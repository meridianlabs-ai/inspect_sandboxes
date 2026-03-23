from __future__ import annotations

import errno
import os
import shlex
import sys
import uuid
from contextvars import ContextVar
from logging import getLogger
from pathlib import PurePosixPath
from typing import Literal, overload

from daytona_sdk import (
    AsyncDaytona,
    AsyncSandbox,
    CreateSandboxFromImageParams,
    CreateSandboxFromSnapshotParams,
    DaytonaError,
    DaytonaNotFoundError,
    DaytonaTimeoutError,
    Image,
)
from inspect_ai.util import (
    ComposeConfig,
    ExecResult,
    OutputLimitExceededError,
    SandboxEnvironment,
    SandboxEnvironmentConfigType,
    SandboxEnvironmentLimits,
    is_compose_yaml,
    is_dockerfile,
    parse_compose_yaml,
    sandboxenv,
    trace_message,
    warn_once,
)
from rich import box, print
from rich.prompt import Confirm
from rich.table import Table
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from typing_extensions import override

from ._compose import convert_compose_to_daytona_params

logger = getLogger(__name__)

INSPECT_SANDBOX_LABEL = {"created_by": "inspect-ai"}

_daytona_client: ContextVar[AsyncDaytona | None] = ContextVar(
    "daytona_client", default=None
)
_running_sandboxes: ContextVar[list[str]] = ContextVar("daytona_running_sandboxes")
_run_id: ContextVar[str] = ContextVar("daytona_run_id")

# Retry decorator for sandbox lifecycle and file I/O operations
_standard_retry = retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type(DaytonaError),
    reraise=True,
)


def _init_context() -> None:
    _daytona_client.set(None)
    _running_sandboxes.set([])
    _run_id.set(uuid.uuid4().hex)


def _run_labels() -> dict[str, str]:
    return {**INSPECT_SANDBOX_LABEL, "inspect_run_id": _run_id.get()}


@sandboxenv(name="daytona")
class DaytonaSandboxEnvironment(SandboxEnvironment):
    def __init__(self, sandbox: AsyncSandbox) -> None:
        super().__init__()
        self.sandbox = sandbox

    @classmethod
    def config_files(cls) -> list[str]:
        return [
            "compose.yaml",
            "compose.yml",
            "docker-compose.yaml",
            "docker-compose.yml",
            "Dockerfile",
        ]

    @classmethod
    def is_docker_compatible(cls) -> bool:
        return True

    @override
    @classmethod
    async def task_init(
        cls, task_name: str, config: SandboxEnvironmentConfigType | None
    ) -> None:
        _init_context()
        client = AsyncDaytona()
        _daytona_client.set(client)

    @override
    @classmethod
    async def sample_init(
        cls,
        task_name: str,
        config: SandboxEnvironmentConfigType | None,
        metadata: dict[str, str],
    ) -> dict[str, SandboxEnvironment]:
        client = _daytona_client.get()
        if client is None:
            raise RuntimeError(
                "Daytona client not initialized. task_init must be called first."
            )

        params: CreateSandboxFromSnapshotParams | CreateSandboxFromImageParams

        if config is None:
            params = CreateSandboxFromSnapshotParams(
                snapshot=None,
                labels=_run_labels(),
                auto_stop_interval=0,
            )
        elif is_dockerfile(config):
            image = Image.from_dockerfile(config)
            params = CreateSandboxFromImageParams(
                image=image,
                labels=_run_labels(),
                auto_stop_interval=0,
            )
        elif is_compose_yaml(config):
            params = cls._compose_to_image_params(
                parse_compose_yaml(config, multiple_services=False), config
            )
        elif isinstance(config, ComposeConfig):
            params = cls._compose_to_image_params(config, None)
        else:
            raise ValueError(
                f"Unrecognized config: {config}. "
                "Expected a compose file (*.yaml/*.yml), Dockerfile, "
                "ComposeConfig object, or None."
            )

        sandbox = await cls._create_sandbox(client, params)
        _running_sandboxes.get().append(sandbox.id)

        return {"default": cls(sandbox)}

    @override
    @classmethod
    async def sample_cleanup(
        cls,
        task_name: str,
        config: SandboxEnvironmentConfigType | None,
        environments: dict[str, SandboxEnvironment],
        interrupted: bool,
    ) -> None:
        if not environments or interrupted:
            return

        client = _daytona_client.get()
        if client is None:
            return

        for env in environments.values():
            sandbox = None
            try:
                sandbox = env.as_type(DaytonaSandboxEnvironment).sandbox
                await cls._delete_sandbox(client, sandbox)
            except Exception as e:
                sandbox_id = cls._get_sandbox_id(sandbox)
                trace_message(
                    logger,
                    "daytona",
                    f"Error deleting Daytona sandbox {sandbox_id} for task '{task_name}': {e}. "
                    "Will retry in task_cleanup.",
                )

    @override
    @classmethod
    async def task_cleanup(
        cls, task_name: str, config: SandboxEnvironmentConfigType | None, cleanup: bool
    ) -> None:
        if not cleanup:
            return

        client = _daytona_client.get()
        if client is None:
            return

        failed_ids: list[str] = []
        deleted_ids: set[str] = set()

        for sandbox_id in _running_sandboxes.get().copy():
            try:
                sandbox = await client.get(sandbox_id)
                await cls._delete_sandbox(client, sandbox)
                deleted_ids.add(sandbox_id)
                trace_message(logger, "daytona", f"Deleted sandbox {sandbox_id}")
            except DaytonaNotFoundError:
                # Already deleted (e.g. by sample_cleanup) — nothing to do.
                deleted_ids.add(sandbox_id)
                trace_message(
                    logger,
                    "daytona",
                    f"Sandbox {sandbox_id} already deleted, skipping.",
                )
            except Exception as e:
                failed_ids.append(sandbox_id)
                logger.error(f"Failed to delete sandbox {sandbox_id}: {e}")

        # Second pass: find orphaned sandboxes by run label.
        # This catches sandboxes where creation failed (e.g. build_failed)
        # and the ID was never tracked in _running_sandboxes.
        run_id = _run_id.get()
        if run_id:
            try:
                paginated = await client.list(labels={"inspect_run_id": run_id})
                for sb in paginated.items:
                    if sb.id in deleted_ids:
                        continue
                    try:
                        await cls._delete_sandbox(client, sb)
                        trace_message(
                            logger,
                            "daytona",
                            f"Deleted orphaned sandbox {sb.id} (state={sb.state})",
                        )
                    except Exception as e:
                        failed_ids.append(sb.id)
                        logger.error(f"Failed to delete orphaned sandbox {sb.id}: {e}")
            except Exception as e:
                logger.warning(f"Failed to list sandboxes for cleanup: {e}")

        if failed_ids:
            logger.warning(
                f"Failed to cleanup {len(failed_ids)} sandbox(es). "
                f"Failed IDs: {', '.join(failed_ids)}"
            )

        _running_sandboxes.get().clear()

        await client.close()
        _daytona_client.set(None)

    @override
    @classmethod
    async def cli_cleanup(cls, id: str | None) -> None:
        client = AsyncDaytona()
        try:
            if id is not None:
                # Single sandbox cleanup
                try:
                    sandbox = await client.get(id)
                    await cls._delete_sandbox(client, sandbox)
                    print(f"Successfully deleted sandbox {id}")
                except Exception as e:
                    print(f"[red]Error deleting sandbox {id}: {e}[/red]")
                    sys.exit(1)
            else:
                # Bulk cleanup
                paginated = await client.list(labels=INSPECT_SANDBOX_LABEL)
                sandboxes = paginated.items

                if not sandboxes:
                    print("No Daytona sandboxes found to clean up.")
                    return

                table = Table(
                    box=box.SQUARE,
                    show_lines=False,
                    title_style="bold",
                    title_justify="left",
                )
                table.add_column("Sandbox ID")
                for sb in sandboxes:
                    table.add_row(sb.id)
                print(table)

                is_interactive = sys.stdin.isatty()
                is_ci = "CI" in os.environ
                is_pytest = "PYTEST_CURRENT_TEST" in os.environ

                if is_interactive and not is_ci and not is_pytest:
                    if not Confirm.ask(
                        f"Are you sure you want to delete ALL {len(sandboxes)} sandbox(es) above?"
                    ):
                        print("Cancelled.")
                        return

                success_count = 0
                failure_count = 0

                for sb in sandboxes:
                    try:
                        await cls._delete_sandbox(client, sb)
                        success_count += 1
                    except Exception as e:
                        print(f"[yellow]Error deleting sandbox {sb.id}: {e}[/yellow]")
                        failure_count += 1

                print(f"\n[green]Successfully deleted: {success_count}[/green]")
                if failure_count > 0:
                    print(f"[red]Failed to delete: {failure_count}[/red]")
                    sys.exit(1)
                else:
                    print("Complete.")
        finally:
            await client.close()

    @override
    async def exec(
        self,
        cmd: list[str],
        input: str | bytes | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        user: str | None = None,
        timeout: int | None = None,
        timeout_retry: bool = True,
        concurrency: bool = True,
    ) -> ExecResult[str]:
        """Execute a command in the sandbox.

        Note:
            stderr is always empty. The Daytona API returns a single combined
            output field; stdout and stderr are not distinguished.
        """
        if user is not None:
            warn_once(
                logger,
                "The 'user' parameter is ignored in DaytonaSandboxEnvironment. "
                "Commands will run as the container's default user.",
            )

        # Daytona's process.exec() doesn't support stdin natively.
        # When input is provided, write it to a temp file and pipe it into the command.
        if input is not None:
            data = input.encode("utf-8") if isinstance(input, str) else input
            stdin_file = f"/tmp/.inspect-stdin-{uuid.uuid4().hex}"
            await self.sandbox.fs.upload_file(data, stdin_file)
            command = (
                f"set -o pipefail; cat {shlex.quote(stdin_file)} | {shlex.join(cmd)}"
                f"; _ec=$?; rm -f {shlex.quote(stdin_file)}; exit $_ec"
            )
        else:
            command = shlex.join(cmd)

        async def _run(t: int | None) -> ExecResult[str]:
            response = await self.sandbox.process.exec(
                command,
                cwd=cwd,
                env=env,
                timeout=t,
            )
            return ExecResult(
                success=response.exit_code == 0,
                returncode=response.exit_code,
                stdout=response.result,
                stderr="",
            )

        # On timeout, retry with a capped timeout: first retry ≤60s, second ≤30s.
        if timeout_retry:
            t1 = min(timeout, 60) if timeout is not None else 60
            t2 = min(timeout, 30) if timeout is not None else 30
            attempt_timeouts: list[int | None] = [timeout, t1, t2]
        else:
            attempt_timeouts = [timeout]

        last_timeout_exc: DaytonaTimeoutError | None = None
        for t in attempt_timeouts:
            try:
                return await _run(t)
            except DaytonaTimeoutError as e:
                last_timeout_exc = e

        assert last_timeout_exc is not None
        raise TimeoutError(
            f"Command timed out after {timeout} seconds"
        ) from last_timeout_exc

    @override
    async def write_file(self, file: str, contents: str | bytes) -> None:
        """Creates parent directories automatically if they don't exist.

        Raises:
            IsADirectoryError: File path already exists as a directory.
        """
        parent = str(PurePosixPath(file).parent)
        if parent and parent not in ("/", "."):
            await self._create_parent_folder(parent)

        if await self._is_directory(file):
            raise IsADirectoryError(errno.EISDIR, "Is a directory", file)

        data = contents.encode("utf-8") if isinstance(contents, str) else contents
        await self._write_file_content(file, data)

    @overload
    async def read_file(self, file: str, text: Literal[True] = True) -> str: ...

    @overload
    async def read_file(self, file: str, text: Literal[False]) -> bytes: ...

    @override
    async def read_file(self, file: str, text: bool = True) -> str | bytes:
        """Read file from sandbox.

        Raises:
            FileNotFoundError: File does not exist.
            IsADirectoryError: Path is a directory.
            UnicodeDecodeError: Encoding error (text mode only).
            OutputLimitExceededError: File exceeds 100 MiB limit.
        """
        await self._verify_read_file_size(file)

        contents_bytes = await self._download_file(file)

        if text:
            try:
                return contents_bytes.decode("utf-8")
            except UnicodeDecodeError as e:
                raise UnicodeDecodeError(
                    e.encoding,
                    e.object,
                    e.start,
                    e.end,
                    f"Failed to decode {file}: {e.reason}",
                ) from e

        return contents_bytes

    @staticmethod
    @_standard_retry
    async def _create_sandbox(
        client: AsyncDaytona,
        params: CreateSandboxFromSnapshotParams | CreateSandboxFromImageParams,
    ) -> AsyncSandbox:
        return await client.create(params)

    @staticmethod
    @_standard_retry
    async def _delete_sandbox(client: AsyncDaytona, sandbox: AsyncSandbox) -> None:
        try:
            await client.delete(sandbox)
        except DaytonaNotFoundError:
            pass  # already deleted - passing to prevent retry

    @staticmethod
    def _get_sandbox_id(sandbox: AsyncSandbox | None) -> str:
        if sandbox is None:
            return "unknown"
        return sandbox.id

    @staticmethod
    def _compose_to_image_params(
        config: ComposeConfig, compose_path: str | None
    ) -> CreateSandboxFromImageParams:
        image, resources, sandbox_params = convert_compose_to_daytona_params(
            config, compose_path
        )
        sandbox_params.setdefault("auto_stop_interval", 0)
        x_labels = sandbox_params.pop("labels", {})
        return CreateSandboxFromImageParams(
            image=image,
            resources=resources,
            labels={**x_labels, **_run_labels()},
            **sandbox_params,
        )

    async def _verify_read_file_size(self, file: str) -> None:
        if await self._is_directory(file):
            raise IsADirectoryError(errno.EISDIR, "Is a directory", file)

        file_size = await self._get_file_size(file)
        if file_size > SandboxEnvironmentLimits.MAX_READ_FILE_SIZE:
            raise OutputLimitExceededError(
                limit_str=SandboxEnvironmentLimits.MAX_READ_FILE_SIZE_STR,
                truncated_output=None,
            )

    @_standard_retry
    async def _get_file_size(self, file: str) -> int:
        try:
            info = await self.sandbox.fs.get_file_info(file)
            return int(info.size or 0)
        except DaytonaNotFoundError as e:
            raise FileNotFoundError(
                errno.ENOENT, "No such file or directory", file
            ) from e

    @_standard_retry
    async def _is_directory(self, file: str) -> bool:
        try:
            info = await self.sandbox.fs.get_file_info(file)
            return bool(info.is_dir)
        except DaytonaNotFoundError:
            return False

    @_standard_retry
    async def _download_file(self, file: str) -> bytes:
        try:
            return await self.sandbox.fs.download_file(file)
        except DaytonaNotFoundError as e:
            raise FileNotFoundError(
                errno.ENOENT, "No such file or directory", file
            ) from e

    @_standard_retry
    async def _create_parent_folder(self, path: str) -> None:
        try:
            await self.sandbox.fs.create_folder(path, "755")
        except DaytonaError as e:
            # No DaytonaConflictError subclass exists, so we check status_code.
            if e.status_code != 409:  # 409 = directory already exists
                raise

    @_standard_retry
    async def _write_file_content(self, file: str, contents: bytes) -> None:
        await self.sandbox.fs.upload_file(contents, file)
