from __future__ import annotations

import asyncio
import errno
import os
import sys
from contextvars import ContextVar
from logging import getLogger
from pathlib import PurePosixPath
from typing import Any, Literal, cast, overload

import modal
import modal.exception
from inspect_ai._util.logger import (
    warn_once,  # noqa: PLC2701 TODO: switch to public import once released on PyPI
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
)
from rich import box, print
from rich.prompt import Confirm
from rich.table import Table
from tenacity import (
    AsyncRetrying,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from typing_extensions import override

from ._compose import convert_compose_to_modal_params

logger = getLogger(__name__)

MODAL_APP_NAME = "inspect_modal_sandbox"
INSPECT_SANDBOX_TAG = {"created_by": "inspect-ai"}

_running_sandboxes: ContextVar[list[str]] = ContextVar("modal_running_sandboxes")


def sandbox_cleanup_startup() -> None:
    _running_sandboxes.set([])


def running_sandboxes() -> list[str]:
    return _running_sandboxes.get()


# Retry decorator for file I/O and sandbox lifecycle ops
_standard_retry = retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    reraise=True,
)


@sandboxenv(name="modal")
class ModalSandboxEnvironment(SandboxEnvironment):
    def __init__(self, sandbox: modal.Sandbox) -> None:
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
        sandbox_cleanup_startup()

    @override
    @classmethod
    async def sample_init(
        cls,
        task_name: str,
        config: SandboxEnvironmentConfigType | None,
        metadata: dict[str, str],
    ) -> dict[str, SandboxEnvironment]:
        app = await cls._lookup_app(MODAL_APP_NAME)

        sandbox_params: dict[str, Any] = {
            "app": app,
            "timeout": 60 * 60 * 24,
        }

        if config is None:
            trace_message(
                logger, "modal", f"Using default Modal image for task '{task_name}'"
            )
        elif is_dockerfile(config):
            sandbox_params["image"] = modal.Image.from_dockerfile(config)
        elif is_compose_yaml(config):
            compose_config = parse_compose_yaml(config, multiple_services=False)
            modal_params = convert_compose_to_modal_params(compose_config, config)
            sandbox_params.update(modal_params)
        elif isinstance(config, ComposeConfig):
            modal_params = convert_compose_to_modal_params(config, None)
            sandbox_params.update(modal_params)
        else:
            raise ValueError(
                f"Unrecognized config: {config}. "
                "Expected a compose file (*.yaml/*.yml), Dockerfile, "
                "ComposeConfig object, or None."
            )

        sandbox = await cls._create_sandbox(sandbox_params)
        await sandbox.set_tags.aio(INSPECT_SANDBOX_TAG)
        running_sandboxes().append(sandbox.object_id)

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

        for env in environments.values():
            sandbox = None
            try:
                sandbox = env.as_type(ModalSandboxEnvironment).sandbox
                await cls._terminate_sandbox(sandbox)

            except Exception as e:
                sandbox_id = cls._get_sandbox_id(sandbox)
                trace_message(
                    logger,
                    "modal",
                    f"Error terminating Modal sandbox {sandbox_id} for task '{task_name}': {e}. "
                    "Will retry in task_cleanup.",
                )

    @override
    @classmethod
    async def task_cleanup(
        cls, task_name: str, config: SandboxEnvironmentConfigType | None, cleanup: bool
    ) -> None:
        """Cleanup sandboxes at task completion.

        Note: terminate() is idempotent (no-op if already terminated).
        """
        if not cleanup:
            return

        failed_ids: list[str] = []

        for sandbox_id in running_sandboxes().copy():
            try:
                sandbox = await modal.Sandbox.from_id.aio(sandbox_id)
                await cls._terminate_sandbox(sandbox)
                trace_message(logger, "modal", f"Terminated sandbox {sandbox_id}")

            except Exception as e:
                failed_ids.append(sandbox_id)
                logger.error(f"Failed to terminate sandbox {sandbox_id}: {e}")

        if failed_ids:
            logger.warning(
                f"Failed to cleanup {len(failed_ids)} sandbox(es). "
                f"Failed IDs: {', '.join(failed_ids)}"
            )

        running_sandboxes().clear()

    @override
    @classmethod
    async def cli_cleanup(cls, id: str | None) -> None:
        if id is not None:
            # Single sandbox cleanup
            try:
                sandbox = await modal.Sandbox.from_id.aio(id)
                await cls._terminate_sandbox(sandbox)
                print(f"Successfully terminated sandbox {id}")
            except Exception as e:
                print(f"[red]Error terminating sandbox {id}: {e}[/red]")
                sys.exit(1)
        else:
            # Bulk cleanup
            sandboxes = [
                sb async for sb in modal.Sandbox.list.aio(tags=INSPECT_SANDBOX_TAG)
            ]

            if not sandboxes:
                print("No Modal sandboxes found to clean up.")
                return

            table = Table(
                box=box.SQUARE,
                show_lines=False,
                title_style="bold",
                title_justify="left",
            )
            table.add_column("Sandbox ID")
            for sb in sandboxes:
                table.add_row(sb.object_id)
            print(table)

            # Only prompt if in an interactive shell
            is_interactive = sys.stdin.isatty()
            is_ci = "CI" in os.environ
            is_pytest = "PYTEST_CURRENT_TEST" in os.environ

            if is_interactive and not is_ci and not is_pytest:
                if not Confirm.ask(
                    f"Are you sure you want to terminate ALL {len(sandboxes)} "
                    "sandbox(es) above?"
                ):
                    print("Cancelled.")
                    return

            success_count = 0
            failure_count = 0

            for sb in sandboxes:
                try:
                    await cls._terminate_sandbox(sb)
                    success_count += 1
                except Exception as e:
                    print(
                        f"[yellow]Error terminating sandbox {sb.object_id}: {e}[/yellow]"
                    )
                    failure_count += 1

            print(f"\n[green]Successfully terminated: {success_count}[/green]")
            if failure_count > 0:
                print(f"[red]Failed to terminate: {failure_count}[/red]")
                sys.exit(1)
            else:
                print("Complete.")

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
        if user is not None:
            warn_once(
                logger,
                "The 'user' parameter is ignored in ModalSandboxEnvironment. "
                "Commands will run as the container's default user.",
            )

        # Modal requires absolute paths for workdir
        workdir = cwd
        if workdir is not None and not PurePosixPath(workdir).is_absolute():
            warn_once(
                logger,
                f"Relative path '{workdir}' for cwd parameter was converted to absolute path '/{workdir}' "
                "(relative to filesystem root). For clarity, consider using absolute paths.",
            )
            workdir = f"/{workdir}"

        async def _run() -> ExecResult[str]:
            modal_env = cast(dict[str, str | None] | None, env)

            process = await self.sandbox.exec.aio(
                *cmd,
                workdir=workdir,
                env=modal_env,
            )

            if input is not None:
                try:
                    data = input.encode("utf-8") if isinstance(input, str) else input
                    process.stdin.write(data)
                except modal.exception.InternalError as e:
                    logger.warning(f"Modal InternalError while writing stdin: {e}.")
                    raise
                finally:
                    # No kill() on Modal's ContainerProcess
                    # Close stdin to unblock the process
                    try:
                        process.stdin.write_eof()
                        await process.stdin.drain.aio()
                    except Exception:
                        pass

            try:
                stdout = await process.stdout.read.aio()
                stderr = await process.stderr.read.aio()
            except UnicodeDecodeError as e:
                raise UnicodeDecodeError(
                    e.encoding,
                    e.object,
                    e.start,
                    e.end,
                    f"Failed to decode command output: {e.reason}",
                ) from e

            await process.wait.aio()

            return ExecResult(
                success=process.returncode == 0,
                returncode=process.returncode if process.returncode is not None else 0,
                stdout=stdout,
                stderr=stderr,
            )

        # Only retry short timeouts: https://inspect.aisi.org.uk/sandboxing.html
        use_retry = timeout_retry and timeout is not None and timeout < 60

        try:
            result: ExecResult[str] | None = None
            if use_retry:
                async for attempt in AsyncRetrying(
                    stop=stop_after_attempt(3),
                    wait=wait_exponential(multiplier=1, min=1, max=10),
                    retry=retry_if_exception_type(
                        (asyncio.TimeoutError, modal.exception.InternalError)
                    ),
                    reraise=True,
                ):
                    with attempt:
                        result = await asyncio.wait_for(_run(), timeout=timeout)
                assert result is not None  # Should always be set after successful retry
            elif timeout:
                result = await asyncio.wait_for(_run(), timeout=timeout)
            else:
                result = await _run()

            return result

        except asyncio.TimeoutError as e:
            raise TimeoutError(f"Command timed out after {timeout} seconds") from e

    @override
    async def write_file(self, file: str, contents: str | bytes) -> None:
        """Creates parent directories automatically if they don't exist.

        Raises:
            IsADirectoryError: File path already exists as a directory.
        """
        parent = str(PurePosixPath(file).parent)
        if parent and parent not in ("/", "."):
            try:
                await self.sandbox.mkdir.aio(parent, parents=True)
            except FileExistsError:
                pass

        try:
            await self._write_file_content(file, contents)
        except IsADirectoryError as e:
            raise IsADirectoryError(errno.EISDIR, "Is a directory", file) from e

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

        try:
            async with await self.sandbox.open.aio(file, "rb") as f:
                contents_bytes = await f.read.aio()
        except modal.exception.FilesystemExecutionError as e:
            if await self._is_directory(file):
                raise IsADirectoryError(errno.EISDIR, "Is a directory", file) from e
            raise FileNotFoundError(
                errno.ENOENT, "No such file or directory", file
            ) from e

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
    async def _lookup_app(app_name: str) -> modal.App:
        return await modal.App.lookup.aio(app_name, create_if_missing=True)

    @staticmethod
    @_standard_retry
    async def _create_sandbox(sandbox_params: dict[str, Any]) -> modal.Sandbox:
        return await modal.Sandbox.create.aio(**sandbox_params)

    @staticmethod
    @_standard_retry
    async def _terminate_sandbox(sandbox: modal.Sandbox) -> None:
        await sandbox.terminate.aio()
        # Verify the sandbox stopped — poll() returns None if still running
        if await sandbox.poll.aio() is None:
            raise RuntimeError(
                f"Sandbox {sandbox.object_id} still running after terminate()"
            )

    @staticmethod
    def _get_sandbox_id(sandbox: modal.Sandbox | None) -> str:
        if sandbox is None:
            return "unknown"
        return getattr(sandbox, "object_id", "unknown")

    @_standard_retry
    async def _write_file_content(self, file: str, contents: str | bytes) -> None:
        if isinstance(contents, str):
            async with await self.sandbox.open.aio(file, "w") as f:
                await f.write.aio(contents)
        else:
            async with await self.sandbox.open.aio(file, "wb") as f:
                await f.write.aio(contents)

    async def _is_directory(self, file: str) -> bool:
        try:
            process = await self.sandbox.exec.aio("test", "-d", file)
            await process.wait.aio()
            return process.returncode == 0
        except Exception:
            return False

    async def _get_file_size(self, file: str) -> int:
        process = await self.sandbox.exec.aio("stat", "-c", "%s", file)
        stdout = await process.stdout.read.aio()
        await process.wait.aio()

        if process.returncode != 0:
            if process.returncode == 1:
                raise FileNotFoundError(errno.ENOENT, "No such file or directory", file)
            stderr = await process.stderr.read.aio()
            raise RuntimeError(
                f"stat command failed with code {process.returncode}: {stderr}"
            )

        try:
            return int(stdout.strip())
        except ValueError as e:
            raise RuntimeError(f"Failed to parse file size for {file}") from e

    async def _verify_read_file_size(self, file: str) -> None:
        if await self._is_directory(file):
            raise IsADirectoryError(errno.EISDIR, "Is a directory", file)

        file_size = await self._get_file_size(file)
        if file_size > SandboxEnvironmentLimits.MAX_READ_FILE_SIZE:
            raise OutputLimitExceededError(
                limit_str=SandboxEnvironmentLimits.MAX_READ_FILE_SIZE_STR,
                truncated_output=None,
            )
