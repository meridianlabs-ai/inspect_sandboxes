"""Single-service E2B sandbox environment."""

from __future__ import annotations

import errno
import shlex
from logging import getLogger
from typing import Literal, overload

from e2b import (
    AsyncSandbox,
    CommandExitException,
    FileNotFoundException,
    FileType,
    NotFoundException,
    SandboxException,
)
from inspect_ai.util import (
    ExecResult,
    OutputLimitExceededError,
    SandboxEnvironment,
    SandboxEnvironmentConfigType,
    SandboxEnvironmentLimits,
    trace_message,
)
from inspect_ai.util._sandbox.environment import (
    HostMapping,
    PortMapping,
    SandboxConnection,
)
from typing_extensions import override

from ._command import (
    exceeds_inline_limit,
    remove_files_command,
    source_script_command,
    temp_file_path,
)
from ._retry import exec_retry, run_with_timeout_retry, standard_retry

logger = getLogger(__name__)

FILE_REQUEST_TIMEOUT = 1800


class E2BSingleServiceEnvironment(SandboxEnvironment):
    """Single-service sandbox using the E2B SDK directly."""

    def __init__(
        self, sandbox: AsyncSandbox, connection_ports: list[int] | None = None
    ) -> None:
        super().__init__()
        self.sandbox = sandbox
        # Container ports declared via Compose `ports`, surfaced through
        # connection() as get_host URLs.
        self._connection_ports = connection_ports or []

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
                sandbox = env.as_type(E2BSingleServiceEnvironment).sandbox
                await cls._kill_sandbox(sandbox)
            except Exception as e:
                sandbox_id = sandbox.sandbox_id if sandbox else "unknown"
                trace_message(
                    logger,
                    "e2b",
                    f"Error killing E2B sandbox {sandbox_id} for task '{task_name}': {e}. "
                    "Will retry in task_cleanup.",
                )

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

        E2B's commands.run() doesn't expose stdin in foreground mode, so we
        write *input* to a temp file inside the sandbox and pipe it via shell
        redirection — same approach Daytona uses.

        commands.run() also passes the whole command line as a single
        ``bash -l -c`` argument, which the kernel caps at 128 KiB, so a longer
        command is staged in a temp script and sourced (see ``_command``).
        Temp files are removed afterwards, as root, in the ``finally`` below.
        """
        temp_files: list[str] = []
        try:
            if input is not None:
                data = input.encode("utf-8") if isinstance(input, str) else input
                stdin_file = temp_file_path("stdin")
                temp_files.append(stdin_file)
                await self._write_file_content(stdin_file, data)
                command = f"{shlex.join(cmd)} < {shlex.quote(stdin_file)}"
            else:
                command = shlex.join(cmd)

            if exceeds_inline_limit(command):
                script_file = temp_file_path("cmd")
                temp_files.append(script_file)
                await self._write_file_content(script_file, command.encode("utf-8"))
                command = source_script_command(script_file)

            return await self._run_command(
                command,
                cwd=cwd,
                env=env,
                user=user,
                timeout=timeout,
                timeout_retry=timeout_retry,
            )
        finally:
            if temp_files:
                await self._remove_temp_files(temp_files)

    async def _run_command(
        self,
        command: str,
        *,
        cwd: str | None,
        env: dict[str, str] | None,
        user: str | None,
        timeout: int | None,
        timeout_retry: bool,
    ) -> ExecResult[str]:
        @exec_retry
        async def _run(t: int | None) -> ExecResult[str]:
            # E2B's commands.run raises CommandExitException on non-zero exit;
            # the SandboxEnvironment.exec contract requires us to surface that
            # as an ExecResult with the failing returncode instead.
            try:
                result = await self.sandbox.commands.run(
                    command,
                    cwd=cwd,
                    envs=env,
                    user=user or "root",
                    # E2B's `timeout` is the per-command wall-clock; 0 disables.
                    timeout=t if t is not None else 0,
                )
            except CommandExitException as e:
                return ExecResult(
                    success=False,
                    returncode=e.exit_code if e.exit_code is not None else 1,
                    stdout=e.stdout,
                    stderr=e.stderr,
                )
            return ExecResult(
                success=result.exit_code == 0,
                returncode=result.exit_code if result.exit_code is not None else 0,
                stdout=result.stdout,
                stderr=result.stderr,
            )

        return await run_with_timeout_retry(_run, timeout, timeout_retry)

    async def _remove_temp_files(self, files: list[str]) -> None:
        # Written as root (see _write_file_content), so removed as root; the
        # exec user may not be allowed to delete them in sticky /tmp.
        try:
            await self.sandbox.commands.run(
                remove_files_command(files), user="root", timeout=10
            )
        except Exception as e:
            trace_message(logger, "e2b", f"Could not remove temp files {files}: {e}")

    @override
    async def write_file(self, file: str, contents: str | bytes) -> None:
        """Creates parent directories automatically.

        Raises:
            IsADirectoryError: File path already exists as a directory.
        """
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
        await self._verify_read_size(file)

        try:
            if text:
                content = await self._read_file_text(file)
                return content
            data = await self._read_file_bytes(file)
            return bytes(data)
        except UnicodeDecodeError as e:
            raise UnicodeDecodeError(
                e.encoding,
                e.object,
                e.start,
                e.end,
                f"Failed to decode {file}: {e.reason}",
            ) from e

    @override
    async def connection(self, *, user: str | None = None) -> SandboxConnection:
        """Surface Compose-declared ports as E2B host URLs.

        ``get_host(port)`` returns a reachable host for any container port, so
        we resolve one per declared container port and place it in
        ``SandboxConnection.ports``. The host is served over HTTPS, so each
        mapping uses the host as ``host_ip`` and port 443.
        """
        ports: list[PortMapping] | None = None
        if self._connection_ports:
            mappings: list[PortMapping] = []
            for container_port in self._connection_ports:
                # get_host is synchronous in the E2B SDK (string formatting).
                host = self.sandbox.get_host(container_port)
                mappings.append(
                    PortMapping(
                        container_port=container_port,
                        protocol="tcp",
                        mappings=[HostMapping(host_ip=host, host_port=443)],
                    )
                )
            ports = mappings or None

        return SandboxConnection(
            type="e2b",
            command=f"e2b sandbox connect {self.sandbox.sandbox_id}",
            ports=ports,
            container=self.sandbox.sandbox_id,
        )

    @staticmethod
    @standard_retry
    async def _kill_sandbox(sandbox: AsyncSandbox) -> None:
        try:
            await sandbox.kill()
        except NotFoundException:
            pass  # already gone

    @standard_retry
    async def _get_file_size(self, file: str) -> int:
        try:
            info = await self.sandbox.files.get_info(file, user="root")
        except NotFoundException as e:
            raise FileNotFoundError(
                errno.ENOENT, "No such file or directory", file
            ) from e
        return int(info.size or 0)

    @standard_retry
    async def _is_directory(self, file: str) -> bool:
        try:
            info = await self.sandbox.files.get_info(file, user="root")
            return info.type == FileType.DIR
        except NotFoundException:
            return False

    async def _verify_read_size(self, file: str) -> None:
        if await self._is_directory(file):
            raise IsADirectoryError(errno.EISDIR, "Is a directory", file)
        size = await self._get_file_size(file)
        if size > SandboxEnvironmentLimits.MAX_READ_FILE_SIZE:
            raise OutputLimitExceededError(
                limit_str=SandboxEnvironmentLimits.MAX_READ_FILE_SIZE_STR,
                truncated_output=None,
            )

    @standard_retry
    async def _read_file_text(self, file: str) -> str:
        # Override the SDK's 60s default — too tight for large files
        # (e.g. multi-GB datasets passed via Sample.files). 30 min matches
        # the Daytona SDK's default for the same operation.
        try:
            return await self.sandbox.files.read(
                file, format="text", user="root", request_timeout=FILE_REQUEST_TIMEOUT
            )
        except (FileNotFoundException, NotFoundException) as e:
            raise FileNotFoundError(
                errno.ENOENT, "No such file or directory", file
            ) from e

    @standard_retry
    async def _read_file_bytes(self, file: str) -> bytes:
        try:
            data = await self.sandbox.files.read(
                file, format="bytes", user="root", request_timeout=FILE_REQUEST_TIMEOUT
            )
            return bytes(data)
        except (FileNotFoundException, NotFoundException) as e:
            raise FileNotFoundError(
                errno.ENOENT, "No such file or directory", file
            ) from e

    @standard_retry
    async def _write_file_content(self, file: str, data: bytes) -> None:
        # E2B's files.write auto-creates parent directories. We don't need
        # the explicit mkdir step that Modal/Daytona do.
        try:
            await self.sandbox.files.write(
                file, data, user="root", request_timeout=FILE_REQUEST_TIMEOUT
            )
        except SandboxException as e:
            msg = str(e).lower()
            if "is a directory" in msg or "isdir" in msg:
                raise IsADirectoryError(errno.EISDIR, "Is a directory", file) from e
            raise
