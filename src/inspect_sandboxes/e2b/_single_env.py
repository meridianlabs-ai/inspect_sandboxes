"""Single-service E2B sandbox environment."""

from __future__ import annotations

import errno
import shlex
import uuid
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
from typing_extensions import override

from ._retry import exec_retry, run_with_timeout_retry, standard_retry

logger = getLogger(__name__)


class E2BSingleServiceEnvironment(SandboxEnvironment):
    """Single-service sandbox using the E2B SDK directly."""

    def __init__(self, sandbox: AsyncSandbox) -> None:
        super().__init__()
        self.sandbox = sandbox

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
        """
        stdin_file: str | None = None
        if input is not None:
            data = input.encode("utf-8") if isinstance(input, str) else input
            stdin_file = f"/tmp/.inspect-stdin-{uuid.uuid4().hex}"
            await self._write_file_content(stdin_file, data)
            command = self._build_stdin_command(cmd, stdin_file, cleanup=user is None)
        else:
            command = shlex.join(cmd)

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

        try:
            return await run_with_timeout_retry(_run, timeout, timeout_retry)
        finally:
            # When running as a non-default user, shell redirection runs in
            # that user's context and may not be able to delete the temp file.
            if stdin_file is not None and user is not None:
                try:
                    await self.sandbox.commands.run(
                        f"rm -f {shlex.quote(stdin_file)}", timeout=10
                    )
                except SandboxException:
                    pass

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

    @staticmethod
    def _build_stdin_command(cmd: list[str], stdin_file: str, *, cleanup: bool) -> str:
        quoted = shlex.quote(stdin_file)
        base = f"{shlex.join(cmd)} < {quoted}"
        if cleanup:
            return f"{base}; _ec=$?; rm -f {quoted}; exit $_ec"
        return f"{base}; _ec=$?; exit $_ec"

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
        try:
            return await self.sandbox.files.read(file, format="text", user="root")
        except (FileNotFoundException, NotFoundException) as e:
            raise FileNotFoundError(
                errno.ENOENT, "No such file or directory", file
            ) from e

    @standard_retry
    async def _read_file_bytes(self, file: str) -> bytes:
        try:
            data = await self.sandbox.files.read(file, format="bytes", user="root")
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
            await self.sandbox.files.write(file, data, user="root")
        except SandboxException as e:
            msg = str(e).lower()
            if "is a directory" in msg or "isdir" in msg:
                raise IsADirectoryError(errno.EISDIR, "Is a directory", file) from e
            raise
