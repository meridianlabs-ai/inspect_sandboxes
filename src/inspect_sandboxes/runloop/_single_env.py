"""Single-service Runloop sandbox environment."""

from __future__ import annotations

import errno
import shlex
import uuid
from logging import getLogger
from typing import Literal, NoReturn, overload

from inspect_ai.util import (
    ExecResult,
    OutputLimitExceededError,
    SandboxEnvironment,
    SandboxEnvironmentConfigType,
    SandboxEnvironmentLimits,
    trace_message,
)
from runloop_api_client import (
    AsyncRunloop,
    BadRequestError,
    NotFoundError,
)
from runloop_api_client.types import DevboxAsyncExecutionDetailView
from typing_extensions import override
from uuid_utils import uuid7

from ._retry import (
    execute_with_poll,
    run_with_timeout_retry,
    shutdown_devbox,
    standard_retry,
)

logger = getLogger(__name__)

FILE_REQUEST_TIMEOUT = 1800
# Runloop's API caps the response stdout/stderr at `last_n` lines (default 100,
# server-side max < 10000). We pass the max so user commands are rarely truncated.
EXEC_LAST_N = "9999"


class RunloopSingleServiceEnvironment(SandboxEnvironment):
    """Single-service sandbox using the Runloop SDK directly."""

    def __init__(self, client: AsyncRunloop, devbox_id: str) -> None:
        super().__init__()
        self.client = client
        self.devbox_id = devbox_id

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
            devbox = None
            try:
                devbox = env.as_type(RunloopSingleServiceEnvironment)
                await shutdown_devbox(devbox.client, devbox.devbox_id)
            except NotFoundError:
                pass  # already gone
            except Exception as e:
                devbox_id = devbox.devbox_id if devbox else "unknown"
                trace_message(
                    logger,
                    "runloop",
                    f"Error shutting down Runloop devbox {devbox_id} for task '{task_name}': {e}. "
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
        """Execute a command in the devbox.

        Runloop's ``execute`` takes a single shell-string command, so we compose
        ``cd``/env-var prefixes and stdin redirection ourselves to honor the
        wider ``exec`` signature. We submit with a stable ``command_id`` and poll
        for completion, so a retried submit is deduped by Runloop rather than
        double-run, and short timeouts are enforced exactly.
        """
        # Runloop's exec API doesn't expose stdin in foreground mode; write
        # input to a temp file and pipe it via shell redirection.
        stdin_file: str | None = None
        if input is not None:
            data = input.encode("utf-8") if isinstance(input, str) else input
            stdin_file = f"/tmp/.inspect-stdin-{uuid.uuid4().hex}"
            # /tmp always exists, so skip _write_file_bytes' parent-dir mkdir.
            await self._upload_file_bytes(stdin_file, data)
            command = self._build_stdin_command(cmd, stdin_file, cleanup=user is None)
        else:
            command = shlex.join(cmd)

        # Prepend env-var assignments (shell-style ``VAR=val cmd``).
        if env:
            env_prefix = " ".join(f"{k}={shlex.quote(v)}" for k, v in env.items())
            command = f"{env_prefix} {command}"

        # Prepend cwd via ``cd``.
        if cwd is not None:
            command = f"cd {shlex.quote(cwd)} && {command}"

        # Runloop devboxes have a fixed default user. To run as a different
        # user, wrap with sudo -u.
        if user is not None:
            user_arg = shlex.quote(f"#{user}") if user.isdigit() else shlex.quote(user)
            command = f"sudo -u {user_arg} bash -c {shlex.quote(command)}"

        # Stable command_id, generated once outside the retry: Runloop dedupes
        # on it, so a retried submit returns the same execution rather than
        # double-running the command.
        command_id = str(uuid7())

        async def _run(t: int | None) -> ExecResult[str]:
            response = await execute_with_poll(
                self.client,
                self.devbox_id,
                command,
                command_id,
                t,
                last_n=EXEC_LAST_N,
            )
            _verify_exec_output_size(response)
            stderr_text = response.stderr or ""
            # POSIX exit code 126 means "command found but not executable" —
            # i.e. the kernel/shell-level exec failed with EACCES. Translate
            # only that to PermissionError so caller code can catch it;
            # application-level "permission denied" stderr (e.g. ``cat`` on
            # an unreadable file) stays as a normal non-zero ExecResult so
            # agents can react to it.
            if (
                response.exit_status == 126
                and "permission denied" in stderr_text.lower()
            ):
                raise PermissionError(errno.EACCES, "Permission denied", stderr_text)
            return ExecResult(
                success=(response.exit_status or 0) == 0,
                returncode=response.exit_status
                if response.exit_status is not None
                else 0,
                stdout=response.stdout or "",
                stderr=stderr_text,
            )

        try:
            return await run_with_timeout_retry(_run, timeout, timeout_retry)
        finally:
            # When running as a non-default user, the wrapped command can't
            # delete the temp file. Clean up as the default user.
            if stdin_file is not None and user is not None:
                try:
                    await self.client.devboxes.execute_and_await_completion(
                        self.devbox_id,
                        command=f"rm -f {shlex.quote(stdin_file)}",
                        timeout=10,
                    )
                except Exception:
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
        await self._write_file_bytes(file, data)

    @overload
    async def read_file(self, file: str, text: Literal[True] = True) -> str: ...

    @overload
    async def read_file(self, file: str, text: Literal[False]) -> bytes: ...

    @override
    async def read_file(self, file: str, text: bool = True) -> str | bytes:
        """Read file from devbox.

        Raises:
            FileNotFoundError: File does not exist.
            IsADirectoryError: Path is a directory.
            UnicodeDecodeError: Encoding error (text mode only).
            OutputLimitExceededError: File exceeds 100 MiB limit.
        """
        await self._verify_read_size(file)

        data = await self._read_file_bytes(file)
        if text:
            try:
                return data.decode("utf-8")
            except UnicodeDecodeError as e:
                raise UnicodeDecodeError(
                    e.encoding,
                    e.object,
                    e.start,
                    e.end,
                    f"Failed to decode {file}: {e.reason}",
                ) from e
        return data

    @staticmethod
    def _build_stdin_command(cmd: list[str], stdin_file: str, *, cleanup: bool) -> str:
        quoted = shlex.quote(stdin_file)
        base = f"{shlex.join(cmd)} < {quoted}"
        if cleanup:
            return f"{base}; _ec=$?; rm -f {quoted}; exit $_ec"
        return f"{base}; _ec=$?; exit $_ec"

    @standard_retry
    async def _get_file_size(self, file: str) -> int:
        """Return file size in bytes. Raises FileNotFoundError if missing."""
        quoted = shlex.quote(file)
        # stat -c %s works on GNU coreutils + busybox; -f %z is the BSD fallback.
        result = await self.client.devboxes.execute_and_await_completion(
            self.devbox_id,
            command=f"stat -c %s {quoted} 2>/dev/null || stat -f %z {quoted} 2>/dev/null",
        )
        if (result.exit_status or 0) == 0:
            try:
                return int((result.stdout or "0").strip())
            except ValueError as e:
                raise RuntimeError(
                    f"Failed to parse file size for {file}: {result.stdout!r}"
                ) from e

        test = await self.client.devboxes.execute_and_await_completion(
            self.devbox_id, command=f"test -e {quoted}"
        )
        if (test.exit_status or 0) != 0:
            raise FileNotFoundError(errno.ENOENT, "No such file or directory", file)
        raise PermissionError(
            errno.EACCES, "Cannot stat (likely permission denied)", file
        )

    @standard_retry
    async def _is_directory(self, file: str) -> bool:
        result = await self.client.devboxes.execute_and_await_completion(
            self.devbox_id,
            command=f"test -d {shlex.quote(file)}",
        )
        return (result.exit_status or 0) == 0

    async def _verify_read_size(self, file: str) -> int:
        if await self._is_directory(file):
            raise IsADirectoryError(errno.EISDIR, "Is a directory", file)
        size = await self._get_file_size(file)
        if size > SandboxEnvironmentLimits.MAX_READ_FILE_SIZE:
            raise OutputLimitExceededError(
                limit_str=SandboxEnvironmentLimits.MAX_READ_FILE_SIZE_STR,
                truncated_output=None,
            )
        return size

    @standard_retry
    async def _read_file_bytes(self, file: str) -> bytes:
        # download_file streams binary content of any size. Runloop reports a
        # missing path, a directory, or a permission denial as a 400 that
        # _raise_filesystem_error maps to the read_file contract.
        try:
            response = await self.client.devboxes.download_file(
                self.devbox_id, path=file, timeout=FILE_REQUEST_TIMEOUT
            )
        except BadRequestError as e:
            _raise_filesystem_error(e, file)
        return await response.read()

    async def _write_file_bytes(self, file: str, data: bytes) -> None:
        # write_file promises to create parent directories; upload_file writes
        # only the file, so create the parent first.
        parent = file.rsplit("/", 1)[0] if "/" in file else ""
        if parent:
            await self.exec(["mkdir", "-p", parent])
        await self._upload_file_bytes(file, data)

    @standard_retry
    async def _upload_file_bytes(self, file: str, data: bytes) -> None:
        # upload_file sends binary content of any size via multipart form data.
        try:
            await self.client.devboxes.upload_file(
                self.devbox_id, path=file, file=data, timeout=FILE_REQUEST_TIMEOUT
            )
        except BadRequestError as e:
            _raise_filesystem_error(e, file)


def _raise_filesystem_error(error: BadRequestError, file: str) -> NoReturn:
    """Translate a filesystem 400 to the matching OSError; re-raise others.

    Runloop reports filesystem failures on upload_file/download_file as a 400
    whose body message names the cause: a permission denial carries the EACCES
    errno ("os error 13"), a missing path reads "File does not exist", and a
    directory target reads "Path is a directory".
    """
    body = getattr(error, "body", None)
    message = body.get("message", "") if isinstance(body, dict) else str(error)
    lowered = message.lower()
    if "os error 13" in lowered or "permission denied" in lowered:
        raise PermissionError(errno.EACCES, "Permission denied", file) from None
    if "does not exist" in lowered:
        raise FileNotFoundError(
            errno.ENOENT, "No such file or directory", file
        ) from None
    if "is a directory" in lowered:
        raise IsADirectoryError(errno.EISDIR, "Is a directory", file) from None
    raise error


def _verify_exec_output_size(response: DevboxAsyncExecutionDetailView) -> None:
    """Verify the command's output wasn't truncated by Runloop.

    Runloop caps returned stdout/stderr at ``EXEC_LAST_N`` lines and flags it;
    the ``exec`` contract is to signal an output-limit overflow, not to silently
    drop the head of the output.

    Raises:
        OutputLimitExceededError: If stdout or stderr was truncated.
    """
    if response.stdout_truncated or response.stderr_truncated:
        # Attach the stream that actually overflowed (prefer stdout if both).
        truncated = response.stdout if response.stdout_truncated else response.stderr
        raise OutputLimitExceededError(
            limit_str=f"{EXEC_LAST_N} lines",
            truncated_output=truncated or "",
        )
