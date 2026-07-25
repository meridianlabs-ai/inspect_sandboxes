"""Single-service Runloop sandbox environment."""

from __future__ import annotations

import base64
import errno
import shlex
import uuid
from logging import getLogger
from typing import Any, Literal, overload

import httpx
from inspect_ai.util import (
    ExecResult,
    OutputLimitExceededError,
    SandboxEnvironment,
    SandboxEnvironmentConfigType,
    SandboxEnvironmentLimits,
    trace_message,
)
from runloop_api_client import APITimeoutError, AsyncRunloop, NotFoundError
from typing_extensions import override
from uuid_utils import uuid7

from ._retry import (
    exec_retry,
    poll_execution,
    run_with_timeout_retry,
    standard_retry,
)

logger = getLogger(__name__)

FILE_REQUEST_TIMEOUT = 1800
# Runloop's API caps the response stdout/stderr at `last_n` lines (default 100,
# server-side max < 10000). We pass the max so user commands are rarely truncated.
EXEC_LAST_N = "9999"
# Files at or above this size switch from base64-over-shell to the Runloop
# Objects API (presigned PUT/GET via S3). Base64-over-shell is one round trip
# for small payloads but exceeds Runloop's request-body cap on multi-MiB files.
LARGE_FILE_THRESHOLD = 4 * 1024 * 1024


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
                await devbox.client.devboxes.shutdown(devbox.devbox_id)
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

        Runloop's ``execute_and_await_completion`` takes a single shell-string
        command. We compose ``cd``/env-var prefixes and stdin redirection
        ourselves so the wider ``exec`` signature is honored.
        """
        # Runloop's exec API doesn't expose stdin in foreground mode; write
        # input to a temp file and pipe it via shell redirection.
        stdin_file: str | None = None
        if input is not None:
            data = input.encode("utf-8") if isinstance(input, str) else input
            stdin_file = f"/tmp/.inspect-stdin-{uuid.uuid4().hex}"
            await self._write_file_bytes(stdin_file, data)
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

        @exec_retry
        async def _execute_and_await(command_id: str) -> Any:
            # Combined submit+await for the no-timeout path. Retried on
            # transient errors; the stable command_id (generated once, outside
            # the retry) lets us kill the in-flight command if an httpx-level
            # timeout fires.
            return await self.client.devboxes.execute_and_await_completion(
                self.devbox_id,
                command=command,
                command_id=command_id,
                last_n=EXEC_LAST_N,
            )

        @exec_retry
        async def _submit_async() -> str:
            # Only the submission is retried. Polling happens separately (via
            # poll_execution) so a transient mid-poll error never re-runs the
            # command.
            started = await self.client.devboxes.execute_async(
                self.devbox_id, command=command
            )
            return started.execution_id

        async def _run(t: int | None) -> ExecResult[str]:
            execution_id: str | None = None
            try:
                if t is None:
                    execution_id = str(uuid7())
                    response: Any = await _execute_and_await(execution_id)
                else:
                    # The SDK's long-poll await has a server-side minimum wait
                    # that doesn't respect short user timeouts. Submit async and
                    # poll ourselves to enforce *t* exactly.
                    execution_id = await _submit_async()
                    response = await poll_execution(
                        self.client,
                        self.devbox_id,
                        execution_id,
                        t,
                        last_n=EXEC_LAST_N,
                    )
            except APITimeoutError:
                if execution_id is not None:
                    await self._kill_execution(execution_id)
                raise
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
        size = await self._verify_read_size(file)

        data = await self._read_file_bytes(file, size=size)
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

    async def _kill_execution(self, execution_id: str) -> None:
        """Best-effort kill of a running execution (and its process group)."""
        try:
            await self.client.devboxes.executions.kill(
                execution_id,
                devbox_id=self.devbox_id,
                kill_process_group=True,
            )
        except Exception:
            pass

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

    async def _read_file_bytes(self, file: str, *, size: int) -> bytes:
        if size >= LARGE_FILE_THRESHOLD:
            return await self._read_file_bytes_via_object(file)
        return await self._read_file_bytes_via_shell(file)

    @standard_retry
    async def _read_file_bytes_via_shell(self, file: str) -> bytes:
        # Runloop's read_file_contents SDK method returns a UTF-8 string;
        # round-trip via base64-in-shell to handle arbitrary bytes.
        result = await self.client.devboxes.execute_and_await_completion(
            self.devbox_id,
            command=f"base64 -w 0 {shlex.quote(file)}",
            timeout=FILE_REQUEST_TIMEOUT,
        )
        if (result.exit_status or 0) != 0:
            stderr = (result.stderr or "").lower()
            if "no such file" in stderr or "not found" in stderr:
                raise FileNotFoundError(errno.ENOENT, "No such file or directory", file)
            if "permission denied" in stderr:
                raise PermissionError(errno.EACCES, "Permission denied", file)
            raise RuntimeError(
                f"Failed to read {file}: exit={result.exit_status} stderr={result.stderr!r}"
            )
        return base64.b64decode((result.stdout or "").strip())

    async def _read_file_bytes_via_object(self, file: str) -> bytes:
        """Read by routing the bytes through a Runloop Object.

        The devbox PUTs the file to a presigned upload URL; we then GET the
        bytes from the presigned download URL. Avoids Runloop's exec-response
        size cap that the base64-over-shell path hits on multi-MiB files.
        """
        obj = await self.client.objects.create(
            content_type="binary",
            name=f"inspect-read-{uuid.uuid4().hex[:12]}",
        )
        if obj.upload_url is None:
            raise RuntimeError(
                "Runloop did not return an upload URL for the read Object."
            )
        try:
            cmd = _devbox_upload_command(file, obj.upload_url)
            result = await self.client.devboxes.execute_and_await_completion(
                self.devbox_id, command=cmd, timeout=FILE_REQUEST_TIMEOUT
            )
            if (result.exit_status or 0) != 0:
                stderr = (result.stderr or "").lower()
                if "no such file" in stderr or "not found" in stderr:
                    raise FileNotFoundError(
                        errno.ENOENT, "No such file or directory", file
                    )
                if "permission denied" in stderr:
                    raise PermissionError(errno.EACCES, "Permission denied", file)
                raise RuntimeError(
                    f"Failed to upload {file} to object store: "
                    f"exit={result.exit_status} stderr={result.stderr!r}"
                )
            await self.client.objects.complete(obj.id)
            download = await self.client.objects.download(obj.id)
            async with httpx.AsyncClient(timeout=FILE_REQUEST_TIMEOUT) as http:
                response = await http.get(download.download_url)
                response.raise_for_status()
                return response.content
        finally:
            try:
                await self.client.objects.delete(obj.id)
            except Exception:
                pass

    async def _write_file_bytes(self, file: str, data: bytes) -> None:
        if len(data) >= LARGE_FILE_THRESHOLD:
            await self._write_file_bytes_via_object(file, data)
            return
        await self._write_file_bytes_via_shell(file, data)

    @standard_retry
    async def _write_file_bytes_via_shell(self, file: str, data: bytes) -> None:
        # Runloop's write_file_contents SDK method only accepts UTF-8 strings;
        # round-trip via base64-in-shell to handle arbitrary bytes.
        encoded = base64.b64encode(data).decode("ascii")
        # Ensure the parent dir exists. Quote the parent as a whole so a parent
        # containing spaces isn't split into multiple tokens.
        parent = file.rsplit("/", 1)[0] if "/" in file else ""
        mkdir_prefix = f"mkdir -p {shlex.quote(parent)} && " if parent else ""
        result = await self.client.devboxes.execute_and_await_completion(
            self.devbox_id,
            command=f"{mkdir_prefix}echo {shlex.quote(encoded)} | base64 -d > {shlex.quote(file)}",
            timeout=FILE_REQUEST_TIMEOUT,
        )
        if (result.exit_status or 0) != 0:
            stderr = (result.stderr or "").lower()
            if "permission denied" in stderr:
                raise PermissionError(errno.EACCES, "Permission denied", file)
            raise RuntimeError(
                f"Failed to write {file}: exit={result.exit_status} stderr={result.stderr!r}"
            )

    async def _write_file_bytes_via_object(self, file: str, data: bytes) -> None:
        """Write by routing the bytes through a Runloop Object.

        We PUT the bytes to a presigned upload URL, then the devbox GETs them
        from the presigned download URL into the destination file. Avoids
        Runloop's request-body cap that the base64-over-shell path hits on
        multi-MiB files.
        """
        obj = await self.client.objects.create(
            content_type="binary",
            name=f"inspect-write-{uuid.uuid4().hex[:12]}",
        )
        if obj.upload_url is None:
            raise RuntimeError(
                "Runloop did not return an upload URL for the write Object."
            )
        try:
            async with httpx.AsyncClient(timeout=FILE_REQUEST_TIMEOUT) as http:
                put = await http.put(obj.upload_url, content=data)
                put.raise_for_status()
            await self.client.objects.complete(obj.id)
            download = await self.client.objects.download(obj.id)
            # Quote the parent as a whole (see _write_file_bytes_via_shell).
            parent = file.rsplit("/", 1)[0] if "/" in file else ""
            mkdir_prefix = f"mkdir -p {shlex.quote(parent)} && " if parent else ""
            cmd = (
                f"{mkdir_prefix}{_devbox_download_command(download.download_url, file)}"
            )
            result = await self.client.devboxes.execute_and_await_completion(
                self.devbox_id, command=cmd, timeout=FILE_REQUEST_TIMEOUT
            )
            if (result.exit_status or 0) != 0:
                stderr = (result.stderr or "").lower()
                if "permission denied" in stderr:
                    raise PermissionError(errno.EACCES, "Permission denied", file)
                raise RuntimeError(
                    f"Failed to write {file}: exit={result.exit_status} stderr={result.stderr!r}"
                )
        finally:
            try:
                await self.client.objects.delete(obj.id)
            except Exception:
                pass


def _devbox_download_command(url: str, file: str) -> str:
    """Shell command that downloads ``url`` into ``file`` on the devbox.

    Tries ``curl``, then ``wget``, then ``python3`` — covers Runloop's default
    image plus most user-supplied eval images (scientific stacks, CTF images,
    Alpine, etc.) where curl isn't guaranteed.

    The url and file are passed as positional arguments (``$1`` and ``$2``)
    rather than interpolated into the script body, so a malicious url can't
    escape the shell context.
    """
    python_fallback = (
        "import sys, shutil, urllib.request\n"
        "with urllib.request.urlopen(sys.argv[1]) as r, "
        "open(sys.argv[2], 'wb') as f:\n"
        "    shutil.copyfileobj(r, f)\n"
    )
    script = f"""\
if command -v curl >/dev/null 2>&1; then
    curl -fsSL -o "$2" "$1"
elif command -v wget >/dev/null 2>&1; then
    wget -qO "$2" "$1"
elif command -v python3 >/dev/null 2>&1; then
    python3 -c {shlex.quote(python_fallback)} "$1" "$2"
else
    echo "no HTTP client (curl/wget/python3) on devbox" >&2
    exit 1
fi
"""
    return f"sh -c {shlex.quote(script)} _ {shlex.quote(url)} {shlex.quote(file)}"


def _devbox_upload_command(file: str, url: str) -> str:
    """Shell command that PUTs ``file`` to ``url`` from the devbox.

    Tries ``curl`` then ``python3``. wget is omitted because it can't do PUT
    in any portable way.

    The url and file are passed as positional arguments (``$1`` and ``$2``)
    rather than interpolated into the script body, so a malicious url can't
    escape the shell context.
    """
    python_fallback = (
        "import sys, urllib.request\n"
        "data = open(sys.argv[2], 'rb').read()\n"
        "req = urllib.request.Request(sys.argv[1], method='PUT', data=data)\n"
        "urllib.request.urlopen(req).read()\n"
    )
    script = f"""\
if command -v curl >/dev/null 2>&1; then
    curl -fsS -X PUT --upload-file "$2" "$1"
elif command -v python3 >/dev/null 2>&1; then
    python3 -c {shlex.quote(python_fallback)} "$1" "$2"
else
    echo "no HTTP client (curl/python3) on devbox" >&2
    exit 1
fi
"""
    return f"sh -c {shlex.quote(script)} _ {shlex.quote(url)} {shlex.quote(file)}"
