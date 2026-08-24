"""Single-service Novita sandbox environment."""

from __future__ import annotations

import errno
import shlex
import uuid
from logging import getLogger
from pathlib import PurePosixPath
from typing import Literal, overload

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
from novita_sandbox.core import (
    AsyncSandbox,
    CommandExitException,
    FileNotFoundException,
    FileType,
    NotFoundException,
    SandboxException,
)
from typing_extensions import override

from ._retry import exec_retry, run_with_timeout_retry, standard_retry

logger = getLogger(__name__)

FILE_REQUEST_TIMEOUT = 1800
# Writes above this size are split into chunks and reassembled in the sandbox:
# a single large upload to the envd files route fails (uncompressed → the
# connection resets with httpx.ReadError; gzipped → the gateway 504s), while a
# 1 GiB Sample.files upload in 8 MiB chunks is reliable.
UPLOAD_CHUNK_SIZE = 8 * 1024 * 1024


class NovitaSingleServiceEnvironment(SandboxEnvironment):
    """Single-service sandbox using the Novita SDK directly."""

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
                sandbox = env.as_type(NovitaSingleServiceEnvironment).sandbox
                await cls._kill_sandbox(sandbox)
            except Exception as e:
                sandbox_id = sandbox.sandbox_id if sandbox else "unknown"
                trace_message(
                    logger,
                    "novita",
                    f"Error killing Novita sandbox {sandbox_id} for task '{task_name}': {e}. "
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

        Novita's commands.run() doesn't expose stdin in foreground mode, so we
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
            # Novita's commands.run raises CommandExitException on non-zero exit;
            # the SandboxEnvironment.exec contract requires us to surface that
            # as an ExecResult with the failing returncode instead.
            try:
                result = await self.sandbox.commands.run(
                    command,
                    cwd=cwd,
                    envs=env,
                    user=user or "root",
                    # Novita's `timeout` is the per-command wall-clock; 0 disables.
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
                except Exception:
                    # Best-effort cleanup: the SDK leaks raw httpx/httpcore
                    # transport errors, and an exception here would replace
                    # the exec result.
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

    @override
    async def connection(self, *, user: str | None = None) -> SandboxConnection:
        """Surface Compose-declared ports as Novita host URLs.

        ``get_host(port)`` returns a reachable host for any container port, so
        we resolve one per declared container port and place it in
        ``SandboxConnection.ports``. The host is served over HTTPS, so each
        mapping uses the host as ``host_ip`` and port 443.
        """
        ports: list[PortMapping] | None = None
        if self._connection_ports:
            mappings: list[PortMapping] = []
            for container_port in self._connection_ports:
                # get_host is synchronous in the Novita SDK (string formatting).
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
            type="novita",
            command=f"novita-sandbox-cli sandbox connect {self.sandbox.sandbox_id}",
            ports=ports,
            container=self.sandbox.sandbox_id,
        )

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

    async def _write_file_content(self, file: str, data: bytes) -> None:
        # Novita's files.write auto-creates parent directories. We don't need
        # the explicit mkdir step that Modal/Daytona do.
        try:
            await write_sandbox_file(self.sandbox, file, data, user="root")
        except SandboxException as e:
            msg = str(e).lower()
            if "is a directory" in msg or "isdir" in msg:
                raise IsADirectoryError(errno.EISDIR, "Is a directory", file) from e
            raise


async def write_sandbox_file(
    sandbox: AsyncSandbox, file: str, data: bytes, user: str | None = None
) -> None:
    """Write *data* to *file* in the sandbox, chunking large payloads.

    Each chunk is uploaded to a temp file and appended to the target, so peak
    extra disk usage stays at one chunk (accumulating every chunk before
    reassembly can exhaust the disk — /tmp may be a small tmpfs).
    """
    if len(data) <= UPLOAD_CHUNK_SIZE:
        await _write_file_bytes(sandbox, file, data, user)
        return

    temp = f"/tmp/.inspect-chunk-{uuid.uuid4().hex}"
    quoted_temp = shlex.quote(temp)
    quoted = shlex.quote(file)
    parent = shlex.quote(str(PurePosixPath(file).parent))
    try:
        await sandbox.commands.run(
            f"mkdir -p {parent} && : > {quoted}",
            user=user,
            timeout=FILE_REQUEST_TIMEOUT,
        )
        for offset in range(0, len(data), UPLOAD_CHUNK_SIZE):
            await _write_file_bytes(
                sandbox, temp, data[offset : offset + UPLOAD_CHUNK_SIZE], user
            )
            # Not retried: a retry after a lost success reply would append the
            # chunk twice.
            await sandbox.commands.run(
                f"cat {quoted_temp} >> {quoted}",
                user=user,
                timeout=FILE_REQUEST_TIMEOUT,
            )
    finally:
        try:
            await sandbox.commands.run(f"rm -f {quoted_temp}", user=user, timeout=60)
        except Exception:
            # Best-effort cleanup: the SDK leaks raw httpx/httpcore transport
            # errors, and a cleanup failure must never mask the primary error.
            pass


@standard_retry
async def _write_file_bytes(
    sandbox: AsyncSandbox, file: str, data: bytes, user: str | None
) -> None:
    # gzip compresses the request body in transit; it only takes effect with
    # use_octet_stream — the default multipart path ignores the flag.
    await sandbox.files.write(
        file,
        data,
        user=user,
        request_timeout=FILE_REQUEST_TIMEOUT,
        gzip=True,
        use_octet_stream=True,
    )
