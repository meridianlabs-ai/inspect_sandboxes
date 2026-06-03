"""Single-service Daytona sandbox environment"""

from __future__ import annotations

import errno
import shlex
import uuid
from logging import getLogger
from pathlib import PurePosixPath
from typing import Literal, overload
from urllib.parse import urlsplit

from daytona_sdk import AsyncSandbox, DaytonaError, DaytonaNotFoundError
from inspect_ai.util import (
    ExecResult,
    SandboxEnvironment,
    SandboxEnvironmentConfigType,
    trace_message,
)
from inspect_ai.util._sandbox.environment import (
    HostMapping,
    PortMapping,
    SandboxConnection,
)
from typing_extensions import override

from ._retry import exec_retry, run_with_timeout_retry, standard_retry
from ._sandbox_utils import (
    build_stdin_command,
    decode_file_content,
    delete_sandbox,
    verify_file_size,
)

logger = getLogger(__name__)


class DaytonaSingleServiceEnvironment(SandboxEnvironment):
    """Single-service sandbox using the Daytona SDK directly."""

    def __init__(
        self, sandbox: AsyncSandbox, connection_ports: list[int] | None = None
    ) -> None:
        super().__init__()
        self.sandbox = sandbox
        # Container ports declared via Compose `ports`, surfaced lazily through
        # connection() as get_preview_link URLs.
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

        # Deferred import to avoid circular dependency:
        # _daytona.py imports _single_env.py (for DaytonaSingleServiceEnvironment),
        # and _single_env.py needs _daytona_client from _daytona.py for cleanup.
        from ._daytona import _daytona_client

        client = _daytona_client.get()
        if client is None:
            return

        for env in environments.values():
            sandbox = None
            try:
                sandbox = env.as_type(DaytonaSingleServiceEnvironment).sandbox
                await delete_sandbox(client, sandbox)
            except Exception as e:
                sandbox_id = sandbox.id if sandbox else "unknown"
                trace_message(
                    logger,
                    "daytona",
                    f"Error deleting Daytona sandbox {sandbox_id} for task '{task_name}': {e}. "
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

        Note:
            stderr is always empty. The Daytona API returns a single combined
            output field; stdout and stderr are not distinguished.

        Timeout: The Daytona server enforces timeouts server-side, killing
            the process tree. No in-container ``timeout`` wrapping needed
            (unlike the Docker sandbox).
        """
        # Daytona's process.exec() doesn't support stdin natively.
        # When input is provided, write it to a temp file and pipe it into the command.
        stdin_file: str | None = None
        if input is not None:
            data = input.encode("utf-8") if isinstance(input, str) else input
            stdin_file = f"/tmp/.inspect-stdin-{uuid.uuid4().hex}"
            await self._write_file_content(stdin_file, data)
            command = build_stdin_command(cmd, stdin_file, cleanup=user is None)
        else:
            command = shlex.join(cmd)

        # Daytona's process.exec() has no user param — use sudo -u to switch.
        if user is not None:
            if user.isdigit():
                user_arg = shlex.quote(f"#{user}")
            else:
                user_arg = shlex.quote(user)
            command = f"sudo -u {user_arg} bash -c {shlex.quote(command)}"

        @exec_retry
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

        try:
            return await run_with_timeout_retry(_run, timeout, timeout_retry)
        finally:
            # When running as a different user, the su'd process can't delete
            # root-owned temp files in sticky /tmp. Clean up as the default user.
            if stdin_file is not None and user is not None:
                await self.sandbox.process.exec(
                    f"rm -f {shlex.quote(stdin_file)}", timeout=10
                )

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
        await verify_file_size(self._is_directory, self._get_file_size, file)

        contents_bytes = await self._download_file(file)

        return decode_file_content(contents_bytes, file, text)

    @override
    async def connection(self, *, user: str | None = None) -> SandboxConnection:
        """Surface Compose-declared ports as Daytona preview URLs.

        ``get_preview_link(port)`` opens the port if it's closed and returns a
        public URL, so we resolve one per declared container port and place it
        in ``SandboxConnection.ports``. The preview URL is a full
        ``https://…`` URL, so we split it into a bare host (``host_ip``) and the
        URL's port (``host_port``, defaulting to 443/80 by scheme) — storing the
        scheme-prefixed URL in ``host_ip`` would corrupt any ``host:port``
        rendering downstream.
        """
        ports: list[PortMapping] | None = None
        if self._connection_ports:
            mappings: list[PortMapping] = []
            for container_port in self._connection_ports:
                try:
                    preview = await self.sandbox.get_preview_link(container_port)
                except Exception as e:
                    trace_message(
                        logger,
                        "daytona",
                        f"Could not get preview link for port {container_port}: {e}",
                    )
                    continue
                parsed = urlsplit(preview.url)
                host_ip = parsed.hostname
                if host_ip is None:
                    trace_message(
                        logger,
                        "daytona",
                        f"Preview URL for port {container_port} has no host: "
                        f"{preview.url!r}; skipping it.",
                    )
                    continue
                host_port = parsed.port or (443 if parsed.scheme == "https" else 80)
                mappings.append(
                    PortMapping(
                        container_port=container_port,
                        protocol="tcp",
                        mappings=[HostMapping(host_ip=host_ip, host_port=host_port)],
                    )
                )
            ports = mappings or None

        return SandboxConnection(
            type="daytona",
            command=f"daytona sandbox ssh {self.sandbox.id}",
            ports=ports,
            container=self.sandbox.id,
        )

    @staticmethod
    def _check_permission_error(e: DaytonaError, path: str) -> None:
        """Translate Daytona permission errors to PermissionError."""
        if e.status_code == 403 or "permission denied" in str(e).lower():
            raise PermissionError(errno.EACCES, "Permission denied", path) from e

    @standard_retry
    async def _get_file_size(self, file: str) -> int:
        try:
            info = await self.sandbox.fs.get_file_info(file)
            return int(info.size or 0)
        except DaytonaNotFoundError as e:
            raise FileNotFoundError(
                errno.ENOENT, "No such file or directory", file
            ) from e
        except DaytonaError as e:
            self._check_permission_error(e, file)
            raise

    @standard_retry
    async def _is_directory(self, file: str) -> bool:
        try:
            info = await self.sandbox.fs.get_file_info(file)
            return bool(info.is_dir)
        except DaytonaNotFoundError:
            return False
        except DaytonaError as e:
            self._check_permission_error(e, file)
            raise

    @standard_retry
    async def _download_file(self, file: str) -> bytes:
        try:
            return await self.sandbox.fs.download_file(file)
        except DaytonaNotFoundError as e:
            raise FileNotFoundError(
                errno.ENOENT, "No such file or directory", file
            ) from e
        except DaytonaError as e:
            self._check_permission_error(e, file)
            raise

    @standard_retry
    async def _create_parent_folder(self, path: str) -> None:
        try:
            await self.sandbox.fs.create_folder(path, "755")
        except DaytonaError as e:
            if e.status_code == 409:  # directory already exists
                return
            self._check_permission_error(e, path)
            raise

    @standard_retry
    async def _write_file_content(self, file: str, contents: bytes) -> None:
        try:
            await self.sandbox.fs.upload_file(contents, file)
        except DaytonaError as e:
            self._check_permission_error(e, file)
            raise
