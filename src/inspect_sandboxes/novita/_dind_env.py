"""DinD per-service sandbox environment"""

from __future__ import annotations

import errno
import shlex
import shutil
import tempfile
import uuid
from logging import getLogger
from pathlib import Path, PurePosixPath
from typing import Literal, overload

import yaml
from inspect_ai.util import (
    ComposeConfig,
    ExecResult,
    OutputLimitExceededError,
    SandboxEnvironment,
    SandboxEnvironmentConfigType,
    SandboxEnvironmentLimits,
    trace_message,
)
from typing_extensions import override

from inspect_sandboxes._util.compose import find_default_service

from ._dind_project import (
    DEFAULT_DIND_CPU,
    DEFAULT_DIND_MEMORY_MB,
    NovitaDinDProject,
    compose_exec,
    create_dind_project,
    destroy_dind_project,
    discover_working_dir,
    vm_exec,
)
from ._retry import run_with_timeout_retry
from ._single_env import FILE_REQUEST_TIMEOUT, write_sandbox_file

logger = getLogger(__name__)


class NovitaDinDServiceEnvironment(SandboxEnvironment):
    """SandboxEnvironment for a single compose service inside a DinD sandbox.

    Routes exec/read/write through ``docker compose exec/cp <service>``
    inside the shared Novita sandbox.
    """

    def __init__(
        self, project: NovitaDinDProject, service: str, working_dir: str
    ) -> None:
        super().__init__()
        self.project = project
        self.service = service
        self._working_dir = working_dir

    @classmethod
    async def sample_init_dind(
        cls,
        config: ComposeConfig,
        compose_file: str | None,
        *,
        metadata: dict[str, str],
        cpu_count: int = DEFAULT_DIND_CPU,
        memory_mb: int = DEFAULT_DIND_MEMORY_MB,
        sandbox_timeout: int | float = 3600,
        sandbox_envs: dict[str, str] | None = None,
    ) -> dict[str, SandboxEnvironment]:
        """Create DinD sandbox and return per-service environments.

        Args:
            config: Parsed compose configuration with >1 service.
            compose_file: Local path to the compose file.
            metadata: Metadata to apply to the sandbox.
            cpu_count: CPUs for the DinD template build.
            memory_mb: Memory (MiB) for the DinD template build.
            sandbox_timeout: Sandbox lifetime in seconds (from
                ``x-novita.timeout`` or default). Novita caps at 3600s (Free) /
                86400s (Paid).
            sandbox_envs: Environment variables set on the sandbox VM (not on
                individual compose services).

        Returns:
            Dict of environments with the default service first.
        """
        tmp_dir: Path | None = None
        if compose_file is None:
            try:
                tmp_dir = Path(tempfile.mkdtemp(prefix="inspect-compose-"))
                tmp_path = tmp_dir / "compose.yaml"
                data = config.model_dump(
                    by_alias=True, exclude_none=True, exclude_defaults=True
                )
                tmp_path.write_text(yaml.dump(data, sort_keys=False))
                compose_file = str(tmp_path)
            except Exception as e:
                if tmp_dir is not None:
                    shutil.rmtree(tmp_dir, ignore_errors=True)
                raise RuntimeError(
                    "Failed to serialize ComposeConfig to a temporary compose file."
                ) from e

        try:
            project = await create_dind_project(
                config,
                compose_file,
                metadata=metadata,
                cpu_count=cpu_count,
                memory_mb=memory_mb,
                sandbox_timeout=sandbox_timeout,
                sandbox_envs=sandbox_envs,
            )
        finally:
            if tmp_dir is not None:
                shutil.rmtree(tmp_dir, ignore_errors=True)

        default_name, _ = find_default_service(config)
        environments: dict[str, SandboxEnvironment] = {}
        for svc_name in project.services:
            wd = await discover_working_dir(project, svc_name)
            environments[svc_name] = cls(project, svc_name, wd)

        default_env = environments.pop(default_name)
        return {default_name: default_env, **environments}

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

        any_env = next(iter(environments.values())).as_type(cls)
        project = any_env.project
        try:
            await destroy_dind_project(project)
            await project.sandbox.kill()
        except Exception as e:
            trace_message(
                logger,
                "novita",
                f"Error cleaning up DinD sandbox {project.sandbox.sandbox_id} "
                f"for task '{task_name}': {e}. Will retry in task_cleanup.",
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
        workdir = cwd if cwd is not None else self._working_dir
        if not PurePosixPath(workdir).is_absolute():
            workdir = str(PurePosixPath(self._working_dir) / workdir)

        exec_cmd = ["exec", "-T", "-w", workdir]
        if user is not None:
            exec_cmd.extend(["--user", user])
        if env:
            for k, v in env.items():
                exec_cmd.extend(["-e", f"{k}={v}"])

        stdin_vm_file: str | None = None
        stdin_container_file: str | None = None
        if input is not None:
            data = input.encode("utf-8") if isinstance(input, str) else input
            stdin_vm_file = f"/tmp/.inspect-stdin-{uuid.uuid4().hex}"
            stdin_container_file = f"/tmp/.inspect-stdin-{uuid.uuid4().hex}"
            await write_sandbox_file(self.project.sandbox, stdin_vm_file, data)
            cp_exit, _, cp_err = await compose_exec(
                self.project,
                ["cp", stdin_vm_file, f"{self.service}:{stdin_container_file}"],
                timeout=30,
            )
            if cp_exit != 0:
                raise RuntimeError(f"Failed to copy stdin to {self.service}: {cp_err}")
            stdin_cmd = self._build_stdin_command(cmd, stdin_container_file)
            exec_cmd.extend([self.service, "sh", "-c", stdin_cmd])
        else:
            exec_cmd.extend([self.service, *cmd])

        async def _run(t: int | None) -> ExecResult[str]:
            exit_code, stdout, stderr = await compose_exec(
                self.project, exec_cmd, timeout=t
            )
            return ExecResult(
                success=exit_code == 0,
                returncode=exit_code,
                stdout=stdout,
                stderr=stderr,
            )

        try:
            return await run_with_timeout_retry(_run, timeout, timeout_retry)
        finally:
            if stdin_vm_file is not None:
                try:
                    await vm_exec(
                        self.project.sandbox,
                        f"rm -f {shlex.quote(stdin_vm_file)}",
                        timeout=10,
                    )
                except Exception:
                    pass

    @override
    async def write_file(self, file: str, contents: str | bytes) -> None:
        """Two-hop write: SDK upload to sandbox temp -> docker compose cp to container."""
        file = self._container_file(file)

        parent = str(PurePosixPath(file).parent)
        if parent and parent not in ("/", "."):
            await self._create_parent_folder(parent)

        if await self._is_directory(file):
            raise IsADirectoryError(errno.EISDIR, "Is a directory", file)

        data = contents.encode("utf-8") if isinstance(contents, str) else contents
        temp = f"/tmp/.inspect-write-{uuid.uuid4().hex}"
        try:
            await write_sandbox_file(self.project.sandbox, temp, data)
            exit_code, _, stderr = await compose_exec(
                self.project,
                ["cp", temp, f"{self.service}:{file}"],
                timeout=120,
            )
            if exit_code != 0:
                raise RuntimeError(
                    f"docker compose cp to {self.service}:{file} failed: {stderr}"
                )
        finally:
            try:
                await vm_exec(
                    self.project.sandbox, f"rm -f {shlex.quote(temp)}", timeout=10
                )
            except Exception:
                pass

    @overload
    async def read_file(self, file: str, text: Literal[True] = True) -> str: ...

    @overload
    async def read_file(self, file: str, text: Literal[False]) -> bytes: ...

    @override
    async def read_file(self, file: str, text: bool = True) -> str | bytes:
        """Two-hop read: docker compose cp from container -> SDK download from sandbox."""
        file = self._container_file(file)
        await self._verify_read_size(file)

        temp = f"/tmp/.inspect-read-{uuid.uuid4().hex}"
        try:
            exit_code, _, stderr = await compose_exec(
                self.project,
                ["cp", f"{self.service}:{file}", temp],
                timeout=120,
            )
            if exit_code != 0:
                msg = stderr.lower()
                if "no such" in msg or "not found" in msg:
                    raise FileNotFoundError(
                        errno.ENOENT, "No such file or directory", file
                    )
                raise RuntimeError(
                    f"docker compose cp from {self.service}:{file} failed: {stderr}"
                )
            data = await self.project.sandbox.files.read(
                temp, format="bytes", request_timeout=FILE_REQUEST_TIMEOUT
            )
            data_bytes = bytes(data)
        finally:
            # docker cp writes the temp as root, so the non-root user can't rm
            # it directly (sticky-bit /tmp). Sudo to avoid leaking into /tmp.
            try:
                await vm_exec(
                    self.project.sandbox,
                    f"sudo rm -f {shlex.quote(temp)}",
                    timeout=10,
                )
            except Exception:
                pass

        if text:
            try:
                return data_bytes.decode("utf-8")
            except UnicodeDecodeError as e:
                raise UnicodeDecodeError(
                    e.encoding,
                    e.object,
                    e.start,
                    e.end,
                    f"Failed to decode {file}: {e.reason}",
                ) from e
        return data_bytes

    @staticmethod
    def _build_stdin_command(cmd: list[str], stdin_file: str) -> str:
        quoted = shlex.quote(stdin_file)
        return f"{shlex.join(cmd)} < {quoted}; _ec=$?; rm -f {quoted}; exit $_ec"

    def _container_file(self, file: str) -> str:
        """Resolve relative path against working directory."""
        path = PurePosixPath(file)
        if not path.is_absolute():
            path = PurePosixPath(self._working_dir) / path
        return str(path)

    async def _is_directory(self, path: str) -> bool:
        exit_code, _, _ = await compose_exec(
            self.project,
            ["exec", "-T", self.service, "test", "-d", path],
            timeout=10,
        )
        return exit_code == 0

    async def _get_file_size(self, path: str) -> int:
        """Return file size in bytes. Raises FileNotFoundError if missing."""
        quoted = shlex.quote(path)
        # stat -c %s works on GNU coreutils + busybox; -f %z is the BSD fallback.
        exit_code, stdout, _ = await compose_exec(
            self.project,
            [
                "exec",
                "-T",
                self.service,
                "sh",
                "-c",
                f"stat -c %s {quoted} 2>/dev/null || stat -f %z {quoted} 2>/dev/null",
            ],
            timeout=10,
        )
        if exit_code == 0:
            try:
                return int(stdout.strip())
            except ValueError as e:
                raise RuntimeError(
                    f"Failed to parse file size for {path}: {stdout!r}"
                ) from e

        test_exit, _, _ = await compose_exec(
            self.project,
            ["exec", "-T", self.service, "test", "-e", path],
            timeout=10,
        )
        if test_exit != 0:
            raise FileNotFoundError(errno.ENOENT, "No such file or directory", path)
        raise PermissionError(
            errno.EACCES, "Cannot stat (likely permission denied)", path
        )

    async def _verify_read_size(self, file: str) -> None:
        if await self._is_directory(file):
            raise IsADirectoryError(errno.EISDIR, "Is a directory", file)
        size = await self._get_file_size(file)
        if size > SandboxEnvironmentLimits.MAX_READ_FILE_SIZE:
            raise OutputLimitExceededError(
                limit_str=SandboxEnvironmentLimits.MAX_READ_FILE_SIZE_STR,
                truncated_output=None,
            )

    async def _create_parent_folder(self, path: str) -> None:
        exit_code, _, stderr = await compose_exec(
            self.project,
            ["exec", "-T", self.service, "mkdir", "-p", path],
            timeout=10,
        )
        if exit_code != 0:
            raise RuntimeError(f"Failed to create directory {path}: {stderr}")
