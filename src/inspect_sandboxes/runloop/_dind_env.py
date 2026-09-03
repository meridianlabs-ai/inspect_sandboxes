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
from runloop_api_client import AsyncRunloop, NotFoundError
from typing_extensions import override

from ._compose import (
    extract_runloop_timeout,
    extract_x_runloop,
    find_default_service,
    normalize_launch_parameters,
)
from ._dind_project import (
    RunloopDinDProject,
    _download_file,
    _upload_file,
    compose_exec,
    create_dind_project,
    destroy_dind_project,
    discover_working_dir,
    vm_exec,
)
from ._retry import run_with_timeout_retry

logger = getLogger(__name__)


class RunloopDinDServiceEnvironment(SandboxEnvironment):
    """SandboxEnvironment for a single compose service inside a DinD Devbox.

    Routes exec/read/write through ``docker compose exec/cp <service>``
    inside the shared Runloop Devbox.
    """

    def __init__(
        self, project: RunloopDinDProject, service: str, working_dir: str
    ) -> None:
        super().__init__()
        self.project = project
        self.service = service
        self._working_dir = working_dir

    @classmethod
    async def sample_init_dind(
        cls,
        client: AsyncRunloop,
        config: ComposeConfig,
        compose_file: str | None,
        *,
        name: str | None = None,
        metadata: dict[str, str],
    ) -> dict[str, SandboxEnvironment]:
        """Create DinD devbox and return per-service environments.

        Args:
            client: Runloop SDK client.
            config: Parsed compose configuration with >1 service.
            compose_file: Local path to the compose file.
            name: Human-readable devbox name to assign.
            metadata: Metadata to apply to the devbox.

        Returns:
            Dict of environments with the default service first.
        """
        # Serialize an in-memory ComposeConfig to a temp file for build context.
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
            ext = extract_x_runloop(config.extensions)
            launch_parameters = normalize_launch_parameters(
                ext.get("launch_parameters")
            )
            env_vars_raw = ext.get("environment_variables")
            environment_variables = (
                {str(k): str(v) for k, v in env_vars_raw.items()}
                if isinstance(env_vars_raw, dict)
                else None
            )
            # Merge x-runloop.metadata into the run metadata.
            ext_meta = ext.get("metadata")
            run_metadata = dict(metadata)
            if isinstance(ext_meta, dict):
                run_metadata = {
                    **{str(k): str(v) for k, v in ext_meta.items()},
                    **run_metadata,
                }

            timeout = extract_runloop_timeout(config.extensions)

            project = await create_dind_project(
                client,
                config,
                compose_file,
                name=name,
                metadata=run_metadata,
                launch_parameters=launch_parameters,
                environment_variables=environment_variables,
                timeout=timeout,
            )
        finally:
            if tmp_dir is not None:
                shutil.rmtree(tmp_dir, ignore_errors=True)

        # Build per-service environments with default first.
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
            await project.client.devboxes.shutdown(project.devbox_id)
        except NotFoundError:
            pass  # already gone
        except Exception as e:
            trace_message(
                logger,
                "runloop",
                f"Error cleaning up DinD devbox {project.devbox_id} for task '{task_name}': {e}. "
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
        workdir = cwd if cwd is not None else self._working_dir
        if not PurePosixPath(workdir).is_absolute():
            workdir = str(PurePosixPath(self._working_dir) / workdir)

        exec_cmd = ["exec", "-T", "-w", workdir]
        if user is not None:
            exec_cmd.extend(["--user", user])
        if env:
            for k, v in env.items():
                exec_cmd.extend(["-e", f"{k}={v}"])

        # Stdin: two-hop upload (VM → compose cp → container), then pipe.
        stdin_vm_file: str | None = None
        stdin_container_file: str | None = None
        if input is not None:
            data = input.encode("utf-8") if isinstance(input, str) else input
            stdin_vm_file = f"/tmp/.inspect-stdin-{uuid.uuid4().hex}"
            stdin_container_file = f"/tmp/.inspect-stdin-{uuid.uuid4().hex}"
            await _upload_file(
                self.project.client, self.project.devbox_id, stdin_vm_file, data
            )
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
                        self.project.client,
                        self.project.devbox_id,
                        f"rm -f {shlex.quote(stdin_vm_file)}",
                        timeout=10,
                    )
                except Exception:
                    pass

    @override
    async def write_file(self, file: str, contents: str | bytes) -> None:
        """Two-hop write: VM temp -> docker compose cp to container."""
        file = self._container_file(file)

        parent = str(PurePosixPath(file).parent)
        if parent and parent not in ("/", "."):
            await self._create_parent_folder(parent)

        if await self._is_directory(file):
            raise IsADirectoryError(errno.EISDIR, "Is a directory", file)

        data = contents.encode("utf-8") if isinstance(contents, str) else contents
        temp = f"/tmp/.inspect-write-{uuid.uuid4().hex}"
        try:
            await _upload_file(self.project.client, self.project.devbox_id, temp, data)
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
                    self.project.client,
                    self.project.devbox_id,
                    f"rm -f {shlex.quote(temp)}",
                    timeout=10,
                )
            except Exception:
                pass

    @overload
    async def read_file(self, file: str, text: Literal[True] = True) -> str: ...

    @overload
    async def read_file(self, file: str, text: Literal[False]) -> bytes: ...

    @override
    async def read_file(self, file: str, text: bool = True) -> str | bytes:
        """Two-hop read: docker compose cp from container -> read from VM."""
        file = self._container_file(file)
        size = await self._verify_read_size(file)

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
            data_bytes = await _download_file(
                self.project.client, self.project.devbox_id, temp, size=size
            )
        finally:
            try:
                await vm_exec(
                    self.project.client,
                    self.project.devbox_id,
                    f"rm -f {shlex.quote(temp)}",
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

    async def _create_parent_folder(self, path: str) -> None:
        exit_code, _, stderr = await compose_exec(
            self.project,
            ["exec", "-T", self.service, "mkdir", "-p", path],
            timeout=10,
        )
        if exit_code != 0:
            raise RuntimeError(f"Failed to create directory {path}: {stderr}")
