"""DinD per-service sandbox environment"""

from __future__ import annotations

import errno
import os
import shlex
import tempfile
import uuid
from logging import getLogger
from pathlib import Path, PurePosixPath
from typing import Any, Literal, overload

import yaml
from daytona_sdk import AsyncDaytona
from inspect_ai.util import (
    ComposeConfig,
    ExecResult,
    SandboxEnvironment,
    SandboxEnvironmentConfigType,
    trace_message,
)
from typing_extensions import override

from ._compose import (
    aggregate_resources,
    apply_daytona_extensions,
    find_default_service,
)
from ._dind_project import (
    DaytonaDinDProject,
    compose_exec,
    create_dind_project,
    destroy_dind_project,
    discover_working_dir,
    vm_exec,
)
from ._retry import run_with_timeout_retry
from ._sandbox_utils import (
    build_stdin_command,
    decode_file_content,
    delete_sandbox,
    sdk_download,
    sdk_upload,
    verify_file_size,
)

logger = getLogger(__name__)


class DaytonaDinDServiceEnvironment(SandboxEnvironment):
    """SandboxEnvironment for a single service inside a DinD compose project.

    Routes exec/read/write through ``docker compose exec/cp <service>``
    inside the shared DinD sandbox.
    """

    def __init__(
        self, project: DaytonaDinDProject, service: str, working_dir: str
    ) -> None:
        super().__init__()
        self.project = project
        self.service = service
        self._working_dir = working_dir

    @classmethod
    async def sample_init_dind(
        cls,
        client: AsyncDaytona,
        config: ComposeConfig,
        compose_file: str | None,
        labels: dict[str, str],
    ) -> dict[str, SandboxEnvironment]:
        """Create DinD sandbox and return per-service environments.

        Args:
            client: Daytona client.
            config: Parsed compose configuration with >1 service.
            compose_file: Local path to the compose file.
            labels: Labels to apply to the sandbox.

        Returns:
            Dict of environments with the default service first.
        """
        # When a ComposeConfig object is passed without a compose file path,
        # serialize it to a temporary YAML file for the DinD build context.
        if compose_file is None:
            tmp_dir: Path | None = None
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
                    import shutil

                    shutil.rmtree(tmp_dir, ignore_errors=True)
                raise RuntimeError(
                    "Failed to serialize ComposeConfig to a temporary compose file."
                ) from e

        # Extract x-daytona sandbox-level params
        sandbox_params: dict[str, Any] = {}
        apply_daytona_extensions(sandbox_params, config.extensions)

        # DinD sandbox must have network — warn if user tried to block it
        if sandbox_params.pop("network_block_all", None):
            logger.warning(
                "network_block_all is ignored for DinD multi-service sandboxes "
                "(Docker daemon requires network access for image pulls)."
            )

        # Merge labels
        x_labels = sandbox_params.pop("labels", {})
        merged_labels = {**x_labels, **labels}

        # Snapshot override: x-daytona.dind_snapshot or DAYTONA_DIND_SNAPSHOT env var
        snapshot = sandbox_params.pop("dind_snapshot", None) or os.environ.get(
            "DAYTONA_DIND_SNAPSHOT"
        )

        # Aggregate resources across all services
        resources = aggregate_resources(config)

        project = await create_dind_project(
            client,
            config,
            compose_file,
            labels=merged_labels,
            resources=resources,
            sandbox_params=sandbox_params,
            snapshot=snapshot,
        )

        # Build per-service environments with default first
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

        from ._daytona import _daytona_client

        client = _daytona_client.get()
        if client is None:
            return

        any_env = next(iter(environments.values())).as_type(cls)
        project = any_env.project
        try:
            await destroy_dind_project(project)
            await delete_sandbox(client, project.sandbox)
        except Exception as e:
            trace_message(
                logger,
                "daytona",
                f"Error cleaning up DinD sandbox {project.sandbox.id} for task '{task_name}': {e}. "
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
        # Resolve working directory
        workdir = cwd if cwd is not None else self._working_dir
        if not PurePosixPath(workdir).is_absolute():
            workdir = str(PurePosixPath(self._working_dir) / workdir)

        # Build compose exec command
        exec_cmd = ["exec", "-T", "-w", workdir]
        if user is not None:
            exec_cmd.extend(["--user", user])
        if env:
            for k, v in env.items():
                exec_cmd.extend(["-e", f"{k}={shlex.quote(v)}"])

        # Stdin: two-hop upload (VM temp -> compose cp -> container), then pipe
        stdin_vm_file: str | None = None
        stdin_container_file: str | None = None
        if input is not None:
            data = input.encode("utf-8") if isinstance(input, str) else input
            stdin_vm_file = f"/tmp/.inspect-stdin-{uuid.uuid4().hex}"
            stdin_container_file = f"/tmp/.inspect-stdin-{uuid.uuid4().hex}"
            await sdk_upload(self.project.sandbox, stdin_vm_file, data)
            await compose_exec(
                self.project,
                ["cp", stdin_vm_file, f"{self.service}:{stdin_container_file}"],
                timeout=30,
            )
            stdin_cmd = build_stdin_command(cmd, stdin_container_file)
            exec_cmd.extend([self.service, "sh", "-c", stdin_cmd])
        else:
            exec_cmd.extend([self.service, *cmd])

        async def _run(t: int | None) -> ExecResult[str]:
            exit_code, output = await compose_exec(
                self.project, exec_cmd, timeout=t
            )
            return ExecResult(
                success=exit_code == 0,
                returncode=exit_code,
                stdout=output,
                stderr="",
            )

        try:
            return await run_with_timeout_retry(_run, timeout, timeout_retry)
        finally:
            if stdin_vm_file is not None:
                await vm_exec(
                    self.project.sandbox,
                    f"rm -f {shlex.quote(stdin_vm_file)}",
                    timeout=10,
                )

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
            await sdk_upload(self.project.sandbox, temp, data)
            exit_code, output = await compose_exec(
                self.project,
                ["cp", temp, f"{self.service}:{file}"],
                timeout=120,
            )
            if exit_code != 0:
                raise RuntimeError(
                    f"docker compose cp to {self.service}:{file} failed: {output}"
                )
        finally:
            await vm_exec(self.project.sandbox, f"rm -f {shlex.quote(temp)}", timeout=10)

    @overload
    async def read_file(self, file: str, text: Literal[True] = True) -> str: ...

    @overload
    async def read_file(self, file: str, text: Literal[False]) -> bytes: ...

    @override
    async def read_file(self, file: str, text: bool = True) -> str | bytes:
        """Two-hop read: docker compose cp from container -> SDK download from sandbox."""
        file = self._container_file(file)
        await verify_file_size(self._is_directory, self._get_file_size, file)

        temp = f"/tmp/.inspect-read-{uuid.uuid4().hex}"
        try:
            exit_code, output = await compose_exec(
                self.project,
                ["cp", f"{self.service}:{file}", temp],
                timeout=120,
            )
            if exit_code != 0:
                msg = output.lower()
                if "no such" in msg or "not found" in msg:
                    raise FileNotFoundError(
                        errno.ENOENT, "No such file or directory", file
                    )
                raise RuntimeError(
                    f"docker compose cp from {self.service}:{file} failed: {output}"
                )
            contents_bytes = await sdk_download(self.project.sandbox, temp)
        finally:
            await vm_exec(self.project.sandbox, f"rm -f {shlex.quote(temp)}", timeout=10)

        return decode_file_content(contents_bytes, file, text)

    def _container_file(self, file: str) -> str:
        """Resolve relative path against working directory."""
        path = PurePosixPath(file)
        if not path.is_absolute():
            path = PurePosixPath(self._working_dir) / path
        return str(path)

    async def _is_directory(self, path: str) -> bool:
        exit_code, _ = await compose_exec(
            self.project,
            ["exec", "-T", self.service, "test", "-d", path],
            timeout=10,
        )
        return exit_code == 0

    async def _get_file_size(self, path: str) -> int:
        # Use wc -c for portability — stat -c %s is GNU/BusyBox-specific
        exit_code, output = await compose_exec(
            self.project,
            ["exec", "-T", self.service, "sh", "-c", f"wc -c < {shlex.quote(path)}"],
            timeout=10,
        )
        if exit_code != 0:
            raise FileNotFoundError(
                errno.ENOENT, "No such file or directory", path
            )
        try:
            return int(output.strip())
        except ValueError as e:
            raise RuntimeError(f"Failed to parse file size for {path}") from e

    async def _create_parent_folder(self, path: str) -> None:
        exit_code, output = await compose_exec(
            self.project,
            ["exec", "-T", self.service, "mkdir", "-p", path],
            timeout=10,
        )
        if exit_code != 0:
            raise RuntimeError(f"Failed to create directory {path}: {output}")
