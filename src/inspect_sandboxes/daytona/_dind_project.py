"""DinD project orchestration"""

from __future__ import annotations

import asyncio
import json
import os
import shlex
import uuid
from copy import deepcopy
from dataclasses import dataclass, field
from logging import getLogger
from pathlib import Path
from typing import Any

import yaml
from daytona_sdk import (
    AsyncDaytona,
    AsyncSandbox,
    CreateSandboxFromImageParams,
    CreateSandboxFromSnapshotParams,
    CreateSnapshotParams,
    DaytonaNotFoundError,
    FileUpload,
    Image,
    Resources,
)
from inspect_ai.util import ComposeConfig, ComposeService
from inspect_ai.util._sandbox.docker.service import parse_duration

from ._retry import exec_retry, standard_retry
from ._sandbox_utils import create_sandbox, delete_sandbox, sdk_upload

logger = getLogger(__name__)

DIND_IMAGE = "docker:28.3.3-dind"
COMPOSE_DIR = "/inspect/compose"
BUILD_CONTEXT_DIR = "/inspect/contexts"
BUILD_TIMEOUT = 600

_DAEMON_POLL_INTERVAL = 2
_DAEMON_TIMEOUT = 60
_SERVICE_POLL_INTERVAL = 2
_SERVICE_TIMEOUT = 120


@dataclass
class DaytonaDinDProject:
    """Shared state for all per-service environments in one DinD sample."""

    sandbox: AsyncSandbox
    project_name: str
    compose_path: str
    services: list[str] = field(default_factory=list)


@exec_retry
async def vm_exec(
    sandbox: AsyncSandbox,
    command: str,
    timeout: int | None = 60,
) -> tuple[int, str]:
    """Execute a command on the DinD sandbox VM (not inside a compose service).

    Uses ``sh -c`` for Alpine compatibility. Returns (exit_code, output).
    """
    response = await sandbox.process.exec(
        f"sh -c {shlex.quote(command)}",
        timeout=timeout,
    )
    return response.exit_code, response.result


async def compose_exec(
    project: DaytonaDinDProject,
    subcommand: list[str],
    env: dict[str, str] | None = None,
    timeout: int | None = 60,
) -> tuple[int, str]:
    """Run a ``docker compose`` subcommand on the DinD sandbox.

    Returns (exit_code, output).
    """
    parts = [
        "docker",
        "compose",
        "-p",
        project.project_name,
        "--project-directory",
        COMPOSE_DIR,
        "-f",
        project.compose_path,
        *subcommand,
    ]
    cmd = shlex.join(parts)

    # Inline env vars as prefix for variable substitution in compose files
    if env:
        prefix = " ".join(f"{k}={shlex.quote(v)}" for k, v in env.items())
        cmd = f"{prefix} {cmd}"

    return await vm_exec(project.sandbox, cmd, timeout=timeout)


async def _wait_for_docker_daemon(sandbox: AsyncSandbox) -> None:
    """Poll ``docker info`` until the Docker daemon is responsive."""
    logger.debug("Waiting for Docker daemon inside DinD sandbox...")
    last_output = ""
    for _ in range(_DAEMON_TIMEOUT // _DAEMON_POLL_INTERVAL):
        exit_code, output = await vm_exec(sandbox, "docker info", timeout=10)
        if exit_code == 0:
            logger.debug("Docker daemon is ready.")
            return
        last_output = output
        await asyncio.sleep(_DAEMON_POLL_INTERVAL)

    raise RuntimeError(
        f"Docker daemon not ready after {_DAEMON_TIMEOUT}s. Last output: {last_output}"
    )


async def _wait_for_services(
    project: DaytonaDinDProject,
    expected: list[str],
    timeout: int = _SERVICE_TIMEOUT,
) -> None:
    """Poll ``docker compose ps`` until all expected services are running."""
    logger.debug("Waiting for compose services: %s", expected)
    last_output = ""
    for _ in range(timeout // _SERVICE_POLL_INTERVAL):
        exit_code, output = await compose_exec(
            project,
            ["ps", "--format", "json", "--status", "running"],
            timeout=15,
        )
        if exit_code == 0 and output.strip():
            # docker compose ps --format json outputs one JSON object per line
            running = set()
            for line in output.strip().splitlines():
                try:
                    entry = json.loads(line)
                    running.add(entry.get("Service", ""))
                except json.JSONDecodeError:
                    continue
            if set(expected) <= running:
                logger.debug("All services running: %s", running)
                return
        last_output = output
        await asyncio.sleep(_SERVICE_POLL_INTERVAL)

    raise RuntimeError(
        f"Not all services running after {timeout}s. "
        f"Expected: {expected}. Last output: {last_output}"
    )


@standard_retry
async def _upload_directory(
    sandbox: AsyncSandbox,
    local_dir: str | Path,
    remote_dir: str,
) -> None:
    """Upload a local directory to the sandbox recursively."""
    local_dir = Path(local_dir)
    uploads: list[FileUpload] = []

    for root, _, files in os.walk(local_dir):
        for filename in files:
            local_path = Path(root) / filename
            if not local_path.is_file():
                continue  # skip sockets, pipes, device files, etc.
            if not os.access(local_path, os.R_OK):
                continue  # skip unreadable files
            rel_path = local_path.relative_to(local_dir)
            remote_path = f"{remote_dir}/{rel_path.as_posix()}"
            uploads.append(FileUpload(source=str(local_path), destination=remote_path))

    if uploads:
        await sandbox.fs.upload_files(uploads)
        logger.debug(
            "Uploaded %d files from %s to %s", len(uploads), local_dir, remote_dir
        )


async def _upload_build_contexts(
    sandbox: AsyncSandbox,
    config: ComposeConfig,
    compose_file: str,
) -> str:
    """Upload compose file and all build contexts to the sandbox.

    If any service has a ``build.context`` that points to a local path
    outside the compose file's parent directory, that context is uploaded
    separately and the compose YAML is rewritten with the remote path.

    Returns the remote path to the compose file.
    """
    compose_path = Path(compose_file)
    compose_dir = compose_path.parent

    # Collect unique local build contexts that need uploading
    # Map: local_abs_path -> remote_path
    context_map: dict[str, str] = {}
    needs_rewrite = False

    for svc_name, service in config.services.items():
        if not service.build:
            continue

        ctx = service.build if isinstance(service.build, str) else service.build.context
        if not ctx:
            continue

        local_ctx = Path(ctx)
        if not local_ctx.is_absolute():
            local_ctx = (compose_dir / local_ctx).resolve()

        if not local_ctx.exists():
            logger.warning(
                "Build context '%s' for service '%s' does not exist, skipping upload",
                local_ctx,
                svc_name,
            )
            continue

        local_key = str(local_ctx)
        if local_key not in context_map:
            remote = f"{BUILD_CONTEXT_DIR}/{svc_name}"
            context_map[local_key] = remote

        # Check if this context is outside the compose dir
        try:
            local_ctx.relative_to(compose_dir)
        except ValueError:
            needs_rewrite = True

    # Upload the compose file's parent directory
    await _upload_directory(sandbox, compose_dir, COMPOSE_DIR)

    # Upload any external build contexts
    for local_path, remote_path in context_map.items():
        await _upload_directory(sandbox, local_path, remote_path)

    if not needs_rewrite:
        return f"{COMPOSE_DIR}/{compose_path.name}"

    # Rewrite compose YAML with remote context paths
    config_copy = deepcopy(config)
    for _, service in config_copy.services.items():
        if not service.build:
            continue

        ctx = service.build if isinstance(service.build, str) else service.build.context
        if not ctx:
            continue

        local_ctx = Path(ctx)
        if not local_ctx.is_absolute():
            local_ctx = (compose_dir / local_ctx).resolve()

        local_key = str(local_ctx)
        if local_key in context_map:
            remote = context_map[local_key]
            if isinstance(service.build, str):
                service.build = remote
            else:
                service.build.context = remote

    # Write rewritten compose YAML to sandbox
    data = config_copy.model_dump(
        by_alias=True, exclude_none=True, exclude_defaults=True
    )
    rewritten_yaml = yaml.dump(data, sort_keys=False)
    rewritten_remote = f"{COMPOSE_DIR}/compose.yaml"

    await sdk_upload(sandbox, rewritten_remote, rewritten_yaml.encode("utf-8"))
    logger.debug("Uploaded rewritten compose YAML to %s", rewritten_remote)

    return rewritten_remote


def _compute_healthcheck_timeout(
    services: dict[str, ComposeService],
    default: int = _SERVICE_TIMEOUT,
) -> int:
    """Compute the maximum wait time from compose healthcheck configs."""
    max_time = 0

    for _, service in services.items():
        hc = service.healthcheck
        if hc is None:
            continue
        retries = hc.retries if hc.retries is not None else 3
        interval = int(parse_duration(hc.interval).seconds) if hc.interval else 30
        timeout = int(parse_duration(hc.timeout).seconds) if hc.timeout else 30
        total_time = retries * (interval + timeout)
        max_time = max(max_time, total_time)

    return max_time if max_time > 0 else default


def _dind_snapshot_name(resources: Resources | None) -> str:
    """Derive a deterministic snapshot name from DinD image and resources."""
    cpu = resources.cpu if resources and resources.cpu else "default"
    mem = resources.memory if resources and resources.memory else "default"
    gpu = resources.gpu if resources and resources.gpu else 0
    return f"inspect-dind-{cpu}cpu-{mem}gb-{gpu}gpu"


async def _ensure_dind_snapshot(
    client: AsyncDaytona,
    snapshot_name: str,
    resources: Resources | None,
) -> str:
    """Ensure a DinD snapshot exists, creating it if needed.

    Returns the snapshot name on success, or empty string if snapshot
    creation is not available (falls back to image-based creation).
    """
    try:
        snapshot = await client.snapshot.get(snapshot_name)
        logger.debug(
            "Using existing DinD snapshot: %s (state=%s)",
            snapshot_name,
            snapshot.state,
        )
        return snapshot_name
    except DaytonaNotFoundError:
        pass  # doesn't exist yet — create below
    except Exception:
        pass  # check failed — still try to create

    try:
        await client.snapshot.create(
            CreateSnapshotParams(
                name=snapshot_name,
                image=Image.base(DIND_IMAGE),
                resources=resources,
            ),
        )
        logger.debug("Created DinD snapshot: %s", snapshot_name)
        return snapshot_name
    except Exception as e:
        logger.warning(
            "Failed to create DinD snapshot '%s': %s. "
            "Falling back to image-based sandbox creation.",
            snapshot_name,
            e,
        )
        return ""


async def create_dind_project(
    client: AsyncDaytona,
    config: ComposeConfig,
    compose_file: str,
    labels: dict[str, str],
    resources: Resources | None = None,
    sandbox_params: dict[str, Any] | None = None,
    snapshot: str | None = None,
) -> DaytonaDinDProject:
    """Create a DinD sandbox, start Docker, and bring up compose services.

    Args:
        client: Daytona client.
        config: Parsed compose configuration.
        compose_file: Local path to the compose file.
        labels: Labels to apply to the sandbox.
        resources: Resource limits for the DinD sandbox.
        sandbox_params: Extra params from x-daytona extensions.
        snapshot: Explicit snapshot name override. If None, auto-creates one.
    """
    project_name = f"inspect-{uuid.uuid4().hex[:8]}"

    extra = {**(sandbox_params or {})}
    extra.setdefault("auto_stop_interval", 0)

    # 1. Resolve snapshot
    if snapshot is None:
        snapshot = await _ensure_dind_snapshot(
            client, _dind_snapshot_name(resources), resources
        )

    # 2. Create sandbox from snapshot or image
    params: CreateSandboxFromSnapshotParams | CreateSandboxFromImageParams
    if snapshot:
        params = CreateSandboxFromSnapshotParams(
            snapshot=snapshot,
            labels=labels,
            network_block_all=False,
            **extra,
        )
    else:
        params = CreateSandboxFromImageParams(
            image=Image.base(DIND_IMAGE),
            resources=resources,
            labels=labels,
            network_block_all=False,
            **extra,
        )

    sandbox = await create_sandbox(client, params)
    logger.debug("Created DinD sandbox %s", sandbox.id)

    try:
        # 3. Start Docker daemon
        await vm_exec(
            sandbox,
            "dockerd-entrypoint.sh dockerd > /var/log/dockerd.log 2>&1 &",
            timeout=10,
        )

        # 4. Wait for daemon
        await _wait_for_docker_daemon(sandbox)

        # 5. Upload build contexts and compose file
        compose_remote_path = await _upload_build_contexts(
            sandbox, config, compose_file
        )

        project = DaytonaDinDProject(
            sandbox=sandbox,
            project_name=project_name,
            compose_path=compose_remote_path,
        )

        # 5. Build services
        logger.debug("Building compose services in DinD sandbox %s...", sandbox.id)
        exit_code, output = await compose_exec(
            project, ["build"], timeout=BUILD_TIMEOUT
        )
        if exit_code != 0:
            raise RuntimeError(f"docker compose build failed:\n{output}")

        # 6. Start services
        healthcheck_timeout = _compute_healthcheck_timeout(config.services)
        logger.debug("Starting compose services in DinD sandbox %s...", sandbox.id)
        exit_code, output = await compose_exec(
            project,
            ["up", "--detach", "--wait", "--wait-timeout", str(healthcheck_timeout)],
            timeout=healthcheck_timeout + 30,
        )
        if exit_code != 0:
            raise RuntimeError(f"docker compose up failed:\n{output}")

        # 7. Verify services are running
        expected_services = list(config.services.keys())
        await _wait_for_services(
            project, expected_services, timeout=healthcheck_timeout
        )
        project.services = expected_services

        return project

    except BaseException:
        # Clean up on any failure
        try:
            await delete_sandbox(client, sandbox)
        except Exception as cleanup_err:
            logger.warning(
                "Failed to clean up DinD sandbox %s: %s", sandbox.id, cleanup_err
            )
        raise


async def destroy_dind_project(project: DaytonaDinDProject) -> None:
    """Tear down compose services inside the DinD sandbox.

    Runs ``docker compose down`` best-effort. The caller is responsible
    for deleting the Daytona sandbox afterwards.
    """
    try:
        exit_code, output = await compose_exec(
            project,
            ["down", "--remove-orphans", "--timeout", "10"],
            timeout=30,
        )
        if exit_code != 0:
            logger.warning("docker compose down failed: %s", output)
    except Exception as e:
        logger.warning("docker compose down error: %s", e)


async def discover_working_dir(
    project: DaytonaDinDProject,
    service: str,
) -> str:
    """Discover a service's working directory via ``pwd``.

    Returns ``/`` if the query fails (e.g. service has no shell).
    """
    exit_code, output = await compose_exec(
        project,
        ["exec", "-T", service, "pwd"],
        timeout=10,
    )
    if exit_code == 0 and output.strip():
        return output.strip()

    logger.warning(
        "Failed to get working directory for service '%s', defaulting to /", service
    )
    return "/"
