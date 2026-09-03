"""DinD project orchestration for the Runloop provider.

Runloop devboxes don't run a Docker daemon by default — there's no init
system, so ``service docker status`` reports "not running". For
multi-service compose, we:

1. Build (or reuse) a cached DinD blueprint that installs a pinned
   ``docker-ce`` from Docker's official apt repo on top of Ubuntu.
2. Create a devbox from that blueprint.
3. ``sudo nohup dockerd > /tmp/dockerd.log 2>&1 &`` to start the daemon.
4. Poll ``sudo docker info`` until the daemon is up (≤60 s).
5. Upload build contexts + the compose file.
6. ``docker compose build`` then ``docker compose up --wait``.
7. Verify services are running.

Each compose service is then exposed as a ``RunloopDinDServiceEnvironment``.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import shlex
import uuid
from dataclasses import dataclass, field
from logging import getLogger
from pathlib import Path

import httpx
from inspect_ai.util import ComposeConfig
from runloop_api_client import AsyncRunloop
from runloop_api_client.types.shared_params import LaunchParameters

from inspect_sandboxes._util.dind_compose import (
    compute_healthcheck_timeout,
    discover_build_contexts,
    rewrite_compose_yaml,
)

from ._blueprint import (
    _BLUEPRINT_POLLING_CONFIG,
    BLUEPRINT_NAME_PREFIX,
    _find_or_await_blueprint,
    _hash_inputs,
    _launch_params_for_hash,
)
from ._retry import exec_retry, poll_execution, standard_retry
from ._single_env import (
    EXEC_LAST_N,
    FILE_REQUEST_TIMEOUT,
    LARGE_FILE_THRESHOLD,
    _devbox_download_command,
    _devbox_upload_command,
)

logger = getLogger(__name__)

COMPOSE_DIR = "/home/user/inspect/compose"
BUILD_CONTEXT_DIR = "/home/user/inspect/contexts"
BUILD_TIMEOUT = 600

_DAEMON_POLL_INTERVAL = 2
_DAEMON_TIMEOUT = 60
_SERVICE_POLL_INTERVAL = 2
_SERVICE_TIMEOUT = 120

# Pinned so cold rebuilds produce byte-identical blueprints. Bump to refresh.
DOCKER_CE_VERSION = "5:29.5.2-1~ubuntu.24.04~noble"

# The Dockerfile used to build the cached DinD blueprint. Installs a pinned
# ``docker-ce`` from Docker's official apt repo so cold rebuilds produce
# byte-identical blueprints. The hash of this content (plus launch
# parameters) determines the blueprint name.
_DIND_DOCKERFILE = f"""\
FROM ubuntu:24.04
RUN apt-get update \\
    && apt-get install -y --no-install-recommends \\
        ca-certificates curl gnupg sudo \\
    && rm -rf /var/lib/apt/lists/*
RUN install -m 0755 -d /etc/apt/keyrings \\
    && curl -fsSL https://download.docker.com/linux/ubuntu/gpg \\
        | gpg --dearmor -o /etc/apt/keyrings/docker.gpg \\
    && chmod a+r /etc/apt/keyrings/docker.gpg \\
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu noble stable" \\
        > /etc/apt/sources.list.d/docker.list \\
    && apt-get update \\
    && apt-get install -y --no-install-recommends \\
        docker-ce={DOCKER_CE_VERSION} \\
        docker-ce-cli={DOCKER_CE_VERSION} \\
        containerd.io docker-buildx-plugin docker-compose-plugin \\
    && rm -rf /var/lib/apt/lists/*
"""


@dataclass
class RunloopDinDProject:
    """Shared state for all per-service environments in one DinD sample."""

    client: AsyncRunloop
    devbox_id: str
    project_name: str
    compose_path: str
    services: list[str] = field(default_factory=list)


@exec_retry
async def _submit_vm_exec(client: AsyncRunloop, devbox_id: str, wrapped: str) -> str:
    """Submit a command to the devbox, returning its execution id.

    Only the submission is retried; ``vm_exec`` polls separately so a transient
    mid-poll error never re-runs (and thus double-executes) the command.
    """
    started = await client.devboxes.execute_async(devbox_id, command=wrapped)
    return started.execution_id


async def vm_exec(
    client: AsyncRunloop,
    devbox_id: str,
    command: str,
    timeout: int | None = 60,
) -> tuple[int, str, str]:
    """Execute a command on the DinD devbox VM (not inside a compose service).

    Wraps with ``sh -c`` so shell features (pipes, &&, redirects) work.
    Returns ``(exit_code, stdout, stderr)``.
    """
    wrapped = f"sh -c {shlex.quote(command)}"
    # Submit once (retried on transient errors), then poll ourselves — the
    # SDK's long-poll await has a server-side minimum wait that doesn't respect
    # short user timeouts.
    execution_id = await _submit_vm_exec(client, devbox_id, wrapped)
    response = await poll_execution(
        client, devbox_id, execution_id, timeout, last_n=EXEC_LAST_N
    )
    return (
        response.exit_status if response.exit_status is not None else 0,
        response.stdout or "",
        response.stderr or "",
    )


async def compose_exec(
    project: RunloopDinDProject,
    subcommand: list[str],
    *,
    env: dict[str, str] | None = None,
    timeout: int | None = 60,
) -> tuple[int, str, str]:
    """Run a ``docker compose`` subcommand on the DinD Devbox.

    Returns (exit_code, stdout, stderr).
    """
    parts = [
        "sudo",
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

    if env:
        prefix = " ".join(f"{k}={shlex.quote(v)}" for k, v in env.items())
        cmd = f"{prefix} {cmd}"

    return await vm_exec(project.client, project.devbox_id, cmd, timeout=timeout)


async def _wait_for_docker_daemon(client: AsyncRunloop, devbox_id: str) -> None:
    """Poll ``docker info`` until the Docker daemon is responsive."""
    logger.debug("Waiting for Docker daemon inside DinD devbox...")
    last_output = ""
    for _ in range(_DAEMON_TIMEOUT // _DAEMON_POLL_INTERVAL):
        exit_code, stdout, stderr = await vm_exec(
            client, devbox_id, "sudo docker info", timeout=10
        )
        if exit_code == 0:
            logger.debug("Docker daemon is ready.")
            return
        last_output = stdout or stderr
        await asyncio.sleep(_DAEMON_POLL_INTERVAL)

    _, daemon_log, _ = await vm_exec(
        client,
        devbox_id,
        "tail -20 /tmp/dockerd.log 2>/dev/null || true",
        timeout=5,
    )
    raise RuntimeError(
        f"Docker daemon not ready after {_DAEMON_TIMEOUT}s.\n"
        f"Last 'docker info' output: {last_output}\n"
        f"dockerd log:\n{daemon_log}"
    )


async def _wait_for_services(
    project: RunloopDinDProject,
    expected: list[str],
    timeout: int = _SERVICE_TIMEOUT,
) -> None:
    """Poll ``docker compose ps`` until all expected services are running."""
    logger.debug("Waiting for compose services: %s", expected)
    last_output = ""
    for _ in range(timeout // _SERVICE_POLL_INTERVAL):
        exit_code, stdout, _ = await compose_exec(
            project,
            ["ps", "--format", "json", "--status", "running"],
            timeout=15,
        )
        if exit_code == 0 and stdout.strip():
            running: set[str] = set()
            for line in stdout.strip().splitlines():
                try:
                    entry = json.loads(line)
                    running.add(entry.get("Service", ""))
                except json.JSONDecodeError:
                    continue
            if set(expected) <= running:
                logger.debug("All services running: %s", running)
                return
        last_output = stdout
        await asyncio.sleep(_SERVICE_POLL_INTERVAL)

    raise RuntimeError(
        f"Not all services running after {timeout}s. "
        f"Expected: {expected}. Last output: {last_output}"
    )


async def _upload_file(
    client: AsyncRunloop, devbox_id: str, remote_path: str, data: bytes
) -> None:
    """Upload a single file to the devbox. Creates parent directories as needed.

    Small files round-trip via base64-over-shell (one exec call); files at or
    above ``LARGE_FILE_THRESHOLD`` go through the Runloop Objects API to avoid
    the request-body cap on the exec endpoint.
    """
    if len(data) >= LARGE_FILE_THRESHOLD:
        await _upload_file_via_object(client, devbox_id, remote_path, data)
        return
    await _upload_file_via_shell(client, devbox_id, remote_path, data)


@standard_retry
async def _upload_file_via_shell(
    client: AsyncRunloop, devbox_id: str, remote_path: str, data: bytes
) -> None:
    encoded = base64.b64encode(data).decode("ascii")
    parent = remote_path.rsplit("/", 1)[0] if "/" in remote_path else ""
    mkdir = f"mkdir -p {shlex.quote(parent)} && " if parent else ""
    cmd = f"{mkdir}echo {shlex.quote(encoded)} | base64 -d > {shlex.quote(remote_path)}"
    exit_code, _, stderr = await vm_exec(
        client, devbox_id, cmd, timeout=FILE_REQUEST_TIMEOUT
    )
    if exit_code != 0:
        raise RuntimeError(
            f"Failed to upload {remote_path}: exit={exit_code} stderr={stderr!r}"
        )


async def _upload_file_via_object(
    client: AsyncRunloop, devbox_id: str, remote_path: str, data: bytes
) -> None:
    """Route the bytes through a Runloop Object (presigned PUT + devbox curl)."""
    obj = await client.objects.create(
        content_type="binary",
        name=f"inspect-vmwrite-{uuid.uuid4().hex[:12]}",
    )
    if obj.upload_url is None:
        raise RuntimeError("Runloop did not return an upload URL for the write Object.")
    try:
        async with httpx.AsyncClient(timeout=FILE_REQUEST_TIMEOUT) as http:
            put = await http.put(obj.upload_url, content=data)
            put.raise_for_status()
        await client.objects.complete(obj.id)
        download = await client.objects.download(obj.id)
        parent = remote_path.rsplit("/", 1)[0] if "/" in remote_path else ""
        mkdir = f"mkdir -p {shlex.quote(parent)} && " if parent else ""
        cmd = f"{mkdir}{_devbox_download_command(download.download_url, remote_path)}"
        exit_code, _, stderr = await vm_exec(
            client, devbox_id, cmd, timeout=FILE_REQUEST_TIMEOUT
        )
        if exit_code != 0:
            raise RuntimeError(
                f"Failed to upload {remote_path}: exit={exit_code} stderr={stderr!r}"
            )
    finally:
        try:
            await client.objects.delete(obj.id)
        except Exception:
            pass


async def _download_file(
    client: AsyncRunloop, devbox_id: str, remote_path: str, *, size: int
) -> bytes:
    """Download a single file from the devbox.

    Small files round-trip via base64-over-shell (one exec call); files at or
    above ``LARGE_FILE_THRESHOLD`` go through the Runloop Objects API.
    """
    if size >= LARGE_FILE_THRESHOLD:
        return await _download_file_via_object(client, devbox_id, remote_path)
    return await _download_file_via_shell(client, devbox_id, remote_path)


@standard_retry
async def _download_file_via_shell(
    client: AsyncRunloop, devbox_id: str, remote_path: str
) -> bytes:
    exit_code, stdout, stderr = await vm_exec(
        client,
        devbox_id,
        f"base64 -w 0 {shlex.quote(remote_path)}",
        timeout=FILE_REQUEST_TIMEOUT,
    )
    if exit_code != 0:
        raise RuntimeError(
            f"Failed to read {remote_path}: exit={exit_code} stderr={stderr!r}"
        )
    return base64.b64decode(stdout.strip())


async def _download_file_via_object(
    client: AsyncRunloop, devbox_id: str, remote_path: str
) -> bytes:
    """Route the bytes through a Runloop Object (devbox curls PUT, we GET)."""
    obj = await client.objects.create(
        content_type="binary",
        name=f"inspect-vmread-{uuid.uuid4().hex[:12]}",
    )
    if obj.upload_url is None:
        raise RuntimeError("Runloop did not return an upload URL for the read Object.")
    try:
        cmd = _devbox_upload_command(remote_path, obj.upload_url)
        exit_code, _, stderr = await vm_exec(
            client, devbox_id, cmd, timeout=FILE_REQUEST_TIMEOUT
        )
        if exit_code != 0:
            raise RuntimeError(
                f"Failed to upload {remote_path} to object store: "
                f"exit={exit_code} stderr={stderr!r}"
            )
        await client.objects.complete(obj.id)
        download = await client.objects.download(obj.id)
        async with httpx.AsyncClient(timeout=FILE_REQUEST_TIMEOUT) as http:
            response = await http.get(download.download_url)
            response.raise_for_status()
            return response.content
    finally:
        try:
            await client.objects.delete(obj.id)
        except Exception:
            pass


async def _upload_directory(
    client: AsyncRunloop,
    devbox_id: str,
    local_dir: str | Path,
    remote_dir: str,
) -> None:
    """Upload a local directory to the devbox recursively."""
    local_dir = Path(local_dir)
    count = 0
    for root, _, files in os.walk(local_dir):
        for filename in files:
            local_path = Path(root) / filename
            if not local_path.is_file():
                continue
            if not os.access(local_path, os.R_OK):
                continue
            rel_path = local_path.relative_to(local_dir)
            remote_path = f"{remote_dir}/{rel_path.as_posix()}"
            data = local_path.read_bytes()
            await _upload_file(client, devbox_id, remote_path, data)
            count += 1
    if count:
        logger.debug("Uploaded %d files from %s to %s", count, local_dir, remote_dir)


async def _upload_build_contexts(
    client: AsyncRunloop,
    devbox_id: str,
    config: ComposeConfig,
    compose_file: str,
) -> str:
    """Upload compose file and all build contexts to the devbox.

    Returns the remote path to the (possibly-rewritten) compose file.
    """
    compose_path = Path(compose_file)
    compose_dir = compose_path.parent

    context_map, needs_rewrite = discover_build_contexts(
        config, compose_dir, BUILD_CONTEXT_DIR
    )

    await _upload_directory(client, devbox_id, compose_dir, COMPOSE_DIR)
    for local_path, remote_path in context_map.items():
        await _upload_directory(client, devbox_id, local_path, remote_path)

    if not needs_rewrite:
        return f"{COMPOSE_DIR}/{compose_path.name}"

    rewritten_yaml = rewrite_compose_yaml(config, compose_dir, context_map)
    rewritten_remote = f"{COMPOSE_DIR}/compose.yaml"
    await _upload_file(
        client, devbox_id, rewritten_remote, rewritten_yaml.encode("utf-8")
    )
    logger.debug("Uploaded rewritten compose YAML to %s", rewritten_remote)
    return rewritten_remote


def _dind_blueprint_name(launch_parameters: LaunchParameters | None = None) -> str:
    """Deterministic blueprint name for the DinD VM, keyed by launch parameters.

    Hash includes the DinD Dockerfile content so any change to the base
    image / install steps produces a new blueprint name.
    """
    h = _hash_inputs(
        {
            "kind": "dind",
            "content": _DIND_DOCKERFILE,
            "launch_parameters": _launch_params_for_hash(launch_parameters),
        }
    )
    return f"{BLUEPRINT_NAME_PREFIX}dind-{h}"


async def _ensure_dind_blueprint(
    client: AsyncRunloop,
    *,
    launch_parameters: LaunchParameters | None = None,
) -> str:
    """Build (or reuse cached) DinD blueprint. Returns the blueprint name."""
    name = _dind_blueprint_name(launch_parameters)
    if await _find_or_await_blueprint(client, name):
        logger.debug("Using existing DinD blueprint: %s", name)
        return name
    # POST without SDK retries: Runloop's blueprint create is not idempotent,
    # so retries spawn duplicate blueprints (and blow the account cap).
    # Polling uses the default client so transient retrieve errors are still
    # retried.
    blueprint = await client.with_options(max_retries=0).blueprints.create(
        name=name,
        dockerfile=_DIND_DOCKERFILE,
        launch_parameters=launch_parameters or {},
        idempotency_key=name,
    )
    await client.blueprints.await_build_complete(
        blueprint.id, polling_config=_BLUEPRINT_POLLING_CONFIG
    )
    logger.debug("Built DinD blueprint: %s", name)
    return name


async def create_dind_project(
    client: AsyncRunloop,
    config: ComposeConfig,
    compose_file: str,
    *,
    name: str | None = None,
    metadata: dict[str, str],
    launch_parameters: LaunchParameters | None = None,
    environment_variables: dict[str, str] | None = None,
    timeout: float | None = None,
) -> RunloopDinDProject:
    """Create a DinD devbox, start dockerd, and bring up compose services.

    Args:
        client: Runloop client.
        config: Parsed compose configuration.
        compose_file: Local path to the compose file.
        name: Human-readable devbox name to assign.
        metadata: Metadata to apply to the devbox.
        launch_parameters: Runloop ``LaunchParameters`` (CPU/RAM/disk overrides)
            applied to the devbox.
        environment_variables: Environment variables set on the devbox VM (not
            on individual compose services).
        timeout: Per-request HTTP timeout forwarded to
            ``devboxes.create_and_await_running``.
    """
    project_name = f"inspect-{uuid.uuid4().hex[:8]}"

    # 1. Ensure DinD blueprint exists.
    blueprint_name = await _ensure_dind_blueprint(
        client, launch_parameters=launch_parameters
    )

    # 2. Create devbox from blueprint.
    create_kwargs: dict[str, object] = {
        "blueprint_name": blueprint_name,
        "metadata": metadata,
    }
    if name is not None:
        create_kwargs["name"] = name
    if launch_parameters is not None:
        create_kwargs["launch_parameters"] = launch_parameters
    if environment_variables:
        create_kwargs["environment_variables"] = environment_variables
    if timeout is not None:
        create_kwargs["timeout"] = timeout

    devbox = await client.devboxes.create_and_await_running(**create_kwargs)  # type: ignore[arg-type]
    logger.debug("Created DinD devbox %s", devbox.id)

    try:
        # 3. Start Docker daemon. Runloop devboxes have no init system.
        await vm_exec(
            client,
            devbox.id,
            "sudo nohup dockerd > /tmp/dockerd.log 2>&1 &",
            timeout=10,
        )

        # 4. Wait for daemon.
        await _wait_for_docker_daemon(client, devbox.id)

        # 5. Upload build contexts and compose file.
        compose_remote_path = await _upload_build_contexts(
            client, devbox.id, config, compose_file
        )

        project = RunloopDinDProject(
            client=client,
            devbox_id=devbox.id,
            project_name=project_name,
            compose_path=compose_remote_path,
        )

        # 6. Build compose services.
        logger.debug("Building compose services in DinD devbox %s...", devbox.id)
        exit_code, stdout, stderr = await compose_exec(
            project, ["build"], timeout=BUILD_TIMEOUT
        )
        if exit_code != 0:
            raise RuntimeError(
                f"docker compose build failed:\nstdout: {stdout}\nstderr: {stderr}"
            )

        # 7. Start services.
        healthcheck_timeout = compute_healthcheck_timeout(config.services)
        logger.debug("Starting compose services in DinD devbox %s...", devbox.id)
        exit_code, stdout, stderr = await compose_exec(
            project,
            [
                "up",
                "--detach",
                "--wait",
                "--wait-timeout",
                str(healthcheck_timeout),
            ],
            timeout=healthcheck_timeout + 30,
        )
        if exit_code != 0:
            raise RuntimeError(
                f"docker compose up failed:\nstdout: {stdout}\nstderr: {stderr}"
            )

        # 8. Verify all expected services are running.
        expected_services = list(config.services.keys())
        await _wait_for_services(
            project, expected_services, timeout=healthcheck_timeout
        )
        project.services = expected_services

        return project

    except BaseException:
        try:
            await client.devboxes.shutdown(devbox.id)
        except Exception as cleanup_err:
            logger.warning(
                "Failed to shut down DinD devbox %s: %s", devbox.id, cleanup_err
            )
        raise


async def destroy_dind_project(project: RunloopDinDProject) -> None:
    """Tear down compose services inside the DinD Devbox.

    Runs ``docker compose down`` best-effort. The caller is responsible
    for shutting down the Runloop Devbox afterwards.
    """
    try:
        exit_code, stdout, _ = await compose_exec(
            project,
            ["down", "--remove-orphans", "--timeout", "10"],
            timeout=30,
        )
        if exit_code != 0:
            logger.warning("docker compose down failed: %s", stdout)
    except Exception as e:
        logger.warning("docker compose down error: %s", e)


async def discover_working_dir(
    project: RunloopDinDProject,
    service: str,
) -> str:
    """Discover a service's working directory via ``pwd``.

    Returns ``/`` if the query fails (e.g. service has no shell).
    """
    exit_code, stdout, _ = await compose_exec(
        project, ["exec", "-T", service, "pwd"], timeout=10
    )
    if exit_code == 0 and stdout.strip():
        return stdout.strip()
    logger.warning(
        "Failed to get working directory for service '%s', defaulting to /", service
    )
    return "/"
