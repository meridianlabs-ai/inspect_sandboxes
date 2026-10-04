"""DinD project orchestration for the Runloop provider.

Runloop devboxes don't run a Docker daemon by default — there's no init
system, so ``service docker status`` reports "not running". For
multi-service compose, we build the stack once and snapshot it, then create a
devbox per sample from that snapshot:

1. Build (or reuse) a cached DinD blueprint that installs a pinned
   ``docker-ce`` from Docker's official apt repo on top of Ubuntu.
2. The first sample for a given compose content-hash creates a devbox from the
   blueprint, starts dockerd, uploads the build contexts + compose file,
   ``docker compose build``, and ``snapshot_disk``s the result — then keeps that
   devbox for its own ``compose up``.
3. Every later sample with the same content-hash creates a devbox from the
   snapshot (docker images already built) and runs only ``docker compose up``,
   skipping the upload and build.

A per-``compose``-hash lock serializes the one-time build; a fixed project name
keeps the pre-built image tags matching. Each compose service is then exposed as
a ``RunloopDinDServiceEnvironment``.
"""

from __future__ import annotations

import asyncio
import shlex
import tempfile
import uuid
from contextvars import ContextVar
from dataclasses import dataclass, field
from logging import getLogger
from pathlib import Path
from typing import IO

from inspect_ai.util import ComposeConfig
from runloop_api_client import AsyncRunloop, BadRequestError
from runloop_api_client.types.shared_params import LaunchParameters
from uuid_utils import uuid7

from inspect_sandboxes._util.dind_compose import (
    compute_healthcheck_timeout,
    discover_build_contexts,
)
from inspect_sandboxes._util.dind_project import (
    build_compose_command,
    compose_down,
    upload_build_contexts,
    wait_for_docker_daemon,
    wait_for_services,
)
from inspect_sandboxes._util.dind_project import (
    discover_working_dir as _discover_working_dir,
)
from inspect_sandboxes._util.hashing import hash_inputs

from ._blueprint import (
    _BLUEPRINT_POLLING_CONFIG,
    BLUEPRINT_NAME_PREFIX,
    _find_or_await_blueprint,
    _hash_build_context,
    _launch_params_for_hash,
    _write_context_tarball,
    blueprint_build_lock,
)
from ._retry import (
    DEVBOX_CREATE_POLLING_CONFIG,
    create_devbox,
    execute_with_poll,
    shutdown_devbox,
    standard_retry,
)
from ._single_env import (
    EXEC_LAST_N,
    FILE_REQUEST_TIMEOUT,
    _raise_filesystem_error,
    _verify_exec_output_size,
)

logger = getLogger(__name__)

COMPOSE_DIR = "/home/user/inspect/compose"
BUILD_CONTEXT_DIR = "/home/user/inspect/contexts"
BUILD_TIMEOUT = 600

# Fixed compose project name. `docker compose build` tags a build-only service's
# image as ``<project>-<service>``, so the build (done once, into the snapshot)
# and every per-sample ``compose up`` must share a project name for the pre-built
# image to be reused. Each sample runs on its own devbox, so a constant is safe.
_PROJECT_NAME = "inspect"

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


async def vm_exec(
    client: AsyncRunloop,
    devbox_id: str,
    command: str,
    timeout: int | None = 60,
    raise_on_truncation: bool = False,
) -> tuple[int, str, str]:
    """Execute a command on the DinD devbox VM (not inside a compose service).

    Wraps with ``sh -c`` so shell features (pipes, &&, redirects) work.
    Returns ``(exit_code, stdout, stderr)``. Internal callers keep
    ``raise_on_truncation=False`` (a long ``docker build`` log is fine to
    truncate); the user-facing ``exec`` path sets it so an over-limit command
    raises ``OutputLimitExceededError`` instead of silently dropping output.
    """
    wrapped = f"sh -c {shlex.quote(command)}"
    # Stable command_id (generated once) lets a retried submit dedupe.
    command_id = str(uuid7())
    response = await execute_with_poll(
        client, devbox_id, wrapped, command_id, timeout, last_n=EXEC_LAST_N
    )
    if raise_on_truncation:
        _verify_exec_output_size(response)
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
    raise_on_truncation: bool = False,
) -> tuple[int, str, str]:
    """Run a ``docker compose`` subcommand on the DinD Devbox.

    Returns (exit_code, stdout, stderr). ``raise_on_truncation`` is forwarded to
    ``vm_exec`` so a user-facing ``exec`` whose output overflows raises rather
    than silently truncating.
    """
    cmd = build_compose_command(
        project.project_name, COMPOSE_DIR, project.compose_path, subcommand, env=env
    )
    return await vm_exec(
        project.client,
        project.devbox_id,
        cmd,
        timeout=timeout,
        raise_on_truncation=raise_on_truncation,
    )


async def _wait_for_docker_daemon(client: AsyncRunloop, devbox_id: str) -> None:
    await wait_for_docker_daemon(
        lambda cmd, t: vm_exec(client, devbox_id, cmd, timeout=t),
        info_cmd="sudo docker info",
        timeout=_DAEMON_TIMEOUT,
        poll_interval=_DAEMON_POLL_INTERVAL,
        log_tail_cmd="tail -20 /tmp/dockerd.log 2>/dev/null || true",
    )


async def _wait_for_services(
    project: RunloopDinDProject,
    expected: list[str],
    timeout: int = _SERVICE_TIMEOUT,
) -> None:
    await wait_for_services(
        compose_exec,
        project,
        expected,
        timeout=timeout,
        poll_interval=_SERVICE_POLL_INTERVAL,
    )


@standard_retry
async def _upload_file(
    client: AsyncRunloop, devbox_id: str, remote_path: str, data: bytes
) -> None:
    """Upload a single file to the devbox, creating parent directories as needed.

    ``upload_file`` sends binary content of any size via multipart form data but
    writes only the file, so the parent directory is created first.
    """
    parent = remote_path.rsplit("/", 1)[0] if "/" in remote_path else ""
    if parent:
        await vm_exec(
            client,
            devbox_id,
            f"mkdir -p {shlex.quote(parent)}",
            timeout=FILE_REQUEST_TIMEOUT,
        )
    try:
        await client.devboxes.upload_file(
            devbox_id, path=remote_path, file=data, timeout=FILE_REQUEST_TIMEOUT
        )
    except BadRequestError as e:
        _raise_filesystem_error(e, remote_path)


@standard_retry
async def _download_file(
    client: AsyncRunloop, devbox_id: str, remote_path: str
) -> bytes:
    """Download a single file from the devbox.

    ``download_file`` streams binary content of any size.
    """
    try:
        response = await client.devboxes.download_file(
            devbox_id, path=remote_path, timeout=FILE_REQUEST_TIMEOUT
        )
    except BadRequestError as e:
        _raise_filesystem_error(e, remote_path)
    return await response.read()


@standard_retry
async def _upload_tarball(
    client: AsyncRunloop, devbox_id: str, remote_path: str, fileobj: IO[bytes]
) -> None:
    """Upload an already-written tarball (a file object, streamed not buffered)."""
    fileobj.seek(0)
    await client.devboxes.upload_file(
        devbox_id, path=remote_path, file=fileobj, timeout=FILE_REQUEST_TIMEOUT
    )


async def _upload_directory(
    client: AsyncRunloop,
    devbox_id: str,
    local_dir: str | Path,
    remote_dir: str,
) -> None:
    """Tar the local dir once, upload the archive, and extract it on the devbox.

    Shipping a single archive turns an O(files) sequence of per-file uploads
    into a constant handful of API calls, and reuses the blueprint context's
    ignore rules (``.git``, ``logs``, ``.venv``, …) so a compose ``build:``
    context is filtered the same way as a Dockerfile context.
    """
    local_dir = Path(local_dir)
    remote_tar = f"/tmp/.inspect-ctx-{uuid.uuid4().hex}.tgz"
    with tempfile.NamedTemporaryFile(suffix=".tgz") as tmp:
        # tar+gzip of the whole dir is blocking CPU/IO; keep it off the loop.
        await asyncio.to_thread(_write_context_tarball, local_dir, tmp)
        tmp.flush()
        # Pass the underlying io object (the SDK rejects the tempfile wrapper).
        await _upload_tarball(client, devbox_id, remote_tar, tmp.file)
    exit_code, _, stderr = await vm_exec(
        client,
        devbox_id,
        f"mkdir -p {shlex.quote(remote_dir)} "
        f"&& tar -xzf {shlex.quote(remote_tar)} -C {shlex.quote(remote_dir)} "
        f"&& rm -f {shlex.quote(remote_tar)}",
        timeout=FILE_REQUEST_TIMEOUT,
    )
    if exit_code != 0:
        raise RuntimeError(
            f"Failed to unpack build context into {remote_dir}: "
            f"exit={exit_code} stderr={stderr!r}"
        )


async def _upload_build_contexts(
    client: AsyncRunloop,
    devbox_id: str,
    config: ComposeConfig,
    compose_file: str,
) -> str:
    """Upload compose file and all build contexts to the devbox.

    Returns the remote path to the (possibly-rewritten) compose file.
    """
    return await upload_build_contexts(
        config,
        compose_file,
        COMPOSE_DIR,
        BUILD_CONTEXT_DIR,
        upload_directory=lambda local, remote: _upload_directory(
            client, devbox_id, local, remote
        ),
        upload_file=lambda remote, data: _upload_file(client, devbox_id, remote, data),
    )


def _dind_blueprint_name(launch_parameters: LaunchParameters | None = None) -> str:
    """Deterministic blueprint name for the DinD VM, keyed by launch parameters.

    Hash includes the DinD Dockerfile content so any change to the base
    image / install steps produces a new blueprint name.
    """
    h = hash_inputs(
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
    async with blueprint_build_lock(name):
        if await _find_or_await_blueprint(client, name):
            logger.debug("Using existing DinD blueprint: %s", name)
            return name
        # POST without SDK retries: Runloop's blueprint create is not
        # idempotent, so retries spawn duplicate blueprints (and blow the
        # account cap). Polling uses the default client so transient retrieve
        # errors are still retried.
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


@dataclass
class _DinDSnapshotInfo:
    """The reusable product of building a compose stack once."""

    snapshot_id: str
    compose_remote_path: str
    expected_services: list[str]


# Per-run cache of built compose-stack snapshots, keyed by
# hash(compose dir + build contexts + launch params). The first sample builds
# the stack once and snapshots it; the rest create devboxes from the snapshot
# and skip the upload + `compose build` entirely. Reset per run by task_init via
# reset_dind_snapshot_cache().
_dind_snapshot_cache: ContextVar[dict[str, _DinDSnapshotInfo]] = ContextVar(
    "runloop_dind_snapshot_cache"
)
# Serialize the build+snapshot for a given key so concurrent first samples don't
# each build (Runloop snapshots, like blueprints, aren't name-deduped).
_snapshot_build_locks: dict[str, asyncio.Lock] = {}


def reset_dind_snapshot_cache() -> None:
    """Start a fresh per-run DinD snapshot cache."""
    _dind_snapshot_cache.set({})


def _cached_dind_snapshot(key: str) -> _DinDSnapshotInfo | None:
    cache = _dind_snapshot_cache.get(None)
    return cache.get(key) if cache is not None else None


def _cache_dind_snapshot(key: str, info: _DinDSnapshotInfo) -> None:
    cache = _dind_snapshot_cache.get(None)
    if cache is not None:  # unset when there was no task_init — skip
        cache[key] = info


def dind_snapshot_ids() -> list[str]:
    """Snapshot ids created this run, for task_cleanup to delete."""
    cache = _dind_snapshot_cache.get(None)
    return [info.snapshot_id for info in cache.values()] if cache is not None else []


def _snapshot_build_lock(key: str) -> asyncio.Lock:
    lock = _snapshot_build_locks.get(key)
    if lock is None:
        lock = asyncio.Lock()
        _snapshot_build_locks[key] = lock
    return lock


def _dind_snapshot_key(
    config: ComposeConfig,
    compose_file: str,
    launch_parameters: LaunchParameters | None,
) -> str:
    """Content hash of everything the compose build depends on.

    Includes the compose directory and every build context (so a changed
    ``requirements.txt`` invalidates the snapshot even if the compose file is
    unchanged) plus launch parameters. Blocking file I/O — call via to_thread.
    """
    compose_dir = Path(compose_file).parent
    context_map, _ = discover_build_contexts(config, compose_dir, BUILD_CONTEXT_DIR)
    h = hash_inputs(
        {
            "kind": "dind-snapshot",
            "compose_dir": _hash_build_context(compose_dir),
            "contexts": {path: _hash_build_context(Path(path)) for path in context_map},
            "launch_parameters": _launch_params_for_hash(launch_parameters),
        }
    )
    return f"inspect-snap-{h}"


async def _start_dind_dockerd(client: AsyncRunloop, devbox_id: str) -> None:
    """Start dockerd (no init system on Runloop devboxes) and wait for it."""
    await vm_exec(
        client, devbox_id, "sudo nohup dockerd > /tmp/dockerd.log 2>&1 &", timeout=10
    )
    await _wait_for_docker_daemon(client, devbox_id)


def _dind_create_kwargs(
    *,
    name: str | None,
    metadata: dict[str, str],
    launch_parameters: LaunchParameters | None,
    environment_variables: dict[str, str] | None,
    timeout: float | None,
) -> dict[str, object]:
    kwargs: dict[str, object] = {
        "metadata": metadata,
        "polling_config": DEVBOX_CREATE_POLLING_CONFIG,
    }
    if name is not None:
        kwargs["name"] = name
    if launch_parameters is not None:
        kwargs["launch_parameters"] = launch_parameters
    if environment_variables:
        kwargs["environment_variables"] = environment_variables
    if timeout is not None:
        kwargs["timeout"] = timeout
    return kwargs


async def _build_dind_snapshot(
    client: AsyncRunloop,
    key: str,
    config: ComposeConfig,
    compose_file: str,
    *,
    name: str | None,
    metadata: dict[str, str],
    launch_parameters: LaunchParameters | None,
    environment_variables: dict[str, str] | None,
    timeout: float | None,
) -> tuple[str, _DinDSnapshotInfo]:
    """Build the compose stack on a fresh devbox and snapshot it for reuse.

    Returns ``(devbox_id, info)``. The devbox is left running (dockerd up, images
    built) so the caller can reuse it for its own sample instead of creating a
    second devbox from the snapshot.
    """
    blueprint_name = await _ensure_dind_blueprint(
        client, launch_parameters=launch_parameters
    )
    create_kwargs = _dind_create_kwargs(
        name=name,
        metadata=metadata,
        launch_parameters=launch_parameters,
        environment_variables=environment_variables,
        timeout=timeout,
    )
    create_kwargs["blueprint_name"] = blueprint_name
    devbox = await create_devbox(client, **create_kwargs)
    logger.debug("Building DinD snapshot on devbox %s", devbox.id)
    try:
        await _start_dind_dockerd(client, devbox.id)
        compose_remote_path = await _upload_build_contexts(
            client, devbox.id, config, compose_file
        )
        project = RunloopDinDProject(
            client=client,
            devbox_id=devbox.id,
            project_name=_PROJECT_NAME,
            compose_path=compose_remote_path,
        )
        exit_code, stdout, stderr = await compose_exec(
            project, ["build"], timeout=BUILD_TIMEOUT
        )
        if exit_code != 0:
            raise RuntimeError(
                f"docker compose build failed:\nstdout: {stdout}\nstderr: {stderr}"
            )
        # Flush the built image layers to disk before snapshotting.
        await vm_exec(client, devbox.id, "sudo sync", timeout=30)
        snapshot = await client.devboxes.snapshot_disk(
            devbox.id, name=key, metadata=metadata
        )
    except BaseException:
        try:
            await shutdown_devbox(client, devbox.id)
        except Exception as cleanup_err:
            logger.warning(
                "Failed to shut down DinD build devbox %s: %s", devbox.id, cleanup_err
            )
        raise

    info = _DinDSnapshotInfo(
        snapshot_id=snapshot.id,
        compose_remote_path=compose_remote_path,
        expected_services=list(config.services.keys()),
    )
    _cache_dind_snapshot(key, info)
    logger.debug("Snapshotted DinD stack %s -> %s", key, snapshot.id)
    return devbox.id, info


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
    """Create a DinD devbox with the compose services running.

    The compose stack is built (and its images snapshotted) only once per unique
    content hash: the first sample builds it on its own devbox and keeps that
    devbox; every later sample creates a devbox from the snapshot and skips the
    upload + ``compose build``, running only ``compose up``.

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
    key = await asyncio.to_thread(
        _dind_snapshot_key, config, compose_file, launch_parameters
    )

    # Resolve (or build once, under a lock) the snapshot for this stack. When we
    # perform the build, we keep the build devbox and reuse it for this sample.
    info = _cached_dind_snapshot(key)
    built_devbox_id: str | None = None
    if info is None:
        async with _snapshot_build_lock(key):
            info = _cached_dind_snapshot(key)
            if info is None:
                built_devbox_id, info = await _build_dind_snapshot(
                    client,
                    key,
                    config,
                    compose_file,
                    name=name,
                    metadata=metadata,
                    launch_parameters=launch_parameters,
                    environment_variables=environment_variables,
                    timeout=timeout,
                )

    if built_devbox_id is not None:
        devbox_id = built_devbox_id  # reuse: dockerd already up, images built
    else:
        create_kwargs = _dind_create_kwargs(
            name=name,
            metadata=metadata,
            launch_parameters=launch_parameters,
            environment_variables=environment_variables,
            timeout=timeout,
        )
        create_kwargs["snapshot_id"] = info.snapshot_id
        devbox = await create_devbox(client, **create_kwargs)
        logger.debug(
            "Created DinD devbox %s from snapshot %s", devbox.id, info.snapshot_id
        )
        await _start_dind_dockerd(client, devbox.id)
        devbox_id = devbox.id

    project = RunloopDinDProject(
        client=client,
        devbox_id=devbox_id,
        project_name=_PROJECT_NAME,
        compose_path=info.compose_remote_path,
        services=info.expected_services,
    )
    try:
        healthcheck_timeout = compute_healthcheck_timeout(config.services)
        logger.debug("Starting compose services in DinD devbox %s...", devbox_id)
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

        await _wait_for_services(
            project, info.expected_services, timeout=healthcheck_timeout
        )
        return project

    except BaseException:
        try:
            await shutdown_devbox(client, devbox_id)
        except Exception as cleanup_err:
            logger.warning(
                "Failed to shut down DinD devbox %s: %s", devbox_id, cleanup_err
            )
        raise


async def destroy_dind_project(project: RunloopDinDProject) -> None:
    """Best-effort ``compose down``; the caller shuts the devbox down after."""
    await compose_down(compose_exec, project)


async def discover_working_dir(project: RunloopDinDProject, service: str) -> str:
    return await _discover_working_dir(compose_exec, project, service)
