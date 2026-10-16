"""DinD project orchestration for the Novita provider.

Multi-service compose runs inside a sandbox that has Docker installed:

    1. Build (or reuse cached) "DinD-capable" Novita template — Ubuntu 24.04
       with Docker engine and the compose plugin.
    2. Create a sandbox from that template.
    3. Upload the compose file and any local build contexts.
    4. ``docker compose build`` + ``pull`` + ``up --wait`` inside.
    5. Per-service environments (in ``_dind_env.py``) route exec/file ops via
       ``docker compose exec/cp``.

The DinD template installs a pinned ``docker-ce`` from Docker's official apt
repo; the post-install configures dockerd to start at boot, so we only need
to wait for the daemon to come up — no explicit start step.
"""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass, field
from logging import getLogger
from pathlib import Path
from typing import Any

import httpx
import pathspec
from inspect_ai.util import ComposeConfig
from novita_sandbox.core import (
    AsyncSandbox,
    AsyncTemplate,
    CommandExitException,
    TemplateClass,
    TimeoutException,
)

from inspect_sandboxes._util.dind_compose import (
    compute_healthcheck_timeout,
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

from ._retry import standard_retry
from ._sandbox_utils import FILE_REQUEST_TIMEOUT, write_sandbox_file
from ._template import TEMPLATE_NAME_PREFIX

logger = getLogger(__name__)

COMPOSE_DIR = "/inspect/compose"
BUILD_CONTEXT_DIR = "/inspect/contexts"
BUILD_TIMEOUT = 600
PULL_TIMEOUT = 600

_DAEMON_POLL_INTERVAL = 2
_DAEMON_TIMEOUT = 60
_SERVICE_POLL_INTERVAL = 2
_SERVICE_TIMEOUT = 120

DEFAULT_DIND_CPU = 2
DEFAULT_DIND_MEMORY_MB = 4096  # docker daemon + at least one service comfortably

DOCKER_CE_VERSION = "5:29.5.2-1~ubuntu.24.04~noble"

# Build-context churn always dropped from DinD uploads, on top of any
# .dockerignore: never build inputs, and `logs` would otherwise bust the
# template cache every run.
_IGNORED_DIR_NAMES = {".git", "__pycache__", "logs", ".venv"}
_IGNORED_FILE_NAMES = {".DS_Store"}


@dataclass
class NovitaDinDProject:
    """Shared state for all per-service environments in one DinD sample."""

    sandbox: AsyncSandbox
    project_name: str
    compose_path: str
    services: list[str] = field(default_factory=list)


async def vm_exec(
    sandbox: AsyncSandbox,
    command: str,
    timeout: int | None = 60,
) -> tuple[int, str, str]:
    """Execute a command on the DinD sandbox VM (not inside a compose service).

    Novita's commands.run raises CommandExitException on non-zero exit; we
    catch it and surface the result as a tuple. Returns (exit_code, stdout,
    stderr).
    """
    try:
        result = await sandbox.commands.run(
            command,
            timeout=timeout if timeout is not None else 0,
        )
    except CommandExitException as e:
        return (
            e.exit_code if e.exit_code is not None else 1,
            e.stdout,
            e.stderr,
        )
    return (
        result.exit_code if result.exit_code is not None else 0,
        result.stdout,
        result.stderr,
    )


async def compose_exec(
    project: NovitaDinDProject,
    subcommand: list[str],
    *,
    env: dict[str, str] | None = None,
    timeout: int | None = 60,
) -> tuple[int, str, str]:
    """Run a ``docker compose`` subcommand on the DinD sandbox.

    Returns (exit_code, stdout, stderr).
    """
    cmd = build_compose_command(
        project.project_name, COMPOSE_DIR, project.compose_path, subcommand, env=env
    )
    return await vm_exec(project.sandbox, cmd, timeout=timeout)


async def _wait_for_docker_daemon(sandbox: AsyncSandbox) -> None:
    # Novita's commands.run raises on an over-time poll; `docker info` blocks
    # while the socket-activated daemon is still starting, so a timed-out poll
    # means "not ready yet", not failure. Translate it to a non-zero result so
    # the shared loop keeps polling instead of propagating the timeout.
    async def _run(cmd: str, t: int | None) -> tuple[int, str, str]:
        try:
            return await vm_exec(sandbox, cmd, timeout=t)
        except (TimeoutException, httpx.TimeoutException):
            return (1, "docker info timed out", "")

    await wait_for_docker_daemon(
        _run,
        info_cmd="sudo docker info",
        timeout=_DAEMON_TIMEOUT,
        poll_interval=_DAEMON_POLL_INTERVAL,
    )


async def _wait_for_services(
    project: NovitaDinDProject,
    expected: list[str],
    timeout: int = _SERVICE_TIMEOUT,
) -> None:
    # Same timeout tolerance as the daemon wait: an over-time `compose ps` poll
    # is "not ready yet", not failure.
    async def _compose_exec(
        proj: NovitaDinDProject,
        subcommand: list[str],
        *,
        timeout: int | None = 60,
    ) -> tuple[int, str, str]:
        try:
            return await compose_exec(proj, subcommand, timeout=timeout)
        except (TimeoutException, httpx.TimeoutException):
            return (1, "docker compose ps timed out", "")

    await wait_for_services(
        _compose_exec,
        project,
        expected,
        timeout=timeout,
        poll_interval=_SERVICE_POLL_INTERVAL,
    )


@standard_retry
def _is_ignored_context_path(rel_path: Path) -> bool:
    for part in rel_path.parts:
        if part in _IGNORED_DIR_NAMES:
            return True
    name = rel_path.name
    return name in _IGNORED_FILE_NAMES or name.endswith(".pyc")


def _load_dockerignore(context_dir: Path) -> pathspec.PathSpec[Any] | None:
    """Parse ``.dockerignore`` at the context root, or None if absent.

    Uses pathspec's gitignore engine — the closest widely-available match for
    Docker's ``.dockerignore`` glob semantics (``**``, ``!`` negation, trailing
    ``/``). The hardcoded churn excludes in ``_is_ignored_context_path`` still
    apply on top: unlike Docker we always drop ``.git``/caches/``logs``/``.venv``
    (never build inputs, and ``logs`` would otherwise bust the cache every run).
    """
    dockerignore = context_dir / ".dockerignore"
    if not dockerignore.is_file():
        return None
    lines = dockerignore.read_text(encoding="utf-8", errors="replace").splitlines()
    return pathspec.GitIgnoreSpec.from_lines(lines)


async def _upload_directory(
    sandbox: AsyncSandbox,
    local_dir: str | Path,
    remote_dir: str,
) -> None:
    """Upload a local directory to the sandbox recursively.

    Honors ``.dockerignore`` at ``local_dir`` plus the hardcoded churn set so
    the DinD upload matches what a single-env build would ship (the Novita SDK
    filters the single-env path itself; DinD does its own walk).
    """
    local_dir = Path(local_dir)
    ignore_spec = _load_dockerignore(local_dir)
    # files.write_files's WriteEntry is a TypedDict; the SDK accepts plain dicts.
    entries: list[Any] = []

    for root, _, files in os.walk(local_dir):
        for filename in files:
            local_path = Path(root) / filename
            if not local_path.is_file():
                continue
            if not os.access(local_path, os.R_OK):
                continue
            rel_path = local_path.relative_to(local_dir)
            if _is_ignored_context_path(rel_path):
                continue
            if ignore_spec is not None and ignore_spec.match_file(rel_path.as_posix()):
                continue
            remote_path = f"{remote_dir}/{rel_path.as_posix()}"
            entries.append({"path": remote_path, "data": local_path.read_bytes()})

    if not entries:
        return

    # write_files doesn't auto-create parents — ensure the destination tree exists.
    distinct_dirs = sorted({str(Path(e["path"]).parent) for e in entries})
    for d in distinct_dirs:
        await sandbox.files.make_dir(d)
    await sandbox.files.write_files(entries, request_timeout=FILE_REQUEST_TIMEOUT)
    logger.debug("Uploaded %d files from %s to %s", len(entries), local_dir, remote_dir)


async def _upload_build_contexts(
    sandbox: AsyncSandbox,
    config: ComposeConfig,
    compose_file: str,
) -> str:
    """Upload compose file and all build contexts to the sandbox.

    Returns the remote path to the (possibly-rewritten) compose file.
    """
    return await upload_build_contexts(
        config,
        compose_file,
        COMPOSE_DIR,
        BUILD_CONTEXT_DIR,
        upload_directory=lambda local, remote: _upload_directory(
            sandbox, local, remote
        ),
        upload_file=lambda remote, data: write_sandbox_file(sandbox, remote, data),
    )


def _dind_template_name(*, cpu_count: int, memory_mb: int) -> str:
    """Derive a deterministic template name from DinD resources."""
    return f"{TEMPLATE_NAME_PREFIX}dind-{cpu_count}cpu-{memory_mb}mb"


def build_dind_template_spec() -> TemplateClass:
    """Build the AsyncTemplate definition for a Docker-capable sandbox.

    Installs a pinned ``docker-ce`` from Docker's official apt repo so cold
    rebuilds produce byte-identical templates. Bump ``DOCKER_CE_VERSION`` to
    pick up a new Docker release — the apt-install ``RUN`` text changes, so
    Novita's per-instruction layer cache rebuilds that layer on next use.
    """
    # Novita's run_cmd runs as an unprivileged user; sudo each step.
    # `> file` redirects happen pre-sudo, so use `| sudo tee file` instead.
    docker_install = " && ".join(
        [
            "sudo install -m 0755 -d /etc/apt/keyrings",
            "curl -fsSL https://download.docker.com/linux/ubuntu/gpg "
            "| sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg",
            "sudo chmod a+r /etc/apt/keyrings/docker.gpg",
            'echo "deb [arch=$(dpkg --print-architecture) '
            "signed-by=/etc/apt/keyrings/docker.gpg] "
            'https://download.docker.com/linux/ubuntu noble stable" '
            "| sudo tee /etc/apt/sources.list.d/docker.list > /dev/null",
            "sudo apt-get update",
            "sudo apt-get install -y --no-install-recommends "
            f"docker-ce={DOCKER_CE_VERSION} "
            f"docker-ce-cli={DOCKER_CE_VERSION} "
            "containerd.io docker-buildx-plugin docker-compose-plugin",
            "sudo rm -rf /var/lib/apt/lists/*",
        ]
    )
    return (
        AsyncTemplate()
        .from_ubuntu_image("24.04")
        .apt_install(["ca-certificates", "curl", "gnupg", "sudo"])
        .run_cmd(docker_install)
    )


async def _ensure_dind_template(
    *, cpu_count: int = DEFAULT_DIND_CPU, memory_mb: int = DEFAULT_DIND_MEMORY_MB
) -> str:
    """Build (or reuse cached) DinD template; returns the template name.

    Relies on Novita's per-instruction layer cache for unchanged builds (see
    ``_template.py`` module docstring).
    """
    name = _dind_template_name(cpu_count=cpu_count, memory_mb=memory_mb)
    await AsyncTemplate.build(
        build_dind_template_spec(),
        name=name,
        cpu_count=cpu_count,
        memory_mb=memory_mb,
    )
    return name


async def create_dind_project(
    config: ComposeConfig,
    compose_file: str,
    *,
    metadata: dict[str, str],
    cpu_count: int = DEFAULT_DIND_CPU,
    memory_mb: int = DEFAULT_DIND_MEMORY_MB,
    sandbox_timeout: int | float = 3600,
    sandbox_envs: dict[str, str] | None = None,
) -> NovitaDinDProject:
    """Build the DinD template, create the sandbox, and bring up compose services.

    Args:
        config: Parsed compose configuration.
        compose_file: Local path to the compose file.
        metadata: Metadata to apply to the sandbox.
        cpu_count: CPUs for the DinD template build.
        memory_mb: Memory (MiB) for the DinD template build.
        sandbox_timeout: Sandbox lifetime in seconds (from ``x-novita.timeout``
            or default). Novita caps at 3600s (Free) / 86400s (Paid).
        sandbox_envs: Environment variables set on the sandbox VM (not on
            individual compose services).
    """
    project_name = f"inspect-{uuid.uuid4().hex[:8]}"

    template = await _ensure_dind_template(cpu_count=cpu_count, memory_mb=memory_mb)
    sandbox = await AsyncSandbox.create(
        template=template,
        timeout=int(sandbox_timeout),
        metadata=metadata,
        envs=sandbox_envs,
    )
    logger.debug("Created DinD sandbox %s", sandbox.sandbox_id)

    try:
        await _wait_for_docker_daemon(sandbox)

        compose_remote_path = await _upload_build_contexts(
            sandbox, config, compose_file
        )
        project = NovitaDinDProject(
            sandbox=sandbox,
            project_name=project_name,
            compose_path=compose_remote_path,
        )

        logger.debug(
            "Building compose services in DinD sandbox %s...", sandbox.sandbox_id
        )
        exit_code, stdout, stderr = await compose_exec(
            project, ["build"], timeout=BUILD_TIMEOUT
        )
        if exit_code != 0:
            raise RuntimeError(
                f"docker compose build failed:\nstdout:\n{stdout}\nstderr:\n{stderr}"
            )

        # Pull image-based services up front: `up` would otherwise spend its
        # healthcheck-sized timeout on the downloads.
        logger.debug("Pulling compose images in DinD sandbox %s...", sandbox.sandbox_id)
        exit_code, stdout, stderr = await compose_exec(
            project, ["pull", "--ignore-buildable"], timeout=PULL_TIMEOUT
        )
        if exit_code != 0:
            raise RuntimeError(
                f"docker compose pull failed:\nstdout:\n{stdout}\nstderr:\n{stderr}"
            )

        healthcheck_timeout = compute_healthcheck_timeout(
            config.services, default=_SERVICE_TIMEOUT
        )
        logger.debug(
            "Starting compose services in DinD sandbox %s...", sandbox.sandbox_id
        )
        exit_code, stdout, stderr = await compose_exec(
            project,
            ["up", "--detach", "--wait", "--wait-timeout", str(healthcheck_timeout)],
            timeout=healthcheck_timeout + 30,
        )
        if exit_code != 0:
            raise RuntimeError(
                f"docker compose up failed:\nstdout:\n{stdout}\nstderr:\n{stderr}"
            )

        expected = list(config.services.keys())
        await _wait_for_services(project, expected, timeout=healthcheck_timeout)
        project.services = expected
        return project

    except BaseException:
        try:
            await sandbox.kill()
        except Exception as e:
            logger.warning(
                "Failed to clean up DinD sandbox %s: %s", sandbox.sandbox_id, e
            )
        raise


async def destroy_dind_project(project: NovitaDinDProject) -> None:
    """Best-effort ``compose down``; the caller kills the Novita sandbox after."""
    await compose_down(compose_exec, project)


async def discover_working_dir(project: NovitaDinDProject, service: str) -> str:
    return await _discover_working_dir(compose_exec, project, service)
