"""DinD project orchestration for the E2B provider.

Multi-service compose runs inside a Devbox that has Docker installed:

    1. Build (or reuse cached) "DinD-capable" E2B template — Ubuntu 24.04 with
       Docker engine and the compose plugin.
    2. Create a Devbox from that template.
    3. Upload the compose file and any local build contexts.
    4. ``docker compose build`` + ``docker compose up --wait`` inside.
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

from e2b import (
    AsyncSandbox,
    AsyncTemplate,
    CommandExitException,
    NotFoundException,
    TemplateClass,
)
from inspect_ai.util import ComposeConfig

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

from ._single_env import FILE_REQUEST_TIMEOUT
from ._template import TEMPLATE_NAME_PREFIX

logger = getLogger(__name__)

COMPOSE_DIR = "/inspect/compose"
BUILD_CONTEXT_DIR = "/inspect/contexts"
BUILD_TIMEOUT = 600

_DAEMON_POLL_INTERVAL = 2
_DAEMON_TIMEOUT = 60
_SERVICE_POLL_INTERVAL = 2
_SERVICE_TIMEOUT = 120

DEFAULT_DIND_CPU = 2
DEFAULT_DIND_MEMORY_MB = 4096  # docker daemon + at least one service comfortably

DOCKER_CE_VERSION = "5:29.5.2-1~ubuntu.24.04~noble"


@dataclass
class E2BDinDProject:
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

    E2B's commands.run raises CommandExitException on non-zero exit; we catch
    it and surface the result as a tuple. Returns (exit_code, stdout, stderr).
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
    project: E2BDinDProject,
    subcommand: list[str],
    *,
    env: dict[str, str] | None = None,
    timeout: int | None = 60,
) -> tuple[int, str, str]:
    """Run a ``docker compose`` subcommand on the DinD Devbox.

    Returns (exit_code, stdout, stderr).
    """
    cmd = build_compose_command(
        project.project_name, COMPOSE_DIR, project.compose_path, subcommand, env=env
    )
    return await vm_exec(project.sandbox, cmd, timeout=timeout)


async def _wait_for_docker_daemon(sandbox: AsyncSandbox) -> None:
    await wait_for_docker_daemon(
        lambda cmd, t: vm_exec(sandbox, cmd, timeout=t),
        info_cmd="sudo docker info",
        timeout=_DAEMON_TIMEOUT,
        poll_interval=_DAEMON_POLL_INTERVAL,
    )


async def _wait_for_services(
    project: E2BDinDProject,
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


async def _upload_directory(
    sandbox: AsyncSandbox,
    local_dir: str | Path,
    remote_dir: str,
) -> None:
    """Upload a local directory to the sandbox recursively."""
    local_dir = Path(local_dir)
    # files.write_files's WriteEntry is a TypedDict; the SDK accepts plain dicts.
    from typing import Any

    entries: list[Any] = []

    for root, _, files in os.walk(local_dir):
        for filename in files:
            local_path = Path(root) / filename
            if not local_path.is_file():
                continue
            if not os.access(local_path, os.R_OK):
                continue
            rel_path = local_path.relative_to(local_dir)
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
    """Upload compose file and all build contexts to the Devbox.

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
        upload_file=lambda remote, data: sandbox.files.write(
            remote, data, request_timeout=FILE_REQUEST_TIMEOUT
        ),
    )


def _dind_template_name(*, cpu_count: int, memory_mb: int) -> str:
    """Derive a deterministic template name from DinD resources."""
    return f"{TEMPLATE_NAME_PREFIX}dind-{cpu_count}cpu-{memory_mb}mb"


def build_dind_template_spec() -> TemplateClass:
    """Build the AsyncTemplate definition for a Docker-capable Devbox.

    Installs a pinned ``docker-ce`` from Docker's official apt repo so cold
    rebuilds produce byte-identical templates. Bump ``DOCKER_CE_VERSION`` to
    pick up a new Docker release — the apt-install ``RUN`` text changes, so
    E2B's per-instruction layer cache rebuilds that layer on next use.
    """
    # E2B's run_cmd runs as an unprivileged user; sudo each step.
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

    Relies on E2B's per-instruction layer cache for unchanged builds (see
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
) -> E2BDinDProject:
    """Build the DinD template, create the Devbox, and bring up compose services.

    Args:
        config: Parsed compose configuration.
        compose_file: Local path to the compose file.
        metadata: Metadata to apply to the Devbox.
        cpu_count: CPUs for the DinD template build.
        memory_mb: Memory (MiB) for the DinD template build.
        sandbox_timeout: Devbox lifetime in seconds (from ``x-e2b.timeout``
            or default). E2B caps at 3600s (Hobby) / 86400s (Pro).
        sandbox_envs: Environment variables set on the Devbox VM (not on
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
        project = E2BDinDProject(
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
        except (NotFoundException, Exception) as e:
            logger.warning(
                "Failed to clean up DinD sandbox %s: %s", sandbox.sandbox_id, e
            )
        raise


async def destroy_dind_project(project: E2BDinDProject) -> None:
    """Best-effort ``compose down``; the caller kills the E2B Devbox after."""
    await compose_down(compose_exec, project)


async def discover_working_dir(project: E2BDinDProject, service: str) -> str:
    return await _discover_working_dir(compose_exec, project, service)
