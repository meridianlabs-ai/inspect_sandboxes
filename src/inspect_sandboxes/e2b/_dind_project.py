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

import asyncio
import json
import os
import shlex
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
    discover_build_contexts,
    rewrite_compose_yaml,
)

from ._command import (
    exceeds_inline_limit,
    remove_files_command,
    source_script_command,
    temp_file_path,
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

    commands.run passes the whole command line as a single ``bash -l -c``
    argument, which the kernel caps at 128 KiB. A ``docker compose exec`` line
    carrying a large container argv is staged in a temp file and sourced so
    the args reach docker intact (see ``_command``).
    """
    script_file: str | None = None
    if exceeds_inline_limit(command):
        script_file = temp_file_path("cmd")
        await sandbox.files.write(
            script_file, command.encode("utf-8"), request_timeout=FILE_REQUEST_TIMEOUT
        )
        command = source_script_command(script_file)

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
    finally:
        # Same default user wrote and sources the file, so it can remove it.
        if script_file is not None:
            try:
                await sandbox.commands.run(
                    remove_files_command([script_file]), timeout=10
                )
            except Exception as e:
                logger.debug(f"Could not remove staged command {script_file}: {e}")
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
    return await vm_exec(project.sandbox, cmd, timeout=timeout)


async def _wait_for_docker_daemon(sandbox: AsyncSandbox) -> None:
    """Poll ``docker info`` until the Docker daemon is responsive."""
    logger.debug("Waiting for Docker daemon inside DinD sandbox...")
    last_output = ""
    for _ in range(_DAEMON_TIMEOUT // _DAEMON_POLL_INTERVAL):
        exit_code, stdout, _ = await vm_exec(sandbox, "sudo docker info", timeout=10)
        if exit_code == 0:
            logger.debug("Docker daemon is ready.")
            return
        last_output = stdout
        await asyncio.sleep(_DAEMON_POLL_INTERVAL)
    raise RuntimeError(
        f"Docker daemon not ready after {_DAEMON_TIMEOUT}s. "
        f"Last 'docker info': {last_output}"
    )


async def _wait_for_services(
    project: E2BDinDProject,
    expected: list[str],
    timeout: int = _SERVICE_TIMEOUT,
) -> None:
    """Poll ``docker compose ps`` until all expected services are running."""
    logger.debug("Waiting for compose services: %s", expected)
    last_output = ""
    for _ in range(timeout // _SERVICE_POLL_INTERVAL):
        exit_code, output, _ = await compose_exec(
            project,
            ["ps", "--format", "json", "--status", "running"],
            timeout=15,
        )
        if exit_code == 0 and output.strip():
            running: set[str] = set()
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
    compose_path = Path(compose_file)
    compose_dir = compose_path.parent

    context_map, needs_rewrite = discover_build_contexts(
        config, compose_dir, BUILD_CONTEXT_DIR
    )

    await _upload_directory(sandbox, compose_dir, COMPOSE_DIR)
    for local_path, remote_path in context_map.items():
        await _upload_directory(sandbox, local_path, remote_path)

    if not needs_rewrite:
        return f"{COMPOSE_DIR}/{compose_path.name}"

    rewritten = rewrite_compose_yaml(config, compose_dir, context_map)
    rewritten_remote = f"{COMPOSE_DIR}/compose.yaml"
    await sandbox.files.write(
        rewritten_remote,
        rewritten.encode("utf-8"),
        request_timeout=FILE_REQUEST_TIMEOUT,
    )
    logger.debug("Uploaded rewritten compose YAML to %s", rewritten_remote)
    return rewritten_remote


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
    """Tear down compose services inside the DinD Devbox.

    Runs ``docker compose down`` best-effort. The caller is responsible
    for killing the E2B Devbox afterwards.
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


async def discover_working_dir(project: E2BDinDProject, service: str) -> str:
    """Discover a service's working directory via ``pwd``.

    Returns ``/`` if the query fails (e.g. service has no shell).
    """
    exit_code, stdout, _ = await compose_exec(
        project,
        ["exec", "-T", service, "pwd"],
        timeout=10,
    )
    if exit_code == 0 and stdout.strip():
        return stdout.strip()
    logger.warning(
        "Failed to get working directory for service '%s', defaulting to /", service
    )
    return "/"
