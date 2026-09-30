"""Shared Docker-in-Docker project orchestration helpers.

Provider-agnostic building blocks for driving a ``docker compose`` stack inside
a DinD VM. The SDK-specific primitives (how a command execs on the VM, how a
directory is uploaded) stay in each provider and are injected here as callables;
these helpers only compose those primitives.
"""

from __future__ import annotations

import asyncio
import json
import shlex
from collections.abc import Awaitable, Callable
from logging import getLogger
from pathlib import Path
from typing import Any

from inspect_ai.util import ComposeConfig

from inspect_sandboxes._util.dind_compose import (
    discover_build_contexts,
    rewrite_compose_yaml,
)

logger = getLogger(__name__)

# A provider's ``compose_exec(project, subcommand, *, timeout, ...)`` — its
# return tuple starts with (exit_code, output, ...); helpers use [0] and [1].
ComposeExec = Callable[..., Awaitable[Any]]
# A VM-command runner ``(command, timeout) -> (exit_code, output, ...)`` already
# bound to a provider's client/devbox or sandbox.
VMExec = Callable[[str, int | None], Awaitable[Any]]


def build_compose_command(
    project_name: str,
    compose_dir: str,
    compose_path: str,
    subcommand: list[str],
    *,
    env: dict[str, str] | None = None,
    sudo: bool = True,
) -> str:
    """Build a ``docker compose`` command line for a DinD VM.

    ``sudo`` prefixes the command (devboxes that run docker as root need it;
    Daytona's VM runs it directly). ``env`` is inlined as ``KEY=VALUE`` prefixes.
    """
    parts = [
        *(["sudo"] if sudo else []),
        "docker",
        "compose",
        "-p",
        project_name,
        "--project-directory",
        compose_dir,
        "-f",
        compose_path,
        *subcommand,
    ]
    cmd = shlex.join(parts)
    if env:
        prefix = " ".join(f"{k}={shlex.quote(v)}" for k, v in env.items())
        cmd = f"{prefix} {cmd}"
    return cmd


def parse_running_services(output: str) -> set[str]:
    """Parse the ``Service`` names from ``docker compose ps --format json``.

    The command emits one JSON object per line; non-JSON lines (warnings mixed
    into combined stdout/stderr) are skipped.
    """
    running: set[str] = set()
    for line in output.strip().splitlines():
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        running.add(entry.get("Service", ""))
    return running


async def upload_build_contexts(
    config: ComposeConfig,
    compose_file: str,
    compose_dir_remote: str,
    build_context_dir: str,
    *,
    upload_directory: Callable[[Path, str], Awaitable[object]],
    upload_file: Callable[[str, bytes], Awaitable[object]],
) -> str:
    """Upload the compose file + all build contexts to a DinD VM.

    The upload primitives are injected: ``upload_directory(local, remote)`` and
    ``upload_file(remote, data)`` are each provider's SDK-specific transfer.
    Returns the remote path to the (possibly-rewritten) compose file.
    """
    compose_path = Path(compose_file)
    compose_dir = compose_path.parent

    context_map, needs_rewrite = discover_build_contexts(
        config, compose_dir, build_context_dir
    )

    await upload_directory(compose_dir, compose_dir_remote)
    for local_path, remote_path in context_map.items():
        await upload_directory(Path(local_path), remote_path)

    if not needs_rewrite:
        return f"{compose_dir_remote}/{compose_path.name}"

    rewritten = rewrite_compose_yaml(config, compose_dir, context_map)
    rewritten_remote = f"{compose_dir_remote}/compose.yaml"
    await upload_file(rewritten_remote, rewritten.encode("utf-8"))
    logger.debug("Uploaded rewritten compose YAML to %s", rewritten_remote)
    return rewritten_remote


async def wait_for_docker_daemon(
    run: VMExec,
    *,
    info_cmd: str,
    timeout: int,
    poll_interval: int,
    log_tail_cmd: str | None = None,
) -> None:
    """Poll ``docker info`` until the Docker daemon is responsive.

    ``run`` is the provider's VM command runner; ``info_cmd`` is the (possibly
    sudo-prefixed) ``docker info`` command. On timeout, if ``log_tail_cmd`` is
    given, its output is appended to the error to aid diagnosis.
    """
    logger.debug("Waiting for Docker daemon inside DinD VM...")
    last_output = ""
    for _ in range(timeout // poll_interval):
        result = await run(info_cmd, 10)
        if result[0] == 0:
            logger.debug("Docker daemon is ready.")
            return
        last_output = result[1]
        await asyncio.sleep(poll_interval)

    msg = (
        f"Docker daemon not ready after {timeout}s.\n"
        f"Last 'docker info' output: {last_output}"
    )
    if log_tail_cmd is not None:
        tail = await run(log_tail_cmd, 5)
        msg += f"\ndockerd log:\n{tail[1]}"
    raise RuntimeError(msg)


async def wait_for_services(
    compose_exec: ComposeExec,
    project: object,
    expected: list[str],
    *,
    timeout: int,
    poll_interval: int,
) -> None:
    """Poll ``docker compose ps`` until all ``expected`` services are running."""
    logger.debug("Waiting for compose services: %s", expected)
    last_output = ""
    for _ in range(timeout // poll_interval):
        result = await compose_exec(
            project, ["ps", "--format", "json", "--status", "running"], timeout=15
        )
        exit_code, output = result[0], result[1]
        if exit_code == 0 and output.strip():
            running = parse_running_services(output)
            if set(expected) <= running:
                logger.debug("All services running: %s", running)
                return
        last_output = output
        await asyncio.sleep(poll_interval)

    raise RuntimeError(
        f"Not all services running after {timeout}s. "
        f"Expected: {expected}. Last output: {last_output}"
    )


async def discover_working_dir(
    compose_exec: ComposeExec, project: object, service: str
) -> str:
    """Discover a service's working directory via ``pwd`` (``/`` on failure)."""
    result = await compose_exec(project, ["exec", "-T", service, "pwd"], timeout=10)
    exit_code, output = result[0], result[1]
    if exit_code == 0 and output.strip():
        return output.strip()
    logger.warning(
        "Failed to get working directory for service '%s', defaulting to /", service
    )
    return "/"


async def compose_down(compose_exec: ComposeExec, project: object) -> None:
    """Best-effort ``docker compose down`` teardown of the stack."""
    try:
        result = await compose_exec(
            project, ["down", "--remove-orphans", "--timeout", "10"], timeout=30
        )
        if result[0] != 0:
            logger.warning("docker compose down failed: %s", result[1])
    except Exception as e:  # noqa: BLE001 — best-effort teardown
        logger.warning("docker compose down error: %s", e)
