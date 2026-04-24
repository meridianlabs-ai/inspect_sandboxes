"""Tests for DinD project orchestration."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from daytona_sdk import Resources
from inspect_ai.util import ComposeConfig, ComposeService
from inspect_ai.util._sandbox.compose import ComposeHealthcheck
from inspect_sandboxes.daytona._dind_project import (
    DaytonaDinDProject,
    _compute_healthcheck_timeout,
    _dind_snapshot_name,
    _wait_for_docker_daemon,
    _wait_for_services,
    compose_exec,
    create_dind_project,
    destroy_dind_project,
    vm_exec,
)


def make_mock_sandbox(sandbox_id: str = "sb-dind-123") -> MagicMock:
    sandbox = MagicMock()
    sandbox.id = sandbox_id
    response = MagicMock()
    response.exit_code = 0
    response.result = ""
    sandbox.process = MagicMock()
    sandbox.process.exec = AsyncMock(return_value=response)
    sandbox.fs = MagicMock()
    sandbox.fs.upload_files = AsyncMock()
    sandbox.fs.upload_file = AsyncMock()
    return sandbox


def make_mock_project(sandbox: MagicMock | None = None) -> DaytonaDinDProject:
    if sandbox is None:
        sandbox = make_mock_sandbox()
    return DaytonaDinDProject(
        sandbox=sandbox,
        project_name="inspect-test1234",
        compose_path="/inspect/compose/compose.yaml",
        services=["web", "helper"],
    )


@pytest.mark.asyncio
async def test_vm_exec_wraps_with_sh_c() -> None:
    """Test vm_exec wraps command with sh -c for Alpine compatibility."""
    sandbox = make_mock_sandbox()
    await vm_exec(sandbox, "echo hello", timeout=10)

    cmd = sandbox.process.exec.call_args[0][0]
    assert cmd == "sh -c 'echo hello'"


@pytest.mark.asyncio
async def test_vm_exec_returns_exit_code_and_output() -> None:
    sandbox = make_mock_sandbox()
    sandbox.process.exec.return_value.exit_code = 1
    sandbox.process.exec.return_value.result = "error msg"

    code, output = await vm_exec(sandbox, "false")
    assert code == 1
    assert output == "error msg"


@pytest.mark.asyncio
async def test_compose_exec_builds_correct_command() -> None:
    project = make_mock_project()

    with patch(
        "inspect_sandboxes.daytona._dind_project.vm_exec",
        new_callable=AsyncMock,
        return_value=(0, ""),
    ) as mock_vm:
        await compose_exec(project, ["ps"], timeout=10)

    cmd = mock_vm.call_args[0][1]
    assert "docker compose" in cmd
    assert "-p inspect-test1234" in cmd
    assert "-f /inspect/compose/compose.yaml" in cmd
    assert "ps" in cmd


@pytest.mark.asyncio
async def test_compose_exec_inlines_env_vars() -> None:
    project = make_mock_project()

    with patch(
        "inspect_sandboxes.daytona._dind_project.vm_exec",
        new_callable=AsyncMock,
        return_value=(0, ""),
    ) as mock_vm:
        await compose_exec(project, ["up"], env={"FOO": "bar", "BAZ": "qux"})

    cmd = mock_vm.call_args[0][1]
    assert cmd.startswith("FOO=bar BAZ=qux docker compose")


@pytest.mark.asyncio
async def test_wait_for_docker_daemon_succeeds() -> None:
    sandbox = make_mock_sandbox()

    with patch(
        "inspect_sandboxes.daytona._dind_project.vm_exec",
        new_callable=AsyncMock,
        return_value=(0, "ok"),
    ):
        await _wait_for_docker_daemon(sandbox)


@pytest.mark.asyncio
async def test_wait_for_docker_daemon_times_out() -> None:
    sandbox = make_mock_sandbox()

    with (
        patch(
            "inspect_sandboxes.daytona._dind_project.vm_exec",
            new_callable=AsyncMock,
            return_value=(1, "not ready"),
        ),
        patch("inspect_sandboxes.daytona._dind_project._DAEMON_TIMEOUT", 2),
        patch("inspect_sandboxes.daytona._dind_project._DAEMON_POLL_INTERVAL", 1),
        patch(
            "inspect_sandboxes.daytona._dind_project.asyncio.sleep",
            new_callable=AsyncMock,
        ),
    ):
        with pytest.raises(RuntimeError, match="Docker daemon not ready"):
            await _wait_for_docker_daemon(sandbox)


@pytest.mark.asyncio
async def test_wait_for_services_succeeds() -> None:
    project = make_mock_project()
    ps_output = '{"Service":"web"}\n{"Service":"helper"}\n'

    with patch(
        "inspect_sandboxes.daytona._dind_project.compose_exec",
        new_callable=AsyncMock,
        return_value=(0, ps_output),
    ):
        await _wait_for_services(project, ["web", "helper"], timeout=10)


@pytest.mark.asyncio
async def test_wait_for_services_times_out() -> None:
    project = make_mock_project()

    with (
        patch(
            "inspect_sandboxes.daytona._dind_project.compose_exec",
            new_callable=AsyncMock,
            return_value=(0, '{"Service":"web"}\n'),
        ),
        patch("inspect_sandboxes.daytona._dind_project._SERVICE_POLL_INTERVAL", 1),
        patch(
            "inspect_sandboxes.daytona._dind_project.asyncio.sleep",
            new_callable=AsyncMock,
        ),
    ):
        with pytest.raises(RuntimeError, match="Not all services running"):
            await _wait_for_services(project, ["web", "helper"], timeout=2)


def test_compute_healthcheck_timeout() -> None:
    services = {
        "web": ComposeService(
            image="python:3.12",
            healthcheck=ComposeHealthcheck(retries=3, interval="10s", timeout="5s"),
        ),
        "db": ComposeService(
            image="postgres:16",
            healthcheck=ComposeHealthcheck(retries=5, interval="5s", timeout="5s"),
        ),
    }
    # web: 3 * (10 + 5) = 45, db: 5 * (5 + 5) = 50 -> max = 50
    assert _compute_healthcheck_timeout(services) == 50


def test_compute_healthcheck_timeout_no_healthchecks() -> None:
    services = {"web": ComposeService(image="python:3.12")}
    assert _compute_healthcheck_timeout(services) == 120


def test_dind_snapshot_name() -> None:
    assert (
        _dind_snapshot_name(Resources(cpu=3, memory=7)) == "inspect-dind-3cpu-7gb-0gpu"
    )
    assert (
        _dind_snapshot_name(Resources(cpu=2, memory=4, gpu=1))
        == "inspect-dind-2cpu-4gb-1gpu"
    )
    assert _dind_snapshot_name(None) == "inspect-dind-defaultcpu-defaultgb-0gpu"


@pytest.mark.asyncio
async def test_create_dind_project_full_sequence(tmp_path: Path) -> None:
    """Test create_dind_project executes the full startup sequence."""
    compose_file = tmp_path / "compose.yaml"
    compose_file.write_text("services:\n  web:\n    image: python:3.12\n")

    config = ComposeConfig(services={"web": ComposeService(image="python:3.12")})
    sandbox = make_mock_sandbox()

    ps_output = '{"Service":"web"}\n'

    with (
        patch(
            "inspect_sandboxes.daytona._dind_project.create_sandbox",
            new_callable=AsyncMock,
            return_value=sandbox,
        ),
        patch(
            "inspect_sandboxes.daytona._dind_project._ensure_dind_snapshot",
            new_callable=AsyncMock,
            return_value="",
        ),
        patch(
            "inspect_sandboxes.daytona._dind_project.vm_exec",
            new_callable=AsyncMock,
            return_value=(0, ""),
        ),
        patch(
            "inspect_sandboxes.daytona._dind_project._wait_for_docker_daemon",
            new_callable=AsyncMock,
        ),
        patch(
            "inspect_sandboxes.daytona._dind_project._upload_build_contexts",
            new_callable=AsyncMock,
            return_value=f"/inspect/compose/{compose_file.name}",
        ),
        patch(
            "inspect_sandboxes.daytona._dind_project.compose_exec",
            new_callable=AsyncMock,
            return_value=(0, ps_output),
        ),
        patch(
            "inspect_sandboxes.daytona._dind_project._wait_for_services",
            new_callable=AsyncMock,
        ),
    ):
        project = await create_dind_project(
            MagicMock(), config, str(compose_file), labels={"test": "true"}
        )

    assert project.sandbox is sandbox
    assert project.services == ["web"]


@pytest.mark.asyncio
async def test_create_dind_project_passes_name_to_params(tmp_path: Path) -> None:
    """The ``name`` kwarg must reach the Daytona sandbox params model."""
    compose_file = tmp_path / "compose.yaml"
    compose_file.write_text("services:\n  web:\n    image: python:3.12\n")
    config = ComposeConfig(services={"web": ComposeService(image="python:3.12")})
    sandbox = make_mock_sandbox()

    with (
        patch(
            "inspect_sandboxes.daytona._dind_project.create_sandbox",
            new_callable=AsyncMock,
            return_value=sandbox,
        ) as mock_create,
        patch(
            "inspect_sandboxes.daytona._dind_project._ensure_dind_snapshot",
            new_callable=AsyncMock,
            return_value="dind-snap",
        ),
        patch(
            "inspect_sandboxes.daytona._dind_project.vm_exec",
            new_callable=AsyncMock,
            return_value=(0, ""),
        ),
        patch(
            "inspect_sandboxes.daytona._dind_project._wait_for_docker_daemon",
            new_callable=AsyncMock,
        ),
        patch(
            "inspect_sandboxes.daytona._dind_project._upload_build_contexts",
            new_callable=AsyncMock,
            return_value=f"/inspect/compose/{compose_file.name}",
        ),
        patch(
            "inspect_sandboxes.daytona._dind_project.compose_exec",
            new_callable=AsyncMock,
            return_value=(0, '{"Service":"web"}\n'),
        ),
        patch(
            "inspect_sandboxes.daytona._dind_project._wait_for_services",
            new_callable=AsyncMock,
        ),
    ):
        await create_dind_project(
            MagicMock(),
            config,
            str(compose_file),
            labels={},
            name="inspect-task-5-abcdef12",
        )

    # Inspect the params object passed to create_sandbox.
    (_, params), _ = mock_create.call_args
    assert params.name == "inspect-task-5-abcdef12"


@pytest.mark.asyncio
async def test_create_dind_project_cleans_up_on_failure(tmp_path: Path) -> None:
    """Test create_dind_project deletes sandbox when startup fails."""
    compose_file = tmp_path / "compose.yaml"
    compose_file.write_text("services:\n  web:\n    image: python:3.12\n")

    config = ComposeConfig(services={"web": ComposeService(image="python:3.12")})
    sandbox = make_mock_sandbox()

    with (
        patch(
            "inspect_sandboxes.daytona._dind_project.create_sandbox",
            new_callable=AsyncMock,
            return_value=sandbox,
        ),
        patch(
            "inspect_sandboxes.daytona._dind_project._ensure_dind_snapshot",
            new_callable=AsyncMock,
            return_value="",
        ),
        patch(
            "inspect_sandboxes.daytona._dind_project.vm_exec",
            new_callable=AsyncMock,
            side_effect=RuntimeError("daemon failed"),
        ),
        patch(
            "inspect_sandboxes.daytona._dind_project.delete_sandbox",
            new_callable=AsyncMock,
        ) as mock_delete,
    ):
        with pytest.raises(RuntimeError, match="daemon failed"):
            await create_dind_project(MagicMock(), config, str(compose_file), labels={})

    mock_delete.assert_called_once()


@pytest.mark.asyncio
async def test_destroy_dind_project_best_effort() -> None:
    """Test destroy_dind_project runs compose down and doesn't raise on failure."""
    project = make_mock_project()

    with patch(
        "inspect_sandboxes.daytona._dind_project.compose_exec",
        new_callable=AsyncMock,
        return_value=(1, "error"),
    ) as mock_exec:
        await destroy_dind_project(project)

    cmd = mock_exec.call_args[0][1]
    assert "down" in cmd
