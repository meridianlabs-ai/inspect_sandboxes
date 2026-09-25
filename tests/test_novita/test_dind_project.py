"""Tests for DinD project orchestration."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from inspect_ai.util import ComposeConfig, ComposeService
from inspect_sandboxes.novita._dind_project import (
    NovitaDinDProject,
    _dind_template_name,
    _ensure_dind_template,
    _upload_build_contexts,
    _wait_for_docker_daemon,
    _wait_for_services,
    compose_exec,
    create_dind_project,
    destroy_dind_project,
    vm_exec,
)
from novita_sandbox.core import CommandExitException, TimeoutException


def _make_command_result(
    *, stdout: str = "", stderr: str = "", exit_code: int = 0
) -> MagicMock:
    cr = MagicMock()
    cr.stdout = stdout
    cr.stderr = stderr
    cr.exit_code = exit_code
    return cr


def make_mock_sandbox(sandbox_id: str = "sb-dind-123") -> MagicMock:
    """Create a mock AsyncSandbox suitable for the DinD VM."""
    sandbox = MagicMock()
    sandbox.sandbox_id = sandbox_id
    sandbox.commands = MagicMock()
    sandbox.commands.run = AsyncMock(
        return_value=_make_command_result(stdout="", exit_code=0)
    )
    sandbox.files = MagicMock()
    sandbox.files.write = AsyncMock()
    sandbox.files.read = AsyncMock(return_value=bytearray(b"file-contents"))
    sandbox.kill = AsyncMock()
    return sandbox


def make_mock_project(sandbox: MagicMock | None = None) -> NovitaDinDProject:
    if sandbox is None:
        sandbox = make_mock_sandbox()
    return NovitaDinDProject(
        sandbox=sandbox,
        project_name="inspect-test1234",
        compose_path="/inspect/compose/compose.yaml",
        services=["web", "helper"],
    )


@pytest.mark.asyncio
async def test_vm_exec_translates_command_exit_exception() -> None:
    """Novita's commands.run RAISES on non-zero exit; vm_exec must surface as a tuple."""
    sandbox = make_mock_sandbox()
    sandbox.commands.run = AsyncMock(
        side_effect=CommandExitException(
            stdout="bad output", stderr="oops", exit_code=3, error=None
        )
    )
    exit_code, stdout, stderr = await vm_exec(sandbox, "false")
    assert exit_code == 3
    assert stdout == "bad output"
    assert stderr == "oops"


@pytest.mark.asyncio
async def test_vm_exec_returns_exit_code_and_output() -> None:
    sandbox = make_mock_sandbox()
    sandbox.commands.run = AsyncMock(
        return_value=_make_command_result(stdout="hello", stderr="warn", exit_code=0)
    )
    exit_code, stdout, stderr = await vm_exec(sandbox, "echo hello")
    assert exit_code == 0
    assert stdout == "hello"
    assert stderr == "warn"


@pytest.mark.asyncio
async def test_compose_exec_builds_correct_command() -> None:
    project = make_mock_project()

    with patch(
        "inspect_sandboxes.novita._dind_project.vm_exec",
        new_callable=AsyncMock,
        return_value=(0, "", ""),
    ) as mock_vm:
        await compose_exec(project, ["ps", "--status", "running"], timeout=15)

    cmd = mock_vm.call_args[0][1]
    assert "sudo" in cmd
    assert "docker compose" in cmd
    assert "-p inspect-test1234" in cmd
    assert "--project-directory /inspect/compose" in cmd
    assert "/inspect/compose/compose.yaml" in cmd
    assert "ps" in cmd
    assert "--status running" in cmd


@pytest.mark.asyncio
async def test_compose_exec_inlines_env_vars() -> None:
    project = make_mock_project()

    with patch(
        "inspect_sandboxes.novita._dind_project.vm_exec",
        new_callable=AsyncMock,
        return_value=(0, "", ""),
    ) as mock_vm:
        await compose_exec(project, ["up"], env={"FOO": "bar", "BAZ": "qux"})

    cmd = mock_vm.call_args[0][1]
    assert cmd.startswith("FOO=bar BAZ=qux sudo docker compose")


@pytest.mark.asyncio
async def test_wait_for_docker_daemon_succeeds() -> None:
    sandbox = make_mock_sandbox()

    with patch(
        "inspect_sandboxes.novita._dind_project.vm_exec",
        new_callable=AsyncMock,
        return_value=(0, "ok", ""),
    ):
        await _wait_for_docker_daemon(sandbox)


@pytest.mark.asyncio
async def test_wait_for_docker_daemon_tolerates_poll_timeout() -> None:
    """A docker info poll that exceeds its timeout means "not ready yet"."""
    sandbox = make_mock_sandbox()

    with patch(
        "inspect_sandboxes.novita._dind_project.vm_exec",
        new_callable=AsyncMock,
        side_effect=[TimeoutException("context deadline exceeded"), (0, "ok", "")],
    ):
        await _wait_for_docker_daemon(sandbox)


@pytest.mark.asyncio
async def test_wait_for_docker_daemon_times_out() -> None:
    sandbox = make_mock_sandbox()

    with (
        patch(
            "inspect_sandboxes.novita._dind_project.vm_exec",
            new_callable=AsyncMock,
            return_value=(1, "not ready", ""),
        ),
        patch("inspect_sandboxes.novita._dind_project._DAEMON_TIMEOUT", 2),
        patch("inspect_sandboxes.novita._dind_project._DAEMON_POLL_INTERVAL", 1),
        patch(
            "inspect_sandboxes.novita._dind_project.asyncio.sleep",
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
        "inspect_sandboxes.novita._dind_project.compose_exec",
        new_callable=AsyncMock,
        return_value=(0, ps_output, ""),
    ):
        await _wait_for_services(project, ["web", "helper"], timeout=10)


@pytest.mark.asyncio
async def test_wait_for_services_tolerates_poll_timeout() -> None:
    """A compose ps poll that exceeds its timeout means "not ready yet"."""
    project = make_mock_project()

    with patch(
        "inspect_sandboxes.novita._dind_project.compose_exec",
        new_callable=AsyncMock,
        side_effect=[
            TimeoutException("context deadline exceeded"),
            (0, '{"Service":"web"}\n{"Service":"helper"}\n', ""),
        ],
    ):
        await _wait_for_services(project, ["web", "helper"], timeout=10)


@pytest.mark.asyncio
async def test_wait_for_services_times_out() -> None:
    project = make_mock_project()

    with (
        patch(
            "inspect_sandboxes.novita._dind_project.compose_exec",
            new_callable=AsyncMock,
            return_value=(0, '{"Service":"web"}\n', ""),
        ),
        patch("inspect_sandboxes.novita._dind_project._SERVICE_POLL_INTERVAL", 1),
        patch(
            "inspect_sandboxes.novita._dind_project.asyncio.sleep",
            new_callable=AsyncMock,
        ),
    ):
        with pytest.raises(RuntimeError, match="Not all services running"):
            await _wait_for_services(project, ["web", "helper"], timeout=2)


def test_dind_template_name() -> None:
    assert (
        _dind_template_name(cpu_count=2, memory_mb=4096) == "inspect-dind-2cpu-4096mb"
    )
    assert (
        _dind_template_name(cpu_count=4, memory_mb=8192) == "inspect-dind-4cpu-8192mb"
    )


@pytest.mark.asyncio
async def test_ensure_dind_template_invokes_sdk() -> None:
    with patch("inspect_sandboxes.novita._dind_project.AsyncTemplate") as mock_cls:
        mock_cls.build = AsyncMock()
        instance = MagicMock()
        # Chain: AsyncTemplate().from_ubuntu_image().apt_install()...
        builder = MagicMock()
        builder.apt_install = MagicMock(return_value=builder)
        builder.run_cmd = MagicMock(return_value=builder)
        instance.from_ubuntu_image = MagicMock(return_value=builder)
        mock_cls.return_value = instance

        name = await _ensure_dind_template(cpu_count=2, memory_mb=4096)
        assert name == _dind_template_name(cpu_count=2, memory_mb=4096)
        mock_cls.build.assert_called_once()


@pytest.mark.asyncio
async def test_create_dind_project_full_sequence() -> None:
    """Test create_dind_project executes the full startup sequence."""
    config = ComposeConfig(services={"web": ComposeService(image="python:3.12")})
    sandbox = make_mock_sandbox()
    ps_output = '{"Service":"web"}\n'

    with (
        patch(
            "inspect_sandboxes.novita._dind_project._ensure_dind_template",
            new_callable=AsyncMock,
            return_value="inspect-dind-abc",
        ),
        patch(
            "inspect_sandboxes.novita._dind_project.AsyncSandbox.create",
            new_callable=AsyncMock,
            return_value=sandbox,
        ),
        patch(
            "inspect_sandboxes.novita._dind_project._wait_for_docker_daemon",
            new_callable=AsyncMock,
        ),
        patch(
            "inspect_sandboxes.novita._dind_project._upload_build_contexts",
            new_callable=AsyncMock,
            return_value="/inspect/compose/compose.yaml",
        ),
        patch(
            "inspect_sandboxes.novita._dind_project.compose_exec",
            new_callable=AsyncMock,
            return_value=(0, ps_output, ""),
        ) as mock_exec,
        patch(
            "inspect_sandboxes.novita._dind_project._wait_for_services",
            new_callable=AsyncMock,
        ),
    ):
        project = await create_dind_project(
            config, "/local/compose.yaml", metadata={"created_by": "test"}
        )

    assert project.sandbox is sandbox
    assert project.services == ["web"]

    subcommands = [c[0][1] for c in mock_exec.call_args_list]
    assert subcommands[0] == ["build"]
    assert subcommands[1] == ["pull", "--ignore-buildable"]
    assert subcommands[2][0] == "up"


@pytest.mark.asyncio
async def test_create_dind_project_cleans_up_on_failure() -> None:
    """Test create_dind_project kills sandbox when startup fails."""
    config = ComposeConfig(services={"web": ComposeService(image="python:3.12")})
    sandbox = make_mock_sandbox()

    with (
        patch(
            "inspect_sandboxes.novita._dind_project._ensure_dind_template",
            new_callable=AsyncMock,
            return_value="inspect-dind-abc",
        ),
        patch(
            "inspect_sandboxes.novita._dind_project.AsyncSandbox.create",
            new_callable=AsyncMock,
            return_value=sandbox,
        ),
        patch(
            "inspect_sandboxes.novita._dind_project._wait_for_docker_daemon",
            new_callable=AsyncMock,
            side_effect=RuntimeError("daemon failed"),
        ),
    ):
        with pytest.raises(RuntimeError, match="daemon failed"):
            await create_dind_project(
                config, "/local/compose.yaml", metadata={"created_by": "test"}
            )

    sandbox.kill.assert_called_once()


@pytest.mark.asyncio
async def test_destroy_dind_project_runs_compose_down() -> None:
    """destroy_dind_project should call docker compose down --remove-orphans."""
    project = make_mock_project()

    with patch(
        "inspect_sandboxes.novita._dind_project.compose_exec",
        new_callable=AsyncMock,
        return_value=(0, "", ""),
    ) as mock_exec:
        await destroy_dind_project(project)

    cmd = mock_exec.call_args[0][1]
    assert "down" in cmd
    assert "--remove-orphans" in cmd


@pytest.mark.asyncio
async def test_destroy_dind_project_swallows_errors() -> None:
    """Best-effort: errors from compose_exec should be logged, not raised."""
    project = make_mock_project()

    with patch(
        "inspect_sandboxes.novita._dind_project.compose_exec",
        new_callable=AsyncMock,
        side_effect=RuntimeError("network blip"),
    ):
        # Should not raise.
        await destroy_dind_project(project)


@pytest.mark.asyncio
async def test_upload_build_contexts_retries_transport_error() -> None:
    """_upload_build_contexts uses write_sandbox_file and retries on ReadError."""
    config = ComposeConfig(services={"web": ComposeService(image="python:3.12")})
    sandbox = make_mock_sandbox()
    sandbox.files.write = AsyncMock(side_effect=[httpx.ReadError("reset"), None])

    with (
        patch(
            "inspect_sandboxes.novita._dind_project._upload_directory",
            new_callable=AsyncMock,
        ),
        patch(
            "inspect_sandboxes.novita._dind_project.discover_build_contexts",
            return_value=({"/elsewhere/web": "/inspect/contexts/web"}, True),
        ),
    ):
        remote_path = await _upload_build_contexts(
            sandbox, config, "/local/project/compose.yaml"
        )

    assert remote_path == "/inspect/compose/compose.yaml"
    assert sandbox.files.write.await_count == 2
