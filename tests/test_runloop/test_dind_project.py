"""Tests for Runloop DinD project orchestration."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from inspect_ai.util import ComposeConfig, ComposeService
from inspect_sandboxes.runloop._dind_project import (
    RunloopDinDProject,
    _dind_blueprint_name,
    _download_file,
    _ensure_dind_blueprint,
    _upload_file,
    _wait_for_docker_daemon,
    _wait_for_services,
    compose_exec,
    create_dind_project,
    destroy_dind_project,
    vm_exec,
)
from inspect_sandboxes.runloop._single_env import LARGE_FILE_THRESHOLD
from runloop_api_client import APIConnectionError


def _make_execution(
    *, stdout: str = "", stderr: str = "", exit_status: int = 0
) -> MagicMock:
    execution = MagicMock()
    execution.stdout = stdout
    execution.stderr = stderr
    execution.exit_status = exit_status
    execution.status = "completed"
    return execution


def _async_iter(items: list[Any]) -> Any:
    async def _aiter():
        for item in items:
            yield item

    return _aiter()


def make_mock_client() -> MagicMock:
    client = MagicMock()
    client.devboxes = MagicMock()
    completed = _make_execution(stdout="ok")
    client.devboxes.execute_and_await_completion = AsyncMock(return_value=completed)
    client.devboxes.execute_async = AsyncMock(return_value=completed)
    client.devboxes.executions = MagicMock()
    client.devboxes.executions.retrieve = AsyncMock(return_value=completed)
    client.devboxes.executions.kill = AsyncMock()
    client.devboxes.create_and_await_running = AsyncMock()
    client.devboxes.shutdown = AsyncMock()
    client.blueprints = MagicMock()
    client.blueprints.list = MagicMock(return_value=_async_iter([]))
    created_bp = MagicMock()
    created_bp.id = "bp_test_created"
    client.blueprints.create = AsyncMock(return_value=created_bp)
    client.blueprints.await_build_complete = AsyncMock()
    client.with_options = MagicMock(return_value=client)
    return client


def make_mock_project(client: MagicMock | None = None) -> RunloopDinDProject:
    if client is None:
        client = make_mock_client()
    return RunloopDinDProject(
        client=client,
        devbox_id="dbx-dind-123",
        project_name="inspect-test1234",
        compose_path="/home/user/inspect/compose/compose.yaml",
        services=["web", "helper"],
    )


@pytest.mark.asyncio
async def test_vm_exec_wraps_with_sh_c() -> None:
    client = make_mock_client()
    await vm_exec(client, "dbx-1", "echo hello", timeout=10)
    call_kwargs = client.devboxes.execute_async.await_args.kwargs
    assert call_kwargs["command"] == "sh -c 'echo hello'"


@pytest.mark.asyncio
async def test_vm_exec_returns_exit_code_and_output() -> None:
    client = make_mock_client()
    completed = _make_execution(stdout="hello", stderr="warn", exit_status=0)
    client.devboxes.executions.retrieve = AsyncMock(return_value=completed)
    exit_code, stdout, stderr = await vm_exec(client, "dbx-1", "echo hello")
    assert exit_code == 0
    assert stdout == "hello"
    assert stderr == "warn"


@pytest.mark.asyncio
async def test_vm_exec_polls_without_resubmitting() -> None:
    """A transient error mid-poll is tolerated: the command is submitted once."""
    client = make_mock_client()
    client.devboxes.execute_async = AsyncMock(
        return_value=MagicMock(execution_id="exec-1")
    )
    client.devboxes.executions.retrieve = AsyncMock(
        side_effect=[
            APIConnectionError(request=MagicMock()),  # transient blip mid-poll
            _make_execution(stdout="ok", exit_status=0),  # then completes
        ]
    )

    exit_code, _, _ = await vm_exec(client, "dbx-1", "echo hi", timeout=60)

    assert exit_code == 0
    assert client.devboxes.execute_async.await_count == 1
    assert client.devboxes.executions.retrieve.await_count == 2


@pytest.mark.asyncio
async def test_compose_exec_builds_correct_command() -> None:
    project = make_mock_project()

    with patch(
        "inspect_sandboxes.runloop._dind_project.vm_exec",
        new_callable=AsyncMock,
        return_value=(0, "", ""),
    ) as mock_vm:
        await compose_exec(project, ["ps", "--status", "running"], timeout=15)

    cmd = mock_vm.call_args[0][2]
    assert "sudo" in cmd
    assert "docker compose" in cmd
    assert "-p inspect-test1234" in cmd
    assert "--project-directory /home/user/inspect/compose" in cmd
    assert "/home/user/inspect/compose/compose.yaml" in cmd
    assert "ps" in cmd


@pytest.mark.asyncio
async def test_compose_exec_inlines_env_vars() -> None:
    project = make_mock_project()

    with patch(
        "inspect_sandboxes.runloop._dind_project.vm_exec",
        new_callable=AsyncMock,
        return_value=(0, "", ""),
    ) as mock_vm:
        await compose_exec(project, ["up"], env={"FOO": "bar", "BAZ": "qux"})

    cmd = mock_vm.call_args[0][2]
    assert cmd.startswith("FOO=bar BAZ=qux sudo docker compose")


@pytest.mark.asyncio
async def test_upload_file_large_uses_object_api() -> None:
    """Files at/above LARGE_FILE_THRESHOLD route through the Objects API."""
    client = make_mock_client()
    obj = MagicMock(id="obj-up", upload_url="https://upload.example/put")
    download = MagicMock(download_url="https://download.example/get")
    client.objects = MagicMock()
    client.objects.create = AsyncMock(return_value=obj)
    client.objects.complete = AsyncMock()
    client.objects.download = AsyncMock(return_value=download)
    client.objects.delete = AsyncMock()

    big = b"\x00" * LARGE_FILE_THRESHOLD
    with (
        patch(
            "inspect_sandboxes.runloop._dind_project.vm_exec",
            new_callable=AsyncMock,
            return_value=(0, "", ""),
        ) as mock_vm,
        patch(
            "inspect_sandboxes.runloop._dind_project.httpx.AsyncClient"
        ) as mock_http_cls,
    ):
        http = mock_http_cls.return_value.__aenter__.return_value
        http.put = AsyncMock(return_value=MagicMock(raise_for_status=MagicMock()))
        await _upload_file(client, "dbx-1", "/tmp/big.bin", big)

    client.objects.create.assert_awaited_once()
    http.put.assert_awaited_once_with("https://upload.example/put", content=big)
    client.objects.complete.assert_awaited_once_with("obj-up")
    cmd = mock_vm.call_args[0][2]
    assert "curl" in cmd
    assert "https://download.example/get" in cmd
    assert "/tmp/big.bin" in cmd
    client.objects.delete.assert_awaited_once_with("obj-up")


@pytest.mark.asyncio
async def test_download_file_large_uses_object_api() -> None:
    """Files at/above LARGE_FILE_THRESHOLD route through the Objects API."""
    client = make_mock_client()
    obj = MagicMock(id="obj-dn", upload_url="https://upload.example/put")
    download = MagicMock(download_url="https://download.example/get")
    client.objects = MagicMock()
    client.objects.create = AsyncMock(return_value=obj)
    client.objects.complete = AsyncMock()
    client.objects.download = AsyncMock(return_value=download)
    client.objects.delete = AsyncMock()

    payload = b"\xff" * LARGE_FILE_THRESHOLD
    with (
        patch(
            "inspect_sandboxes.runloop._dind_project.vm_exec",
            new_callable=AsyncMock,
            return_value=(0, "", ""),
        ) as mock_vm,
        patch(
            "inspect_sandboxes.runloop._dind_project.httpx.AsyncClient"
        ) as mock_http_cls,
    ):
        http = mock_http_cls.return_value.__aenter__.return_value
        http.get = AsyncMock(
            return_value=MagicMock(content=payload, raise_for_status=MagicMock())
        )
        result = await _download_file(
            client, "dbx-1", "/tmp/big.bin", size=LARGE_FILE_THRESHOLD
        )

    assert result == payload
    cmd = mock_vm.call_args[0][2]
    assert "curl" in cmd
    assert "--upload-file" in cmd
    assert "/tmp/big.bin" in cmd
    assert "https://upload.example/put" in cmd
    http.get.assert_awaited_once_with("https://download.example/get")
    client.objects.delete.assert_awaited_once_with("obj-dn")


@pytest.mark.asyncio
async def test_wait_for_docker_daemon_succeeds() -> None:
    client = make_mock_client()
    with patch(
        "inspect_sandboxes.runloop._dind_project.vm_exec",
        new_callable=AsyncMock,
        return_value=(0, "ok", ""),
    ):
        await _wait_for_docker_daemon(client, "dbx-dind-123")


@pytest.mark.asyncio
async def test_wait_for_docker_daemon_times_out() -> None:
    client = make_mock_client()
    with (
        patch(
            "inspect_sandboxes.runloop._dind_project.vm_exec",
            new_callable=AsyncMock,
            return_value=(1, "not ready", ""),
        ),
        patch("inspect_sandboxes.runloop._dind_project._DAEMON_TIMEOUT", 2),
        patch("inspect_sandboxes.runloop._dind_project._DAEMON_POLL_INTERVAL", 1),
        patch(
            "inspect_sandboxes.runloop._dind_project.asyncio.sleep",
            new_callable=AsyncMock,
        ),
    ):
        with pytest.raises(RuntimeError, match="Docker daemon not ready"):
            await _wait_for_docker_daemon(client, "dbx-dind-123")


@pytest.mark.asyncio
async def test_wait_for_services_succeeds() -> None:
    project = make_mock_project()
    ps_output = '{"Service":"web"}\n{"Service":"helper"}\n'

    with patch(
        "inspect_sandboxes.runloop._dind_project.compose_exec",
        new_callable=AsyncMock,
        return_value=(0, ps_output, ""),
    ):
        await _wait_for_services(project, ["web", "helper"], timeout=10)


@pytest.mark.asyncio
async def test_wait_for_services_times_out() -> None:
    project = make_mock_project()

    with (
        patch(
            "inspect_sandboxes.runloop._dind_project.compose_exec",
            new_callable=AsyncMock,
            return_value=(0, '{"Service":"web"}\n', ""),
        ),
        patch("inspect_sandboxes.runloop._dind_project._SERVICE_POLL_INTERVAL", 1),
        patch(
            "inspect_sandboxes.runloop._dind_project.asyncio.sleep",
            new_callable=AsyncMock,
        ),
    ):
        with pytest.raises(RuntimeError, match="Not all services running"):
            await _wait_for_services(project, ["web", "helper"], timeout=2)


def test_dind_blueprint_name_is_derived() -> None:
    # Pinned hashes so any change to _DIND_DOCKERFILE (or how it's hashed)
    # forces a deliberate test update — and a fresh blueprint build —
    # rather than silently reusing a stale cached blueprint.
    assert _dind_blueprint_name() == "inspect-dind-f6e19e98e5aa"
    assert (
        _dind_blueprint_name(launch_parameters={"custom_cpu_cores": 4})
        == "inspect-dind-6603708926e7"
    )


@pytest.mark.asyncio
async def test_ensure_dind_blueprint_invokes_sdk() -> None:
    client = make_mock_client()
    name = await _ensure_dind_blueprint(client, launch_parameters=None)
    assert name == _dind_blueprint_name(launch_parameters=None)
    client.blueprints.create.assert_called_once()
    kwargs = client.blueprints.create.await_args.kwargs
    # Idempotency key sent for forward-compat; SDK retries disabled at the
    # top via with_options(max_retries=0).
    assert kwargs["idempotency_key"] == name
    client.with_options.assert_called_once_with(max_retries=0)


@pytest.mark.asyncio
async def test_create_dind_project_full_sequence() -> None:
    """Test create_dind_project executes the full startup sequence."""
    config = ComposeConfig(services={"web": ComposeService(image="python:3.12")})

    devbox = MagicMock()
    devbox.id = "dbx-dind-123"
    client = make_mock_client()
    client.devboxes.create_and_await_running = AsyncMock(return_value=devbox)

    ps_output = '{"Service":"web"}\n'
    with (
        patch(
            "inspect_sandboxes.runloop._dind_project._ensure_dind_blueprint",
            new_callable=AsyncMock,
            return_value="inspect-dind-xyz",
        ),
        patch(
            "inspect_sandboxes.runloop._dind_project.vm_exec",
            new_callable=AsyncMock,
            return_value=(0, "", ""),
        ),
        patch(
            "inspect_sandboxes.runloop._dind_project._wait_for_docker_daemon",
            new_callable=AsyncMock,
        ),
        patch(
            "inspect_sandboxes.runloop._dind_project._upload_build_contexts",
            new_callable=AsyncMock,
            return_value="/home/user/inspect/compose/compose.yaml",
        ),
        patch(
            "inspect_sandboxes.runloop._dind_project.compose_exec",
            new_callable=AsyncMock,
            return_value=(0, ps_output, ""),
        ),
        patch(
            "inspect_sandboxes.runloop._dind_project._wait_for_services",
            new_callable=AsyncMock,
        ),
    ):
        project = await create_dind_project(
            client, config, "/local/compose.yaml", metadata={"created_by": "test"}
        )

    assert project.devbox_id == "dbx-dind-123"
    assert project.services == ["web"]


@pytest.mark.asyncio
async def test_create_dind_project_cleans_up_on_failure() -> None:
    """Test create_dind_project shuts down devbox when startup fails."""
    config = ComposeConfig(services={"web": ComposeService(image="python:3.12")})

    devbox = MagicMock()
    devbox.id = "dbx-dind-123"
    client = make_mock_client()
    client.devboxes.create_and_await_running = AsyncMock(return_value=devbox)

    with (
        patch(
            "inspect_sandboxes.runloop._dind_project._ensure_dind_blueprint",
            new_callable=AsyncMock,
            return_value="inspect-dind-xyz",
        ),
        patch(
            "inspect_sandboxes.runloop._dind_project.vm_exec",
            new_callable=AsyncMock,
            return_value=(0, "", ""),
        ),
        patch(
            "inspect_sandboxes.runloop._dind_project._wait_for_docker_daemon",
            new_callable=AsyncMock,
            side_effect=RuntimeError("daemon failed"),
        ),
    ):
        with pytest.raises(RuntimeError, match="daemon failed"):
            await create_dind_project(
                client, config, "/local/compose.yaml", metadata={"created_by": "test"}
            )

    client.devboxes.shutdown.assert_awaited_with("dbx-dind-123")


@pytest.mark.asyncio
async def test_destroy_dind_project_runs_compose_down() -> None:
    """destroy_dind_project should call docker compose down --remove-orphans."""
    project = make_mock_project()

    with patch(
        "inspect_sandboxes.runloop._dind_project.compose_exec",
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
        "inspect_sandboxes.runloop._dind_project.compose_exec",
        new_callable=AsyncMock,
        side_effect=RuntimeError("network blip"),
    ):
        # Should not raise.
        await destroy_dind_project(project)
