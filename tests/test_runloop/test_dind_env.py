"""Tests for RunloopDinDServiceEnvironment."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from inspect_ai.util import ComposeConfig, ComposeService
from inspect_sandboxes.runloop._dind_env import RunloopDinDServiceEnvironment
from inspect_sandboxes.runloop._dind_project import RunloopDinDProject


def make_mock_client() -> MagicMock:
    client = MagicMock()
    client.devboxes = MagicMock()
    client.devboxes.execute_and_await_completion = AsyncMock()
    client.devboxes.shutdown = AsyncMock()
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


def make_env(
    project: RunloopDinDProject | None = None,
    service: str = "web",
    working_dir: str = "/app",
) -> RunloopDinDServiceEnvironment:
    if project is None:
        project = make_mock_project()
    return RunloopDinDServiceEnvironment(project, service, working_dir)


def test_container_file_resolves_relative_path() -> None:
    env = make_env(working_dir="/app")
    assert env._container_file("test.txt") == "/app/test.txt"
    assert env._container_file("sub/dir/file.py") == "/app/sub/dir/file.py"


def test_container_file_preserves_absolute_path() -> None:
    env = make_env(working_dir="/app")
    assert env._container_file("/tmp/test.txt") == "/tmp/test.txt"


@pytest.mark.asyncio
async def test_exec_routes_to_service_with_correct_command() -> None:
    """Test exec builds compose exec args targeting the correct service."""
    env = make_env(service="helper", working_dir="/work")
    with patch(
        "inspect_sandboxes.runloop._dind_env.compose_exec",
        new_callable=AsyncMock,
        return_value=(0, "output", ""),
    ) as mock_exec:
        result = await env.exec(["echo", "hi"])

    cmd = mock_exec.call_args[0][1]
    assert cmd == ["exec", "-T", "-w", "/work", "helper", "echo", "hi"]
    assert result.success
    assert result.stdout == "output"


@pytest.mark.asyncio
async def test_exec_with_user_adds_user_flag() -> None:
    env = make_env(service="web", working_dir="/app")
    with patch(
        "inspect_sandboxes.runloop._dind_env.compose_exec",
        new_callable=AsyncMock,
        return_value=(0, "", ""),
    ) as mock_exec:
        await env.exec(["whoami"], user="testuser")

    cmd = mock_exec.call_args[0][1]
    assert cmd == ["exec", "-T", "-w", "/app", "--user", "testuser", "web", "whoami"]


@pytest.mark.asyncio
async def test_exec_with_env_vars_no_double_quoting() -> None:
    """Test env vars are passed as raw values — shlex.join in compose_exec handles quoting."""
    env = make_env(service="web", working_dir="/app")
    with patch(
        "inspect_sandboxes.runloop._dind_env.compose_exec",
        new_callable=AsyncMock,
        return_value=(0, "", ""),
    ) as mock_exec:
        await env.exec(
            ["sh", "-c", "echo $MY_VAR"],
            env={"MY_VAR": "hello world", "OTHER": "simple"},
        )

    cmd = mock_exec.call_args[0][1]
    # Values should be raw (no shlex.quote wrapping) — compose_exec's shlex.join handles it
    assert cmd == [
        "exec",
        "-T",
        "-w",
        "/app",
        "-e",
        "MY_VAR=hello world",
        "-e",
        "OTHER=simple",
        "web",
        "sh",
        "-c",
        "echo $MY_VAR",
    ]


@pytest.mark.asyncio
async def test_exec_with_cwd() -> None:
    env = make_env(working_dir="/app")
    with patch(
        "inspect_sandboxes.runloop._dind_env.compose_exec",
        new_callable=AsyncMock,
        return_value=(0, "", ""),
    ) as mock_exec:
        await env.exec(["pwd"], cwd="/tmp")

    cmd = mock_exec.call_args[0][1]
    assert cmd[3] == "/tmp"


@pytest.mark.asyncio
async def test_exec_resolves_relative_cwd() -> None:
    env = make_env(working_dir="/app")
    with patch(
        "inspect_sandboxes.runloop._dind_env.compose_exec",
        new_callable=AsyncMock,
        return_value=(0, "", ""),
    ) as mock_exec:
        await env.exec(["pwd"], cwd="subdir")

    cmd = mock_exec.call_args[0][1]
    assert cmd[3] == "/app/subdir"


@pytest.mark.asyncio
async def test_exec_stdin_two_hop_upload() -> None:
    """Test stdin is uploaded to VM then compose cp'd to the container."""
    env = make_env()

    with (
        patch(
            "inspect_sandboxes.runloop._dind_env.compose_exec",
            new_callable=AsyncMock,
            return_value=(0, "stdin data", ""),
        ),
        patch(
            "inspect_sandboxes.runloop._dind_env._upload_file",
            new_callable=AsyncMock,
        ) as mock_upload,
        patch(
            "inspect_sandboxes.runloop._dind_env.vm_exec",
            new_callable=AsyncMock,
            return_value=(0, "", ""),
        ),
    ):
        result = await env.exec(["cat"], input="hello")

    mock_upload.assert_called_once()
    upload_args = mock_upload.call_args[0]
    assert upload_args[2].startswith("/tmp/.inspect-stdin-")
    assert upload_args[3] == b"hello"
    assert result.stdout == "stdin data"


@pytest.mark.asyncio
async def test_write_file_two_hop() -> None:
    """Test write_file uploads to VM temp, then compose cp to service."""
    env = make_env(service="web")
    env._is_directory = AsyncMock(return_value=False)  # type: ignore[method-assign]
    env._create_parent_folder = AsyncMock()  # type: ignore[method-assign]

    with (
        patch(
            "inspect_sandboxes.runloop._dind_env.compose_exec",
            new_callable=AsyncMock,
            return_value=(0, "", ""),
        ) as mock_exec,
        patch(
            "inspect_sandboxes.runloop._dind_env._upload_file",
            new_callable=AsyncMock,
        ),
        patch(
            "inspect_sandboxes.runloop._dind_env.vm_exec",
            new_callable=AsyncMock,
            return_value=(0, "", ""),
        ),
    ):
        await env.write_file("/app/test.txt", "hello")

    cp_calls = [c for c in mock_exec.call_args_list if "cp" in c[0][1]]
    assert len(cp_calls) == 1
    assert "web:/app/test.txt" in cp_calls[0][0][1]


@pytest.mark.asyncio
async def test_read_file_two_hop() -> None:
    """Test read_file compose cp's from service, then reads bytes from VM temp."""
    env = make_env(service="web")
    env._is_directory = AsyncMock(return_value=False)  # type: ignore[method-assign]
    env._get_file_size = AsyncMock(return_value=100)  # type: ignore[method-assign]

    with (
        patch(
            "inspect_sandboxes.runloop._dind_env.compose_exec",
            new_callable=AsyncMock,
            return_value=(0, "", ""),
        ),
        patch(
            "inspect_sandboxes.runloop._dind_env.vm_exec",
            new_callable=AsyncMock,
            return_value=(0, "", ""),
        ),
        patch(
            "inspect_sandboxes.runloop._dind_env._download_file",
            new_callable=AsyncMock,
            return_value=b"file content",
        ),
    ):
        result = await env.read_file("/app/test.txt")

    assert result == "file content"


@pytest.mark.asyncio
async def test_read_file_not_found() -> None:
    env = make_env()
    env._is_directory = AsyncMock(return_value=False)  # type: ignore[method-assign]
    env._get_file_size = AsyncMock(return_value=100)  # type: ignore[method-assign]

    with patch(
        "inspect_sandboxes.runloop._dind_env.compose_exec",
        new_callable=AsyncMock,
        return_value=(1, "", "No such file or directory"),
    ):
        with pytest.raises(FileNotFoundError):
            await env.read_file("/app/missing.txt")


@pytest.mark.asyncio
async def test_sample_cleanup_destroys_and_shuts_down() -> None:
    client = make_mock_client()
    project = make_mock_project(client)
    env = make_env(project)
    with patch(
        "inspect_sandboxes.runloop._dind_env.destroy_dind_project",
        new_callable=AsyncMock,
    ) as mock_destroy:
        await RunloopDinDServiceEnvironment.sample_cleanup(
            "task", None, {"web": env}, False
        )

    mock_destroy.assert_awaited_once_with(project)
    client.devboxes.shutdown.assert_awaited_once_with("dbx-dind-123")


@pytest.mark.asyncio
async def test_sample_cleanup_skips_when_interrupted() -> None:
    client = make_mock_client()
    env = make_env(make_mock_project(client))
    with patch(
        "inspect_sandboxes.runloop._dind_env.destroy_dind_project",
        new_callable=AsyncMock,
    ) as mock_destroy:
        await RunloopDinDServiceEnvironment.sample_cleanup(
            "task", None, {"web": env}, True
        )

    mock_destroy.assert_not_called()
    client.devboxes.shutdown.assert_not_called()


@pytest.mark.asyncio
async def test_sample_init_dind_serializes_compose_config() -> None:
    """Test that sample_init_dind creates a temp YAML when compose_file is None."""
    client = make_mock_client()
    config = ComposeConfig(
        services={
            "web": ComposeService(image="python:3.12", **{"x-default": True}),  # type: ignore[arg-type]
            "helper": ComposeService(image="alpine:3.20"),
        }
    )

    mock_project = MagicMock()
    mock_project.devbox_id = "dbx-dind-123"
    mock_project.services = ["web", "helper"]

    with (
        patch(
            "inspect_sandboxes.runloop._dind_env.create_dind_project",
            new_callable=AsyncMock,
            return_value=mock_project,
        ) as mock_create,
        patch(
            "inspect_sandboxes.runloop._dind_env.discover_working_dir",
            new_callable=AsyncMock,
            return_value="/",
        ),
    ):
        envs = await RunloopDinDServiceEnvironment.sample_init_dind(
            client, config, None, metadata={"created_by": "test"}
        )

    compose_file_arg = (
        mock_create.call_args[1].get("compose_file") or mock_create.call_args[0][2]
    )
    assert compose_file_arg is not None
    assert compose_file_arg.endswith("compose.yaml")

    assert list(envs.keys())[0] == "web"
    assert "helper" in envs


@pytest.mark.asyncio
async def test_sample_init_dind_defaults_custom_size() -> None:
    """Test the DinD path defaults resource_size_request to CUSTOM_SIZE — Runloop rejects custom_* without it."""
    client = make_mock_client()
    config = ComposeConfig(
        services={
            "web": ComposeService(image="python:3.12", **{"x-default": True}),  # type: ignore[arg-type]
            "helper": ComposeService(image="alpine:3.20"),
        },
        **{"x-runloop": {"launch_parameters": {"custom_cpu_cores": 4}}},  # type: ignore[arg-type]
    )

    mock_project = MagicMock()
    mock_project.devbox_id = "dbx-dind-123"
    mock_project.services = ["web", "helper"]

    with (
        patch(
            "inspect_sandboxes.runloop._dind_env.create_dind_project",
            new_callable=AsyncMock,
            return_value=mock_project,
        ) as mock_create,
        patch(
            "inspect_sandboxes.runloop._dind_env.discover_working_dir",
            new_callable=AsyncMock,
            return_value="/",
        ),
    ):
        await RunloopDinDServiceEnvironment.sample_init_dind(
            client, config, None, metadata={"created_by": "test"}
        )

    launch_parameters = mock_create.call_args.kwargs["launch_parameters"]
    assert launch_parameters["custom_cpu_cores"] == 4
    assert launch_parameters["resource_size_request"] == "CUSTOM_SIZE"
