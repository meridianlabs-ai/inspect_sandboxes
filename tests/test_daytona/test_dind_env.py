"""Tests for DinD per-service sandbox environment."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from inspect_ai.util import ComposeConfig, ComposeService
from inspect_sandboxes.daytona._daytona import _daytona_client, _init_context
from inspect_sandboxes.daytona._dind_env import DaytonaDinDServiceEnvironment
from inspect_sandboxes.daytona._dind_project import DaytonaDinDProject


def make_mock_sandbox(sandbox_id: str = "sb-dind-123") -> MagicMock:
    sandbox = MagicMock()
    sandbox.id = sandbox_id
    sandbox.process = MagicMock()
    sandbox.process.exec = AsyncMock(return_value=MagicMock(exit_code=0, result=""))
    sandbox.fs = MagicMock()
    sandbox.fs.upload_file = AsyncMock()
    sandbox.fs.download_file = AsyncMock(return_value=b"content")
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


def make_env(
    project: DaytonaDinDProject | None = None,
    service: str = "web",
    working_dir: str = "/app",
) -> DaytonaDinDServiceEnvironment:
    if project is None:
        project = make_mock_project()
    return DaytonaDinDServiceEnvironment(project, service, working_dir)


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
        "inspect_sandboxes.daytona._dind_env.compose_exec",
        new_callable=AsyncMock,
        return_value=(0, "output"),
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
        "inspect_sandboxes.daytona._dind_env.compose_exec",
        new_callable=AsyncMock,
        return_value=(0, ""),
    ) as mock_exec:
        await env.exec(["whoami"], user="testuser")

    cmd = mock_exec.call_args[0][1]
    assert cmd == ["exec", "-T", "-w", "/app", "--user", "testuser", "web", "whoami"]


@pytest.mark.asyncio
async def test_exec_with_env_vars_no_double_quoting() -> None:
    """Test env vars are passed as raw values — shlex.join in compose_exec handles quoting."""
    env = make_env(service="web", working_dir="/app")

    with patch(
        "inspect_sandboxes.daytona._dind_env.compose_exec",
        new_callable=AsyncMock,
        return_value=(0, ""),
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
        "inspect_sandboxes.daytona._dind_env.compose_exec",
        new_callable=AsyncMock,
        return_value=(0, ""),
    ) as mock_exec:
        await env.exec(["pwd"], cwd="/tmp")

    cmd = mock_exec.call_args[0][1]
    assert cmd[3] == "/tmp"


@pytest.mark.asyncio
async def test_exec_resolves_relative_cwd() -> None:
    env = make_env(working_dir="/app")

    with patch(
        "inspect_sandboxes.daytona._dind_env.compose_exec",
        new_callable=AsyncMock,
        return_value=(0, ""),
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
            "inspect_sandboxes.daytona._dind_env.sdk_upload", new_callable=AsyncMock
        ) as mock_upload,
        patch(
            "inspect_sandboxes.daytona._dind_env.compose_exec",
            new_callable=AsyncMock,
            return_value=(0, "stdin data"),
        ),
        patch(
            "inspect_sandboxes.daytona._dind_env.vm_exec",
            new_callable=AsyncMock,
            return_value=(0, ""),
        ),
    ):
        result = await env.exec(["cat"], input="hello")

    mock_upload.assert_called_once()
    assert mock_upload.call_args[0][2] == b"hello"
    assert result.stdout == "stdin data"


@pytest.mark.asyncio
async def test_write_file_two_hop() -> None:
    """Test write_file uploads to VM temp, then compose cp to service."""
    env = make_env(service="web")
    env._is_directory = AsyncMock(return_value=False)  # type: ignore[method-assign]

    with (
        patch("inspect_sandboxes.daytona._dind_env.sdk_upload", new_callable=AsyncMock),
        patch(
            "inspect_sandboxes.daytona._dind_env.compose_exec",
            new_callable=AsyncMock,
            return_value=(0, ""),
        ) as mock_exec,
        patch(
            "inspect_sandboxes.daytona._dind_env.vm_exec",
            new_callable=AsyncMock,
            return_value=(0, ""),
        ),
    ):
        await env.write_file("/app/test.txt", "hello")

    cp_calls = [c for c in mock_exec.call_args_list if "cp" in c[0][1]]
    assert len(cp_calls) == 1
    assert "web:/app/test.txt" in cp_calls[0][0][1]


@pytest.mark.asyncio
async def test_read_file_two_hop() -> None:
    """Test read_file compose cp's from service, then SDK downloads from VM."""
    env = make_env()
    env._is_directory = AsyncMock(return_value=False)  # type: ignore[method-assign]
    env._get_file_size = AsyncMock(return_value=100)  # type: ignore[method-assign]

    with (
        patch(
            "inspect_sandboxes.daytona._dind_env.sdk_download",
            new_callable=AsyncMock,
            return_value=b"file content",
        ),
        patch(
            "inspect_sandboxes.daytona._dind_env.compose_exec",
            new_callable=AsyncMock,
            return_value=(0, ""),
        ),
        patch(
            "inspect_sandboxes.daytona._dind_env.vm_exec",
            new_callable=AsyncMock,
            return_value=(0, ""),
        ),
    ):
        result = await env.read_file("/app/test.txt")

    assert result == "file content"


@pytest.mark.asyncio
async def test_read_file_not_found() -> None:
    env = make_env()

    with (
        patch(
            "inspect_sandboxes.daytona._dind_env.compose_exec",
            new_callable=AsyncMock,
            return_value=(1, "No such file or directory"),
        ),
        patch(
            "inspect_sandboxes.daytona._dind_env.vm_exec",
            new_callable=AsyncMock,
            return_value=(0, ""),
        ),
    ):
        with pytest.raises(FileNotFoundError):
            await env.read_file("/app/missing.txt")


@pytest.mark.asyncio
async def test_sample_cleanup_destroys_and_deletes() -> None:
    project = make_mock_project()
    env = DaytonaDinDServiceEnvironment(project, "web", "/app")

    _init_context()
    mock_client = MagicMock()
    mock_client.delete = AsyncMock()
    _daytona_client.set(mock_client)

    with patch(
        "inspect_sandboxes.daytona._dind_env.destroy_dind_project",
        new_callable=AsyncMock,
    ) as mock_destroy:
        await DaytonaDinDServiceEnvironment.sample_cleanup(
            "task",
            None,
            {"web": env},
            False,
        )

    mock_destroy.assert_called_once_with(project)
    mock_client.delete.assert_called_once()


@pytest.mark.asyncio
async def test_sample_cleanup_skips_when_interrupted() -> None:
    project = make_mock_project()
    env = DaytonaDinDServiceEnvironment(project, "web", "/app")

    _init_context()
    mock_client = MagicMock()
    mock_client.delete = AsyncMock()
    _daytona_client.set(mock_client)

    with patch(
        "inspect_sandboxes.daytona._dind_env.destroy_dind_project",
        new_callable=AsyncMock,
    ) as mock_destroy:
        await DaytonaDinDServiceEnvironment.sample_cleanup(
            "task",
            None,
            {"web": env},
            True,
        )

    mock_destroy.assert_not_called()
    mock_client.delete.assert_not_called()


@pytest.mark.asyncio
async def test_sample_init_dind_serializes_compose_config() -> None:
    """Test that sample_init_dind creates a temp YAML when compose_file is None."""
    config = ComposeConfig(
        services={
            "web": ComposeService(image="python:3.12", **{"x-default": True}),  # type: ignore[arg-type]
            "helper": ComposeService(image="alpine:3.20"),
        }
    )

    mock_project = MagicMock()
    mock_project.sandbox.id = "sb-123"
    mock_project.services = ["web", "helper"]

    with (
        patch(
            "inspect_sandboxes.daytona._dind_env.create_dind_project",
            new_callable=AsyncMock,
            return_value=mock_project,
        ) as mock_create,
        patch(
            "inspect_sandboxes.daytona._dind_env.discover_working_dir",
            new_callable=AsyncMock,
            return_value="/",
        ),
    ):
        envs = await DaytonaDinDServiceEnvironment.sample_init_dind(
            MagicMock(), config, None, {"created_by": "test"}
        )

    compose_file_arg = (
        mock_create.call_args[1].get("compose_file") or mock_create.call_args[0][2]
    )
    assert compose_file_arg is not None
    assert compose_file_arg.endswith("compose.yaml")

    assert list(envs.keys())[0] == "web"
    assert "helper" in envs
