"""Tests for DinD per-service sandbox environment."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from inspect_ai.util import ComposeConfig, ComposeService, SandboxUserUnsupportedError
from inspect_sandboxes.e2b._dind_env import E2BDinDServiceEnvironment
from inspect_sandboxes.e2b._dind_project import E2BDinDProject


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


def make_mock_project(sandbox: MagicMock | None = None) -> E2BDinDProject:
    if sandbox is None:
        sandbox = make_mock_sandbox()
    return E2BDinDProject(
        sandbox=sandbox,
        project_name="inspect-test1234",
        compose_path="/inspect/compose/compose.yaml",
        services=["web", "helper"],
    )


def make_env(
    project: E2BDinDProject | None = None,
    service: str = "web",
    working_dir: str = "/app",
) -> E2BDinDServiceEnvironment:
    if project is None:
        project = make_mock_project()
    return E2BDinDServiceEnvironment(project, service, working_dir)


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
        "inspect_sandboxes.e2b._dind_env.compose_exec",
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
        "inspect_sandboxes.e2b._dind_env.compose_exec",
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
        "inspect_sandboxes.e2b._dind_env.compose_exec",
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
        "inspect_sandboxes.e2b._dind_env.compose_exec",
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
        "inspect_sandboxes.e2b._dind_env.compose_exec",
        new_callable=AsyncMock,
        return_value=(0, "", ""),
    ) as mock_exec:
        await env.exec(["pwd"], cwd="subdir")

    cmd = mock_exec.call_args[0][1]
    assert cmd[3] == "/app/subdir"


@pytest.mark.asyncio
async def test_exec_stdin_two_hop_upload() -> None:
    """Test stdin is uploaded to VM then compose cp'd to the container."""
    sandbox = make_mock_sandbox()
    env = make_env(make_mock_project(sandbox))

    with (
        patch(
            "inspect_sandboxes.e2b._dind_env.compose_exec",
            new_callable=AsyncMock,
            return_value=(0, "stdin data", ""),
        ),
        patch(
            "inspect_sandboxes.e2b._dind_env.vm_exec",
            new_callable=AsyncMock,
            return_value=(0, "", ""),
        ),
    ):
        result = await env.exec(["cat"], input="hello")

    sandbox.files.write.assert_called_once()
    assert sandbox.files.write.call_args[0][0].startswith("/tmp/.inspect-stdin-")
    assert sandbox.files.write.call_args[0][1] == b"hello"
    assert result.stdout == "stdin data"


@pytest.mark.asyncio
async def test_file_operations_reject_an_explicit_user() -> None:
    """An explicit file user is refused rather than ignored."""
    env = make_env(service="web")
    with pytest.raises(SandboxUserUnsupportedError):
        await env.write_file("/app/test.txt", "hello", user="nobody")
    with pytest.raises(SandboxUserUnsupportedError):
        await env.read_file("/app/test.txt", user="nobody")


@pytest.mark.parametrize("user", [None, "root"])
@pytest.mark.asyncio
async def test_file_preflight_runs_as_the_file_user(user: str | None) -> None:
    """The mkdir, directory and size checks run in the service as the file user.

    The VM download runs as root: `compose cp` stages the file as root.
    """
    sandbox = make_mock_sandbox()
    sandbox.files.read = AsyncMock(return_value=bytearray(b"hello"))
    env = make_env(make_mock_project(sandbox), service="web")

    async def fake_compose_exec(
        project: E2BDinDProject, subcommand: list[str], **kwargs: Any
    ) -> tuple[int, str, str]:
        if "test" in subcommand:
            return (1, "", "")  # not a directory
        if "sh" in subcommand:
            return (0, "5\n", "")  # stat
        return (0, "", "")

    with (
        patch(
            "inspect_sandboxes.e2b._dind_env.compose_exec",
            side_effect=fake_compose_exec,
        ) as mock_exec,
        patch(
            "inspect_sandboxes.e2b._dind_env.vm_exec",
            new_callable=AsyncMock,
            return_value=(0, "", ""),
        ),
    ):
        await env.write_file("/app/sub/test.txt", "hello", user=user)
        assert await env.read_file("/app/sub/test.txt", user=user) == "hello"

    prefix = ["exec", "-T", *(["--user", user] if user else []), "web"]
    exec_calls = [c.args[1] for c in mock_exec.call_args_list if c.args[1][0] == "exec"]
    # write: mkdir, directory check; read: directory check, size
    assert len(exec_calls) == 4
    assert all(c[: len(prefix)] == prefix for c in exec_calls), exec_calls
    # utilities never resolve through the image's PATH
    assert all(c[len(prefix)] == "/bin/sh" for c in exec_calls), exec_calls
    assert sandbox.files.read.call_args.kwargs["user"] == "root"


@pytest.mark.asyncio
async def test_write_file_two_hop() -> None:
    """Test write_file uploads to VM temp, then compose cp to service."""
    env = make_env(service="web")
    env._is_directory = AsyncMock(return_value=False)  # type: ignore[method-assign]

    with (
        patch(
            "inspect_sandboxes.e2b._dind_env.compose_exec",
            new_callable=AsyncMock,
            return_value=(0, "", ""),
        ) as mock_exec,
        patch(
            "inspect_sandboxes.e2b._dind_env.vm_exec",
            new_callable=AsyncMock,
            return_value=(0, "", ""),
        ),
    ):
        await env.write_file("/app/test.txt", "hello")

    cp_calls = [c for c in mock_exec.call_args_list if "cp" in c[0][1]]
    assert len(cp_calls) == 1
    assert "web:/app/test.txt" in cp_calls[0][0][1]


@pytest.mark.parametrize("user", [None, "root"])
@pytest.mark.asyncio
async def test_read_file_two_hop(user: str | None) -> None:
    """Test read_file compose cp's from service, then SDK downloads from VM."""
    sandbox = make_mock_sandbox()
    sandbox.files.read = AsyncMock(return_value=bytearray(b"file content"))
    env = make_env(make_mock_project(sandbox))
    env._is_directory = AsyncMock(return_value=False)  # type: ignore[method-assign]
    env._get_file_size = AsyncMock(return_value=100)  # type: ignore[method-assign]

    with (
        patch(
            "inspect_sandboxes.e2b._dind_env.compose_exec",
            new_callable=AsyncMock,
            return_value=(0, "", ""),
        ),
        patch(
            "inspect_sandboxes.e2b._dind_env.vm_exec",
            new_callable=AsyncMock,
            return_value=(0, "", ""),
        ),
    ):
        result = await env.read_file("/app/test.txt", user=user)

    assert result == "file content"


@pytest.mark.asyncio
async def test_read_file_not_found() -> None:
    env = make_env()
    env._is_directory = AsyncMock(return_value=False)  # type: ignore[method-assign]

    with (
        patch(
            "inspect_sandboxes.e2b._dind_env.compose_exec",
            new_callable=AsyncMock,
            return_value=(1, "", "No such file or directory"),
        ),
        patch(
            "inspect_sandboxes.e2b._dind_env.vm_exec",
            new_callable=AsyncMock,
            return_value=(0, "", ""),
        ),
    ):
        with pytest.raises(FileNotFoundError):
            await env.read_file("/app/missing.txt")


@pytest.mark.asyncio
async def test_sample_cleanup_destroys_and_kills() -> None:
    sandbox = make_mock_sandbox()
    project = make_mock_project(sandbox)
    env = make_env(project)
    with patch(
        "inspect_sandboxes.e2b._dind_env.destroy_dind_project",
        new_callable=AsyncMock,
    ) as mock_destroy:
        await E2BDinDServiceEnvironment.sample_cleanup(
            "task", None, {"web": env}, False
        )

    mock_destroy.assert_awaited_once_with(project)
    sandbox.kill.assert_called_once()


@pytest.mark.asyncio
async def test_sample_cleanup_skips_when_interrupted() -> None:
    sandbox = make_mock_sandbox()
    env = make_env(make_mock_project(sandbox))
    with patch(
        "inspect_sandboxes.e2b._dind_env.destroy_dind_project",
        new_callable=AsyncMock,
    ) as mock_destroy:
        await E2BDinDServiceEnvironment.sample_cleanup("task", None, {"web": env}, True)

    mock_destroy.assert_not_called()
    sandbox.kill.assert_not_called()


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
    mock_project.sandbox.sandbox_id = "sb-123"
    mock_project.services = ["web", "helper"]

    with (
        patch(
            "inspect_sandboxes.e2b._dind_env.create_dind_project",
            new_callable=AsyncMock,
            return_value=mock_project,
        ) as mock_create,
        patch(
            "inspect_sandboxes.e2b._dind_env.discover_working_dir",
            new_callable=AsyncMock,
            return_value="/",
        ),
    ):
        envs = await E2BDinDServiceEnvironment.sample_init_dind(
            config, None, metadata={"created_by": "test"}
        )

    compose_file_arg = (
        mock_create.call_args[1].get("compose_file") or mock_create.call_args[0][1]
    )
    assert compose_file_arg is not None
    assert compose_file_arg.endswith("compose.yaml")

    assert list(envs.keys())[0] == "web"
    assert "helper" in envs
