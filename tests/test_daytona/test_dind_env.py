"""Tests for DinD per-service sandbox environment."""

from __future__ import annotations

import asyncio
import shlex
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from daytona import DaytonaError
from inspect_ai.util import ComposeConfig, ComposeService
from inspect_sandboxes.daytona._daytona import _daytona_client, _init_context
from inspect_sandboxes.daytona._dind_env import DaytonaDinDServiceEnvironment
from inspect_sandboxes.daytona._dind_project import DaytonaDinDProject, compose_command
from inspect_sandboxes.daytona._sandbox_utils import (
    TIMEOUT_GRACE,
    TIMEOUT_PATH,
    build_remove_command,
    build_session_command,
)


def session_response(exit_code: int, stdout: str = "", stderr: str = "") -> MagicMock:
    response = MagicMock()
    response.exit_code = exit_code
    response.stdout = stdout
    response.stderr = stderr
    return response


def scripted_session(*responses: tuple[int, str, str] | Exception) -> AsyncMock:
    """An ``execute_session_command`` fake answering each call in turn."""
    remaining = list(responses)

    async def run(
        session_id: str, request: Any, timeout: int | None = None
    ) -> MagicMock:
        response = remaining.pop(0)
        if isinstance(response, Exception):
            raise response
        return session_response(*response)

    return AsyncMock(side_effect=run)


def make_mock_sandbox(sandbox_id: str = "sb-dind-123") -> MagicMock:
    sandbox = MagicMock()
    sandbox.id = sandbox_id
    sandbox.process = MagicMock()
    # Housekeeping via process.exec (merged output) succeeds silently; the
    # services' exec() runs in VM sessions.
    sandbox.process.exec = AsyncMock(return_value=MagicMock(exit_code=0, result=""))
    sandbox.process.create_session = AsyncMock()
    sandbox.process.delete_session = AsyncMock()
    sandbox.process.execute_session_command = scripted_session((0, "", ""))
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


def session_commands(sandbox: MagicMock) -> list[str]:
    return [
        c.args[1].command
        for c in sandbox.process.execute_session_command.call_args_list
    ]


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
    env.project.sandbox.process.execute_session_command = scripted_session(
        (0, "output", "")
    )

    with patch(
        "inspect_sandboxes.daytona._dind_env.compose_command", wraps=compose_command
    ) as mock_cmd:
        result = await env.exec(["echo", "hi"])

    cmd = mock_cmd.call_args[0][1]
    assert cmd == ["exec", "-T", "-w", "/work", "helper", "echo", "hi"]
    assert result.success
    assert result.stdout == "output"


@pytest.mark.asyncio
async def test_exec_with_user_adds_user_flag() -> None:
    env = make_env(service="web", working_dir="/app")

    with patch(
        "inspect_sandboxes.daytona._dind_env.compose_command", wraps=compose_command
    ) as mock_cmd:
        await env.exec(["whoami"], user="testuser")

    cmd = mock_cmd.call_args[0][1]
    assert cmd == ["exec", "-T", "-w", "/app", "--user", "testuser", "web", "whoami"]


@pytest.mark.asyncio
async def test_exec_with_env_vars_no_double_quoting() -> None:
    """Test env vars are passed as raw values — shlex.join in compose_command handles quoting."""
    env = make_env(service="web", working_dir="/app")

    with patch(
        "inspect_sandboxes.daytona._dind_env.compose_command", wraps=compose_command
    ) as mock_cmd:
        await env.exec(
            ["sh", "-c", "echo $MY_VAR"],
            env={"MY_VAR": "hello world", "OTHER": "simple"},
        )

    cmd = mock_cmd.call_args[0][1]
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
        "inspect_sandboxes.daytona._dind_env.compose_command", wraps=compose_command
    ) as mock_cmd:
        await env.exec(["pwd"], cwd="/tmp")

    assert mock_cmd.call_args[0][1][3] == "/tmp"


@pytest.mark.asyncio
async def test_exec_resolves_relative_cwd() -> None:
    env = make_env(working_dir="/app")

    with patch(
        "inspect_sandboxes.daytona._dind_env.compose_command", wraps=compose_command
    ) as mock_cmd:
        await env.exec(["pwd"], cwd="subdir")

    assert mock_cmd.call_args[0][1][3] == "/app/subdir"


EXPECTED_COMPOSE = shlex.join(
    [
        "docker",
        "compose",
        "-p",
        "inspect-test1234",
        "--project-directory",
        "/inspect/compose",
        "-f",
        "/inspect/compose/compose.yaml",
        "exec",
        "-T",
        "-w",
        "/app",
        "web",
    ]
)


@pytest.mark.asyncio
async def test_exec_runs_the_compose_exec_in_a_vm_session() -> None:
    """The compose exec runs on the VM in a session, which returns both streams."""
    sandbox = make_mock_sandbox()
    sandbox.process.execute_session_command = scripted_session((0, "out\n", "err\n"))
    env = DaytonaDinDServiceEnvironment(make_mock_project(sandbox), "web", "/app")

    result = await env.exec(["sh", "-c", "echo out; echo err >&2"])

    assert (result.stdout, result.stderr, result.success) == ("out\n", "err\n", True)
    assert session_commands(sandbox) == [
        build_session_command(f"{EXPECTED_COMPOSE} sh -c 'echo out; echo err >&2'")
    ]
    assert sandbox.process.execute_session_command.call_args.kwargs["timeout"] is None


@pytest.mark.asyncio
async def test_exec_timeout_runs_inside_the_container() -> None:
    """The timeout wraps the container command, as the Docker sandbox does."""
    sandbox = make_mock_sandbox()
    sandbox.process.execute_session_command = scripted_session((124, "partial", ""))
    env = DaytonaDinDServiceEnvironment(make_mock_project(sandbox), "web", "/app")

    with pytest.raises(TimeoutError, match="timed out after 30 seconds") as info:
        await env.exec(["sleep", "99"], timeout=30)

    assert vars(info.value)["truncated_output"] == "partial"
    assert session_commands(sandbox) == [
        build_session_command(f"{EXPECTED_COMPOSE} {TIMEOUT_PATH} -k 5s 30s sleep 99")
    ]
    assert sandbox.process.execute_session_command.call_args.kwargs["timeout"] == (
        30 + TIMEOUT_GRACE
    )


@pytest.mark.asyncio
async def test_exec_stdin_two_hop_upload() -> None:
    """Test stdin is uploaded to VM then compose cp'd to the container."""
    sandbox = make_mock_sandbox()
    sandbox.process.execute_session_command = scripted_session((0, "stdin data", ""))
    env = DaytonaDinDServiceEnvironment(make_mock_project(sandbox), "web", "/app")

    with (
        patch(
            "inspect_sandboxes.daytona._dind_env.sdk_upload", new_callable=AsyncMock
        ) as mock_upload,
        patch(
            "inspect_sandboxes.daytona._dind_env.compose_exec",
            new_callable=AsyncMock,
            return_value=(0, ""),
        ) as mock_cp,
        patch(
            "inspect_sandboxes.daytona._dind_env.vm_exec",
            new_callable=AsyncMock,
            return_value=(0, ""),
        ) as mock_vm_exec,
    ):
        result = await env.exec(["cat"], input="hello")

    mock_upload.assert_called_once()
    stdin_vm_file = mock_upload.call_args[0][1]
    assert mock_upload.call_args[0][2] == b"hello"
    assert mock_cp.call_args[0][1][0] == "cp"
    assert result.stdout == "stdin data"
    command = session_commands(sandbox)[0]
    assert command.startswith(build_session_command(f"{EXPECTED_COMPOSE} sh -c ")[:-1])
    assert "< /tmp/.inspect-stdin-" in command
    # The VM's stdin temp file is removed afterwards, with rm from SYSTEM_PATH.
    mock_vm_exec.assert_called_once()
    assert mock_vm_exec.call_args[0][1] == build_remove_command([stdin_vm_file])


def make_local_shell_sandbox() -> MagicMock:
    """A DinD VM sandbox whose sessions run commands in the local shell.

    The ``docker compose ... exec -T -w /app web`` prefix is dropped so the
    service command runs directly; the session command built around it and
    the session pool are the real code path.
    """
    sandbox = make_mock_sandbox()

    async def run(
        session_id: str, request: Any, timeout: int | None = None
    ) -> MagicMock:
        command = request.command.replace(
            shlex.quote(EXPECTED_COMPOSE + " "), ""
        ).replace(EXPECTED_COMPOSE + " ", "")
        proc = await asyncio.create_subprocess_shell(
            command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        out, err = await proc.communicate()
        return session_response(proc.returncode or 0, out.decode(), err.decode())

    sandbox.process.execute_session_command = AsyncMock(side_effect=run)
    return sandbox


@pytest.mark.asyncio
async def test_exec_streams_round_trip_through_the_vm_shell() -> None:
    """Acceptance through a real ``sh``: stdout and stderr separate and intact."""
    sandbox = make_local_shell_sandbox()
    env = DaytonaDinDServiceEnvironment(make_mock_project(sandbox), "web", "/app")

    result = await env.exec(["sh", "-c", "echo out; echo err >&2; exit 5"])

    assert result.stdout == "out\n"
    assert result.stderr == "err\n"
    assert result.returncode == 5


@pytest.mark.asyncio
async def test_exec_waits_for_background_writers_on_the_vm() -> None:
    sandbox = make_local_shell_sandbox()
    env = DaytonaDinDServiceEnvironment(make_mock_project(sandbox), "web", "/app")

    result = await env.exec(
        ["sh", "-c", "echo early; (sleep 0.3; echo late; echo late-err >&2) &"]
    )

    assert (result.stdout, result.stderr) == ("early\nlate\n", "late-err\n")


@pytest.mark.asyncio
async def test_exec_retries_a_dead_vm_session() -> None:
    sandbox = make_mock_sandbox()
    sandbox.process.execute_session_command = scripted_session(
        DaytonaError("session process has exited"), (0, "ok", "")
    )
    env = DaytonaDinDServiceEnvironment(make_mock_project(sandbox), "web", "/app")

    result = await env.exec(["true"])

    assert result.stdout == "ok"
    calls = sandbox.process.execute_session_command.call_args_list
    assert calls[0].args[0] != calls[1].args[0]
    sandbox.process.delete_session.assert_awaited_once_with(calls[0].args[0])


@pytest.mark.asyncio
async def test_services_share_the_project_session_pool() -> None:
    sandbox = make_mock_sandbox()
    sandbox.process.execute_session_command = scripted_session((0, "", ""), (0, "", ""))
    project = make_mock_project(sandbox)
    await DaytonaDinDServiceEnvironment(project, "web", "/app").exec(["true"])
    await DaytonaDinDServiceEnvironment(project, "helper", "/").exec(["true"])

    sandbox.process.create_session.assert_awaited_once()


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
