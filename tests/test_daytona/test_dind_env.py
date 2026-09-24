"""Tests for DinD per-service sandbox environment."""

from __future__ import annotations

import asyncio
import base64
import re
import shlex
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from daytona import DaytonaError
from inspect_ai.util import ComposeConfig, ComposeService
from inspect_sandboxes.daytona._daytona import _daytona_client, _init_context
from inspect_sandboxes.daytona._dind_env import (
    DaytonaDinDServiceEnvironment,
    _timed_out,
)
from inspect_sandboxes.daytona._dind_project import DaytonaDinDProject, compose_command
from inspect_sandboxes.daytona._exec_capture import (
    ExecCapture,
    OutputCollectionError,
    build_capture_command,
    build_remove_command,
)

TAG_RE = re.compile(r"inspect-exec-([0-9a-f]{32})")


def tag_of(command: str) -> str:
    match = TAG_RE.search(command)
    assert match is not None, f"no capture tag in {command!r}"
    return match.group(1)


def framed(command: str, stdout: str = "", stderr: str = "") -> str:
    """What the capture wrapper in *command* prints for the given streams."""
    tag = tag_of(command)
    out = base64.b64encode(stdout.encode()).decode()
    err = base64.b64encode(stderr.encode()).decode()
    return (
        f"<<inspect-exec-{tag}:stdout>>{out}<<inspect-exec-{tag}:stderr>>"
        f"{err}<<inspect-exec-{tag}:end>>"
    )


def scripted_vm_exec(*responses: tuple[int, str, str] | Exception) -> AsyncMock:
    """A ``vm_exec`` fake answering each call in turn, framed for the command's tag."""
    remaining = list(responses)

    async def run(
        sandbox: Any, command: str, timeout: int | None = 60
    ) -> tuple[int, str]:
        response = remaining.pop(0)
        if isinstance(response, Exception):
            raise response
        exit_code, stdout, stderr = response
        # Housekeeping commands (rm -f ...) carry no capture tag and no frame.
        if not TAG_RE.search(command):
            return exit_code, stdout
        return exit_code, framed(command, stdout, stderr)

    return AsyncMock(side_effect=run)


def make_mock_sandbox(sandbox_id: str = "sb-dind-123") -> MagicMock:
    sandbox = MagicMock()
    sandbox.id = sandbox_id
    sandbox.process = MagicMock()

    # Every VM command succeeds with empty streams, framed when captured.
    async def run(command: str, **kwargs: Any) -> MagicMock:
        result = framed(command) if TAG_RE.search(command) else ""
        return MagicMock(exit_code=0, result=result)

    sandbox.process.exec = AsyncMock(side_effect=run)
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

    with (
        patch(
            "inspect_sandboxes.daytona._dind_env.compose_command",
            wraps=compose_command,
        ) as mock_cmd,
        patch(
            "inspect_sandboxes.daytona._dind_env.vm_exec",
            scripted_vm_exec((0, "output", "")),
        ),
    ):
        result = await env.exec(["echo", "hi"])

    cmd = mock_cmd.call_args[0][1]
    assert cmd == ["exec", "-T", "-w", "/work", "helper", "echo", "hi"]
    assert result.success
    assert result.stdout == "output"


@pytest.mark.asyncio
async def test_exec_with_user_adds_user_flag() -> None:
    env = make_env(service="web", working_dir="/app")

    with patch(
        "inspect_sandboxes.daytona._dind_env.compose_command",
        wraps=compose_command,
    ) as mock_cmd:
        await env.exec(["whoami"], user="testuser")

    cmd = mock_cmd.call_args[0][1]
    assert cmd == ["exec", "-T", "-w", "/app", "--user", "testuser", "web", "whoami"]


@pytest.mark.asyncio
async def test_exec_with_env_vars_no_double_quoting() -> None:
    """Test env vars are passed as raw values — shlex.join in compose_exec handles quoting."""
    env = make_env(service="web", working_dir="/app")

    with patch(
        "inspect_sandboxes.daytona._dind_env.compose_command",
        wraps=compose_command,
    ) as mock_cmd:
        await env.exec(
            ["sh", "-c", "echo $MY_VAR"],
            env={"MY_VAR": "hello world", "OTHER": "simple"},
        )

    cmd = mock_cmd.call_args[0][1]
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
        "inspect_sandboxes.daytona._dind_env.compose_command",
        wraps=compose_command,
    ) as mock_cmd:
        await env.exec(["pwd"], cwd="/tmp")

    cmd = mock_cmd.call_args[0][1]
    assert cmd[3] == "/tmp"


@pytest.mark.asyncio
async def test_exec_resolves_relative_cwd() -> None:
    env = make_env(working_dir="/app")

    with patch(
        "inspect_sandboxes.daytona._dind_env.compose_command",
        wraps=compose_command,
    ) as mock_cmd:
        await env.exec(["pwd"], cwd="subdir")

    cmd = mock_cmd.call_args[0][1]
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
            return_value=(0, ""),
        ) as mock_cp,
        patch(
            "inspect_sandboxes.daytona._dind_env.vm_exec",
            scripted_vm_exec((0, "stdin data", ""), (0, "", "")),
        ) as mock_vm_exec,
    ):
        result = await env.exec(["cat"], input="hello")

    mock_upload.assert_called_once()
    stdin_vm_file = mock_upload.call_args[0][1]
    assert mock_upload.call_args[0][2] == b"hello"
    assert mock_cp.call_args[0][1][0] == "cp"
    assert result.stdout == "stdin data"
    # The captured exec, then removal of the VM's stdin temp file.
    commands = [c[0][1] for c in mock_vm_exec.call_args_list]
    assert len(commands) == 2
    assert TAG_RE.search(commands[0])
    assert commands[1] == build_remove_command([stdin_vm_file])


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
async def test_exec_captures_both_streams_on_the_vm() -> None:
    """The compose exec runs on the VM inside the capture wrapper, one round trip."""
    project = make_mock_project()
    env = DaytonaDinDServiceEnvironment(project, "web", "/app")

    with patch(
        "inspect_sandboxes.daytona._dind_env.vm_exec",
        scripted_vm_exec((0, "out\n", "err\n")),
    ) as mock_vm_exec:
        result = await env.exec(["sh", "-c", "echo out; echo err >&2"])

    assert result.success
    assert result.stdout == "out\n"
    assert result.stderr == "err\n"

    mock_vm_exec.assert_called_once()
    assert mock_vm_exec.call_args[0][0] is project.sandbox
    command = mock_vm_exec.call_args[0][1]
    inner = f"{EXPECTED_COMPOSE} sh -c 'echo out; echo err >&2'"
    assert command == build_capture_command(
        inner, ExecCapture.from_tag(tag_of(command))
    )
    # No timeout: no in-container timeout and no server deadline.
    assert mock_vm_exec.call_args[1]["timeout"] is None


@pytest.mark.asyncio
async def test_exec_timeout_runs_under_in_container_timeout() -> None:
    """The container command runs under /usr/bin/timeout; the server gets 10 s slack."""
    env = make_env()

    with (
        patch(
            "inspect_sandboxes.daytona._dind_env.compose_command",
            wraps=compose_command,
        ) as mock_cmd,
        patch(
            "inspect_sandboxes.daytona._dind_env.vm_exec",
            scripted_vm_exec((0, "out", "")),
        ) as mock_vm_exec,
    ):
        result = await env.exec(["sleep", "1"], timeout=7)

    assert result.stdout == "out"
    cmd = mock_cmd.call_args[0][1]
    assert cmd[-6:] == ["/usr/bin/timeout", "-k", "5s", "7s", "sleep", "1"]
    assert mock_vm_exec.call_args[1]["timeout"] == 17


@pytest.mark.asyncio
async def test_exec_timeout_with_stdin_keeps_the_cleanup_outside_the_timeout() -> None:
    env = make_env()

    with (
        patch("inspect_sandboxes.daytona._dind_env.sdk_upload", new_callable=AsyncMock),
        patch(
            "inspect_sandboxes.daytona._dind_env.compose_exec",
            new_callable=AsyncMock,
            return_value=(0, ""),
        ) as mock_cp,
        patch(
            "inspect_sandboxes.daytona._dind_env.compose_command",
            wraps=compose_command,
        ) as mock_cmd,
        patch(
            "inspect_sandboxes.daytona._dind_env.vm_exec",
            scripted_vm_exec((0, "", ""), (0, "", "")),
        ),
    ):
        await env.exec(["cat"], input="hi", timeout=3)

    container_file = mock_cp.call_args[0][1][2].split(":", 1)[1]
    cmd = mock_cmd.call_args[0][1]
    assert cmd[-3:-1] == ["sh", "-c"]
    assert cmd[-1].startswith(
        f"/usr/bin/timeout -k 5s 3s cat < {container_file}; _ec=$?; "
    )


@pytest.mark.parametrize("returncode", [124, 137, 143])
@pytest.mark.asyncio
async def test_exec_in_container_timeout_raises(returncode: int) -> None:
    env = make_env()

    with (
        patch(
            "inspect_sandboxes.daytona._dind_env.vm_exec",
            scripted_vm_exec((returncode, "early\n", "")),
        ),
        patch("inspect_sandboxes.daytona._dind_env._timed_out", return_value=True),
    ):
        with pytest.raises(TimeoutError, match="timed out after 2 seconds") as e:
            await env.exec(["sh", "-c", "echo early; sleep 60"], timeout=2)

    assert getattr(e.value, "truncated_output", None) == "early\n"


@pytest.mark.parametrize(
    ("returncode", "elapsed", "expected"),
    [
        (124, 0.1, True),  # GNU timeout: unambiguous
        (137, 2.5, True),  # SIGKILL after -k, once the deadline passed
        (143, 2.0, True),  # BusyBox timeout (SIGTERM)
        (137, 0.5, False),  # too fast: an OOM kill, not the timeout
        (143, 0.5, False),  # too fast: some other SIGTERM
        (1, 5.0, False),
        (0, 5.0, False),
    ],
)
def test_timed_out_classifies_exit_statuses(
    returncode: int, elapsed: float, expected: bool
) -> None:
    assert _timed_out(returncode, elapsed, 2) is expected


@pytest.mark.asyncio
async def test_exec_fast_signal_exit_is_returned() -> None:
    """A 137 well before the deadline (an OOM kill) is the command's result."""
    env = make_env()

    with patch(
        "inspect_sandboxes.daytona._dind_env.vm_exec",
        scripted_vm_exec((137, "", "Killed\n")),
    ):
        result = await env.exec(["big"], timeout=60)

    assert (result.returncode, result.stderr) == (137, "Killed\n")


@pytest.mark.asyncio
async def test_exec_stdin_cleanup_failure_keeps_the_result() -> None:
    """A failing VM-side rm of the stdin file does not replace the exec's result."""
    env = make_env()

    with (
        patch("inspect_sandboxes.daytona._dind_env.sdk_upload", new_callable=AsyncMock),
        patch(
            "inspect_sandboxes.daytona._dind_env.compose_exec",
            new_callable=AsyncMock,
            return_value=(0, ""),
        ),
        patch(
            "inspect_sandboxes.daytona._dind_env.vm_exec",
            scripted_vm_exec((0, "ok", ""), DaytonaError("VM gone")),
        ),
    ):
        result = await env.exec(["cat"], input="hi")

    assert result.stdout == "ok"


def make_local_shell_sandbox() -> MagicMock:
    """A DinD VM sandbox whose ``process.exec`` runs commands in the local shell.

    The ``docker compose ... exec -T -w /app web`` prefix is dropped so the
    service command runs directly; everything else (the ``sh -c`` VM wrapper,
    the capture wrapper, ``vm_exec``'s retry) is the real code path.
    """
    sandbox = make_mock_sandbox()

    async def run(command: str, **kwargs: Any) -> MagicMock:
        proc = await asyncio.create_subprocess_shell(
            command.replace(shlex.quote(EXPECTED_COMPOSE + " "), "").replace(
                EXPECTED_COMPOSE + " ", ""
            ),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        output, _ = await proc.communicate()
        return MagicMock(exit_code=proc.returncode, result=output.decode().strip())

    sandbox.process.exec = AsyncMock(side_effect=run)
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
    command = sandbox.process.exec.call_args[0][0]
    for file in ExecCapture.from_tag(tag_of(command)).files:
        assert not Path(file).exists(), f"{file} left behind"


@pytest.mark.skipif(
    not Path("/usr/bin/timeout").exists(), reason="needs /usr/bin/timeout"
)
@pytest.mark.asyncio
async def test_exec_timeout_kills_the_command_through_the_vm_shell() -> None:
    """The in-container timeout stops the command and its children, and raises."""
    sandbox = make_local_shell_sandbox()
    env = DaytonaDinDServiceEnvironment(make_mock_project(sandbox), "web", "/app")

    with pytest.raises(TimeoutError) as e:
        await env.exec(
            ["sh", "-c", "echo early; (sleep 30; echo late) & sleep 30"], timeout=1
        )

    assert getattr(e.value, "truncated_output", None) == "early\n"


@pytest.mark.asyncio
async def test_exec_reruns_command_when_the_vm_response_is_lost() -> None:
    """A DaytonaError after the VM ran the command retries it via vm_exec's retry."""
    sandbox = make_local_shell_sandbox()
    real_run = sandbox.process.exec.side_effect
    calls = 0

    async def lossy(command: str, **kwargs: Any) -> MagicMock:
        nonlocal calls
        calls += 1
        response = await real_run(command, **kwargs)
        if calls == 1:
            raise DaytonaError("response lost after the command ran")
        return response

    sandbox.process.exec = AsyncMock(side_effect=lossy)
    env = DaytonaDinDServiceEnvironment(make_mock_project(sandbox), "web", "/app")

    result = await env.exec(["sh", "-c", "echo out; echo err >&2"])

    assert (result.stdout, result.stderr) == ("out\n", "err\n")
    assert calls == 2


@pytest.mark.asyncio
async def test_exec_waits_for_background_writers_on_the_vm() -> None:
    sandbox = make_local_shell_sandbox()
    env = DaytonaDinDServiceEnvironment(make_mock_project(sandbox), "web", "/app")

    result = await env.exec(
        ["sh", "-c", "echo early; (sleep 0.3; echo late; echo late-err >&2) &"]
    )

    assert (result.stdout, result.stderr) == ("early\nlate\n", "late-err\n")


@pytest.mark.asyncio
async def test_exec_vm_collection_failure_raises() -> None:
    env = make_env()

    async def run(
        sandbox: Any, command: str, timeout: int | None = 60
    ) -> tuple[int, str]:
        tag = tag_of(command)
        return (
            0,
            f"<<inspect-exec-{tag}:stdout>><<inspect-exec-{tag}:failed>>sh: base64: not found",
        )

    with patch("inspect_sandboxes.daytona._dind_env.vm_exec", run):
        with pytest.raises(OutputCollectionError, match="base64"):
            await env.exec(["true"])


@pytest.mark.asyncio
async def test_exec_unframed_vm_output_is_a_failed_exec() -> None:
    env = make_env()

    with patch(
        "inspect_sandboxes.daytona._dind_env.vm_exec",
        new_callable=AsyncMock,
        return_value=(1, "sh: can't create /tmp/.inspect-exec-x.out: No space left"),
    ):
        result = await env.exec(["true"])

    assert not result.success
    assert result.stdout == ""
    assert "No space left" in result.stderr


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
