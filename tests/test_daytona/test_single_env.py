"""Tests for DaytonaSingleServiceEnvironment."""

import asyncio
import os
import re
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from daytona_sdk import DaytonaError, DaytonaNotFoundError, DaytonaTimeoutError
from inspect_ai.util import (
    ExecResult,
    OutputLimitExceededError,
    SandboxEnvironment,
    SandboxEnvironmentLimits,
)
from inspect_sandboxes.daytona._daytona import _daytona_client, _init_context
from inspect_sandboxes.daytona._sandbox_utils import (
    STDERR_SENTINEL,
    build_stderr_capture_command,
    build_stderr_readback_command,
)
from inspect_sandboxes.daytona._single_env import DaytonaSingleServiceEnvironment

STDERR_FILE_RE = re.compile(r"/tmp/\.inspect-stderr-[0-9a-f]{32}")


def exec_response(exit_code: int = 0, result: str = "output") -> MagicMock:
    """A fake ``process.exec`` response."""
    response = MagicMock()
    response.exit_code = exit_code
    response.result = result
    return response


def readback_response(stderr: str = "") -> MagicMock:
    """The response of the stderr readback exec that follows every command."""
    return exec_response(0, f"{STDERR_SENTINEL}{stderr}{STDERR_SENTINEL}")


def make_mock_sandbox(sandbox_id: str = "sb-test-123") -> MagicMock:
    """Create a mock AsyncSandbox.

    ``process.exec`` answers a command exec and then the stderr readback that
    ``exec()`` issues afterwards; tests that make more calls script their own.
    """
    sandbox = MagicMock()
    sandbox.id = sandbox_id

    sandbox.process = MagicMock()
    sandbox.process.exec = AsyncMock(side_effect=[exec_response(), readback_response()])

    sandbox.fs = MagicMock()
    sandbox.fs.upload_file = AsyncMock()
    sandbox.fs.download_file = AsyncMock(return_value=b"content")
    sandbox.fs.get_file_info = AsyncMock()
    sandbox.fs.create_folder = AsyncMock()

    return sandbox


def make_local_shell_sandbox() -> MagicMock:
    """A fake AsyncSandbox whose ``process.exec`` runs commands in the local ``sh``.

    Behaves like the Daytona API: one merged output stream (stderr folded
    into stdout) with the trailing newline stripped (emulated here as a full
    trim, the harsher case), plus the exit code.
    ``fs.upload_file`` writes to the local path so stdin temp files work.
    """
    sandbox = make_mock_sandbox()

    async def run(
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout: int | None = None,
    ) -> MagicMock:
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=cwd,
            env={**os.environ, **env} if env else None,
        )
        output, _ = await proc.communicate()
        assert proc.returncode is not None
        return exec_response(proc.returncode, output.decode().strip())

    async def upload(data: bytes, path: str) -> None:
        Path(path).write_bytes(data)

    sandbox.process.exec = AsyncMock(side_effect=run)
    sandbox.fs.upload_file = AsyncMock(side_effect=upload)
    return sandbox


def exec_commands(sandbox: MagicMock) -> list[str]:
    """The command strings passed to ``process.exec``, in order."""
    return [call[0][0] for call in sandbox.process.exec.call_args_list]


def stderr_file_of(command: str) -> str:
    match = STDERR_FILE_RE.search(command)
    assert match is not None, f"no stderr temp file in {command!r}"
    return match.group(0)


@pytest.fixture
def mock_sandbox() -> MagicMock:
    return make_mock_sandbox()


@pytest.mark.parametrize(
    ("cmd", "returncode", "expected_stdout", "expected_stderr"),
    [
        (["echo", "hello"], 0, "output", ""),
        (["false"], 1, "output", "failed\n"),
        (["ls", "-la"], 0, "output", "warning\n"),
    ],
)
@pytest.mark.asyncio
async def test_exec_basic(
    cmd: list[str],
    returncode: int,
    expected_stdout: str,
    expected_stderr: str,
    mock_sandbox: MagicMock,
) -> None:
    """Test exec with various command combinations."""
    mock_sandbox.process.exec.side_effect = [
        exec_response(returncode, expected_stdout),
        readback_response(expected_stderr),
    ]

    env = DaytonaSingleServiceEnvironment(mock_sandbox)
    result = await env.exec(cmd)

    assert isinstance(result, ExecResult)
    assert result.success == (returncode == 0)
    assert result.returncode == returncode
    assert result.stdout == expected_stdout
    assert result.stderr == expected_stderr


@pytest.mark.asyncio
async def test_exec_joins_args_with_shlex(mock_sandbox: MagicMock) -> None:
    """Test that exec correctly joins args into a shell command string."""
    env = DaytonaSingleServiceEnvironment(mock_sandbox)
    await env.exec(["echo", "hello world"])

    command_arg = exec_commands(mock_sandbox)[0]
    assert "{ echo 'hello world'; }" in command_arg


@pytest.mark.asyncio
async def test_exec_passes_cwd_and_env(mock_sandbox: MagicMock) -> None:
    """Test that exec passes cwd and env to process.exec."""
    env = DaytonaSingleServiceEnvironment(mock_sandbox)
    await env.exec(["ls"], cwd="/workspace", env={"MY_VAR": "value"})

    call_kwargs = mock_sandbox.process.exec.call_args_list[0][1]
    assert call_kwargs["cwd"] == "/workspace"
    assert call_kwargs["env"] == {"MY_VAR": "value"}


@pytest.mark.asyncio
async def test_exec_with_user_wraps_with_su(mock_sandbox: MagicMock) -> None:
    """Test that exec wraps command with su when user is specified."""
    env = DaytonaSingleServiceEnvironment(mock_sandbox)
    await env.exec(["whoami"], user="testuser")

    command = exec_commands(mock_sandbox)[0]
    assert "sudo -u testuser bash -c" in command


@pytest.mark.asyncio
async def test_exec_with_numeric_user_resolves_via_getent(
    mock_sandbox: MagicMock,
) -> None:
    """Test that numeric UIDs are resolved via getent."""
    env = DaytonaSingleServiceEnvironment(mock_sandbox)
    await env.exec(["whoami"], user="1000")

    command = exec_commands(mock_sandbox)[0]
    assert "sudo -u '#1000'" in command


@pytest.mark.asyncio
async def test_exec_separates_stderr_from_stdout() -> None:
    """Stderr comes back on its own, byte for byte, through the merged Daytona output.

    Runs the generated commands in the local shell with the API's behaviour
    emulated (one merged stream, trimmed), so this covers the capture wrapper,
    the readback and the sentinel framing end to end.
    """
    sandbox = make_local_shell_sandbox()
    env = DaytonaSingleServiceEnvironment(sandbox)

    result = await env.exec(["sh", "-c", "echo out; echo err >&2"])

    # stdout's trailing newline is lost to the API's trimming (a known
    # limitation); stderr is read back intact.
    assert result.stdout == "out"
    assert result.stderr == "err\n"
    assert result.success

    commands = exec_commands(sandbox)
    assert len(commands) == 2
    stderr_file = stderr_file_of(commands[0])
    assert commands[1] == build_stderr_readback_command(stderr_file)
    assert not Path(stderr_file).exists(), "readback must remove the temp file"


@pytest.mark.asyncio
async def test_exec_stderr_keeps_exit_code_and_empty_stdout() -> None:
    sandbox = make_local_shell_sandbox()
    env = DaytonaSingleServiceEnvironment(sandbox)

    result = await env.exec(["sh", "-c", "echo bad >&2; exit 7"])

    assert result.returncode == 7
    assert not result.success
    assert result.stdout == ""
    assert result.stderr == "bad\n"


@pytest.mark.asyncio
async def test_exec_stderr_with_stdin() -> None:
    """The stdin redirection and the stderr capture compose."""
    sandbox = make_local_shell_sandbox()
    env = DaytonaSingleServiceEnvironment(sandbox)

    result = await env.exec(["sh", "-c", "cat; echo e >&2"], input="hi")

    assert result.stdout == "hi"
    assert result.stderr == "e\n"
    stdin_file = sandbox.fs.upload_file.call_args[0][1]
    assert not Path(stdin_file).exists(), "the command's own rm removes stdin"
    assert not Path(stderr_file_of(exec_commands(sandbox)[0])).exists()


@pytest.mark.asyncio
async def test_exec_stderr_readback_failure_raises(mock_sandbox: MagicMock) -> None:
    """A failed readback is an error, not silently empty stderr."""
    mock_sandbox.process.exec.side_effect = [
        exec_response(0, "out"),
        exec_response(1, "cat: /tmp/.inspect-stderr-x: No such file or directory"),
        exec_response(0, ""),
    ]
    env = DaytonaSingleServiceEnvironment(mock_sandbox)

    with pytest.raises(RuntimeError, match="Failed to read the command's stderr"):
        await env.exec(["echo", "out"])

    # The failed readback is followed by a best-effort removal of the temp file.
    commands = exec_commands(mock_sandbox)
    assert len(commands) == 3
    assert commands[2] == f"rm -f {stderr_file_of(commands[0])}"


@pytest.mark.asyncio
async def test_exec_with_stdin_string(mock_sandbox: MagicMock) -> None:
    """Test exec redirects string stdin through a temp file."""
    env = DaytonaSingleServiceEnvironment(mock_sandbox)
    await env.exec(["cat"], input="hello")

    mock_sandbox.fs.upload_file.assert_called_once()
    call_args = mock_sandbox.fs.upload_file.call_args
    assert call_args[0][0] == b"hello"
    stdin_path = call_args[0][1]
    assert stdin_path.startswith("/tmp/.inspect-stdin-")

    exec_command = exec_commands(mock_sandbox)[0]
    assert f"< {stdin_path}" in exec_command
    assert f"rm -f {stdin_path}" in exec_command


@pytest.mark.asyncio
async def test_exec_with_stdin_bytes(mock_sandbox: MagicMock) -> None:
    """Test exec redirects bytes stdin through a temp file."""
    env = DaytonaSingleServiceEnvironment(mock_sandbox)
    await env.exec(["wc", "-c"], input=b"\x00\x01\x02")

    call_args = mock_sandbox.fs.upload_file.call_args
    assert call_args[0][0] == b"\x00\x01\x02"


@pytest.mark.asyncio
async def test_exec_without_stdin_no_upload(mock_sandbox: MagicMock) -> None:
    """Test exec without stdin does not upload any file."""
    env = DaytonaSingleServiceEnvironment(mock_sandbox)
    await env.exec(["echo", "hi"])

    mock_sandbox.fs.upload_file.assert_not_called()
    command = exec_commands(mock_sandbox)[0]
    assert command == build_stderr_capture_command("echo hi", stderr_file_of(command))


@pytest.mark.asyncio
async def test_exec_with_stdin_and_user_skips_inline_cleanup(
    mock_sandbox: MagicMock,
) -> None:
    """Test that stdin + no baked-in rm (cleanup done in finally as root)."""
    env = DaytonaSingleServiceEnvironment(mock_sandbox)
    await env.exec(["cat"], input="hello", user="testuser")

    commands = exec_commands(mock_sandbox)
    # First call: the sudo-wrapped command (no baked-in rm -f)
    exec_command = commands[0]
    assert "sudo -u testuser" in exec_command
    assert "rm -f" not in exec_command
    # Second call: the stderr readback removes the stdin file too, as root
    assert len(commands) == 2
    stdin_file = mock_sandbox.fs.upload_file.call_args[0][1]
    assert commands[1] == build_stderr_readback_command(
        stderr_file_of(exec_command), [stdin_file]
    )


@pytest.mark.asyncio
async def test_exec_retries_transient_error(mock_sandbox: MagicMock) -> None:
    """Test that exec retries on transient DaytonaError."""
    call_count = 0
    responses: list[Any] = [
        DaytonaError("transient API failure"),
        exec_response(0, "ok"),
        readback_response(),
    ]

    async def flaky_exec(*args: Any, **kwargs: Any) -> MagicMock:
        nonlocal call_count
        call_count += 1
        response = responses[call_count - 1]
        if isinstance(response, Exception):
            raise response
        return response

    mock_sandbox.process.exec = AsyncMock(side_effect=flaky_exec)
    env = DaytonaSingleServiceEnvironment(mock_sandbox)
    result = await env.exec(["echo", "test"])

    assert result.success
    assert result.stdout == "ok"
    assert call_count == 3  # failed attempt, retry, stderr readback


@pytest.mark.asyncio
async def test_exec_does_not_retry_timeout(mock_sandbox: MagicMock) -> None:
    """Test that DaytonaTimeoutError propagates to the timeout retry loop."""
    mock_sandbox.process.exec = AsyncMock(side_effect=DaytonaTimeoutError("timed out"))
    env = DaytonaSingleServiceEnvironment(mock_sandbox)

    with pytest.raises(TimeoutError):
        await env.exec(["sleep", "100"], timeout=5)

    # Outer timeout loop makes 3 attempts (original, 5s cap, 5s cap), then
    # the stderr temp file is removed best-effort (that exec fails too).
    commands = exec_commands(mock_sandbox)
    assert len(commands) == 4
    assert commands[3] == f"rm -f {stderr_file_of(commands[0])}"


@pytest.mark.asyncio
async def test_exec_does_not_retry_non_daytona_error(mock_sandbox: MagicMock) -> None:
    """Test that non-DaytonaError exceptions are not retried."""
    mock_sandbox.process.exec = AsyncMock(side_effect=RuntimeError("unexpected"))
    env = DaytonaSingleServiceEnvironment(mock_sandbox)

    with pytest.raises(RuntimeError, match="unexpected"):
        await env.exec(["echo", "test"])

    # One attempt plus the best-effort temp file removal.
    assert mock_sandbox.process.exec.call_count == 2


@pytest.mark.asyncio
async def test_write_file_text(mock_sandbox: MagicMock) -> None:
    """Test write_file with text content."""
    env = DaytonaSingleServiceEnvironment(mock_sandbox)
    file_info = MagicMock()
    file_info.is_dir = False
    mock_sandbox.fs.get_file_info = AsyncMock(return_value=file_info)

    await env.write_file("/workspace/test.txt", "hello")

    mock_sandbox.fs.upload_file.assert_called_once_with(b"hello", "/workspace/test.txt")


@pytest.mark.asyncio
async def test_write_file_binary(mock_sandbox: MagicMock) -> None:
    """Test write_file with binary content."""
    env = DaytonaSingleServiceEnvironment(mock_sandbox)
    file_info = MagicMock()
    file_info.is_dir = False
    mock_sandbox.fs.get_file_info = AsyncMock(return_value=file_info)

    await env.write_file("/workspace/data.bin", b"\x00\x01\x02")

    mock_sandbox.fs.upload_file.assert_called_once_with(
        b"\x00\x01\x02", "/workspace/data.bin"
    )


@pytest.mark.asyncio
async def test_write_file_creates_parent_dirs(mock_sandbox: MagicMock) -> None:
    """Test write_file calls create_folder for parent directory."""
    env = DaytonaSingleServiceEnvironment(mock_sandbox)
    file_info = MagicMock()
    file_info.is_dir = False
    mock_sandbox.fs.get_file_info = AsyncMock(return_value=file_info)

    await env.write_file("/deep/nested/dir/file.txt", "content")

    mock_sandbox.fs.create_folder.assert_called_once_with("/deep/nested/dir", "755")


@pytest.mark.asyncio
async def test_write_file_raises_for_directory(mock_sandbox: MagicMock) -> None:
    """Test write_file raises IsADirectoryError when path is a directory."""
    env = DaytonaSingleServiceEnvironment(mock_sandbox)
    dir_info = MagicMock()
    dir_info.is_dir = True
    mock_sandbox.fs.get_file_info = AsyncMock(return_value=dir_info)

    with pytest.raises(IsADirectoryError):
        await env.write_file("/existing/dir", "content")


@pytest.mark.asyncio
async def test_read_file_text(mock_sandbox: MagicMock) -> None:
    """Test read_file in text mode."""
    env = DaytonaSingleServiceEnvironment(mock_sandbox)
    file_info = MagicMock()
    file_info.is_dir = False
    file_info.size = 12
    mock_sandbox.fs.get_file_info = AsyncMock(return_value=file_info)
    mock_sandbox.fs.download_file = AsyncMock(return_value=b"hello world\n")

    result = await env.read_file("/test.txt", text=True)
    assert result == "hello world\n"


@pytest.mark.asyncio
async def test_read_file_binary(mock_sandbox: MagicMock) -> None:
    """Test read_file in binary mode."""
    env = DaytonaSingleServiceEnvironment(mock_sandbox)
    file_info = MagicMock()
    file_info.is_dir = False
    file_info.size = 4
    mock_sandbox.fs.get_file_info = AsyncMock(return_value=file_info)
    mock_sandbox.fs.download_file = AsyncMock(return_value=b"\x00\x01\x02\x03")

    result = await env.read_file("/test.bin", text=False)
    assert result == b"\x00\x01\x02\x03"


@pytest.mark.asyncio
async def test_read_file_not_found(mock_sandbox: MagicMock) -> None:
    """Test read_file raises FileNotFoundError when file doesn't exist."""
    env = DaytonaSingleServiceEnvironment(mock_sandbox)
    file_info = MagicMock()
    file_info.is_dir = False
    file_info.size = 10
    mock_sandbox.fs.get_file_info = AsyncMock(return_value=file_info)
    mock_sandbox.fs.download_file = AsyncMock(
        side_effect=DaytonaNotFoundError("not found")
    )

    with pytest.raises(FileNotFoundError):
        await env.read_file("/missing.txt")


@pytest.mark.asyncio
async def test_read_file_is_directory(mock_sandbox: MagicMock) -> None:
    """Test read_file raises IsADirectoryError for directories."""
    env = DaytonaSingleServiceEnvironment(mock_sandbox)
    dir_info = MagicMock()
    dir_info.is_dir = True
    dir_info.size = 0
    mock_sandbox.fs.get_file_info = AsyncMock(return_value=dir_info)

    with pytest.raises(IsADirectoryError):
        await env.read_file("/some/dir")


@pytest.mark.asyncio
async def test_read_file_size_limit(mock_sandbox: MagicMock) -> None:
    """Test read_file raises OutputLimitExceededError for oversized files."""
    env = DaytonaSingleServiceEnvironment(mock_sandbox)
    file_info = MagicMock()
    file_info.is_dir = False
    file_info.size = SandboxEnvironmentLimits.MAX_READ_FILE_SIZE + 1
    mock_sandbox.fs.get_file_info = AsyncMock(return_value=file_info)

    with pytest.raises(OutputLimitExceededError):
        await env.read_file("/huge.bin")


@pytest.mark.asyncio
async def test_sample_cleanup_deletes_sandboxes() -> None:
    """Test sample_cleanup deletes all sandboxes in environments dict."""
    _init_context()
    sb1 = make_mock_sandbox("sb-1")
    sb2 = make_mock_sandbox("sb-2")
    mock_client = MagicMock()
    mock_client.delete = AsyncMock()
    _daytona_client.set(mock_client)

    envs: dict[str, SandboxEnvironment] = {
        "default": DaytonaSingleServiceEnvironment(sb1),
        "other": DaytonaSingleServiceEnvironment(sb2),
    }
    await DaytonaSingleServiceEnvironment.sample_cleanup("task", None, envs, False)

    assert mock_client.delete.call_count == 2


@pytest.mark.asyncio
async def test_sample_cleanup_skips_when_interrupted() -> None:
    """Test sample_cleanup does nothing when interrupted."""
    _init_context()
    mock_client = MagicMock()
    mock_client.delete = AsyncMock()
    _daytona_client.set(mock_client)

    sb = make_mock_sandbox()
    envs: dict[str, SandboxEnvironment] = {
        "default": DaytonaSingleServiceEnvironment(sb)
    }
    await DaytonaSingleServiceEnvironment.sample_cleanup("task", None, envs, True)

    mock_client.delete.assert_not_called()


@pytest.mark.asyncio
async def test_sample_cleanup_skips_when_no_client() -> None:
    """Test sample_cleanup no-ops when client is None."""
    _init_context()
    _daytona_client.set(None)

    sb = make_mock_sandbox()
    envs: dict[str, SandboxEnvironment] = {
        "default": DaytonaSingleServiceEnvironment(sb)
    }
    # Should not raise
    await DaytonaSingleServiceEnvironment.sample_cleanup("task", None, envs, False)


@pytest.mark.asyncio
async def test_sample_cleanup_continues_on_delete_failure() -> None:
    """Test sample_cleanup logs error and continues when a delete fails."""
    _init_context()
    sb1 = make_mock_sandbox("sb-fail")
    sb2 = make_mock_sandbox("sb-ok")
    mock_client = MagicMock()
    mock_client.delete = AsyncMock(side_effect=[Exception("fail"), None])
    _daytona_client.set(mock_client)

    envs: dict[str, SandboxEnvironment] = {
        "a": DaytonaSingleServiceEnvironment(sb1),
        "b": DaytonaSingleServiceEnvironment(sb2),
    }
    # Should not raise — logs the error and continues
    await DaytonaSingleServiceEnvironment.sample_cleanup("task", None, envs, False)

    assert mock_client.delete.call_count == 2


@pytest.mark.asyncio
async def test_connection_surfaces_preview_links_as_ports(
    mock_sandbox: MagicMock,
) -> None:
    """connection() splits each preview URL into a bare host and its port."""
    preview = MagicMock()
    preview.url = "https://3000-sb-test-123.proxy.daytona.work"
    mock_sandbox.get_preview_link = AsyncMock(return_value=preview)

    env = DaytonaSingleServiceEnvironment(mock_sandbox, connection_ports=[3000])
    conn = await env.connection()

    assert conn.type == "daytona"
    assert conn.ports is not None
    assert len(conn.ports) == 1
    assert conn.ports[0].container_port == 3000
    mapping = conn.ports[0].mappings[0]
    # host_ip is a bare host (no scheme); the URL implies HTTPS, so port 443.
    assert mapping.host_ip == "3000-sb-test-123.proxy.daytona.work"
    assert mapping.host_port == 443
    mock_sandbox.get_preview_link.assert_awaited_once_with(3000)


@pytest.mark.asyncio
async def test_connection_preview_link_honors_explicit_port(
    mock_sandbox: MagicMock,
) -> None:
    """An explicit port in the preview URL is preserved, not clobbered by 443."""
    preview = MagicMock()
    preview.url = "https://3000-sb-test-123.proxy.daytona.work:8443"
    mock_sandbox.get_preview_link = AsyncMock(return_value=preview)

    env = DaytonaSingleServiceEnvironment(mock_sandbox, connection_ports=[3000])
    conn = await env.connection()

    assert conn.ports is not None
    mapping = conn.ports[0].mappings[0]
    assert mapping.host_ip == "3000-sb-test-123.proxy.daytona.work"
    assert mapping.host_port == 8443


@pytest.mark.asyncio
async def test_connection_without_ports_is_empty(mock_sandbox: MagicMock) -> None:
    """No declared ports yields a connection with ports=None."""
    env = DaytonaSingleServiceEnvironment(mock_sandbox)
    conn = await env.connection()

    assert conn.type == "daytona"
    assert conn.ports is None
    assert conn.container == "sb-test-123"
