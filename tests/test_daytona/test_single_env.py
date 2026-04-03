"""Tests for DaytonaSingleServiceEnvironment."""

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
from inspect_sandboxes.daytona._single_env import DaytonaSingleServiceEnvironment


def make_mock_sandbox(sandbox_id: str = "sb-test-123") -> MagicMock:
    """Create a mock AsyncSandbox."""
    sandbox = MagicMock()
    sandbox.id = sandbox_id

    execute_response = MagicMock()
    execute_response.exit_code = 0
    execute_response.result = "output"
    sandbox.process = MagicMock()
    sandbox.process.exec = AsyncMock(return_value=execute_response)

    sandbox.fs = MagicMock()
    sandbox.fs.upload_file = AsyncMock()
    sandbox.fs.download_file = AsyncMock(return_value=b"content")
    sandbox.fs.get_file_info = AsyncMock()
    sandbox.fs.create_folder = AsyncMock()

    return sandbox


@pytest.fixture
def mock_sandbox() -> MagicMock:
    return make_mock_sandbox()


@pytest.mark.parametrize(
    ("cmd", "returncode", "expected_stdout"),
    [
        (["echo", "hello"], 0, "output"),
        (["false"], 1, "output"),
        (["ls", "-la"], 0, "output"),
    ],
)
@pytest.mark.asyncio
async def test_exec_basic(
    cmd: list[str],
    returncode: int,
    expected_stdout: str,
    mock_sandbox: MagicMock,
) -> None:
    """Test exec with various command combinations."""
    mock_sandbox.process.exec.return_value.exit_code = returncode
    mock_sandbox.process.exec.return_value.result = expected_stdout

    env = DaytonaSingleServiceEnvironment(mock_sandbox)
    result = await env.exec(cmd)

    assert isinstance(result, ExecResult)
    assert result.success == (returncode == 0)
    assert result.returncode == returncode
    assert result.stdout == expected_stdout
    assert result.stderr == ""


@pytest.mark.asyncio
async def test_exec_joins_args_with_shlex(mock_sandbox: MagicMock) -> None:
    """Test that exec correctly joins args into a shell command string."""
    env = DaytonaSingleServiceEnvironment(mock_sandbox)
    await env.exec(["echo", "hello world"])

    command_arg = mock_sandbox.process.exec.call_args[0][0]
    assert command_arg == "echo 'hello world'"


@pytest.mark.asyncio
async def test_exec_passes_cwd_and_env(mock_sandbox: MagicMock) -> None:
    """Test that exec passes cwd and env to process.exec."""
    env = DaytonaSingleServiceEnvironment(mock_sandbox)
    await env.exec(["ls"], cwd="/workspace", env={"MY_VAR": "value"})

    call_kwargs = mock_sandbox.process.exec.call_args[1]
    assert call_kwargs["cwd"] == "/workspace"
    assert call_kwargs["env"] == {
        "MY_VAR": '"value"'
    }  # double-quoted for Daytona SDK bug


@pytest.mark.asyncio
async def test_exec_with_user_wraps_with_su(mock_sandbox: MagicMock) -> None:
    """Test that exec wraps command with su when user is specified."""
    env = DaytonaSingleServiceEnvironment(mock_sandbox)
    await env.exec(["whoami"], user="testuser")

    command = mock_sandbox.process.exec.call_args[0][0]
    assert "sudo -u testuser bash -c" in command


@pytest.mark.asyncio
async def test_exec_with_numeric_user_resolves_via_getent(
    mock_sandbox: MagicMock,
) -> None:
    """Test that numeric UIDs are resolved via getent."""
    env = DaytonaSingleServiceEnvironment(mock_sandbox)
    await env.exec(["whoami"], user="1000")

    command = mock_sandbox.process.exec.call_args[0][0]
    assert "sudo -u '#1000'" in command


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

    exec_command = mock_sandbox.process.exec.call_args[0][0]
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
    command = mock_sandbox.process.exec.call_args[0][0]
    assert command == "echo hi"


@pytest.mark.asyncio
async def test_exec_with_stdin_and_user_skips_inline_cleanup(
    mock_sandbox: MagicMock,
) -> None:
    """Test that stdin + no baked-in rm (cleanup done in finally as root)."""
    env = DaytonaSingleServiceEnvironment(mock_sandbox)
    await env.exec(["cat"], input="hello", user="testuser")

    calls = mock_sandbox.process.exec.call_args_list
    # First call: the sudo-wrapped command (no baked-in rm -f)
    exec_command = calls[0][0][0]
    assert "sudo -u testuser" in exec_command
    assert "rm -f" not in exec_command
    # Second call: cleanup the temp file as root
    assert len(calls) == 2
    cleanup_command = calls[1][0][0]
    assert "rm -f" in cleanup_command


@pytest.mark.asyncio
async def test_exec_retries_transient_error(mock_sandbox: MagicMock) -> None:
    """Test that exec retries on transient DaytonaError."""
    call_count = 0
    success_response = MagicMock()
    success_response.exit_code = 0
    success_response.result = "ok"

    async def flaky_exec(*args: Any, **kwargs: Any) -> MagicMock:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise DaytonaError("transient API failure")
        return success_response

    mock_sandbox.process.exec = AsyncMock(side_effect=flaky_exec)
    env = DaytonaSingleServiceEnvironment(mock_sandbox)
    result = await env.exec(["echo", "test"])

    assert result.success
    assert call_count == 2


@pytest.mark.asyncio
async def test_exec_does_not_retry_timeout(mock_sandbox: MagicMock) -> None:
    """Test that DaytonaTimeoutError propagates to the timeout retry loop."""
    mock_sandbox.process.exec = AsyncMock(side_effect=DaytonaTimeoutError("timed out"))
    env = DaytonaSingleServiceEnvironment(mock_sandbox)

    with pytest.raises(TimeoutError):
        await env.exec(["sleep", "100"], timeout=5)

    # Outer timeout loop makes 3 attempts (original, 5s cap, 5s cap)
    assert mock_sandbox.process.exec.call_count == 3


@pytest.mark.asyncio
async def test_exec_does_not_retry_non_daytona_error(mock_sandbox: MagicMock) -> None:
    """Test that non-DaytonaError exceptions are not retried."""
    mock_sandbox.process.exec = AsyncMock(side_effect=RuntimeError("unexpected"))
    env = DaytonaSingleServiceEnvironment(mock_sandbox)

    with pytest.raises(RuntimeError, match="unexpected"):
        await env.exec(["echo", "test"])

    assert mock_sandbox.process.exec.call_count == 1


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
