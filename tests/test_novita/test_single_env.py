"""Tests for NovitaSingleServiceEnvironment."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpcore
import httpx
import pytest
from inspect_ai.util import OutputLimitExceededError, SandboxEnvironmentLimits
from inspect_sandboxes.novita._single_env import NovitaSingleServiceEnvironment
from novita_sandbox.core import (
    CommandExitException,
    FileNotFoundException,
    FileType,
    NotFoundException,
    SandboxException,
    TimeoutException,
)


def _make_command_result(
    *, stdout: str = "", stderr: str = "", exit_code: int = 0
) -> MagicMock:
    cr = MagicMock()
    cr.stdout = stdout
    cr.stderr = stderr
    cr.exit_code = exit_code
    return cr


def make_mock_sandbox(sandbox_id: str = "sb-test-123") -> MagicMock:
    """Create a mock AsyncSandbox."""
    sandbox = MagicMock()
    sandbox.sandbox_id = sandbox_id
    sandbox.commands = MagicMock()
    sandbox.commands.run = AsyncMock(return_value=_make_command_result(stdout="ok"))
    sandbox.files = MagicMock()
    sandbox.files.read = AsyncMock(return_value="content")
    sandbox.files.write = AsyncMock()
    sandbox.files.get_info = AsyncMock()
    sandbox.kill = AsyncMock()
    return sandbox


@pytest.fixture
def mock_sandbox() -> MagicMock:
    return make_mock_sandbox()


@pytest.mark.asyncio
async def test_exec_basic(mock_sandbox: MagicMock) -> None:
    """Test exec returns ExecResult for a successful command."""
    mock_sandbox.commands.run = AsyncMock(
        return_value=_make_command_result(stdout="hi\n", stderr="", exit_code=0)
    )
    env = NovitaSingleServiceEnvironment(mock_sandbox)
    result = await env.exec(["echo", "hi"])
    assert result.success is True
    assert result.returncode == 0
    assert result.stdout == "hi\n"
    assert result.stderr == ""


@pytest.mark.asyncio
async def test_exec_joins_args_with_shlex(mock_sandbox: MagicMock) -> None:
    """Test that exec correctly joins args into a shell command string."""
    env = NovitaSingleServiceEnvironment(mock_sandbox)
    await env.exec(["echo", "hello world"])

    command_arg = mock_sandbox.commands.run.call_args[0][0]
    assert command_arg == "echo 'hello world'"


@pytest.mark.asyncio
async def test_exec_passes_cwd_and_env(mock_sandbox: MagicMock) -> None:
    """Test that exec passes cwd and env to commands.run."""
    env = NovitaSingleServiceEnvironment(mock_sandbox)
    await env.exec(["ls"], cwd="/workspace", env={"MY_VAR": "value"})

    call_kwargs = mock_sandbox.commands.run.call_args[1]
    assert call_kwargs["cwd"] == "/workspace"
    assert call_kwargs["envs"] == {"MY_VAR": "value"}


@pytest.mark.asyncio
async def test_exec_defaults_user_to_root(mock_sandbox: MagicMock) -> None:
    """Novita's SDK defaults to user='user' (uid 1000); we override to root."""
    env = NovitaSingleServiceEnvironment(mock_sandbox)

    await env.exec(["whoami"])
    assert mock_sandbox.commands.run.call_args[1]["user"] == "root"

    mock_sandbox.commands.run.reset_mock()
    await env.exec(["whoami"], user="agent")
    assert mock_sandbox.commands.run.call_args[1]["user"] == "agent"


@pytest.mark.asyncio
async def test_file_ops_default_user_to_root(mock_sandbox: MagicMock) -> None:
    """files.* SDK calls must run as root by default for the same reason as exec."""
    env = NovitaSingleServiceEnvironment(mock_sandbox)
    file_info = MagicMock()
    file_info.type = FileType.FILE
    file_info.size = 5
    mock_sandbox.files.get_info = AsyncMock(return_value=file_info)
    mock_sandbox.files.read = AsyncMock(return_value="hello")

    await env.read_file("/etc/anywhere", text=True)
    assert mock_sandbox.files.get_info.call_args[1]["user"] == "root"
    assert mock_sandbox.files.read.call_args[1]["user"] == "root"

    mock_sandbox.files.get_info = AsyncMock(side_effect=NotFoundException("missing"))
    await env.write_file("/etc/anywhere/new.txt", "x")
    assert mock_sandbox.files.write.call_args[1]["user"] == "root"


@pytest.mark.asyncio
async def test_exec_failure_returncode(mock_sandbox: MagicMock) -> None:
    """Novita's commands.run RAISES on non-zero exit; we must surface as ExecResult."""
    mock_sandbox.commands.run = AsyncMock(
        side_effect=CommandExitException(
            stdout="", stderr="boom", exit_code=2, error=None
        )
    )
    env = NovitaSingleServiceEnvironment(mock_sandbox)
    result = await env.exec(["false"])
    assert result.success is False
    assert result.returncode == 2
    assert result.stderr == "boom"


@pytest.mark.asyncio
async def test_exec_with_stdin_string(mock_sandbox: MagicMock) -> None:
    """Test exec redirects string stdin through a temp file."""
    env = NovitaSingleServiceEnvironment(mock_sandbox)
    await env.exec(["cat"], input="hello")

    mock_sandbox.files.write.assert_called_once()
    call_args = mock_sandbox.files.write.call_args
    stdin_path = call_args[0][0]
    assert call_args[0][1] == b"hello"
    assert stdin_path.startswith("/tmp/.inspect-stdin-")

    exec_command = mock_sandbox.commands.run.call_args[0][0]
    assert f"< {stdin_path}" in exec_command
    assert f"rm -f {stdin_path}" in exec_command


@pytest.mark.asyncio
async def test_exec_with_stdin_bytes(mock_sandbox: MagicMock) -> None:
    """Test exec redirects bytes stdin through a temp file."""
    env = NovitaSingleServiceEnvironment(mock_sandbox)
    await env.exec(["wc", "-c"], input=b"\x00\x01\x02")

    call_args = mock_sandbox.files.write.call_args
    assert call_args[0][1] == b"\x00\x01\x02"


@pytest.mark.asyncio
async def test_exec_without_stdin_no_upload(mock_sandbox: MagicMock) -> None:
    """Test exec without stdin does not upload any file."""
    env = NovitaSingleServiceEnvironment(mock_sandbox)
    await env.exec(["echo", "hi"])

    mock_sandbox.files.write.assert_not_called()
    command = mock_sandbox.commands.run.call_args[0][0]
    assert command == "echo hi"


@pytest.mark.asyncio
async def test_exec_with_stdin_and_user_skips_inline_cleanup(
    mock_sandbox: MagicMock,
) -> None:
    """Test that stdin + user defers temp-file cleanup to the finally block."""
    env = NovitaSingleServiceEnvironment(mock_sandbox)
    await env.exec(["cat"], input="hello", user="testuser")

    calls = mock_sandbox.commands.run.call_args_list
    # First call: exec with user (no baked-in rm -f)
    exec_command = calls[0][0][0]
    assert "rm -f" not in exec_command
    # Second call: cleanup the temp file
    assert len(calls) == 2
    cleanup_command = calls[1][0][0]
    assert "rm -f" in cleanup_command


@pytest.mark.asyncio
async def test_exec_stdin_cleanup_failure_does_not_mask_result(
    mock_sandbox: MagicMock,
) -> None:
    """A leaked transport error during stdin temp cleanup must not replace the exec result."""
    mock_sandbox.commands.run = AsyncMock(
        side_effect=[
            _make_command_result(stdout="ok", exit_code=0),
            httpcore.ReadError("reset"),
        ]
    )
    env = NovitaSingleServiceEnvironment(mock_sandbox)

    result = await env.exec(["cat"], input="hello", user="testuser")

    assert result.success
    assert result.stdout == "ok"


@pytest.mark.asyncio
async def test_exec_retries_transient_error(mock_sandbox: MagicMock) -> None:
    """Test that exec retries on transient SandboxException."""
    call_count = 0
    success_result = _make_command_result(stdout="ok", exit_code=0)

    async def flaky_run(*args: Any, **kwargs: Any) -> MagicMock:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise SandboxException("transient API failure")
        return success_result

    mock_sandbox.commands.run = AsyncMock(side_effect=flaky_run)
    env = NovitaSingleServiceEnvironment(mock_sandbox)
    result = await env.exec(["echo", "test"])

    assert result.success
    assert call_count == 2


@pytest.mark.asyncio
async def test_exec_retries_transport_error(mock_sandbox: MagicMock) -> None:
    """A dropped connection (httpcore.ReadError) leaks from the SDK; must be retried."""
    mock_sandbox.commands.run = AsyncMock(
        side_effect=[
            httpcore.ReadError("reset"),
            _make_command_result(stdout="ok", exit_code=0),
        ]
    )
    env = NovitaSingleServiceEnvironment(mock_sandbox)
    result = await env.exec(["echo", "test"])

    assert result.success
    assert mock_sandbox.commands.run.await_count == 2


@pytest.mark.asyncio
async def test_exec_does_not_retry_non_novita_error(mock_sandbox: MagicMock) -> None:
    """Test that non-SandboxException, non-transport exceptions are not retried."""
    mock_sandbox.commands.run = AsyncMock(side_effect=RuntimeError("unexpected"))
    env = NovitaSingleServiceEnvironment(mock_sandbox)

    with pytest.raises(RuntimeError, match="unexpected"):
        await env.exec(["echo", "test"])

    assert mock_sandbox.commands.run.await_count == 1


@pytest.mark.asyncio
async def test_exec_timeout_retry_uses_capped_timeouts(
    mock_sandbox: MagicMock,
) -> None:
    """First retry caps at 60s, second at 30s, then raises TimeoutError."""
    mock_sandbox.commands.run = AsyncMock(side_effect=TimeoutException("timed out"))
    env = NovitaSingleServiceEnvironment(mock_sandbox)
    with pytest.raises(TimeoutError):
        await env.exec(["sleep", "1000"], timeout=120)
    run = mock_sandbox.commands.run
    assert run.await_count == 3
    timeouts = [call.kwargs["timeout"] for call in run.call_args_list]
    assert timeouts == [120, 60, 30]


@pytest.mark.asyncio
async def test_write_file_text(mock_sandbox: MagicMock) -> None:
    """Test write_file with text content."""
    env = NovitaSingleServiceEnvironment(mock_sandbox)
    mock_sandbox.files.get_info = AsyncMock(side_effect=NotFoundException("missing"))

    await env.write_file("/workspace/test.txt", "hello")

    mock_sandbox.files.write.assert_called_once_with(
        "/workspace/test.txt",
        b"hello",
        user="root",
        request_timeout=1800,
        gzip=True,
        use_octet_stream=True,
    )


@pytest.mark.asyncio
async def test_write_file_binary(mock_sandbox: MagicMock) -> None:
    """Test write_file with binary content."""
    env = NovitaSingleServiceEnvironment(mock_sandbox)
    mock_sandbox.files.get_info = AsyncMock(side_effect=NotFoundException("missing"))

    await env.write_file("/workspace/data.bin", b"\x00\x01\x02")

    mock_sandbox.files.write.assert_called_once_with(
        "/workspace/data.bin",
        b"\x00\x01\x02",
        user="root",
        request_timeout=1800,
        gzip=True,
        use_octet_stream=True,
    )


@pytest.mark.asyncio
async def test_write_file_large_uploads_in_chunks(mock_sandbox: MagicMock) -> None:
    """Writes above UPLOAD_CHUNK_SIZE go up as chunk files, then cat + cleanup."""
    env = NovitaSingleServiceEnvironment(mock_sandbox)
    mock_sandbox.files.get_info = AsyncMock(side_effect=NotFoundException("missing"))

    with patch("inspect_sandboxes.novita._single_env.UPLOAD_CHUNK_SIZE", 4):
        await env.write_file("/workspace/big.bin", b"0123456789")

    write_calls = mock_sandbox.files.write.call_args_list
    assert [c[0][1] for c in write_calls] == [b"0123", b"4567", b"89"]
    # Chunks reuse one temp file: each is appended to the target as it lands.
    chunk_paths = {c[0][0] for c in write_calls}
    assert len(chunk_paths) == 1
    temp = chunk_paths.pop()
    assert temp.startswith("/tmp/.inspect-chunk-")

    run_commands = [c[0][0] for c in mock_sandbox.commands.run.call_args_list]
    assert run_commands == [
        "mkdir -p /workspace && : > /workspace/big.bin",
        *[f"cat {temp} >> /workspace/big.bin"] * 3,
        f"rm -f {temp}",
    ]


@pytest.mark.asyncio
async def test_write_file_retries_transport_error(mock_sandbox: MagicMock) -> None:
    """A dropped connection (httpx.ReadError) leaks from the SDK; must be retried."""
    mock_sandbox.files.get_info = AsyncMock(side_effect=NotFoundException("missing"))
    mock_sandbox.files.write = AsyncMock(side_effect=[httpx.ReadError("reset"), None])
    env = NovitaSingleServiceEnvironment(mock_sandbox)

    await env.write_file("/workspace/test.txt", "hello")

    assert mock_sandbox.files.write.await_count == 2


@pytest.mark.asyncio
async def test_write_file_does_not_retry_timeout(mock_sandbox: MagicMock) -> None:
    """Timeout exceptions (httpx.ReadTimeout) are excluded from retry."""
    mock_sandbox.files.get_info = AsyncMock(side_effect=NotFoundException("missing"))
    mock_sandbox.files.write = AsyncMock(side_effect=httpx.ReadTimeout("timed out"))
    env = NovitaSingleServiceEnvironment(mock_sandbox)

    with pytest.raises(httpx.ReadTimeout):
        await env.write_file("/workspace/test.txt", "hello")

    assert mock_sandbox.files.write.await_count == 1


@pytest.mark.asyncio
async def test_write_file_raises_for_directory(mock_sandbox: MagicMock) -> None:
    """Test write_file raises IsADirectoryError when path is a directory."""
    env = NovitaSingleServiceEnvironment(mock_sandbox)
    dir_info = MagicMock()
    dir_info.type = FileType.DIR
    mock_sandbox.files.get_info = AsyncMock(return_value=dir_info)

    with pytest.raises(IsADirectoryError):
        await env.write_file("/existing/dir", "content")


@pytest.mark.asyncio
async def test_write_file_translates_isdir_sdk_error(mock_sandbox: MagicMock) -> None:
    mock_sandbox.files.get_info = AsyncMock(side_effect=NotFoundException("missing"))
    mock_sandbox.files.write = AsyncMock(side_effect=SandboxException("Is a directory"))
    env = NovitaSingleServiceEnvironment(mock_sandbox)
    with pytest.raises(IsADirectoryError):
        await env.write_file("/workspace/test.txt", "y")


@pytest.mark.asyncio
async def test_read_file_text(mock_sandbox: MagicMock) -> None:
    """Test read_file in text mode."""
    env = NovitaSingleServiceEnvironment(mock_sandbox)
    file_info = MagicMock()
    file_info.type = FileType.FILE
    file_info.size = 12
    mock_sandbox.files.get_info = AsyncMock(return_value=file_info)
    mock_sandbox.files.read = AsyncMock(return_value="hello world\n")

    result = await env.read_file("/test.txt", text=True)
    assert result == "hello world\n"


@pytest.mark.asyncio
async def test_read_file_binary(mock_sandbox: MagicMock) -> None:
    """Test read_file in binary mode."""
    env = NovitaSingleServiceEnvironment(mock_sandbox)
    file_info = MagicMock()
    file_info.type = FileType.FILE
    file_info.size = 4
    mock_sandbox.files.get_info = AsyncMock(return_value=file_info)
    mock_sandbox.files.read = AsyncMock(return_value=bytearray(b"\x00\x01\x02\x03"))

    result = await env.read_file("/test.bin", text=False)
    assert result == b"\x00\x01\x02\x03"


@pytest.mark.asyncio
async def test_read_file_retries_transport_error(mock_sandbox: MagicMock) -> None:
    """A dropped connection (httpx.ReadError) during read must be retried."""
    env = NovitaSingleServiceEnvironment(mock_sandbox)
    file_info = MagicMock()
    file_info.type = FileType.FILE
    file_info.size = 12
    mock_sandbox.files.get_info = AsyncMock(return_value=file_info)
    mock_sandbox.files.read = AsyncMock(
        side_effect=[httpx.ReadError("reset"), "hello world\n"]
    )

    result = await env.read_file("/test.txt", text=True)

    assert result == "hello world\n"
    assert mock_sandbox.files.read.await_count == 2


@pytest.mark.asyncio
async def test_read_file_not_found(mock_sandbox: MagicMock) -> None:
    """Test read_file raises FileNotFoundError when file doesn't exist."""
    env = NovitaSingleServiceEnvironment(mock_sandbox)
    mock_sandbox.files.get_info = AsyncMock(side_effect=NotFoundException("missing"))

    with pytest.raises(FileNotFoundError):
        await env.read_file("/missing.txt")


@pytest.mark.asyncio
async def test_read_file_is_directory(mock_sandbox: MagicMock) -> None:
    """Test read_file raises IsADirectoryError for directories."""
    env = NovitaSingleServiceEnvironment(mock_sandbox)
    dir_info = MagicMock()
    dir_info.type = FileType.DIR
    dir_info.size = 0
    mock_sandbox.files.get_info = AsyncMock(return_value=dir_info)

    with pytest.raises(IsADirectoryError):
        await env.read_file("/some/dir")


@pytest.mark.asyncio
async def test_read_file_size_limit(mock_sandbox: MagicMock) -> None:
    """Test read_file raises OutputLimitExceededError for oversized files."""
    env = NovitaSingleServiceEnvironment(mock_sandbox)
    file_info = MagicMock()
    file_info.type = FileType.FILE
    file_info.size = SandboxEnvironmentLimits.MAX_READ_FILE_SIZE + 1
    mock_sandbox.files.get_info = AsyncMock(return_value=file_info)

    with pytest.raises(OutputLimitExceededError):
        await env.read_file("/huge.bin")


@pytest.mark.asyncio
async def test_read_file_filenotfound_during_read(mock_sandbox: MagicMock) -> None:
    info = MagicMock()
    info.type = FileType.FILE
    info.size = 100
    mock_sandbox.files.get_info = AsyncMock(return_value=info)
    mock_sandbox.files.read = AsyncMock(side_effect=FileNotFoundException("gone"))
    env = NovitaSingleServiceEnvironment(mock_sandbox)
    with pytest.raises(FileNotFoundError):
        await env.read_file("/test.txt")


@pytest.mark.asyncio
async def test_sample_cleanup_kills_sandbox(mock_sandbox: MagicMock) -> None:
    """Test sample_cleanup kills all sandboxes in environments dict."""
    env = NovitaSingleServiceEnvironment(mock_sandbox)
    await NovitaSingleServiceEnvironment.sample_cleanup(
        "task", None, {"default": env}, False
    )
    mock_sandbox.kill.assert_awaited_once()


@pytest.mark.asyncio
async def test_sample_cleanup_skips_when_interrupted(mock_sandbox: MagicMock) -> None:
    """Test sample_cleanup does nothing when interrupted."""
    env = NovitaSingleServiceEnvironment(mock_sandbox)
    await NovitaSingleServiceEnvironment.sample_cleanup(
        "task", None, {"default": env}, True
    )
    mock_sandbox.kill.assert_not_called()


@pytest.mark.asyncio
async def test_sample_cleanup_skips_when_no_environments() -> None:
    """No-op when environments dict is empty (e.g. sample_init never returned)."""
    await NovitaSingleServiceEnvironment.sample_cleanup("task", None, {}, False)
    # nothing to assert; verifying no exception


@pytest.mark.asyncio
async def test_sample_cleanup_continues_on_kill_failure(
    mock_sandbox: MagicMock,
) -> None:
    """Test sample_cleanup logs error and continues when a kill fails."""
    mock_sandbox.kill = AsyncMock(side_effect=RuntimeError("flaky"))
    env = NovitaSingleServiceEnvironment(mock_sandbox)
    # Should not raise — error is traced and deferred.
    await NovitaSingleServiceEnvironment.sample_cleanup(
        "task", None, {"default": env}, False
    )


@pytest.mark.asyncio
async def test_connection_surfaces_get_host_as_ports(
    mock_sandbox: MagicMock,
) -> None:
    """connection() maps each declared port to a get_host URL."""
    # get_host is synchronous in the Novita SDK.
    mock_sandbox.get_host = MagicMock(return_value="3000-sb-test-123.sandbox.novita.ai")

    env = NovitaSingleServiceEnvironment(mock_sandbox, connection_ports=[3000])
    conn = await env.connection()

    assert conn.type == "novita"
    assert conn.ports is not None
    assert len(conn.ports) == 1
    assert conn.ports[0].container_port == 3000
    mapping = conn.ports[0].mappings[0]
    # get_host returns a bare host (no scheme); Novita serves it over HTTPS:443.
    assert mapping.host_ip == "3000-sb-test-123.sandbox.novita.ai"
    assert "://" not in mapping.host_ip
    assert mapping.host_port == 443
    mock_sandbox.get_host.assert_called_once_with(3000)


@pytest.mark.asyncio
async def test_connection_without_ports_is_empty(mock_sandbox: MagicMock) -> None:
    """No declared ports yields a connection with ports=None."""
    env = NovitaSingleServiceEnvironment(mock_sandbox)
    conn = await env.connection()

    assert conn.type == "novita"
    assert conn.ports is None
    assert conn.container == "sb-test-123"
