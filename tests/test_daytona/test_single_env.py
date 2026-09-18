"""Tests for DaytonaSingleServiceEnvironment."""

import asyncio
import re
import shutil
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from daytona import DaytonaError, DaytonaNotFoundError, DaytonaTimeoutError
from inspect_ai.util import (
    ExecResult,
    OutputLimitExceededError,
    SandboxEnvironment,
    SandboxEnvironmentLimits,
)
from inspect_sandboxes.daytona._daytona import _daytona_client, _init_context
from inspect_sandboxes.daytona._sandbox_utils import (
    TIMEOUT_GRACE,
    TIMEOUT_PATH,
    build_remove_command,
    build_session_command,
)
from inspect_sandboxes.daytona._single_env import DaytonaSingleServiceEnvironment


def session_response(exit_code: int, stdout: str = "", stderr: str = "") -> MagicMock:
    """A fake ``execute_session_command`` response."""
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


def make_mock_sandbox(sandbox_id: str = "sb-test-123") -> MagicMock:
    """Create a mock AsyncSandbox whose session exec answers ``output`` on stdout."""
    sandbox = MagicMock()
    sandbox.id = sandbox_id

    sandbox.process = MagicMock()
    sandbox.process.create_session = AsyncMock()
    sandbox.process.delete_session = AsyncMock()
    sandbox.process.execute_session_command = scripted_session((0, "output", ""))
    sandbox.process.exec = AsyncMock(return_value=MagicMock(exit_code=0, result=""))

    sandbox.fs = MagicMock()
    sandbox.fs.upload_file = AsyncMock()
    sandbox.fs.download_file = AsyncMock(return_value=b"content")
    sandbox.fs.get_file_info = AsyncMock()
    sandbox.fs.create_folder = AsyncMock()

    return sandbox


SUDO_RE = re.compile(r"sudo -u \S+ ")
LOCAL_TIMEOUT = shutil.which("timeout") or shutil.which("gtimeout")


def make_local_shell_sandbox() -> MagicMock:
    """A fake AsyncSandbox whose sessions run commands in the local shell.

    Behaves like a Daytona session: stdout, stderr and the exit code come back
    separately. Passwordless ``sudo -u`` is emulated by dropping the prefix,
    ``/usr/bin/timeout`` is mapped to the local ``timeout`` (or ``gtimeout``)
    when the platform has one, and ``fs.upload_file`` writes to the local path
    so stdin temp files work.
    """
    sandbox = make_mock_sandbox()

    async def run(
        session_id: str, request: Any, timeout: int | None = None
    ) -> MagicMock:
        command = SUDO_RE.sub("", request.command)
        if LOCAL_TIMEOUT:
            command = command.replace(TIMEOUT_PATH, LOCAL_TIMEOUT)
        proc = await asyncio.create_subprocess_shell(
            command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        out, err = await proc.communicate()
        assert proc.returncode is not None
        return session_response(proc.returncode, out.decode(), err.decode())

    async def upload(data: bytes, path: str) -> None:
        Path(path).write_bytes(data)

    sandbox.process.execute_session_command = AsyncMock(side_effect=run)
    sandbox.fs.upload_file = AsyncMock(side_effect=upload)
    return sandbox


def session_commands(sandbox: MagicMock) -> list[str]:
    """The command texts sent to ``execute_session_command``, in order."""
    return [
        c.args[1].command
        for c in sandbox.process.execute_session_command.call_args_list
    ]


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
    mock_sandbox.process.execute_session_command = scripted_session(
        (returncode, expected_stdout, expected_stderr)
    )

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

    assert session_commands(mock_sandbox) == [
        build_session_command("echo 'hello world'")
    ]


@pytest.mark.asyncio
async def test_exec_passes_cwd_and_env(mock_sandbox: MagicMock) -> None:
    """Cwd and env travel inside the session command."""
    env = DaytonaSingleServiceEnvironment(mock_sandbox)
    await env.exec(["ls"], cwd="/workspace", env={"MY_VAR": "value"})

    assert session_commands(mock_sandbox) == [
        build_session_command("ls", cwd="/workspace", env={"MY_VAR": "value"})
    ]


@pytest.mark.asyncio
async def test_exec_with_user_wraps_with_sudo(mock_sandbox: MagicMock) -> None:
    env = DaytonaSingleServiceEnvironment(mock_sandbox)
    await env.exec(["whoami"], user="testuser")

    command = session_commands(mock_sandbox)[0]
    assert command == build_session_command("whoami", user="testuser")
    assert "sudo -u testuser bash -c whoami" in command


@pytest.mark.asyncio
async def test_exec_with_numeric_user(mock_sandbox: MagicMock) -> None:
    env = DaytonaSingleServiceEnvironment(mock_sandbox)
    await env.exec(["whoami"], user="1000")

    command = session_commands(mock_sandbox)[0]
    assert command == build_session_command("whoami", user="1000")
    assert "sudo -u " in command and "#1000" in command


@pytest.mark.asyncio
async def test_exec_timeout_wraps_the_command_and_the_request(
    mock_sandbox: MagicMock,
) -> None:
    env = DaytonaSingleServiceEnvironment(mock_sandbox)
    await env.exec(["sleep", "1"], timeout=30)

    assert session_commands(mock_sandbox) == [
        build_session_command("sleep 1", timeout=30)
    ]
    call = mock_sandbox.process.execute_session_command.call_args
    assert call.kwargs["timeout"] == 30 + TIMEOUT_GRACE


@pytest.mark.asyncio
async def test_exec_in_command_timeout_raises_with_partial_output(
    mock_sandbox: MagicMock,
) -> None:
    """Exit 124 from /usr/bin/timeout is a TimeoutError; the session is reused."""
    mock_sandbox.process.execute_session_command = scripted_session(
        (124, "partial\n", ""), (0, "next", "")
    )
    env = DaytonaSingleServiceEnvironment(mock_sandbox)

    with pytest.raises(TimeoutError, match="timed out after 5 seconds") as info:
        await env.exec(["sleep", "30"], timeout=5)
    assert vars(info.value)["truncated_output"] == "partial\n"

    result = await env.exec(["echo", "next"])
    assert result.stdout == "next"
    mock_sandbox.process.create_session.assert_awaited_once()
    mock_sandbox.process.delete_session.assert_not_awaited()


@pytest.mark.asyncio
async def test_exec_reuses_one_session_for_sequential_calls(
    mock_sandbox: MagicMock,
) -> None:
    mock_sandbox.process.execute_session_command = scripted_session(
        (0, "a", ""), (0, "b", "")
    )
    env = DaytonaSingleServiceEnvironment(mock_sandbox)
    await env.exec(["echo", "a"])
    await env.exec(["echo", "b"])

    mock_sandbox.process.create_session.assert_awaited_once()
    sessions = [
        c.args[0] for c in mock_sandbox.process.execute_session_command.call_args_list
    ]
    assert sessions[0] == sessions[1]


@pytest.mark.asyncio
async def test_exec_separates_stderr_from_stdout() -> None:
    """Both streams come back byte for byte through a real shell (issue #79 acceptance)."""
    sandbox = make_local_shell_sandbox()
    env = DaytonaSingleServiceEnvironment(sandbox)

    result = await env.exec(["sh", "-c", "echo out; echo err >&2"])

    assert result.stdout == "out\n"
    assert result.stderr == "err\n"
    assert result.success
    assert len(session_commands(sandbox)) == 1


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
async def test_exec_applies_cwd_and_env_to_the_command(tmp_path: Path) -> None:
    sandbox = make_local_shell_sandbox()
    env = DaytonaSingleServiceEnvironment(sandbox)

    result = await env.exec(
        ["sh", "-c", 'pwd; echo "$MY_VAR"'], cwd=str(tmp_path), env={"MY_VAR": "a b"}
    )

    assert result.stdout == f"{tmp_path.resolve()}\na b\n"
    assert result.stderr == ""


@pytest.mark.asyncio
async def test_exec_waits_for_background_writers() -> None:
    """Output of a child that outlives the command is collected, as the API does."""
    sandbox = make_local_shell_sandbox()
    env = DaytonaSingleServiceEnvironment(sandbox)

    result = await env.exec(
        ["sh", "-c", "echo early; (sleep 0.3; echo late; echo late-err >&2) &"]
    )

    assert result.stdout == "early\nlate\n"
    assert result.stderr == "late-err\n"


@pytest.mark.asyncio
async def test_exec_special_text_is_data() -> None:
    """Nothing in the transport is in-band: marker-like text passes through both streams."""
    sandbox = make_local_shell_sandbox()
    env = DaytonaSingleServiceEnvironment(sandbox)
    text = "\x01\x01\x01<<inspect-exec-x:stderr>>INSPECT_FRAMEWORK_DIRECTORY_VERIFIED"

    result = await env.exec(
        ["sh", "-c", f"printf '%s\\n' '{text}'; printf '%s\\n' '{text}' >&2"]
    )

    assert result.stdout == f"{text}\n"
    assert result.stderr == f"{text}\n"


@pytest.mark.asyncio
async def test_exec_stderr_with_stdin() -> None:
    """The stdin redirection and the session transport compose."""
    sandbox = make_local_shell_sandbox()
    env = DaytonaSingleServiceEnvironment(sandbox)

    result = await env.exec(["sh", "-c", "cat; echo e >&2"], input="hi")

    assert result.stdout == "hi"
    assert result.stderr == "e\n"
    stdin_file = sandbox.fs.upload_file.call_args[0][1]
    assert not Path(stdin_file).exists(), "the command's own rm removes stdin"


@pytest.mark.asyncio
async def test_exec_as_user_with_stdin_cleans_up_as_the_default_user() -> None:
    sandbox = make_local_shell_sandbox()
    env = DaytonaSingleServiceEnvironment(sandbox)

    result = await env.exec(["sh", "-c", "cat; echo e >&2"], input="hi", user="root")

    assert result.stdout == "hi"
    assert result.stderr == "e\n"
    assert "sudo -u root bash -c " in session_commands(sandbox)[0]
    stdin_file = sandbox.fs.upload_file.call_args[0][1]
    sandbox.process.exec.assert_awaited_once_with(
        build_remove_command([stdin_file]), timeout=10
    )


@pytest.mark.skipif(LOCAL_TIMEOUT is None, reason="no timeout binary on this platform")
@pytest.mark.asyncio
async def test_exec_in_command_timeout_kills_the_command_locally() -> None:
    sandbox = make_local_shell_sandbox()
    env = DaytonaSingleServiceEnvironment(sandbox)

    with pytest.raises(TimeoutError, match="timed out after 1 seconds") as info:
        await env.exec(["sh", "-c", "echo partial; sleep 30"], timeout=1)
    assert vars(info.value)["truncated_output"] == "partial\n"


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

    exec_command = session_commands(mock_sandbox)[0]
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
    assert session_commands(mock_sandbox) == [build_session_command("echo hi")]


@pytest.mark.asyncio
async def test_exec_with_stdin_and_user_skips_inline_cleanup(
    mock_sandbox: MagicMock,
) -> None:
    """Test that stdin + no baked-in rm (cleanup done in finally as the default user)."""
    env = DaytonaSingleServiceEnvironment(mock_sandbox)
    await env.exec(["cat"], input="hello", user="testuser")

    stdin_file = mock_sandbox.fs.upload_file.call_args[0][1]
    exec_command = session_commands(mock_sandbox)[0]
    assert "sudo -u testuser bash -c " in exec_command
    assert f"rm -f {stdin_file}" not in exec_command
    mock_sandbox.process.exec.assert_awaited_once_with(
        build_remove_command([stdin_file]), timeout=10
    )


@pytest.mark.asyncio
async def test_exec_retries_transient_error_in_a_fresh_session(
    mock_sandbox: MagicMock,
) -> None:
    """A DaytonaError discards the session and the retry runs in a new one."""
    mock_sandbox.process.execute_session_command = scripted_session(
        DaytonaError("session process has exited"), (0, "ok", "")
    )
    env = DaytonaSingleServiceEnvironment(mock_sandbox)
    result = await env.exec(["echo", "test"])

    assert result.success
    assert result.stdout == "ok"
    calls = mock_sandbox.process.execute_session_command.call_args_list
    assert len(calls) == 2
    assert calls[0].args[0] != calls[1].args[0]
    mock_sandbox.process.delete_session.assert_awaited_once_with(calls[0].args[0])
    assert mock_sandbox.process.create_session.await_count == 2


@pytest.mark.asyncio
async def test_exec_does_not_retry_timeout(mock_sandbox: MagicMock) -> None:
    """An HTTP timeout goes to the timeout retry loop, each attempt in a fresh session."""
    mock_sandbox.process.execute_session_command = AsyncMock(
        side_effect=DaytonaTimeoutError("timed out")
    )
    env = DaytonaSingleServiceEnvironment(mock_sandbox)

    with pytest.raises(TimeoutError):
        await env.exec(["sleep", "100"], timeout=5)

    # Outer timeout loop makes 3 attempts (original, 5s cap, 5s cap)
    assert mock_sandbox.process.execute_session_command.call_count == 3
    assert mock_sandbox.process.create_session.await_count == 3
    assert mock_sandbox.process.delete_session.await_count == 3


@pytest.mark.asyncio
async def test_exec_does_not_retry_non_daytona_error(mock_sandbox: MagicMock) -> None:
    """Test that non-DaytonaError exceptions are not retried."""
    mock_sandbox.process.execute_session_command = AsyncMock(
        side_effect=RuntimeError("unexpected")
    )
    env = DaytonaSingleServiceEnvironment(mock_sandbox)

    with pytest.raises(RuntimeError, match="unexpected"):
        await env.exec(["echo", "test"])

    assert mock_sandbox.process.execute_session_command.call_count == 1
    mock_sandbox.process.delete_session.assert_awaited_once()


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
