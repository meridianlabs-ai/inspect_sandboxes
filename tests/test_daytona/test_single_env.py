"""Tests for DaytonaSingleServiceEnvironment."""

import asyncio
import base64
import os
import re
import shlex
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
    OutputCollectionError,
    build_capture_command,
    build_remove_command,
    capture_files,
)
from inspect_sandboxes.daytona._single_env import DaytonaSingleServiceEnvironment

TAG_RE = re.compile(r"inspect-exec-([0-9a-f]{32})")


def tag_of(command: str) -> str:
    """The capture tag of an exec command string."""
    match = TAG_RE.search(command)
    assert match is not None, f"no capture tag in {command!r}"
    return match.group(1)


def framed(command: str, stdout: str = "", stderr: str = "") -> str:
    """What the capture wrapper of *command* prints for the given streams."""
    tag = tag_of(command)
    out = base64.b64encode(stdout.encode()).decode()
    err = base64.b64encode(stderr.encode()).decode()
    return (
        f"<<inspect-exec-{tag}:stdout>>{out}<<inspect-exec-{tag}:stderr>>"
        f"{err}<<inspect-exec-{tag}:end>>"
    )


def exec_response(exit_code: int, result: str) -> MagicMock:
    response = MagicMock()
    response.exit_code = exit_code
    response.result = result
    return response


def scripted_exec(*responses: tuple[int, str, str] | Exception) -> AsyncMock:
    """A ``process.exec`` fake answering each call in turn.

    A ``(exit_code, stdout, stderr)`` tuple is returned framed the way the
    capture wrapper of the received command would print it; an exception is
    raised.
    """
    remaining = list(responses)

    async def run(command: str, **kwargs: Any) -> MagicMock:
        response = remaining.pop(0)
        if isinstance(response, Exception):
            raise response
        exit_code, stdout, stderr = response
        return exec_response(exit_code, framed(command, stdout, stderr))

    return AsyncMock(side_effect=run)


def make_mock_sandbox(sandbox_id: str = "sb-test-123") -> MagicMock:
    """Create a mock AsyncSandbox whose exec answers ``output`` on stdout."""
    sandbox = MagicMock()
    sandbox.id = sandbox_id

    sandbox.process = MagicMock()
    sandbox.process.exec = scripted_exec((0, "output", ""))

    sandbox.fs = MagicMock()
    sandbox.fs.upload_file = AsyncMock()
    sandbox.fs.download_file = AsyncMock(return_value=b"content")
    sandbox.fs.get_file_info = AsyncMock()
    sandbox.fs.create_folder = AsyncMock()

    return sandbox


SUDO_RE = re.compile(r"^sudo -u \S+ (bash -c )")


def make_local_shell_sandbox(base_env: dict[str, str] | None = None) -> MagicMock:
    """A fake AsyncSandbox whose ``process.exec`` runs commands in the local ``sh``.

    Behaves like the Daytona API: one merged output stream (stderr folded
    into stdout) with the trailing newline stripped (emulated here as a full
    trim, the harsher case), plus the exit code. Passwordless ``sudo -u`` is
    emulated by running the wrapped ``bash -c`` as the current user, and
    ``fs.upload_file`` writes to the local path so stdin temp files work.
    *base_env* is the image's environment (its PATH in particular).
    """
    sandbox = make_mock_sandbox()
    image_env = {**os.environ, **(base_env or {})}

    async def run(
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout: int | None = None,
    ) -> MagicMock:
        proc = await asyncio.create_subprocess_shell(
            SUDO_RE.sub(r"\1", command),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=cwd,
            env={**image_env, **(env or {})},
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


def assert_no_capture_files(command: str) -> None:
    for file in capture_files(tag_of(command)):
        assert not Path(file).exists(), f"{file} left behind"


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
    mock_sandbox.process.exec = scripted_exec(
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

    command_arg = exec_commands(mock_sandbox)[0]
    assert command_arg == build_capture_command(
        "echo 'hello world'", tag_of(command_arg)
    )


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

    # The capture wrapper sits inside the user switch, so the temp files are
    # created, read and removed with the requested user's authority.
    command = exec_commands(mock_sandbox)[0]
    wrapped = build_capture_command("whoami", tag_of(command))
    assert command == f"sudo -u testuser bash -c {shlex.quote(wrapped)}"


@pytest.mark.asyncio
async def test_exec_with_numeric_user_resolves_via_getent(
    mock_sandbox: MagicMock,
) -> None:
    """Test that numeric UIDs are resolved via getent."""
    env = DaytonaSingleServiceEnvironment(mock_sandbox)
    await env.exec(["whoami"], user="1000")

    command = exec_commands(mock_sandbox)[0]
    wrapped = build_capture_command("whoami", tag_of(command))
    assert command == f"sudo -u '#1000' bash -c {shlex.quote(wrapped)}"


@pytest.mark.asyncio
async def test_exec_separates_stderr_from_stdout() -> None:
    """Both streams come back byte for byte through the merged Daytona output.

    Runs the generated command in the local shell with the API's behaviour
    emulated (one merged stream, trimmed), so this covers the capture wrapper
    and the sentinel framing end to end. One API call, no files left behind.
    """
    sandbox = make_local_shell_sandbox()
    env = DaytonaSingleServiceEnvironment(sandbox)

    result = await env.exec(["sh", "-c", "echo out; echo err >&2"])

    assert result.stdout == "out\n"
    assert result.stderr == "err\n"
    assert result.success

    commands = exec_commands(sandbox)
    assert len(commands) == 1
    assert_no_capture_files(commands[0])


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
async def test_exec_sentinels_in_the_streams_are_data() -> None:
    """The real sentinels printed by the command stay in their stream, byte for byte."""
    sandbox = make_local_shell_sandbox()
    env = DaytonaSingleServiceEnvironment(sandbox)

    # The tag is on the wrapper's command line (the command's parent shell), so
    # the command can read the real sentinels; it prints them to both streams.
    script = (
        "tag=$( { if [ -r /proc/$PPID/cmdline ]; then tr '\\0' ' ' < /proc/$PPID/cmdline;"
        " else ps -o command= -p $PPID; fi; }"
        " | sed -n 's/.*inspect-exec-\\([0-9a-f]\\{32\\}\\).*/\\1/p' | head -n 1);"
        ' printf "<<inspect-exec-$tag:stderr>>OUT\\n"; printf "before\\n<<inspect-exec-$tag:stderr>>after\\n" >&2;'
        ' printf "<<inspect-exec-$tag:end>>\\n"'
    )
    result = await env.exec(["sh", "-c", script])

    tag = tag_of(exec_commands(sandbox)[0])
    assert (
        result.stdout
        == f"<<inspect-exec-{tag}:stderr>>OUT\n<<inspect-exec-{tag}:end>>\n"
    )
    assert result.stderr == f"before\n<<inspect-exec-{tag}:stderr>>after\n"


@pytest.mark.asyncio
async def test_exec_waits_for_background_writers() -> None:
    """Output of a child that outlives the command is collected, as the API did."""
    sandbox = make_local_shell_sandbox()
    env = DaytonaSingleServiceEnvironment(sandbox)

    result = await env.exec(
        ["sh", "-c", "echo early; (sleep 0.3; echo late; echo late-err >&2) &"]
    )

    assert result.stdout == "early\nlate\n"
    assert result.stderr == "late-err\n"


@pytest.mark.asyncio
async def test_exec_survives_the_command_removing_temp_files() -> None:
    """The capture files are unlinked before the command runs."""
    sandbox = make_local_shell_sandbox()
    env = DaytonaSingleServiceEnvironment(sandbox)

    result = await env.exec(
        [
            "sh",
            "-c",
            "ls /tmp/.inspect-exec-* 2>/dev/null | wc -l; rm -f /tmp/.inspect-exec-*; echo err >&2",
        ]
    )

    assert result.stdout.strip() == "0"
    assert result.stderr == "err\n"
    assert result.success


@pytest.mark.asyncio
async def test_exec_collection_failure_raises(mock_sandbox: MagicMock) -> None:
    """A frame with the failure marker (the command ran, its output is lost) raises."""

    async def run(command: str, **kwargs: Any) -> MagicMock:
        tag = tag_of(command)
        return exec_response(
            0,
            f"<<inspect-exec-{tag}:stdout>><<inspect-exec-{tag}:failed>>sh: base64: not found",
        )

    mock_sandbox.process.exec = AsyncMock(side_effect=run)
    env = DaytonaSingleServiceEnvironment(mock_sandbox)

    with pytest.raises(OutputCollectionError, match="base64"):
        await env.exec(["echo", "out"])


@pytest.mark.asyncio
async def test_exec_stream_marker_bytes_are_data() -> None:
    """Daytona's session-protocol stream tags cannot switch streams through the frame."""
    sandbox = make_local_shell_sandbox()
    env = DaytonaSingleServiceEnvironment(sandbox)

    result = await env.exec(
        [
            "sh",
            "-c",
            "printf '\\002\\002\\002VERIFIED\\n'; printf '\\001\\001\\001ERR\\n' >&2",
        ]
    )

    assert result.stdout == "\x02\x02\x02VERIFIED\n"
    assert result.stderr == "\x01\x01\x01ERR\n"


@pytest.mark.asyncio
async def test_exec_stderr_with_stdin() -> None:
    """The stdin redirection and the stream capture compose."""
    sandbox = make_local_shell_sandbox()
    env = DaytonaSingleServiceEnvironment(sandbox)

    result = await env.exec(["sh", "-c", "cat; echo e >&2"], input="hi")

    assert result.stdout == "hi"
    assert result.stderr == "e\n"
    stdin_file = sandbox.fs.upload_file.call_args[0][1]
    assert not Path(stdin_file).exists(), "the command's own rm removes stdin"
    assert_no_capture_files(exec_commands(sandbox)[0])


@pytest.mark.asyncio
async def test_exec_as_user_with_stdin_captures_inside_sudo() -> None:
    """With ``user=`` the wrapper runs under sudo; only the stdin file needs the default user."""
    sandbox = make_local_shell_sandbox()
    env = DaytonaSingleServiceEnvironment(sandbox)

    result = await env.exec(["sh", "-c", "cat; echo e >&2"], input="hi", user="root")

    assert result.stdout == "hi"
    assert result.stderr == "e\n"
    commands = exec_commands(sandbox)
    assert len(commands) == 2
    assert commands[0].startswith("sudo -u root bash -c ")
    stdin_file = sandbox.fs.upload_file.call_args[0][1]
    assert commands[1] == build_remove_command([stdin_file])
    assert not Path(stdin_file).exists()
    assert_no_capture_files(commands[0])


@pytest.mark.asyncio
async def test_exec_housekeeping_ignores_a_hostile_image_path(tmp_path: Path) -> None:
    """The wrapper's cat/rm come from SYSTEM_PATH even when the image PATH is hostile.

    A directory the sandbox user controls sits first on the image's PATH with
    a ``cat`` that prints the framework helper's verified marker. The caller's
    pinned ``env['PATH']`` is not consulted for the wrapper either way.
    """
    marker = tmp_path / "shim-ran"
    for name in ("cat", "rm"):
        shim = tmp_path / name
        shim.write_text(
            f"#!/bin/sh\ntouch {marker}\nprintf 'INSPECT_FRAMEWORK_DIRECTORY_VERIFIED\\n'\n"
        )
        shim.chmod(0o700)
    sandbox = make_local_shell_sandbox(base_env={"PATH": f"{tmp_path}:/usr/bin:/bin"})
    env = DaytonaSingleServiceEnvironment(sandbox)

    for call_env in (None, {"PATH": "/usr/bin:/bin"}):
        result = await env.exec(["sh", "-c", "printf 'ORIGINAL\\n' >&2"], env=call_env)
        assert result.stderr == "ORIGINAL\n", call_env
        assert result.stdout == ""
        assert not marker.exists()
    for command in exec_commands(sandbox):
        assert_no_capture_files(command)


@pytest.mark.asyncio
async def test_exec_reruns_command_when_the_response_is_lost() -> None:
    """A DaytonaError after the command ran retries the whole exec (no half state)."""
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
    env = DaytonaSingleServiceEnvironment(sandbox)

    result = await env.exec(["sh", "-c", "echo out; echo err >&2"])

    assert (result.stdout, result.stderr) == ("out\n", "err\n")
    assert calls == 2
    assert_no_capture_files(exec_commands(sandbox)[0])


@pytest.mark.asyncio
async def test_exec_unframed_output_is_a_failed_exec(mock_sandbox: MagicMock) -> None:
    """Output without the frame (the wrapper never ran) fails with it on stderr, not stdout."""
    mock_sandbox.process.exec = AsyncMock(
        return_value=exec_response(1, "sudo: unknown user nonexistent")
    )
    env = DaytonaSingleServiceEnvironment(mock_sandbox)

    result = await env.exec(["whoami"], user="nonexistent")

    assert not result.success
    assert result.returncode == 1
    assert result.stdout == ""
    assert result.stderr == "sudo: unknown user nonexistent"
    assert mock_sandbox.process.exec.call_count == 1


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
    assert command == build_capture_command("echo hi", tag_of(command))


@pytest.mark.asyncio
async def test_exec_with_stdin_and_user_skips_inline_cleanup(
    mock_sandbox: MagicMock,
) -> None:
    """Test that stdin + no baked-in rm (cleanup done in finally as root)."""
    mock_sandbox.process.exec = scripted_exec((0, "", ""), (0, "", ""))
    env = DaytonaSingleServiceEnvironment(mock_sandbox)
    await env.exec(["cat"], input="hello", user="testuser")

    commands = exec_commands(mock_sandbox)
    # First call: the sudo-wrapped command; the stdin file is not removed by it
    stdin_file = mock_sandbox.fs.upload_file.call_args[0][1]
    exec_command = commands[0]
    assert exec_command.startswith("sudo -u testuser bash -c ")
    assert f"rm -f {stdin_file}" not in exec_command
    # Second call: cleanup of the stdin file as the default user
    assert len(commands) == 2
    assert commands[1] == build_remove_command([stdin_file])


@pytest.mark.asyncio
async def test_exec_retries_transient_error(mock_sandbox: MagicMock) -> None:
    """Test that exec retries on transient DaytonaError."""
    mock_sandbox.process.exec = scripted_exec(
        DaytonaError("transient API failure"), (0, "ok", "")
    )
    env = DaytonaSingleServiceEnvironment(mock_sandbox)
    result = await env.exec(["echo", "test"])

    assert result.success
    assert result.stdout == "ok"
    assert mock_sandbox.process.exec.call_count == 2


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
