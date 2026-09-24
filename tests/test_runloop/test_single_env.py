"""Tests for RunloopSingleServiceEnvironment."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from inspect_ai.util import OutputLimitExceededError
from inspect_sandboxes.runloop._retry import poll_execution, shutdown_devbox
from inspect_sandboxes.runloop._single_env import RunloopSingleServiceEnvironment
from runloop_api_client import (
    APIConnectionError,
    APITimeoutError,
    BadRequestError,
    NotFoundError,
)


def _make_execution(
    *,
    stdout: str = "",
    stderr: str = "",
    exit_status: int = 0,
    stdout_truncated: bool = False,
    stderr_truncated: bool = False,
) -> MagicMock:
    execution = MagicMock()
    execution.stdout = stdout
    execution.stderr = stderr
    execution.exit_status = exit_status
    execution.status = "completed"
    execution.execution_id = "exec-1"
    execution.stdout_truncated = stdout_truncated
    execution.stderr_truncated = stderr_truncated
    return execution


def _execution(status: str) -> MagicMock:
    execution = MagicMock()
    execution.status = status
    return execution


def _poll_client(retrieve: AsyncMock) -> MagicMock:
    """A mock client exposing devboxes.executions.retrieve/kill for poll_execution."""
    client = MagicMock()
    client.devboxes = MagicMock()
    client.devboxes.executions = MagicMock()
    client.devboxes.executions.retrieve = retrieve
    client.devboxes.executions.kill = AsyncMock()
    return client


def _make_client() -> MagicMock:
    """A mock AsyncRunloop client wired for exec (execute) and file I/O."""
    client = MagicMock()
    client.devboxes = MagicMock()
    completed = _make_execution(stdout="ok")
    # exec submits via execute; probes (stat/test/rm) use
    # execute_and_await_completion; file bytes move via upload_file/download_file.
    client.devboxes.execute = AsyncMock(return_value=completed)
    client.devboxes.execute_and_await_completion = AsyncMock(return_value=completed)
    client.devboxes.executions = MagicMock()
    client.devboxes.executions.retrieve = AsyncMock(return_value=completed)
    client.devboxes.executions.kill = AsyncMock()
    client.devboxes.upload_file = AsyncMock()
    download_response = MagicMock()
    download_response.read = AsyncMock(return_value=b"")
    client.devboxes.download_file = AsyncMock(return_value=download_response)
    return client


@pytest.fixture
def client() -> MagicMock:
    return _make_client()


@pytest.mark.asyncio
async def test_exec_basic(client: MagicMock) -> None:
    """Test exec returns ExecResult for a successful command."""
    env = RunloopSingleServiceEnvironment(client, "dbx-test-123")
    # A command that finishes within the optimistic window comes back completed
    # from execute itself, so no poll is needed.
    client.devboxes.execute = AsyncMock(
        return_value=_make_execution(stdout="hi\n", exit_status=0)
    )

    result = await env.exec(["echo", "hi"])

    assert result.success
    assert result.returncode == 0
    assert result.stdout == "hi\n"
    await_args = client.devboxes.execute.await_args
    assert await_args is not None
    command = await_args.kwargs["command"]
    assert command == "echo hi"


@pytest.mark.asyncio
async def test_exec_joins_args_with_shlex(client: MagicMock) -> None:
    """Args with spaces are shell-quoted so they're not split into multiple tokens."""
    env = RunloopSingleServiceEnvironment(client, "dbx-test-123")
    await env.exec(["echo", "hello world"])

    await_args = client.devboxes.execute.await_args
    assert await_args is not None
    command = await_args.kwargs["command"]
    assert command == "echo 'hello world'"


@pytest.mark.asyncio
async def test_exec_with_cwd(client: MagicMock) -> None:
    env = RunloopSingleServiceEnvironment(client, "dbx-test-123")
    await env.exec(["pwd"], cwd="/work")

    await_args = client.devboxes.execute.await_args
    assert await_args is not None
    command = await_args.kwargs["command"]
    assert command == "cd /work && pwd"


@pytest.mark.asyncio
async def test_exec_with_env(client: MagicMock) -> None:
    env = RunloopSingleServiceEnvironment(client, "dbx-test-123")
    await env.exec(["sh", "-c", "echo $FOO"], env={"FOO": "bar"})

    await_args = client.devboxes.execute.await_args
    assert await_args is not None
    command = await_args.kwargs["command"]
    assert command == "FOO=bar sh -c 'echo $FOO'"


@pytest.mark.asyncio
async def test_exec_with_user_wraps_in_sudo(client: MagicMock) -> None:
    env = RunloopSingleServiceEnvironment(client, "dbx-test-123")
    await env.exec(["whoami"], user="root")

    await_args = client.devboxes.execute.await_args
    assert await_args is not None
    command = await_args.kwargs["command"]
    assert command.startswith("sudo -u root bash -c ")


@pytest.mark.asyncio
async def test_exec_with_numeric_user_uses_uid_form(client: MagicMock) -> None:
    env = RunloopSingleServiceEnvironment(client, "dbx-test-123")
    await env.exec(["whoami"], user="4444")

    await_args = client.devboxes.execute.await_args
    assert await_args is not None
    command = await_args.kwargs["command"]
    assert command.startswith("sudo -u '#4444' bash -c ")


@pytest.mark.asyncio
async def test_exec_failure_returncode(client: MagicMock) -> None:
    """Non-zero exit_status from the SDK should surface as ExecResult.returncode."""
    env = RunloopSingleServiceEnvironment(client, "dbx-test-123")
    client.devboxes.execute = AsyncMock(
        return_value=_make_execution(stdout="", stderr="boom", exit_status=1)
    )

    result = await env.exec(["false"])
    assert not result.success
    assert result.returncode == 1
    assert result.stderr == "boom"


@pytest.mark.asyncio
async def test_exec_with_stdin_string(client: MagicMock) -> None:
    """Test exec redirects string stdin through a temp file."""
    env = RunloopSingleServiceEnvironment(client, "dbx-test-123")

    await env.exec(["cat"], input="hello")

    # The stdin temp is uploaded via upload_file.
    upload_kwargs = client.devboxes.upload_file.await_args.kwargs
    assert upload_kwargs["path"].startswith("/tmp/.inspect-stdin-")
    assert upload_kwargs["file"] == b"hello"
    # The command itself runs via execute, redirecting from the tmp file.
    exec_cmd = client.devboxes.execute.await_args.kwargs["command"]
    assert "cat < " in exec_cmd
    assert "/tmp/.inspect-stdin-" in exec_cmd
    assert "rm -f" in exec_cmd  # cleanup inline


@pytest.mark.asyncio
async def test_exec_with_stdin_bytes(client: MagicMock) -> None:
    """Test exec redirects bytes stdin through a temp file."""
    env = RunloopSingleServiceEnvironment(client, "dbx-test-123")
    await env.exec(["wc", "-c"], input=b"\x00\x01\x02")

    # Raw bytes are uploaded verbatim (no base64 round-trip).
    assert client.devboxes.upload_file.await_args.kwargs["file"] == b"\x00\x01\x02"


@pytest.mark.asyncio
async def test_exec_without_stdin_no_upload(client: MagicMock) -> None:
    """Test exec without stdin does not upload any file."""
    env = RunloopSingleServiceEnvironment(client, "dbx-test-123")
    await env.exec(["echo", "hi"])

    client.devboxes.upload_file.assert_not_awaited()
    assert client.devboxes.execute.await_count == 1
    assert client.devboxes.execute.await_args.kwargs["command"] == "echo hi"


@pytest.mark.asyncio
async def test_exec_with_stdin_and_user_skips_inline_cleanup(client: MagicMock) -> None:
    """Test that stdin + user defers temp-file cleanup to the finally block."""
    env = RunloopSingleServiceEnvironment(client, "dbx-test-123")
    await env.exec(["cat"], input="hello", user="testuser")

    # Stdin is uploaded; the deferred cleanup runs the rm as the default user.
    client.devboxes.upload_file.assert_awaited_once()
    cleanup_command = client.devboxes.execute_and_await_completion.await_args.kwargs[
        "command"
    ]
    assert cleanup_command.startswith("rm -f ")
    # The exec command runs via execute (wrapped in sudo, no baked-in rm -f).
    exec_command = client.devboxes.execute.await_args.kwargs["command"]
    assert "sudo -u" in exec_command
    assert "rm -f" not in exec_command


@pytest.mark.asyncio
async def test_exec_retries_transient_error(client: MagicMock) -> None:
    """Test that exec retries on transient APIError."""
    env = RunloopSingleServiceEnvironment(client, "dbx-test-123")
    call_count = 0
    success = _make_execution(stdout="ok", exit_status=0)

    async def flaky_run(*_args: Any, **_kwargs: Any) -> MagicMock:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise APIConnectionError(request=MagicMock())
        return success

    client.devboxes.execute = AsyncMock(side_effect=flaky_run)
    result = await env.exec(["echo", "test"])

    assert result.success
    assert call_count == 2


@pytest.mark.asyncio
async def test_exec_timeout_retry_uses_capped_timeouts(client: MagicMock) -> None:
    """First retry caps at 60s, second at 30s, then raises TimeoutError."""
    env = RunloopSingleServiceEnvironment(client, "dbx-test-123")
    client.devboxes.execute = AsyncMock(
        side_effect=APITimeoutError(request=MagicMock())
    )
    with pytest.raises(TimeoutError):
        await env.exec(["sleep", "1000"], timeout=120)
    # Three attempts: initial + 2 retries (timeouts capped at 120, 60, 30 by
    # run_with_timeout_retry — verified by call count since we use our own
    # polling for the actual timeout now).
    assert client.devboxes.execute.await_count == 3


@pytest.mark.asyncio
async def test_exec_raises_when_output_truncated(client: MagicMock) -> None:
    """Runloop truncating output surfaces as OutputLimitExceededError, not a silent drop."""
    env = RunloopSingleServiceEnvironment(client, "dbx-test-123")
    client.devboxes.execute = AsyncMock(
        return_value=_make_execution(
            stdout="tail", exit_status=0, stdout_truncated=True
        )
    )
    with pytest.raises(OutputLimitExceededError) as exc:
        await env.exec(["cat", "big.txt"])
    # The truncated stream is attached (stdout here).
    assert exc.value.truncated_output == "tail"


@pytest.mark.asyncio
async def test_exec_truncation_attaches_stderr_when_only_stderr_overflows(
    client: MagicMock,
) -> None:
    """When only stderr overflows, attach stderr — not the intact stdout."""
    env = RunloopSingleServiceEnvironment(client, "dbx-test-123")
    client.devboxes.execute = AsyncMock(
        return_value=_make_execution(
            stdout="fine", stderr="errtail", exit_status=0, stderr_truncated=True
        )
    )
    with pytest.raises(OutputLimitExceededError) as exc:
        await env.exec(["noisy"])
    assert exc.value.truncated_output == "errtail"


@pytest.mark.asyncio
async def test_exec_reuses_command_id_across_retries(client: MagicMock) -> None:
    """A retried submit reuses the command_id so Runloop dedupes it, not re-runs it."""
    env = RunloopSingleServiceEnvironment(client, "dbx-test-123")
    call_count = 0

    async def flaky(*_args: Any, **_kwargs: Any) -> MagicMock:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise APIConnectionError(request=MagicMock())
        return _make_execution(stdout="ok", exit_status=0)

    client.devboxes.execute = AsyncMock(side_effect=flaky)
    await env.exec(["echo", "hi"])

    ids = [c.kwargs["command_id"] for c in client.devboxes.execute.await_args_list]
    assert len(ids) == 2
    assert ids[0] == ids[1]


@pytest.mark.asyncio
async def test_exec_untimed_polls_instead_of_blocking(client: MagicMock) -> None:
    """timeout=None submits then polls (no 25s-window last_n drop, no ~52min cap)."""
    env = RunloopSingleServiceEnvironment(client, "dbx-test-123")
    started = MagicMock(execution_id="exec-1", status="running")
    client.devboxes.execute = AsyncMock(return_value=started)
    client.devboxes.executions.retrieve = AsyncMock(
        return_value=_make_execution(stdout="done", exit_status=0)
    )

    result = await env.exec(["long-running"], timeout=None)

    assert result.stdout == "done"
    client.devboxes.execute_and_await_completion.assert_not_awaited()
    # last_n stays applied on every poll.
    retrieve_args = client.devboxes.executions.retrieve.await_args
    assert retrieve_args is not None
    assert retrieve_args.kwargs["last_n"] == "9999"


@pytest.mark.asyncio
async def test_exec_completed_within_optimistic_window_skips_poll(
    client: MagicMock,
) -> None:
    """A command that finishes inside the optimistic window needs no poll."""
    env = RunloopSingleServiceEnvironment(client, "dbx-test-123")
    client.devboxes.execute = AsyncMock(
        return_value=_make_execution(stdout="fast", exit_status=0)  # status=completed
    )

    result = await env.exec(["echo", "fast"])

    assert result.stdout == "fast"
    client.devboxes.executions.retrieve.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("timeout", "expected"),
    [(None, 25), (100, 25), (10, 10), (1, 1), (0, 1)],
)
async def test_exec_bounds_optimistic_timeout(
    client: MagicMock, timeout: int | None, expected: int
) -> None:
    """optimistic_timeout is capped at 25s and floored at 1s (0 makes Runloop 408)."""
    env = RunloopSingleServiceEnvironment(client, "dbx-test-123")
    await env.exec(["echo", "hi"], timeout=timeout)
    assert client.devboxes.execute.await_args.kwargs["optimistic_timeout"] == expected


@pytest.mark.asyncio
async def test_write_file_text(client: MagicMock) -> None:
    """Test write_file with text content uploads the bytes via upload_file."""
    env = RunloopSingleServiceEnvironment(client, "dbx-test-123")
    # The _is_directory probe returns non-zero → not a dir.
    client.devboxes.execute_and_await_completion = AsyncMock(
        return_value=_make_execution(exit_status=1)
    )

    await env.write_file("/tmp/out.txt", "hello")

    kwargs = client.devboxes.upload_file.await_args.kwargs
    assert kwargs["path"] == "/tmp/out.txt"
    assert kwargs["file"] == b"hello"


@pytest.mark.asyncio
async def test_write_file_binary(client: MagicMock) -> None:
    """Test write_file with binary content uploads the raw bytes."""
    env = RunloopSingleServiceEnvironment(client, "dbx-test-123")
    client.devboxes.execute_and_await_completion = AsyncMock(
        return_value=_make_execution(exit_status=1)  # not a directory
    )

    await env.write_file("/tmp/out.bin", b"\x00\x01\x02")

    assert client.devboxes.upload_file.await_args.kwargs["file"] == b"\x00\x01\x02"


@pytest.mark.asyncio
async def test_write_file_creates_parent_dir(client: MagicMock) -> None:
    """write_file mkdir -p's the parent (upload_file only writes the file)."""
    env = RunloopSingleServiceEnvironment(client, "dbx-test-123")
    client.devboxes.execute_and_await_completion = AsyncMock(
        return_value=_make_execution(exit_status=1)  # not a directory
    )

    await env.write_file("/tmp/nested/dir/baz.txt", "hi")

    # The parent mkdir runs through the exec path (execute), quoted as one token.
    mkdir_cmds = [c.kwargs["command"] for c in client.devboxes.execute.await_args_list]
    assert any("mkdir -p /tmp/nested/dir" in c for c in mkdir_cmds)
    assert client.devboxes.upload_file.await_args.kwargs["path"] == (
        "/tmp/nested/dir/baz.txt"
    )


@pytest.mark.asyncio
async def test_write_file_raises_for_directory(client: MagicMock) -> None:
    """Test write_file raises IsADirectoryError when path is a directory."""
    env = RunloopSingleServiceEnvironment(client, "dbx-test-123")
    client.devboxes.execute_and_await_completion = AsyncMock(
        return_value=_make_execution(exit_status=0)  # is_directory: yes
    )

    with pytest.raises(IsADirectoryError):
        await env.write_file("/tmp", "hello")
    client.devboxes.upload_file.assert_not_awaited()


@pytest.mark.asyncio
async def test_write_file_maps_permission_error(client: MagicMock) -> None:
    """A permission-denied 400 from upload_file surfaces as PermissionError."""
    env = RunloopSingleServiceEnvironment(client, "dbx-test-123")
    client.devboxes.execute_and_await_completion = AsyncMock(
        return_value=_make_execution(exit_status=1)  # not a directory
    )
    # Runloop reports filesystem permission failures as a 400 carrying EACCES.
    response = httpx.Response(400, request=httpx.Request("POST", "https://example"))
    client.devboxes.upload_file = AsyncMock(
        side_effect=BadRequestError(
            "denied",
            response=response,
            body={"message": "Permission denied: Permission denied (os error 13)"},
        )
    )

    with pytest.raises(PermissionError):
        await env.write_file("/root/locked.txt", "hi")


@pytest.mark.asyncio
async def test_write_file_reraises_non_permission_bad_request(
    client: MagicMock,
) -> None:
    """A 400 that isn't a permission error is not masked as PermissionError."""
    env = RunloopSingleServiceEnvironment(client, "dbx-test-123")
    client.devboxes.execute_and_await_completion = AsyncMock(
        return_value=_make_execution(exit_status=1)  # not a directory
    )
    response = httpx.Response(400, request=httpx.Request("POST", "https://example"))
    client.devboxes.upload_file = AsyncMock(
        side_effect=BadRequestError(
            "bad", response=response, body={"message": "invalid path"}
        )
    )

    with pytest.raises(BadRequestError):
        await env.write_file("/tmp/x.txt", "hi")


@pytest.mark.asyncio
async def test_read_file_text(client: MagicMock) -> None:
    """Test read_file in text mode downloads and decodes the bytes."""
    env = RunloopSingleServiceEnvironment(client, "dbx-test-123")
    client.devboxes.execute_and_await_completion = AsyncMock(
        side_effect=[
            _make_execution(exit_status=1),  # is_directory: no
            _make_execution(stdout="13", exit_status=0),  # stat -c %s
        ]
    )
    response = MagicMock()
    response.read = AsyncMock(return_value=b"file contents")
    client.devboxes.download_file = AsyncMock(return_value=response)

    result = await env.read_file("/tmp/file.txt")

    assert result == "file contents"
    download_args = client.devboxes.download_file.await_args
    assert download_args is not None
    assert download_args.kwargs["path"] == "/tmp/file.txt"


@pytest.mark.asyncio
async def test_read_file_binary(client: MagicMock) -> None:
    """Test read_file in binary mode returns the raw downloaded bytes."""
    env = RunloopSingleServiceEnvironment(client, "dbx-test-123")
    payload = b"\xff\xfe\x00"
    client.devboxes.execute_and_await_completion = AsyncMock(
        side_effect=[
            _make_execution(exit_status=1),  # is_directory: no
            _make_execution(stdout=str(len(payload)), exit_status=0),  # stat -c %s
        ]
    )
    response = MagicMock()
    response.read = AsyncMock(return_value=payload)
    client.devboxes.download_file = AsyncMock(return_value=response)

    result = await env.read_file("/tmp/file.bin", text=False)
    assert result == payload


@pytest.mark.asyncio
async def test_read_file_not_found(client: MagicMock) -> None:
    """Test read_file raises FileNotFoundError when file doesn't exist."""
    env = RunloopSingleServiceEnvironment(client, "dbx-test-123")
    client.devboxes.execute_and_await_completion = AsyncMock(
        side_effect=[
            _make_execution(exit_status=1),  # is_directory: no
            _make_execution(stderr="No such file or directory", exit_status=1),  # stat
            _make_execution(exit_status=1),  # test -e: missing
        ]
    )

    with pytest.raises(FileNotFoundError):
        await env.read_file("/tmp/missing.txt")


@pytest.mark.asyncio
async def test_read_file_maps_missing_from_download(client: MagicMock) -> None:
    """A "File does not exist" 400 from download_file surfaces as FileNotFoundError."""
    env = RunloopSingleServiceEnvironment(client, "dbx-test-123")
    # Pass the size checks so the download is actually attempted.
    client.devboxes.execute_and_await_completion = AsyncMock(
        side_effect=[
            _make_execution(exit_status=1),  # is_directory: no
            _make_execution(stdout="10", exit_status=0),  # stat -c %s
        ]
    )
    response = httpx.Response(400, request=httpx.Request("POST", "https://example"))
    client.devboxes.download_file = AsyncMock(
        side_effect=BadRequestError(
            "bad",
            response=response,
            body={"message": "File does not exist: /tmp/vanished.txt"},
        )
    )

    with pytest.raises(FileNotFoundError):
        await env.read_file("/tmp/vanished.txt")


@pytest.mark.asyncio
async def test_read_file_is_directory(client: MagicMock) -> None:
    """Test read_file raises IsADirectoryError for directories."""
    env = RunloopSingleServiceEnvironment(client, "dbx-test-123")
    client.devboxes.execute_and_await_completion = AsyncMock(
        return_value=_make_execution(exit_status=0)  # is_directory: yes
    )

    with pytest.raises(IsADirectoryError):
        await env.read_file("/tmp")


@pytest.mark.asyncio
async def test_read_file_size_limit(client: MagicMock) -> None:
    """Test read_file raises OutputLimitExceededError for oversized files."""
    env = RunloopSingleServiceEnvironment(client, "dbx-test-123")
    huge = 200 * 1024 * 1024  # 200 MiB, above 100 MiB read cap
    client.devboxes.execute_and_await_completion = AsyncMock(
        side_effect=[
            _make_execution(exit_status=1),  # is_directory: no
            _make_execution(stdout=str(huge), exit_status=0),  # stat -c %s
        ]
    )

    with pytest.raises(OutputLimitExceededError):
        await env.read_file("/tmp/huge.bin")
    client.devboxes.download_file.assert_not_awaited()


@pytest.mark.asyncio
async def test_exec_timeout_path_polls_without_resubmitting(client: MagicMock) -> None:
    """A transient error mid-poll is tolerated: the command is submitted once."""
    env = RunloopSingleServiceEnvironment(client, "dbx-test-123")
    # optimistic_timeout=0 returns still-running, so we poll.
    started = MagicMock(execution_id="exec-1", status="running")
    client.devboxes.execute = AsyncMock(return_value=started)
    client.devboxes.executions.retrieve = AsyncMock(
        side_effect=[
            APIConnectionError(request=MagicMock()),  # transient blip mid-poll
            _make_execution(stdout="ok", exit_status=0),  # then completes
        ]
    )

    result = await env.exec(["echo", "hi"], timeout=120)

    assert result.success
    assert client.devboxes.execute.await_count == 1
    assert client.devboxes.executions.retrieve.await_count == 2


@pytest.mark.asyncio
async def test_sample_cleanup_shuts_down_devbox(client: MagicMock) -> None:
    """Test sample_cleanup shuts down all devboxes in environments dict."""
    env = RunloopSingleServiceEnvironment(client, "dbx-test-123")
    client.devboxes.shutdown = AsyncMock()
    await RunloopSingleServiceEnvironment.sample_cleanup(
        "task", None, {"default": env}, False
    )
    client.devboxes.shutdown.assert_awaited_once_with("dbx-test-123")


@pytest.mark.asyncio
async def test_sample_cleanup_skips_when_interrupted(client: MagicMock) -> None:
    """Test sample_cleanup does nothing when interrupted."""
    env = RunloopSingleServiceEnvironment(client, "dbx-test-123")
    client.devboxes.shutdown = AsyncMock()
    await RunloopSingleServiceEnvironment.sample_cleanup(
        "task", None, {"default": env}, True
    )
    client.devboxes.shutdown.assert_not_called()


@pytest.mark.asyncio
async def test_sample_cleanup_skips_when_no_environments() -> None:
    """No-op when environments dict is empty (e.g. sample_init never returned)."""
    await RunloopSingleServiceEnvironment.sample_cleanup("task", None, {}, False)
    # nothing to assert; verifying no exception


@pytest.mark.asyncio
async def test_sample_cleanup_continues_on_shutdown_failure(client: MagicMock) -> None:
    """Test sample_cleanup logs error and continues when a shutdown fails."""
    env = RunloopSingleServiceEnvironment(client, "dbx-test-123")
    client.devboxes.shutdown = AsyncMock(side_effect=RuntimeError("flaky"))
    # Should not raise — error is traced and deferred.
    await RunloopSingleServiceEnvironment.sample_cleanup(
        "task", None, {"default": env}, False
    )


@pytest.mark.asyncio
async def test_poll_execution_returns_when_done() -> None:
    client = _poll_client(AsyncMock(return_value=_execution("completed")))
    result = await poll_execution(
        client, "dbx-1", "exec-1", timeout=None, last_n="9999"
    )
    assert result.status == "completed"


@pytest.mark.asyncio
async def test_poll_execution_times_out_and_kills() -> None:
    """On deadline overrun it raises TimeoutError and kills the execution."""
    client = _poll_client(AsyncMock(return_value=_execution("running")))
    with pytest.raises(TimeoutError):
        await poll_execution(client, "dbx-1", "exec-1", timeout=0, last_n="9999")
    client.devboxes.executions.kill.assert_awaited_once_with(
        "exec-1", devbox_id="dbx-1", kill_process_group=True
    )


@pytest.mark.asyncio
async def test_poll_execution_tolerates_transient_error_in_place() -> None:
    """A transient retrieve error is retried in place, never re-submitting."""
    retrieve = AsyncMock(
        side_effect=[
            APIConnectionError(request=MagicMock()),
            _execution("completed"),
        ]
    )
    client = _poll_client(retrieve)
    result = await poll_execution(
        client, "dbx-1", "exec-1", timeout=None, last_n="9999"
    )
    assert result.status == "completed"
    assert retrieve.await_count == 2


@pytest.mark.asyncio
async def test_poll_execution_reraises_non_retryable_error() -> None:
    response = httpx.Response(404, request=httpx.Request("GET", "https://example"))
    retrieve = AsyncMock(
        side_effect=NotFoundError("gone", response=response, body=None)
    )
    client = _poll_client(retrieve)
    with pytest.raises(NotFoundError):
        await poll_execution(client, "dbx-1", "exec-1", timeout=None, last_n="9999")


@pytest.mark.asyncio
async def test_shutdown_devbox_retries_transient_error() -> None:
    """A transient API error on shutdown is retried rather than surfaced."""
    call_count = 0

    async def _shutdown(_devbox_id: str) -> None:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise APIConnectionError(request=MagicMock())

    client = MagicMock()
    client.devboxes = MagicMock()
    client.devboxes.shutdown = AsyncMock(side_effect=_shutdown)

    await shutdown_devbox(client, "dbx-1")

    assert call_count == 2


@pytest.mark.asyncio
async def test_shutdown_devbox_does_not_retry_not_found() -> None:
    """NotFoundError is permanent — it propagates so callers treat it as gone."""
    response = httpx.Response(404, request=httpx.Request("POST", "https://example"))
    client = MagicMock()
    client.devboxes = MagicMock()
    client.devboxes.shutdown = AsyncMock(
        side_effect=NotFoundError("gone", response=response, body=None)
    )

    with pytest.raises(NotFoundError):
        await shutdown_devbox(client, "dbx-1")

    client.devboxes.shutdown.assert_awaited_once()
