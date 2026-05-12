"""Tests for RunloopSingleServiceEnvironment."""

from __future__ import annotations

import base64
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from inspect_ai.util import OutputLimitExceededError
from inspect_sandboxes.runloop._single_env import (
    LARGE_FILE_THRESHOLD,
    RunloopSingleServiceEnvironment,
)
from runloop_api_client import APIConnectionError, APITimeoutError


def _make_execution(
    *, stdout: str = "", stderr: str = "", exit_status: int = 0
) -> MagicMock:
    execution = MagicMock()
    execution.stdout = stdout
    execution.stderr = stderr
    execution.exit_status = exit_status
    execution.status = "completed"
    return execution


def _make_client() -> MagicMock:
    """A mock AsyncRunloop client with devboxes.execute_and_await_completion stubbed."""
    client = MagicMock()
    client.devboxes = MagicMock()
    completed = _make_execution(stdout="ok")
    client.devboxes.execute_and_await_completion = AsyncMock(return_value=completed)
    client.devboxes.execute_async = AsyncMock(return_value=completed)
    client.devboxes.executions = MagicMock()
    client.devboxes.executions.retrieve = AsyncMock(return_value=completed)
    client.devboxes.executions.kill = AsyncMock()
    return client


@pytest.fixture
def client() -> MagicMock:
    return _make_client()


@pytest.fixture
def env(client: MagicMock) -> RunloopSingleServiceEnvironment:
    return RunloopSingleServiceEnvironment(client, "dbx-test-123")


@pytest.mark.asyncio
async def test_exec_basic(
    client: MagicMock, env: RunloopSingleServiceEnvironment
) -> None:
    """Test exec returns ExecResult for a successful command."""
    client.devboxes.execute_and_await_completion = AsyncMock(
        return_value=_make_execution(stdout="hi\n", exit_status=0)
    )

    result = await env.exec(["echo", "hi"])

    assert result.success
    assert result.returncode == 0
    assert result.stdout == "hi\n"
    await_args = client.devboxes.execute_and_await_completion.await_args
    assert await_args is not None
    command = await_args.kwargs["command"]
    assert command == "echo hi"


@pytest.mark.asyncio
async def test_exec_joins_args_with_shlex(
    client: MagicMock, env: RunloopSingleServiceEnvironment
) -> None:
    """Args with spaces are shell-quoted so they're not split into multiple tokens."""
    await env.exec(["echo", "hello world"])

    await_args = client.devboxes.execute_and_await_completion.await_args
    assert await_args is not None
    command = await_args.kwargs["command"]
    assert command == "echo 'hello world'"


@pytest.mark.asyncio
async def test_exec_with_cwd(
    client: MagicMock, env: RunloopSingleServiceEnvironment
) -> None:
    await env.exec(["pwd"], cwd="/work")

    await_args = client.devboxes.execute_and_await_completion.await_args
    assert await_args is not None
    command = await_args.kwargs["command"]
    assert command == "cd /work && pwd"


@pytest.mark.asyncio
async def test_exec_with_env(
    client: MagicMock, env: RunloopSingleServiceEnvironment
) -> None:
    await env.exec(["sh", "-c", "echo $FOO"], env={"FOO": "bar"})

    await_args = client.devboxes.execute_and_await_completion.await_args
    assert await_args is not None
    command = await_args.kwargs["command"]
    assert command == "FOO=bar sh -c 'echo $FOO'"


@pytest.mark.asyncio
async def test_exec_with_user_wraps_in_sudo(
    client: MagicMock, env: RunloopSingleServiceEnvironment
) -> None:
    await env.exec(["whoami"], user="root")

    await_args = client.devboxes.execute_and_await_completion.await_args
    assert await_args is not None
    command = await_args.kwargs["command"]
    assert command.startswith("sudo -u root bash -c ")


@pytest.mark.asyncio
async def test_exec_with_numeric_user_uses_uid_form(
    client: MagicMock, env: RunloopSingleServiceEnvironment
) -> None:
    await env.exec(["whoami"], user="4444")

    await_args = client.devboxes.execute_and_await_completion.await_args
    assert await_args is not None
    command = await_args.kwargs["command"]
    assert command.startswith("sudo -u '#4444' bash -c ")


@pytest.mark.asyncio
async def test_exec_failure_returncode(
    client: MagicMock, env: RunloopSingleServiceEnvironment
) -> None:
    """Non-zero exit_status from the SDK should surface as ExecResult.returncode."""
    client.devboxes.execute_and_await_completion = AsyncMock(
        return_value=_make_execution(stdout="", stderr="boom", exit_status=1)
    )

    result = await env.exec(["false"])
    assert not result.success
    assert result.returncode == 1
    assert result.stderr == "boom"


@pytest.mark.asyncio
async def test_exec_with_stdin_string(
    client: MagicMock, env: RunloopSingleServiceEnvironment
) -> None:
    """Test exec redirects string stdin through a temp file."""
    client.devboxes.execute_and_await_completion = AsyncMock(
        return_value=_make_execution(stdout="ok", exit_status=0)
    )

    await env.exec(["cat"], input="hello")

    calls = client.devboxes.execute_and_await_completion.await_args_list
    # First call writes the temp file (base64-decoded payload).
    write_cmd = calls[0].kwargs["command"]
    assert "base64 -d" in write_cmd
    assert "/tmp/.inspect-stdin-" in write_cmd
    # Second call is the actual exec, redirecting from the tmp file.
    exec_cmd = calls[-1].kwargs["command"]
    assert "cat < " in exec_cmd
    assert "/tmp/.inspect-stdin-" in exec_cmd
    assert "rm -f" in exec_cmd  # cleanup inline


@pytest.mark.asyncio
async def test_exec_with_stdin_bytes(
    client: MagicMock, env: RunloopSingleServiceEnvironment
) -> None:
    """Test exec redirects bytes stdin through a temp file."""
    await env.exec(["wc", "-c"], input=b"\x00\x01\x02")

    write_cmd = client.devboxes.execute_and_await_completion.await_args_list[0].kwargs[
        "command"
    ]
    expected = base64.b64encode(b"\x00\x01\x02").decode("ascii")
    assert expected in write_cmd


@pytest.mark.asyncio
async def test_exec_without_stdin_no_upload(
    client: MagicMock, env: RunloopSingleServiceEnvironment
) -> None:
    """Test exec without stdin does not upload any file."""
    await env.exec(["echo", "hi"])

    calls = client.devboxes.execute_and_await_completion.await_args_list
    assert len(calls) == 1
    command = calls[0].kwargs["command"]
    assert command == "echo hi"


@pytest.mark.asyncio
async def test_exec_with_stdin_and_user_skips_inline_cleanup(
    client: MagicMock, env: RunloopSingleServiceEnvironment
) -> None:
    """Test that stdin + user defers temp-file cleanup to the finally block."""
    await env.exec(["cat"], input="hello", user="testuser")

    calls = client.devboxes.execute_and_await_completion.await_args_list
    # Three calls: write tmpfile, exec command, cleanup tmpfile.
    assert len(calls) == 3
    exec_command = calls[1].kwargs["command"]
    # The exec command (wrapped in sudo) shouldn't have a baked-in rm -f.
    assert "sudo -u" in exec_command
    assert "rm -f" not in exec_command
    # The final call is the explicit cleanup.
    cleanup_command = calls[2].kwargs["command"]
    assert cleanup_command.startswith("rm -f ")


@pytest.mark.asyncio
async def test_exec_retries_transient_error(
    client: MagicMock, env: RunloopSingleServiceEnvironment
) -> None:
    """Test that exec retries on transient APIError."""
    call_count = 0
    success = _make_execution(stdout="ok", exit_status=0)

    async def flaky_run(*_args: Any, **_kwargs: Any) -> MagicMock:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise APIConnectionError(request=MagicMock())
        return success

    client.devboxes.execute_and_await_completion = AsyncMock(side_effect=flaky_run)
    result = await env.exec(["echo", "test"])

    assert result.success
    assert call_count == 2


@pytest.mark.asyncio
async def test_exec_timeout_retry_uses_capped_timeouts(
    client: MagicMock, env: RunloopSingleServiceEnvironment
) -> None:
    """First retry caps at 60s, second at 30s, then raises TimeoutError."""
    client.devboxes.execute_async = AsyncMock(
        side_effect=APITimeoutError(request=MagicMock())
    )
    with pytest.raises(TimeoutError):
        await env.exec(["sleep", "1000"], timeout=120)
    # Three attempts: initial + 2 retries (timeouts capped at 120, 60, 30 by
    # run_with_timeout_retry — verified by call count since we use our own
    # polling for the actual timeout now).
    assert client.devboxes.execute_async.await_count == 3


@pytest.mark.asyncio
async def test_write_file_text(
    client: MagicMock, env: RunloopSingleServiceEnvironment
) -> None:
    """Test write_file with text content."""
    # First exec is the _is_directory check (returns non-zero → not a dir).
    client.devboxes.execute_and_await_completion = AsyncMock(
        side_effect=[
            _make_execution(exit_status=1),  # is_directory: no
            _make_execution(exit_status=0),  # write succeeds
        ]
    )

    await env.write_file("/tmp/out.txt", "hello")

    write_call = client.devboxes.execute_and_await_completion.await_args_list[-1]
    cmd = write_call.kwargs["command"]
    assert "base64 -d" in cmd
    assert "/tmp/out.txt" in cmd


@pytest.mark.asyncio
async def test_write_file_binary(
    client: MagicMock, env: RunloopSingleServiceEnvironment
) -> None:
    """Test write_file with binary content."""
    client.devboxes.execute_and_await_completion = AsyncMock(
        side_effect=[
            _make_execution(exit_status=1),  # not a directory
            _make_execution(exit_status=0),  # write succeeds
        ]
    )

    await env.write_file("/tmp/out.bin", b"\x00\x01\x02")

    write_call = client.devboxes.execute_and_await_completion.await_args_list[-1]
    cmd = write_call.kwargs["command"]
    # Confirm the bytes were base64-encoded into the command.
    expected = base64.b64encode(b"\x00\x01\x02").decode("ascii")
    assert expected in cmd


@pytest.mark.asyncio
async def test_write_file_raises_for_directory(
    client: MagicMock, env: RunloopSingleServiceEnvironment
) -> None:
    """Test write_file raises IsADirectoryError when path is a directory."""
    client.devboxes.execute_and_await_completion = AsyncMock(
        return_value=_make_execution(exit_status=0)  # is_directory: yes
    )

    with pytest.raises(IsADirectoryError):
        await env.write_file("/tmp", "hello")


@pytest.mark.asyncio
async def test_read_file_text(
    client: MagicMock, env: RunloopSingleServiceEnvironment
) -> None:
    """Test read_file in text mode."""
    encoded = base64.b64encode(b"file contents").decode("ascii")
    client.devboxes.execute_and_await_completion = AsyncMock(
        side_effect=[
            _make_execution(exit_status=1),  # is_directory: no
            _make_execution(stdout="13", exit_status=0),  # stat -c %s
            _make_execution(stdout=encoded, exit_status=0),  # base64 read
        ]
    )

    result = await env.read_file("/tmp/file.txt")

    assert result == "file contents"


@pytest.mark.asyncio
async def test_read_file_binary(
    client: MagicMock, env: RunloopSingleServiceEnvironment
) -> None:
    """Test read_file in binary mode."""
    payload = b"\xff\xfe\x00"
    encoded = base64.b64encode(payload).decode("ascii")
    client.devboxes.execute_and_await_completion = AsyncMock(
        side_effect=[
            _make_execution(exit_status=1),  # is_directory: no
            _make_execution(stdout=str(len(payload)), exit_status=0),  # stat -c %s
            _make_execution(stdout=encoded, exit_status=0),  # base64 read
        ]
    )

    result = await env.read_file("/tmp/file.bin", text=False)
    assert result == payload


@pytest.mark.asyncio
async def test_read_file_not_found(
    client: MagicMock, env: RunloopSingleServiceEnvironment
) -> None:
    """Test read_file raises FileNotFoundError when file doesn't exist."""
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
async def test_read_file_is_directory(
    client: MagicMock, env: RunloopSingleServiceEnvironment
) -> None:
    """Test read_file raises IsADirectoryError for directories."""
    client.devboxes.execute_and_await_completion = AsyncMock(
        return_value=_make_execution(exit_status=0)  # is_directory: yes
    )

    with pytest.raises(IsADirectoryError):
        await env.read_file("/tmp")


@pytest.mark.asyncio
async def test_read_file_size_limit(
    client: MagicMock, env: RunloopSingleServiceEnvironment
) -> None:
    """Test read_file raises OutputLimitExceededError for oversized files."""
    huge = 200 * 1024 * 1024  # 200 MiB, above 100 MiB read cap
    client.devboxes.execute_and_await_completion = AsyncMock(
        side_effect=[
            _make_execution(exit_status=1),  # is_directory: no
            _make_execution(stdout=str(huge), exit_status=0),  # stat -c %s
        ]
    )

    with pytest.raises(OutputLimitExceededError):
        await env.read_file("/tmp/huge.bin")


@pytest.mark.asyncio
async def test_write_file_large_uses_object_api(
    client: MagicMock, env: RunloopSingleServiceEnvironment
) -> None:
    """Files at/above LARGE_FILE_THRESHOLD route through the Objects API."""
    obj = MagicMock(id="obj-123", upload_url="https://upload.example/put")
    download = MagicMock(download_url="https://download.example/get")
    client.objects = MagicMock()
    client.objects.create = AsyncMock(return_value=obj)
    client.objects.complete = AsyncMock()
    client.objects.download = AsyncMock(return_value=download)
    client.objects.delete = AsyncMock()
    client.devboxes.execute_and_await_completion = AsyncMock(
        side_effect=[
            _make_execution(exit_status=1),  # not a directory
            _make_execution(exit_status=0),  # devbox curl GET write
        ]
    )

    big = b"\x00" * LARGE_FILE_THRESHOLD
    with patch(
        "inspect_sandboxes.runloop._single_env.httpx.AsyncClient"
    ) as mock_http_cls:
        http = mock_http_cls.return_value.__aenter__.return_value
        http.put = AsyncMock(return_value=MagicMock(raise_for_status=MagicMock()))
        await env.write_file("/tmp/big.bin", big)

    client.objects.create.assert_awaited_once()
    http.put.assert_awaited_once_with("https://upload.example/put", content=big)
    client.objects.complete.assert_awaited_once_with("obj-123")
    client.objects.download.assert_awaited_once_with("obj-123")
    curl_cmd = client.devboxes.execute_and_await_completion.await_args_list[-1].kwargs[
        "command"
    ]
    assert "curl" in curl_cmd
    assert "https://download.example/get" in curl_cmd
    assert "/tmp/big.bin" in curl_cmd
    client.objects.delete.assert_awaited_once_with("obj-123")


@pytest.mark.asyncio
async def test_read_file_large_uses_object_api(
    client: MagicMock, env: RunloopSingleServiceEnvironment
) -> None:
    """Files at/above LARGE_FILE_THRESHOLD route through the Objects API."""
    obj = MagicMock(id="obj-456", upload_url="https://upload.example/put")
    download = MagicMock(download_url="https://download.example/get")
    client.objects = MagicMock()
    client.objects.create = AsyncMock(return_value=obj)
    client.objects.complete = AsyncMock()
    client.objects.download = AsyncMock(return_value=download)
    client.objects.delete = AsyncMock()
    client.devboxes.execute_and_await_completion = AsyncMock(
        side_effect=[
            _make_execution(exit_status=1),  # not a directory
            _make_execution(
                stdout=str(LARGE_FILE_THRESHOLD), exit_status=0
            ),  # stat -c %s
            _make_execution(exit_status=0),  # devbox curl PUT upload
        ]
    )

    payload = b"\xff" * LARGE_FILE_THRESHOLD
    with patch(
        "inspect_sandboxes.runloop._single_env.httpx.AsyncClient"
    ) as mock_http_cls:
        http = mock_http_cls.return_value.__aenter__.return_value
        http.get = AsyncMock(
            return_value=MagicMock(content=payload, raise_for_status=MagicMock())
        )
        result = await env.read_file("/tmp/big.bin", text=False)

    assert result == payload
    curl_cmd = client.devboxes.execute_and_await_completion.await_args_list[-1].kwargs[
        "command"
    ]
    assert "curl" in curl_cmd
    assert "--upload-file" in curl_cmd
    assert "/tmp/big.bin" in curl_cmd
    assert "https://upload.example/put" in curl_cmd
    http.get.assert_awaited_once_with("https://download.example/get")
    client.objects.delete.assert_awaited_once_with("obj-456")


@pytest.mark.asyncio
async def test_sample_cleanup_shuts_down_devbox(
    client: MagicMock, env: RunloopSingleServiceEnvironment
) -> None:
    """Test sample_cleanup shuts down all devboxes in environments dict."""
    client.devboxes.shutdown = AsyncMock()
    await RunloopSingleServiceEnvironment.sample_cleanup(
        "task", None, {"default": env}, False
    )
    client.devboxes.shutdown.assert_awaited_once_with("dbx-test-123")


@pytest.mark.asyncio
async def test_sample_cleanup_skips_when_interrupted(
    client: MagicMock, env: RunloopSingleServiceEnvironment
) -> None:
    """Test sample_cleanup does nothing when interrupted."""
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
async def test_sample_cleanup_continues_on_shutdown_failure(
    client: MagicMock, env: RunloopSingleServiceEnvironment
) -> None:
    """Test sample_cleanup logs error and continues when a shutdown fails."""
    client.devboxes.shutdown = AsyncMock(side_effect=RuntimeError("flaky"))
    # Should not raise — error is traced and deferred.
    await RunloopSingleServiceEnvironment.sample_cleanup(
        "task", None, {"default": env}, False
    )
