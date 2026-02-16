"""Tests for Modal sandbox environment implementation."""

from collections.abc import AsyncGenerator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import modal
import pytest
import pytest_asyncio
from inspect_ai.util import (
    ComposeConfig,
    ComposeService,
    ExecResult,
    SandboxEnvironment,
    SandboxEnvironmentLimits,
)
from inspect_ai.util._sandbox.limits import OutputLimitExceededError
from inspect_ai.util._sandbox.self_check import self_check
from inspect_sandboxes.modal._modal import (
    ModalSandboxEnvironment,
    running_sandboxes,
    sandbox_cleanup_startup,
)


@pytest.fixture
def mock_modal_app() -> MagicMock:
    """Create a mock Modal app."""
    app = MagicMock(spec=modal.App)
    app.object_id = "app-123"
    return app


@pytest.fixture
def mock_modal_sandbox() -> MagicMock:
    """Create a mock Modal sandbox."""
    sandbox = MagicMock(spec=modal.Sandbox)
    sandbox.object_id = "sb-test-123"

    # Setup async mock methods
    async def mock_exec(*args: Any, **kwargs: Any) -> MagicMock:
        process = MagicMock()
        process.returncode = 0
        process.stdout = MagicMock()
        process.stdout.read = AsyncMock(return_value="output")
        process.stderr = MagicMock()
        process.stderr.read = AsyncMock(return_value="")
        process.stdin = MagicMock()
        process.stdin.write = MagicMock()
        process.stdin.write_eof = MagicMock()
        process.stdin.drain = AsyncMock()
        process.wait = AsyncMock()
        return process

    sandbox.exec = MagicMock()
    sandbox.exec.aio = mock_exec
    sandbox.open = MagicMock()
    sandbox.mkdir = MagicMock()
    sandbox.mkdir.aio = AsyncMock()
    sandbox.terminate = MagicMock()
    sandbox.terminate.aio = AsyncMock()

    return sandbox


@pytest.fixture
def sandbox_env(mock_modal_sandbox: MagicMock) -> ModalSandboxEnvironment:
    """Create a ModalSandboxEnvironment instance."""
    return ModalSandboxEnvironment(mock_modal_sandbox)


@pytest.mark.asyncio
async def test_full_lifecycle(
    mock_modal_app: MagicMock,
    mock_modal_sandbox: MagicMock,
) -> None:
    """Test the full sandbox lifecycle: task_init → sample_init → sample_cleanup → task_cleanup."""

    async def mock_from_id(sandbox_id: str) -> MagicMock:
        return mock_modal_sandbox

    with (
        patch.object(
            ModalSandboxEnvironment,
            "_lookup_app",
            new_callable=AsyncMock,
            return_value=mock_modal_app,
        ),
        patch.object(
            ModalSandboxEnvironment,
            "_create_sandbox",
            new_callable=AsyncMock,
            return_value=mock_modal_sandbox,
        ),
        patch.object(
            ModalSandboxEnvironment, "_terminate_sandbox", new_callable=AsyncMock
        ) as mock_terminate,
        patch("modal.Sandbox.from_id") as mock_from_id_class,
    ):
        mock_from_id_class.aio = mock_from_id

        # task_init: initialize tracking
        await ModalSandboxEnvironment.task_init("test_task", None)
        assert running_sandboxes() == []

        # sample_init: create sandbox and track it
        envs = await ModalSandboxEnvironment.sample_init("test_task", None, {})
        assert len(running_sandboxes()) == 1
        assert running_sandboxes()[0] == "sb-test-123"
        assert "default" in envs

        # sample_cleanup: attempt cleanup but don't remove from tracking
        await ModalSandboxEnvironment.sample_cleanup("test_task", None, envs, False)
        assert mock_terminate.called
        assert len(running_sandboxes()) == 1  # Still tracked

        # task_cleanup: final cleanup and clear tracking
        mock_terminate.reset_mock()
        await ModalSandboxEnvironment.task_cleanup("test_task", None, cleanup=True)
        assert mock_terminate.called
        assert running_sandboxes() == []


@pytest.mark.parametrize("interrupted", [True, False])
@pytest.mark.asyncio
async def test_sample_cleanup(
    interrupted: bool,
    mock_modal_sandbox: MagicMock,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test sample_cleanup with interrupted=True/False."""
    env = ModalSandboxEnvironment(mock_modal_sandbox)

    with patch.object(
        ModalSandboxEnvironment,
        "_terminate_sandbox",
        new_callable=AsyncMock,
        side_effect=Exception("Termination failed"),
    ):
        await ModalSandboxEnvironment.sample_cleanup(
            "test_task", None, {"default": env}, interrupted
        )

        if interrupted:
            # Should silently ignore errors when interrupted
            assert "Error terminating" not in caplog.text
        else:
            # Should log warnings when not interrupted
            assert "Error terminating" in caplog.text
            assert "Will retry in task_cleanup" in caplog.text


@pytest.mark.parametrize(
    ("cleanup", "num_sandboxes", "num_failures"),
    [
        (False, 0, 0),  # cleanup=False, no-op
        (True, 0, 0),  # No sandboxes to clean
        (True, 1, 0),  # Single sandbox, success
        (True, 3, 0),  # Multiple sandboxes, all success
        (True, 3, 1),  # Multiple sandboxes, one failure
        (True, 3, 3),  # Multiple sandboxes, all failures
    ],
)
@pytest.mark.asyncio
async def test_task_cleanup(
    cleanup: bool,
    num_sandboxes: int,
    num_failures: int,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test task_cleanup with various scenarios."""
    sandbox_cleanup_startup()

    # Add sandboxes to tracking
    for i in range(num_sandboxes):
        running_sandboxes().append(f"sb-{i}")

    # Mock Sandbox.from_id and terminate
    async def from_id_mock(sandbox_id: str) -> MagicMock:
        sandbox = MagicMock(spec=modal.Sandbox)
        sandbox.object_id = sandbox_id
        return sandbox

    failure_ids = {f"sb-{i}" for i in range(num_failures)}

    async def terminate_mock(sandbox: MagicMock) -> None:
        if sandbox.object_id in failure_ids:
            raise Exception(f"Failed to terminate {sandbox.object_id}")

    with (
        patch("modal.Sandbox.from_id") as mock_from_id,
        patch.object(ModalSandboxEnvironment, "_terminate_sandbox", new=terminate_mock),
    ):
        mock_from_id.aio = from_id_mock

        await ModalSandboxEnvironment.task_cleanup("test_task", None, cleanup)

        if not cleanup:
            # Should be no-op
            assert len(running_sandboxes()) == num_sandboxes
        else:
            # Should always clear tracking
            assert running_sandboxes() == []

            if num_failures > 0:
                assert "Failed to cleanup" in caplog.text
                for i in range(num_failures):
                    assert f"sb-{i}" in caplog.text


@pytest.mark.parametrize(
    ("config_type", "should_raise"),
    [
        ("none", False),  # No config, use default
        ("dockerfile", False),  # Dockerfile path
        ("compose_yaml", False),  # Compose YAML file
        ("compose_config", False),  # ComposeConfig object
        ("invalid", True),  # Invalid config type
    ],
    ids=["none", "dockerfile", "compose_yaml", "compose_config", "invalid"],
)
@pytest.mark.asyncio
async def test_sample_init_configs(
    config_type: str,
    should_raise: bool,
    mock_modal_app: MagicMock,
    mock_modal_sandbox: MagicMock,
    tmp_path: Any,
) -> None:
    """Test sample_init with different config types."""
    sandbox_cleanup_startup()

    # Create the appropriate config based on config_type
    config: Any
    if config_type == "none":
        config = None
    elif config_type == "dockerfile":
        dockerfile = tmp_path / "Dockerfile"
        dockerfile.write_text("FROM python:3.12\nRUN echo 'test'")
        config = str(dockerfile)
    elif config_type == "compose_yaml":
        compose_file = tmp_path / "compose.yaml"
        compose_file.write_text("""
services:
  default:
    image: python:3.12
    environment:
      - TEST_VAR=value
""")
        config = str(compose_file)
    elif config_type == "compose_config":
        config = ComposeConfig(
            services={
                "default": ComposeService(
                    image="python:3.12",
                    environment={"TEST_VAR": "value"},
                )
            }
        )
    else:  # invalid
        config = 12345

    with (
        patch.object(
            ModalSandboxEnvironment,
            "_lookup_app",
            new_callable=AsyncMock,
            return_value=mock_modal_app,
        ),
        patch.object(
            ModalSandboxEnvironment,
            "_create_sandbox",
            new_callable=AsyncMock,
            return_value=mock_modal_sandbox,
        ),
        patch("modal.Image.from_dockerfile") as mock_from_dockerfile,
        patch("modal.Image.from_registry") as mock_from_registry,
    ):
        mock_from_dockerfile.return_value = MagicMock(spec=modal.Image)
        mock_from_registry.return_value = MagicMock(spec=modal.Image)

        if should_raise:
            with pytest.raises(ValueError, match="Unrecognized config"):
                await ModalSandboxEnvironment.sample_init("test_task", config, {})
        else:
            result = await ModalSandboxEnvironment.sample_init("test_task", config, {})
            assert "default" in result

            if config_type == "dockerfile":
                mock_from_dockerfile.assert_called_once()
            elif config_type in ("compose_yaml", "compose_config"):
                mock_from_registry.assert_called_once_with("python:3.12")


@pytest.mark.parametrize(
    ("cmd", "returncode", "timeout"),
    [
        (["echo", "hello"], 0, None),  # Basic success
        (["false"], 1, None),  # Command failure
        (["sleep", "1"], 0, 5),  # With timeout
    ],
)
@pytest.mark.asyncio
async def test_exec_variations(
    cmd: list[str],
    returncode: int,
    timeout: int | None,
    sandbox_env: ModalSandboxEnvironment,
) -> None:
    """Test exec with various parameter combinations."""

    async def mock_exec(*args: Any, **kwargs: Any) -> MagicMock:
        process = MagicMock()
        process.returncode = returncode
        process.stdout = MagicMock()
        process.stdout.read = AsyncMock(return_value="output")
        process.stderr = MagicMock()
        process.stderr.read = AsyncMock(return_value="")
        process.stdin = MagicMock()
        process.stdin.write = MagicMock()
        process.stdin.write_eof = MagicMock()
        process.stdin.drain = AsyncMock()
        process.wait = AsyncMock()
        return process

    sandbox_env.sandbox.exec = MagicMock()
    sandbox_env.sandbox.exec.aio = mock_exec

    result = await sandbox_env.exec(cmd, timeout=timeout)

    assert isinstance(result, ExecResult)
    assert result.success == (returncode == 0)
    assert result.returncode == returncode


@pytest.mark.parametrize(
    ("param_name", "param_value", "expected_warning"),
    [
        ("user", "root", "user' parameter is ignored"),
        ("concurrency", False, "concurrency' parameter is not supported"),
        ("cwd", "relative/path", "Relative path"),
    ],
)
@pytest.mark.asyncio
async def test_exec_warnings(
    param_name: str,
    param_value: Any,
    expected_warning: str,
    sandbox_env: ModalSandboxEnvironment,
) -> None:
    """Test that exec issues warnings for unsupported parameters."""
    kwargs: dict[str, Any] = {"cmd": ["echo", "test"]}
    kwargs[param_name] = param_value

    with pytest.warns(UserWarning, match=expected_warning):
        await sandbox_env.exec(**kwargs)


@pytest.mark.asyncio
async def test_write_file_text(sandbox_env: ModalSandboxEnvironment) -> None:
    """Test write_file with text content."""
    mock_file = AsyncMock()
    mock_context = AsyncMock()
    mock_context.__aenter__ = AsyncMock(return_value=mock_file)
    mock_context.__aexit__ = AsyncMock(return_value=None)

    sandbox_env.sandbox.open = MagicMock()
    sandbox_env.sandbox.open.aio = AsyncMock(return_value=mock_context)
    sandbox_env.sandbox.mkdir = MagicMock()
    sandbox_env.sandbox.mkdir.aio = AsyncMock()

    await sandbox_env.write_file("/test.txt", "text content")

    sandbox_env.sandbox.open.aio.assert_called_once_with("/test.txt", "w")
    mock_file.write.aio.assert_called_once_with("text content")


@pytest.mark.asyncio
async def test_write_file_binary(sandbox_env: ModalSandboxEnvironment) -> None:
    """Test write_file with binary content."""
    mock_file = AsyncMock()
    mock_context = AsyncMock()
    mock_context.__aenter__ = AsyncMock(return_value=mock_file)
    mock_context.__aexit__ = AsyncMock(return_value=None)

    sandbox_env.sandbox.open = MagicMock()
    sandbox_env.sandbox.open.aio = AsyncMock(return_value=mock_context)
    sandbox_env.sandbox.mkdir = MagicMock()
    sandbox_env.sandbox.mkdir.aio = AsyncMock()

    await sandbox_env.write_file("/test.bin", b"binary content")

    sandbox_env.sandbox.open.aio.assert_called_once_with("/test.bin", "wb")
    mock_file.write.aio.assert_called_once_with(b"binary content")


@pytest.mark.asyncio
async def test_read_file_text(sandbox_env: ModalSandboxEnvironment) -> None:
    """Test read_file in text mode."""
    mock_file = AsyncMock()
    mock_file.read.aio = AsyncMock(return_value=b"test content")
    mock_context = AsyncMock()
    mock_context.__aenter__ = AsyncMock(return_value=mock_file)
    mock_context.__aexit__ = AsyncMock(return_value=None)

    with (
        patch.object(
            sandbox_env, "_is_directory", new_callable=AsyncMock, return_value=False
        ),
        patch.object(sandbox_env, "_get_file_size", return_value=100),
        patch.object(sandbox_env.sandbox, "open") as mock_open,
    ):
        mock_open.aio = AsyncMock(return_value=mock_context)
        result = await sandbox_env.read_file("/test.txt", text=True)
        assert isinstance(result, str)
        assert result == "test content"


@pytest.mark.asyncio
async def test_read_file_binary(sandbox_env: ModalSandboxEnvironment) -> None:
    """Test read_file in binary mode."""
    mock_file = AsyncMock()
    mock_file.read.aio = AsyncMock(return_value=b"test content")
    mock_context = AsyncMock()
    mock_context.__aenter__ = AsyncMock(return_value=mock_file)
    mock_context.__aexit__ = AsyncMock(return_value=None)

    with (
        patch.object(
            sandbox_env, "_is_directory", new_callable=AsyncMock, return_value=False
        ),
        patch.object(sandbox_env, "_get_file_size", return_value=100),
        patch.object(sandbox_env.sandbox, "open") as mock_open,
    ):
        mock_open.aio = AsyncMock(return_value=mock_context)
        result = await sandbox_env.read_file("/test.bin", text=False)
        assert isinstance(result, bytes)
        assert result == b"test content"


@pytest.mark.asyncio
async def test_read_file_size_limit(sandbox_env: ModalSandboxEnvironment) -> None:
    """Test read_file with file exceeding size limit."""
    with (
        patch.object(
            sandbox_env, "_is_directory", new_callable=AsyncMock, return_value=False
        ),
        patch.object(
            sandbox_env,
            "_get_file_size",
            return_value=SandboxEnvironmentLimits.MAX_READ_FILE_SIZE + 1,
        ),
    ):
        with pytest.raises(OutputLimitExceededError):
            await sandbox_env.read_file("/large.txt")


@pytest.mark.parametrize(
    ("success", "sandbox_id"),
    [
        (True, "sb-123"),
        (False, "sb-456"),
    ],
)
@pytest.mark.asyncio
async def test_cli_cleanup_single(
    success: bool,
    sandbox_id: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Test CLI cleanup of a single sandbox."""
    mock_sandbox = MagicMock(spec=modal.Sandbox)
    mock_sandbox.object_id = sandbox_id

    async def mock_from_id(sid: str) -> MagicMock:
        return mock_sandbox

    with (
        patch("modal.Sandbox.from_id") as mock_from_id_class,
        patch.object(
            ModalSandboxEnvironment, "_terminate_sandbox", new_callable=AsyncMock
        ) as mock_terminate,
    ):
        mock_from_id_class.aio = mock_from_id

        if not success:
            mock_terminate.side_effect = Exception("Termination failed")

        if success:
            await ModalSandboxEnvironment.cli_cleanup(sandbox_id)
            captured = capsys.readouterr()
            assert "Successfully terminated" in captured.out
        else:
            with pytest.raises(SystemExit) as exc_info:
                await ModalSandboxEnvironment.cli_cleanup(sandbox_id)
            assert exc_info.value.code == 1
            captured = capsys.readouterr()
            assert "Error terminating" in captured.out


@pytest.mark.parametrize(
    ("sandbox_ids", "fail_ids", "expected_exit"),
    [
        (["sb-001", "sb-002", "sb-003"], set(), False),  # All succeed
        (["sb-001", "sb-002", "sb-003"], {"sb-002"}, True),  # Partial failure
        (["sb-001", "sb-002"], {"sb-001", "sb-002"}, True),  # All fail
    ],
    ids=["all_succeed", "partial_failure", "all_fail"],
)
@pytest.mark.asyncio
async def test_cli_cleanup_bulk_with_sandboxes(
    sandbox_ids: list[str],
    fail_ids: set[str],
    expected_exit: bool,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Test CLI bulk cleanup with multiple sandboxes."""
    # Create mock sandboxes
    mock_sandboxes: list[MagicMock] = []
    for sid in sandbox_ids:
        mock_sb = MagicMock(spec=modal.Sandbox)
        mock_sb.object_id = sid
        mock_sandboxes.append(mock_sb)

    async def mock_list_generator() -> AsyncGenerator[MagicMock, None]:
        for sb in mock_sandboxes:
            yield sb

    async def mock_terminate(sandbox: modal.Sandbox) -> None:
        if sandbox.object_id in fail_ids:
            raise Exception(f"Failed to terminate {sandbox.object_id}")

    with (
        patch("modal.Sandbox.list") as mock_list,
        patch.object(
            ModalSandboxEnvironment, "_terminate_sandbox", new_callable=AsyncMock
        ) as mock_terminate_patch,
    ):
        mock_list.aio = mock_list_generator
        mock_terminate_patch.side_effect = mock_terminate

        if expected_exit:
            with pytest.raises(SystemExit) as exc_info:
                await ModalSandboxEnvironment.cli_cleanup(None)
            assert exc_info.value.code == 1
        else:
            await ModalSandboxEnvironment.cli_cleanup(None)

        captured = capsys.readouterr()

        # Verify all sandbox IDs are shown in output
        for sid in sandbox_ids:
            assert sid in captured.out

        # Check success/failure messages
        success_count = len(sandbox_ids) - len(fail_ids)
        failure_count = len(fail_ids)

        assert f"Successfully terminated: {success_count}" in captured.out

        if failure_count > 0:
            assert f"Failed to terminate: {failure_count}" in captured.out
            for failed_id in fail_ids:
                assert f"Error terminating sandbox {failed_id}" in captured.out
        else:
            assert "Complete." in captured.out


@pytest.mark.asyncio
async def test_cli_cleanup_bulk_no_sandboxes(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Test CLI bulk cleanup with no sandboxes."""

    async def mock_list_generator():
        # Create an async generator that yields nothing
        for _ in []:
            yield

    with patch("modal.Sandbox.list") as mock_list:
        mock_list.aio = mock_list_generator

        await ModalSandboxEnvironment.cli_cleanup(None)
        captured = capsys.readouterr()
        assert "No Modal sandboxes found" in captured.out


@pytest.mark.asyncio
async def test_get_sandbox_id_none() -> None:
    """Test _get_sandbox_id with None."""
    assert ModalSandboxEnvironment._get_sandbox_id(None) == "unknown"


@pytest_asyncio.fixture
async def modal_sandbox_environment() -> AsyncGenerator[SandboxEnvironment, None]:
    """Create a real Modal sandbox environment for integration testing."""
    sandbox_cleanup_startup()

    envs = await ModalSandboxEnvironment.sample_init("test_self_check", None, {})
    sandbox_env = envs["default"]

    yield sandbox_env

    try:
        await ModalSandboxEnvironment.sample_cleanup(
            "test_self_check", None, envs, False
        )
        await ModalSandboxEnvironment.task_cleanup(
            "test_self_check", None, cleanup=True
        )
    except Exception as e:
        print(f"Cleanup error: {e}")


def check_results_of_self_check(
    results: dict[str, bool | str], known_failures: list[str]
) -> None:
    """Check self_check results, ignoring known failures."""
    passed = []
    failed = []
    known_failed = []

    for test_name, result in results.items():
        if result is True:
            passed.append(test_name)
        elif test_name in known_failures:
            known_failed.append(test_name)
        else:
            failed.append((test_name, result))

    if failed:
        failure_details = "\n".join([f"  {name}: {error}" for name, error in failed])
        raise AssertionError(
            f"{len(failed)} unexpected test(s) failed:\n{failure_details}"
        )


@pytest.mark.asyncio
@pytest.mark.integration
async def test_self_check(modal_sandbox_environment: ModalSandboxEnvironment) -> None:
    """Run Inspect AI's self-check suite against Modal sandbox."""
    known_failures = [
        "test_read_file_not_allowed",  # user is root, so this doesn't work
        "test_exec_as_user",  # unsupported
        "test_exec_as_nonexistent_user",  # unsupported
        "test_write_text_file_without_permissions",  # user is root
        "test_write_binary_file_without_permissions",  # user is root
        "test_exec_permission_error",  # user is root
    ]

    results = await self_check(modal_sandbox_environment)
    check_results_of_self_check(results, known_failures)
