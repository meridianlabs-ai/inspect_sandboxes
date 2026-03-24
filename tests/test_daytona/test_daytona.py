"""Tests for Daytona sandbox environment implementation."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from daytona_sdk import DaytonaError, DaytonaTimeoutError
from inspect_ai.util import (
    ComposeConfig,
    ComposeService,
    ExecResult,
    OutputLimitExceededError,
    SandboxEnvironmentLimits,
)
from inspect_sandboxes.daytona._daytona import (
    INSPECT_SANDBOX_LABEL,
    DaytonaSandboxEnvironment,
    _daytona_client,
    _init_context,
    _run_id,
    _running_sandboxes,
)


def _assert_has_run_labels(labels: dict[str, str] | None) -> None:
    """Assert labels contain INSPECT_SANDBOX_LABEL entries and an inspect_run_id."""
    assert labels is not None
    for key, value in INSPECT_SANDBOX_LABEL.items():
        assert labels[key] == value
    assert "inspect_run_id" in labels
    assert len(labels["inspect_run_id"]) == 32  # uuid4 hex


def make_mock_sandbox(sandbox_id: str = "sb-test-123") -> MagicMock:
    """Create a mock AsyncSandbox."""
    sandbox = MagicMock()
    sandbox.id = sandbox_id

    # process.exec returns ExecuteResponse
    execute_response = MagicMock()
    execute_response.exit_code = 0
    execute_response.result = "output"
    sandbox.process = MagicMock()
    sandbox.process.exec = AsyncMock(return_value=execute_response)

    # fs methods
    sandbox.fs = MagicMock()
    sandbox.fs.upload_file = AsyncMock()
    sandbox.fs.download_file = AsyncMock(return_value=b"content")
    sandbox.fs.get_file_info = AsyncMock()
    sandbox.fs.create_folder = AsyncMock()

    return sandbox


def make_mock_client(sandbox: MagicMock) -> MagicMock:
    """Create a mock AsyncDaytona client."""
    client = MagicMock()
    client.create = AsyncMock(return_value=sandbox)
    client.get = AsyncMock(return_value=sandbox)
    client.delete = AsyncMock()
    client.close = AsyncMock()

    paginated = MagicMock()
    paginated.items = []
    client.list = AsyncMock(return_value=paginated)

    return client


@pytest.fixture
def mock_sandbox() -> MagicMock:
    """Create a mock AsyncSandbox fixture."""
    return make_mock_sandbox()


@pytest.fixture
def mock_client(mock_sandbox: MagicMock) -> MagicMock:
    """Create a mock AsyncDaytona client fixture."""
    return make_mock_client(mock_sandbox)


@pytest.fixture
def sandbox_env(mock_sandbox: MagicMock) -> DaytonaSandboxEnvironment:
    """Create a DaytonaSandboxEnvironment fixture."""
    return DaytonaSandboxEnvironment(mock_sandbox)


@pytest.mark.asyncio
async def test_full_lifecycle(
    mock_client: MagicMock,
    mock_sandbox: MagicMock,
) -> None:
    """Test the full sandbox lifecycle: task_init → sample_init → sample_cleanup → task_cleanup."""
    with patch(
        "inspect_sandboxes.daytona._daytona.AsyncDaytona", return_value=mock_client
    ):
        # task_init: initialize client only
        await DaytonaSandboxEnvironment.task_init("test_task", None)
        mock_client.create.assert_not_called()

        # sample_init: create sandbox
        envs = await DaytonaSandboxEnvironment.sample_init("test_task", None, {})
        assert "default" in envs
        assert _running_sandboxes.get() == ["sb-test-123"]

        # sample_cleanup
        await DaytonaSandboxEnvironment.sample_cleanup("test_task", None, envs, False)
        mock_client.delete.assert_called_once_with(mock_sandbox)

        # task_cleanup: clean remaining + close client
        mock_client.delete.reset_mock()
        await DaytonaSandboxEnvironment.task_cleanup("test_task", None, cleanup=True)
        assert _running_sandboxes.get() == []
        mock_client.close.assert_called_once()


@pytest.mark.asyncio
async def test_task_init_initializes_context() -> None:
    """Test task_init populates the Daytona client context var and sets running sandboxes to []."""
    mock_sandbox = make_mock_sandbox()
    mock_client = make_mock_client(mock_sandbox)

    with patch(
        "inspect_sandboxes.daytona._daytona.AsyncDaytona", return_value=mock_client
    ):
        await DaytonaSandboxEnvironment.task_init("test_task", None)

    assert _daytona_client.get() is mock_client
    assert _running_sandboxes.get() == []


@pytest.mark.asyncio
async def test_task_init_with_any_config_only_creates_client(tmp_path: Any) -> None:
    """Test task_init with a config still only initializes the client."""
    dockerfile = tmp_path / "Dockerfile"
    dockerfile.write_text("FROM python:3.12\n")

    mock_sandbox = make_mock_sandbox()
    mock_client = make_mock_client(mock_sandbox)

    with patch(
        "inspect_sandboxes.daytona._daytona.AsyncDaytona", return_value=mock_client
    ):
        await DaytonaSandboxEnvironment.task_init("test_task", str(dockerfile))

    mock_client.create.assert_not_called()


@pytest.mark.asyncio
async def test_sample_init_no_config_uses_snapshot_params(
    mock_client: MagicMock,
    mock_sandbox: MagicMock,
) -> None:
    """Test sample_init with no config uses CreateSandboxFromSnapshotParams(snapshot=None)."""
    from daytona_sdk import CreateSandboxFromSnapshotParams

    with patch(
        "inspect_sandboxes.daytona._daytona.AsyncDaytona", return_value=mock_client
    ):
        await DaytonaSandboxEnvironment.task_init("test_task", None)
        await DaytonaSandboxEnvironment.sample_init("test_task", None, {})

    call_args = mock_client.create.call_args[0][0]
    assert isinstance(call_args, CreateSandboxFromSnapshotParams)
    assert call_args.snapshot is None
    assert call_args.auto_stop_interval == 0
    _assert_has_run_labels(call_args.labels)


@pytest.mark.asyncio
async def test_sample_init_dockerfile_uses_image_params(
    mock_client: MagicMock,
    tmp_path: Any,
) -> None:
    """Test sample_init with Dockerfile uses CreateSandboxFromImageParams."""
    from daytona_sdk import CreateSandboxFromImageParams

    dockerfile = tmp_path / "Dockerfile"
    dockerfile.write_text("FROM python:3.12\n")

    with patch(
        "inspect_sandboxes.daytona._daytona.AsyncDaytona", return_value=mock_client
    ):
        await DaytonaSandboxEnvironment.task_init("test_task", None)
        await DaytonaSandboxEnvironment.sample_init("test_task", str(dockerfile), {})

    call_args = mock_client.create.call_args[0][0]
    assert isinstance(call_args, CreateSandboxFromImageParams)
    assert call_args.auto_stop_interval == 0
    _assert_has_run_labels(call_args.labels)


@pytest.mark.asyncio
async def test_sample_init_compose_yaml_uses_image_params(
    mock_client: MagicMock,
    tmp_path: Any,
) -> None:
    """Test sample_init with compose YAML uses CreateSandboxFromImageParams with env vars."""
    from daytona_sdk import CreateSandboxFromImageParams

    compose_file = tmp_path / "compose.yaml"
    compose_file.write_text("""
services:
  default:
    image: python:3.12
    environment:
      - TEST_VAR=value
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 2g
""")

    with patch(
        "inspect_sandboxes.daytona._daytona.AsyncDaytona", return_value=mock_client
    ):
        await DaytonaSandboxEnvironment.task_init("test_task", None)
        await DaytonaSandboxEnvironment.sample_init("test_task", str(compose_file), {})

    call_args = mock_client.create.call_args[0][0]
    assert isinstance(call_args, CreateSandboxFromImageParams)
    assert call_args.image == "python:3.12"
    assert call_args.env_vars == {"TEST_VAR": "value"}
    assert call_args.resources is not None
    assert call_args.resources.cpu == 2
    assert call_args.resources.memory == 2
    assert call_args.auto_stop_interval == 0
    _assert_has_run_labels(call_args.labels)


@pytest.mark.asyncio
async def test_sample_init_compose_config_uses_image_params(
    mock_client: MagicMock,
) -> None:
    """Test sample_init with ComposeConfig uses CreateSandboxFromImageParams."""
    from daytona_sdk import CreateSandboxFromImageParams
    from inspect_ai.util._sandbox.compose import (
        ComposeDeploy,
        ComposeResourceConfig,
        ComposeResources,
    )

    config = ComposeConfig(
        services={
            "default": ComposeService(
                image="python:3.12",
                deploy=ComposeDeploy(
                    resources=ComposeResourceConfig(
                        limits=ComposeResources(cpus="4", memory="4g")
                    )
                ),
            )
        }
    )

    with patch(
        "inspect_sandboxes.daytona._daytona.AsyncDaytona", return_value=mock_client
    ):
        await DaytonaSandboxEnvironment.task_init("test_task", None)
        await DaytonaSandboxEnvironment.sample_init("test_task", config, {})

    call_args = mock_client.create.call_args[0][0]
    assert isinstance(call_args, CreateSandboxFromImageParams)
    assert call_args.image == "python:3.12"
    assert call_args.resources is not None
    assert call_args.resources.cpu == 4
    assert call_args.resources.memory == 4
    assert call_args.auto_stop_interval == 0
    _assert_has_run_labels(call_args.labels)


@pytest.mark.asyncio
async def test_sample_init_different_configs_per_sample(
    mock_client: MagicMock,
) -> None:
    """Test that different samples can use different configs independently."""
    from daytona_sdk import CreateSandboxFromImageParams

    config_a = ComposeConfig(services={"default": ComposeService(image="python:3.12")})
    config_b = ComposeConfig(services={"default": ComposeService(image="node:20")})

    sb_a = make_mock_sandbox("sb-a")
    sb_b = make_mock_sandbox("sb-b")
    mock_client.create = AsyncMock(side_effect=[sb_a, sb_b])

    with patch(
        "inspect_sandboxes.daytona._daytona.AsyncDaytona", return_value=mock_client
    ):
        await DaytonaSandboxEnvironment.task_init("test_task", None)

        envs_a = await DaytonaSandboxEnvironment.sample_init("test_task", config_a, {})
        envs_b = await DaytonaSandboxEnvironment.sample_init("test_task", config_b, {})

    calls = mock_client.create.call_args_list
    assert isinstance(calls[0][0][0], CreateSandboxFromImageParams)
    assert calls[0][0][0].image == "python:3.12"
    assert isinstance(calls[1][0][0], CreateSandboxFromImageParams)
    assert calls[1][0][0].image == "node:20"

    assert "default" in envs_a
    assert "default" in envs_b


@pytest.mark.asyncio
async def test_sample_init_invalid_config() -> None:
    """Test sample_init raises ValueError for unrecognized config."""
    mock_client = make_mock_client(make_mock_sandbox())

    with patch(
        "inspect_sandboxes.daytona._daytona.AsyncDaytona", return_value=mock_client
    ):
        await DaytonaSandboxEnvironment.task_init("test_task", None)
        with pytest.raises(ValueError, match="Unrecognized config"):
            await DaytonaSandboxEnvironment.sample_init(
                "test_task",
                12345,  # type: ignore[arg-type]
                {},
            )


@pytest.mark.asyncio
async def test_sample_init_compose_auto_stop_from_extension(
    mock_client: MagicMock,
) -> None:
    """Test that x-daytona auto_stop_interval extension overrides the default 0."""
    from daytona_sdk import CreateSandboxFromImageParams

    config = ComposeConfig(
        services={"default": ComposeService(image="python:3.12")},
        **{"x-daytona": {"auto_stop_interval": 30}},
    )

    with patch(
        "inspect_sandboxes.daytona._daytona.AsyncDaytona", return_value=mock_client
    ):
        await DaytonaSandboxEnvironment.task_init("test_task", None)
        await DaytonaSandboxEnvironment.sample_init("test_task", config, {})

    call_args = mock_client.create.call_args[0][0]
    assert isinstance(call_args, CreateSandboxFromImageParams)
    assert call_args.auto_stop_interval == 30


@pytest.mark.asyncio
async def test_sample_cleanup_skips_when_interrupted(mock_sandbox: MagicMock) -> None:
    """Test sample_cleanup does nothing when interrupted=True."""
    _init_context()
    env = DaytonaSandboxEnvironment(mock_sandbox)
    mock_client = make_mock_client(mock_sandbox)

    with patch(
        "inspect_sandboxes.daytona._daytona.AsyncDaytona", return_value=mock_client
    ):
        await DaytonaSandboxEnvironment.task_init("test_task", None)
        await DaytonaSandboxEnvironment.sample_cleanup(
            "test_task", None, {"default": env}, interrupted=True
        )

    mock_client.delete.assert_not_called()


@pytest.mark.asyncio
async def test_task_cleanup_no_op_when_cleanup_false() -> None:
    """Test task_cleanup is a no-op when cleanup=False."""
    _init_context()
    mock_client = make_mock_client(make_mock_sandbox())

    with patch(
        "inspect_sandboxes.daytona._daytona.AsyncDaytona", return_value=mock_client
    ):
        await DaytonaSandboxEnvironment.task_init("test_task", None)
        await DaytonaSandboxEnvironment.task_cleanup("test_task", None, cleanup=False)

    mock_client.delete.assert_not_called()
    mock_client.close.assert_not_called()


@pytest.mark.asyncio
async def test_task_cleanup_closes_client(mock_client: MagicMock) -> None:
    """Test task_cleanup closes the client when no sandboxes are running."""
    with patch(
        "inspect_sandboxes.daytona._daytona.AsyncDaytona", return_value=mock_client
    ):
        await DaytonaSandboxEnvironment.task_init("test_task", None)
        await DaytonaSandboxEnvironment.task_cleanup("test_task", None, cleanup=True)

    mock_client.close.assert_called_once()


@pytest.mark.asyncio
async def test_task_cleanup_handles_sandbox_failures(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test task_cleanup logs errors for failed sandbox deletions."""
    _init_context()
    _running_sandboxes.set(["sb-001", "sb-002"])

    mock_sb = make_mock_sandbox()
    mock_sb.id = "sb-001"

    mock_client = MagicMock()
    mock_client.get = AsyncMock(return_value=mock_sb)
    mock_client.delete = AsyncMock(side_effect=Exception("Delete failed"))
    mock_client.close = AsyncMock()

    with patch(
        "inspect_sandboxes.daytona._daytona.AsyncDaytona", return_value=mock_client
    ):
        await DaytonaSandboxEnvironment.task_init("test_task", None)
        # Manually set running sandboxes (bypassing sample_init)
        _running_sandboxes.set(["sb-001", "sb-002"])
        await DaytonaSandboxEnvironment.task_cleanup("test_task", None, cleanup=True)

    assert _running_sandboxes.get() == []
    assert "Failed to cleanup" in caplog.text


@pytest.mark.asyncio
async def test_task_init_generates_run_id() -> None:
    """Test that task_init generates a unique run_id."""
    mock_sb = make_mock_sandbox()
    mock_client = make_mock_client(mock_sb)

    with patch(
        "inspect_sandboxes.daytona._daytona.AsyncDaytona", return_value=mock_client
    ):
        await DaytonaSandboxEnvironment.task_init("test_task", None)

    run_id = _run_id.get()
    assert isinstance(run_id, str)
    assert len(run_id) == 32  # uuid4 hex


@pytest.mark.asyncio
async def test_sandbox_labels_include_run_id(
    mock_client: MagicMock,
    mock_sandbox: MagicMock,
) -> None:
    """Test that sandbox labels include inspect_run_id matching the task's run ID."""
    with patch(
        "inspect_sandboxes.daytona._daytona.AsyncDaytona", return_value=mock_client
    ):
        await DaytonaSandboxEnvironment.task_init("test_task", None)
        await DaytonaSandboxEnvironment.sample_init("test_task", None, {})

    call_args = mock_client.create.call_args[0][0]
    assert call_args.labels["inspect_run_id"] == _run_id.get()


@pytest.mark.asyncio
async def test_task_cleanup_deletes_orphaned_sandboxes() -> None:
    """Test that task_cleanup finds and deletes build-failed orphaned sandboxes."""
    orphan = make_mock_sandbox()
    orphan.id = "sb-orphan"
    orphan.state = "build_failed"

    paginated = MagicMock()
    paginated.items = [orphan]

    mock_client = MagicMock()
    mock_client.get = AsyncMock(side_effect=Exception("not tracked"))
    mock_client.list = AsyncMock(return_value=paginated)
    mock_client.delete = AsyncMock()
    mock_client.close = AsyncMock()

    with patch(
        "inspect_sandboxes.daytona._daytona.AsyncDaytona", return_value=mock_client
    ):
        await DaytonaSandboxEnvironment.task_init("test_task", None)
        # No sample_init — simulating the case where create failed
        await DaytonaSandboxEnvironment.task_cleanup("test_task", None, cleanup=True)

    # The orphan should have been found via list and deleted
    mock_client.list.assert_called_once_with(labels={"inspect_run_id": _run_id.get()})
    mock_client.delete.assert_called_once_with(orphan)


@pytest.mark.asyncio
async def test_task_cleanup_skips_already_deleted_in_orphan_pass(
    mock_sandbox: MagicMock,
) -> None:
    """Test that task_cleanup doesn't double-delete sandboxes found in both passes."""
    # The sandbox is tracked AND appears in the list response
    paginated = MagicMock()
    paginated.items = [mock_sandbox]

    mock_client = MagicMock()
    mock_client.get = AsyncMock(return_value=mock_sandbox)
    mock_client.list = AsyncMock(return_value=paginated)
    mock_client.delete = AsyncMock()
    mock_client.close = AsyncMock()

    with patch(
        "inspect_sandboxes.daytona._daytona.AsyncDaytona", return_value=mock_client
    ):
        await DaytonaSandboxEnvironment.task_init("test_task", None)
        _running_sandboxes.set([mock_sandbox.id])
        await DaytonaSandboxEnvironment.task_cleanup("test_task", None, cleanup=True)

    # Should only be deleted once (in the first pass), not again in the orphan pass
    mock_client.delete.assert_called_once_with(mock_sandbox)


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

    env = DaytonaSandboxEnvironment(mock_sandbox)
    result = await env.exec(cmd)

    assert isinstance(result, ExecResult)
    assert result.success == (returncode == 0)
    assert result.returncode == returncode
    assert result.stdout == expected_stdout
    assert result.stderr == ""


@pytest.mark.asyncio
async def test_exec_joins_args_with_shlex(mock_sandbox: MagicMock) -> None:
    """Test that exec correctly joins args into a shell command string."""
    env = DaytonaSandboxEnvironment(mock_sandbox)
    await env.exec(["echo", "hello world"])

    call_args = mock_sandbox.process.exec.call_args
    command_arg = call_args[0][0]  # first positional arg
    assert command_arg == "echo 'hello world'"


@pytest.mark.asyncio
async def test_exec_passes_cwd_and_env(mock_sandbox: MagicMock) -> None:
    """Test that exec passes cwd and env to process.exec."""
    env = DaytonaSandboxEnvironment(mock_sandbox)
    await env.exec(
        ["ls"],
        cwd="/workspace",
        env={"MY_VAR": "value"},
    )

    call_kwargs = mock_sandbox.process.exec.call_args[1]
    assert call_kwargs["cwd"] == "/workspace"
    assert call_kwargs["env"] == {"MY_VAR": "value"}


@pytest.mark.asyncio
async def test_write_file_text(mock_sandbox: MagicMock) -> None:
    """Test write_file with text content."""
    env = DaytonaSandboxEnvironment(mock_sandbox)
    # Parent dir exists (no EISDIR)
    file_info = MagicMock()
    file_info.is_dir = False
    mock_sandbox.fs.get_file_info = AsyncMock(return_value=file_info)

    await env.write_file("/workspace/test.txt", "hello")

    mock_sandbox.fs.upload_file.assert_called_once_with(b"hello", "/workspace/test.txt")


@pytest.mark.asyncio
async def test_write_file_binary(mock_sandbox: MagicMock) -> None:
    """Test write_file with binary content."""
    env = DaytonaSandboxEnvironment(mock_sandbox)
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
    env = DaytonaSandboxEnvironment(mock_sandbox)
    file_info = MagicMock()
    file_info.is_dir = False
    mock_sandbox.fs.get_file_info = AsyncMock(return_value=file_info)

    await env.write_file("/deep/nested/dir/file.txt", "content")

    mock_sandbox.fs.create_folder.assert_called_once_with("/deep/nested/dir", "755")


@pytest.mark.asyncio
async def test_write_file_raises_for_directory(mock_sandbox: MagicMock) -> None:
    """Test write_file raises IsADirectoryError when path is a directory."""
    env = DaytonaSandboxEnvironment(mock_sandbox)

    # create_folder succeeds, but get_file_info reports it's a directory
    dir_info = MagicMock()
    dir_info.is_dir = True
    mock_sandbox.fs.get_file_info = AsyncMock(return_value=dir_info)

    with pytest.raises(IsADirectoryError):
        await env.write_file("/existing/dir", "content")


@pytest.mark.asyncio
async def test_read_file_text(mock_sandbox: MagicMock) -> None:
    """Test read_file in text mode."""
    env = DaytonaSandboxEnvironment(mock_sandbox)

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
    env = DaytonaSandboxEnvironment(mock_sandbox)

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
    from daytona_sdk import DaytonaNotFoundError

    env = DaytonaSandboxEnvironment(mock_sandbox)

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
    env = DaytonaSandboxEnvironment(mock_sandbox)

    dir_info = MagicMock()
    dir_info.is_dir = True
    dir_info.size = 0
    mock_sandbox.fs.get_file_info = AsyncMock(return_value=dir_info)

    with pytest.raises(IsADirectoryError):
        await env.read_file("/some/dir")


@pytest.mark.asyncio
async def test_read_file_size_limit(mock_sandbox: MagicMock) -> None:
    """Test read_file raises OutputLimitExceededError for oversized files."""
    env = DaytonaSandboxEnvironment(mock_sandbox)

    file_info = MagicMock()
    file_info.is_dir = False
    file_info.size = SandboxEnvironmentLimits.MAX_READ_FILE_SIZE + 1
    mock_sandbox.fs.get_file_info = AsyncMock(return_value=file_info)

    with pytest.raises(OutputLimitExceededError):
        await env.read_file("/huge.bin")


@pytest.mark.asyncio
async def test_cli_cleanup_single_success(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Test CLI cleanup of a single sandbox by ID."""
    mock_sandbox = make_mock_sandbox("sb-abc")
    mock_client = make_mock_client(mock_sandbox)

    with patch(
        "inspect_sandboxes.daytona._daytona.AsyncDaytona", return_value=mock_client
    ):
        await DaytonaSandboxEnvironment.cli_cleanup("sb-abc")

    mock_client.delete.assert_called_once_with(mock_sandbox)
    captured = capsys.readouterr()
    assert "Successfully deleted" in captured.out


@pytest.mark.asyncio
async def test_cli_cleanup_single_failure(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Test CLI cleanup failure exits with code 1."""
    mock_sandbox = make_mock_sandbox("sb-abc")
    mock_client = make_mock_client(mock_sandbox)
    mock_client.delete = AsyncMock(side_effect=Exception("delete failed"))

    with patch(
        "inspect_sandboxes.daytona._daytona.AsyncDaytona", return_value=mock_client
    ):
        with pytest.raises(SystemExit) as exc_info:
            await DaytonaSandboxEnvironment.cli_cleanup("sb-abc")

    assert exc_info.value.code == 1


@pytest.mark.asyncio
async def test_cli_cleanup_bulk_no_sandboxes(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Test CLI bulk cleanup with no sandboxes found."""
    mock_client = MagicMock()
    paginated = MagicMock()
    paginated.items = []
    mock_client.list = AsyncMock(return_value=paginated)
    mock_client.close = AsyncMock()

    with patch(
        "inspect_sandboxes.daytona._daytona.AsyncDaytona", return_value=mock_client
    ):
        await DaytonaSandboxEnvironment.cli_cleanup(None)

    captured = capsys.readouterr()
    assert "No Daytona sandboxes found" in captured.out


@pytest.mark.asyncio
async def test_cli_cleanup_bulk_with_sandboxes(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Test CLI bulk cleanup terminates all found sandboxes."""
    sb1 = make_mock_sandbox("sb-001")
    sb2 = make_mock_sandbox("sb-002")

    mock_client = MagicMock()
    paginated = MagicMock()
    paginated.items = [sb1, sb2]
    mock_client.list = AsyncMock(return_value=paginated)
    mock_client.delete = AsyncMock()
    mock_client.close = AsyncMock()

    with patch(
        "inspect_sandboxes.daytona._daytona.AsyncDaytona", return_value=mock_client
    ):
        await DaytonaSandboxEnvironment.cli_cleanup(None)

    mock_client.list.assert_called_once_with(labels=INSPECT_SANDBOX_LABEL)
    assert mock_client.delete.call_count == 2
    captured = capsys.readouterr()
    assert "sb-001" in captured.out
    assert "sb-002" in captured.out
    assert "Successfully deleted: 2" in captured.out


@pytest.mark.asyncio
async def test_cli_cleanup_bulk_partial_failure(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Test CLI bulk cleanup exits with code 1 on partial failure."""
    sb1 = make_mock_sandbox("sb-001")
    sb2 = make_mock_sandbox("sb-002")

    async def failing_delete(sb: MagicMock) -> None:
        if sb.id == "sb-001":
            raise Exception("delete failed")

    mock_client = MagicMock()
    paginated = MagicMock()
    paginated.items = [sb1, sb2]
    mock_client.list = AsyncMock(return_value=paginated)
    mock_client.delete = AsyncMock(side_effect=failing_delete)
    mock_client.close = AsyncMock()

    with patch(
        "inspect_sandboxes.daytona._daytona.AsyncDaytona", return_value=mock_client
    ):
        with pytest.raises(SystemExit) as exc_info:
            await DaytonaSandboxEnvironment.cli_cleanup(None)

    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    assert "Failed to delete: 1" in captured.out


@pytest.mark.asyncio
async def test_exec_with_stdin_string(mock_sandbox: MagicMock) -> None:
    """Test exec pipes string stdin through a temp file with pipefail."""
    env = DaytonaSandboxEnvironment(mock_sandbox)
    await env.exec(["cat"], input="hello")

    # Verify file was uploaded with UTF-8 encoded content
    mock_sandbox.fs.upload_file.assert_called_once()
    call_args = mock_sandbox.fs.upload_file.call_args
    assert call_args[0][0] == b"hello"
    stdin_path = call_args[0][1]
    assert stdin_path.startswith("/tmp/.inspect-stdin-")

    # Verify the command uses pipefail and cleans up the temp file
    exec_command = mock_sandbox.process.exec.call_args[0][0]
    assert "set -o pipefail" in exec_command
    assert f"cat {stdin_path}" in exec_command
    assert f"rm -f {stdin_path}" in exec_command


@pytest.mark.asyncio
async def test_exec_with_stdin_bytes(mock_sandbox: MagicMock) -> None:
    """Test exec pipes bytes stdin through a temp file."""
    env = DaytonaSandboxEnvironment(mock_sandbox)
    await env.exec(["wc", "-c"], input=b"\x00\x01\x02")

    call_args = mock_sandbox.fs.upload_file.call_args
    assert call_args[0][0] == b"\x00\x01\x02"


@pytest.mark.asyncio
async def test_exec_without_stdin_no_upload(mock_sandbox: MagicMock) -> None:
    """Test exec without stdin does not upload any file."""
    env = DaytonaSandboxEnvironment(mock_sandbox)
    await env.exec(["echo", "hi"])

    mock_sandbox.fs.upload_file.assert_not_called()
    command = mock_sandbox.process.exec.call_args[0][0]
    assert command == "echo hi"
    assert "pipefail" not in command


def test_get_sandbox_id_none() -> None:
    """Test _get_sandbox_id returns 'unknown' for None."""
    assert DaytonaSandboxEnvironment._get_sandbox_id(None) == "unknown"


def test_get_sandbox_id_valid(mock_sandbox: MagicMock) -> None:
    """Test _get_sandbox_id returns the sandbox ID."""
    assert DaytonaSandboxEnvironment._get_sandbox_id(mock_sandbox) == "sb-test-123"


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
    env = DaytonaSandboxEnvironment(mock_sandbox)
    result = await env.exec(["echo", "test"])

    assert result.success
    assert call_count == 2


@pytest.mark.asyncio
async def test_exec_does_not_retry_timeout(mock_sandbox: MagicMock) -> None:
    """Test that exec does NOT retry DaytonaTimeoutError inside _run (outer loop handles it)."""
    mock_sandbox.process.exec = AsyncMock(side_effect=DaytonaTimeoutError("timed out"))
    env = DaytonaSandboxEnvironment(mock_sandbox)

    with pytest.raises(TimeoutError):
        await env.exec(["sleep", "100"], timeout=5)

    # Outer timeout loop makes 3 attempts (original, 5s cap, 5s cap)
    assert mock_sandbox.process.exec.call_count == 3


@pytest.mark.asyncio
async def test_exec_does_not_retry_non_daytona_error(mock_sandbox: MagicMock) -> None:
    """Test that exec does NOT retry non-DaytonaError exceptions."""
    mock_sandbox.process.exec = AsyncMock(side_effect=RuntimeError("unexpected"))
    env = DaytonaSandboxEnvironment(mock_sandbox)

    with pytest.raises(RuntimeError, match="unexpected"):
        await env.exec(["echo", "test"])

    # Called only once — no retry
    assert mock_sandbox.process.exec.call_count == 1
