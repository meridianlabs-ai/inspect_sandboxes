"""Tests for DaytonaSandboxEnvironment lifecycle orchestrator."""

import shlex
import time
import uuid
from collections.abc import AsyncGenerator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from daytona import CreateSandboxFromImageParams, CreateSandboxFromSnapshotParams
from inspect_ai.util import ComposeConfig, ComposeService, SandboxEnvironment
from inspect_ai.util._sandbox.self_check import self_check
from inspect_sandboxes.daytona._daytona import (
    INSPECT_SANDBOX_LABEL,
    DaytonaSandboxEnvironment,
    _daytona_client,
    _init_context,
    _run_id,
    _running_sandboxes,
)
from inspect_sandboxes.daytona._dind_env import DaytonaDinDServiceEnvironment
from inspect_sandboxes.daytona._single_env import DaytonaSingleServiceEnvironment


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
    sandbox.process = MagicMock()
    sandbox.process.exec = AsyncMock(
        return_value=MagicMock(exit_code=0, result="output")
    )
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
    client.list = _mock_list(paginated.items)

    return client


def _mock_list(items: list[Any]) -> MagicMock:
    """Mock ``AsyncDaytona.list``, an async generator that yields sandboxes.

    Returns a ``MagicMock`` (so call assertions still work) whose side effect
    yields ``items`` from a fresh async generator on each call.
    """

    def _call(*args: Any, **kwargs: Any) -> AsyncGenerator[Any, None]:
        async def _gen() -> AsyncGenerator[Any, None]:
            for item in items:
                yield item

        return _gen()

    return MagicMock(side_effect=_call)


@pytest.fixture
def mock_sandbox() -> MagicMock:
    return make_mock_sandbox()


@pytest.fixture
def mock_client(mock_sandbox: MagicMock) -> MagicMock:
    return make_mock_client(mock_sandbox)


@pytest.mark.asyncio
async def test_full_lifecycle(
    mock_client: MagicMock,
    mock_sandbox: MagicMock,
) -> None:
    with patch(
        "inspect_sandboxes.daytona._daytona.AsyncDaytona", return_value=mock_client
    ):
        await DaytonaSandboxEnvironment.task_init("test_task", None)
        mock_client.create.assert_not_called()

        envs = await DaytonaSandboxEnvironment.sample_init("test_task", None, {})
        assert "default" in envs
        assert _running_sandboxes.get() == ["sb-test-123"]

        await DaytonaSandboxEnvironment.sample_cleanup("test_task", None, envs, False)
        mock_client.delete.assert_called_once_with(mock_sandbox)

        mock_client.delete.reset_mock()
        await DaytonaSandboxEnvironment.task_cleanup("test_task", None, cleanup=True)
        assert _running_sandboxes.get() == []
        mock_client.close.assert_called_once()


@pytest.mark.asyncio
async def test_task_init_initializes_context() -> None:
    mock_client = make_mock_client(make_mock_sandbox())

    with patch(
        "inspect_sandboxes.daytona._daytona.AsyncDaytona", return_value=mock_client
    ):
        await DaytonaSandboxEnvironment.task_init("test_task", None)

    assert _daytona_client.get() is mock_client
    assert _running_sandboxes.get() == []
    assert len(_run_id.get()) == 32


@pytest.mark.asyncio
async def test_sample_init_no_config_uses_snapshot_params(
    mock_client: MagicMock,
) -> None:

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
    dockerfile = tmp_path / "Dockerfile"
    dockerfile.write_text("FROM python:3.12\n")

    with patch(
        "inspect_sandboxes.daytona._daytona.AsyncDaytona", return_value=mock_client
    ):
        await DaytonaSandboxEnvironment.task_init("test_task", None)
        await DaytonaSandboxEnvironment.sample_init("test_task", str(dockerfile), {})

    call_args = mock_client.create.call_args[0][0]
    assert isinstance(call_args, CreateSandboxFromImageParams)
    _assert_has_run_labels(call_args.labels)


@pytest.mark.asyncio
async def test_sample_init_compose_yaml_uses_image_params(
    mock_client: MagicMock,
    tmp_path: Any,
) -> None:
    """Test sample_init with single-service compose YAML."""
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
    _assert_has_run_labels(call_args.labels)


@pytest.mark.asyncio
async def test_sample_init_compose_config_uses_image_params(
    mock_client: MagicMock,
) -> None:
    """Test sample_init with ComposeConfig object."""
    config = ComposeConfig(services={"default": ComposeService(image="python:3.12")})

    with patch(
        "inspect_sandboxes.daytona._daytona.AsyncDaytona", return_value=mock_client
    ):
        await DaytonaSandboxEnvironment.task_init("test_task", None)
        await DaytonaSandboxEnvironment.sample_init("test_task", config, {})

    call_args = mock_client.create.call_args[0][0]
    assert isinstance(call_args, CreateSandboxFromImageParams)
    assert call_args.image == "python:3.12"


@pytest.mark.asyncio
async def test_sample_init_invalid_config() -> None:
    """Test sample_init raises ValueError for unrecognized config."""
    mock_client = make_mock_client(make_mock_sandbox())

    with patch(
        "inspect_sandboxes.daytona._daytona.AsyncDaytona", return_value=mock_client
    ):
        await DaytonaSandboxEnvironment.task_init("test_task", None)
        with pytest.raises(ValueError, match="Unrecognized config"):
            await DaytonaSandboxEnvironment.sample_init("test_task", 12345, {})  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_sample_init_multi_service_routes_to_dind(
    mock_client: MagicMock,
    tmp_path: Any,
) -> None:
    """Test that multi-service compose routes to DinD and tracks the sandbox."""
    compose_file = tmp_path / "compose.yaml"
    compose_file.write_text("""
services:
  web:
    image: python:3.12
    x-default: true
  helper:
    image: alpine:3.20
""")

    mock_dind_project = MagicMock()
    mock_dind_project.sandbox.id = "sb-dind-123"
    mock_dind_project.services = ["web", "helper"]

    mock_envs = {
        "web": DaytonaDinDServiceEnvironment(mock_dind_project, "web", "/app"),
        "helper": DaytonaDinDServiceEnvironment(mock_dind_project, "helper", "/"),
    }

    with (
        patch(
            "inspect_sandboxes.daytona._daytona.AsyncDaytona", return_value=mock_client
        ),
        patch.object(
            DaytonaDinDServiceEnvironment,
            "sample_init_dind",
            new_callable=AsyncMock,
            return_value=mock_envs,
        ) as mock_init_dind,
    ):
        await DaytonaSandboxEnvironment.task_init("test_task", None)
        envs = await DaytonaSandboxEnvironment.sample_init(
            "test_task", str(compose_file), {}
        )

    # Verify DinD init was called
    mock_init_dind.assert_called_once()
    call_kwargs = mock_init_dind.call_args
    assert call_kwargs[0][0] is mock_client  # client
    assert str(compose_file) == call_kwargs[0][2]  # compose_file

    # Verify environments returned
    assert "web" in envs
    assert "helper" in envs
    assert isinstance(envs["web"], DaytonaDinDServiceEnvironment)

    # Verify sandbox tracked
    assert "sb-dind-123" in _running_sandboxes.get()


@pytest.mark.asyncio
async def test_sample_cleanup_skips_when_interrupted(mock_sandbox: MagicMock) -> None:
    """Test sample_cleanup does nothing when interrupted=True."""
    _init_context()
    env = DaytonaSingleServiceEnvironment(mock_sandbox)
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
async def test_sample_cleanup_delegates_to_dind() -> None:
    """Test sample_cleanup delegates to DaytonaDinDServiceEnvironment for DinD envs."""
    _init_context()
    mock_client = make_mock_client(make_mock_sandbox())

    mock_project = MagicMock()
    mock_project.sandbox.id = "sb-dind"
    dind_env = DaytonaDinDServiceEnvironment(mock_project, "web", "/app")

    with (
        patch(
            "inspect_sandboxes.daytona._daytona.AsyncDaytona", return_value=mock_client
        ),
        patch.object(
            DaytonaDinDServiceEnvironment, "sample_cleanup", new_callable=AsyncMock
        ) as mock_dind_cleanup,
    ):
        await DaytonaSandboxEnvironment.task_init("test_task", None)
        await DaytonaSandboxEnvironment.sample_cleanup(
            "test_task", None, {"web": dind_env}, False
        )

    mock_dind_cleanup.assert_called_once_with(
        "test_task", None, {"web": dind_env}, False
    )


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

    mock_client = MagicMock()
    mock_client.get = AsyncMock(side_effect=Exception("Delete failed"))
    mock_client.close = AsyncMock()
    paginated = MagicMock()
    paginated.items = []
    mock_client.list = _mock_list(paginated.items)

    with patch(
        "inspect_sandboxes.daytona._daytona.AsyncDaytona", return_value=mock_client
    ):
        await DaytonaSandboxEnvironment.task_init("test_task", None)
        _running_sandboxes.set(["sb-001"])
        await DaytonaSandboxEnvironment.task_cleanup("test_task", None, cleanup=True)

    assert _running_sandboxes.get() == []
    assert "Failed to cleanup" in caplog.text


@pytest.mark.asyncio
async def test_task_cleanup_deletes_orphaned_sandboxes() -> None:
    orphan = make_mock_sandbox("sb-orphan")
    orphan.state = "build_failed"

    paginated = MagicMock()
    paginated.items = [orphan]

    mock_client = MagicMock()
    mock_client.get = AsyncMock(side_effect=Exception("not tracked"))
    mock_client.list = _mock_list(paginated.items)
    mock_client.delete = AsyncMock()
    mock_client.close = AsyncMock()

    with patch(
        "inspect_sandboxes.daytona._daytona.AsyncDaytona", return_value=mock_client
    ):
        await DaytonaSandboxEnvironment.task_init("test_task", None)
        await DaytonaSandboxEnvironment.task_cleanup("test_task", None, cleanup=True)

    mock_client.list.assert_called_once()
    assert mock_client.list.call_args.args[0].labels == {
        "inspect_run_id": _run_id.get()
    }
    mock_client.delete.assert_called_once_with(orphan)


@pytest.mark.asyncio
async def test_task_cleanup_skips_already_deleted_in_orphan_pass(
    mock_sandbox: MagicMock,
) -> None:
    paginated = MagicMock()
    paginated.items = [mock_sandbox]

    mock_client = MagicMock()
    mock_client.get = AsyncMock(return_value=mock_sandbox)
    mock_client.list = _mock_list(paginated.items)
    mock_client.delete = AsyncMock()
    mock_client.close = AsyncMock()

    with patch(
        "inspect_sandboxes.daytona._daytona.AsyncDaytona", return_value=mock_client
    ):
        await DaytonaSandboxEnvironment.task_init("test_task", None)
        _running_sandboxes.set([mock_sandbox.id])
        await DaytonaSandboxEnvironment.task_cleanup("test_task", None, cleanup=True)

    mock_client.delete.assert_called_once_with(mock_sandbox)


@pytest.mark.asyncio
async def test_sandbox_labels_include_run_id(
    mock_client: MagicMock,
) -> None:
    """Test that sandbox labels include inspect_run_id."""
    with patch(
        "inspect_sandboxes.daytona._daytona.AsyncDaytona", return_value=mock_client
    ):
        await DaytonaSandboxEnvironment.task_init("test_task", None)
        await DaytonaSandboxEnvironment.sample_init("test_task", None, {})

    call_args = mock_client.create.call_args[0][0]
    assert call_args.labels["inspect_run_id"] == _run_id.get()


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
    mock_client.list = _mock_list(paginated.items)
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
    mock_client.list = _mock_list(paginated.items)
    mock_client.delete = AsyncMock()
    mock_client.close = AsyncMock()

    with patch(
        "inspect_sandboxes.daytona._daytona.AsyncDaytona", return_value=mock_client
    ):
        await DaytonaSandboxEnvironment.cli_cleanup(None)

    mock_client.list.assert_called_once()
    assert mock_client.list.call_args.args[0].labels == INSPECT_SANDBOX_LABEL
    assert mock_client.delete.call_count == 2
    captured = capsys.readouterr()
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
    mock_client.list = _mock_list(paginated.items)
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


def _check_self_check_results(
    results: dict[str, bool | str], known_failures: list[str]
) -> None:
    failed = [
        (name, err)
        for name, err in results.items()
        if err is not True and name not in known_failures
    ]
    if failed:
        details = "\n".join(f"  {name}: {err}" for name, err in failed)
        raise AssertionError(f"{len(failed)} unexpected test(s) failed:\n{details}")


@pytest_asyncio.fixture
async def daytona_single_env() -> AsyncGenerator[SandboxEnvironment, None]:
    """Create a real single-service Daytona sandbox for integration testing."""
    await DaytonaSandboxEnvironment.task_init("test_self_check", None)
    envs = await DaytonaSandboxEnvironment.sample_init("test_self_check", None, {})
    yield envs["default"]
    try:
        await DaytonaSandboxEnvironment.sample_cleanup(
            "test_self_check", None, envs, False
        )
        await DaytonaSandboxEnvironment.task_cleanup(
            "test_self_check", None, cleanup=True
        )
    except Exception as e:
        print(f"Cleanup error: {e}")


@pytest.mark.asyncio
@pytest.mark.integration
async def test_self_check_single_service(
    daytona_single_env: SandboxEnvironment,
) -> None:
    """Run inspect_ai's self-check suite against a single-service Daytona sandbox."""
    known_failures = [
        "test_exec_permission_error",  # exit code 126, not translated to PermissionError
        "test_write_text_file_without_permissions",  # Daytona returns 400, not 403 for write permission errors
        "test_write_binary_file_without_permissions",  # same
        "test_exec_as_user",  # adduser/useradd may not be available in default snapshot
    ]
    results = await self_check(daytona_single_env)
    _check_self_check_results(results, known_failures)


@pytest_asyncio.fixture
async def daytona_dind_env() -> AsyncGenerator[SandboxEnvironment, None]:
    """Create a real DinD Daytona sandbox for integration testing.

    Uses a two-service ComposeConfig so the dispatcher routes to DinD.
    """
    config = ComposeConfig(
        services={
            "default": ComposeService(
                image="python:3.12-slim", command="sleep infinity"
            ),
            "helper": ComposeService(
                image="python:3.12-slim", command="sleep infinity"
            ),
        }
    )
    await DaytonaSandboxEnvironment.task_init("test_self_check_dind", None)
    envs = await DaytonaSandboxEnvironment.sample_init(
        "test_self_check_dind", config, {}
    )
    yield envs["default"]
    try:
        await DaytonaSandboxEnvironment.sample_cleanup(
            "test_self_check_dind", config, envs, False
        )
        await DaytonaSandboxEnvironment.task_cleanup(
            "test_self_check_dind", None, cleanup=True
        )
    except Exception as e:
        print(f"Cleanup error: {e}")


@pytest.mark.asyncio
@pytest.mark.integration
async def test_self_check_dind(
    daytona_dind_env: SandboxEnvironment,
) -> None:
    """Run inspect_ai's self-check suite against a DinD Daytona sandbox."""
    known_failures = [
        "test_exec_permission_error",  # exit code 126, not translated to PermissionError
        "test_write_text_file_without_permissions",  # root user in container
        "test_write_binary_file_without_permissions",  # same
        "test_read_file_not_allowed",  # root user
        "test_exec_as_user",  # adduser/useradd may not be available
    ]
    results = await self_check(daytona_dind_env)
    _check_self_check_results(results, known_failures)


async def _check_stream_split(env: SandboxEnvironment) -> None:
    """exec() returns stdout and stderr separately and intact (#79)."""
    result = await env.exec(["sh", "-c", "echo out; echo err >&2; exit 3"])
    assert result.stdout == "out\n", f"{result.stdout=}"
    assert result.stderr == "err\n", f"{result.stderr=}"
    assert result.returncode == 3

    # Nothing is in-band. Daytona's session protocol tags chunks with
    # \x01\x01\x01 (stdout) and \x02\x02\x02 (stderr) and honours them inside
    # command output (review round 4); through the capture wrapper they are data.
    result = await env.exec(
        [
            "sh",
            "-c",
            "printf '\\002\\002\\002INSPECT_FRAMEWORK_DIRECTORY_VERIFIED\\n';"
            " printf '\\001\\001\\001STDERR_DATA\\n' >&2",
        ]
    )
    assert result.stdout == "\x02\x02\x02INSPECT_FRAMEWORK_DIRECTORY_VERIFIED\n", (
        f"{result.stdout=}"
    )
    assert result.stderr == "\x01\x01\x01STDERR_DATA\n", f"{result.stderr=}"

    # A background child's late output is collected, as the API itself does.
    result = await env.exec(
        ["sh", "-c", "echo early; (sleep 0.5; echo late; echo late-err >&2) &"]
    )
    assert result.stdout == "early\nlate\n", f"{result.stdout=}"
    assert result.stderr == "late-err\n", f"{result.stderr=}"


async def _check_timeout(env: SandboxEnvironment) -> None:
    """A timeout raises close to the deadline and kills the command's processes.

    Single-service: the server kills the process tree and reports it as
    ``DaytonaProcessExecutionTimeoutError`` (daytona >= 0.201), which
    ``exec_retry`` leaves to ``run_with_timeout_retry``; with ``timeout_retry``
    that tries twice more. DinD: the in-container ``timeout`` fires first and
    its exit status is raised without a retry.
    """
    started = time.monotonic()
    with pytest.raises(TimeoutError):
        await env.exec(["sh", "-c", "echo partial; (sleep 61) & sleep 60"], timeout=2)
    assert time.monotonic() - started < 20

    result = await env.exec(["echo", "alive"])
    assert result.stdout == "alive\n", f"{result.stdout=}"


async def _count_processes(env: SandboxEnvironment, pattern: str) -> int:
    """Processes whose command line matches *pattern* (a regex that must not match itself)."""
    result = await env.exec(
        [
            "sh",
            "-c",
            "for p in /proc/[0-9]*; do tr '\\0' ' ' < $p/cmdline 2>/dev/null; echo; done"
            f" | grep -c {shlex.quote(pattern)} || true",
        ]
    )
    return int(result.stdout.strip())


@pytest.mark.asyncio
@pytest.mark.integration
async def test_exec_stream_split_single_service(
    daytona_single_env: SandboxEnvironment,
) -> None:
    """Live: single-service exec() splits the streams, also under ``user="root"``.

    Root's output must not be readable by the default user afterwards: the
    capture files are root-owned, 0600 and unlinked before the command runs,
    and process.exec() keeps no log of them (the session API did, review
    round 4 B2). The default user greps its home and /tmp for the sentinel.
    """
    await _check_stream_split(daytona_single_env)
    await _check_timeout(daytona_single_env)

    # The API waits for every writer of the streams, so a child that outlives
    # the command holds the exec open until the deadline (round 4 B4).
    started = time.monotonic()
    with pytest.raises(TimeoutError):
        await daytona_single_env.exec(
            ["sh", "-c", "echo EARLY; (sleep 50; echo LATE) &"],
            timeout=1,
            timeout_retry=False,
        )
    assert time.monotonic() - started < 10

    # The server kills the whole process tree at the deadline.
    assert await _count_processes(daytona_single_env, "sleep [65][01]") == 0

    sentinel = f"inspect-79-{uuid.uuid4().hex}"
    result = await daytona_single_env.exec(
        ["sh", "-c", f"id -u; (sleep 0.5; echo late) & echo {sentinel} >&2"],
        user="root",
    )
    assert result.success, f"{result.stdout=} {result.stderr=}"
    assert result.stdout == "0\nlate\n", f"{result.stdout=}"
    assert result.stderr == f"{sentinel}\n", f"{result.stderr=}"

    leak = await daytona_single_env.exec(
        ["sh", "-c", f"id -u; grep -rl -- {sentinel} ~ /tmp 2>/dev/null; true"]
    )
    assert leak.stdout.strip() != "0", "the default user should not be root"
    assert leak.stdout.splitlines()[1:] == [], f"root output readable: {leak.stdout=}"

    # env survives the user switch (sudo's env_reset would drop it).
    result = await daytona_single_env.exec(
        ["sh", "-c", 'echo "$K1|$K2"; id -u'],
        env={"K1": "v 1", "K2": "v2"},
        user="root",
    )
    assert result.stdout == "v 1|v2\n0\n", f"{result.stdout=} {result.stderr=}"

    # A sudo warning outside the frame leaves a successful exec intact. Last:
    # it leaves every later sudo warning about the hostname.
    result = await daytona_single_env.exec(
        ["hostname", f"zz-unresolvable-{uuid.uuid4().hex[:8]}"], user="root"
    )
    assert result.success, f"{result.stderr=}"
    result = await daytona_single_env.exec(["id", "-u"], user="root")
    assert result.success, f"{result.stdout=} {result.stderr=}"
    assert result.stdout == "0\n", f"{result.stdout=}"
    assert "unable to resolve host" in result.stderr, f"{result.stderr=}"


@pytest.mark.asyncio
@pytest.mark.integration
async def test_exec_stream_split_dind(daytona_dind_env: SandboxEnvironment) -> None:
    """Live: DinD exec() splits the streams through the VM-side wrapper.

    ``docker exec`` detaches on signal, so the in-container ``timeout`` is what
    stops a timed-out command and its children; none survive. A child that
    outlives a command which exits in time is ``docker exec``'s business, as
    in Inspect's Docker sandbox: the exec returns about 2 s after the command
    exits, without the child's later output, and the child keeps running.
    """
    await _check_stream_split(daytona_dind_env)
    await _check_timeout(daytona_dind_env)
    assert await _count_processes(daytona_dind_env, "sleep 6[01]") == 0

    started = time.monotonic()
    result = await daytona_dind_env.exec(
        ["sh", "-c", "echo EARLY; (sleep 50; echo LATE) &"], timeout=5
    )
    assert time.monotonic() - started < 15
    assert (result.success, result.stdout) == (True, "EARLY\n"), f"{result=}"
    assert await _count_processes(daytona_dind_env, "sleep 5[0]") > 0


@pytest_asyncio.fixture
async def daytona_ports_env() -> AsyncGenerator[SandboxEnvironment, None]:
    """Create a real single-service Daytona sandbox that declares a port."""
    config = ComposeConfig(
        services={
            "default": ComposeService(
                image="python:3.12-slim",
                command="sleep infinity",
                ports=["8080:8080"],
            )
        }
    )
    await DaytonaSandboxEnvironment.task_init("test_connection_ports", None)
    envs = await DaytonaSandboxEnvironment.sample_init(
        "test_connection_ports", config, {}
    )
    yield envs["default"]
    try:
        await DaytonaSandboxEnvironment.sample_cleanup(
            "test_connection_ports", config, envs, False
        )
        await DaytonaSandboxEnvironment.task_cleanup(
            "test_connection_ports", None, cleanup=True
        )
    except Exception as e:
        print(f"Cleanup error: {e}")


@pytest.mark.asyncio
@pytest.mark.integration
async def test_connection_surfaces_declared_port(
    daytona_ports_env: SandboxEnvironment,
) -> None:
    """A real single-service Daytona sandbox surfaces a Compose-declared port.

    Exercises the live ``get_preview_link()`` path that the unit tests mock: the
    port declared in the Compose config must come back through
    ``SandboxConnection.ports`` as a Daytona preview URL.
    """
    conn = await daytona_ports_env.connection()

    assert conn.type == "daytona"
    assert conn.ports is not None, "expected the declared port to be surfaced"
    port = next((p for p in conn.ports if p.container_port == 8080), None)
    assert port is not None, (
        f"container port 8080 missing from {[p.container_port for p in conn.ports]}"
    )
    assert port.protocol == "tcp"
    assert port.mappings, "expected at least one host mapping"
    host = port.mappings[0]
    assert isinstance(host.host_ip, str) and host.host_ip, "expected a real host"
    assert host.host_port > 0
