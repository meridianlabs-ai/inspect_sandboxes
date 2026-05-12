"""Tests for RunloopSandboxEnvironment lifecycle orchestrator."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from inspect_ai.util import ComposeConfig, ComposeService, SandboxEnvironment
from inspect_ai.util._sandbox.self_check import self_check
from inspect_sandboxes.runloop._runloop import (
    RunloopSandboxEnvironment,
    _run_id,
    _running_sandboxes,
)
from inspect_sandboxes.runloop._single_env import RunloopSingleServiceEnvironment


def make_mock_devbox(devbox_id: str = "dbx-test-123") -> MagicMock:
    devbox = MagicMock()
    devbox.id = devbox_id
    return devbox


def _make_async_iter(items: list[Any]) -> Any:
    async def _aiter():
        for item in items:
            yield item

    return _aiter()


def make_mock_client(devbox: MagicMock) -> MagicMock:
    client = MagicMock()
    client.devboxes = MagicMock()
    client.devboxes.create_and_await_running = AsyncMock(return_value=devbox)
    client.devboxes.shutdown = AsyncMock()
    client.devboxes.list = MagicMock(return_value=_make_async_iter([]))
    client.blueprints = MagicMock()
    client.blueprints.list = MagicMock(return_value=_make_async_iter([]))
    created_bp = MagicMock()
    created_bp.id = "bp_test_created"
    client.blueprints.create = AsyncMock(return_value=created_bp)
    client.blueprints.await_build_complete = AsyncMock()
    client.with_options = MagicMock(return_value=client)

    created_object = MagicMock()
    created_object.id = "obj_test_123"
    created_object.upload_url = "https://upload.example/test"
    client.objects = MagicMock()
    client.objects.create = AsyncMock(return_value=created_object)
    client.objects.complete = AsyncMock()
    client.objects.delete = AsyncMock()
    return client


def _patch_blueprint_httpx() -> Any:
    """Patch the ``httpx.AsyncClient`` used by ``_blueprint._upload_build_context``."""
    response = MagicMock()
    response.raise_for_status = MagicMock()

    http_client = MagicMock()
    http_client.put = AsyncMock(return_value=response)
    http_client.__aenter__ = AsyncMock(return_value=http_client)
    http_client.__aexit__ = AsyncMock(return_value=None)
    return patch(
        "inspect_sandboxes.runloop._blueprint.httpx.AsyncClient",
        return_value=http_client,
    )


@pytest.fixture
def mock_devbox() -> MagicMock:
    return make_mock_devbox()


@pytest.fixture
def mock_client(mock_devbox: MagicMock) -> MagicMock:
    return make_mock_client(mock_devbox)


@pytest.mark.asyncio
async def test_full_lifecycle(
    mock_client: MagicMock,
) -> None:
    """task_init → sample_init(config=None) → sample_cleanup → task_cleanup."""
    with patch(
        "inspect_sandboxes.runloop._runloop.AsyncRunloop", return_value=mock_client
    ):
        await RunloopSandboxEnvironment.task_init("test_task", None)

        envs = await RunloopSandboxEnvironment.sample_init("test_task", None, {})

        assert "default" in envs
        assert isinstance(envs["default"], RunloopSingleServiceEnvironment)
        assert _running_sandboxes.get() == ["dbx-test-123"]

        await_args = mock_client.devboxes.create_and_await_running.await_args
        assert await_args is not None
        create_kwargs = await_args.kwargs
        assert "blueprint_id" not in create_kwargs
        assert "blueprint_name" not in create_kwargs
        assert create_kwargs["metadata"]["created_by"] == "inspect-ai"
        assert "inspect_run_id" in create_kwargs["metadata"]
        assert create_kwargs["metadata"]["task"] == "test_task"

        await RunloopSandboxEnvironment.sample_cleanup("test_task", None, envs, False)
        mock_client.devboxes.shutdown.assert_any_await("dbx-test-123")
        assert _running_sandboxes.get() == []

        mock_client.devboxes.shutdown.reset_mock()
        # Re-arm list for the orphan pass.
        mock_client.devboxes.list = MagicMock(return_value=_make_async_iter([]))
        await RunloopSandboxEnvironment.task_cleanup("test_task", None, cleanup=True)
        # Both passes: pass-1 has nothing tracked, pass-2 list returns no items.
        mock_client.devboxes.shutdown.assert_not_awaited()
        assert _running_sandboxes.get() == []


@pytest.mark.asyncio
async def test_task_init_initializes_context(mock_client: MagicMock) -> None:
    with patch(
        "inspect_sandboxes.runloop._runloop.AsyncRunloop", return_value=mock_client
    ):
        await RunloopSandboxEnvironment.task_init("test_task", None)

    assert _running_sandboxes.get() == []
    assert len(_run_id.get()) == 32


@pytest.mark.asyncio
async def test_dockerfile_config_builds_blueprint(
    mock_client: MagicMock,
    tmp_path: Any,
) -> None:
    """Dockerfile config builds a blueprint and creates a devbox referencing it."""
    dockerfile = tmp_path / "Dockerfile"
    dockerfile.write_text("FROM python:3.12\n")
    with (
        patch(
            "inspect_sandboxes.runloop._runloop.AsyncRunloop", return_value=mock_client
        ),
        _patch_blueprint_httpx(),
    ):
        await RunloopSandboxEnvironment.task_init("t", None)
        envs = await RunloopSandboxEnvironment.sample_init("t", str(dockerfile), {})

    assert "default" in envs
    # Blueprint was built (list returned no cached match → create called).
    mock_client.blueprints.create.assert_awaited_once()
    bp_kwargs = mock_client.blueprints.create.await_args.kwargs
    assert bp_kwargs["dockerfile"] == "FROM python:3.12\n"
    assert bp_kwargs["name"].startswith("inspect-")
    assert bp_kwargs["build_context"] == {"object_id": "obj_test_123", "type": "object"}

    # Devbox creation referenced that blueprint by name.
    create_kwargs = mock_client.devboxes.create_and_await_running.await_args.kwargs
    assert create_kwargs["blueprint_name"] == bp_kwargs["name"]


@pytest.mark.asyncio
async def test_single_service_compose_image_builds_blueprint(
    mock_client: MagicMock,
    tmp_path: Any,
) -> None:
    compose = tmp_path / "compose.yaml"
    compose.write_text(
        "services:\n  default:\n    image: python:3.12\n    environment:\n"
        "      - FOO=bar\n"
        "x-runloop:\n  timeout: 1234\n"
    )
    with patch(
        "inspect_sandboxes.runloop._runloop.AsyncRunloop", return_value=mock_client
    ):
        await RunloopSandboxEnvironment.task_init("t", None)
        await RunloopSandboxEnvironment.sample_init("t", str(compose), {})

    mock_client.blueprints.create.assert_awaited_once()
    bp_kwargs = mock_client.blueprints.create.await_args.kwargs
    assert bp_kwargs["dockerfile"] == "FROM python:3.12\n"

    create_kwargs = mock_client.devboxes.create_and_await_running.await_args.kwargs
    assert create_kwargs["blueprint_name"] == bp_kwargs["name"]
    assert create_kwargs["environment_variables"] == {"FOO": "bar"}
    assert create_kwargs["timeout"] == 1234.0


@pytest.mark.asyncio
async def test_x_runloop_blueprint_name_skips_build(
    mock_client: MagicMock,
    tmp_path: Any,
) -> None:
    compose = tmp_path / "compose.yaml"
    compose.write_text(
        "services:\n  default:\n    image: alpine\n"
        "x-runloop:\n  blueprint_name: my-prebuilt\n"
    )
    with patch(
        "inspect_sandboxes.runloop._runloop.AsyncRunloop", return_value=mock_client
    ):
        await RunloopSandboxEnvironment.task_init("t", None)
        await RunloopSandboxEnvironment.sample_init("t", str(compose), {})

    mock_client.blueprints.create.assert_not_called()
    create_kwargs = mock_client.devboxes.create_and_await_running.await_args.kwargs
    assert create_kwargs["blueprint_name"] == "my-prebuilt"


@pytest.mark.asyncio
async def test_x_runloop_blueprint_id_skips_build(
    mock_client: MagicMock,
    tmp_path: Any,
) -> None:
    compose = tmp_path / "compose.yaml"
    compose.write_text(
        "services:\n  default:\n    image: alpine\nx-runloop:\n  blueprint_id: bp_xyz\n"
    )
    with patch(
        "inspect_sandboxes.runloop._runloop.AsyncRunloop", return_value=mock_client
    ):
        await RunloopSandboxEnvironment.task_init("t", None)
        await RunloopSandboxEnvironment.sample_init("t", str(compose), {})

    mock_client.blueprints.create.assert_not_called()
    create_kwargs = mock_client.devboxes.create_and_await_running.await_args.kwargs
    assert create_kwargs["blueprint_id"] == "bp_xyz"


@pytest.mark.asyncio
async def test_sample_init_multi_service_routes_to_dind(
    mock_client: MagicMock,
    tmp_path: Any,
) -> None:
    """Multi-service compose calls into the DinD orchestrator with mapped params."""
    from inspect_sandboxes.runloop._dind_env import RunloopDinDServiceEnvironment
    from inspect_sandboxes.runloop._dind_project import RunloopDinDProject

    compose = tmp_path / "compose.yaml"
    compose.write_text(
        "services:\n  a:\n    image: alpine\n  b:\n    image: ubuntu\n"
        "x-runloop:\n  timeout: 1800\n"
    )

    project = RunloopDinDProject(
        client=mock_client,
        devbox_id="dbx-dind-1",
        project_name="inspect-x",
        compose_path="/home/user/inspect/compose/compose.yaml",
        services=["a", "b"],
    )
    real_envs = {
        "a": RunloopDinDServiceEnvironment(project, "a", "/"),
        "b": RunloopDinDServiceEnvironment(project, "b", "/"),
    }
    with (
        patch(
            "inspect_sandboxes.runloop._runloop.AsyncRunloop", return_value=mock_client
        ),
        patch.object(
            RunloopDinDServiceEnvironment,
            "sample_init_dind",
            new=AsyncMock(return_value=real_envs),
        ) as init_dind,
    ):
        await RunloopSandboxEnvironment.task_init("test_task", None)
        envs = await RunloopSandboxEnvironment.sample_init(
            "test_task", str(compose), {}
        )
        assert set(envs.keys()) == {"a", "b"}

        assert init_dind.await_args is not None
        kwargs = init_dind.await_args.kwargs
        assert kwargs["metadata"]["created_by"] == "inspect-ai"


@pytest.mark.asyncio
async def test_sample_init_resolves_sample_override_when_task_config_none(
    mock_client: MagicMock,
    tmp_path: Any,
) -> None:
    """sample_init resolves the blueprint from the sample's config even when task_init saw None."""
    dockerfile = tmp_path / "Dockerfile"
    dockerfile.write_text("FROM python:3.12\n")
    with (
        patch(
            "inspect_sandboxes.runloop._runloop.AsyncRunloop", return_value=mock_client
        ),
        _patch_blueprint_httpx(),
    ):
        await RunloopSandboxEnvironment.task_init("t", None)
        envs = await RunloopSandboxEnvironment.sample_init("t", str(dockerfile), {})

    assert "default" in envs
    create_kwargs = mock_client.devboxes.create_and_await_running.await_args.kwargs
    assert "blueprint_name" in create_kwargs
    assert create_kwargs["blueprint_name"].startswith("inspect-")


@pytest.mark.asyncio
async def test_sample_init_uses_sample_config_when_differs_from_task(
    mock_client: MagicMock,
    tmp_path: Any,
) -> None:
    """sample_init uses the sample's Dockerfile even when task_init saw a different one."""
    dockerfile_python = tmp_path / "Dockerfile.python"
    dockerfile_python.write_text("FROM python:3.12\n")
    dockerfile_alpine = tmp_path / "Dockerfile.alpine"
    dockerfile_alpine.write_text("FROM alpine:3.20\n")

    def fake_build(_client: Any, path: str, **_: Any) -> str:
        return f"inspect-{path.rsplit('/', 1)[-1]}"

    with (
        patch(
            "inspect_sandboxes.runloop._runloop.AsyncRunloop", return_value=mock_client
        ),
        patch(
            "inspect_sandboxes.runloop._runloop.build_blueprint_for_dockerfile",
            side_effect=fake_build,
        ),
    ):
        await RunloopSandboxEnvironment.task_init("t", str(dockerfile_python))
        await RunloopSandboxEnvironment.sample_init("t", str(dockerfile_alpine), {})

    create_kwargs = mock_client.devboxes.create_and_await_running.await_args.kwargs
    assert create_kwargs["blueprint_name"] == "inspect-Dockerfile.alpine"


@pytest.mark.asyncio
async def test_sample_init_invalid_config(
    mock_client: MagicMock,
) -> None:
    with patch(
        "inspect_sandboxes.runloop._runloop.AsyncRunloop", return_value=mock_client
    ):
        await RunloopSandboxEnvironment.task_init("test_task", None)
        bad_config: Any = 12345
        with pytest.raises(ValueError, match="Unrecognized config"):
            await RunloopSandboxEnvironment.sample_init(
                "test_task",
                bad_config,
                {},
            )


@pytest.mark.asyncio
async def test_task_cleanup_no_op_when_cleanup_false(
    mock_client: MagicMock,
) -> None:
    with patch(
        "inspect_sandboxes.runloop._runloop.AsyncRunloop", return_value=mock_client
    ):
        await RunloopSandboxEnvironment.task_init("t", None)
        await RunloopSandboxEnvironment.task_cleanup("t", None, cleanup=False)
        mock_client.devboxes.shutdown.assert_not_called()


@pytest.mark.asyncio
async def test_task_cleanup_kills_orphaned_devboxes(
    mock_client: MagicMock,
) -> None:
    """A devbox tagged with this run's metadata but never tracked should still be shut down."""
    orphan = MagicMock()
    orphan.id = "dbx-orphan-1"
    mock_client.devboxes.list = MagicMock(return_value=_make_async_iter([orphan]))
    with patch(
        "inspect_sandboxes.runloop._runloop.AsyncRunloop", return_value=mock_client
    ):
        await RunloopSandboxEnvironment.task_init("t", None)
        # The list mock filters client-side via metadata, so tag with the live run_id.
        orphan.metadata = {"inspect_run_id": _run_id.get()}
        await RunloopSandboxEnvironment.task_cleanup("t", None, cleanup=True)
        mock_client.devboxes.shutdown.assert_any_await("dbx-orphan-1")


@pytest.mark.asyncio
async def test_task_cleanup_skips_already_killed_in_orphan_pass(
    mock_client: MagicMock,
) -> None:
    """Tracked devbox shut down in first pass; metadata listing returns it too."""
    same = MagicMock()
    same.id = "dbx-test-123"
    mock_client.devboxes.list = MagicMock(return_value=_make_async_iter([same]))
    with patch(
        "inspect_sandboxes.runloop._runloop.AsyncRunloop", return_value=mock_client
    ):
        await RunloopSandboxEnvironment.task_init("t", None)
        same.metadata = {"inspect_run_id": _run_id.get()}
        envs = await RunloopSandboxEnvironment.sample_init("t", None, {})
        assert envs

        await RunloopSandboxEnvironment.task_cleanup("t", None, cleanup=True)
        # The tracked devbox is shut down exactly once,
        # not a second time during the orphan pass.
        shutdown_calls = [
            c.args[0] for c in mock_client.devboxes.shutdown.await_args_list
        ]
        assert shutdown_calls.count("dbx-test-123") == 1


@pytest.mark.asyncio
async def test_task_cleanup_continues_when_list_fails(mock_client: MagicMock) -> None:
    """If listing orphans throws, task_cleanup should warn and continue."""
    mock_client.devboxes.list = MagicMock(side_effect=RuntimeError("api blew up"))
    with patch(
        "inspect_sandboxes.runloop._runloop.AsyncRunloop", return_value=mock_client
    ):
        await RunloopSandboxEnvironment.task_init("t", None)
        # Should not raise.
        await RunloopSandboxEnvironment.task_cleanup("t", None, cleanup=True)


@pytest.mark.asyncio
async def test_devbox_metadata_include_run_id(mock_client: MagicMock) -> None:
    """Test that devbox metadata includes inspect_run_id."""
    with patch(
        "inspect_sandboxes.runloop._runloop.AsyncRunloop", return_value=mock_client
    ):
        await RunloopSandboxEnvironment.task_init("test_task", None)
        await RunloopSandboxEnvironment.sample_init("test_task", None, {})

    create_kwargs = mock_client.devboxes.create_and_await_running.await_args.kwargs
    assert create_kwargs["metadata"]["inspect_run_id"] == _run_id.get()


@pytest.mark.asyncio
async def test_cli_cleanup_single_success(mock_client: MagicMock) -> None:
    with patch(
        "inspect_sandboxes.runloop._runloop.AsyncRunloop", return_value=mock_client
    ):
        await RunloopSandboxEnvironment.cli_cleanup("dbx-x")
        mock_client.devboxes.shutdown.assert_awaited_once_with("dbx-x")


@pytest.mark.asyncio
async def test_cli_cleanup_single_failure(mock_client: MagicMock) -> None:
    """Test CLI cleanup failure exits with code 1."""
    mock_client.devboxes.shutdown = AsyncMock(side_effect=Exception("shutdown failed"))
    with patch(
        "inspect_sandboxes.runloop._runloop.AsyncRunloop", return_value=mock_client
    ):
        with pytest.raises(SystemExit) as exc_info:
            await RunloopSandboxEnvironment.cli_cleanup("dbx-x")

    assert exc_info.value.code == 1


@pytest.mark.asyncio
async def test_cli_cleanup_bulk_no_devboxes(mock_client: MagicMock) -> None:
    mock_client.devboxes.list = MagicMock(return_value=_make_async_iter([]))
    with patch(
        "inspect_sandboxes.runloop._runloop.AsyncRunloop", return_value=mock_client
    ):
        await RunloopSandboxEnvironment.cli_cleanup(None)
        mock_client.devboxes.shutdown.assert_not_called()


@pytest.mark.asyncio
async def test_cli_cleanup_bulk_with_devboxes(mock_client: MagicMock) -> None:
    a = MagicMock()
    a.id = "dbx-a"
    a.metadata = {"created_by": "inspect-ai"}
    b = MagicMock()
    b.id = "dbx-b"
    b.metadata = {"created_by": "inspect-ai"}
    mock_client.devboxes.list = MagicMock(return_value=_make_async_iter([a, b]))
    with patch(
        "inspect_sandboxes.runloop._runloop.AsyncRunloop", return_value=mock_client
    ):
        await RunloopSandboxEnvironment.cli_cleanup(None)
        shut = sorted(c.args[0] for c in mock_client.devboxes.shutdown.await_args_list)
        assert shut == ["dbx-a", "dbx-b"]


@pytest.mark.asyncio
async def test_cli_cleanup_bulk_partial_failure(mock_client: MagicMock) -> None:
    """Test CLI bulk cleanup exits with code 1 on partial failure."""
    a = MagicMock()
    a.id = "dbx-a"
    a.metadata = {"created_by": "inspect-ai"}
    b = MagicMock()
    b.id = "dbx-b"
    b.metadata = {"created_by": "inspect-ai"}
    mock_client.devboxes.list = MagicMock(return_value=_make_async_iter([a, b]))

    async def failing_shutdown(devbox_id: str) -> bool:
        if devbox_id == "dbx-a":
            raise Exception("shutdown failed")
        return True

    mock_client.devboxes.shutdown = AsyncMock(side_effect=failing_shutdown)

    with patch(
        "inspect_sandboxes.runloop._runloop.AsyncRunloop", return_value=mock_client
    ):
        with pytest.raises(SystemExit) as exc_info:
            await RunloopSandboxEnvironment.cli_cleanup(None)

    assert exc_info.value.code == 1


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
async def runloop_single_env() -> AsyncGenerator[SandboxEnvironment, None]:
    """Create a real single-service Runloop devbox (default image)."""
    await RunloopSandboxEnvironment.task_init("test_self_check", None)
    envs = await RunloopSandboxEnvironment.sample_init("test_self_check", None, {})
    yield envs["default"]
    try:
        await RunloopSandboxEnvironment.sample_cleanup(
            "test_self_check", None, envs, False
        )
        await RunloopSandboxEnvironment.task_cleanup(
            "test_self_check", None, cleanup=True
        )
    except Exception as e:
        print(f"Cleanup error: {e}")


@pytest.mark.asyncio
@pytest.mark.integration
async def test_self_check_single_service(
    runloop_single_env: SandboxEnvironment,
) -> None:
    """Run inspect_ai's self-check suite against a single-service Runloop devbox."""
    known_failures = [
        # Runloop's default image may not have useradd preinstalled in all variants.
        "test_exec_as_user",
        # Runloop's runtime SIGKILLs the process during signal-exit cleanup, so
        # the shell's self-SIGTERM (kill -TERM $$) surfaces as -137 (SIGKILL),
        # not 143 (128+SIGTERM). Platform behavior, not fixable client-side.
        "test_exec_timeout_not_raised_on_fast_signal_death",
    ]
    results = await self_check(runloop_single_env)
    _check_self_check_results(results, known_failures)


@pytest_asyncio.fixture
async def runloop_dind_env() -> AsyncGenerator[SandboxEnvironment, None]:
    """Create a real DinD Runloop devbox (two-service compose)."""
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
    await RunloopSandboxEnvironment.task_init("test_self_check_dind", None)
    envs = await RunloopSandboxEnvironment.sample_init(
        "test_self_check_dind", config, {}
    )
    yield envs["default"]
    try:
        await RunloopSandboxEnvironment.sample_cleanup(
            "test_self_check_dind", config, envs, False
        )
        await RunloopSandboxEnvironment.task_cleanup(
            "test_self_check_dind", None, cleanup=True
        )
    except Exception as e:
        print(f"Cleanup error: {e}")


@pytest.mark.asyncio
@pytest.mark.integration
async def test_self_check_dind(
    runloop_dind_env: SandboxEnvironment,
) -> None:
    """Run inspect_ai's self-check suite against a DinD Runloop devbox."""
    known_failures = [
        # docker compose exec routes through sh; permission/output edges differ.
        "test_exec_permission_error",
        "test_write_text_file_without_permissions",
        "test_write_binary_file_without_permissions",
        "test_read_file_not_allowed",
        "test_exec_as_user",
    ]
    results = await self_check(runloop_dind_env)
    _check_self_check_results(results, known_failures)
