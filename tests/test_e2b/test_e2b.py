"""Tests for E2BSandboxEnvironment lifecycle orchestrator."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from inspect_ai.util import ComposeConfig, ComposeService, SandboxEnvironment
from inspect_ai.util._sandbox.self_check import self_check
from inspect_sandboxes.e2b._e2b import (
    E2BSandboxEnvironment,
    _run_id,
    _running_sandboxes,
)
from inspect_sandboxes.e2b._single_env import E2BSingleServiceEnvironment


def make_mock_sandbox(sandbox_id: str = "sb-test-123") -> MagicMock:
    """Create a mock AsyncSandbox instance."""
    sandbox = MagicMock()
    sandbox.sandbox_id = sandbox_id
    sandbox.kill = AsyncMock()
    return sandbox


def make_mock_async_sandbox_cls(
    sandbox: MagicMock, list_items: tuple[MagicMock, ...] = ()
) -> MagicMock:
    """Create a mock AsyncSandbox *class*. Models the static methods used by _e2b.py."""
    cls = MagicMock()
    cls.create = AsyncMock(return_value=sandbox)
    cls.kill = AsyncMock()
    paginator = MagicMock()
    paginator.next_items = AsyncMock(return_value=list(list_items))
    paginator.has_next = False
    cls.list = MagicMock(return_value=paginator)
    return cls


@pytest.fixture
def mock_sandbox() -> MagicMock:
    return make_mock_sandbox()


@pytest.mark.asyncio
async def test_full_lifecycle(mock_sandbox: MagicMock) -> None:
    """task_init → sample_init(config=None) → sample_cleanup → task_cleanup."""
    mock_cls = make_mock_async_sandbox_cls(mock_sandbox)
    with patch("inspect_sandboxes.e2b._e2b.AsyncSandbox", new=mock_cls):
        await E2BSandboxEnvironment.task_init("test_task", None)

        envs = await E2BSandboxEnvironment.sample_init("test_task", None, {})

        assert "default" in envs
        assert isinstance(envs["default"], E2BSingleServiceEnvironment)
        assert _running_sandboxes.get() == ["sb-test-123"]

        # Default base template (template=None) was requested.
        create_kwargs = mock_cls.create.await_args.kwargs
        assert create_kwargs["template"] is None
        assert create_kwargs["metadata"]["created_by"] == "inspect-ai"
        assert "inspect_run_id" in create_kwargs["metadata"]
        assert create_kwargs["metadata"]["task"] == "test_task"

        await E2BSandboxEnvironment.sample_cleanup("test_task", None, envs, False)
        mock_sandbox.kill.assert_awaited_once()

        await E2BSandboxEnvironment.task_cleanup("test_task", None, cleanup=True)
        # Tracked sandbox is killed via AsyncSandbox.kill(sandbox_id).
        mock_cls.kill.assert_any_await("sb-test-123")
        assert _running_sandboxes.get() == []


@pytest.mark.asyncio
async def test_task_init_initializes_context(mock_sandbox: MagicMock) -> None:
    mock_cls = make_mock_async_sandbox_cls(mock_sandbox)
    with patch("inspect_sandboxes.e2b._e2b.AsyncSandbox", new=mock_cls):
        await E2BSandboxEnvironment.task_init("test_task", None)

    assert _running_sandboxes.get() == []
    assert len(_run_id.get()) == 32


@pytest.mark.asyncio
async def test_dockerfile_config_builds_template_in_task_init(
    mock_sandbox: MagicMock,
    tmp_path: Any,
) -> None:
    """task_init builds the template once; sample_init passes the cached name."""
    dockerfile = tmp_path / "Dockerfile"
    dockerfile.write_text("FROM python:3.12\n")
    mock_cls = make_mock_async_sandbox_cls(mock_sandbox)
    with (
        patch("inspect_sandboxes.e2b._e2b.AsyncSandbox", new=mock_cls),
        patch("inspect_sandboxes.e2b._e2b.build_template_for_dockerfile") as build,
    ):
        build.return_value = "inspect-abc123"

        await E2BSandboxEnvironment.task_init("test_task", str(dockerfile))
        build.assert_awaited_once_with(str(dockerfile))

        envs = await E2BSandboxEnvironment.sample_init("test_task", str(dockerfile), {})
        assert "default" in envs
        create_kwargs = mock_cls.create.await_args.kwargs
        assert create_kwargs["template"] == "inspect-abc123"


@pytest.mark.asyncio
async def test_single_service_compose_image_builds_template(
    mock_sandbox: MagicMock,
    tmp_path: Any,
) -> None:
    compose = tmp_path / "compose.yaml"
    compose.write_text(
        "services:\n  default:\n    image: python:3.12\n    environment:\n"
        "      - FOO=bar\n"
        "x-e2b:\n  timeout: 1234\n"
    )
    mock_cls = make_mock_async_sandbox_cls(mock_sandbox)
    with (
        patch("inspect_sandboxes.e2b._e2b.AsyncSandbox", new=mock_cls),
        patch("inspect_sandboxes.e2b._e2b.build_template_for_image") as build,
    ):
        build.return_value = "inspect-image-hash"

        await E2BSandboxEnvironment.task_init("test_task", str(compose))
        build.assert_awaited_once()
        assert build.await_args is not None
        assert build.await_args.args[0] == "python:3.12"

        await E2BSandboxEnvironment.sample_init("test_task", str(compose), {})
        create_kwargs = mock_cls.create.await_args.kwargs
        assert create_kwargs["template"] == "inspect-image-hash"
        assert create_kwargs["envs"] == {"FOO": "bar"}
        assert create_kwargs["timeout"] == 1234.0


@pytest.mark.asyncio
async def test_single_service_compose_with_x_e2b_template_skips_build(
    mock_sandbox: MagicMock,
    tmp_path: Any,
) -> None:
    compose = tmp_path / "compose.yaml"
    compose.write_text(
        "services:\n  default:\n    image: alpine\nx-e2b:\n  template: my-prebuilt\n"
    )
    mock_cls = make_mock_async_sandbox_cls(mock_sandbox)
    with (
        patch("inspect_sandboxes.e2b._e2b.AsyncSandbox", new=mock_cls),
        patch("inspect_sandboxes.e2b._e2b.build_template_for_image") as bi,
        patch("inspect_sandboxes.e2b._e2b.build_template_for_dockerfile") as bd,
    ):
        await E2BSandboxEnvironment.task_init("test_task", str(compose))
        bi.assert_not_called()
        bd.assert_not_called()

        await E2BSandboxEnvironment.sample_init("test_task", str(compose), {})
        assert mock_cls.create.await_args.kwargs["template"] == "my-prebuilt"


@pytest.mark.asyncio
async def test_sample_init_multi_service_routes_to_dind(
    mock_sandbox: MagicMock,
    tmp_path: Any,
) -> None:
    """Multi-service compose calls into the DinD orchestrator with mapped params."""
    from inspect_sandboxes.e2b._dind_env import E2BDinDServiceEnvironment
    from inspect_sandboxes.e2b._dind_project import E2BDinDProject

    compose = tmp_path / "compose.yaml"
    compose.write_text(
        "services:\n  a:\n    image: alpine\n  b:\n    image: ubuntu\n"
        "x-e2b:\n  cpu_count: 4\n  memory_mb: 8192\n  timeout: 1800\n"
    )

    project_sandbox = MagicMock()
    project_sandbox.sandbox_id = "sb-dind-1"
    project = E2BDinDProject(
        sandbox=project_sandbox,
        project_name="inspect-x",
        compose_path="/inspect/compose/compose.yaml",
        services=["a", "b"],
    )
    real_envs = {
        "a": E2BDinDServiceEnvironment(project, "a", "/"),
        "b": E2BDinDServiceEnvironment(project, "b", "/"),
    }
    mock_cls = make_mock_async_sandbox_cls(mock_sandbox)
    with (
        patch("inspect_sandboxes.e2b._e2b.AsyncSandbox", new=mock_cls),
        patch.object(
            E2BDinDServiceEnvironment,
            "sample_init_dind",
            new=AsyncMock(return_value=real_envs),
        ) as init_dind,
    ):
        await E2BSandboxEnvironment.task_init("test_task", str(compose))
        envs = await E2BSandboxEnvironment.sample_init("test_task", str(compose), {})
        assert set(envs.keys()) == {"a", "b"}

        assert init_dind.await_args is not None
        kwargs = init_dind.await_args.kwargs
        assert kwargs["cpu_count"] == 4
        assert kwargs["memory_mb"] == 8192
        assert kwargs["sandbox_timeout"] == 1800.0
        assert kwargs["metadata"]["created_by"] == "inspect-ai"


@pytest.mark.asyncio
async def test_sample_init_resolves_sample_override_when_task_config_none(
    mock_sandbox: MagicMock,
    tmp_path: Any,
) -> None:
    """sample_init resolves the template from the sample's config even when task_init saw None."""
    dockerfile = tmp_path / "Dockerfile"
    dockerfile.write_text("FROM python:3.12\n")
    mock_cls = make_mock_async_sandbox_cls(mock_sandbox)
    with (
        patch("inspect_sandboxes.e2b._e2b.AsyncSandbox", new=mock_cls),
        patch("inspect_sandboxes.e2b._e2b.build_template_for_dockerfile") as build,
    ):
        build.return_value = "inspect-sample-only"

        await E2BSandboxEnvironment.task_init("test_task", None)
        envs = await E2BSandboxEnvironment.sample_init("test_task", str(dockerfile), {})

        assert "default" in envs
        create_kwargs = mock_cls.create.await_args.kwargs
        assert create_kwargs["template"] == "inspect-sample-only"


@pytest.mark.asyncio
async def test_sample_init_uses_sample_config_when_differs_from_task(
    mock_sandbox: MagicMock,
    tmp_path: Any,
) -> None:
    """sample_init uses the sample's Dockerfile even when task_init saw a different one."""
    dockerfile_python = tmp_path / "Dockerfile.python"
    dockerfile_python.write_text("FROM python:3.12\n")
    dockerfile_alpine = tmp_path / "Dockerfile.alpine"
    dockerfile_alpine.write_text("FROM alpine:3.20\n")

    def fake_build(path: str, **_: Any) -> str:
        return f"inspect-{path.rsplit('/', 1)[-1]}"

    mock_cls = make_mock_async_sandbox_cls(mock_sandbox)
    with (
        patch("inspect_sandboxes.e2b._e2b.AsyncSandbox", new=mock_cls),
        patch(
            "inspect_sandboxes.e2b._e2b.build_template_for_dockerfile",
            side_effect=fake_build,
        ),
    ):
        await E2BSandboxEnvironment.task_init("t", str(dockerfile_python))
        await E2BSandboxEnvironment.sample_init("t", str(dockerfile_alpine), {})

    create_kwargs = mock_cls.create.await_args.kwargs
    assert create_kwargs["template"] == "inspect-Dockerfile.alpine"


@pytest.mark.asyncio
async def test_sample_init_invalid_config(mock_sandbox: MagicMock) -> None:
    mock_cls = make_mock_async_sandbox_cls(mock_sandbox)
    with patch("inspect_sandboxes.e2b._e2b.AsyncSandbox", new=mock_cls):
        await E2BSandboxEnvironment.task_init("test_task", None)
        bad_config: Any = 12345
        with pytest.raises(ValueError, match="Unrecognized config"):
            await E2BSandboxEnvironment.sample_init(
                "test_task",
                bad_config,
                {},
            )


@pytest.mark.asyncio
async def test_task_cleanup_no_op_when_cleanup_false(mock_sandbox: MagicMock) -> None:
    mock_cls = make_mock_async_sandbox_cls(mock_sandbox)
    with patch("inspect_sandboxes.e2b._e2b.AsyncSandbox", new=mock_cls):
        await E2BSandboxEnvironment.task_init("t", None)
        await E2BSandboxEnvironment.task_cleanup("t", None, cleanup=False)
        mock_cls.kill.assert_not_called()


@pytest.mark.asyncio
async def test_task_cleanup_kills_orphaned_sandboxes(mock_sandbox: MagicMock) -> None:
    """A sandbox tagged with this run's metadata but never tracked should still be killed."""
    orphan = MagicMock()
    orphan.sandbox_id = "sb-orphan-1"
    mock_cls = make_mock_async_sandbox_cls(mock_sandbox, list_items=(orphan,))
    with patch("inspect_sandboxes.e2b._e2b.AsyncSandbox", new=mock_cls):
        await E2BSandboxEnvironment.task_init("t", None)
        await E2BSandboxEnvironment.task_cleanup("t", None, cleanup=True)
        mock_cls.kill.assert_any_await("sb-orphan-1")


@pytest.mark.asyncio
async def test_task_cleanup_skips_already_killed_in_orphan_pass(
    mock_sandbox: MagicMock,
) -> None:
    """Tracked sandbox killed in first pass; metadata listing returns it too."""
    same = MagicMock()
    same.sandbox_id = "sb-test-123"
    mock_cls = make_mock_async_sandbox_cls(mock_sandbox, list_items=(same,))
    with patch("inspect_sandboxes.e2b._e2b.AsyncSandbox", new=mock_cls):
        await E2BSandboxEnvironment.task_init("t", None)
        envs = await E2BSandboxEnvironment.sample_init("t", None, {})
        assert envs

        await E2BSandboxEnvironment.task_cleanup("t", None, cleanup=True)
        # The tracked sandbox is killed exactly once via AsyncSandbox.kill (string id),
        # not a second time during the orphan pass.
        kill_calls = [c.args[0] for c in mock_cls.kill.await_args_list]
        assert kill_calls.count("sb-test-123") == 1


@pytest.mark.asyncio
async def test_task_cleanup_continues_when_list_fails(mock_sandbox: MagicMock) -> None:
    """If listing orphans throws, task_cleanup should warn and continue."""
    mock_cls = make_mock_async_sandbox_cls(mock_sandbox)
    mock_cls.list = MagicMock(side_effect=RuntimeError("api blew up"))
    with patch("inspect_sandboxes.e2b._e2b.AsyncSandbox", new=mock_cls):
        await E2BSandboxEnvironment.task_init("t", None)
        # Should not raise.
        await E2BSandboxEnvironment.task_cleanup("t", None, cleanup=True)


@pytest.mark.asyncio
async def test_sandbox_metadata_include_run_id(mock_sandbox: MagicMock) -> None:
    """Test that sandbox metadata includes inspect_run_id."""
    mock_cls = make_mock_async_sandbox_cls(mock_sandbox)
    with patch("inspect_sandboxes.e2b._e2b.AsyncSandbox", new=mock_cls):
        await E2BSandboxEnvironment.task_init("test_task", None)
        await E2BSandboxEnvironment.sample_init("test_task", None, {})

    create_kwargs = mock_cls.create.await_args.kwargs
    assert create_kwargs["metadata"]["inspect_run_id"] == _run_id.get()


@pytest.mark.asyncio
async def test_cli_cleanup_single_success(mock_sandbox: MagicMock) -> None:
    mock_cls = make_mock_async_sandbox_cls(mock_sandbox)
    with patch("inspect_sandboxes.e2b._e2b.AsyncSandbox", new=mock_cls):
        await E2BSandboxEnvironment.cli_cleanup("sb-x")
        mock_cls.kill.assert_awaited_once_with("sb-x")


@pytest.mark.asyncio
async def test_cli_cleanup_single_failure(mock_sandbox: MagicMock) -> None:
    """Test CLI cleanup failure exits with code 1."""
    mock_cls = make_mock_async_sandbox_cls(mock_sandbox)
    mock_cls.kill = AsyncMock(side_effect=Exception("kill failed"))

    with patch("inspect_sandboxes.e2b._e2b.AsyncSandbox", new=mock_cls):
        with pytest.raises(SystemExit) as exc_info:
            await E2BSandboxEnvironment.cli_cleanup("sb-x")

    assert exc_info.value.code == 1


@pytest.mark.asyncio
async def test_cli_cleanup_bulk_no_sandboxes(mock_sandbox: MagicMock) -> None:
    mock_cls = make_mock_async_sandbox_cls(mock_sandbox)
    with patch("inspect_sandboxes.e2b._e2b.AsyncSandbox", new=mock_cls):
        await E2BSandboxEnvironment.cli_cleanup(None)
        mock_cls.kill.assert_not_called()


@pytest.mark.asyncio
async def test_cli_cleanup_bulk_with_sandboxes(mock_sandbox: MagicMock) -> None:
    a = MagicMock()
    a.sandbox_id = "sb-a"
    b = MagicMock()
    b.sandbox_id = "sb-b"
    mock_cls = make_mock_async_sandbox_cls(mock_sandbox, list_items=(a, b))
    with patch("inspect_sandboxes.e2b._e2b.AsyncSandbox", new=mock_cls):
        await E2BSandboxEnvironment.cli_cleanup(None)
        killed = sorted(c.args[0] for c in mock_cls.kill.await_args_list)
        assert killed == ["sb-a", "sb-b"]


@pytest.mark.asyncio
async def test_cli_cleanup_bulk_partial_failure(mock_sandbox: MagicMock) -> None:
    """Test CLI bulk cleanup exits with code 1 on partial failure."""
    a = MagicMock()
    a.sandbox_id = "sb-a"
    b = MagicMock()
    b.sandbox_id = "sb-b"
    mock_cls = make_mock_async_sandbox_cls(mock_sandbox, list_items=(a, b))

    async def failing_kill(sandbox_id: str) -> bool:
        if sandbox_id == "sb-a":
            raise Exception("kill failed")
        return True

    mock_cls.kill = AsyncMock(side_effect=failing_kill)

    with patch("inspect_sandboxes.e2b._e2b.AsyncSandbox", new=mock_cls):
        with pytest.raises(SystemExit) as exc_info:
            await E2BSandboxEnvironment.cli_cleanup(None)

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
async def e2b_single_env() -> AsyncGenerator[SandboxEnvironment, None]:
    """Create a real single-service E2B sandbox (default `base` template)."""
    await E2BSandboxEnvironment.task_init("test_self_check", None)
    envs = await E2BSandboxEnvironment.sample_init("test_self_check", None, {})
    yield envs["default"]
    try:
        await E2BSandboxEnvironment.sample_cleanup("test_self_check", None, envs, False)
        await E2BSandboxEnvironment.task_cleanup("test_self_check", None, cleanup=True)
    except Exception as e:
        print(f"Cleanup error: {e}")


@pytest.mark.asyncio
@pytest.mark.integration
async def test_self_check_single_service(
    e2b_single_env: SandboxEnvironment,
) -> None:
    """Run inspect_ai's self-check suite against a single-service E2B sandbox."""
    known_failures = [
        "test_read_file_not_allowed",  # files.read doesn't translate permission denials to PermissionError
        "test_write_text_file_without_permissions",  # files.write returns HTTP 400 for permission errors, not translated
        "test_write_binary_file_without_permissions",  # same
        "test_exec_permission_error",  # exit code 126 from shell, not translated to PermissionError
        "test_exec_timeout_not_raised_on_fast_signal_death",  # E2B reports exit -1 for self-SIGTERM, not 143
        "test_exec_as_nonexistent_user",  # E2B raises AuthenticationException, not the inspect_ai-expected error
        "test_exec_as_user",  # default `base` template doesn't have useradd preinstalled
        "test_exec_timeout",  # flaky: httpx.ReadTimeout escapes outside the wrapped path (any of these 3 rotates)
        "test_exec_timeout_kills_process",  # same
        "test_exec_timeout_kills_child_processes",  # same
    ]
    results = await self_check(e2b_single_env)
    _check_self_check_results(results, known_failures)


@pytest_asyncio.fixture
async def e2b_dind_env() -> AsyncGenerator[SandboxEnvironment, None]:
    """Create a real DinD E2B sandbox (two-service compose, routes to DinD)."""
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
    await E2BSandboxEnvironment.task_init("test_self_check_dind", None)
    envs = await E2BSandboxEnvironment.sample_init("test_self_check_dind", config, {})
    yield envs["default"]
    try:
        await E2BSandboxEnvironment.sample_cleanup(
            "test_self_check_dind", config, envs, False
        )
        await E2BSandboxEnvironment.task_cleanup(
            "test_self_check_dind", None, cleanup=True
        )
    except Exception as e:
        print(f"Cleanup error: {e}")


@pytest.mark.asyncio
@pytest.mark.integration
async def test_self_check_dind(
    e2b_dind_env: SandboxEnvironment,
) -> None:
    """Run inspect_ai's self-check suite against a DinD E2B sandbox."""
    known_failures = [
        "test_exec_permission_error",  # docker compose exec routes through sh; permission edges differ
        "test_write_text_file_without_permissions",  # same
        "test_write_binary_file_without_permissions",  # same
        "test_read_file_not_allowed",  # same
        "test_exec_as_user",  # docker compose exec; user creation edge
        "test_exec_timeout",  # flaky: httpx.ReadTimeout escapes outside the wrapped path (any of these 3 rotates)
        "test_exec_timeout_kills_process",  # same
        "test_exec_timeout_kills_child_processes",  # same
    ]
    results = await self_check(e2b_dind_env)
    _check_self_check_results(results, known_failures)


@pytest_asyncio.fixture
async def e2b_ports_env() -> AsyncGenerator[SandboxEnvironment, None]:
    """Create a real single-service E2B sandbox that declares a port."""
    config = ComposeConfig(
        services={
            "default": ComposeService(
                image="python:3.12-slim",
                command="sleep infinity",
                ports=["8080:8080"],
            )
        }
    )
    await E2BSandboxEnvironment.task_init("test_connection_ports", None)
    envs = await E2BSandboxEnvironment.sample_init("test_connection_ports", config, {})
    yield envs["default"]
    try:
        await E2BSandboxEnvironment.sample_cleanup(
            "test_connection_ports", config, envs, False
        )
        await E2BSandboxEnvironment.task_cleanup(
            "test_connection_ports", None, cleanup=True
        )
    except Exception as e:
        print(f"Cleanup error: {e}")


@pytest.mark.asyncio
@pytest.mark.integration
async def test_connection_surfaces_declared_port(
    e2b_ports_env: SandboxEnvironment,
) -> None:
    """A real single-service E2B sandbox surfaces a Compose-declared port.

    Exercises the live ``get_host()`` path that the unit tests mock: the port
    declared in the Compose config must come back through
    ``SandboxConnection.ports`` as an E2B host URL.
    """
    conn = await e2b_ports_env.connection()

    assert conn.type == "e2b"
    assert conn.ports is not None, "expected the declared port to be surfaced"
    port = next((p for p in conn.ports if p.container_port == 8080), None)
    assert port is not None, (
        f"container port 8080 missing from {[p.container_port for p in conn.ports]}"
    )
    assert port.protocol == "tcp"
    assert port.mappings, "expected at least one host mapping"
    host = port.mappings[0]
    assert isinstance(host.host_ip, str) and host.host_ip, "expected a real host"
    assert host.host_port == 443
