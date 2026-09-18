"""Tests for Daytona sandbox create-retry, zombie-reap and session exec helpers."""

import asyncio
import shlex
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import anyio
import pytest
from daytona import (
    CreateSandboxFromImageParams,
    CreateSandboxFromSnapshotParams,
    DaytonaError,
    DaytonaNotFoundError,
    DaytonaTimeoutError,
)
from inspect_sandboxes.daytona._sandbox_utils import (
    CREATE_SANDBOX_ATTEMPTS,
    SHELL_PATH,
    SYSTEM_PATH,
    TIMEOUT_GRACE,
    TIMEOUT_PATH,
    SessionPool,
    _respin_create_params,
    build_remove_command,
    build_session_command,
    create_sandbox,
    reap_zombie_sandboxes,
    reset_zombie_registry,
    session_exec,
    session_exec_result,
    zombie_registry,
)


def test_respin_create_params_swaps_existing_uuid_suffix() -> None:
    """An existing 8-char hex suffix is swapped, keeping the name length."""
    params = CreateSandboxFromImageParams(
        image="python:3.12", name="inspect-foo-1-abcdef12"
    )

    _respin_create_params(params)

    assert params.name is not None
    assert params.name != "inspect-foo-1-abcdef12"
    assert params.name.startswith("inspect-foo-1-")
    assert len(params.name) == len("inspect-foo-1-abcdef12")


def test_respin_create_params_appends_suffix_when_none_present() -> None:
    """A name with no existing hex suffix gets one appended."""
    params = CreateSandboxFromImageParams(image="python:3.12", name="inspect-foo-1")

    _respin_create_params(params)

    assert params.name is not None
    assert params.name.startswith("inspect-foo-1-")
    assert len(params.name) == len("inspect-foo-1") + 9


def test_respin_create_params_handles_missing_name() -> None:
    """No name set (None/empty) is a no-op, not an error."""
    params = CreateSandboxFromImageParams(image="python:3.12", name=None)

    _respin_create_params(params)

    assert params.name is None


def test_respin_create_params_snapshot_params() -> None:
    """Respin also works on the snapshot-params variant."""
    params = CreateSandboxFromSnapshotParams(
        snapshot="my-snapshot", name="inspect-foo-1-abcdef12"
    )

    _respin_create_params(params)

    assert params.name is not None
    assert params.name != "inspect-foo-1-abcdef12"
    assert len(params.name) == len("inspect-foo-1-abcdef12")


@pytest.mark.asyncio
async def test_reap_zombie_sandboxes_already_gone() -> None:
    """A zombie that 404s on lookup is treated as already reaped."""
    client = MagicMock()
    client.get = AsyncMock(side_effect=DaytonaNotFoundError("not found"))

    remaining = await reap_zombie_sandboxes(client, ["zombie-1"])

    assert remaining == []


@pytest.mark.asyncio
async def test_reap_zombie_sandboxes_deletes_successfully() -> None:
    """A zombie that can be fetched and deleted on the first pass is reaped."""
    client = MagicMock()
    client.get = AsyncMock(return_value=MagicMock())
    client.delete = AsyncMock()

    remaining = await reap_zombie_sandboxes(client, ["zombie-1"])

    assert remaining == []
    client.delete.assert_awaited_once()


@pytest.mark.asyncio
async def test_reap_zombie_sandboxes_delete_404_is_success() -> None:
    """A zombie that 404s on delete (vanished after get) is treated as reaped."""
    client = MagicMock()
    client.get = AsyncMock(return_value=MagicMock())
    client.delete = AsyncMock(side_effect=DaytonaNotFoundError("not found"))

    remaining = await reap_zombie_sandboxes(client, ["zombie-1"])

    assert remaining == []
    client.delete.assert_awaited_once()


@pytest.mark.asyncio
async def test_reap_zombie_sandboxes_retries_until_deletable() -> None:
    """A zombie undeletable on the first pass is reaped on a later pass."""
    client = MagicMock()
    client.get = AsyncMock(return_value=MagicMock(state="creating"))
    client.delete = AsyncMock(
        side_effect=[DaytonaError("state change in progress"), None]
    )

    with (
        patch(
            "inspect_sandboxes.daytona._sandbox_utils._monotonic",
            side_effect=[1000.0, 1010.0, 1020.0],
        ),
        patch(
            "inspect_sandboxes.daytona._sandbox_utils.asyncio.sleep",
            new=AsyncMock(),
        ) as sleep_mock,
    ):
        remaining = await reap_zombie_sandboxes(client, ["zombie-1"])

    assert remaining == []
    assert client.delete.await_count == 2
    sleep_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_reap_zombie_sandboxes_gives_up_at_ceiling() -> None:
    """A zombie still undeletable once the ceiling passes is reported, not retried forever."""
    client = MagicMock()
    client.get = AsyncMock(return_value=MagicMock(state="creating"))
    client.delete = AsyncMock(side_effect=DaytonaError("state change in progress"))

    with (
        patch(
            "inspect_sandboxes.daytona._sandbox_utils._monotonic",
            side_effect=[1000.0, 1010.0, 3000.0],
        ),
        patch(
            "inspect_sandboxes.daytona._sandbox_utils.asyncio.sleep",
            new=AsyncMock(),
        ),
    ):
        remaining = await reap_zombie_sandboxes(client, ["zombie-1"], ceiling_sec=1500)

    assert remaining == ["zombie-1"]


@pytest.mark.asyncio
async def test_reap_zombie_sandboxes_dedupes_names() -> None:
    """Duplicate names (e.g. repeated create-retry failures) are only reaped once."""
    client = MagicMock()
    client.get = AsyncMock(return_value=MagicMock())
    client.delete = AsyncMock()

    remaining = await reap_zombie_sandboxes(client, ["zombie-1", "zombie-1"])

    assert remaining == []
    client.delete.assert_awaited_once()


@pytest.mark.asyncio
async def test_create_sandbox_respins_and_succeeds_after_failure() -> None:
    """A failed first attempt respins the name, registers the zombie, then succeeds."""
    reset_zombie_registry()
    sandbox = MagicMock()
    client = MagicMock()
    client.create = AsyncMock(side_effect=[DaytonaError("boom"), sandbox])
    client.get = AsyncMock(return_value=MagicMock())
    client.delete = AsyncMock()
    params = CreateSandboxFromImageParams(
        image="python:3.12", name="inspect-foo-abcdef12"
    )

    with patch(
        "inspect_sandboxes.daytona._sandbox_utils.asyncio.sleep", new=AsyncMock()
    ):
        result = await create_sandbox(client, params)

    assert result is sandbox
    assert client.create.await_count == 2
    assert params.name is not None
    assert params.name != "inspect-foo-abcdef12"  # respun
    assert params.name.startswith("inspect-foo-")
    assert "inspect-foo-abcdef12" in zombie_registry()  # failed name registered


@pytest.mark.asyncio
async def test_create_sandbox_retries_timeout_error() -> None:
    """DaytonaTimeoutError is retried (not excluded as it was pre-respin)."""
    reset_zombie_registry()
    sandbox = MagicMock()
    client = MagicMock()
    client.create = AsyncMock(side_effect=[DaytonaTimeoutError("timed out"), sandbox])
    client.get = AsyncMock(return_value=MagicMock())
    client.delete = AsyncMock()
    params = CreateSandboxFromImageParams(image="python:3.12", name="inspect-foo")

    with patch(
        "inspect_sandboxes.daytona._sandbox_utils.asyncio.sleep", new=AsyncMock()
    ):
        result = await create_sandbox(client, params)

    assert result is sandbox
    assert client.create.await_count == 2


@pytest.mark.asyncio
async def test_create_sandbox_reraises_after_exhausting_attempts() -> None:
    """All attempts failing re-raises the last error after CREATE_SANDBOX_ATTEMPTS."""
    reset_zombie_registry()
    client = MagicMock()
    client.create = AsyncMock(side_effect=DaytonaError("persistent"))
    client.get = AsyncMock(return_value=MagicMock())
    client.delete = AsyncMock()
    params = CreateSandboxFromImageParams(image="python:3.12", name="inspect-foo")

    with patch(
        "inspect_sandboxes.daytona._sandbox_utils.asyncio.sleep", new=AsyncMock()
    ):
        with pytest.raises(DaytonaError, match="persistent"):
            await create_sandbox(client, params)

    assert client.create.await_count == CREATE_SANDBOX_ATTEMPTS


@pytest.mark.asyncio
async def test_zombie_registry_visible_across_child_task() -> None:
    """A zombie appended inside a child task is visible to the parent context.

    inspect runs each sample in its own anyio task with a copied context, so
    the registry must be primed in the parent (reset_zombie_registry, from
    task_init) for appends made inside sample tasks to survive to task_cleanup.
    """
    reset_zombie_registry()

    async def sample_task() -> None:
        zombie_registry().append("zombie-from-child")

    async with anyio.create_task_group() as tg:
        tg.start_soon(sample_task)

    assert "zombie-from-child" in zombie_registry()


# --- session exec transport ---------------------------------------------------


def sh_c(snippet: str) -> str:
    return f"{SHELL_PATH} -c {shlex.quote(snippet)}"


def test_build_session_command_plain() -> None:
    assert build_session_command("echo 'a b'") == sh_c("echo 'a b'")


def test_build_session_command_applies_env_and_cwd_in_a_child_shell() -> None:
    command = build_session_command(
        "echo hi", cwd="/work dir", env={"A": "x y", "B": "1"}
    )
    assert command == sh_c(
        "export A='x y' && export B=1 && cd -- '/work dir' && exec " + sh_c("echo hi")
    )


def test_build_session_command_user_switch() -> None:
    assert build_session_command("whoami", user="tester") == sh_c(
        "exec sudo -u tester bash -c whoami"
    )
    assert build_session_command("whoami", user="1000") == sh_c(
        "exec sudo -u '#1000' bash -c whoami"
    )


def test_build_session_command_timeout_runs_as_the_requested_user() -> None:
    assert build_session_command("sleep 9", timeout=30) == sh_c(
        f"exec {TIMEOUT_PATH} -k 5s 30s {sh_c('sleep 9')}"
    )
    assert build_session_command("sleep 9", user="root", timeout=30) == sh_c(
        f"exec sudo -u root {TIMEOUT_PATH} -k 5s 30s bash -c 'sleep 9'"
    )


def test_build_remove_command_pins_path() -> None:
    assert build_remove_command(["/tmp/a", "/tmp/b c"]) == (
        f"PATH={SYSTEM_PATH}; export PATH; rm -f /tmp/a '/tmp/b c'"
    )


async def _sh(command: str) -> tuple[int, str, str]:
    """Run *command* like a Daytona session: separate streams plus the exit code."""
    proc = await asyncio.create_subprocess_shell(
        command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    out, err = await proc.communicate()
    assert proc.returncode is not None
    return proc.returncode, out.decode(), err.decode()


@pytest.mark.asyncio
async def test_session_command_round_trip_in_a_posix_shell(tmp_path: Path) -> None:
    """Cwd and env reach the command, the streams stay apart, the status is the command's."""
    command = build_session_command(
        'pwd; echo "$GREETING"; echo err >&2; exit 3',
        cwd=str(tmp_path),
        env={"GREETING": "hi there"},
    )
    exit_code, stdout, stderr = await _sh(command)
    assert exit_code == 3
    assert stdout == f"{tmp_path.resolve()}\nhi there\n"
    assert stderr == "err\n"


@pytest.mark.asyncio
async def test_session_command_waits_for_background_writers() -> None:
    command = build_session_command(
        "echo early; (sleep 0.3; echo late; echo late-err >&2) &"
    )
    assert await _sh(command) == (0, "early\nlate\n", "late-err\n")


def make_session_sandbox() -> MagicMock:
    sandbox = MagicMock()
    sandbox.id = "sb"
    sandbox.process.create_session = AsyncMock()
    sandbox.process.delete_session = AsyncMock()
    sandbox.process.execute_session_command = AsyncMock(
        return_value=MagicMock(exit_code=0, stdout="out\n", stderr="err\n")
    )
    return sandbox


@pytest.mark.asyncio
async def test_session_pool_reuses_an_idle_session() -> None:
    sandbox = make_session_sandbox()
    pool = SessionPool(sandbox)

    async with pool.session() as first:
        pass
    async with pool.session() as second:
        pass

    assert first == second
    assert first.startswith("inspect-")
    sandbox.process.create_session.assert_awaited_once_with(first)
    sandbox.process.delete_session.assert_not_awaited()


@pytest.mark.asyncio
async def test_session_pool_lends_distinct_sessions_concurrently() -> None:
    """Commands on one session run one after another, so concurrent execs get their own."""
    sandbox = make_session_sandbox()
    pool = SessionPool(sandbox)
    seen: list[str] = []

    async def use() -> None:
        async with pool.session() as session_id:
            seen.append(session_id)
            await asyncio.sleep(0.05)

    await asyncio.gather(use(), use(), use())

    assert len(set(seen)) == 3
    assert sandbox.process.create_session.await_count == 3
    async with pool.session() as reused:
        assert reused in seen  # and no fourth session is created
    assert sandbox.process.create_session.await_count == 3


@pytest.mark.asyncio
async def test_session_pool_discards_a_session_whose_command_raised() -> None:
    sandbox = make_session_sandbox()
    pool = SessionPool(sandbox)

    lent: list[str] = []
    with pytest.raises(DaytonaError):
        async with pool.session() as dead:
            lent.append(dead)
            raise DaytonaError("session process has exited")
    sandbox.process.delete_session.assert_awaited_once_with(lent[0])

    async with pool.session() as fresh:
        pass
    assert fresh != lent[0]
    assert sandbox.process.create_session.await_count == 2


@pytest.mark.asyncio
async def test_session_pool_discard_failure_does_not_mask_the_error() -> None:
    sandbox = make_session_sandbox()
    sandbox.process.delete_session = AsyncMock(side_effect=DaytonaError("gone"))
    pool = SessionPool(sandbox)

    with pytest.raises(RuntimeError, match="original"):
        async with pool.session():
            raise RuntimeError("original")


@pytest.mark.asyncio
async def test_session_exec_returns_the_streams_and_sets_the_http_timeout() -> None:
    sandbox = make_session_sandbox()
    pool = SessionPool(sandbox)

    result = await session_exec(pool, "echo hi", 30)

    assert result == (0, "out\n", "err\n")
    call = sandbox.process.execute_session_command.await_args
    assert call.args[1].command == "echo hi"
    assert call.args[1].run_async is False
    assert call.kwargs["timeout"] == 30 + TIMEOUT_GRACE

    await session_exec(pool, "echo hi", None)
    assert sandbox.process.execute_session_command.await_args.kwargs["timeout"] is None


@pytest.mark.asyncio
async def test_session_exec_without_exit_code_is_an_error() -> None:
    sandbox = make_session_sandbox()
    sandbox.process.execute_session_command = AsyncMock(
        return_value=MagicMock(exit_code=None, stdout="", stderr="")
    )
    with pytest.raises(RuntimeError, match="no exit code"):
        await session_exec(SessionPool(sandbox), "echo hi", None)


def test_session_exec_result_maps_timeout_exit_codes() -> None:
    result = session_exec_result(3, "o", "e", None, 0.1)
    assert (result.success, result.returncode, result.stdout, result.stderr) == (
        False,
        3,
        "o",
        "e",
    )

    # GNU timeout reports the kill with 124, at any elapsed time.
    with pytest.raises(TimeoutError, match="timed out after 5 seconds") as info:
        session_exec_result(124, "partial", "err", 5, 0.5)
    assert vars(info.value)["truncated_output"] == "partialerr"

    # SIGKILL escalation / BusyBox SIGTERM count only after the timeout elapsed.
    assert session_exec_result(137, "", "", 5, 0.5).returncode == 137
    with pytest.raises(TimeoutError):
        session_exec_result(143, "", "", 5, 5.2)
    # Without a timeout these are ordinary exit codes.
    assert session_exec_result(124, "", "", None, 100).returncode == 124
