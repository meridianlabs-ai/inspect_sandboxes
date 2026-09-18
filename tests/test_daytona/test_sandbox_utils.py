"""Tests for Daytona sandbox create-retry, zombie-reap and exec stderr helpers."""

import asyncio
import stat
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import anyio
import pytest
from daytona_sdk import (
    CreateSandboxFromImageParams,
    CreateSandboxFromSnapshotParams,
    DaytonaError,
    DaytonaNotFoundError,
    DaytonaTimeoutError,
)
from inspect_sandboxes.daytona._sandbox_utils import (
    CREATE_SANDBOX_ATTEMPTS,
    STDERR_SENTINEL,
    _respin_create_params,
    build_stderr_capture_command,
    build_stderr_readback_command,
    create_sandbox,
    parse_stderr_readback,
    read_stderr_file,
    reap_zombie_sandboxes,
    reset_zombie_registry,
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


# --- exec stderr capture -----------------------------------------------------


def test_build_stderr_capture_command_redirects_into_a_private_file() -> None:
    command = build_stderr_capture_command("echo 'a b' >&2", "/tmp/.inspect-stderr-1")
    assert command == (
        "(umask 077 && : > /tmp/.inspect-stderr-1) && "
        "{ echo 'a b' >&2; } 2>>/tmp/.inspect-stderr-1"
    )


def test_build_stderr_readback_command_removes_extra_files() -> None:
    command = build_stderr_readback_command("/tmp/err", ["/tmp/in put"])
    assert command == (
        f"printf %s '{STDERR_SENTINEL}'; cat /tmp/err; _ec=$?; "
        f"rm -f /tmp/err '/tmp/in put'; printf %s '{STDERR_SENTINEL}'; exit $_ec"
    )


@pytest.mark.parametrize(
    ("output", "expected"),
    [
        (f"{STDERR_SENTINEL}err\n{STDERR_SENTINEL}", "err\n"),
        (f"{STDERR_SENTINEL}{STDERR_SENTINEL}", ""),
        # The API trims the merged output; the sentinels keep that away from
        # the stderr bytes, including a trailing newline or a leading blank.
        (f"  {STDERR_SENTINEL}\n  x \n{STDERR_SENTINEL}\n", "\n  x \n"),
        # Only one sentinel is stripped from each end.
        (
            f"{STDERR_SENTINEL}{STDERR_SENTINEL}!{STDERR_SENTINEL}",
            f"{STDERR_SENTINEL}!",
        ),
    ],
)
def test_parse_stderr_readback_strips_sentinels(output: str, expected: str) -> None:
    assert parse_stderr_readback(0, output, "/tmp/err") == expected


def test_parse_stderr_readback_rejects_a_failed_cat() -> None:
    output = f"{STDERR_SENTINEL}cat: /tmp/err: No such file{STDERR_SENTINEL}"
    with pytest.raises(RuntimeError, match=r"/tmp/err \(exit code 1\): cat: /tmp/err"):
        parse_stderr_readback(1, output, "/tmp/err")


@pytest.mark.parametrize(
    "output", ["", "err", f"{STDERR_SENTINEL}err", STDERR_SENTINEL]
)
def test_parse_stderr_readback_rejects_unframed_output(output: str) -> None:
    with pytest.raises(RuntimeError, match="Failed to read the command's stderr"):
        parse_stderr_readback(0, output, "/tmp/err")


async def _sh(command: str) -> tuple[int, str]:
    """Run *command* like the Daytona API does: merged output, trimmed."""
    proc = await asyncio.create_subprocess_shell(
        command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT
    )
    output, _ = await proc.communicate()
    assert proc.returncode is not None
    return proc.returncode, output.decode().strip()


@pytest.mark.asyncio
async def test_stderr_capture_round_trip_in_a_posix_shell(tmp_path: Path) -> None:
    """Capture then readback through a real ``sh`` returns stderr byte for byte."""
    stderr_file = str(tmp_path / "err")
    (tmp_path / "keep").write_text("umask must not leak into the command")

    exit_code, output = await _sh(
        build_stderr_capture_command(
            f"echo out; printf '\\n err \\n' >&2; umask > {tmp_path / 'umask'}; exit 3",
            stderr_file,
        )
    )
    assert (exit_code, output) == (3, "out")
    # The file is private to the user running the command...
    assert stat.S_IMODE(Path(stderr_file).stat().st_mode) == 0o600
    # ...and the command itself ran with its own umask, not 077.
    assert (tmp_path / "umask").read_text().strip() != "0077"

    stderr = await read_stderr_file(_sh, stderr_file, [str(tmp_path / "keep")])
    assert stderr == "\n err \n"
    assert not Path(stderr_file).exists()
    assert not (tmp_path / "keep").exists()


@pytest.mark.asyncio
async def test_read_stderr_file_reports_a_missing_file(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="Failed to read the command's stderr"):
        await read_stderr_file(_sh, str(tmp_path / "missing"))
