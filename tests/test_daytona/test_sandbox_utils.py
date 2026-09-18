"""Tests for Daytona sandbox create-retry, zombie-reap and exec capture helpers."""

import asyncio
import os
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
    SYSTEM_PATH,
    OutputNotCapturedError,
    _respin_create_params,
    build_capture_command,
    build_remove_command,
    capture_files,
    captured_exec_result,
    create_sandbox,
    parse_captured_output,
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


# --- exec stream capture ------------------------------------------------------

TAG = "0123456789abcdef0123456789abcdef"
OUT_FILE, ERR_FILE = capture_files(TAG)
START = f"<<inspect-exec-{TAG}:stdout>>"
MID = f"<<inspect-exec-{TAG}:stderr>>"
END = f"<<inspect-exec-{TAG}:end>>"


def framed(stdout: str, stderr: str) -> str:
    return f"{START}{stdout}{MID}{stderr}{END}"


def test_build_capture_command_shape() -> None:
    command = build_capture_command("echo 'a b' >&2", TAG)
    pin = f"PATH={SYSTEM_PATH}; export PATH"
    assert command == (
        f"({pin}; umask 077; set -C; rm -f {OUT_FILE} {ERR_FILE}"
        f" && : > {OUT_FILE} && : > {ERR_FILE})"
        f" && (echo 'a b' >&2) >>{OUT_FILE} 2>>{ERR_FILE}; _ec=$?; "
        f"({pin}; printf %s '{START}'; cat {OUT_FILE}; printf %s '{MID}'; cat {ERR_FILE};"
        f" rm -f {OUT_FILE} {ERR_FILE}; printf %s '{END}'); exit $_ec"
    )


def test_build_remove_command_pins_path() -> None:
    assert build_remove_command(["/tmp/a", "/tmp/b c"]) == (
        f"PATH={SYSTEM_PATH}; export PATH; rm -f /tmp/a '/tmp/b c'"
    )


@pytest.mark.parametrize(
    ("output", "expected"),
    [
        (framed("out\n", "err\n"), ("out\n", "err\n")),
        (framed("", ""), ("", "")),
        # The API strips the trailing newline (or more); the frame keeps that
        # away from the streams, including stdout's own trailing newline.
        (
            f"  {framed('out' + chr(10), chr(10) + ' e ' + chr(10))}\n",
            ("out\n", "\n e \n"),
        ),
        # A command that prints the stderr sentinel to stdout stays in stdout:
        # the real sentinel is the last one, printed after the command finished.
        (framed(f"x{MID}FORGED\n", "real\n"), (f"x{MID}FORGED\n", "real\n")),
        # Printing it to stderr can only shrink the command's own stderr.
        (framed("out\n", f"{MID}late\n"), (f"out\n{MID}", "late\n")),
    ],
)
def test_parse_captured_output_splits_on_the_last_sentinel(
    output: str, expected: tuple[str, str]
) -> None:
    assert parse_captured_output(output, TAG) == expected


@pytest.mark.parametrize(
    "output",
    [
        "",
        "sh: can't create /tmp/.inspect-exec-x.out: Read-only file system",
        f"{START}out{MID}err",  # no end
        f"out{MID}err{END}",  # no start
        f"{START}out{END}",  # no stderr sentinel
        # A frame from another exec (wrong tag) is not accepted.
        framed("o", "e").replace(TAG, "f" * 32),
    ],
)
def test_parse_captured_output_rejects_an_incomplete_frame(output: str) -> None:
    with pytest.raises(OutputNotCapturedError, match="Command output was not captured"):
        parse_captured_output(output, TAG)


def test_captured_exec_result_reports_a_missing_frame_as_a_failed_exec() -> None:
    """The wrapper never ran (sudo refused the user): failure, diagnostics on stderr."""
    result = captured_exec_result(1, "sudo: unknown user nonexistent", TAG)
    assert not result.success
    assert result.returncode == 1
    assert result.stdout == ""
    assert result.stderr == "sudo: unknown user nonexistent"

    # An unframed exit 0 is still a failure: the command did not run.
    assert captured_exec_result(0, "", TAG).returncode == 1

    framed_result = captured_exec_result(3, framed("out\n", "err\n"), TAG)
    assert (framed_result.returncode, framed_result.stdout, framed_result.stderr) == (
        3,
        "out\n",
        "err\n",
    )


async def _sh(command: str, env: dict[str, str] | None = None) -> tuple[int, str]:
    """Run *command* like the Daytona API does: merged output, trimmed."""
    proc = await asyncio.create_subprocess_shell(
        command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        env=env,
    )
    output, _ = await proc.communicate()
    assert proc.returncode is not None
    return proc.returncode, output.decode().strip()


@pytest.mark.asyncio
async def test_capture_round_trip_in_a_posix_shell(tmp_path: Path) -> None:
    """The wrapper returns both streams byte for byte through a real ``sh``.

    The command stats its own capture files and prints its umask, so the
    output shows the files are private and the command's umask is untouched.
    """
    out_file, err_file = capture_files(TAG)
    inner = (
        "printf 'out\\n'; printf '\\n err \\n' >&2; umask; "
        f"stat -f '%Lp' {out_file} {err_file} 2>/dev/null || stat -c '%a' {out_file} {err_file}; "
        "exit 3"
    )
    exit_code, output = await _sh(build_capture_command(inner, TAG))

    assert exit_code == 3
    stdout, stderr = parse_captured_output(output, TAG)
    assert stdout == "out\n0022\n600\n600\n"
    assert stderr == "\n err \n"
    assert not Path(out_file).exists() and not Path(err_file).exists()


@pytest.mark.asyncio
async def test_capture_replaces_stale_files_from_a_killed_attempt() -> None:
    out_file, err_file = capture_files(TAG)
    Path(out_file).write_text("stale stdout")
    Path(err_file).write_text("stale stderr")

    exit_code, output = await _sh(build_capture_command("echo fresh", TAG))

    assert exit_code == 0
    assert parse_captured_output(output, TAG) == ("fresh\n", "")
    assert not Path(out_file).exists() and not Path(err_file).exists()


@pytest.mark.asyncio
async def test_capture_housekeeping_ignores_a_hostile_path(tmp_path: Path) -> None:
    """``cat``/``rm`` come from SYSTEM_PATH, not from a directory the user put first on PATH."""
    marker = tmp_path / "shim-ran"
    for name in ("cat", "rm"):
        shim = tmp_path / name
        shim.write_text(
            f"#!/bin/sh\ntouch {marker}\nprintf 'INSPECT_FRAMEWORK_DIRECTORY_VERIFIED\\n'\n"
        )
        shim.chmod(0o700)
    hostile = {
        **os.environ,
        "PATH": f"{tmp_path}:{os.environ.get('PATH', '/usr/bin:/bin')}",
    }

    # The command itself sees the image PATH (that is the caller's business)...
    exit_code, output = await _sh(
        build_capture_command("command -v cat; echo real >&2", TAG), env=hostile
    )

    assert exit_code == 0
    stdout, stderr = parse_captured_output(output, TAG)
    assert stdout == f"{tmp_path}/cat\n"
    # ...but the wrapper's own cat and rm never ran from there.
    assert stderr == "real\n"
    assert not marker.exists()
    out_file, err_file = capture_files(TAG)
    assert not Path(out_file).exists() and not Path(err_file).exists()
