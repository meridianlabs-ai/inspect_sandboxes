"""Tests for Daytona sandbox create-retry, zombie-reap and exec capture helpers."""

import asyncio
import base64
import os
import shutil
import sys
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
from inspect_sandboxes.daytona import _sandbox_utils as sandbox_utils
from inspect_sandboxes.daytona._sandbox_utils import (
    CREATE_SANDBOX_ATTEMPTS,
    SYSTEM_PATH,
    OutputCollectionError,
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
OUT_FILE, ERR_FILE, STATUS_FILE = capture_files(TAG)
START = f"<<inspect-exec-{TAG}:stdout>>"
MID = f"<<inspect-exec-{TAG}:stderr>>"
END = f"<<inspect-exec-{TAG}:end>>"
FAILED = f"<<inspect-exec-{TAG}:failed>>"


def b64(text: str) -> str:
    return base64.b64encode(text.encode()).decode()


def framed(stdout: str, stderr: str) -> str:
    return f"{START}{b64(stdout)}{MID}{b64(stderr)}{END}"


def test_build_capture_command_shape() -> None:
    command = build_capture_command("echo 'a b' >&2", TAG)
    pin = f"PATH={SYSTEM_PATH}; export PATH"
    files = f"{OUT_FILE} {ERR_FILE} {STATUS_FILE}"
    assert command == (
        f"if ({pin}; umask 077; set -C; rm -f {files}"
        f" && : > {OUT_FILE} && : > {ERR_FILE} && : > {STATUS_FILE})"
        f" && exec 3>>{OUT_FILE} 4>>{ERR_FILE} 9>>{STATUS_FILE}"
        f" 6<{OUT_FILE} 7<{ERR_FILE} 8<{STATUS_FILE}"
        f" && ({pin}; rm -f {files}); then "
        "{ { ( (echo 'a b' >&2) 3>&- 4>&- 5>&- 6<&- 7<&- 8<&- 9>&-; echo $? >&9; )"
        f" 2>&5 | ({pin}; exec cat >&3); }} 5>&1 | ({pin}; exec cat >&4); }}; "
        "read -r _ec <&8; "
        f"({pin}; printf %s '{START}' && base64 <&6 && printf %s '{MID}'"
        f" && base64 <&7 && printf %s '{END}') || printf %s '{FAILED}'; "
        "exit ${_ec:-1}; fi; exit 1"
    )


def test_build_remove_command_pins_path() -> None:
    assert build_remove_command(["/tmp/a", "/tmp/b c"]) == (
        f"PATH={SYSTEM_PATH}; export PATH; rm -f /tmp/a '/tmp/b c'"
    )


@pytest.mark.parametrize(
    ("stdout", "stderr"),
    [
        ("out\n", "err\n"),
        ("", ""),
        ("\n  x \n", "\n e \n"),
        # The sentinels are data like anything else, in either stream.
        (f"x{MID}FORGED\n", "real\n"),
        ("out\n", f"before\n{MID}after\n"),
        (f"{START}{END}{FAILED}", f"{FAILED}{MID}"),
        # Non-ASCII survives.
        ("héllo ✓\n", "ünïcode\n"),
    ],
)
def test_parse_captured_output_round_trips_both_streams(
    stdout: str, stderr: str
) -> None:
    # The API may strip whitespace around the output, and base64 may be wrapped.
    output = f"  {framed(stdout, stderr)}\n"
    assert parse_captured_output(output, TAG) == (stdout, stderr)


def test_parse_captured_output_accepts_wrapped_base64() -> None:
    long = "x" * 200 + "\n"
    wrapped = "\n".join(b64(long)[i : i + 76] for i in range(0, len(b64(long)), 76))
    output = f"{START}{wrapped}\n{MID}{b64('e')}{END}"
    assert parse_captured_output(output, TAG) == (long, "e")


def test_parse_captured_output_replaces_invalid_utf8() -> None:
    raw = base64.b64encode(b"ok\xff\n").decode()
    output = f"{START}{raw}{MID}{END}"
    assert parse_captured_output(output, TAG) == ("ok\ufffd\n", "")


@pytest.mark.parametrize(
    "output",
    [
        "",
        "sh: can't create /tmp/.inspect-exec-x.out: Read-only file system",
        f"{START}b3V0{MID}ZXJy",  # no end
        f"b3V0{MID}ZXJy{END}",  # no start
        f"{START}b3V0{END}",  # no stderr sentinel
        f"{START}not base64!{MID}{END}",  # data outside the alphabet
        # A frame from another exec (wrong tag) is not accepted.
        framed("o", "e").replace(TAG, "f" * 32),
    ],
)
def test_parse_captured_output_rejects_an_incomplete_frame(output: str) -> None:
    with pytest.raises(OutputNotCapturedError, match="Command output was not captured"):
        parse_captured_output(output, TAG)


@pytest.mark.parametrize(
    "output",
    [
        f"{START}{FAILED}sh: base64: not found",
        f"{START}sh: base64: not found\n{FAILED}",
        f"{START}b3V0{MID}{FAILED}",
    ],
)
def test_parse_captured_output_reports_a_collection_failure(output: str) -> None:
    with pytest.raises(OutputCollectionError, match="could not be collected"):
        parse_captured_output(output, TAG)


def test_parse_captured_output_reports_bad_base64() -> None:
    with pytest.raises(OutputCollectionError, match="could not be decoded"):
        parse_captured_output(f"{START}b3V=0{MID}{END}", TAG)


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

    # A collection failure after the command ran is a transport error.
    with pytest.raises(OutputCollectionError):
        captured_exec_result(0, f"{START}{FAILED}", TAG)


async def _sh(command: str, env: dict[str, str] | None = None) -> tuple[int, str]:
    """Run *command* like the Daytona API does: merged output, plus the exit code."""
    proc = await asyncio.create_subprocess_shell(
        command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        env=env,
    )
    output, _ = await proc.communicate()
    assert proc.returncode is not None
    return proc.returncode, output.decode()


def _assert_no_capture_files() -> None:
    for file in capture_files(TAG):
        assert not Path(file).exists(), f"{file} left behind"


def _system_path_with_recording_rm(tmp_path: Path, record: Path) -> str:
    """A SYSTEM_PATH whose ``rm`` records mode and owner of each existing file first.

    The wrapper's own ``rm`` runs right after it created and opened the capture
    files, so the record shows what other users could have seen in ``/tmp``.
    ``cat`` and ``base64`` are the real programs.
    """
    for name in ("cat", "base64"):
        (tmp_path / name).symlink_to(shutil.which(name) or f"/usr/bin/{name}")
    probe = (
        "import os, stat, sys\n"
        f"with open({str(record)!r}, 'a') as f:\n"
        "    for p in sys.argv[1:]:\n"
        "        if os.path.exists(p):\n"
        "            st = os.stat(p)\n"
        "            f.write(f'{p} {oct(stat.S_IMODE(st.st_mode))} {st.st_uid == os.getuid()}\\n')\n"
    )
    (tmp_path / "probe.py").write_text(probe)
    rm = tmp_path / "rm"
    rm.write_text(
        f'#!/bin/sh\n{sys.executable} {tmp_path / "probe.py"} "$@"\nexec {shutil.which("rm")} "$@"\n'
    )
    rm.chmod(0o700)
    return str(tmp_path)


@pytest.mark.asyncio
async def test_capture_round_trip_in_a_posix_shell(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The wrapper returns both streams byte for byte through a real ``sh``.

    The capture files are created ``0600`` for the running user and unlinked
    before the command starts (it cannot see them by name), the command's
    incoming umask (027 here) is untouched, and the exit status is the
    command's.
    """
    record = tmp_path / "record"
    monkeypatch.setattr(
        sandbox_utils, "SYSTEM_PATH", _system_path_with_recording_rm(tmp_path, record)
    )
    inner = (
        "printf 'out\\n'; printf '\\n err \\n' >&2; umask; "
        f"ls {OUT_FILE} {ERR_FILE} {STATUS_FILE} >/dev/null 2>&1 && echo present || echo unlinked; "
        "exit 3"
    )
    exit_code, output = await _sh(f"umask 027; {build_capture_command(inner, TAG)}")

    assert exit_code == 3
    stdout, stderr = parse_captured_output(output, TAG)
    assert stdout == "out\n0027\nunlinked\n"
    assert stderr == "\n err \n"
    _assert_no_capture_files()
    # The only rm that saw existing files is the unlink after creation: three
    # files, mode 0600, owned by the running user.
    assert sorted(record.read_text().splitlines()) == sorted(
        f"{file} 0o600 True" for file in (OUT_FILE, ERR_FILE, STATUS_FILE)
    )


@pytest.mark.asyncio
async def test_capture_waits_for_background_writers() -> None:
    """Late output from a child that outlives the command is collected, as the API did."""
    inner = "echo early; (sleep 0.3; echo late; echo late-err >&2) &"
    exit_code, output = await _sh(build_capture_command(inner, TAG))

    assert exit_code == 0
    assert parse_captured_output(output, TAG) == ("early\nlate\n", "late-err\n")


@pytest.mark.asyncio
async def test_capture_survives_the_command_removing_temp_files() -> None:
    """The files are unlinked before the command runs; deleting more changes nothing."""
    inner = "rm -f /tmp/.inspect-exec-*; echo out; echo err >&2; exit 2"
    exit_code, output = await _sh(build_capture_command(inner, TAG))

    assert exit_code == 2
    assert parse_captured_output(output, TAG) == ("out\n", "err\n")


@pytest.mark.asyncio
async def test_capture_replaces_stale_files_from_a_killed_attempt() -> None:
    for file, text in zip(
        capture_files(TAG), ("stale out", "stale err", "9"), strict=True
    ):
        Path(file).write_text(text)

    exit_code, output = await _sh(build_capture_command("echo fresh", TAG))

    assert exit_code == 0
    assert parse_captured_output(output, TAG) == ("fresh\n", "")
    _assert_no_capture_files()


@pytest.mark.asyncio
async def test_capture_reports_a_collection_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without ``base64`` on SYSTEM_PATH the frame carries the failure marker."""
    for name in ("cat", "rm"):
        (tmp_path / name).symlink_to(shutil.which(name) or f"/bin/{name}")
    monkeypatch.setattr(sandbox_utils, "SYSTEM_PATH", str(tmp_path))

    exit_code, output = await _sh(build_capture_command("echo out; exit 4", TAG))

    assert exit_code == 4
    with pytest.raises(OutputCollectionError, match="base64"):
        parse_captured_output(output, TAG)
    _assert_no_capture_files()


@pytest.mark.asyncio
async def test_capture_housekeeping_ignores_a_hostile_path(tmp_path: Path) -> None:
    """``cat``/``rm``/``base64`` come from SYSTEM_PATH, not from a directory first on PATH."""
    marker = tmp_path / "shim-ran"
    for name in ("cat", "rm", "base64"):
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
    # ...but the wrapper's own programs never ran from there.
    assert stderr == "real\n"
    assert not marker.exists()
    _assert_no_capture_files()
