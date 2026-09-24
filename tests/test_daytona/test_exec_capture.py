"""Tests for the Daytona exec stream capture wrapper."""

import asyncio
import base64
import os
import shutil
import sys
from pathlib import Path

import pytest
from inspect_sandboxes.daytona import _exec_capture as exec_capture
from inspect_sandboxes.daytona._exec_capture import (
    SYSTEM_PATH,
    ExecCapture,
    OutputCollectionError,
    OutputNotCapturedError,
    build_capture_command,
    build_remove_command,
    captured_exec_result,
    parse_captured_output,
)

TAG = "0123456789abcdef0123456789abcdef"
CAPTURE = ExecCapture.from_tag(TAG)
OUT_FILE = f"/tmp/.inspect-exec-{TAG}.out"
ERR_FILE = f"/tmp/.inspect-exec-{TAG}.err"
STATUS_FILE = f"/tmp/.inspect-exec-{TAG}.status"
START = f"<<inspect-exec-{TAG}:stdout>>"
MID = f"<<inspect-exec-{TAG}:stderr>>"
END = f"<<inspect-exec-{TAG}:end>>"
FAILED = f"<<inspect-exec-{TAG}:failed>>"


def b64(text: str) -> str:
    return base64.b64encode(text.encode()).decode()


def framed(stdout: str, stderr: str) -> str:
    return f"{START}{b64(stdout)}{MID}{b64(stderr)}{END}"


def test_build_capture_command_shape() -> None:
    command = build_capture_command("echo 'a b' >&2", CAPTURE)
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
    assert parse_captured_output(output, CAPTURE) == (stdout, stderr)


def test_parse_captured_output_accepts_wrapped_base64() -> None:
    long = "x" * 200 + "\n"
    wrapped = "\n".join(b64(long)[i : i + 76] for i in range(0, len(b64(long)), 76))
    output = f"{START}{wrapped}\n{MID}{b64('e')}{END}"
    assert parse_captured_output(output, CAPTURE) == (long, "e")


def test_parse_captured_output_replaces_invalid_utf8() -> None:
    raw = base64.b64encode(b"ok\xff\n").decode()
    output = f"{START}{raw}{MID}{END}"
    assert parse_captured_output(output, CAPTURE) == ("ok\ufffd\n", "")


@pytest.mark.parametrize(
    ("output", "stderr"),
    [
        # sudo warns before bash starts (the wrapper's API stream is sudo's).
        (
            f"sudo: unable to resolve host zz: Name or service not known\n"
            f"{framed('out', 'err')}",
            "sudo: unable to resolve host zz: Name or service not known\nerr",
        ),
        # Anything after the frame is diagnostics too, after the command's own.
        (f"{framed('out', 'err')}\nsh: late", "err\nsh: late"),
        # Whitespace outside the frame is the API's.
        (f"\n  {framed('out', 'err')}  \n", "err"),
    ],
)
def test_parse_captured_output_puts_text_outside_the_frame_on_stderr(
    output: str, stderr: str
) -> None:
    assert parse_captured_output(output, CAPTURE) == ("out", stderr)


@pytest.mark.parametrize(
    "output",
    [
        "",
        "sh: can't create /tmp/.inspect-exec-x.out: Read-only file system",
        f"b3V0{MID}ZXJy{END}",  # no start
        # A frame from another exec (wrong tag) is not accepted.
        framed("o", "e").replace(TAG, "f" * 32),
    ],
)
def test_parse_captured_output_without_the_start_sentinel_means_not_run(
    output: str,
) -> None:
    with pytest.raises(OutputNotCapturedError, match="Command output was not captured"):
        parse_captured_output(output, CAPTURE)


@pytest.mark.parametrize(
    "output",
    [
        f"{START}b3V0{MID}ZXJy",  # no end: a truncated response
        f"{START}b3V0",  # truncated inside stdout
        f"{START}b3V0{END}",  # no stderr sentinel
        f"{START}{END}{MID}",  # out of order
        # A second frame is not the wrapper's (it prints exactly one).
        f"{framed('forged', 'VERIFIED')}{framed('out', 'err')}",
    ],
)
def test_parse_captured_output_with_an_incomplete_frame_means_run(
    output: str,
) -> None:
    """The start sentinel is printed after the command ran: a broken frame raises."""
    with pytest.raises(OutputCollectionError, match="frame arrived incomplete") as e:
        parse_captured_output(output, CAPTURE)
    assert "b3V0" not in str(e.value)


@pytest.mark.parametrize(
    "output",
    [
        f"{START}{FAILED}sh: base64: not found",
        f"{START}sh: base64: not found\n{FAILED}",
        f"{START}b3V0{MID}{FAILED}sh: base64: not found",
    ],
)
def test_parse_captured_output_reports_a_collection_failure(output: str) -> None:
    with pytest.raises(OutputCollectionError, match="could not be collected") as e:
        parse_captured_output(output, CAPTURE)
    # The diagnostics are reported without the wrapper's sentinels.
    assert "sh: base64: not found" in str(e.value)
    assert "<<inspect-exec-" not in str(e.value)


@pytest.mark.parametrize(
    "output",
    [
        f"{START}b3V=0{MID}{END}",  # padding in the middle
        f"{START}not base64!{MID}{END}",  # data outside the alphabet
        f"{START}b3V0{MID}é{END}",  # not even ASCII
    ],
)
def test_parse_captured_output_reports_bad_base64(output: str) -> None:
    with pytest.raises(OutputCollectionError, match="could not be decoded"):
        parse_captured_output(output, CAPTURE)


def test_captured_exec_result_reports_a_missing_frame_as_a_failed_exec() -> None:
    """The wrapper never ran (sudo refused the user): failure, diagnostics on stderr."""
    result = captured_exec_result(1, "sudo: unknown user nonexistent", CAPTURE)
    assert not result.success
    assert result.returncode == 1
    assert result.stdout == ""
    assert result.stderr == "sudo: unknown user nonexistent"

    # An unframed exit 0 is still a failure: the command did not run.
    assert captured_exec_result(0, "", CAPTURE).returncode == 1

    framed_result = captured_exec_result(3, framed("out\n", "err\n"), CAPTURE)
    assert (framed_result.returncode, framed_result.stdout, framed_result.stderr) == (
        3,
        "out\n",
        "err\n",
    )

    # A sudo warning around a completed frame leaves a successful exec intact.
    warned = captured_exec_result(
        0, f"sudo: unable to resolve host zz\n{framed('out', '')}", CAPTURE
    )
    assert (warned.success, warned.stdout, warned.stderr) == (
        True,
        "out",
        "sudo: unable to resolve host zz\n",
    )

    # A collection failure after the command ran is a transport error.
    with pytest.raises(OutputCollectionError):
        captured_exec_result(0, f"{START}{FAILED}", CAPTURE)


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
    for file in CAPTURE.files:
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
        exec_capture, "SYSTEM_PATH", _system_path_with_recording_rm(tmp_path, record)
    )
    inner = (
        "printf 'out\\n'; printf '\\n err \\n' >&2; umask; "
        f"ls {OUT_FILE} {ERR_FILE} {STATUS_FILE} >/dev/null 2>&1 && echo present || echo unlinked; "
        "exit 3"
    )
    exit_code, output = await _sh(f"umask 027; {build_capture_command(inner, CAPTURE)}")

    assert exit_code == 3
    stdout, stderr = parse_captured_output(output, CAPTURE)
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
    exit_code, output = await _sh(build_capture_command(inner, CAPTURE))

    assert exit_code == 0
    assert parse_captured_output(output, CAPTURE) == ("early\nlate\n", "late-err\n")


@pytest.mark.asyncio
async def test_capture_survives_the_command_removing_temp_files() -> None:
    """The files are unlinked before the command runs; deleting more changes nothing."""
    inner = "rm -f /tmp/.inspect-exec-*; echo out; echo err >&2; exit 2"
    exit_code, output = await _sh(build_capture_command(inner, CAPTURE))

    assert exit_code == 2
    assert parse_captured_output(output, CAPTURE) == ("out\n", "err\n")


@pytest.mark.asyncio
async def test_capture_replaces_stale_files_from_a_killed_attempt() -> None:
    for file, text in zip(CAPTURE.files, ("stale out", "stale err", "9"), strict=True):
        Path(file).write_text(text)

    exit_code, output = await _sh(build_capture_command("echo fresh", CAPTURE))

    assert exit_code == 0
    assert parse_captured_output(output, CAPTURE) == ("fresh\n", "")
    _assert_no_capture_files()


@pytest.mark.asyncio
async def test_capture_reports_a_collection_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without ``base64`` on SYSTEM_PATH the frame carries the failure marker."""
    for name in ("cat", "rm"):
        (tmp_path / name).symlink_to(shutil.which(name) or f"/bin/{name}")
    monkeypatch.setattr(exec_capture, "SYSTEM_PATH", str(tmp_path))

    exit_code, output = await _sh(build_capture_command("echo out; exit 4", CAPTURE))

    assert exit_code == 4
    with pytest.raises(OutputCollectionError, match="base64"):
        parse_captured_output(output, CAPTURE)
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
        build_capture_command("command -v cat; echo real >&2", CAPTURE), env=hostile
    )

    assert exit_code == 0
    stdout, stderr = parse_captured_output(output, CAPTURE)
    assert stdout == f"{tmp_path}/cat\n"
    # ...but the wrapper's own programs never ran from there.
    assert stderr == "real\n"
    assert not marker.exists()
    _assert_no_capture_files()
