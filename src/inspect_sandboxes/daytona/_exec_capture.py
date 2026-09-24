"""Separate stdout and stderr for Daytona exec.

Daytona's exec API returns one merged output stream, so exec() does not hand
the command's streams to the API directly: a wrapper shell runs the command
with stdout and stderr piped into two private temp files, then prints both
files base64-encoded between per-call sentinels. The command's streams go to
the collectors, never to the API stream, so nothing it prints can land in the
frame, base64 keeps the sentinels out of the data, and the pipes make the
wrapper wait, as the API itself did, until every writer of the command's
streams (background children included) has closed them.
"""

from __future__ import annotations

import base64
import shlex
import uuid
from collections.abc import Sequence
from dataclasses import dataclass

from inspect_ai.util import ExecResult

# Where the wrapper's own programs (cat, rm, base64) are looked up. Matches the
# pin inspect_ai uses for its sandbox commands, so a directory the sandbox user
# controls on the image's PATH cannot supply them. It applies only to the
# wrapper's housekeeping subshells; the command keeps its own environment.
SYSTEM_PATH = "/usr/sbin:/usr/bin:/sbin:/bin"


@dataclass(frozen=True)
class ExecCapture:
    """The temp files and output sentinels of one captured exec."""

    tag: str
    out_file: str
    err_file: str
    status_file: str
    stdout_sentinel: str
    stderr_sentinel: str
    end_sentinel: str
    failed_sentinel: str

    @classmethod
    def new(cls) -> ExecCapture:
        """A capture with a fresh random tag."""
        return cls.from_tag(uuid.uuid4().hex)

    @classmethod
    def from_tag(cls, tag: str) -> ExecCapture:
        base = f"/tmp/.inspect-exec-{tag}"
        return cls(
            tag=tag,
            out_file=f"{base}.out",
            err_file=f"{base}.err",
            status_file=f"{base}.status",
            stdout_sentinel=f"<<inspect-exec-{tag}:stdout>>",
            stderr_sentinel=f"<<inspect-exec-{tag}:stderr>>",
            end_sentinel=f"<<inspect-exec-{tag}:end>>",
            failed_sentinel=f"<<inspect-exec-{tag}:failed>>",
        )

    @property
    def files(self) -> list[str]:
        return [self.out_file, self.err_file, self.status_file]

    @property
    def sentinels(self) -> list[str]:
        return [
            self.stdout_sentinel,
            self.stderr_sentinel,
            self.end_sentinel,
            self.failed_sentinel,
        ]


class OutputNotCapturedError(RuntimeError):
    """The output of a :func:`build_capture_command` run carries no frame.

    The wrapper prints the frame's first sentinel after the command has run,
    so without it the command never ran: ``sudo`` refused the user, ``/tmp``
    was not writable, the shell was killed. The raw output is the diagnostics.
    """


class OutputCollectionError(RuntimeError):
    """The command ran, but the wrapper could not deliver its streams intact."""


def build_capture_command(command: str, capture: ExecCapture) -> str:
    """Wrap shell *command* so its stdout and stderr come back separately.

    The wrapper creates the three temp files ``0600`` and exclusively (``umask
    077`` and ``set -C`` in a subshell, so the command's own umask and options
    are untouched; a stale file from a killed attempt is removed first, a
    planted entry makes creation fail), opens them on descriptors 3/4/9 for
    writing and 6/7/8 for reading, and unlinks them before the command runs:
    the command cannot reach them by name, and nothing is left behind if the
    process tree is killed. The command runs in a subshell with those
    descriptors closed, its stdout and stderr piped into two ``cat`` collectors
    that append to the files. Its exit status is written to the status file.
    Then ``<stdout>``, base64(stdout), ``<stderr>``, base64(stderr), ``<end>``
    are printed, or ``<failed>`` if collection failed, and the wrapper exits
    with the command's status. Its own programs are resolved through
    ``SYSTEM_PATH``. Decode the output with :func:`parse_captured_output`.

    The pipes only close once every writer has, so a background child's late
    output is collected too, and a daemon started without redirecting its
    stdout and stderr keeps the exec waiting until the timeout.
    """
    q = shlex.quote
    out_file, err_file, status_file = (
        q(capture.out_file),
        q(capture.err_file),
        q(capture.status_file),
    )
    start, mid, end = (
        q(capture.stdout_sentinel),
        q(capture.stderr_sentinel),
        q(capture.end_sentinel),
    )
    failed = q(capture.failed_sentinel)
    pin = f"PATH={SYSTEM_PATH}; export PATH"
    files = f"{out_file} {err_file} {status_file}"
    return (
        # setup: private, exclusive files; open them; unlink them
        f"if ({pin}; umask 077; set -C; rm -f {files}"
        f" && : > {out_file} && : > {err_file} && : > {status_file})"
        f" && exec 3>>{out_file} 4>>{err_file} 9>>{status_file}"
        f" 6<{out_file} 7<{err_file} 8<{status_file}"
        f" && ({pin}; rm -f {files}); then "
        # run: stdout -> pipe -> cat -> fd 3 (out); stderr -> pipe -> cat -> fd 4 (err)
        f"{{ {{ ( ({command}) 3>&- 4>&- 5>&- 6<&- 7<&- 8<&- 9>&-; echo $? >&9; )"
        f" 2>&5 | ({pin}; exec cat >&3); }} 5>&1 | ({pin}; exec cat >&4); }}; "
        f"read -r _ec <&8; "
        # emit: both streams base64, framed; a collection failure is marked
        f"({pin}; printf %s {start} && base64 <&6 && printf %s {mid}"
        f" && base64 <&7 && printf %s {end}) || printf %s {failed}; "
        f"exit ${{_ec:-1}}; fi; exit 1"
    )


def parse_captured_output(output: str, capture: ExecCapture) -> tuple[str, str]:
    """Split the output of a :func:`build_capture_command` run into (stdout, stderr).

    Both streams are base64 in the frame, so a sentinel can never occur inside
    them and both come back byte for byte (decoded as UTF-8, invalid bytes
    replaced). Text outside the frame is not the command's stdout or stderr
    (those go to the collectors) but diagnostics printed by ``sudo`` or the wrapper shell,
    such as ``sudo: unable to resolve host``; it is added to stderr, around
    the command's own. Whitespace-only text there is the API's and dropped.

    Raises:
        OutputNotCapturedError: The frame's first sentinel is missing, so the
            command never ran.
        OutputCollectionError: The command ran, but the wrapper marked the
            collection failed, or the frame is incomplete (a truncated
            response), duplicated or not valid base64.
    """
    # Only the wrapper prints the failure marker; the shell's own error text
    # may land before or after it, so look for it anywhere.
    if capture.failed_sentinel in output:
        raise OutputCollectionError(
            "Command ran but its output could not be collected: "
            + _without_sentinels(output, capture)
        )
    if capture.stdout_sentinel not in output:
        raise OutputNotCapturedError(
            "Command output was not captured (the exec wrapper did not reach "
            f"the command): {output.strip()}"
        )

    start = output.find(capture.stdout_sentinel)
    mid = output.find(capture.stderr_sentinel)
    end = output.find(capture.end_sentinel)
    frame = [capture.stdout_sentinel, capture.stderr_sentinel, capture.end_sentinel]
    if any(output.count(s) != 1 for s in frame) or not start < mid < end:
        raise OutputCollectionError(
            "Command ran but its output frame arrived incomplete "
            f"({len(output)} characters received)"
        )

    try:
        stdout = _decode_stream(output[start + len(capture.stdout_sentinel) : mid])
        stderr = _decode_stream(output[mid + len(capture.stderr_sentinel) : end])
    except ValueError as e:
        raise OutputCollectionError(
            f"Command ran but its output could not be decoded: {e}"
        ) from e
    before = _diagnostics(output[:start])
    after = _diagnostics(output[end + len(capture.end_sentinel) :])
    return stdout, before + stderr + after


def captured_exec_result(
    exit_code: int, output: str, capture: ExecCapture
) -> ExecResult[str]:
    """The :class:`ExecResult` of a :func:`build_capture_command` run.

    Output without the frame means the wrapper never reached the command (for
    example ``sudo: unknown user``, or ``/tmp`` not writable), so the result is
    a failed exec with the diagnostics on stderr, as a provider with native
    streams would report a failed user switch. The command cannot produce
    this itself: it runs with both streams redirected into the collectors.
    A collection failure after the command ran raises
    :class:`OutputCollectionError`.
    """
    try:
        stdout, stderr = parse_captured_output(output, capture)
    except OutputNotCapturedError:
        return ExecResult(
            success=False,
            returncode=exit_code if exit_code != 0 else 1,
            stdout="",
            stderr=output,
        )
    return ExecResult(
        success=exit_code == 0, returncode=exit_code, stdout=stdout, stderr=stderr
    )


def build_remove_command(files: Sequence[str]) -> str:
    """A shell command removing temp *files*, with ``rm`` resolved via ``SYSTEM_PATH``."""
    quoted = " ".join(shlex.quote(f) for f in files)
    return f"PATH={SYSTEM_PATH}; export PATH; rm -f {quoted}"


def _decode_stream(encoded: str) -> str:
    data = base64.b64decode("".join(encoded.split()), validate=True)
    return data.decode("utf-8", errors="replace")


def _diagnostics(text: str) -> str:
    return text if text.strip() else ""


def _without_sentinels(output: str, capture: ExecCapture) -> str:
    for sentinel in capture.sentinels:
        output = output.replace(sentinel, " ")
    return " ".join(output.split())
