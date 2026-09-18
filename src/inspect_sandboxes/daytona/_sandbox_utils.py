"""Shared helpers for Daytona sandbox environments"""

from __future__ import annotations

import asyncio
import base64
import binascii
import errno
import re
import shlex
import string
import time
import uuid
from collections.abc import Awaitable, Callable, Sequence
from contextvars import ContextVar
from logging import getLogger

from daytona_sdk import (
    AsyncDaytona,
    AsyncSandbox,
    CreateSandboxFromImageParams,
    CreateSandboxFromSnapshotParams,
    DaytonaError,
    DaytonaNotFoundError,
    ListSandboxesQuery,
)
from inspect_ai.util import (
    ExecResult,
    OutputLimitExceededError,
    SandboxEnvironmentLimits,
)

from inspect_sandboxes._util.naming import _HEX_LEN

from ._retry import standard_retry

logger = getLogger(__name__)

# Indirection so tests can substitute a fake clock by patching this alias,
# instead of patching time.monotonic process-wide (which the asyncio event
# loop also calls, making a finite side_effect list flaky).
_monotonic = time.monotonic


def build_stdin_command(cmd: list[str], stdin_file: str, cleanup: bool = True) -> str:
    """Build a shell command that redirects a temp file as stdin into *cmd*.

    Args:
        cmd: Command to redirect stdin into.
        stdin_file: Path to the temp file containing stdin data.
        cleanup: If True, remove the temp file after the command.
            Set to False when the caller handles cleanup separately
            (e.g., when running as a different user who can't delete the file).
    """
    quoted_file = shlex.quote(stdin_file)
    base = f"{shlex.join(cmd)} < {quoted_file}"
    if cleanup:
        return f"{base}; _ec=$?; rm -f {quoted_file}; exit $_ec"
    return f"{base}; _ec=$?; exit $_ec"


# Daytona's exec API returns one merged output stream, so exec() does not hand
# the command's streams to the API directly: a wrapper shell runs the command
# with stdout and stderr piped into two private temp files, then prints both
# files base64-encoded between per-call sentinels. The wrapper is the only
# writer of the API stream, so nothing the command runs can put bytes into the
# stderr segment, base64 keeps the sentinels out of the data, and the pipes
# make the wrapper wait, as the API itself did, until every writer of the
# command's streams (background children included) has closed them.

# Where the wrapper's own programs (cat, rm, base64) are looked up. Matches the
# pin inspect_ai uses for its sandbox commands, so a directory the sandbox user
# controls on the image's PATH cannot supply them. It applies only to the
# wrapper's housekeeping subshells; the command keeps its own environment.
SYSTEM_PATH = "/usr/sbin:/usr/bin:/sbin:/bin"


def new_capture_tag() -> str:
    """A fresh per-exec tag naming the temp files and the output sentinels."""
    return uuid.uuid4().hex


def capture_files(tag: str) -> tuple[str, str, str]:
    """The stdout, stderr and exit-status temp files for a capture *tag*."""
    base = f"/tmp/.inspect-exec-{tag}"
    return f"{base}.out", f"{base}.err", f"{base}.status"


def _sentinels(tag: str) -> tuple[str, str, str, str]:
    return (
        f"<<inspect-exec-{tag}:stdout>>",
        f"<<inspect-exec-{tag}:stderr>>",
        f"<<inspect-exec-{tag}:end>>",
        f"<<inspect-exec-{tag}:failed>>",
    )


def build_capture_command(command: str, tag: str) -> str:
    """Wrap shell *command* so its stdout and stderr come back separately.

    The wrapper creates the three temp files ``0600`` and exclusively (``umask
    077`` and ``set -C`` in a subshell, so the command's own umask and options
    are untouched; a stale file from a killed attempt is removed first, a
    planted entry makes creation fail), opens them on descriptors 3/4/9 for
    writing and 6/7/8 for reading, and unlinks them before the command runs:
    the command cannot reach them by name, and nothing is left behind if the
    process tree is killed. The command runs in a subshell with those
    descriptors closed, its stdout and stderr piped into two ``cat`` collectors
    that append to the files; the pipes only close once every writer has,
    so a background child's late output is collected too. Its exit status is
    written to the status file. Then ``<stdout>``, base64(stdout),
    ``<stderr>``, base64(stderr), ``<end>`` are printed, or ``<failed>`` if
    collection failed, and the wrapper exits with the command's status. Its
    own programs are resolved through ``SYSTEM_PATH``. Decode the output with
    :func:`parse_captured_output`.
    """
    out_file, err_file, status_file = (shlex.quote(f) for f in capture_files(tag))
    start, mid, end, failed = (shlex.quote(s) for s in _sentinels(tag))
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


class OutputNotCapturedError(RuntimeError):
    """The output of a :func:`build_capture_command` run carries no frame.

    The wrapper prints the frame after the command has run, so a missing frame
    means the command never ran: ``sudo`` refused the user, ``/tmp`` was not
    writable, the shell was killed. The raw output is the diagnostics.
    """


class OutputCollectionError(RuntimeError):
    """The command ran, but the wrapper could not deliver its streams intact."""


_BASE64_TEXT = r"[A-Za-z0-9+/=\s]*"


def parse_captured_output(output: str, tag: str) -> tuple[str, str]:
    """Split the output of a :func:`build_capture_command` run into (stdout, stderr).

    Both streams are base64 in the frame, so a sentinel can never occur inside
    them and both come back byte for byte (decoded as UTF-8, invalid bytes
    replaced). Whitespace outside the outer sentinels is the API's to strip or
    keep.

    Raises:
        OutputCollectionError: The wrapper ran the command but marked the
            collection failed, or a segment is not valid base64.
        OutputNotCapturedError: The frame is missing or incomplete.
    """
    start, mid, end, failed = _sentinels(tag)
    body = output.strip()
    # Only the wrapper can print the failure marker (the command's bytes are
    # never on this stream); the shell's own error text may land before or
    # after it, so look for it anywhere.
    if failed in body:
        raise OutputCollectionError(
            "Command ran but its output could not be collected: "
            + body.replace(failed, " ").strip()
        )
    frame = re.fullmatch(
        f"{re.escape(start)}(?P<out>{_BASE64_TEXT}){re.escape(mid)}"
        f"(?P<err>{_BASE64_TEXT}){re.escape(end)}",
        body,
    )
    if frame is None:
        raise OutputNotCapturedError(
            f"Command output was not captured (the exec wrapper did not complete): "
            f"{body}"
        )
    try:
        return (
            _decode_stream(frame.group("out")),
            _decode_stream(frame.group("err")),
        )
    except binascii.Error as e:
        raise OutputCollectionError(
            f"Command ran but its output could not be decoded: {e}"
        ) from e


def _decode_stream(encoded: str) -> str:
    data = base64.b64decode(re.sub(r"\s", "", encoded), validate=True)
    return data.decode("utf-8", errors="replace")


def captured_exec_result(exit_code: int, output: str, tag: str) -> ExecResult[str]:
    """The :class:`ExecResult` of a :func:`build_capture_command` run.

    Output without the frame means the wrapper never reached the command (for
    example ``sudo: unknown user``, or ``/tmp`` not writable), so the result is
    a failed exec with the diagnostics on stderr, as a provider with native
    streams would report a failed user switch. The command cannot produce
    this itself: it runs with both streams redirected into the collectors, so
    it has no way to write to the API stream, framed or not. A collection
    failure after the command ran raises :class:`OutputCollectionError`.
    """
    try:
        stdout, stderr = parse_captured_output(output, tag)
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


async def verify_file_size(
    is_dir_fn: Callable[[str], Awaitable[bool]],
    get_size_fn: Callable[[str], Awaitable[int]],
    file: str,
) -> None:
    """Raise if *file* is a directory or exceeds the read size limit."""
    if await is_dir_fn(file):
        raise IsADirectoryError(errno.EISDIR, "Is a directory", file)

    file_size = await get_size_fn(file)
    if file_size > SandboxEnvironmentLimits.MAX_READ_FILE_SIZE:
        raise OutputLimitExceededError(
            limit_str=SandboxEnvironmentLimits.MAX_READ_FILE_SIZE_STR,
            truncated_output=None,
        )


def decode_file_content(data: bytes, file: str, text: bool) -> str | bytes:
    """Decode *data* to UTF-8 string if *text* is True, else return raw bytes."""
    if text:
        try:
            return data.decode("utf-8")
        except UnicodeDecodeError as e:
            raise UnicodeDecodeError(
                e.encoding,
                e.object,
                e.start,
                e.end,
                f"Failed to decode {file}: {e.reason}",
            ) from e
    return data


CREATE_SANDBOX_ATTEMPTS = 3

# Sandbox names left behind by failed create attempts, reaped by task_cleanup.
# Primed once in the parent context (reset_zombie_registry(), from task_init)
# before sample tasks fork — inspect copies the context per sample, so a .set()
# from inside a sample task would be invisible to task_cleanup. Same pattern as
# _running_sandboxes. (Kept here, not co-located in _daytona.py, to avoid a
# circular import.)
_zombie_names: ContextVar[list[str] | None] = ContextVar(
    "daytona_zombie_names", default=None
)


def reset_zombie_registry() -> None:
    """Prime a fresh zombie registry in the current (parent) context."""
    _zombie_names.set([])


def zombie_registry() -> list[str]:
    """The zombie-name registry for the current context (created on demand)."""
    registry = _zombie_names.get()
    if registry is None:
        registry = []
        _zombie_names.set(registry)
    return registry


async def reap_zombie_sandboxes(
    client: AsyncDaytona,
    names: list[str],
    *,
    ceiling_sec: float = 120,
    poll_sec: float = 30,
) -> list[str]:
    """Best-effort delete of zombie sandboxes, retrying each a few times.

    A zombie is undeletable ("state change in progress") until the server
    moves it to ERROR, which can take much longer than task_cleanup should
    block for — so this is a short best-effort pass, not a wait-until-gone
    loop. Returns the names still undeleted within ceiling_sec (also logged
    by name so they can be cleaned up out of band).
    """
    logger.warning(
        "Reaping %d zombie sandbox(es) from failed creates "
        "(best-effort, up to %.0fs): %s",
        len(set(names)),
        ceiling_sec,
        ", ".join(dict.fromkeys(names)),
    )
    deadline = _monotonic() + ceiling_sec
    remaining = list(dict.fromkeys(names))  # de-dupe, keep order
    while remaining and _monotonic() < deadline:
        still_remaining: list[str] = []
        for name in remaining:
            try:
                sandbox = await client.get(name)
            except DaytonaNotFoundError:
                logger.info("Zombie sandbox %s already gone", name)
                continue
            except DaytonaError as e:
                logger.debug("Zombie lookup failed for %s (will retry): %s", name, e)
                still_remaining.append(name)
                continue
            try:
                await client.delete(sandbox)
                logger.info("Reaped zombie sandbox %s", name)
            except DaytonaNotFoundError:
                # Vanished between get and delete (e.g. server removed it) —
                # treat as reaped, don't waste a poll cycle retrying.
                logger.info("Zombie sandbox %s already gone", name)
            except DaytonaError as e:
                logger.debug(
                    "Zombie %s not deletable yet (state=%s, will retry): %s",
                    name,
                    sandbox.state,
                    e,
                )
                still_remaining.append(name)
        remaining = still_remaining
        if remaining:
            await asyncio.sleep(poll_sec)
    if remaining:
        logger.warning(
            "Gave up reaping %d zombie sandbox(es) after %.0fs: %s",
            len(remaining),
            ceiling_sec,
            ", ".join(remaining),
        )
    return remaining


async def create_sandbox(
    client: AsyncDaytona,
    params: CreateSandboxFromSnapshotParams | CreateSandboxFromImageParams,
    *,
    timeout: float | None = None,
) -> AsyncSandbox:
    """Create a sandbox, recovering from failed create attempts.

    A failed create can leave a zombie holding the requested name, so a retry
    with unchanged params hits a name conflict; each retry respins the name
    suffix (and best-effort deletes the zombie) with backoff.

    Retries include DaytonaTimeoutError, so a persistently timing-out create
    can take up to CREATE_SANDBOX_ATTEMPTS x ``x-daytona.timeout`` to fail.
    """
    last_error: DaytonaError | None = None
    for attempt in range(CREATE_SANDBOX_ATTEMPTS):
        if attempt > 0:
            # exponential backoff between attempts (1s, 2s, capped at 10s)
            await asyncio.sleep(min(2 ** (attempt - 1), 10))
            await _cleanup_failed_create(client, params)
            _respin_create_params(params)
        try:
            if timeout is None:
                return await client.create(params)
            return await client.create(params, timeout=timeout)
        except DaytonaError as e:
            last_error = e
            if params.name:
                zombie_registry().append(params.name)
            logger.warning(
                "Sandbox create attempt %d/%d (name=%s) failed: %s: %s",
                attempt + 1,
                CREATE_SANDBOX_ATTEMPTS,
                params.name,
                type(e).__name__,
                str(e)[:300],
            )
    assert last_error is not None
    raise last_error


async def _cleanup_failed_create(
    client: AsyncDaytona,
    params: CreateSandboxFromSnapshotParams | CreateSandboxFromImageParams,
) -> None:
    """Best-effort delete of the zombie left behind by a failed create.

    Deletion is expected to fail while the zombie is still in CREATING
    ("state change in progress"); that's fine — the retry uses a fresh name
    and the zombie self-destructs once the server-side watchdog errors it.
    """
    if not params.name:
        return
    try:
        sandbox = await client.get(params.name)
        await client.delete(sandbox)
        logger.info("Deleted zombie sandbox from failed create: %s", params.name)
    except Exception as e:
        logger.debug(
            "Could not delete zombie sandbox %s (continuing with a fresh name): %s",
            params.name,
            e,
        )


def _respin_create_params(
    params: CreateSandboxFromSnapshotParams | CreateSandboxFromImageParams,
) -> None:
    """Regenerate the name suffix in place."""
    if params.name:
        base, sep, suffix = params.name.rpartition("-")
        if (
            sep
            and len(suffix) == _HEX_LEN
            and all(c in string.hexdigits for c in suffix)
        ):
            # Swap the existing unique suffix so the name length stays put.
            params.name = f"{base}-{uuid.uuid4().hex[:_HEX_LEN]}"
        else:
            params.name = f"{params.name}-{uuid.uuid4().hex[:_HEX_LEN]}"


@standard_retry
async def delete_sandbox(client: AsyncDaytona, sandbox: AsyncSandbox) -> None:
    try:
        await client.delete(sandbox)
    except DaytonaNotFoundError:
        pass  # already deleted — avoid triggering retry


@standard_retry
async def list_sandboxes(
    client: AsyncDaytona, labels: dict[str, str]
) -> list[AsyncSandbox]:
    query = ListSandboxesQuery(labels=labels)
    return [sandbox async for sandbox in client.list(query)]


@standard_retry
async def close_client(client: AsyncDaytona) -> None:
    await client.close()


@standard_retry
async def sdk_upload(sandbox: AsyncSandbox, remote_path: str, data: bytes) -> None:
    await sandbox.fs.upload_file(data, remote_path)


@standard_retry
async def sdk_download(sandbox: AsyncSandbox, remote_path: str) -> bytes:
    return await sandbox.fs.download_file(remote_path)
