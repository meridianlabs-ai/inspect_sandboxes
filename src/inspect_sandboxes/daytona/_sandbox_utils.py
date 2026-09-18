"""Shared helpers for Daytona sandbox environments"""

from __future__ import annotations

import asyncio
import errno
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


# Daytona's exec API returns one merged output stream and strips its trailing
# newline, so exec() does not hand the command's streams to the API directly:
# the command runs with stdout and stderr redirected to two private temp files,
# and the same shell then prints both files framed by per-call sentinels. The
# wrapper shell is the only writer of the API stream, so nothing the command
# runs (not even a background child) can put bytes into the stderr segment, and
# both streams come back byte for byte.

# Where the wrapper's own programs (cat, rm) are looked up. Matches the pin
# inspect_ai uses for its sandbox commands, so a directory the sandbox user
# controls on the image's PATH cannot supply them. It applies only to the
# wrapper's housekeeping subshells; the command keeps its own environment.
SYSTEM_PATH = "/usr/sbin:/usr/bin:/sbin:/bin"


def new_capture_tag() -> str:
    """A fresh per-exec tag naming the temp files and the output sentinels."""
    return uuid.uuid4().hex


def capture_files(tag: str) -> tuple[str, str]:
    """The stdout and stderr temp files for a capture *tag*."""
    return f"/tmp/.inspect-exec-{tag}.out", f"/tmp/.inspect-exec-{tag}.err"


def _sentinels(tag: str) -> tuple[str, str, str]:
    return (
        f"<<inspect-exec-{tag}:stdout>>",
        f"<<inspect-exec-{tag}:stderr>>",
        f"<<inspect-exec-{tag}:end>>",
    )


def build_capture_command(command: str, tag: str) -> str:
    """Wrap shell *command* so its stdout and stderr come back separately.

    The wrapper creates the two temp files ``0600`` and exclusively (``umask
    077`` and ``set -C`` inside a subshell, so the command's own umask and
    options are untouched; a stale file from a killed attempt is removed first,
    a planted entry makes creation fail), runs the command in a subshell with
    stdout and stderr appended to them, then prints ``<start>``, stdout,
    ``<stderr>``, stderr, ``<end>`` and removes the files, exiting with the
    command's status. Its own programs are resolved through ``SYSTEM_PATH``.
    Decode the output with :func:`parse_captured_output`.
    """
    out_file, err_file = (shlex.quote(f) for f in capture_files(tag))
    start, mid, end = (shlex.quote(s) for s in _sentinels(tag))
    pin = f"PATH={SYSTEM_PATH}; export PATH"
    return (
        f"({pin}; umask 077; set -C; rm -f {out_file} {err_file}"
        f" && : > {out_file} && : > {err_file})"
        f" && ({command}) >>{out_file} 2>>{err_file}; _ec=$?; "
        f"({pin}; printf %s {start}; cat {out_file}; printf %s {mid}; cat {err_file};"
        f" rm -f {out_file} {err_file}; printf %s {end}); exit $_ec"
    )


class OutputNotCapturedError(RuntimeError):
    """The output of a :func:`build_capture_command` run carries no frame.

    The wrapper prints the frame after the command has run, so a missing frame
    means the command never ran: ``sudo`` refused the user, ``/tmp`` was not
    writable, the shell was killed. The raw output is the diagnostics.
    """


def parse_captured_output(output: str, tag: str) -> tuple[str, str]:
    """Split the output of a :func:`build_capture_command` run into (stdout, stderr).

    The stderr sentinel is searched from the end: the wrapper prints it after
    the command has finished, so whatever the command wrote to stdout, even a
    copy of the sentinel, stays in stdout. Whitespace outside the outer
    sentinels is the API's to strip or keep.

    Raises:
        OutputNotCapturedError: The frame is missing or incomplete.
    """
    start, mid, end = _sentinels(tag)
    body = output.strip()
    if not (
        body.startswith(start)
        and body.endswith(end)
        and len(body) >= len(start) + len(end)
    ):
        raise OutputNotCapturedError(
            f"Command output was not captured (the exec wrapper did not complete): "
            f"{output.strip()}"
        )
    body = body[len(start) : len(body) - len(end)]
    split = body.rfind(mid)
    if split < 0:
        raise OutputNotCapturedError(
            f"Command output was not captured (stderr sentinel missing): {output.strip()}"
        )
    return body[:split], body[split + len(mid) :]


def captured_exec_result(exit_code: int, output: str, tag: str) -> ExecResult[str]:
    """The :class:`ExecResult` of a :func:`build_capture_command` run.

    Output without the frame means the wrapper never reached the command (for
    example ``sudo: unknown user``, or ``/tmp`` not writable), so the result is
    a failed exec with the diagnostics on stderr, as a provider with native
    streams would report a failed user switch. The command cannot produce
    this itself: it runs with both streams redirected to files, so it has no
    way to write to the API stream, framed or not.
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
