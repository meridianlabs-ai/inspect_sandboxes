"""Shared helpers for Daytona sandbox environments"""

from __future__ import annotations

import asyncio
import errno
import shlex
import string
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from contextlib import asynccontextmanager
from contextvars import ContextVar
from logging import getLogger

from daytona import (
    AsyncDaytona,
    AsyncSandbox,
    CreateSandboxFromImageParams,
    CreateSandboxFromSnapshotParams,
    DaytonaError,
    DaytonaNotFoundError,
    ListSandboxesQuery,
    SessionExecuteRequest,
)
from inspect_ai.util import (
    ExecResult,
    OutputLimitExceededError,
    SandboxEnvironmentLimits,
    trace_message,
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


# Daytona's process.exec() API returns one merged output stream, so exec() runs
# commands through the sandbox's session API instead: the Daytona agent tags
# every chunk a session command writes with its stream, and the SDK hands back
# stdout, stderr and the exit status separately. A session is a persistent
# shell in the sandbox, and commands sent to one session run one after another,
# so each environment keeps a pool of idle sessions (one per concurrent exec),
# and every command runs in its own `/bin/sh -c` so cwd and environment never
# leak between calls. Measured on the default image (2026-09-18): a session
# `echo` costs about 25 ms more than process.exec(), creating a session about
# 40 ms; large outputs are slower (5 MiB: 8 s against 0.65 s). The agent
# appends a newline to an unterminated last line of either stream and replaces
# invalid UTF-8; NUL bytes and a background child's late output are kept.

# Where the wrapper's own programs are looked up, and the absolute paths of the
# programs it runs before the caller's command: the image's PATH is never
# consulted for them (a directory the sandbox user controls could be first on
# it). Same pins as inspect_ai's own sandbox commands and Docker sandbox.
SYSTEM_PATH = "/usr/sbin:/usr/bin:/sbin:/bin"
SHELL_PATH = "/bin/sh"
TIMEOUT_PATH = "/usr/bin/timeout"

# Grace between the in-command timeout firing and the HTTP request timing out:
# `timeout -k 5s` escalates to SIGKILL after 5 s, plus agent round trip.
TIMEOUT_GRACE = 10


def build_session_command(
    command: str,
    *,
    cwd: str | None = None,
    env: dict[str, str] | None = None,
    user: str | None = None,
    timeout: int | None = None,
) -> str:
    """The text a Daytona session runs for shell snippet *command*.

    The session request carries only the command text, so the working
    directory, environment, user switch and timeout are applied inside it:
    ``export``s and ``cd`` in a fresh ``/bin/sh -c`` (nothing reaches the
    session's own shell, whose dialect and state are the image's), then
    ``exec`` of the command under ``sudo -u <user> bash -c`` when *user* is
    set (as before) and under ``/usr/bin/timeout -k 5s <timeout>s`` when a
    timeout is set, run as the requested user so the whole process tree is
    killed when it fires (the Docker sandbox does the same; exit 124 means it
    did). The environment is exported before ``sudo``, so ``sudo``'s own
    policy decides what the switched user sees, as with the exec API.
    """
    if user is not None:
        user_arg = shlex.quote(f"#{user}") if user.isdigit() else shlex.quote(user)
        runner = f"bash -c {shlex.quote(command)}"
        if timeout is not None:
            runner = f"{TIMEOUT_PATH} -k 5s {timeout}s {runner}"
        runner = f"sudo -u {user_arg} {runner}"
    else:
        runner = f"{SHELL_PATH} -c {shlex.quote(command)}"
        if timeout is not None:
            runner = f"{TIMEOUT_PATH} -k 5s {timeout}s {runner}"
    parts = [f"export {k}={shlex.quote(v)}" for k, v in (env or {}).items()]
    if cwd is not None:
        parts.append(f"cd -- {shlex.quote(cwd)}")
    if not parts and runner == f"{SHELL_PATH} -c {shlex.quote(command)}":
        return runner
    parts.append(f"exec {runner}")
    return f"{SHELL_PATH} -c {shlex.quote(' && '.join(parts))}"


class SessionPool:
    """Idle Daytona sessions of one sandbox, handed out one per running exec.

    Commands sent to one session run sequentially, so a session is lent to a
    single exec at a time and returned when the command has finished. A
    session whose command raised is never reused: it may still be running the
    command, or its shell may have died ("session process has exited"), so it
    is deleted best-effort and the next exec creates a fresh one. Sessions die
    with the sandbox.
    """

    def __init__(self, sandbox: AsyncSandbox) -> None:
        self.sandbox = sandbox
        self._idle: list[str] = []

    @asynccontextmanager
    async def session(self) -> AsyncIterator[str]:
        """Lend an idle session (creating one if none is idle) for one command."""
        if self._idle:
            session_id = self._idle.pop()
        else:
            session_id = f"inspect-{uuid.uuid4().hex[:12]}"
            await self.sandbox.process.create_session(session_id)
        try:
            yield session_id
        except BaseException:
            await self._discard(session_id)
            raise
        self._idle.append(session_id)

    async def _discard(self, session_id: str) -> None:
        try:
            await asyncio.wait_for(self.sandbox.process.delete_session(session_id), 10)
        except BaseException as e:  # noqa: BLE001 - best effort during error handling
            trace_message(
                logger,
                "daytona",
                f"Could not delete session {session_id} of sandbox {self.sandbox.id}: {e}",
            )


async def session_exec(
    pool: SessionPool, command: str, timeout: int | None
) -> tuple[int, str, str]:
    """Run *command* (from :func:`build_session_command`) in a pooled session.

    Returns ``(exit_code, stdout, stderr)``. The HTTP request is given
    ``timeout + TIMEOUT_GRACE`` seconds when a timeout is set, so the
    in-command ``timeout`` normally fires first and the result comes back
    cleanly; an HTTP timeout (``DaytonaTimeoutError``) means the agent did not
    answer, and the session is discarded.
    """
    request = SessionExecuteRequest(command=command, run_async=False)
    http_timeout = timeout + TIMEOUT_GRACE if timeout is not None else None
    async with pool.session() as session_id:
        response = await pool.sandbox.process.execute_session_command(
            session_id, request, timeout=http_timeout
        )
    if response.exit_code is None:
        raise RuntimeError("Daytona session command returned no exit code")
    return int(response.exit_code), response.stdout or "", response.stderr or ""


def session_exec_result(
    exit_code: int, stdout: str, stderr: str, timeout: int | None, elapsed: float
) -> ExecResult[str]:
    """The :class:`ExecResult` of a timed command, or ``TimeoutError`` if it timed out.

    Mirrors inspect_ai's Docker sandbox: exit 124 is GNU ``timeout`` reporting
    the kill, 137 (SIGKILL escalation) and 143 (BusyBox ``timeout``, SIGTERM)
    are ambiguous with other signal deaths and count only when the command ran
    at least *timeout* seconds. The partial output travels on the error's
    ``truncated_output`` attribute.
    """
    if (
        timeout is not None
        and exit_code in (124, 137, 143)
        and (exit_code == 124 or elapsed >= timeout)
    ):
        error = TimeoutError(f"Command timed out after {timeout} seconds")
        if stdout or stderr:
            setattr(error, "truncated_output", stdout + stderr)  # noqa: B010
        raise error
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
