"""Shared helpers for Daytona sandbox environments"""

from __future__ import annotations

import asyncio
import errno
import shlex
import string
import time
import uuid
from collections.abc import Awaitable, Callable
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
from inspect_ai.util import OutputLimitExceededError, SandboxEnvironmentLimits

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
