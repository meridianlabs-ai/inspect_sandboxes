"""Shared helpers for Daytona sandbox environments"""

from __future__ import annotations

import errno
import shlex
import string
import uuid
from collections.abc import Awaitable, Callable
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

from ._retry import standard_retry

logger = getLogger(__name__)


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


async def create_sandbox(
    client: AsyncDaytona,
    params: CreateSandboxFromSnapshotParams | CreateSandboxFromImageParams,
    *,
    timeout: float | None = None,
) -> AsyncSandbox:
    """Create a sandbox, recovering from failed create attempts.

    A failed create can leave a zombie sandbox that still holds the requested
    name (observed stuck in CREATING for 15-20 min until a server-side watchdog
    errors it), so retrying with unchanged params always fails with a name
    conflict. Each retry therefore respins the identity: best-effort deletes
    the zombie, then regenerates the name suffix.
    """
    last_error: DaytonaError | None = None
    for attempt in range(CREATE_SANDBOX_ATTEMPTS):
        if attempt > 0:
            await _cleanup_failed_create(client, params)
            _respin_create_params(params)
        try:
            if timeout is None:
                return await client.create(params)
            return await client.create(params, timeout=timeout)
        except DaytonaError as e:
            last_error = e
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
        if sep and len(suffix) == 8 and all(c in string.hexdigits for c in suffix):
            # Swap the existing unique suffix so the name length stays put.
            params.name = f"{base}-{uuid.uuid4().hex[:8]}"
        else:
            params.name = f"{params.name}-{uuid.uuid4().hex[:8]}"


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
