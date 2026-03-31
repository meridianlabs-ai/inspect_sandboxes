"""Shared helpers for Daytona sandbox environments"""

from __future__ import annotations

import errno
import shlex
from collections.abc import Awaitable, Callable

from daytona_sdk import (
    AsyncDaytona,
    AsyncSandbox,
    CreateSandboxFromImageParams,
    CreateSandboxFromSnapshotParams,
    DaytonaNotFoundError,
)
from inspect_ai.util import OutputLimitExceededError, SandboxEnvironmentLimits

from ._retry import exec_retry, standard_retry


def build_stdin_command(cmd: list[str], stdin_file: str) -> str:
    """Build a shell command that pipes a temp file into *cmd* and cleans up."""
    return (
        f"set -o pipefail; cat {shlex.quote(stdin_file)} | {shlex.join(cmd)}"
        f"; _ec=$?; rm -f {shlex.quote(stdin_file)}; exit $_ec"
    )


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


@exec_retry
async def create_sandbox(
    client: AsyncDaytona,
    params: CreateSandboxFromSnapshotParams | CreateSandboxFromImageParams,
) -> AsyncSandbox:
    """Uses exec_retry to avoid retrying on DaytonaTimeoutError"""
    return await client.create(params)


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
    paginated = await client.list(labels=labels)
    return paginated.items


@standard_retry
async def close_client(client: AsyncDaytona) -> None:
    await client.close()


@standard_retry
async def sdk_upload(sandbox: AsyncSandbox, remote_path: str, data: bytes) -> None:
    await sandbox.fs.upload_file(data, remote_path)


@standard_retry
async def sdk_download(sandbox: AsyncSandbox, remote_path: str) -> bytes:
    return await sandbox.fs.download_file(remote_path)
