"""Shared helpers for SDK-backed sandbox environments.

Provider-agnostic building blocks for exec and file I/O: constructing the
stdin-redirect shell command, checking read-size limits via injected probes,
and decoding file bytes. Provider-specific pieces (how a probe execs, how a
transfer happens) stay in the providers.
"""

from __future__ import annotations

import errno
import shlex
from collections.abc import Awaitable, Callable

from inspect_ai.util import OutputLimitExceededError, SandboxEnvironmentLimits


def build_stdin_command(
    cmd: list[str],
    stdin_file: str,
    cleanup: bool = True,
    *,
    remove_command: Callable[[list[str]], str] | None = None,
) -> str:
    """Build a shell command that redirects a temp file as stdin into *cmd*.

    Args:
        cmd: Command to redirect stdin into.
        stdin_file: Path to the temp file containing stdin data.
        cleanup: If True, remove the temp file after the command. Set to False
            when the caller handles cleanup separately (e.g. when running as a
            different user who can't delete the file).
        remove_command: Optional builder for the cleanup command, given the files
            to remove. Defaults to a plain ``rm -f``; a provider can inject one
            that hardens it (e.g. a PATH-safe ``rm`` for sudo'd execs).
    """
    quoted_file = shlex.quote(stdin_file)
    base = f"{shlex.join(cmd)} < {quoted_file}"
    if not cleanup:
        return f"{base}; _ec=$?; exit $_ec"
    rm = (
        f"({remove_command([stdin_file])})"
        if remove_command
        else f"rm -f {quoted_file}"
    )
    return f"{base}; _ec=$?; {rm}; exit $_ec"


async def verify_file_size(
    is_dir_fn: Callable[[str], Awaitable[bool]],
    get_size_fn: Callable[[str], Awaitable[int]],
    file: str,
) -> None:
    """Raise if *file* is a directory or exceeds the read size limit.

    The directory and size checks are injected as ``is_dir_fn`` / ``get_size_fn``
    so each provider supplies its own probe.
    """
    if await is_dir_fn(file):
        raise IsADirectoryError(errno.EISDIR, "Is a directory", file)

    file_size = await get_size_fn(file)
    if file_size > SandboxEnvironmentLimits.MAX_READ_FILE_SIZE:
        raise OutputLimitExceededError(
            limit_str=SandboxEnvironmentLimits.MAX_READ_FILE_SIZE_STR,
            truncated_output=None,
        )


def decode_file_content(data: bytes, file: str, text: bool) -> str | bytes:
    """Decode *data* to a UTF-8 string if *text*, else return the raw bytes."""
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
