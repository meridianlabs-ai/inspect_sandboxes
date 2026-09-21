"""Helpers for running shell commands on E2B sandboxes.

E2B's ``commands.run(cmd)`` starts ``/bin/bash -l -c <cmd>``: the whole command
line travels as ONE argv element, which the Linux kernel caps at MAX_ARG_STRLEN
(128 KiB) regardless of ARG_MAX. Beyond that, envd fails the process start
with ``InvalidArgumentException("error starting process ...")``. Commands over
:data:`MAX_INLINE_COMMAND_BYTES` are therefore staged in a temp script inside
the sandbox and sourced, which has no such limit.

Temp files (staged scripts, stdin payloads) are removed by a separate command
after the run rather than by an inline ``rm -f``: the retry helpers re-run the
very same command string, and an inline cleanup would leave a retry sourcing a
file the first attempt already deleted.
"""

from __future__ import annotations

import shlex
import uuid

# Half the kernel's 128 KiB per-argument cap (MAX_ARG_STRLEN), so a redirection
# suffix or any future wrapper can never push an inline command over it while
# ordinary commands stay on the inline path.
MAX_INLINE_COMMAND_BYTES = 64 * 1024


def exceeds_inline_limit(command: str) -> bool:
    """Whether *command* is too long to pass as a single ``bash -c`` argument."""
    return len(command.encode("utf-8")) > MAX_INLINE_COMMAND_BYTES


def temp_file_path(kind: str) -> str:
    """A unique ``/tmp`` path for a staged file of the given *kind* (e.g. ``cmd``)."""
    return f"/tmp/.inspect-{kind}-{uuid.uuid4().hex}"


def source_script_command(script_file: str) -> str:
    """Run *script_file* in the calling shell.

    Sourcing (rather than ``bash script``) keeps the inline semantics: same
    login shell, environment and working directory as the direct path.
    """
    return f". {shlex.quote(script_file)}"


def remove_files_command(files: list[str]) -> str:
    """A command that removes *files*, ignoring ones that are already gone."""
    return "rm -f " + " ".join(shlex.quote(file) for file in files)
