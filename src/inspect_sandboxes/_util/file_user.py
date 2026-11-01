"""File-operation user handling shared across providers."""

from typing import Literal

from inspect_ai.util import SandboxUserUnsupportedError


def reject_file_user(
    sandbox: object, operation: Literal["read", "write"], user: str | None
) -> None:
    """Refuse an explicit `user` for `read_file()` or `write_file()`.

    inspect_ai requires a provider to enforce an explicit file user's permissions or raise `SandboxUserUnsupportedError`. These providers read and write files with privileged APIs, so they cannot enforce one yet.
    """
    if user is not None:
        raise SandboxUserUnsupportedError(
            f"{type(sandbox).__name__} cannot {operation} files as user {user!r}."
        )
