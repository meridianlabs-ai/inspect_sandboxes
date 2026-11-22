"""File-operation user handling shared across providers."""

from typing import Literal

from inspect_ai.util import SandboxUserUnsupportedError


def check_file_user(
    sandbox: object,
    operation: Literal["read", "write"],
    user: str | None,
    *,
    as_root: bool,
) -> None:
    """Refuse a `read_file()` or `write_file()` user the provider cannot enforce.

    inspect_ai requires a provider to enforce an explicit file user's permissions or raise `SandboxUserUnsupportedError`. A provider whose file API acts as root (`as_root`) enforces `root`, which inspect_ai's checkpointing asks for. No provider here can act as any other user yet.
    """
    if user is None or (as_root and user == "root"):
        return
    raise SandboxUserUnsupportedError(
        f"{type(sandbox).__name__} cannot {operation} files as user {user!r}."
    )
