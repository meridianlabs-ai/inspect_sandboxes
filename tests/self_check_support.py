"""Shared scaffolding for the per-provider conformance runs of inspect_ai's checks.

The checks themselves live in ``inspect_ai.util._sandbox.self_check``.
"""

from dataclasses import dataclass, field
from typing import NamedTuple

import pytest
from inspect_ai.util import ComposeConfig, ComposeService, SandboxEnvironment


@dataclass(frozen=True)
class XFail:
    """An expected-failure marker: reason plus strictness.

    Strict (the default) means a check that starts passing fails the run, so
    stale entries get pruned.
    """

    reason: str
    strict: bool = True


FILE_USER_XFAILS = {
    "test_read_file_as_user": XFail("file operations cannot act as the requested user"),
    "test_write_file_as_user": XFail(
        "file operations cannot act as the requested user"
    ),
}
"""Checks every provider fails until its file operations can enforce a user."""


@dataclass(frozen=True)
class SandboxConfig:
    """A sandbox configuration to run the check suite against."""

    id: str
    config: ComposeConfig | None
    xfails: dict[str, XFail] = field(default_factory=dict)


class ConfigAndEnv(NamedTuple):
    """A sandbox config paired with its initialized environment."""

    cfg: SandboxConfig
    env: SandboxEnvironment


def dind_config(user: str | None = None) -> ComposeConfig:
    """Two-service ComposeConfig so the dispatcher routes to DinD.

    Args:
        user: User the default service runs as (the image's user when None).
    """
    return ComposeConfig(
        services={
            "default": ComposeService(
                image="python:3.12-slim", command="sleep infinity", user=user
            ),
            "helper": ComposeService(
                image="python:3.12-slim", command="sleep infinity"
            ),
        }
    )


async def check_root_file_ops_in_root_only_dir(env: SandboxEnvironment) -> None:
    """Root reads and writes reach a directory the service's default user can't.

    Run against a DinD service whose default user is not root (``dind_config("nobody")``).
    """
    setup = await env.exec(
        [
            "sh",
            "-c",
            "mkdir -m 700 /root-only && echo secret > /root-only/existing.txt"
            " && chmod 600 /root-only/existing.txt",
        ],
        user="root",
    )
    assert setup.success, setup.stderr
    assert not (await env.exec(["cat", "/root-only/existing.txt"])).success

    assert await env.read_file("/root-only/existing.txt", user="root") == "secret\n"
    await env.write_file("/root-only/new/written.txt", "written", user="root")
    assert await env.read_file("/root-only/new/written.txt", user="root") == "written"


def apply_xfail(request: pytest.FixtureRequest, xfails: dict[str, XFail]) -> None:
    """Mark the requesting check as an expected failure if it is listed."""
    xfail = xfails.get(request.node.originalname)
    if xfail is not None:
        request.node.add_marker(
            pytest.mark.xfail(reason=xfail.reason, strict=xfail.strict)
        )
