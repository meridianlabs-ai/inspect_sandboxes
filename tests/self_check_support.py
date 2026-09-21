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


def dind_config() -> ComposeConfig:
    """Two-service ComposeConfig so the dispatcher routes to DinD."""
    return ComposeConfig(
        services={
            "default": ComposeService(
                image="python:3.12-slim", command="sleep infinity"
            ),
            "helper": ComposeService(
                image="python:3.12-slim", command="sleep infinity"
            ),
        }
    )


def apply_xfail(request: pytest.FixtureRequest, xfails: dict[str, XFail]) -> None:
    """Mark the requesting check as an expected failure if it is listed."""
    xfail = xfails.get(request.node.originalname)
    if xfail is not None:
        request.node.add_marker(
            pytest.mark.xfail(reason=xfail.reason, strict=xfail.strict)
        )
