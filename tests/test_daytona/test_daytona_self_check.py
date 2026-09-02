"""Conformance runs of inspect_ai's portable sandbox checks against Daytona.

Kept separate from test_daytona.py so the `import *` of check functions
doesn't pollute the unit-test module. See the self_check module docstring
for the consumption contract (sandbox_env fixture + per-check xfails).
"""

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import NamedTuple

import pytest
import pytest_asyncio
from inspect_ai.util import ComposeConfig, ComposeService, SandboxEnvironment

# Pull the portable check functions into this module so pytest collects them
# as tests, each driven by the `sandbox_env` fixture below.
from inspect_ai.util._sandbox.self_check import *  # noqa: F401, F403
from inspect_sandboxes.daytona._daytona import DaytonaSandboxEnvironment

# All checks share one sandbox per config (module-scoped loop + env): a
# fresh Daytona sandbox per check would multiply runtime and API cost ~60x.
pytestmark = [pytest.mark.asyncio(loop_scope="module"), pytest.mark.integration]


@dataclass(frozen=True)
class XFail:
    """An expected-failure marker: reason plus strictness."""

    reason: str
    strict: bool = True


@dataclass(frozen=True)
class SandboxConfig:
    """A sandbox configuration to run the check suite against."""

    id: str
    config: ComposeConfig | None
    xfails: dict[str, XFail] = field(default_factory=dict)


def _dind_config() -> ComposeConfig:
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


SANDBOX_CONFIGS = [
    SandboxConfig(
        id="single",
        config=None,
        xfails={
            "test_exec_stderr": XFail(
                "Daytona merges stdout+stderr; stderr always empty"
            ),
            "test_exec_permission_error": XFail(
                "exit code 126, not translated to PermissionError"
            ),
            "test_exec_output": XFail("Daytona strips trailing newline from output"),
            "test_exec_env_vars": XFail(
                "trailing newline stripped (env vars themselves work)"
            ),
            "test_write_text_file_without_permissions": XFail(
                "Daytona returns 400, not 403 for write permission errors"
            ),
            "test_write_binary_file_without_permissions": XFail(
                "Daytona returns 400, not 403 for write permission errors"
            ),
            "test_exec_as_user": XFail(
                "adduser/useradd may not be available in default snapshot"
            ),
        },
    ),
    SandboxConfig(
        id="dind",
        config=_dind_config(),
        xfails={
            "test_exec_stderr": XFail(
                "DinD routes through compose exec; stderr merged"
            ),
            "test_exec_permission_error": XFail(
                "exit code 126, not translated to PermissionError"
            ),
            "test_exec_output": XFail("trailing newline stripped by compose exec"),
            "test_exec_env_vars": XFail("trailing newline stripped"),
            "test_write_text_file_without_permissions": XFail("root user in container"),
            "test_write_binary_file_without_permissions": XFail(
                "root user in container"
            ),
            "test_read_file_not_allowed": XFail("root user"),
            "test_exec_as_user": XFail("adduser/useradd may not be available"),
        },
    ),
]


class ConfigAndEnv(NamedTuple):
    """A sandbox config paired with its initialized environment."""

    cfg: SandboxConfig
    env: SandboxEnvironment


# Module-scoped: one sandbox per config, shared by all checks (like the old
# self_check() runner). Checks clean up after themselves.
@pytest_asyncio.fixture(
    scope="module",
    loop_scope="module",
    params=SANDBOX_CONFIGS,
    ids=lambda cfg: cfg.id,
)
async def _config_and_env(
    request: pytest.FixtureRequest,
) -> AsyncIterator[ConfigAndEnv]:
    cfg: SandboxConfig = request.param
    task_name = f"test_self_check_{cfg.id}"
    await DaytonaSandboxEnvironment.task_init(task_name, None)
    envs = await DaytonaSandboxEnvironment.sample_init(task_name, cfg.config, {})
    try:
        yield ConfigAndEnv(cfg=cfg, env=envs["default"])
    finally:
        try:
            await DaytonaSandboxEnvironment.sample_cleanup(
                task_name, cfg.config, envs, False
            )
            await DaytonaSandboxEnvironment.task_cleanup(task_name, None, cleanup=True)
        except Exception as e:
            print(f"Cleanup error: {e}")


# Must stay function-scoped: xfails are applied per check via request.node.
@pytest.fixture
def sandbox_env(
    request: pytest.FixtureRequest, _config_and_env: ConfigAndEnv
) -> SandboxEnvironment:
    xfail = _config_and_env.cfg.xfails.get(request.node.originalname)
    if xfail is not None:
        request.node.add_marker(
            pytest.mark.xfail(reason=xfail.reason, strict=xfail.strict)
        )
    return _config_and_env.env
