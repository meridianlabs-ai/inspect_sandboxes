"""Conformance runs of inspect_ai's portable sandbox checks against Runloop.

Kept separate from test_runloop.py so the `import *` of check functions doesn't
pollute the unit-test module. See the self_check module docstring for the
consumption contract (sandbox_env fixture + per-check xfails).
"""

from collections.abc import AsyncIterator

import pytest
import pytest_asyncio
from inspect_ai.util import SandboxEnvironment

# Pull the portable check functions into this module so pytest collects them
# as tests, each driven by the `sandbox_env` fixture below.
from inspect_ai.util._sandbox.self_check import *  # noqa: F403  # pyright: ignore[reportWildcardImportFromLibrary]
from inspect_sandboxes.runloop._runloop import RunloopSandboxEnvironment

from tests.self_check_support import (
    ConfigAndEnv,
    SandboxConfig,
    XFail,
    apply_xfail,
    dind_config,
)

# All checks share one devbox per config (module-scoped loop + env); a fresh
# Runloop devbox per check would multiply runtime and API cost by the check count.
pytestmark = [pytest.mark.asyncio(loop_scope="module"), pytest.mark.integration]

SANDBOX_CONFIGS = [
    SandboxConfig(
        id="single",
        config=None,
        xfails={
            "test_exec_as_user": XFail(
                "Runloop's default image may not have useradd preinstalled."
            ),
            "test_exec_timeout_not_raised_on_fast_signal_death": XFail(
                "Runloop SIGKILLs the process during signal-exit cleanup, so the "
                "shell's self-SIGTERM surfaces as -137, not 143. Platform behavior."
            ),
        },
    ),
    SandboxConfig(
        id="dind",
        config=dind_config(),
        xfails={
            "test_exec_permission_error": XFail(
                "docker compose exec routes through sh; permission/output edges differ"
            ),
            "test_write_text_file_without_permissions": XFail(
                "docker compose exec routes through sh; permission/output edges differ"
            ),
            "test_write_binary_file_without_permissions": XFail(
                "docker compose exec routes through sh; permission/output edges differ"
            ),
            "test_read_file_not_allowed": XFail(
                "docker compose exec routes through sh; permission/output edges differ"
            ),
            "test_exec_as_user": XFail(
                "Runloop's default image may not have useradd preinstalled."
            ),
        },
    ),
]


# Module-scoped: one devbox per config, shared by all checks, which clean up
# after themselves.
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
    await RunloopSandboxEnvironment.task_init(task_name, None)
    envs = await RunloopSandboxEnvironment.sample_init(task_name, cfg.config, {})
    try:
        yield ConfigAndEnv(cfg=cfg, env=envs["default"])
    finally:
        try:
            await RunloopSandboxEnvironment.sample_cleanup(
                task_name, cfg.config, envs, False
            )
            await RunloopSandboxEnvironment.task_cleanup(task_name, None, cleanup=True)
        except Exception as e:
            print(f"Cleanup error: {e}")


# Must stay function-scoped: xfails are applied per check via request.node.
@pytest.fixture
def sandbox_env(
    request: pytest.FixtureRequest, _config_and_env: ConfigAndEnv
) -> SandboxEnvironment:
    apply_xfail(request, _config_and_env.cfg.xfails)
    return _config_and_env.env
