"""Conformance runs of inspect_ai's portable sandbox checks against Daytona.

Kept separate from test_daytona.py so the `import *` of check functions doesn't
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
from inspect_sandboxes.daytona._daytona import DaytonaSandboxEnvironment

from tests.self_check_support import (
    ConfigAndEnv,
    SandboxConfig,
    XFail,
    apply_xfail,
    dind_config,
)

# All checks share one sandbox per config (module-scoped loop + env); a fresh
# Daytona sandbox per check would multiply runtime and API cost by the check count.
pytestmark = [pytest.mark.asyncio(loop_scope="module"), pytest.mark.integration]


SANDBOX_CONFIGS = [
    SandboxConfig(
        id="single",
        config=None,
        xfails={
            "test_exec_permission_error": XFail(
                "exit code 126, not translated to PermissionError"
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
        config=dind_config(),
        xfails={
            "test_exec_permission_error": XFail(
                "exit code 126, not translated to PermissionError"
            ),
            "test_write_text_file_without_permissions": XFail("root user in container"),
            "test_write_binary_file_without_permissions": XFail(
                "root user in container"
            ),
            "test_read_file_not_allowed": XFail("root user"),
            "test_exec_large_command": XFail(
                "vm_exec passes the whole VM command to sh -c as one argument,"
                " capped at 128 KiB (MAX_ARG_STRLEN)"
            ),
            "test_exec_as_user": XFail("adduser/useradd may not be available"),
        },
    ),
]


# Module-scoped: one sandbox per config, shared by all checks, which clean up
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
    apply_xfail(request, _config_and_env.cfg.xfails)
    return _config_and_env.env
