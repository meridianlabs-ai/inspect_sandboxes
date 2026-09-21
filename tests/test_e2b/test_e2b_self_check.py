"""Conformance runs of inspect_ai's portable sandbox checks against E2B.

Kept separate from test_e2b.py so the `import *` of check functions doesn't
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
from inspect_sandboxes.e2b._e2b import E2BSandboxEnvironment

from tests.self_check_support import (
    ConfigAndEnv,
    SandboxConfig,
    XFail,
    apply_xfail,
    dind_config,
)

# All checks share one sandbox per config (module-scoped loop + env); a fresh
# E2B sandbox per check would multiply runtime and API cost by the check count.
pytestmark = [pytest.mark.asyncio(loop_scope="module"), pytest.mark.integration]


# The three exec-timeout checks are flaky rather than reliably broken
# (httpx.ReadTimeout escapes outside the wrapped path; any of the three
# rotates), so they are non-strict: a pass shouldn't fail the run.
_FLAKY_TIMEOUT_XFAILS = {
    "test_exec_timeout": XFail(
        "flaky: httpx.ReadTimeout escapes outside the wrapped path", strict=False
    ),
    "test_exec_timeout_kills_process": XFail(
        "flaky: httpx.ReadTimeout escapes outside the wrapped path", strict=False
    ),
    "test_exec_timeout_kills_child_processes": XFail(
        "flaky: httpx.ReadTimeout escapes outside the wrapped path", strict=False
    ),
}

SANDBOX_CONFIGS = [
    SandboxConfig(
        id="single",
        config=None,
        xfails={
            "test_read_file_not_allowed": XFail(
                "files.read doesn't translate permission denials to PermissionError"
            ),
            "test_write_text_file_without_permissions": XFail(
                "files.write returns HTTP 400 for permission errors, not translated"
            ),
            "test_write_binary_file_without_permissions": XFail(
                "files.write returns HTTP 400 for permission errors, not translated"
            ),
            "test_exec_permission_error": XFail(
                "exit code 126 from shell, not translated to PermissionError"
            ),
            "test_exec_timeout_not_raised_on_fast_signal_death": XFail(
                "E2B reports exit -1 for self-SIGTERM, not 143"
            ),
            "test_exec_as_nonexistent_user": XFail(
                "E2B raises AuthenticationException, not the inspect_ai-expected error"
            ),
            "test_exec_large_command": XFail(
                "E2B process-start RPC caps command size; raw ConnectException "
                "(https://github.com/meridianlabs-ai/inspect_sandboxes/issues/64)"
            ),
            **_FLAKY_TIMEOUT_XFAILS,
        },
    ),
    SandboxConfig(
        id="dind",
        config=dind_config(),
        xfails={
            "test_exec_permission_error": XFail(
                "docker compose exec routes through sh; permission edges differ"
            ),
            "test_write_text_file_without_permissions": XFail(
                "docker compose exec routes through sh; permission edges differ"
            ),
            "test_write_binary_file_without_permissions": XFail(
                "docker compose exec routes through sh; permission edges differ"
            ),
            "test_read_file_not_allowed": XFail(
                "docker compose exec routes through sh; permission edges differ"
            ),
            "test_exec_large_command": XFail(
                "E2B process-start RPC caps command size; raw ConnectException "
                "(https://github.com/meridianlabs-ai/inspect_sandboxes/issues/64)"
            ),
            **_FLAKY_TIMEOUT_XFAILS,
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
    await E2BSandboxEnvironment.task_init(task_name, None)
    envs = await E2BSandboxEnvironment.sample_init(task_name, cfg.config, {})
    try:
        yield ConfigAndEnv(cfg=cfg, env=envs["default"])
    finally:
        try:
            await E2BSandboxEnvironment.sample_cleanup(
                task_name, cfg.config, envs, False
            )
            await E2BSandboxEnvironment.task_cleanup(task_name, None, cleanup=True)
        except Exception as e:
            print(f"Cleanup error: {e}")


# Must stay function-scoped: xfails are applied per check via request.node.
@pytest.fixture
def sandbox_env(
    request: pytest.FixtureRequest, _config_and_env: ConfigAndEnv
) -> SandboxEnvironment:
    apply_xfail(request, _config_and_env.cfg.xfails)
    return _config_and_env.env
