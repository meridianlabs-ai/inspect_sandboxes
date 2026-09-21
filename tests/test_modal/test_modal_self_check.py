"""Conformance runs of inspect_ai's portable sandbox checks against Modal.

Kept separate from test_modal.py so the `import *` of check functions doesn't
pollute the unit-test module. See the self_check module docstring for the
consumption contract (sandbox_env fixture + per-check xfails).
"""

from collections.abc import AsyncIterator

import pytest
import pytest_asyncio
from inspect_ai.util import SandboxEnvironment

# Pull the portable check functions into this module so pytest collects them
# as tests, each driven by the `sandbox_env` fixture below.
from inspect_ai.util._sandbox.self_check import *  # noqa: F401, F403  # pyright: ignore[reportWildcardImportFromLibrary]
from inspect_sandboxes.modal._modal import (
    ModalSandboxEnvironment,
    sandbox_cleanup_startup,
)

from tests.self_check_support import (
    ConfigAndEnv,
    SandboxConfig,
    XFail,
    apply_xfail,
)

# All checks share one sandbox per config (module-scoped loop + env): a
# fresh Modal sandbox per check would multiply runtime and API cost ~60x.
pytestmark = [pytest.mark.asyncio(loop_scope="module"), pytest.mark.integration]


SANDBOX_CONFIGS = [
    SandboxConfig(
        id="default",
        config=None,
        xfails={
            "test_exec_input_large": XFail(
                "exec() writes stdin in one shot; Modal's stdin buffer overflows "
                "with BufferError for large input"
            ),
            "test_exec_large_command": XFail(
                "Modal caps CMD at 64 KiB (ARG_MAX); raw InvalidError "
                "(https://github.com/meridianlabs-ai/inspect_sandboxes/issues/65)"
            ),
            "test_read_file_not_allowed": XFail("user is root"),
            "test_write_text_file_without_permissions": XFail("user is root"),
            "test_write_binary_file_without_permissions": XFail("user is root"),
            "test_exec_permission_error": XFail("user is root"),
        },
    ),
]


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
    sandbox_cleanup_startup()
    envs = await ModalSandboxEnvironment.sample_init(task_name, cfg.config, {})
    try:
        yield ConfigAndEnv(cfg=cfg, env=envs["default"])
    finally:
        try:
            await ModalSandboxEnvironment.sample_cleanup(
                task_name, cfg.config, envs, False
            )
            await ModalSandboxEnvironment.task_cleanup(task_name, None, cleanup=True)
        except Exception as e:
            print(f"Cleanup error: {e}")


# Must stay function-scoped: xfails are applied per check via request.node.
@pytest.fixture
def sandbox_env(
    request: pytest.FixtureRequest, _config_and_env: ConfigAndEnv
) -> SandboxEnvironment:
    apply_xfail(request, _config_and_env.cfg.xfails)
    return _config_and_env.env
