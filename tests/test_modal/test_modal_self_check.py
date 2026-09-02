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
from inspect_ai.util._sandbox.self_check import *  # noqa: F401, F403
from inspect_sandboxes.modal._modal import (
    ModalSandboxEnvironment,
    sandbox_cleanup_startup,
)

# All checks share one sandbox (module-scoped loop + env): a fresh Modal
# sandbox per check would multiply runtime and API cost ~60x.
pytestmark = [pytest.mark.asyncio(loop_scope="module"), pytest.mark.integration]

# Check name -> reason, for checks the Modal sandbox can't satisfy (strict:
# an entry that starts passing fails the run so stale entries get pruned).
# Modal hard-removed the legacy Sandbox filesystem API the provider's file
# I/O (and exec stdin / cwd handling) still uses, so every check touching it
# fails (https://github.com/meridianlabs-ai/inspect_sandboxes/issues/63).
_FS_API_REMOVED = "Modal removed the legacy Sandbox filesystem API (#63)"

XFAILS = {
    "test_cwd_absolute": _FS_API_REMOVED,
    "test_cwd_relative": _FS_API_REMOVED,
    "test_cwd_unspecified": _FS_API_REMOVED,
    "test_exec_input_binary": _FS_API_REMOVED,
    "test_exec_input_large": _FS_API_REMOVED,
    "test_read_and_write_file_binary": _FS_API_REMOVED,
    "test_read_and_write_file_including_directory_absolute": _FS_API_REMOVED,
    "test_read_and_write_file_including_directory_relative": _FS_API_REMOVED,
    "test_read_and_write_file_text": _FS_API_REMOVED,
    "test_read_and_write_large_file_binary": _FS_API_REMOVED,
    "test_read_file_limit": _FS_API_REMOVED,
    "test_read_file_zero_length": _FS_API_REMOVED,
    "test_write_binary_file_exists": _FS_API_REMOVED,
    "test_write_binary_file_is_directory": _FS_API_REMOVED,
    "test_write_binary_file_space": _FS_API_REMOVED,
    "test_write_binary_file_zero_length": _FS_API_REMOVED,
    "test_write_file_text_utf": _FS_API_REMOVED,
    "test_write_text_file_exists": _FS_API_REMOVED,
    "test_write_text_file_is_directory": _FS_API_REMOVED,
    "test_write_text_file_space": _FS_API_REMOVED,
    "test_write_text_file_zero_length": _FS_API_REMOVED,
    "test_exec_large_command": (
        "Modal caps CMD at 64 KiB (ARG_MAX); raw InvalidError "
        "(https://github.com/meridianlabs-ai/inspect_sandboxes/issues/65)"
    ),
    "test_read_file_not_allowed": "user is root, so this doesn't work",
    "test_exec_as_user": "unsupported",
    "test_exec_as_nonexistent_user": "unsupported",
    "test_write_text_file_without_permissions": "user is root",
    "test_write_binary_file_without_permissions": "user is root",
    "test_exec_permission_error": "user is root",
}


# Module-scoped: one sandbox shared by all checks (like the old self_check()
# runner). Checks clean up after themselves.
@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def _modal_env() -> AsyncIterator[SandboxEnvironment]:
    sandbox_cleanup_startup()
    envs = await ModalSandboxEnvironment.sample_init("test_self_check", None, {})
    try:
        yield envs["default"]
    finally:
        try:
            await ModalSandboxEnvironment.sample_cleanup(
                "test_self_check", None, envs, False
            )
            await ModalSandboxEnvironment.task_cleanup(
                "test_self_check", None, cleanup=True
            )
        except Exception as e:
            print(f"Cleanup error: {e}")


# Must stay function-scoped: xfails are applied per check via request.node.
@pytest.fixture
def sandbox_env(
    request: pytest.FixtureRequest, _modal_env: SandboxEnvironment
) -> SandboxEnvironment:
    reason = XFAILS.get(request.node.originalname)
    if reason is not None:
        request.node.add_marker(pytest.mark.xfail(reason=reason, strict=True))
    return _modal_env
