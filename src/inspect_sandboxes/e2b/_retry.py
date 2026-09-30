"""Retry decorators for the E2B sandbox provider.

E2B's SDK does some retry under the hood for HTTP transients, but not for
sandbox-level errors. We wrap lifecycle and exec operations with tenacity so
rate-limits and transient sandbox errors don't surface as test failures.

Permanent errors (NotFoundException, AuthenticationException,
InvalidArgumentException, CommandExitException) are not retried.

TimeoutException is also not retried by exec_retry — the exec() timeout retry
loop in _single_env.py handles those, applying the SandboxEnvironment contract
of "first retry ≤60s, second ≤30s".
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TypeVar

import httpx
from e2b import (
    AuthenticationException,
    CommandExitException,
    InvalidArgumentException,
    NotFoundException,
    SandboxException,
    TimeoutException,
)

from inspect_sandboxes._util.retry import (
    make_retry_decorators,
)
from inspect_sandboxes._util.retry import (
    run_with_timeout_retry as _run_with_timeout_retry,
)

T = TypeVar("T")

_PERMANENT_EXCEPTIONS = (
    NotFoundException,
    AuthenticationException,
    InvalidArgumentException,
    CommandExitException,
)


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, _PERMANENT_EXCEPTIONS):
        return False
    return isinstance(exc, SandboxException)


def _is_retryable_for_exec(exc: BaseException) -> bool:
    # TimeoutException is handled by run_with_timeout_retry below.
    if isinstance(exc, TimeoutException):
        return False
    return _is_retryable(exc)


def _is_timeout(exc: BaseException) -> bool:
    # E2B's SDK normally wraps timeouts as TimeoutException, but for short
    # command timeouts the underlying httpx.ReadTimeout can surface unwrapped.
    return isinstance(exc, (TimeoutException, httpx.TimeoutException))


# standard_retry: sandbox lifecycle + file I/O (5 attempts).
# exec_retry: exec operations (3 attempts; excludes TimeoutException).
standard_retry, exec_retry = make_retry_decorators(
    _is_retryable, _is_retryable_for_exec
)


async def run_with_timeout_retry(
    run_fn: Callable[[int | None], Awaitable[T]],
    timeout: int | None,
    timeout_retry: bool,
) -> T:
    """Execute *run_fn* with decreasing timeout caps on TimeoutException.

    On the first timeout, retries with cap ≤60 s, then ≤30 s. ``_is_timeout``
    also catches an unwrapped ``httpx.TimeoutException``.
    """
    return await _run_with_timeout_retry(
        run_fn, timeout, timeout_retry, is_timeout=_is_timeout
    )
