"""Retry decorators for the Novita sandbox provider.

Novita's SDK does some retry under the hood for HTTP transients, but not for
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

import httpcore
import httpx
from novita_sandbox.core import (
    AuthenticationException,
    CommandExitException,
    InvalidArgumentException,
    NotFoundException,
    SandboxException,
    TimeoutException,
)
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
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
    # The SDK lets raw httpx/httpcore transport errors (e.g. ReadError from a
    # dropped connection) escape unwrapped; retry those too. Timeouts are
    # excluded — run_with_timeout_retry owns them.
    if isinstance(exc, (httpx.TimeoutException, httpcore.TimeoutException)):
        return False
    return isinstance(
        exc, (SandboxException, httpx.TransportError, httpcore.NetworkError)
    )


# Retry decorator for sandbox lifecycle and file I/O operations.
standard_retry = retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception(_is_retryable),
    reraise=True,
)


def _is_retryable_for_exec(exc: BaseException) -> bool:
    # TimeoutException is handled by run_with_timeout_retry below.
    if isinstance(exc, TimeoutException):
        return False
    return _is_retryable(exc)


# Retry decorator for exec operations.
exec_retry = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception(_is_retryable_for_exec),
    reraise=True,
)


async def run_with_timeout_retry(
    run_fn: Callable[[int | None], Awaitable[T]],
    timeout: int | None,
    timeout_retry: bool,
) -> T:
    """Execute *run_fn* with decreasing timeout caps on TimeoutException.

    On the first timeout, retries with cap ≤60 s, then ≤30 s.
    """
    if timeout_retry:
        t1 = min(timeout, 60) if timeout is not None else 60
        t2 = min(timeout, 30) if timeout is not None else 30
        attempt_timeouts: list[int | None] = [timeout, t1, t2]
    else:
        attempt_timeouts = [timeout]

    # Novita's SDK normally wraps timeouts as TimeoutException, but for short
    # command timeouts the underlying httpx.ReadTimeout can surface unwrapped.
    last_timeout_exc: TimeoutException | httpx.TimeoutException | None = None
    for t in attempt_timeouts:
        try:
            return await run_fn(t)
        except (TimeoutException, httpx.TimeoutException) as e:
            last_timeout_exc = e

    assert last_timeout_exc is not None
    raise TimeoutError(
        f"Command timed out after {timeout} seconds"
    ) from last_timeout_exc
