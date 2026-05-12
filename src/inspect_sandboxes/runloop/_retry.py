"""Retry decorators for the Runloop sandbox provider.

The Runloop SDK retries some transients internally but doesn't retry
sandbox-level errors. We wrap lifecycle and exec operations with tenacity
so rate-limits and transient API errors don't surface as test failures.

Permanent errors (NotFoundError, AuthenticationError, BadRequestError,
ConflictError, UnprocessableEntityError, PermissionDeniedError) are not
retried.

``APITimeoutError`` is handled separately by ``run_with_timeout_retry``,
which mirrors the ``SandboxEnvironment.exec`` contract of "first retry
≤60 s, second ≤30 s".
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TypeVar

from runloop_api_client import (
    APIError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    ConflictError,
    NotFoundError,
    PermissionDeniedError,
    UnprocessableEntityError,
)
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

T = TypeVar("T")

_PERMANENT_EXCEPTIONS = (
    NotFoundError,
    AuthenticationError,
    BadRequestError,
    ConflictError,
    PermissionDeniedError,
    UnprocessableEntityError,
)


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, _PERMANENT_EXCEPTIONS):
        return False
    return isinstance(exc, APIError)


# Retry decorator for devbox lifecycle and file I/O operations.
standard_retry = retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception(_is_retryable),
    reraise=True,
)


def _is_retryable_for_exec(exc: BaseException) -> bool:
    # APITimeoutError is handled by run_with_timeout_retry below.
    if isinstance(exc, APITimeoutError):
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
    """Execute *run_fn* with decreasing timeout caps on APITimeoutError.

    On the first timeout, retries with cap ≤60 s, then ≤30 s.
    """
    if timeout_retry:
        t1 = min(timeout, 60) if timeout is not None else 60
        t2 = min(timeout, 30) if timeout is not None else 30
        attempt_timeouts: list[int | None] = [timeout, t1, t2]
    else:
        attempt_timeouts = [timeout]

    # Runloop's SDK raises APITimeoutError directly on HTTP-layer timeouts;
    # no httpx unwrapping needed (unlike E2B).
    last_timeout_exc: APITimeoutError | None = None
    for t in attempt_timeouts:
        try:
            return await run_fn(t)
        except APITimeoutError as e:
            last_timeout_exc = e

    assert last_timeout_exc is not None
    raise TimeoutError(
        f"Command timed out after {timeout} seconds"
    ) from last_timeout_exc
