"""Retry utilities for the Daytona sandbox provider.

The Daytona SDK has no built-in retry, so we handle it here. All DaytonaError
subclasses (including DaytonaRateLimitError) are retried with exponential
backoff, except DaytonaTimeoutError which is handled separately by
run_with_timeout_retry.

Note: create_sandbox does not use these decorators — it has its own retry
loop that respins the name on each attempt and intentionally retries
DaytonaTimeoutError (a timed-out create is the canonical name-holding zombie).
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TypeVar

from daytona import DaytonaError, DaytonaTimeoutError

from inspect_sandboxes._util.retry import (
    make_retry_decorators,
)
from inspect_sandboxes._util.retry import (
    run_with_timeout_retry as _run_with_timeout_retry,
)

T = TypeVar("T")


def _is_retryable(exc: BaseException) -> bool:
    return isinstance(exc, DaytonaError)


def _is_retryable_for_exec(exc: BaseException) -> bool:
    # DaytonaTimeoutError is handled by run_with_timeout_retry below.
    return isinstance(exc, DaytonaError) and not isinstance(exc, DaytonaTimeoutError)


def _is_timeout(exc: BaseException) -> bool:
    # The SDK sometimes raises a plain DaytonaError (not DaytonaTimeoutError)
    # for exec timeouts — detect those via message.
    if isinstance(exc, DaytonaTimeoutError):
        return True
    return isinstance(exc, DaytonaError) and "timeout" in str(exc).lower()


# standard_retry: sandbox lifecycle + file I/O (5 attempts).
# exec_retry: exec/VM commands (3 attempts; excludes DaytonaTimeoutError).
standard_retry, exec_retry = make_retry_decorators(
    _is_retryable, _is_retryable_for_exec
)


async def run_with_timeout_retry(
    run_fn: Callable[[int | None], Awaitable[T]],
    timeout: int | None,
    timeout_retry: bool,
) -> T:
    """Execute *run_fn* with decreasing timeout caps on DaytonaTimeoutError.

    On the first timeout, retries with cap ≤60 s, then ≤30 s. ``_is_timeout``
    also detects a message-only timeout (a plain DaytonaError).
    """
    return await _run_with_timeout_retry(
        run_fn, timeout, timeout_retry, is_timeout=_is_timeout
    )
