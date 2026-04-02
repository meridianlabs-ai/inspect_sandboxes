"""Shared retry utilities for the Daytona sandbox provider.

The Daytona SDK has no built-in retry, so we handle it here.
All DaytonaError subclasses (including DaytonaRateLimitError)
are retried with exponential backoff, except DaytonaTimeoutError which
is handled separately by run_with_timeout_retry.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TypeVar

from daytona_sdk import DaytonaError, DaytonaTimeoutError
from tenacity import (
    retry,
    retry_if_exception,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

T = TypeVar("T")

# Retry decorator for sandbox lifecycle and file I/O operations
standard_retry = retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type(DaytonaError),
    reraise=True,
)

# Retry decorator for exec and VM command operations — excludes
# DaytonaTimeoutError which should propagate immediately (either to
# exec()'s timeout retry loop, or as a real timeout for long-running ops).
exec_retry = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception(
        lambda e: isinstance(e, DaytonaError) and not isinstance(e, DaytonaTimeoutError)
    ),
    reraise=True,
)


async def run_with_timeout_retry(
    run_fn: Callable[[int | None], Awaitable[T]],
    timeout: int | None,
    timeout_retry: bool,
) -> T:
    """Execute *run_fn* with decreasing timeout caps on DaytonaTimeoutError.

    On the first timeout, retries with cap ≤60 s, then ≤30 s.
    """
    if timeout_retry:
        t1 = min(timeout, 60) if timeout is not None else 60
        t2 = min(timeout, 30) if timeout is not None else 30
        attempt_timeouts: list[int | None] = [timeout, t1, t2]
    else:
        attempt_timeouts = [timeout]

    last_timeout_exc: DaytonaError | None = None
    for t in attempt_timeouts:
        try:
            return await run_fn(t)
        except DaytonaTimeoutError as e:
            last_timeout_exc = e
        except DaytonaError as e:
            # The SDK sometimes raises DaytonaError (not DaytonaTimeoutError)
            # for exec timeouts — detect via message.
            if "timeout" in str(e).lower():
                last_timeout_exc = e
            else:
                raise

    assert last_timeout_exc is not None
    raise TimeoutError(
        f"Command timed out after {timeout} seconds"
    ) from last_timeout_exc
