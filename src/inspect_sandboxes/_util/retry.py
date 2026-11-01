"""Shared retry machinery for SDK-backed sandbox providers.

The tenacity configuration and the ``SandboxEnvironment.exec`` timeout-retry
ladder ("first retry ≤60 s, second ≤30 s") live here so a fix reaches every
provider. Each provider supplies only its SDK-specific predicates: which
exceptions are retryable, and which represent a timeout.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

T = TypeVar("T")

# Predicate over a raised exception (retryable? / timeout?).
Predicate = Callable[[BaseException], bool]

_WAIT = wait_exponential(multiplier=1, min=1, max=10)


def make_retry_decorators(
    is_retryable: Predicate, is_exec_retryable: Predicate
) -> tuple[Callable[..., Any], Callable[..., Any]]:
    """Build the ``(standard_retry, exec_retry)`` decorators for a provider.

    ``standard_retry`` (lifecycle + file I/O) retries up to 5 attempts;
    ``exec_retry`` (exec/VM commands) up to 3. Both use the same exponential
    backoff and ``reraise=True``, differing only in the supplied predicate —
    ``is_exec_retryable`` excludes the provider's timeout exception so the
    timeout-retry ladder handles it instead.
    """
    standard_retry = retry(
        stop=stop_after_attempt(5),
        wait=_WAIT,
        retry=retry_if_exception(is_retryable),
        reraise=True,
    )
    exec_retry = retry(
        stop=stop_after_attempt(3),
        wait=_WAIT,
        retry=retry_if_exception(is_exec_retryable),
        reraise=True,
    )
    return standard_retry, exec_retry


async def run_with_timeout_retry(
    run_fn: Callable[[int | None], Awaitable[T]],
    timeout: int | None,
    timeout_retry: bool,
    *,
    is_timeout: Predicate,
) -> T:
    """Execute ``run_fn`` with decreasing timeout caps on a timeout.

    On the first timeout, retries with cap ≤60 s, then ≤30 s. A raised
    exception engages the ladder only when ``is_timeout`` accepts it; anything
    else propagates immediately.
    """
    if timeout_retry:
        t1 = min(timeout, 60) if timeout is not None else 60
        t2 = min(timeout, 30) if timeout is not None else 30
        attempt_timeouts: list[int | None] = [timeout, t1, t2]
    else:
        attempt_timeouts = [timeout]

    last_timeout_exc: BaseException | None = None
    for t in attempt_timeouts:
        try:
            return await run_fn(t)
        except Exception as e:
            if not is_timeout(e):
                raise
            last_timeout_exc = e

    assert last_timeout_exc is not None
    raise TimeoutError(
        f"Command timed out after {timeout} seconds"
    ) from last_timeout_exc
