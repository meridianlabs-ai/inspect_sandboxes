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

import asyncio
import time
from collections.abc import Awaitable, Callable
from typing import TypeVar

from runloop_api_client import (
    APIError,
    APITimeoutError,
    AsyncRunloop,
    AuthenticationError,
    BadRequestError,
    ConflictError,
    NotFoundError,
    PermissionDeniedError,
    UnprocessableEntityError,
)
from runloop_api_client.types import DevboxAsyncExecutionDetailView
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


# Polling backoff for async executions: start fast, back off to a modest cap so
# long-running commands don't hammer the API. Transient retrieve errors are
# tolerated in place (bounded) so we keep polling the *same* execution instead
# of re-submitting the command.
_POLL_INTERVAL_INITIAL = 0.5
_POLL_INTERVAL_MAX = 5.0
_POLL_MAX_TRANSIENT_ERRORS = 5


async def _sleep_bounded(interval: float, deadline: float | None) -> None:
    """Sleep ``interval`` but never past ``deadline`` (so timeouts fire tight)."""
    if deadline is not None:
        interval = min(interval, max(0.0, deadline - time.monotonic()))
    await asyncio.sleep(interval)


async def _kill_execution(
    client: AsyncRunloop, devbox_id: str, execution_id: str
) -> None:
    """Best-effort kill of a running execution and its process group."""
    try:
        await client.devboxes.executions.kill(
            execution_id, devbox_id=devbox_id, kill_process_group=True
        )
    except Exception:
        pass


async def poll_execution(
    client: AsyncRunloop,
    devbox_id: str,
    execution_id: str,
    timeout: int | None,
    *,
    last_n: str,
) -> DevboxAsyncExecutionDetailView:
    """Poll a devbox execution until it completes, then return it.

    Polls ``devboxes.executions.retrieve`` for ``execution_id``, backing off
    from 0.5 s to 5 s between polls. A transient (retryable) API error is
    swallowed and retried in place — we keep polling the same execution rather
    than re-running the command — up to ``_POLL_MAX_TRANSIENT_ERRORS``
    consecutive failures, after which it propagates. Non-retryable errors
    propagate immediately.

    Raises ``TimeoutError`` if ``timeout`` seconds elapse first, after a
    best-effort kill of the execution.
    """
    deadline = time.monotonic() + timeout if timeout is not None else None
    interval = _POLL_INTERVAL_INITIAL
    transient = 0
    while True:
        try:
            execution = await client.devboxes.executions.retrieve(
                execution_id, devbox_id=devbox_id, last_n=last_n
            )
            transient = 0
        except Exception as exc:  # noqa: BLE001 — reraised unless retryable
            if not _is_retryable(exc):
                raise
            transient += 1
            if transient > _POLL_MAX_TRANSIENT_ERRORS:
                raise
            await _sleep_bounded(interval, deadline)
            interval = min(interval * 2, _POLL_INTERVAL_MAX)
            continue

        if execution.status == "completed":
            return execution
        if deadline is not None and time.monotonic() >= deadline:
            await _kill_execution(client, devbox_id, execution_id)
            raise TimeoutError(f"Command timed out after {timeout} seconds")
        await _sleep_bounded(interval, deadline)
        interval = min(interval * 2, _POLL_INTERVAL_MAX)


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

    # Two flavors of timeout can surface from run_fn:
    #   - APITimeoutError: the SDK's HTTP-layer timeout (raised directly; no
    #     httpx unwrapping needed, unlike E2B).
    #   - TimeoutError: our own deadline in poll_execution, when we poll an
    #     async execution ourselves to enforce a short user timeout exactly.
    # Both must engage the decreasing-cap retry, so we catch both.
    last_timeout_exc: BaseException | None = None
    for t in attempt_timeouts:
        try:
            return await run_fn(t)
        except (APITimeoutError, TimeoutError) as e:
            last_timeout_exc = e

    assert last_timeout_exc is not None
    raise TimeoutError(
        f"Command timed out after {timeout} seconds"
    ) from last_timeout_exc
