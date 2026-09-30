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
from runloop_api_client.lib.polling import PollingConfig
from runloop_api_client.types import DevboxAsyncExecutionDetailView

from inspect_sandboxes._util.retry import (
    make_retry_decorators,
)
from inspect_sandboxes._util.retry import (
    run_with_timeout_retry as _run_with_timeout_retry,
)

T = TypeVar("T")

# Devbox provisioning (image pull + boot) can exceed the SDK's default poll
# window (~2 min). Use a longer guardrail so a slow provision doesn't raise
# PollingTimeout and leak a half-provisioned devbox before it's tracked.
DEVBOX_CREATE_POLLING_CONFIG = PollingConfig(interval_seconds=2.0, timeout_seconds=300)

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


def _is_retryable_for_exec(exc: BaseException) -> bool:
    # APITimeoutError is handled by run_with_timeout_retry below.
    if isinstance(exc, APITimeoutError):
        return False
    return _is_retryable(exc)


def _is_timeout(exc: BaseException) -> bool:
    # APITimeoutError: the SDK's HTTP-layer timeout (raised directly; no httpx
    # unwrapping needed, unlike E2B). TimeoutError: our own poll_execution
    # deadline, when we poll an async execution to enforce a short user timeout.
    return isinstance(exc, (APITimeoutError, TimeoutError))


# standard_retry: devbox lifecycle + file I/O (5 attempts).
# exec_retry: exec operations (3 attempts; excludes APITimeoutError).
standard_retry, exec_retry = make_retry_decorators(
    _is_retryable, _is_retryable_for_exec
)


@standard_retry
async def shutdown_devbox(client: AsyncRunloop, devbox_id: str) -> None:
    """Shut down a devbox, retrying transient API errors.

    ``NotFoundError`` is permanent (see ``_is_retryable``) so it propagates to
    the caller, which treats an already-gone devbox as success.
    """
    await client.devboxes.shutdown(devbox_id)


# Polling backoff for async executions: start fast, back off to a ~1 s cap so a
# command that finishes mid-interval is observed promptly (a larger cap adds
# dead time proportional to the cap). Transient retrieve errors are tolerated in
# place (bounded) so we keep polling the *same* execution instead of
# re-submitting the command.
_POLL_INTERVAL_INITIAL = 0.5
_POLL_INTERVAL_MAX = 1.0
_POLL_MAX_TRANSIENT_ERRORS = 5
# Intermediate polls only need the status, so they fetch a tiny tail; the full
# output is fetched once on completion with the caller's last_n. Polling with
# the full last_n would re-transfer (and discard) the whole stream every poll.
_POLL_LAST_N = "1"


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
    from 0.5 s to 1 s between polls with a tiny ``last_n``, then re-fetches the
    full output (the caller's ``last_n``) once on completion. A transient
    (retryable) API error is swallowed and retried in place — we keep polling
    the same execution rather than re-running the command — up to
    ``_POLL_MAX_TRANSIENT_ERRORS`` consecutive failures, after which it
    propagates. Non-retryable errors propagate immediately.

    Raises ``TimeoutError`` if ``timeout`` seconds elapse first, after a
    best-effort kill of the execution.
    """
    deadline = time.monotonic() + timeout if timeout is not None else None
    interval = _POLL_INTERVAL_INITIAL
    transient = 0
    while True:
        try:
            execution = await client.devboxes.executions.retrieve(
                execution_id, devbox_id=devbox_id, last_n=_POLL_LAST_N
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
            return await client.devboxes.executions.retrieve(
                execution_id, devbox_id=devbox_id, last_n=last_n
            )
        if deadline is not None and time.monotonic() >= deadline:
            await _kill_execution(client, devbox_id, execution_id)
            raise TimeoutError(f"Command timed out after {timeout} seconds")
        await _sleep_bounded(interval, deadline)
        interval = min(interval * 2, _POLL_INTERVAL_MAX)


# `execute` waits up to this many seconds for the command to finish before
# returning the still-running execution to poll (Runloop's server-side max).
# The wait must be non-zero: `optimistic_timeout=0` makes Runloop return HTTP
# 408 instead of the running execution.
OPTIMISTIC_TIMEOUT_MAX = 25


async def execute_with_poll(
    client: AsyncRunloop,
    devbox_id: str,
    command: str,
    command_id: str,
    timeout: int | None,
    *,
    last_n: str,
) -> DevboxAsyncExecutionDetailView:
    """Submit a command via ``execute`` and return its completed execution.

    ``execute`` waits up to ``optimistic_timeout`` for the command to finish
    (the command is not killed if it overruns) and then returns the still-
    running execution, which we poll for the rest of the timeout. The stable
    ``command_id`` lets a retried submit dedupe rather than double-run.

    An ``APITimeoutError`` propagates uncaught: the caller's retry resubmits
    under the same ``command_id`` and reattaches to this still-running
    execution, so killing it here would force a re-run instead.
    """
    # Bound the optimistic wait by the timeout so a short one still fires on
    # schedule; it must stay non-zero (0 => HTTP 408).
    optimistic_timeout = (
        OPTIMISTIC_TIMEOUT_MAX
        if timeout is None
        else max(1, min(timeout, OPTIMISTIC_TIMEOUT_MAX))
    )

    @exec_retry
    async def _submit() -> DevboxAsyncExecutionDetailView:
        return await client.devboxes.execute(
            devbox_id,
            command=command,
            command_id=command_id,
            optimistic_timeout=optimistic_timeout,
            last_n=last_n,
        )

    start = time.monotonic()
    response = await _submit()
    if response.status != "completed":
        # Deduct the optimistic wait already spent so the total honors the
        # timeout.
        remaining = (
            None if timeout is None else max(0, timeout - int(time.monotonic() - start))
        )
        response = await poll_execution(
            client, devbox_id, response.execution_id, remaining, last_n=last_n
        )
    return response


async def run_with_timeout_retry(
    run_fn: Callable[[int | None], Awaitable[T]],
    timeout: int | None,
    timeout_retry: bool,
) -> T:
    """Execute *run_fn* with decreasing timeout caps on a timeout.

    On the first timeout, retries with cap ≤60 s, then ≤30 s. Both timeout
    flavors engage the ladder (``APITimeoutError`` and our own
    ``poll_execution`` ``TimeoutError``) — see ``_is_timeout``.
    """
    return await _run_with_timeout_retry(
        run_fn, timeout, timeout_retry, is_timeout=_is_timeout
    )
