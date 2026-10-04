"""Devbox lifecycle helpers for the Runloop sandbox provider."""

from __future__ import annotations

from contextlib import suppress
from typing import Any

from runloop_api_client import AsyncRunloop
from runloop_api_client.lib.polling import PollingTimeout
from runloop_api_client.types import DevboxView

from ._retry import standard_retry

# Page size for paginated list responses.
_LIST_PAGE_LIMIT = 100

# Devbox statuses that are already terminal — excluded from cleanup listing so
# we don't try to reclaim them. Every other status (provisioning, suspended,
# etc.) is still alive and billable, so it must be reclaimable.
_TERMINAL_DEVBOX_STATUSES = {"shutdown", "failure"}


@standard_retry
async def shutdown_devbox(client: AsyncRunloop, devbox_id: str) -> None:
    await client.devboxes.shutdown(devbox_id)


async def create_devbox(client: AsyncRunloop, **create_kwargs: Any) -> DevboxView:
    """Create a devbox and wait for it to reach running.

    ``create_and_await_running`` creates the devbox and *then* polls, so a
    ``PollingTimeout`` means the devbox exists but never came up. Shut it down
    before re-raising rather than leaking it until the task-cleanup orphan
    sweep — a stranded devbox holds a concurrency slot for the rest of the run.
    """
    try:
        return await client.devboxes.create_and_await_running(**create_kwargs)  # type: ignore[arg-type]
    except PollingTimeout as e:
        devbox = e.last_value
        if isinstance(devbox, DevboxView):
            with suppress(Exception):
                await shutdown_devbox(client, devbox.id)
        raise


async def list_devboxes(client: AsyncRunloop, metadata: dict[str, str]) -> list[Any]:
    """List non-terminal devboxes whose metadata matches *all* key/value pairs.

    Runloop's ``devboxes.list`` API doesn't accept a metadata filter, so we
    paginate and filter client-side. We keep every non-terminal devbox (not just
    ``running``) so provisioning/suspended ones are still reclaimed by cleanup.
    """
    matches: list[Any] = []
    async for devbox in client.devboxes.list(limit=_LIST_PAGE_LIMIT):
        if getattr(devbox, "status", None) in _TERMINAL_DEVBOX_STATUSES:
            continue
        meta = getattr(devbox, "metadata", None) or {}
        if all(meta.get(k) == v for k, v in metadata.items()):
            matches.append(devbox)
    return matches
