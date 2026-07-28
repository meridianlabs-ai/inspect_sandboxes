"""Sandbox naming helpers shared across providers."""

from __future__ import annotations

import re
import uuid
from typing import Any, Final

_SLUG_RE = re.compile(r"[^a-z0-9_-]+")
_SLUG_STRIP = "-_"
_SLUG_MAX_LEN = 40
_HEX_LEN = 8
_SANDBOX_NAME_MAX_LEN: Final[int] = 64
_SANDBOX_NAME_PREFIX: Final[str] = "inspect"


def _slug(value: Any) -> str:
    """Lowercase; replace non-alnum runs with '-'; preserve underscores; trim."""
    s = _SLUG_RE.sub("-", str(value).lower()).strip(_SLUG_STRIP)
    return s[:_SLUG_MAX_LEN].rstrip(_SLUG_STRIP)


def _compact_slug(slug: str, maximum_length: int) -> str:
    if len(slug) <= maximum_length:
        return slug

    prefix_length = (maximum_length - 1) // 2
    suffix_length = maximum_length - prefix_length - 1
    return f"{slug[:prefix_length]}-{slug[-suffix_length:]}"


def make_sandbox_name(task_name: str | None, metadata: dict[str, Any]) -> str:
    """Build a human-readable, globally-unique sandbox name.

    Format (components joined with ``-``):
        - Both task_name and ``metadata['__sample_id__']`` present:
            ``inspect-{task}-{sample}-{hex}``
        - Only task_name:  ``inspect-{task}-{hex}``
        - Only sample id:  ``inspect-{sample}-{hex}``
        - Neither:         ``inspect-{hex}``

    The trailing 8-char hex suffix keeps names unique across re-runs.
    """
    slugs: list[str] = []
    if task_name:
        task_slug = _slug(task_name)
        if task_slug:
            slugs.append(task_slug)
    sample_id = metadata.get("__sample_id__")
    if sample_id is not None:
        sample_slug = _slug(sample_id)
        if sample_slug:
            slugs.append(sample_slug)

    slug_budget = _SANDBOX_NAME_MAX_LEN - (
        len(_SANDBOX_NAME_PREFIX) + _HEX_LEN + len(slugs) + 1
    )
    if slugs and sum(map(len, slugs)) > slug_budget:
        fair_share = slug_budget // len(slugs)
        slug_lengths = [min(len(slug), fair_share) for slug in slugs]
        remaining = slug_budget - sum(slug_lengths)
        for index, slug in enumerate(slugs):
            additional = min(remaining, len(slug) - slug_lengths[index])
            slug_lengths[index] += additional
            remaining -= additional
        slugs = [
            _compact_slug(slug, maximum_length)
            for slug, maximum_length in zip(slugs, slug_lengths, strict=True)
        ]

    parts = [_SANDBOX_NAME_PREFIX, *slugs, uuid.uuid4().hex[:_HEX_LEN]]
    return "-".join(parts)
