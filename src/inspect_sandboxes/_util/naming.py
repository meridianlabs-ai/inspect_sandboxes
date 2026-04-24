"""Sandbox naming helpers shared across providers."""

from __future__ import annotations

import re
import uuid
from typing import Any

_SLUG_RE = re.compile(r"[^a-z0-9_-]+")
_SLUG_STRIP = "-_"
_SLUG_MAX_LEN = 40
_HEX_LEN = 8


def _slug(value: Any) -> str:
    """Lowercase; replace non-alnum runs with '-'; preserve underscores; trim."""
    s = _SLUG_RE.sub("-", str(value).lower()).strip(_SLUG_STRIP)
    return s[:_SLUG_MAX_LEN].rstrip(_SLUG_STRIP)


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
    parts: list[str] = ["inspect"]
    if task_name:
        task_slug = _slug(task_name)
        if task_slug:
            parts.append(task_slug)
    sample_id = metadata.get("__sample_id__")
    if sample_id is not None:
        sample_slug = _slug(sample_id)
        if sample_slug:
            parts.append(sample_slug)
    parts.append(uuid.uuid4().hex[:_HEX_LEN])
    return "-".join(parts)
