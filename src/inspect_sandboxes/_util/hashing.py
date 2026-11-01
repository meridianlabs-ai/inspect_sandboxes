"""Shared hashing helpers for deriving stable cache keys."""

from __future__ import annotations

import hashlib
import json

_HASH_LEN = 12


def hash_inputs(payload: dict[str, object]) -> str:
    """Return a short, stable hex digest of a JSON-serializable payload.

    Keys are sorted so the digest is independent of insertion order; the result
    is the first ``_HASH_LEN`` characters of the SHA-256 hex digest.
    """
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:_HASH_LEN]
