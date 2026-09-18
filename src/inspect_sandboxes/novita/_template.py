"""Novita template build and content-hash caching.

Novita templates must be pre-built before a sandbox can be created from them.
This module computes a deterministic name from the build inputs (Dockerfile
contents or base image, plus resource budget) so identical inputs share the
same cached template across users and runs.

The Dockerfile path does *not* short-circuit via ``AsyncTemplate.exists(name)``:
our hash captures only the Dockerfile text, not the contents of files pulled in
by ``COPY``/``ADD``, so a stale cached template could outlive valid input
changes. Instead we call ``AsyncTemplate.build()`` every time and rely on the
SDK's per-instruction layer cache (see
``novita_sandbox.core.template_async.main.AsyncTemplate._build``), which hashes
COPY sources and reuses prior layers on a match — unchanged builds remain fast.

The image path *does* short-circuit via ``exists()``: there are no COPY/ADD
sources to invalidate against, so the only thing the build call would do on a
cache hit is round-trip the API.

The cache key is global (no project prefix) so identical inputs share builds
across users and runs.
"""

from __future__ import annotations

import hashlib
import json
from logging import getLogger
from pathlib import Path

from inspect_ai.util import trace_message
from novita_sandbox.core import AsyncTemplate

logger = getLogger(__name__)

TEMPLATE_NAME_PREFIX = "inspect-"
DEFAULT_CPU_COUNT = 2
DEFAULT_MEMORY_MB = 1024
_HASH_LEN = 12


def _hash_inputs(payload: dict[str, object]) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:_HASH_LEN]


def template_name_for_dockerfile(
    dockerfile_path: str,
    *,
    cpu_count: int = DEFAULT_CPU_COUNT,
    memory_mb: int = DEFAULT_MEMORY_MB,
) -> str:
    """Cached template name for a Dockerfile + resource budget."""
    content = Path(dockerfile_path).read_bytes().decode("utf-8", errors="replace")
    h = _hash_inputs(
        {
            "kind": "dockerfile",
            "content": content,
            "cpu_count": cpu_count,
            "memory_mb": memory_mb,
        }
    )
    return f"{TEMPLATE_NAME_PREFIX}{h}"


def template_name_for_image(
    image: str,
    *,
    cpu_count: int = DEFAULT_CPU_COUNT,
    memory_mb: int = DEFAULT_MEMORY_MB,
) -> str:
    """Cached template name for a base image + resource budget."""
    h = _hash_inputs(
        {
            "kind": "image",
            "image": image,
            "cpu_count": cpu_count,
            "memory_mb": memory_mb,
        }
    )
    return f"{TEMPLATE_NAME_PREFIX}{h}"


async def build_template_for_dockerfile(
    dockerfile_path: str,
    *,
    cpu_count: int = DEFAULT_CPU_COUNT,
    memory_mb: int = DEFAULT_MEMORY_MB,
) -> str:
    """Build (or reuse cached) Novita template from a Dockerfile."""
    name = template_name_for_dockerfile(
        dockerfile_path, cpu_count=cpu_count, memory_mb=memory_mb
    )
    template = AsyncTemplate(
        file_context_path=Path(dockerfile_path).parent
    ).from_dockerfile(dockerfile_path)
    trace_message(logger, "novita", f"Building template {name} from {dockerfile_path}")
    await AsyncTemplate.build(
        template, name=name, cpu_count=cpu_count, memory_mb=memory_mb
    )
    return name


async def build_template_for_image(
    image: str,
    *,
    cpu_count: int = DEFAULT_CPU_COUNT,
    memory_mb: int = DEFAULT_MEMORY_MB,
) -> str:
    """Build (or reuse cached) Novita template from a base image."""
    name = template_name_for_image(image, cpu_count=cpu_count, memory_mb=memory_mb)
    if await AsyncTemplate.exists(name):
        trace_message(logger, "novita", f"Template {name} cached, reusing")
        return name

    template = AsyncTemplate().from_image(image)
    trace_message(logger, "novita", f"Building template {name} from image {image}")
    await AsyncTemplate.build(
        template, name=name, cpu_count=cpu_count, memory_mb=memory_mb
    )
    return name
