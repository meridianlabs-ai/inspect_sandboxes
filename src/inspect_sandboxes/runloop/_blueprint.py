"""Runloop blueprint build and content-hash caching.

Runloop blueprints must be pre-built before a devbox can be created from
them. This module computes a deterministic name from the build inputs
(Dockerfile contents, build-context file contents, base image, plus
launch parameters) and short-circuits when a blueprint with that name
already exists (``build_complete``) or is in flight (we wait for it).

Runloop's API does not deduplicate by name — every call to
``blueprints.create`` produces a new blueprint with a fresh id, even when
the name and inputs are unchanged. To avoid accumulating duplicates we:

1. Probe ``blueprints.list(name=name)`` and reuse any ``build_complete``
   match, or wait on any in-flight (``queued``/``provisioning``/
   ``building``) match.
2. Issue ``blueprints.create`` with ``max_retries=0`` so SDK auto-retries
   on transient errors can't spawn duplicate blueprints.

We always tar the Dockerfile's parent directory, upload it as a Runloop
Object, and pass the object id as ``build_context`` — mirroring
``docker build``, which ships the whole context so any ``COPY``/``ADD``
in the Dockerfile can resolve its local references. The tarball is
streamed through a temp file rather than buffered whole in memory, and
context files are hashed in chunks for the same reason.

The cache key is global (no project prefix) so identical inputs share builds
across users and runs.
"""

from __future__ import annotations

import hashlib
import json
import os
import tarfile
import tempfile
from collections.abc import AsyncIterator, Iterator
from logging import getLogger
from pathlib import Path
from typing import IO
from uuid import uuid4

import httpx
from inspect_ai.util import trace_message
from runloop_api_client import AsyncRunloop
from runloop_api_client.lib.polling import PollingConfig
from runloop_api_client.types.shared_params import LaunchParameters

from ._single_env import FILE_REQUEST_TIMEOUT

logger = getLogger(__name__)

BLUEPRINT_NAME_PREFIX = "inspect-"
BLUEPRINT_BUILD_TIMEOUT = 1800
_HASH_LEN = 12

# Read/stream context files in 1 MiB blocks so a large file is never held in
# memory whole — for hashing or for building the upload tarball.
_CONTEXT_CHUNK_SIZE = 1024 * 1024

# Runloop SDK's default polling is 120 attempts × 1.0s ≈ 2 min, which isn't
# enough for queue+build of a non-trivial Dockerfile (pip install, etc.).
# Match the FILE_REQUEST_TIMEOUT convention (30 min defensive guardrail).
_BLUEPRINT_POLLING_CONFIG = PollingConfig(
    interval_seconds=2.0, timeout_seconds=BLUEPRINT_BUILD_TIMEOUT
)

# Best-effort exclusions when hashing/tarring a build context. These mirror
# the entries most ``.dockerignore`` files include; we don't parse
# ``.dockerignore`` itself yet.
_IGNORED_DIR_NAMES = {".git", "__pycache__"}
_IGNORED_FILE_NAMES = {".DS_Store"}


def _hash_inputs(payload: dict[str, object]) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:_HASH_LEN]


def _is_ignored_context_path(rel_path: Path) -> bool:
    for part in rel_path.parts:
        if part in _IGNORED_DIR_NAMES:
            return True
    name = rel_path.name
    return name in _IGNORED_FILE_NAMES or name.endswith(".pyc")


def _iter_context_files(context_dir: Path) -> Iterator[Path]:
    for path in sorted(context_dir.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(context_dir)
        if _is_ignored_context_path(rel):
            continue
        yield path


def _hash_build_context(context_dir: Path) -> str:
    """Hash every file in the build context.

    Any local change invalidates the cached blueprint name (mirrors
    Docker's COPY/ADD invalidation). Files are read in chunks so a large
    context file is never held in memory whole.
    """
    h = hashlib.sha256()
    for path in _iter_context_files(context_dir):
        rel = path.relative_to(context_dir).as_posix()
        h.update(rel.encode("utf-8"))
        h.update(b"\0")
        with path.open("rb") as f:
            while True:
                chunk = f.read(_CONTEXT_CHUNK_SIZE)
                if not chunk:
                    break
                h.update(chunk)
        h.update(b"\0")
    return h.hexdigest()


def _write_context_tarball(context_dir: Path, fileobj: IO[bytes]) -> None:
    """Stream a gzipped tar of the build context into ``fileobj``.

    ``tarfile`` writes incrementally, so the whole archive is never buffered
    in memory (unlike building it in a ``BytesIO``).
    """
    with tarfile.open(fileobj=fileobj, mode="w:gz") as tar:
        for path in _iter_context_files(context_dir):
            arcname = path.relative_to(context_dir).as_posix()
            tar.add(str(path), arcname=arcname)


async def _upload_build_context(client: AsyncRunloop, context_dir: Path) -> str:
    """Tar the build context and upload it as a Runloop Object.

    The tarball is streamed through a temp file on disk and PUT to the presigned
    URL in chunks (with an explicit Content-Length, since S3 rejects chunked
    transfer encoding), so multi-hundred-MiB contexts don't load into memory.

    Returns the Object id, suitable for ``build_context.object_id``.
    """
    obj = await client.objects.create(
        content_type="tgz",
        name=f"inspect-context-{uuid4().hex[:12]}",
    )
    if obj.upload_url is None:
        raise RuntimeError(
            "Runloop did not return an upload URL for the build-context Object."
        )
    with tempfile.NamedTemporaryFile(suffix=".tgz") as tmp:
        _write_context_tarball(context_dir, tmp)
        tmp.flush()
        size = os.fstat(tmp.fileno()).st_size

        # httpx.AsyncClient needs an async iterable for a streamed request body.
        async def _chunks() -> AsyncIterator[bytes]:
            tmp.seek(0)
            while True:
                block = tmp.read(_CONTEXT_CHUNK_SIZE)
                if not block:
                    break
                yield block

        async with httpx.AsyncClient(timeout=FILE_REQUEST_TIMEOUT) as http:
            response = await http.put(
                obj.upload_url,
                content=_chunks(),
                headers={"Content-Length": str(size)},
            )
            response.raise_for_status()
    await client.objects.complete(obj.id)
    return obj.id


def _launch_params_for_hash(
    launch_parameters: LaunchParameters | None,
) -> dict[str, object]:
    """Project LaunchParameters to a plain-dict form for hashing."""
    if not launch_parameters:
        return {}
    # TypedDict instances are already dicts at runtime; copy to a plain dict
    # so json.dumps can serialize deterministically with sort_keys.
    return {k: v for k, v in dict(launch_parameters).items() if v is not None}


def blueprint_name_for_dockerfile(
    dockerfile_path: str,
    *,
    launch_parameters: LaunchParameters | None = None,
) -> str:
    """Cached blueprint name for a Dockerfile + build context + launch params."""
    content = Path(dockerfile_path).read_bytes().decode("utf-8", errors="replace")
    context_hash = _hash_build_context(Path(dockerfile_path).parent)
    h = _hash_inputs(
        {
            "kind": "dockerfile",
            "content": content,
            "context": context_hash,
            "launch_parameters": _launch_params_for_hash(launch_parameters),
        }
    )
    return f"{BLUEPRINT_NAME_PREFIX}{h}"


def blueprint_name_for_image(
    image: str,
    *,
    launch_parameters: LaunchParameters | None = None,
) -> str:
    """Cached blueprint name for a base image + launch parameters."""
    h = _hash_inputs(
        {
            "kind": "image",
            "image": image,
            "launch_parameters": _launch_params_for_hash(launch_parameters),
        }
    )
    return f"{BLUEPRINT_NAME_PREFIX}{h}"


async def _find_or_await_blueprint(client: AsyncRunloop, name: str) -> bool:
    """Return True if a usable blueprint with this name exists.

    Runloop's API allows multiple blueprints to share a name. "Usable" means
    either ``build_complete``, or in-flight (``queued``/``provisioning``/
    ``building``) that we successfully wait to completion. Returns False
    only when nothing matches or all in-flight attempts failed — at which
    point the caller creates a fresh one.

    This dedupes concurrent ``sample_init`` calls and the case where a prior
    run left an in-flight blueprint behind.
    """
    in_flight: list[str] = []
    async for bp in client.blueprints.list(name=name):
        if bp.status == "build_complete":
            return True
        if bp.status in ("queued", "provisioning", "building"):
            in_flight.append(bp.id)

    for bp_id in in_flight:
        try:
            await client.blueprints.await_build_complete(
                bp_id, polling_config=_BLUEPRINT_POLLING_CONFIG
            )
            return True
        except Exception:
            continue
    return False


async def build_blueprint_for_dockerfile(
    client: AsyncRunloop,
    dockerfile_path: str,
    *,
    launch_parameters: LaunchParameters | None = None,
) -> str:
    """Build (or reuse cached) Runloop blueprint from a Dockerfile.

    Returns the blueprint name.
    """
    name = blueprint_name_for_dockerfile(
        dockerfile_path, launch_parameters=launch_parameters
    )
    if await _find_or_await_blueprint(client, name):
        trace_message(logger, "runloop", f"Blueprint {name} cached, reusing")
        return name

    path = Path(dockerfile_path)
    dockerfile = path.read_bytes().decode("utf-8", errors="replace")
    object_id = await _upload_build_context(client, path.parent)
    trace_message(
        logger, "runloop", f"Building blueprint {name} from {dockerfile_path}"
    )
    try:
        # POST without SDK retries: Runloop's blueprint create is not
        # idempotent, so retries spawn duplicate blueprints (and blow the
        # account cap). `idempotency_key` is also sent, but Runloop doesn't
        # currently honor it for this endpoint. Polling uses the default
        # client so transient retrieve errors are still retried.
        blueprint = await client.with_options(max_retries=0).blueprints.create(
            name=name,
            dockerfile=dockerfile,
            build_context={"object_id": object_id, "type": "object"},
            launch_parameters=launch_parameters or {},
            idempotency_key=name,
        )
        await client.blueprints.await_build_complete(
            blueprint.id, polling_config=_BLUEPRINT_POLLING_CONFIG
        )
    finally:
        # The Object is only needed during the build; the resulting blueprint
        # carries the layers. Delete it so we don't blow Runloop's per-account
        # Object cap (3 on the free tier).
        try:
            await client.objects.delete(object_id)
        except Exception:
            pass
    return name


async def build_blueprint_for_image(
    client: AsyncRunloop,
    image: str,
    *,
    launch_parameters: LaunchParameters | None = None,
) -> str:
    """Build (or reuse cached) Runloop blueprint from a base image."""
    name = blueprint_name_for_image(image, launch_parameters=launch_parameters)
    if await _find_or_await_blueprint(client, name):
        trace_message(logger, "runloop", f"Blueprint {name} cached, reusing")
        return name

    dockerfile = f"FROM {image}\n"
    trace_message(logger, "runloop", f"Building blueprint {name} from image {image}")
    # POST without SDK retries: Runloop's blueprint create is not idempotent,
    # so retries spawn duplicate blueprints (and blow the account cap).
    # `idempotency_key` is also sent, but Runloop doesn't currently honor it
    # for this endpoint. Polling uses the default client so transient
    # retrieve errors are still retried.
    blueprint = await client.with_options(max_retries=0).blueprints.create(
        name=name,
        dockerfile=dockerfile,
        launch_parameters=launch_parameters or {},
        idempotency_key=name,
    )
    await client.blueprints.await_build_complete(
        blueprint.id, polling_config=_BLUEPRINT_POLLING_CONFIG
    )
    return name
