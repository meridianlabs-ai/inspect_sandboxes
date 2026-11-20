"""Pure compose-handling helpers shared by DinD-capable providers.

These functions don't touch any provider SDK — they operate on parsed
compose configs and on local filesystem paths. Provider-specific upload
and exec stay in each provider's ``_dind_project.py``.
"""

from __future__ import annotations

from copy import deepcopy
from logging import getLogger
from pathlib import Path

import yaml
from inspect_ai.util import ComposeConfig, ComposeService
from inspect_ai.util._sandbox.docker.service import parse_duration

logger = getLogger(__name__)

DEFAULT_SERVICE_TIMEOUT = 120


def compute_healthcheck_timeout(
    services: dict[str, ComposeService],
    default: int = DEFAULT_SERVICE_TIMEOUT,
) -> int:
    """Return the longest ``retries * (interval + timeout)`` across services.

    Used to size the ``docker compose up --wait-timeout`` and the wait loop.
    """
    max_time = 0
    for service in services.values():
        hc = service.healthcheck
        if hc is None:
            continue
        retries = hc.retries if hc.retries is not None else 3
        interval = int(parse_duration(hc.interval).seconds) if hc.interval else 30
        timeout = int(parse_duration(hc.timeout).seconds) if hc.timeout else 30
        max_time = max(max_time, retries * (interval + timeout))
    return max_time if max_time > 0 else default


_REMOTE_CONTEXT_PREFIXES = ("github.com/", "git@")


def _is_local_relative_context(ctx: str) -> bool:
    """True if *ctx* is a relative local path (not absolute, not a remote URL).

    Remote contexts follow Compose's rule: anything containing ``://`` (git,
    http(s), ssh, ``docker-image://``, ...) or starting with ``github.com/``
    or ``git@``.
    """
    if "://" in ctx or ctx.startswith(_REMOTE_CONTEXT_PREFIXES):
        return False
    return not Path(ctx).is_absolute()


def resolve_build_contexts(config: ComposeConfig, base_dir: Path) -> ComposeConfig:
    """Return a copy of *config* with relative build contexts made absolute.

    A ``ComposeConfig`` passed without a compose file is serialized to a temp
    directory, so its relative build contexts must be resolved against
    *base_dir* (the working directory, as Inspect's Docker provider does)
    before that happens. Absolute paths and remote (git/URL) contexts are left
    unchanged.
    """
    config_copy = deepcopy(config)
    for service in config_copy.services.values():
        if isinstance(service.build, str):
            if _is_local_relative_context(service.build):
                service.build = str(base_dir / service.build)
        elif service.build is not None:
            ctx = service.build.context or "."
            if _is_local_relative_context(ctx):
                service.build.context = str(base_dir / ctx)
    return config_copy


def discover_build_contexts(
    config: ComposeConfig,
    compose_dir: Path,
    build_context_dir: str,
) -> tuple[dict[str, str], bool]:
    """Discover local build contexts that need to be uploaded into the sandbox.

    Args:
        config: Parsed compose config.
        compose_dir: Local directory containing the compose file.
        build_context_dir: Sandbox-side directory under which external contexts
            are placed (e.g. ``/inspect/contexts``). Each context gets its own
            subdirectory named after the service.

    Returns:
        A ``(context_map, needs_rewrite)`` tuple where ``context_map`` maps
        absolute local paths to remote sandbox paths, and ``needs_rewrite`` is
        True iff at least one context lives outside ``compose_dir`` (in which
        case the compose file must be rewritten to point at the remote paths).
    """
    context_map: dict[str, str] = {}
    needs_rewrite = False

    for svc_name, service in config.services.items():
        if not service.build:
            continue
        ctx = service.build if isinstance(service.build, str) else service.build.context
        if not ctx:
            continue
        local_ctx = Path(ctx)
        if not local_ctx.is_absolute():
            local_ctx = (compose_dir / local_ctx).resolve()
        if not local_ctx.exists():
            logger.warning(
                "Build context '%s' for service '%s' does not exist, skipping upload",
                local_ctx,
                svc_name,
            )
            continue
        local_key = str(local_ctx)
        if local_key not in context_map:
            context_map[local_key] = f"{build_context_dir}/{svc_name}"
        try:
            local_ctx.relative_to(compose_dir)
        except ValueError:
            needs_rewrite = True

    return context_map, needs_rewrite


def rewrite_compose_yaml(
    config: ComposeConfig,
    compose_dir: Path,
    context_map: dict[str, str],
) -> str:
    """Return YAML for *config* with build contexts replaced by remote paths.

    Use after :func:`discover_build_contexts` reports ``needs_rewrite=True``.
    """
    config_copy = deepcopy(config)
    for service in config_copy.services.values():
        if not service.build:
            continue
        ctx = service.build if isinstance(service.build, str) else service.build.context
        if not ctx:
            continue
        local_ctx = Path(ctx)
        if not local_ctx.is_absolute():
            local_ctx = (compose_dir / local_ctx).resolve()
        local_key = str(local_ctx)
        if local_key in context_map:
            remote = context_map[local_key]
            if isinstance(service.build, str):
                service.build = remote
            else:
                service.build.context = remote

    return yaml.dump(
        config_copy.model_dump(by_alias=True, exclude_none=True, exclude_defaults=True),
        sort_keys=False,
    )
