from __future__ import annotations

from logging import getLogger
from pathlib import Path
from typing import Any, NamedTuple

from inspect_ai.util import ComposeBuild, ComposeConfig, ComposeService, warn_once

from inspect_sandboxes._util.compose import (
    extract_extension,
    extract_extension_timeout,
    find_default_service,
    parse_environment,
    parse_service_ports,
    resolve_dockerfile_path,
    resolve_service_resources,
)

logger = getLogger(__name__)

DEFAULT_CPU_COUNT = 2
DEFAULT_MEMORY_MB = 1024


class NovitaSingleServiceParams(NamedTuple):
    """Resolved Novita params for a single-service compose config.

    Exactly one of ``template``, ``dockerfile_path``, ``image`` is set:

    - ``template``: pre-built template name, supplied via ``x-novita.template``.
    - ``dockerfile_path``: build a template from this Dockerfile.
    - ``image``: build a template from this base image.

    The orchestrator decides which build helper to call.
    """

    template: str | None
    dockerfile_path: str | None
    context_dir: str | None
    image: str | None
    cpu_count: int
    memory_mb: int
    envs: dict[str, str]
    user: str | None
    timeout: float | None
    metadata: dict[str, str]
    allow_internet_access: bool


def resolve_build_context(build: str | ComposeBuild, compose_dir: Path) -> Path:
    """Resolve the build *context* directory, separate from the Dockerfile path.

    Compose's ``build.context`` is what gets shipped to the build; the Dockerfile
    (``build.dockerfile``) may sit in a subdirectory of it, so the context is
    *not* simply the Dockerfile's parent (e.g. context ``.`` with dockerfile
    ``docker/Dockerfile`` -> context is ``.``, not ``docker/``).
    """
    if isinstance(build, str):
        return compose_dir / build
    return compose_dir / (build.context or ".")


def resolve_single_service_params(
    config: ComposeConfig,
    compose_path: str | None,
) -> NovitaSingleServiceParams:
    """Map a single-service compose config to Novita params.

    Args:
        config: Parsed compose configuration.
        compose_path: Path to the compose file for resolving relative
            Dockerfile paths. Pass ``None`` for an in-memory ``ComposeConfig``.
    """
    _, service = find_default_service(config)
    compose_dir = Path(compose_path).parent if compose_path else Path.cwd()
    extensions = extract_x_novita(config.extensions)

    template: str | None = extensions.get("template")
    dockerfile_path: Path | None = None
    context_dir: Path | None = None
    image: str | None = None
    if template is not None:
        pass
    elif service.build is not None:
        dockerfile_path = resolve_dockerfile_path(service.build, compose_dir)
        if not dockerfile_path.exists():
            raise FileNotFoundError(f"Dockerfile not found: {dockerfile_path}")
        context_dir = resolve_build_context(service.build, compose_dir)
    elif service.image:
        image = service.image
    else:
        raise ValueError(
            "Compose service must specify 'image', 'build', or x-novita.template"
        )

    cpu_count, memory_mb = _service_to_resources(service, extensions)

    envs: dict[str, str] = {}
    if service.environment:
        envs.update(parse_environment(service.environment))
    ext_envs = extensions.get("envs")
    if isinstance(ext_envs, dict):
        envs.update({str(k): str(v) for k, v in ext_envs.items()})

    user = extensions.get("user") or service.user

    timeout = extract_novita_timeout(config.extensions)

    # Translate Docker network_mode to Novita allow_internet_access;
    # x-novita.allow_internet_access takes precedence.
    if extensions.get("allow_internet_access") is not None:
        allow_internet_access = bool(extensions["allow_internet_access"])
    else:
        allow_internet_access = service.network_mode != "none"

    metadata_raw = extensions.get("metadata")
    metadata: dict[str, str] = {}
    if isinstance(metadata_raw, dict):
        metadata = {str(k): str(v) for k, v in metadata_raw.items()}

    return NovitaSingleServiceParams(
        template=template,
        dockerfile_path=str(dockerfile_path) if dockerfile_path else None,
        context_dir=str(context_dir) if context_dir else None,
        image=image,
        cpu_count=cpu_count,
        memory_mb=memory_mb,
        envs=envs,
        user=user,
        timeout=timeout,
        metadata=metadata,
        allow_internet_access=allow_internet_access,
    )


def service_connection_ports(service: ComposeService) -> list[int]:
    """Return the container ports to surface through ``connection()``.

    Novita mirrors Daytona's runtime model: ``get_host(port)`` returns a
    reachable host for any container port with no creation-time declaration. So
    we don't translate ports at creation; we record the container side of each
    ``service.ports`` entry for ``connection()`` to turn into ``get_host`` URLs.

    ``expose`` is host-private and is warned about, never surfaced. Port ranges
    and UDP entries are warned about and skipped.
    """
    if service.expose:
        warn_once(
            logger,
            "Novita does not surface Compose 'expose' ports. They stay "
            "host-private (reachable only by sibling services). Use 'ports' to "
            "get a host URL through connection().",
        )

    if not service.ports:
        return []

    parsed, unparseable = parse_service_ports(service.ports)
    for raw in unparseable:
        warn_once(
            logger,
            f"Novita host URLs can't represent the port entry '{raw}' "
            "(port range or malformed); skipping it.",
        )

    container_ports: list[int] = []
    for port in parsed:
        if port.protocol != "tcp":
            warn_once(
                logger,
                f"Novita host URLs are HTTP(S) only; skipping the "
                f"{port.protocol.upper()} port '{port.raw}'.",
            )
            continue
        if port.container_port not in container_ports:
            container_ports.append(port.container_port)
    return container_ports


def extract_x_novita(extensions: dict[str, Any] | None) -> dict[str, Any]:
    """Return the parsed ``x-novita`` extension block, normalized.

    Supported keys:

    - ``template`` (str): Pre-built template name; skips the build step.
    - ``timeout`` (number, seconds): Sandbox lifetime.
    - ``cpu_count`` (int), ``memory_mb`` (int): Override resources at template
      build time. Take precedence over ``deploy.resources`` and service-level
      ``cpus`` / ``mem_limit``.
    - ``envs`` (dict): Extra env vars; merged with ``service.environment`` (these win).
    - ``user`` (str): OS user; overrides ``service.user``.
    - ``allow_internet_access`` (bool): Allow outbound network access;
      overrides the ``network_mode`` translation. Ignored (with a warning)
      for DinD.
    - ``metadata`` (dict): Custom metadata; merged with the run's tracking
      labels (run-level wins for keys it owns: ``created_by``, ``inspect_run_id``,
      ``task``, ``name``).
    """
    return extract_extension(extensions, "x-novita")


def extract_novita_timeout(extensions: dict[str, Any] | None) -> float | None:
    """Return the ``x-novita.timeout`` value (seconds), or None if unset.

    Raises:
        ValueError: If ``x-novita.timeout`` is set to something that can't be
            coerced to ``float`` (e.g. a non-numeric string). YAML will parse
            ``timeout: "30"`` as the string ``"30"``; we coerce here so the
            SDK receives a proper number.
    """
    return extract_extension_timeout(extensions, "x-novita")


def _service_to_resources(
    service: ComposeService, extensions: dict[str, Any]
) -> tuple[int, int]:
    """Resolve template build resources from compose + ``x-novita`` overrides.

    Priority (each axis independently):
      1. ``x-novita.cpu_count`` / ``x-novita.memory_mb``
      2. ``deploy.resources.limits``
      3. ``deploy.resources.reservations``
      4. service-level ``cpus`` / ``mem_limit``
      5. defaults (2 vCPU / 1024 MiB)
    """
    cpu: int | None = None
    memory_mb: int | None = None

    if extensions.get("cpu_count") is not None:
        cpu = int(extensions["cpu_count"])
    if extensions.get("memory_mb") is not None:
        memory_mb = int(extensions["memory_mb"])

    # The x-novita override wins per axis; fall back to the shared ladder for any
    # axis it leaves unset, then to defaults.
    ladder_cpu, ladder_mib = resolve_service_resources(service)
    if cpu is None:
        cpu = ladder_cpu
    if memory_mb is None:
        memory_mb = ladder_mib

    return (cpu or DEFAULT_CPU_COUNT, memory_mb or DEFAULT_MEMORY_MB)
