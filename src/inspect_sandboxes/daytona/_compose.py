import math
from logging import getLogger
from pathlib import Path
from typing import Any

from daytona_sdk import (
    CreateSandboxFromImageParams,
    CreateSandboxFromSnapshotParams,
    Image,
    Resources,
)
from inspect_ai.util import ComposeConfig, ComposeService, warn_once

from inspect_sandboxes._util.compose import (
    find_default_service,
    parse_environment,
    parse_memory,
    parse_service_ports,
    resolve_dockerfile_path,
)

logger = getLogger(__name__)


def create_single_service_params(
    config: ComposeConfig,
    compose_path: str | None,
    labels: dict[str, str],
    name: str | None = None,
) -> CreateSandboxFromImageParams | CreateSandboxFromSnapshotParams:
    """Create Daytona sandbox params from a single-service compose config.

    Args:
        config: Parsed compose configuration.
        compose_path: Path to the compose file for resolving relative paths.
            Pass None when using a ComposeConfig object directly.
        labels: Labels to apply (merged with x-daytona labels).
        name: Optional sandbox name (visible in the Daytona dashboard).
    """
    _, service = find_default_service(config)

    compose_dir = Path(compose_path).parent if compose_path else Path.cwd()

    # Resolve image
    if service.build:
        dockerfile_path = resolve_dockerfile_path(service.build, compose_dir)
        if not dockerfile_path.exists():
            raise FileNotFoundError(f"Dockerfile not found: {dockerfile_path}")
        image: str | Image = Image.from_dockerfile(str(dockerfile_path))
    elif service.image:
        image = service.image
    else:
        raise ValueError("Compose service must specify either 'image' or 'build'")

    # Resources (from service, may be overridden by x-daytona.resources below)
    resources = _service_to_resources(service)

    # Sandbox-level params
    sandbox_params: dict[str, Any] = {}

    if service.environment:
        sandbox_params["env_vars"] = parse_environment(service.environment)

    if service.user:
        sandbox_params["os_user"] = service.user

    # Translate Docker network_mode to Daytona network_block_all.
    # Only set as a default; x-daytona extensions can override below.
    if service.network_mode is not None and "network_block_all" not in sandbox_params:
        sandbox_params["network_block_all"] = service.network_mode == "none"

    apply_daytona_extensions(sandbox_params, config.extensions)

    sandbox_params.setdefault("auto_stop_interval", 0)
    x_labels = sandbox_params.pop("labels", {})
    merged_labels = {**x_labels, **labels}

    # x-daytona.resources overrides service-level resources
    resources_override = sandbox_params.pop("resources", None)
    if resources_override:
        resources = Resources(
            cpu=resources_override.get("cpu"),
            memory=resources_override.get("memory"),
            gpu=resources_override.get("gpu"),
            disk=resources_override.get("disk"),
        )

    # x-daytona.snapshot: use pre-built snapshot instead of building from image
    snapshot = sandbox_params.pop("snapshot", None)
    if snapshot:
        return CreateSandboxFromSnapshotParams(
            snapshot=snapshot,
            name=name,
            labels=merged_labels,
            **sandbox_params,
        )

    return CreateSandboxFromImageParams(
        image=image,
        name=name,
        resources=resources,
        labels=merged_labels,
        **sandbox_params,
    )


def service_connection_ports(service: ComposeService) -> list[int]:
    """Return the container ports to surface through ``connection()``.

    Daytona has no creation-time port declaration: ``get_preview_link(port)``
    opens a closed port on demand and returns a URL. So we don't translate
    ports at creation; we record the container side of each ``service.ports``
    entry and let ``connection()`` turn them into preview URLs.

    ``expose`` is host-private and is warned about, never surfaced. Port ranges
    and UDP entries are warned about and skipped (a preview URL is a single
    HTTP(S) endpoint).
    """
    if service.expose:
        warn_once(
            logger,
            "Daytona does not surface Compose 'expose' ports. They stay "
            "host-private (reachable only by sibling services). Use 'ports' to "
            "get a preview URL through connection().",
        )

    if not service.ports:
        return []

    parsed, unparseable = parse_service_ports(service.ports)
    for raw in unparseable:
        warn_once(
            logger,
            f"Daytona preview URLs can't represent the port entry '{raw}' "
            "(port range or malformed); skipping it.",
        )

    container_ports: list[int] = []
    for port in parsed:
        if port.protocol != "tcp":
            warn_once(
                logger,
                f"Daytona preview URLs are HTTP(S) only; skipping the "
                f"{port.protocol.upper()} port '{port.raw}'.",
            )
            continue
        if port.container_port not in container_ports:
            container_ports.append(port.container_port)
    return container_ports


def aggregate_resources(config: ComposeConfig) -> Resources | None:
    """Sum per-service resources across all services + (Docker daemon) overhead."""
    total_cpu = 0
    total_memory = 0
    total_gpu = 0
    has_any = False

    for svc in config.services.values():
        r = _service_to_resources(svc)
        if r:
            has_any = True
            total_cpu += r.cpu or 0
            total_memory += r.memory or 0
            total_gpu += r.gpu or 0

    if not has_any:
        return None

    return Resources(
        cpu=total_cpu + 1,
        memory=total_memory + 1,
        gpu=total_gpu or None,
    )


def apply_daytona_extensions(
    params: dict[str, Any], extensions: dict[str, Any]
) -> None:
    """Apply Daytona-specific extensions from x-daytona compose key.

    Supported extensions:
        - auto_stop_interval (int): Minutes of inactivity before sandbox auto-stops.
            Default: 0 (disabled).
        - auto_archive_interval (int): Minutes before stopped sandbox auto-archives.
        - auto_delete_interval (int): Minutes before stopped sandbox auto-deletes.
        - network_block_all (bool): Block all network access.
        - network_allow_list (str): Comma-separated CIDR allowlist.
        - language (str): Programming language (e.g. "python", "typescript", "javascript").
        - os_user (str): OS user to run commands as. Overrides the service-level user field.
        - public (bool): Whether the sandbox should be publicly accessible.
        - ephemeral (bool): If True, sandbox is auto-deleted when stopped.
        - timeout (float): Seconds to wait for sandbox creation. NOT applied to
            the params dict by this function — it's a kwarg on
            ``AsyncDaytona.create()`` rather than a field on the sandbox params
            model. Callers should use :func:`extract_daytona_timeout` to
            retrieve it and forward to :func:`create_sandbox`.
        - env_vars (dict): Environment variables, merged with those from `environment:`.
            x-daytona values take precedence over service-level environment.
        - labels (dict): Custom labels. Merged by the caller with inspect's own labels,
            which take precedence.
        - snapshot (str): Pre-created Daytona snapshot name. For single-service,
            skips image building. For DinD, uses as the DinD VM snapshot.
        - resources (dict): Sandbox-level resource overrides (cpu, memory, gpu,
            disk). ``disk`` (GiB) can only be set here. For DinD, overrides
            the per-service aggregation.
        - volumes: Not yet supported.

    Args:
        params: Sandbox params dict to modify in-place.
        extensions: Extensions dict from compose config.
    """
    ext = extensions.get("x-daytona", {})

    simple_keys = [
        "auto_stop_interval",
        "auto_archive_interval",
        "auto_delete_interval",
        "network_block_all",
        "network_allow_list",
        "language",
        "os_user",
        "public",
        "ephemeral",
        "labels",
        "snapshot",
        "resources",
    ]

    for key in simple_keys:
        if ext.get(key) is not None:
            params[key] = ext[key]

    # env_vars: merge with service-level environment; x-daytona takes precedence
    if ext.get("env_vars") is not None:
        params["env_vars"] = {**params.get("env_vars", {}), **ext["env_vars"]}


def extract_daytona_timeout(extensions: dict[str, Any]) -> float | None:
    """Return the ``x-daytona.timeout`` value (seconds), or None if unset.

    Kept separate from :func:`apply_daytona_extensions` because ``timeout`` is
    a kwarg on ``AsyncDaytona.create()``, not a field on the sandbox params
    model — unpacking it alongside the other params would be silently
    dropped by Pydantic.

    Raises:
        ValueError: If ``x-daytona.timeout`` is set to something that can't be
            coerced to ``float`` (e.g. a non-numeric string). YAML will parse
            ``timeout: "30"`` as the string ``"30"``; we coerce here so the
            SDK receives a proper number.
    """
    raw = (extensions.get("x-daytona") or {}).get("timeout")
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError) as e:
        raise ValueError(
            f"x-daytona.timeout must be a number (seconds), got {raw!r}"
        ) from e


def _service_to_resources(service: ComposeService) -> Resources | None:
    """Extract resource configuration from a compose service.

    Returns:
        Resources for Daytona (cpu in cores, memory in GiB, gpu count), or None.

    Note:
        Daytona resources use GiB for memory (not MiB or bytes).
        CPU is an integer number of cores.
        Priority: deploy.resources > service-level fields.
    """
    cpu: int | None = None
    memory_gib: int | None = None
    gpu: int | None = None

    if service.deploy and service.deploy.resources:
        resources = service.deploy.resources

        if resources.limits and resources.limits.cpus:
            cpu = max(1, math.ceil(float(resources.limits.cpus)))
        elif resources.reservations and resources.reservations.cpus:
            cpu = max(1, math.ceil(float(resources.reservations.cpus)))

        if resources.limits and resources.limits.memory:
            memory_gib = _to_gib(resources.limits.memory)
        elif resources.reservations and resources.reservations.memory:
            memory_gib = _to_gib(resources.reservations.memory)

        # GPU count from compose deploy.resources.reservations.devices
        if resources.reservations and resources.reservations.devices:
            for device in resources.reservations.devices:
                if device.capabilities and "gpu" in device.capabilities:
                    if device.count:
                        gpu = int(device.count)
                    elif device.device_ids:
                        gpu = len(device.device_ids)
                    else:
                        gpu = 1
                    break

    # Fall back to service-level fields (v2 format)
    if cpu is None and service.cpus:
        cpu = max(1, math.ceil(service.cpus))

    if memory_gib is None and service.mem_limit:
        memory_gib = _to_gib(service.mem_limit)

    if cpu is None and memory_gib is None and gpu is None:
        return None

    return Resources(cpu=cpu, memory=memory_gib, gpu=gpu)


def _to_gib(mem_str: str) -> int:
    """Convert a memory string to GiB, ceiling-rounded, minimum 1."""
    return max(1, math.ceil(parse_memory(mem_str) / 1024))
