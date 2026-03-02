import math
from pathlib import Path
from typing import Any

from daytona_sdk import Image, Resources
from inspect_ai.util import ComposeConfig, ComposeService

from inspect_sandboxes._util.compose import (
    parse_environment,
    parse_memory,
    resolve_dockerfile_path,
)


def convert_compose_to_daytona_params(
    config: ComposeConfig, compose_path: str | None
) -> tuple[str | Image, Resources | None, dict[str, Any]]:
    """Convert a ComposeConfig to Daytona sandbox creation parameters.

    Args:
        config: Parsed compose configuration.
        compose_path: Path to the compose file for resolving relative paths.
            Pass None when using a ComposeConfig object directly.

    Returns:
        Tuple of (image, resources, sandbox_params) where:
            - image: Docker image name or Image object
            - resources: Resources config, or None if not specified
            - sandbox_params: Extra sandbox params (env_vars, network settings, etc.)
    """
    service = next((svc for svc in config.services.values() if svc.x_default), None)
    if service is None:
        service = config.services.get("default") or next(iter(config.services.values()))

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

    # Resources
    resources = _service_to_resources(service)

    # Sandbox-level params
    sandbox_params: dict[str, Any] = {}

    if service.environment:
        sandbox_params["env_vars"] = parse_environment(service.environment)

    if service.user:
        sandbox_params["os_user"] = service.user

    _apply_daytona_extensions(sandbox_params, config.extensions)

    return image, resources, sandbox_params


def _apply_daytona_extensions(
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
        - timeout (float): Seconds to wait for sandbox creation.
        - env_vars (dict): Environment variables, merged with those from `environment:`.
            x-daytona values take precedence over service-level environment.
        - labels (dict): Custom labels. Merged by the caller with inspect's own labels,
            which take precedence.
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
        "timeout",
        "labels",
    ]

    for key in simple_keys:
        if ext.get(key) is not None:
            params[key] = ext[key]

    # env_vars: merge with service-level environment; x-daytona takes precedence
    if ext.get("env_vars") is not None:
        params["env_vars"] = {**params.get("env_vars", {}), **ext["env_vars"]}


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
