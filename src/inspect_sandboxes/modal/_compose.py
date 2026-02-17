import re
from pathlib import Path
from typing import Any

import modal
from inspect_ai.util import ComposeBuild, ComposeConfig, ComposeService


def convert_compose_to_modal_params(
    config: ComposeConfig, compose_path: str | None
) -> dict[str, Any]:
    """Convert a ComposeConfig to Modal Sandbox.create() parameters.

    Args:
        config: Parsed compose configuration.
        compose_path: Path to the compose file (for resolving relative paths).
            None if config is a ComposeConfig object without an associated file.

    Returns:
        Dictionary of parameters for Modal Sandbox.create().
    """
    # Select service (prefer x-default, then "default", then first)
    service = next((svc for svc in config.services.values() if svc.x_default), None)
    if service is None:
        service = config.services.get("default") or next(iter(config.services.values()))

    params: dict[str, Any] = {}
    compose_dir = Path(compose_path).parent if compose_path else Path.cwd()

    if service.build:
        dockerfile_path = _resolve_dockerfile_path(service.build, compose_dir)
        if not dockerfile_path.exists():
            raise FileNotFoundError(f"Dockerfile not found: {dockerfile_path}")
        params["image"] = modal.Image.from_dockerfile(str(dockerfile_path))
    elif service.image:
        params["image"] = modal.Image.from_registry(service.image)

    if service.working_dir:
        params["workdir"] = service.working_dir

    if service.environment:
        params["env"] = _parse_environment(service.environment)

    memory = _service_to_memory(service)
    if memory is not None:
        params["memory"] = memory

    cpu = _service_to_cpu(service)
    if cpu is not None:
        params["cpu"] = cpu

    gpu = _service_to_gpu(service)
    if gpu is not None:
        params["gpu"] = gpu

    _apply_modal_extensions(params, config.extensions)

    return params


def _apply_modal_extensions(params: dict[str, Any], extensions: dict[str, Any]) -> None:
    """Apply Modal-specific extensions to params dict.

    Supported extensions:
        - gpu (str): GPU type ("A10G", "A100", "T4", "ANY", "A10G:2", etc.)
                     Overrides GPU config from compose deploy.resources.
        - block_network (bool): Block all network access
        - cidr_allowlist (list[str]): Allowed CIDR blocks for network access
        - timeout (int): Sandbox timeout in seconds
        - cloud (str): Cloud provider ("aws", "gcp", "oci", "auto")
        - region (str | list[str]): Cloud region(s) to use
        - idle_timeout (int): Idle timeout in seconds
        - pty (bool): Enable pseudo-TTY for interactive sessions
        - encrypted_ports (list[int]): HTTPS ports for web services
        - h2_ports (list[int]): HTTP/2 ports
        - unencrypted_ports (list[int]): HTTP ports
        - custom_domain (str): Custom domain for web services
        - verbose (bool): Enable verbose logging

    Unsupported Modal parameters:
        - secrets: Requires modal.Secret objects
        - network_file_systems: Requires modal.NetworkFileSystem objects
        - volumes: Requires modal.Volume or modal.CloudBucketMount objects
        - proxy: Requires modal.Proxy object

    Args:
        params: Parameters dict to modify.
        extensions: Extensions dict from compose config.
    """
    modal_extensions = extensions.get("x-inspect_modal_sandbox", {})

    extension_keys = [
        "gpu",
        "block_network",
        "cidr_allowlist",
        "timeout",
        "cloud",
        "region",
        "idle_timeout",
        "pty",
        "encrypted_ports",
        "h2_ports",
        "unencrypted_ports",
        "custom_domain",
        "verbose",
    ]

    for key in extension_keys:
        if modal_extensions.get(key) is not None:
            params[key] = modal_extensions[key]


def _service_to_cpu(service: ComposeService) -> float | tuple[float, float] | None:
    """Extract CPU configuration from compose service.

    Args:
        service: Compose service configuration.

    Returns:
        CPU specification for Modal, or None if no CPU config.
        - float: Single CPU limit (e.g., 2.0)
        - tuple[float, float]: (reservation, limit) for both soft and hard limits

    Note:
        Priority: deploy.resources.{reservations,limits}.cpus > service.cpus
    """
    cpu_reservation = None
    cpu_limit = None

    # Check deploy.resources (v3 format) first
    if service.deploy and service.deploy.resources:
        resources = service.deploy.resources

        if resources.reservations and resources.reservations.cpus:
            cpu_reservation = float(resources.reservations.cpus)

        if resources.limits and resources.limits.cpus:
            cpu_limit = float(resources.limits.cpus)

    # Fall back to service-level field (v2 format)
    if cpu_limit is None and service.cpus:
        cpu_limit = service.cpus

    # Return tuple if both, single value if only one
    if cpu_reservation and cpu_limit:
        return (cpu_reservation, cpu_limit)
    elif cpu_limit:
        return cpu_limit
    elif cpu_reservation:
        return cpu_reservation
    return None


def _service_to_memory(service: ComposeService) -> int | tuple[int, int] | None:
    """Extract memory configuration from compose service.

    Args:
        service: Compose service configuration.

    Returns:
        Memory specification in MiB for Modal, or None if no memory config.
        - int: Single memory limit in MiB (e.g., 1024)
        - tuple[int, int]: (reservation, limit) in MiB for both soft and hard limits

    Note:
        Priority: deploy.resources.{reservations,limits}.memory > service.mem_limit
    """
    mem_reservation = None
    mem_limit = None

    # Check deploy.resources (v3 format) first
    if service.deploy and service.deploy.resources:
        resources = service.deploy.resources

        if resources.reservations and resources.reservations.memory:
            try:
                mem_reservation = _convert_byte_value(resources.reservations.memory)
            except ValueError as e:
                raise ValueError(
                    f"Invalid memory reservation in deploy.resources: {e}"
                ) from e

        if resources.limits and resources.limits.memory:
            try:
                mem_limit = _convert_byte_value(resources.limits.memory)
            except ValueError as e:
                raise ValueError(
                    f"Invalid memory limit in deploy.resources: {e}"
                ) from e

    # Fall back to service-level field (v2 format)
    if mem_limit is None and service.mem_limit:
        try:
            mem_limit = _convert_byte_value(service.mem_limit)
        except ValueError as e:
            raise ValueError(f"Invalid mem_limit in service: {e}") from e

    # Return tuple if both, single value if only one
    if mem_reservation and mem_limit:
        return (mem_reservation, mem_limit)
    elif mem_limit:
        return mem_limit
    elif mem_reservation:
        return mem_reservation
    return None


def _service_to_gpu(service: ComposeService) -> str | None:
    """Extract GPU configuration from compose service.

    Args:
        service: Compose service configuration.

    Returns:
        GPU specification string for Modal, or None if no GPU requested.
        - "ANY:<count>": Any GPU with specified count (e.g., "ANY:2")
        - "ANY": Any single GPU

    Note:
        Compose GPU config doesn't specify GPU types (A10G, T4, etc.), so we
        default to "ANY". Use x-inspect_modal_sandbox.gpu extension to specify
        a particular GPU type, which will override this value.
    """
    if not service.deploy or not service.deploy.resources:
        return None

    reservations = service.deploy.resources.reservations
    if not reservations or not reservations.devices:
        return None

    gpu_device = None
    for device in reservations.devices:
        if device.capabilities and "gpu" in device.capabilities:
            gpu_device = device
            break

    if not gpu_device:
        return None

    if gpu_device.count:
        return f"ANY:{gpu_device.count}"
    if gpu_device.device_ids:
        # Modal doesn't support specific device IDs in cloud environments
        # Convert to count based on number of device IDs specified
        return f"ANY:{len(gpu_device.device_ids)}"
    return "ANY"


def _resolve_dockerfile_path(build: str | ComposeBuild, compose_dir: Path) -> Path:
    """Resolve Dockerfile path from build configuration.

    Args:
        build: Build configuration (string or ComposeBuild object).
        compose_dir: Directory containing the compose file.

    Returns:
        Path to the Dockerfile.
    """
    if isinstance(build, str):
        return compose_dir / build / "Dockerfile"
    else:
        context = build.context or "."
        dockerfile = build.dockerfile or "Dockerfile"
        return compose_dir / context / dockerfile


def _parse_environment(
    environment: list[str] | dict[str, str | None],
) -> dict[str, str]:
    """Parse environment variables from list or dict format.

    Args:
        environment: Environment variables as list of "KEY=VALUE" strings
            or dict mapping keys to values.

    Returns:
        Dictionary of environment variables (excluding None values).
    """
    if isinstance(environment, list):
        env_dict = {}
        for item in environment:
            if "=" in item:
                key, value = item.split("=", 1)
                env_dict[key] = value
        return env_dict
    else:
        return {k: v for k, v in environment.items() if v is not None}


def _convert_byte_value(mem_limit: str) -> int:
    """Convert memory limit string to MiB.

    Supports formats like: "512m", "1g", "1.5gb", "1024k"

    Args:
        mem_limit: Memory limit string (e.g., "512m", "1g").

    Returns:
        Memory in MiB (mebibytes).
    """
    mem_limit = mem_limit.lower().strip()
    match = re.match(r"^(\d+(?:\.\d+)?)\s*([kmgt]?)b?$", mem_limit)

    if not match:
        raise ValueError(
            f"Invalid memory format: '{mem_limit}'. "
            "Expected format: <number>[k|m|g|t][b] (e.g., '512m', '1g', '1.5gb')"
        )

    value = float(match.group(1))
    unit = match.group(2)

    # Convert to MiB (Modal's expected unit)
    # k=kibibytes, m=mebibytes, g=gibibytes, t=tebibytes
    multipliers = {"": 1, "k": 1 / 1024, "m": 1, "g": 1024, "t": 1024 * 1024}
    result = int(value * multipliers[unit])

    if result <= 0:
        raise ValueError(f"Memory limit must be positive, got: {mem_limit}")

    return result
