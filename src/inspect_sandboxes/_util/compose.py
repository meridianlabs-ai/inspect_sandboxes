import re
from dataclasses import dataclass
from pathlib import Path

from inspect_ai.util import ComposeBuild, ComposeConfig, ComposeService


def parse_environment(
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


def parse_memory(mem_limit: str) -> int:
    """Convert a memory string to MiB.

    Supports formats: "512m", "1g", "1.5gb", "1024k"
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

    # Convert to MiB
    # k=kibibytes, m=mebibytes, g=gibibytes, t=tebibytes
    multipliers = {"": 1, "k": 1 / 1024, "m": 1, "g": 1024, "t": 1024 * 1024}
    result = int(value * multipliers[unit])

    if value <= 0:
        raise ValueError(f"Memory must be positive, got: {mem_limit}")

    # A positive but sub-MiB request (e.g. "256k") floors to 0 MiB; clamp it to a
    # 1 MiB minimum rather than rejecting it (mirrors daytona/_compose.py:_to_gib).
    return max(1, result)


def resolve_dockerfile_path(build: str | ComposeBuild, compose_dir: Path) -> Path:
    if isinstance(build, str):
        return compose_dir / build / "Dockerfile"
    else:
        context = build.context or "."
        dockerfile = build.dockerfile or "Dockerfile"
        return compose_dir / context / dockerfile


def find_default_service(config: ComposeConfig) -> tuple[str, ComposeService]:
    """Find the default service in a compose config.

    Priority: x-default: true -> service named "default" or "main" -> first service.

    Returns:
        Tuple of (service_name, service_config).
    """
    for name, svc in config.services.items():
        if svc.x_default:
            return name, svc
    for candidate in ("default", "main"):
        if candidate in config.services:
            return candidate, config.services[candidate]
    name = next(iter(config.services))
    return name, config.services[name]


@dataclass(frozen=True)
class ParsedPort:
    """A single Compose ``ports`` entry parsed from short syntax.

    Attributes:
        container_port: The container-side port the service listens on.
        host_port: The requested host-side port, or None when the entry only
            names a container port (Docker would auto-assign a host port).
        protocol: "tcp" or "udp" (defaults to "tcp" when no suffix is given).
        raw: The original entry, kept for warning messages.
    """

    container_port: int
    host_port: int | None
    protocol: str
    raw: str


def _strip_protocol(token: str) -> tuple[str, str]:
    """Split a "port/proto" token into (port_part, protocol)."""
    if "/" in token:
        port_part, proto = token.rsplit("/", 1)
        return port_part, proto.lower()
    return token, "tcp"


def parse_service_ports(ports: list[str | int]) -> tuple[list[ParsedPort], list[str]]:
    """Parse Compose ``ports`` short syntax into structured entries.

    Handles the forms ``"3000"``, ``"8080:80"``, ``"127.0.0.1:8080:80"``,
    ``"53:53/udp"`` and bare integers. Port ranges (``"8000-8005:8000-8005"``)
    are not representable as a single mapping, so they are returned as
    unparseable rather than guessed at.

    Args:
        ports: The ``service.ports`` list.

    Returns:
        A tuple ``(parsed, unparseable)``. ``parsed`` holds the entries that
        map to a single container port; ``unparseable`` holds the raw strings
        of entries a caller should ``warn_once`` about and skip (port ranges or
        malformed values).
    """
    parsed: list[ParsedPort] = []
    unparseable: list[str] = []

    for entry in ports:
        raw = str(entry)
        # The host side may carry an IP ("127.0.0.1:8080:80"); the port pair is
        # always the last one or two colon-separated tokens.
        parts = raw.split(":")
        port_tokens = parts[-2:] if len(parts) >= 2 else parts

        if len(port_tokens) == 2:
            host_token, container_token = port_tokens
        else:
            host_token, container_token = None, port_tokens[0]

        container_str, protocol = _strip_protocol(container_token)
        host_str = None
        if host_token is not None:
            host_str, _ = _strip_protocol(host_token)

        # Ranges contain a hyphen; we can't collapse them to one port.
        if "-" in container_str or (host_str is not None and "-" in host_str):
            unparseable.append(raw)
            continue

        try:
            container_port = int(container_str)
            host_port = int(host_str) if host_str else None
        except ValueError:
            unparseable.append(raw)
            continue

        parsed.append(
            ParsedPort(
                container_port=container_port,
                host_port=host_port,
                protocol=protocol,
                raw=raw,
            )
        )

    return parsed, unparseable
