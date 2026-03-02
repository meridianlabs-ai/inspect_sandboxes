import re
from pathlib import Path

from inspect_ai.util import ComposeBuild


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

    if result <= 0:
        raise ValueError(f"Memory must be positive, got: {mem_limit}")

    return result


def resolve_dockerfile_path(build: str | ComposeBuild, compose_dir: Path) -> Path:
    if isinstance(build, str):
        return compose_dir / build / "Dockerfile"
    else:
        context = build.context or "."
        dockerfile = build.dockerfile or "Dockerfile"
        return compose_dir / context / dockerfile
