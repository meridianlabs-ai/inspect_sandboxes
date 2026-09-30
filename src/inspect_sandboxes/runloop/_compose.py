from __future__ import annotations

from pathlib import Path
from typing import Any, NamedTuple

from inspect_ai.util import ComposeConfig, ComposeService
from runloop_api_client.types.shared_params import LaunchParameters

from inspect_sandboxes._util.compose import (
    extract_extension,
    extract_extension_timeout,
    find_default_service,
    mib_to_gib,
    parse_environment,
    resolve_dockerfile_path,
    resolve_service_resources,
)


class RunloopSingleServiceParams(NamedTuple):
    """Resolved Runloop params for a single-service compose config.

    Exactly one of ``blueprint_id``, ``blueprint_name``, ``snapshot_id``,
    ``dockerfile_path``, or ``image`` is set:

    - ``blueprint_id`` / ``blueprint_name`` / ``snapshot_id``: pre-built
      blueprint/snapshot references, supplied via ``x-runloop``. Skip the
      build step.
    - ``dockerfile_path``: build a blueprint from this Dockerfile.
    - ``image``: build a thin blueprint from this base image.

    The orchestrator decides which build helper to call.
    """

    blueprint_id: str | None
    blueprint_name: str | None
    snapshot_id: str | None
    dockerfile_path: str | None
    image: str | None
    launch_parameters: LaunchParameters | None
    environment_variables: dict[str, str]
    metadata: dict[str, str]
    timeout: float | None


def resolve_single_service_params(
    config: ComposeConfig,
    compose_path: str | None,
) -> RunloopSingleServiceParams:
    """Map a single-service compose config to Runloop params.

    Args:
        config: Parsed compose configuration.
        compose_path: Path to the compose file for resolving relative
            Dockerfile paths. Pass ``None`` for an in-memory ``ComposeConfig``.
    """
    _, service = find_default_service(config)
    compose_dir = Path(compose_path).parent if compose_path else Path.cwd()
    extensions = extract_x_runloop(config.extensions)

    blueprint_id: str | None = extensions.get("blueprint_id")
    blueprint_name: str | None = extensions.get("blueprint_name")
    snapshot_id: str | None = extensions.get("snapshot_id")
    dockerfile_path: Path | None = None
    image: str | None = None

    if (
        blueprint_id is not None
        or blueprint_name is not None
        or snapshot_id is not None
    ):
        pass
    elif service.build is not None:
        dockerfile_path = resolve_dockerfile_path(service.build, compose_dir)
        if not dockerfile_path.exists():
            raise FileNotFoundError(f"Dockerfile not found: {dockerfile_path}")
    elif service.image:
        image = service.image
    else:
        raise ValueError(
            "Compose service must specify 'image', 'build', or "
            "x-runloop.{blueprint_id,blueprint_name,snapshot_id}."
        )

    launch_parameters = _service_to_launch_parameters(service, extensions)

    environment_variables: dict[str, str] = {}
    if service.environment:
        environment_variables.update(parse_environment(service.environment))
    ext_env = extensions.get("environment_variables")
    if isinstance(ext_env, dict):
        environment_variables.update({str(k): str(v) for k, v in ext_env.items()})

    timeout = extract_runloop_timeout(config.extensions)

    metadata_raw = extensions.get("metadata")
    metadata: dict[str, str] = {}
    if isinstance(metadata_raw, dict):
        metadata = {str(k): str(v) for k, v in metadata_raw.items()}

    return RunloopSingleServiceParams(
        blueprint_id=blueprint_id,
        blueprint_name=blueprint_name,
        snapshot_id=snapshot_id,
        dockerfile_path=str(dockerfile_path) if dockerfile_path else None,
        image=image,
        launch_parameters=launch_parameters,
        environment_variables=environment_variables,
        metadata=metadata,
        timeout=timeout,
    )


def extract_x_runloop(extensions: dict[str, Any] | None) -> dict[str, Any]:
    """Return the parsed ``x-runloop`` extension block, normalized.

    Supported keys:

    - ``blueprint_id`` / ``blueprint_name`` / ``snapshot_id`` (str): Pre-built
      blueprint or snapshot references; skip the build step.
    - ``timeout`` (number, seconds): Per-request HTTP timeout forwarded to
      ``create_and_await_running``.
    - ``launch_parameters`` (dict): Merged into the resolved ``LaunchParameters``.
      Takes precedence over ``deploy.resources`` and service-level
      ``cpus`` / ``mem_limit``. Set ``keep_alive_time_seconds`` here to control
      the devbox lifetime (Runloop's own default is 1 hour).
    - ``environment_variables`` (dict): Extra env vars; merged with
      ``service.environment`` (these win).
    - ``metadata`` (dict): Custom metadata; merged with the run's tracking
      labels (run-level wins for keys it owns: ``created_by``, ``inspect_run_id``,
      ``task``, ``name``).
    """
    return extract_extension(extensions, "x-runloop")


def extract_runloop_timeout(extensions: dict[str, Any] | None) -> float | None:
    """Return the ``x-runloop.timeout`` value (seconds), or None if unset.

    Raises:
        ValueError: If ``x-runloop.timeout`` is set to something that can't be
            coerced to ``float`` (e.g. a non-numeric string). YAML will parse
            ``timeout: "30"`` as the string ``"30"``; we coerce here so the
            SDK receives a proper number.
    """
    return extract_extension_timeout(extensions, "x-runloop")


def _service_to_launch_parameters(
    service: ComposeService, extensions: dict[str, Any]
) -> LaunchParameters | None:
    """Resolve devbox launch parameters from compose + ``x-runloop`` overrides.

    Priority (each axis independently):
      1. ``x-runloop.launch_parameters``
      2. ``deploy.resources.limits``
      3. ``deploy.resources.reservations``
      4. service-level ``cpus`` / ``mem_limit``
    """
    ext_launch: dict[str, Any] = dict(extensions.get("launch_parameters") or {})
    cpu: int | None = ext_launch.get("custom_cpu_cores")
    memory_gb: int | None = ext_launch.get("custom_gb_memory")

    # The x-runloop override wins per axis; fall back to the shared ladder
    # (MiB) for any axis it leaves unset, converting memory to GiB.
    ladder_cpu, ladder_mib = resolve_service_resources(service)
    if cpu is None:
        cpu = ladder_cpu
    if memory_gb is None and ladder_mib is not None:
        memory_gb = mib_to_gib(ladder_mib)

    params: dict[str, Any] = dict(ext_launch)
    if cpu is not None:
        params["custom_cpu_cores"] = cpu
    if memory_gb is not None:
        params["custom_gb_memory"] = memory_gb

    return normalize_launch_parameters(params)


def normalize_launch_parameters(
    launch_parameters: dict[str, Any] | None,
) -> LaunchParameters | None:
    """Normalize raw launch parameters for the Runloop API.

    Runloop's API rejects ``custom_cpu_cores`` / ``custom_gb_memory`` /
    ``custom_disk_size`` unless ``resource_size_request`` is ``"CUSTOM_SIZE"``,
    so we default it whenever any custom sizing field is present. Returns None
    for empty or absent parameters.
    """
    params = dict(launch_parameters or {})
    if not params:
        return None
    if any(
        k in params
        for k in ("custom_cpu_cores", "custom_gb_memory", "custom_disk_size")
    ):
        params.setdefault("resource_size_request", "CUSTOM_SIZE")
    return params  # type: ignore[return-value]  # plain dict matches TypedDict shape
