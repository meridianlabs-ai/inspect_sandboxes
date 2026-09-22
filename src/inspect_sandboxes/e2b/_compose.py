from __future__ import annotations

import math
from logging import getLogger
from pathlib import Path
from typing import Any, NamedTuple

from inspect_ai.util import ComposeConfig, ComposeService, warn_once

from inspect_sandboxes._util.compose import (
    find_default_service,
    parse_environment,
    parse_memory,
    parse_service_ports,
    resolve_dockerfile_path,
)
from inspect_sandboxes._util.compose_support import (
    ComposeSupport,
    ignored,
    partial,
    rejected,
    supported,
    validate_compose_support,
)

logger = getLogger(__name__)

DEFAULT_CPU_COUNT = 2
DEFAULT_MEMORY_MB = 1024

# Every x-e2b key this provider reads (see extract_x_e2b).
_E2B_EXTENSION_KEYS = frozenset(
    {
        "template",
        "timeout",
        "cpu_count",
        "memory_mb",
        "envs",
        "user",
        "metadata",
        "allow_internet_access",
    }
)

_NOT_EXECUTED = (
    "E2B sandboxes run envd; the Compose process is not started, so a "
    "keep-alive such as `sleep infinity` is unnecessary."
)

# How the single-service converter treats each Compose field. Validated (and
# rendered in docs/e2b.qmd) by validate_compose_support; see
# _util/compose_support.py. The DinD path runs Compose itself and is exempt.
E2B_COMPOSE_SUPPORT = ComposeSupport(
    provider="E2B",
    service={
        "image": supported(
            "A template is built from the image; ignored when `x-e2b.template` is set."
        ),
        "build": partial(
            "`context` and `dockerfile` locate the Dockerfile, but the Dockerfile's own directory is the build context, so `COPY` sources differ from Compose when the Dockerfile is not directly inside `context`. Ignored when `x-e2b.template` is set."
        ),
        "command": ignored(_NOT_EXECUTED),
        "entrypoint": ignored(_NOT_EXECUTED),
        "working_dir": ignored(
            "Commands start in the sandbox user's home; pass `cwd=` to `exec()`."
        ),
        "environment": supported(
            "Set on the sandbox at creation; `x-e2b.envs` adds to (and overrides) these."
        ),
        "env_file": ignored(
            "Files are not read; put the variables under `environment` or `x-e2b.envs`."
        ),
        "user": partial(
            "Default OS user for `exec()`; `x-e2b.user` overrides, and file operations still run as root. Must be a username that exists in the image: E2B does not accept numeric uids, and a `user:group` value is used for its user part only."
        ),
        "healthcheck": ignored("The sandbox is ready once creation returns."),
        "ports": partial(
            "Container ports are surfaced as E2B host URLs through `connection()`. Host bindings are dropped; UDP entries and port ranges are dropped with a warning."
        ),
        "expose": ignored(
            "`expose` ports are host-private and have no E2B equivalent; use `ports` to get a host URL."
        ),
        "volumes": rejected(
            "Bind mounts and named volumes cannot be attached; ship files with `Sample.files` / `write_file()`."
        ),
        "devices": ignored("Host devices cannot be mapped into a E2B sandbox."),
        "networks": ignored("Single service; there is no network to join."),
        "network_mode": partial(
            "`none` creates the sandbox with `allow_internet_access=False`, denying all outbound traffic; unlike Docker, the sandbox's public URLs (used by `connection()`) stay reachable. Every other value keeps E2B's default internet access. `x-e2b.allow_internet_access` overrides."
        ),
        "hostname": ignored("The hostname is assigned by E2B."),
        "runtime": ignored("The runtime is chosen by E2B; there are no GPUs."),
        "init": ignored("E2B has no init-process option."),
        "privileged": ignored("E2B sandboxes always run unprivileged."),
        "shm_size": ignored("`/dev/shm` size is fixed by E2B."),
        "ulimits": ignored("Resource limits cannot be set at creation."),
        "depends_on": ignored("Single service; nothing to depend on."),
        "pull_policy": ignored("E2B pulls the image when it builds the template."),
        "platform": ignored(
            "E2B templates are `linux/amd64` only; there is no architecture selection."
        ),
        "extra_hosts": ignored("`/etc/hosts` entries cannot be added at creation."),
        "cap_add": ignored(
            "Capabilities cannot be granted beyond the sandbox defaults."
        ),
        "cap_drop": rejected(
            "Capabilities cannot be dropped, so the sandbox would run with more privilege than the file requests."
        ),
        "security_opt": rejected(
            "seccomp, AppArmor and no-new-privileges options cannot be applied, so the sandbox would be less confined than the file requests."
        ),
        "tmpfs": ignored("tmpfs mounts cannot be declared at creation."),
        "restart": ignored("Sandboxes are never restarted."),
        "stdin_open": ignored("`exec()` supplies stdin per command."),
        "tty": ignored("Commands run without a pseudo-TTY."),
        "deploy": partial(
            "`cpus` round up to whole cores without warning and `memory` is taken in MiB, both baked into the template; limits win over reservations; GPU `devices` are dropped with a warning. `x-e2b.cpu_count` / `memory_mb` override; all of it is ignored when `x-e2b.template` is set."
        ),
        "mem_limit": supported(
            "Baked into the template; ignored when `x-e2b.template` is set."
        ),
        "mem_reservation": ignored("E2B has no memory reservation."),
        "memswap_limit": ignored("Swap cannot be configured."),
        "cpus": partial(
            "Rounded up to whole cores and baked into the template; ignored when `x-e2b.template` is set."
        ),
        "x_default": supported("Selects the service to run."),
    },
    top_level={
        "volumes": ignored("Named volume definitions have no E2B mapping."),
        "networks": ignored("Network definitions have no E2B mapping."),
    },
    extension="x-e2b",
    extension_keys=_E2B_EXTENSION_KEYS,
)


class E2BSingleServiceParams(NamedTuple):
    """Resolved E2B params for a single-service compose config.

    Exactly one of ``template``, ``dockerfile_path``, ``image`` is set:

    - ``template``: pre-built template name, supplied via ``x-e2b.template``.
    - ``dockerfile_path``: build a template from this Dockerfile.
    - ``image``: build a template from this base image.

    The orchestrator decides which build helper to call.

    ``allow_internet_access`` is False when the service sets
    ``network_mode: none`` (or ``x-e2b.allow_internet_access: false``).
    """

    template: str | None
    dockerfile_path: str | None
    image: str | None
    cpu_count: int
    memory_mb: int
    envs: dict[str, str]
    user: str | None
    timeout: float | None
    metadata: dict[str, str]
    allow_internet_access: bool


def resolve_single_service_params(
    config: ComposeConfig,
    compose_path: str | None,
) -> E2BSingleServiceParams:
    """Map a single-service compose config to E2B params.

    Args:
        config: Parsed compose configuration.
        compose_path: Path to the compose file for resolving relative
            Dockerfile paths. Pass ``None`` for an in-memory ``ComposeConfig``.

    Raises:
        ValueError: If the config uses a Compose field E2B cannot honor (see
            ``E2B_COMPOSE_SUPPORT``).
    """
    validate_compose_support(config, E2B_COMPOSE_SUPPORT)
    _, service = find_default_service(config)
    extensions = extract_x_e2b(config.extensions)
    compose_dir = Path(compose_path).parent if compose_path else Path.cwd()

    template: str | None = extensions.get("template")
    dockerfile_path: Path | None = None
    image: str | None = None
    if template is not None:
        pass
    elif service.build is not None:
        dockerfile_path = resolve_dockerfile_path(service.build, compose_dir)
        if not dockerfile_path.exists():
            raise FileNotFoundError(f"Dockerfile not found: {dockerfile_path}")
    elif service.image:
        image = service.image
    else:
        raise ValueError(
            "Compose service must specify 'image', 'build', or x-e2b.template"
        )

    cpu_count, memory_mb = _service_to_resources(service, extensions)

    envs: dict[str, str] = {}
    if service.environment:
        envs.update(parse_environment(service.environment))
    ext_envs = extensions.get("envs")
    if isinstance(ext_envs, dict):
        envs.update({str(k): str(v) for k, v in ext_envs.items()})

    user = extensions.get("user") or service.user
    if user is not None:
        # envd authenticates commands by username; a uid fails at the first
        # exec() with an opaque AuthenticationException, so fail here instead.
        name = user.split(":", 1)[0]
        if name.isdigit():
            raise ValueError(
                f"E2B needs a username for `user`, not a uid (got {user!r}); "
                "use the account name that exists in the image."
            )

    timeout = extract_e2b_timeout(config.extensions)

    # Docker's `network_mode: none` is E2B's allow_internet_access=False
    # (deny all outbound traffic); every other mode keeps E2B's default.
    # x-e2b.allow_internet_access overrides either way, like Modal's
    # x-modal.block_network and Daytona's x-daytona.network_block_all.
    allow_internet_access = service.network_mode != "none"
    if extensions.get("allow_internet_access") is not None:
        allow_internet_access = bool(extensions["allow_internet_access"])

    metadata_raw = extensions.get("metadata")
    metadata: dict[str, str] = {}
    if isinstance(metadata_raw, dict):
        metadata = {str(k): str(v) for k, v in metadata_raw.items()}

    return E2BSingleServiceParams(
        template=template,
        dockerfile_path=str(dockerfile_path) if dockerfile_path else None,
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

    E2B mirrors Daytona's runtime model: ``get_host(port)`` returns a reachable
    host for any container port with no creation-time declaration. So we don't
    translate ports at creation; we record the container side of each
    ``service.ports`` entry for ``connection()`` to turn into ``get_host`` URLs.

    ``expose`` is host-private and never surfaced; validate_compose_support
    warns about it (see E2B_COMPOSE_SUPPORT). Port ranges and UDP entries are
    warned about and skipped.
    """
    if not service.ports:
        return []

    parsed, unparseable = parse_service_ports(service.ports)
    for raw in unparseable:
        warn_once(
            logger,
            f"E2B host URLs can't represent the port entry '{raw}' "
            "(port range or malformed); skipping it.",
        )

    container_ports: list[int] = []
    for port in parsed:
        if port.protocol != "tcp":
            warn_once(
                logger,
                f"E2B host URLs are HTTP(S) only; skipping the "
                f"{port.protocol.upper()} port '{port.raw}'.",
            )
            continue
        if port.container_port not in container_ports:
            container_ports.append(port.container_port)
    return container_ports


def extract_x_e2b(extensions: dict[str, Any] | None) -> dict[str, Any]:
    """Return the parsed ``x-e2b`` extension block, normalized.

    Supported keys:

    - ``template`` (str): Pre-built template name; skips the build step.
    - ``timeout`` (number, seconds): Sandbox lifetime.
    - ``cpu_count`` (int), ``memory_mb`` (int): Override resources at template
      build time. Take precedence over ``deploy.resources`` and service-level
      ``cpus`` / ``mem_limit``.
    - ``envs`` (dict): Extra env vars; merged with ``service.environment`` (these win).
    - ``user`` (str): OS user; overrides ``service.user``.
    - ``allow_internet_access`` (bool): Overrides the network access derived
      from ``network_mode`` (``none`` blocks outbound traffic).
    - ``metadata`` (dict): Custom metadata; merged with the run's tracking
      labels (run-level wins for keys it owns: ``created_by``, ``inspect_run_id``,
      ``task``, ``name``).
    """
    if not extensions:
        return {}
    raw = extensions.get("x-e2b")
    if not isinstance(raw, dict):
        return {}
    return raw


def extract_e2b_timeout(extensions: dict[str, Any] | None) -> float | None:
    """Return the ``x-e2b.timeout`` value (seconds), or None if unset.

    Raises:
        ValueError: If ``x-e2b.timeout`` is set to something that can't be
            coerced to ``float`` (e.g. a non-numeric string). YAML will parse
            ``timeout: "30"`` as the string ``"30"``; we coerce here so the
            SDK receives a proper number.
    """
    raw = extract_x_e2b(extensions).get("timeout")
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError) as e:
        raise ValueError(
            f"x-e2b.timeout must be a number (seconds), got {raw!r}"
        ) from e


def _service_to_resources(
    service: ComposeService, extensions: dict[str, Any]
) -> tuple[int, int]:
    """Resolve template build resources from compose + ``x-e2b`` overrides.

    Priority (each axis independently):
      1. ``x-e2b.cpu_count`` / ``x-e2b.memory_mb``
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

    if service.deploy and service.deploy.resources:
        reservations = service.deploy.resources.reservations
        devices = (reservations.devices if reservations else None) or []
        if any(d.capabilities and "gpu" in d.capabilities for d in devices):
            warn_once(
                logger,
                "E2B has no GPU allocation; ignoring the GPU reservation in "
                "deploy.resources.reservations.devices.",
            )

    if (
        (cpu is None or memory_mb is None)
        and service.deploy
        and service.deploy.resources
    ):
        resources = service.deploy.resources
        if cpu is None:
            if resources.limits and resources.limits.cpus:
                cpu = max(1, math.ceil(float(resources.limits.cpus)))
            elif resources.reservations and resources.reservations.cpus:
                cpu = max(1, math.ceil(float(resources.reservations.cpus)))
        if memory_mb is None:
            if resources.limits and resources.limits.memory:
                memory_mb = parse_memory(resources.limits.memory)
            elif resources.reservations and resources.reservations.memory:
                memory_mb = parse_memory(resources.reservations.memory)

    if cpu is None and service.cpus:
        cpu = max(1, math.ceil(service.cpus))
    if memory_mb is None and service.mem_limit:
        memory_mb = parse_memory(service.mem_limit)

    return (cpu or DEFAULT_CPU_COUNT, memory_mb or DEFAULT_MEMORY_MB)
