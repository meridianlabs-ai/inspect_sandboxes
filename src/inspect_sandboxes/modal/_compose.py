import shlex
from dataclasses import dataclass, field
from logging import getLogger
from pathlib import Path
from typing import Any, NamedTuple

import modal
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

# x-modal keys that declare tunnels. If any is set, it overrides the ports
# translated from service.ports (these extensions are explicit overrides).
_MODAL_PORT_KEYS = ("encrypted_ports", "h2_ports", "unencrypted_ports")

# Every x-modal key _apply_modal_extensions reads, plus image_registry_secret
# (consumed by convert_compose_to_modal_params).
_MODAL_EXTENSION_KEYS = frozenset(
    {
        "block_network",
        "cidr_allowlist",
        "cloud",
        "custom_domain",
        "encrypted_ports",
        "experimental_options",
        "gpu",
        "h2_ports",
        "idle_timeout",
        "image_registry_secret",
        "pty",
        "region",
        "secrets",
        "timeout",
        "unencrypted_ports",
        "verbose",
        "volumes",
    }
)

# The x-modal keys copied straight onto Sandbox.create() kwargs (everything but
# image_registry_secret, which shapes the image pull instead).
_MODAL_PARAM_KEYS = frozenset(_MODAL_EXTENSION_KEYS - {"image_registry_secret"})

# How this converter treats each Compose field. Validated (and rendered in
# docs/modal.qmd) by validate_compose_support; see _util/compose_support.py.
MODAL_COMPOSE_SUPPORT = ComposeSupport(
    provider="Modal",
    service={
        "image": supported(),
        "build": partial(
            "`context` and `dockerfile` locate the Dockerfile, but the Dockerfile's own directory is the build context, so `COPY` sources differ from Compose when the Dockerfile is not directly inside `context`."
        ),
        "command": supported(),
        "entrypoint": supported(
            "Replaces the image ENTRYPOINT and suppresses its CMD."
        ),
        "working_dir": supported(),
        "environment": supported(),
        "env_file": ignored(
            "Files are not read; put the variables under `environment` or in a Modal secret (`x-modal.secrets`)."
        ),
        "user": rejected(
            "Modal sandboxes have no per-service user, so the sandbox would run as a different identity; pass `user=` to `exec()` instead."
        ),
        "healthcheck": ignored("The sandbox is ready once creation returns."),
        "ports": partial(
            "Container ports become unencrypted TCP tunnels surfaced through `connection()`. Host bindings, UDP entries and port ranges are dropped with a warning. `x-modal.*_ports` overrides."
        ),
        "expose": ignored(
            "`expose` ports are host-private and have no Modal equivalent; use `ports` or `x-modal.*_ports` to publish a port."
        ),
        "volumes": rejected(
            "Bind mounts and named volumes cannot be attached; mount a Modal Volume with `x-modal.volumes`, or ship files with `Sample.files` / `write_file()`."
        ),
        "devices": ignored("Host devices cannot be mapped into a Modal sandbox."),
        "networks": ignored("Single service; there is no network to join."),
        "network_mode": partial(
            "`none` sets `block_network`; every other value allows network access. `x-modal.block_network` overrides."
        ),
        "hostname": ignored("The hostname is assigned by Modal."),
        "runtime": ignored(
            "The runtime is chosen by Modal; request GPUs via `deploy.resources` or `x-modal.gpu`."
        ),
        "init": ignored("Modal has no init-process option."),
        "privileged": ignored("Modal sandboxes always run unprivileged."),
        "shm_size": ignored("`/dev/shm` size is fixed by Modal."),
        "ulimits": ignored("Resource limits cannot be set at creation."),
        "depends_on": ignored("Single service; nothing to depend on."),
        "pull_policy": ignored("Modal pulls the image when it builds its image layer."),
        "platform": ignored(
            "Modal runs `linux/amd64` only; there is no architecture selection."
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
        "tty": ignored("Use `x-modal.pty` to request a pseudo-TTY."),
        "deploy": partial(
            "`resources.limits` / `reservations` `cpus` and `memory` map to Modal cpu and memory (reservation, limit). GPU `devices` map to a count of `ANY` GPUs (`x-modal.gpu` picks a type); `driver`, `options`, specific `device_ids` and non-GPU devices are dropped silently."
        ),
        "mem_limit": supported(),
        "mem_reservation": ignored(
            "Only `deploy.resources.reservations.memory` sets a reservation."
        ),
        "memswap_limit": ignored("Swap cannot be configured."),
        "cpus": supported(),
        "x_default": supported("Selects the service to run."),
    },
    top_level={
        "volumes": ignored(
            "Named volume definitions have no Modal mapping; see `x-modal.volumes`."
        ),
        "networks": ignored("Network definitions have no Modal mapping."),
    },
    extension="x-modal",
    extension_keys=_MODAL_EXTENSION_KEYS,
)


class ModalVolumeSpec(NamedTuple):
    """A named Modal Volume and its read-only sandbox mount."""

    name: str
    mount_path: str
    read_only: bool = False


@dataclass
class ModalSandboxParams:
    """Parameters for modal.Sandbox.create().

    Attributes:
        command: Positional command args passed before **kwargs to Sandbox.create().
        kwargs: Keyword arguments passed to Sandbox.create().
        volumes: Named Modal Volumes to attach when creating the sandbox.
    """

    command: list[str] = field(default_factory=list)
    kwargs: dict[str, Any] = field(default_factory=dict)
    volumes: list[ModalVolumeSpec] = field(default_factory=list)


def convert_compose_to_modal_params(
    config: ComposeConfig, compose_path: str | None
) -> ModalSandboxParams:
    """Convert a ComposeConfig to Modal Sandbox.create() parameters.

    Args:
        config: Parsed compose configuration.
        compose_path: Path to the compose file for resolving relative paths.
            Pass None when using a ComposeConfig object directly.
    """
    # Select service (prefer x-default, then "default"/"main", then first).
    # parse_compose_yaml(multiple_services=False) already rejects multi-service
    # files; an in-memory ComposeConfig can still carry several, so say which
    # ones are dropped rather than silently running one. Only the selected
    # service is validated: settings on the dropped ones never take effect.
    service_name, service = find_default_service(config)
    if len(config.services) > 1:
        others = sorted(name for name in config.services if name != service_name)
        warn_once(
            logger,
            f"Modal runs a single service; using '{service_name}' and ignoring "
            f"{others}. Mark the intended service with `x-default: true`.",
        )
    validate_compose_support(config, MODAL_COMPOSE_SUPPORT, services=[service_name])

    params: dict[str, Any] = {}
    command: list[str] = []
    compose_dir = Path(compose_path).parent if compose_path else Path.cwd()

    if service.build:
        if (
            config.extensions.get("x-modal", {}).get("image_registry_secret")
            is not None
        ):
            # Fail loudly rather than silently ignoring the key: it authenticates
            # registry PULLS for image:-based services only. A private base image in a
            # Dockerfile would otherwise fail later with Modal's opaque build error.
            raise ValueError(
                "x-modal.image_registry_secret is not supported with build:; it "
                "authenticates the registry pull for image:-based services. For a "
                "private base image in a Dockerfile, pull it via image: instead."
            )
        dockerfile_path = resolve_dockerfile_path(service.build, compose_dir)
        if not dockerfile_path.exists():
            raise FileNotFoundError(f"Dockerfile not found: {dockerfile_path}")
        context_dir = dockerfile_path.parent
        params["image"] = modal.Image.from_dockerfile(
            str(dockerfile_path), context_dir=str(context_dir)
        )
    elif service.image:
        # A private registry needs credentials at PULL time. Without them Modal fails
        # with an opaque `RemoteError('Image build for im-... failed')`, which reads like
        # a build problem rather than an auth problem. `x-modal.image_registry_secret`
        # names a Modal secret holding the registry credentials; it is distinct from
        # `x-modal.secrets`, which injects env vars into the running sandbox and has no
        # effect on the image pull.
        registry_secret = config.extensions.get("x-modal", {}).get(
            "image_registry_secret"
        )
        if registry_secret is not None:
            if not isinstance(registry_secret, str):
                raise TypeError(
                    "x-modal.image_registry_secret must be a Modal secret name (str), "
                    f"got {type(registry_secret).__name__}"
                )
            params["image"] = modal.Image.from_registry(
                service.image, secret=modal.Secret.from_name(registry_secret)
            )
        else:
            params["image"] = modal.Image.from_registry(service.image)

    if service.command:
        command = (
            service.command
            if isinstance(service.command, list)
            else shlex.split(service.command)
        )

    if service.entrypoint is not None:
        entrypoint = (
            service.entrypoint
            if isinstance(service.entrypoint, list)
            else shlex.split(service.entrypoint)
        )
        command = [*entrypoint, *command]
        if not command:
            raise ValueError(
                "Compose entrypoint is empty and no command is set, so the Modal "
                "sandbox would exit immediately. Set entrypoint or command."
            )
        # Modal prepends the image ENTRYPOINT to the Sandbox.create() args, so
        # clear ENTRYPOINT and CMD for the Compose entrypoint to run verbatim.
        image = params.get("image")
        if image is not None:
            params["image"] = image.entrypoint([]).cmd([])

    if service.working_dir:
        params["workdir"] = service.working_dir

    if service.environment:
        params["env"] = parse_environment(service.environment)

    memory = _service_to_memory(service)
    if memory is not None:
        params["memory"] = memory

    cpu = _service_to_cpu(service)
    if cpu is not None:
        params["cpu"] = cpu

    gpu = _service_to_gpu(service)
    if gpu is not None:
        params["gpu"] = gpu

    # Translate Docker network_mode to Modal block_network.
    # Only set as a default; x-modal extensions can override below.
    if service.network_mode is not None:
        params["block_network"] = service.network_mode == "none"

    # Translate service.ports into unencrypted_ports (raw TCP tunnels).
    # Set as a default; an explicit x-modal.*_ports overrides it below.
    _apply_service_ports(params, service)

    volumes = _apply_modal_extensions(params, config.extensions)

    return ModalSandboxParams(command=command, kwargs=params, volumes=volumes)


def _apply_service_ports(params: dict[str, Any], service: ComposeService) -> None:
    """Default Compose ``ports`` / ``expose`` into Modal tunnel params.

    Modal tunnels are declared at creation as a flat list of container ports.
    We take the container side of each ``service.ports`` entry and put it into
    ``unencrypted_ports`` (raw TCP, the only protocol-agnostic choice for a port
    that might be Postgres, ssh, or anything else, not just HTTP).

    Caveats, each surfaced via ``warn_once`` once per process:
      - A ``host:container`` mapping with a differing host port can't be
        honored; Modal assigns the tunnel URL and there is no host binding.
      - UDP entries and port ranges aren't representable as tunnels; skip them.

    ``expose`` is host-private and never translated; validate_compose_support
    warns about it (see MODAL_COMPOSE_SUPPORT).
    """
    if not service.ports:
        return

    parsed, unparseable = parse_service_ports(service.ports)

    container_ports: list[int] = []
    for port in parsed:
        if port.protocol != "tcp":
            warn_once(
                logger,
                f"Modal tunnels can't represent the {port.protocol.upper()} "
                f"port '{port.raw}'; skipping it.",
            )
            continue
        if port.host_port is not None and port.host_port != port.container_port:
            warn_once(
                logger,
                f"Modal can't honor the host port in '{port.raw}'; it assigns "
                f"the tunnel URL and has no host binding. Exposing container "
                f"port {port.container_port} as an unencrypted tunnel instead.",
            )
        if port.container_port not in container_ports:
            container_ports.append(port.container_port)

    for raw in unparseable:
        warn_once(
            logger,
            f"Modal tunnels can't represent the port entry '{raw}' "
            "(port range or malformed); skipping it.",
        )

    if container_ports:
        params["unencrypted_ports"] = container_ports


def _apply_modal_extensions(
    params: dict[str, Any], extensions: dict[str, Any]
) -> list[ModalVolumeSpec]:
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
        - experimental_options (dict[str, Any]): Experimental Modal Sandbox options
        - secrets (str | list[str]): Modal secret name(s) to attach
        - volumes (list[dict[str, str | bool]]): Named Modal Volumes to attach.
          Each entry requires `name` and `mount_path`; `read_only` defaults to
          `False`. Volumes are resolved at sandbox creation so missing volumes
          fail loudly.
        - image_registry_secret (str): Modal secret name holding registry credentials,
          used for the image PULL (see convert_compose_to_modal_params). Not applied here.

    Unsupported Modal parameters:
        - network_file_systems: Requires modal.NetworkFileSystem objects
        - proxy: Requires modal.Proxy object

    Args:
        params: Parameters dict to modify.
        extensions: Extensions dict from compose config.

    Returns:
        Volume specifications extracted from ``x-modal.volumes``. Other supported
        extension values are applied directly to ``params``.
    """
    modal_extensions = extensions.get("x-modal", {})
    volumes: list[ModalVolumeSpec] = []

    # An explicit x-modal port declaration overrides the ports translated from
    # service.ports. Any of the three port keys takes over the whole tunnel set,
    # so drop the translated default before applying the extension keys below.
    if any(modal_extensions.get(key) is not None for key in _MODAL_PORT_KEYS):
        params.pop("unencrypted_ports", None)

    for key in sorted(_MODAL_PARAM_KEYS):
        if modal_extensions.get(key) is not None:
            if key == "secrets":
                secrets = modal_extensions[key]
                if not isinstance(secrets, list):
                    secrets = [secrets]
                params[key] = [modal.Secret.from_name(s) for s in secrets]
            elif key == "experimental_options":
                options = modal_extensions[key]
                if not isinstance(options, dict):
                    raise TypeError(
                        "x-modal.experimental_options must be a mapping, got "
                        f"{type(options).__name__}"
                    )
                params[key] = options
            elif key == "volumes":
                volumes = [
                    _parse_modal_volume_spec(volume) for volume in modal_extensions[key]
                ]
            else:
                params[key] = modal_extensions[key]

    return volumes


_MODAL_VOLUME_KEYS = frozenset(ModalVolumeSpec._fields)


def _parse_modal_volume_spec(entry: Any) -> ModalVolumeSpec:
    """Validate and build a ModalVolumeSpec from an ``x-modal.volumes`` entry.

    Raises a descriptive error naming ``x-modal.volumes`` instead of letting
    malformed entries surface as opaque ``NamedTuple`` construction errors
    (e.g. "unexpected keyword argument" or "must be a mapping, not str").
    """
    if not isinstance(entry, dict):
        raise TypeError(
            "x-modal.volumes entries must be mappings with 'name' and "
            "'mount_path' keys (e.g. {name: myvol, mount_path: /data}), got "
            f"{type(entry).__name__}. Docker Compose's short volume syntax "
            "(e.g. 'myvol:/data:ro') is not supported here."
        )
    unknown_keys = set(entry) - _MODAL_VOLUME_KEYS
    if unknown_keys:
        raise ValueError(
            f"x-modal.volumes entry has unexpected key(s) {sorted(unknown_keys)}; "
            f"expected only {sorted(_MODAL_VOLUME_KEYS)}"
        )
    missing_keys = {"name", "mount_path"} - set(entry)
    if missing_keys:
        raise ValueError(
            f"x-modal.volumes entry is missing required key(s) {sorted(missing_keys)}"
        )
    return ModalVolumeSpec(**entry)


def _service_to_cpu(service: ComposeService) -> float | tuple[float, float] | None:
    """Extract CPU configuration from compose service.

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
                mem_reservation = parse_memory(resources.reservations.memory)
            except ValueError as e:
                raise ValueError(
                    f"Invalid memory reservation in deploy.resources: {e}"
                ) from e

        if resources.limits and resources.limits.memory:
            try:
                mem_limit = parse_memory(resources.limits.memory)
            except ValueError as e:
                raise ValueError(
                    f"Invalid memory limit in deploy.resources: {e}"
                ) from e

    # Fall back to service-level field (v2 format)
    if mem_limit is None and service.mem_limit:
        try:
            mem_limit = parse_memory(service.mem_limit)
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

    Returns:
        GPU specification string for Modal, or None if no GPU requested.
        - "ANY:<count>": Any GPU with specified count (e.g., "ANY:2")
        - "ANY": Any single GPU

    Note:
        Compose GPU config doesn't specify GPU types (A10G, T4, etc.), so we
        default to "ANY". Use x-modal.gpu extension to specify a particular GPU
        type, which will override this value.
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
