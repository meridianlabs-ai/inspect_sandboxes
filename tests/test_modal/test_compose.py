"""Tests for Modal compose configuration conversion."""

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, call, patch

import pytest
from inspect_ai.util import (
    ComposeConfig,
    ComposeService,
    parse_compose_yaml,
)
from inspect_sandboxes.modal._compose import (
    ModalVolumeSpec,
    _apply_modal_extensions,
    _apply_service_ports,
    _service_to_gpu,
    convert_compose_to_modal_params,
)


@pytest.fixture(autouse=True)
def _reset_warn_once() -> None:
    """warn_once dedupes globally by message; clear it between tests."""
    from inspect_ai._util import logger as inspect_logger

    inspect_logger._warned.clear()


@pytest.mark.parametrize(
    ("service_config", "expected"),
    [
        # No deploy config
        ({}, None),
        # No resources
        ({"deploy": {}}, None),
        # No GPU devices
        ({"deploy": {"resources": {"reservations": {}}}}, None),
        # GPU with count
        (
            {
                "deploy": {
                    "resources": {
                        "reservations": {
                            "devices": [{"capabilities": ["gpu"], "count": 2}]
                        }
                    }
                }
            },
            "ANY:2",
        ),
        # GPU with device_ids
        (
            {
                "deploy": {
                    "resources": {
                        "reservations": {
                            "devices": [
                                {"capabilities": ["gpu"], "device_ids": ["0", "1"]}
                            ]
                        }
                    }
                }
            },
            "ANY:2",
        ),
        # GPU with no count or device_ids
        (
            {
                "deploy": {
                    "resources": {
                        "reservations": {"devices": [{"capabilities": ["gpu"]}]}
                    }
                }
            },
            "ANY",
        ),
    ],
)
def test_service_to_gpu(
    service_config: dict[str, Any],
    expected: str | None,
) -> None:
    """Test GPU configuration extraction from service."""
    service = ComposeService(**service_config)
    assert _service_to_gpu(service) == expected


@pytest.mark.parametrize(
    ("extensions", "expected_params"),
    [
        # No extensions
        ({}, {}),
        # Empty x-modal
        ({"x-modal": {}}, {}),
        # Single extension
        ({"x-modal": {"timeout": 300}}, {"timeout": 300}),
        # Multiple extensions
        (
            {"x-modal": {"timeout": 300, "cloud": "aws"}},
            {"timeout": 300, "cloud": "aws"},
        ),
        # Extension with None (should not be applied)
        ({"x-modal": {"timeout": None}}, {}),
        # Mix of None and valid
        (
            {"x-modal": {"timeout": 300, "cloud": None}},
            {"timeout": 300},
        ),
        # All supported extensions
        (
            {
                "x-modal": {
                    "block_network": True,
                    "cidr_allowlist": ["10.0.0.0/8"],
                    "timeout": 300,
                    "cloud": "aws",
                    "region": "us-east-1",
                    "idle_timeout": 60,
                    "pty": True,
                    "encrypted_ports": [443],
                    "h2_ports": [8080],
                    "unencrypted_ports": [80],
                    "custom_domain": "example.com",
                    "verbose": True,
                }
            },
            {
                "block_network": True,
                "cidr_allowlist": ["10.0.0.0/8"],
                "timeout": 300,
                "cloud": "aws",
                "region": "us-east-1",
                "idle_timeout": 60,
                "pty": True,
                "encrypted_ports": [443],
                "h2_ports": [8080],
                "unencrypted_ports": [80],
                "custom_domain": "example.com",
                "verbose": True,
            },
        ),
    ],
)
def test_apply_modal_extensions(
    extensions: dict[str, Any],
    expected_params: dict[str, Any],
) -> None:
    """Test Modal extensions are correctly applied to params dict."""
    params: dict[str, Any] = {}
    _apply_modal_extensions(params, extensions)
    assert params == expected_params


@pytest.mark.parametrize(
    ("secret_config", "expected_names"),
    [
        ("service-secret", ["service-secret"]),
        (["service-secret", "shared-secret"], ["service-secret", "shared-secret"]),
    ],
)
def test_apply_modal_extensions_secrets(
    secret_config: str | list[str],
    expected_names: list[str],
) -> None:
    """Test x-modal secrets are converted via modal.Secret.from_name."""
    params: dict[str, Any] = {}
    secret_objects = [MagicMock(name=f"secret:{name}") for name in expected_names]

    with patch(
        "inspect_sandboxes.modal._compose.modal.Secret.from_name"
    ) as mock_secret:
        mock_secret.side_effect = secret_objects

        _apply_modal_extensions(params, {"x-modal": {"secrets": secret_config}})

    assert mock_secret.call_args_list == [call(name) for name in expected_names]
    assert params["secrets"] == secret_objects


@pytest.mark.parametrize(
    ("services", "expected_service_name"),
    [
        # Service with x-default=True
        (
            {
                "web": {"x-default": False, "working_dir": "/web"},
                "api": {"x-default": True, "working_dir": "/api"},
            },
            "api",
        ),
        # Service named "default" when no x_default
        (
            {
                "web": {"working_dir": "/web"},
                "default": {"working_dir": "/default"},
            },
            "default",
        ),
        # First service when no "default" or x_default
        (
            {
                "web": {"working_dir": "/web"},
                "api": {"working_dir": "/api"},
            },
            "web",  # First in iteration order
        ),
    ],
)
def test_convert_compose_service_selection(
    services: dict[str, dict[str, Any]],
    expected_service_name: str,
) -> None:
    """Test that the correct service is selected based on priority."""
    compose_services = {
        name: ComposeService(**config) for name, config in services.items()
    }
    config = ComposeConfig(services=compose_services)

    with patch("inspect_sandboxes.modal._compose.Path"):
        result = convert_compose_to_modal_params(config, None)

    assert result.kwargs["workdir"] == f"/{expected_service_name}"


@pytest.mark.parametrize(
    ("service_config", "compose_path", "expected_params"),
    [
        # Image from registry
        (
            {"image": "python:3.12"},
            None,
            {"image": "registry:python:3.12"},
        ),
        # Working directory
        (
            {"image": "python:3.12", "working_dir": "/app"},
            None,
            {"image": "registry:python:3.12", "workdir": "/app"},
        ),
        # Environment variables
        (
            {
                "image": "python:3.12",
                "environment": ["KEY=VALUE"],
            },
            None,
            {"image": "registry:python:3.12", "env": {"KEY": "VALUE"}},
        ),
        # Memory limit
        (
            {"image": "python:3.12", "mem_limit": "512m"},
            None,
            {"image": "registry:python:3.12", "memory": 512},
        ),
        # CPU count
        (
            {"image": "python:3.12", "cpus": 2.0},
            None,
            {"image": "registry:python:3.12", "cpu": 2.0},
        ),
        # GPU configuration
        (
            {
                "image": "python:3.12",
                "deploy": {
                    "resources": {
                        "reservations": {
                            "devices": [{"capabilities": ["gpu"], "count": 1}]
                        }
                    }
                },
            },
            None,
            {"image": "registry:python:3.12", "gpu": "ANY:1"},
        ),
        # All parameters combined
        (
            {
                "image": "python:3.12",
                "working_dir": "/app",
                "environment": {"KEY": "VALUE"},
                "mem_limit": "1g",
                "cpus": 2.0,
                "deploy": {
                    "resources": {
                        "reservations": {
                            "devices": [{"capabilities": ["gpu"], "count": 2}]
                        }
                    }
                },
            },
            None,
            {
                "image": "registry:python:3.12",
                "workdir": "/app",
                "env": {"KEY": "VALUE"},
                "memory": 1024,
                "cpu": 2.0,
                "gpu": "ANY:2",
            },
        ),
    ],
)
def test_convert_compose_to_modal_params(
    service_config: dict[str, Any],
    compose_path: str | None,
    expected_params: dict[str, Any],
) -> None:
    """Test conversion of compose config to Modal params."""
    service = ComposeService(**service_config)
    config = ComposeConfig(services={"default": service})

    # Mock modal.Image methods
    with (
        patch("inspect_sandboxes.modal._compose.modal.Image") as mock_image,
        patch("inspect_sandboxes.modal._compose.Path") as mock_path,
    ):
        # Setup mocks
        mock_image.from_registry.side_effect = lambda x: f"registry:{x}"
        mock_image.from_dockerfile.side_effect = lambda x: f"dockerfile:{x}"
        mock_path_instance = MagicMock()
        mock_path_instance.exists.return_value = True
        mock_path.return_value = mock_path_instance

        result = convert_compose_to_modal_params(config, compose_path)

    assert result.kwargs == expected_params


@pytest.mark.parametrize(
    ("compose_yaml", "expected_cpu", "expected_memory"),
    [
        # Both reservations and limits (should return tuples)
        (
            """
services:
  default:
    image: python:3.12
    deploy:
      resources:
        reservations:
          cpus: "0.5"
          memory: 512m
        limits:
          cpus: "2.0"
          memory: 1g
""",
            (0.5, 2.0),
            (512, 1024),
        ),
        # Limits only (should return single values)
        (
            """
services:
  default:
    image: python:3.12
    deploy:
      resources:
        limits:
          cpus: "2.0"
          memory: 1g
""",
            2.0,
            1024,
        ),
        # Reservations only (should return single values)
        (
            """
services:
  default:
    image: python:3.12
    deploy:
      resources:
        reservations:
          cpus: "0.5"
          memory: 512m
""",
            0.5,
            512,
        ),
        # Service-level fallback (v2 format - no deploy.resources)
        (
            """
services:
  default:
    image: python:3.12
    cpus: 2.0
    mem_limit: 1g
""",
            2.0,
            1024,
        ),
    ],
)
def test_convert_compose_resource_tuples(
    tmp_path: Path,
    compose_yaml: str,
    expected_cpu: float | tuple[float, float],
    expected_memory: int | tuple[int, int],
) -> None:
    """Test CPU and memory (request, limit) tuple handling."""
    compose_file = tmp_path / "compose.yaml"
    compose_file.write_text(compose_yaml)

    config = parse_compose_yaml(str(compose_file), multiple_services=False)

    with patch("inspect_sandboxes.modal._compose.modal.Image") as mock_image:
        mock_image.from_registry.side_effect = lambda x: f"registry:{x}"
        result = convert_compose_to_modal_params(config, None)

    assert result.kwargs["cpu"] == expected_cpu
    assert result.kwargs["memory"] == expected_memory


def test_convert_compose_with_build() -> None:
    """Test conversion with build configuration."""
    service = ComposeService(build="myapp")
    config = ComposeConfig(services={"default": service})

    with (
        patch("inspect_sandboxes.modal._compose.modal.Image") as mock_image,
        patch("inspect_sandboxes.modal._compose.Path") as mock_path,
    ):
        mock_image.from_dockerfile.return_value = "dockerfile:myapp"
        mock_path_instance = MagicMock()
        mock_path_instance.parent = Path("/tmp")
        mock_path_instance.exists.return_value = True
        mock_path.return_value = mock_path_instance

        # Mock the dockerfile path
        with patch(
            "inspect_sandboxes.modal._compose.resolve_dockerfile_path"
        ) as mock_resolve:
            mock_dockerfile_path = MagicMock(spec=Path)
            mock_dockerfile_path.exists.return_value = True
            mock_resolve.return_value = mock_dockerfile_path

            result = convert_compose_to_modal_params(config, "/tmp/compose.yml")

    assert "image" in result.kwargs


def test_network_mode_none_sets_block_network() -> None:
    """Test that network_mode='none' translates to block_network=True."""
    config = ComposeConfig(
        services={"default": ComposeService(image="python:3.12", network_mode="none")}
    )

    with patch("inspect_sandboxes.modal._compose.modal.Image") as mock_image:
        mock_image.from_registry.side_effect = lambda x: f"registry:{x}"
        result = convert_compose_to_modal_params(config, None)

    assert result.kwargs.get("block_network") is True


def test_network_mode_bridge_allows_network() -> None:
    """Test that network_mode='bridge' translates to block_network=False."""
    config = ComposeConfig(
        services={"default": ComposeService(image="python:3.12", network_mode="bridge")}
    )

    with patch("inspect_sandboxes.modal._compose.modal.Image") as mock_image:
        mock_image.from_registry.side_effect = lambda x: f"registry:{x}"
        result = convert_compose_to_modal_params(config, None)

    assert result.kwargs.get("block_network") is False


def test_x_modal_overrides_network_mode() -> None:
    """Test that x-modal block_network overrides service network_mode."""
    config = ComposeConfig(
        services={"default": ComposeService(image="python:3.12", network_mode="none")},
        **{"x-modal": {"block_network": False}},
    )

    with patch("inspect_sandboxes.modal._compose.modal.Image") as mock_image:
        mock_image.from_registry.side_effect = lambda x: f"registry:{x}"
        result = convert_compose_to_modal_params(config, None)

    assert result.kwargs.get("block_network") is False


def test_convert_compose_missing_dockerfile() -> None:
    """Test that FileNotFoundError is raised when Dockerfile is missing."""
    service = ComposeService(build="myapp")
    config = ComposeConfig(services={"default": service})

    with (
        patch("inspect_sandboxes.modal._compose.Path") as mock_path,
        patch(
            "inspect_sandboxes.modal._compose.resolve_dockerfile_path"
        ) as mock_resolve,
    ):
        mock_path_instance = MagicMock()
        mock_path_instance.parent = Path("/tmp")
        mock_path.return_value = mock_path_instance

        mock_dockerfile_path = MagicMock(spec=Path)
        mock_dockerfile_path.exists.return_value = False
        mock_resolve.return_value = mock_dockerfile_path

        with pytest.raises(FileNotFoundError, match="Dockerfile not found"):
            convert_compose_to_modal_params(config, "/tmp/compose.yml")


def test_convert_compose_invalid_mem_limit() -> None:
    """Test that ValueError is raised for invalid mem_limit."""
    service = ComposeService(image="python:3.12", mem_limit="invalid")
    config = ComposeConfig(services={"default": service})

    with patch("inspect_sandboxes.modal._compose.modal.Image"):
        with pytest.raises(ValueError, match="Invalid mem_limit"):
            convert_compose_to_modal_params(config, None)


def test_convert_compose_with_extensions() -> None:
    """Test that Modal extensions are applied."""
    service = ComposeService(image="python:3.12")
    config = ComposeConfig(
        services={"default": service},
        **{"x-modal": {"timeout": 300, "cloud": "aws"}},
    )

    with patch("inspect_sandboxes.modal._compose.modal.Image") as mock_image:
        mock_image.from_registry.side_effect = lambda x: f"registry:{x}"

        result = convert_compose_to_modal_params(config, None)

    assert result.kwargs["timeout"] == 300
    assert result.kwargs["cloud"] == "aws"


def test_convert_compose_with_secret_extensions() -> None:
    """Test that x-modal secrets are converted and included in kwargs."""
    service = ComposeService(image="python:3.12")
    config = ComposeConfig(
        services={"default": service},
        **{"x-modal": {"secrets": ["service-secret", "shared-secret"]}},
    )
    secret_objects = [
        MagicMock(name="secret:service-secret"),
        MagicMock(name="secret:shared-secret"),
    ]

    with (
        patch("inspect_sandboxes.modal._compose.modal.Image") as mock_image,
        patch("inspect_sandboxes.modal._compose.modal.Secret.from_name") as mock_secret,
    ):
        mock_image.from_registry.side_effect = lambda x: f"registry:{x}"
        mock_secret.side_effect = secret_objects

        result = convert_compose_to_modal_params(config, None)

    assert mock_secret.call_args_list == [
        call("service-secret"),
        call("shared-secret"),
    ]
    assert result.kwargs["secrets"] == secret_objects


def test_convert_compose_modal_volume_specs() -> None:
    """Compose x-modal volumes become read-only Modal volume specifications."""
    config = ComposeConfig(
        services={"default": ComposeService(image="python:3.12")},
        **{
            "x-modal": {
                "volumes": [
                    {
                        "name": "agent-cli-claude-2-1-205",
                        "mount_path": "/opt/agent-cli/claude",
                        "read_only": True,
                    }
                ]
            }
        },
    )

    with patch("inspect_sandboxes.modal._compose.modal.Image") as mock_image:
        mock_image.from_registry.side_effect = lambda image: f"registry:{image}"
        params = convert_compose_to_modal_params(config, None)

    assert params.volumes == [
        ModalVolumeSpec(
            name="agent-cli-claude-2-1-205",
            mount_path="/opt/agent-cli/claude",
            read_only=True,
        )
    ]
    assert isinstance(params.volumes[0], ModalVolumeSpec)


class TestModalVolumeValidation:
    """Malformed x-modal.volumes entries raise errors that name the key."""

    @staticmethod
    def _config(volumes: list[Any]) -> ComposeConfig:
        return ComposeConfig(
            services={"default": ComposeService(image="python:3.12")},
            **{"x-modal": {"volumes": volumes}},
        )

    def test_read_only_defaults_to_false_when_omitted(self) -> None:
        """Omitting read_only is valid; it defaults to False like Modal's own default."""
        config = self._config(
            [
                {
                    "name": "agent-cli-claude-2-1-205",
                    "mount_path": "/opt/agent-cli/claude",
                }
            ]
        )

        with patch("inspect_sandboxes.modal._compose.modal.Image") as mock_image:
            mock_image.from_registry.side_effect = lambda image: f"registry:{image}"
            params = convert_compose_to_modal_params(config, None)

        assert params.volumes == [
            ModalVolumeSpec(
                name="agent-cli-claude-2-1-205",
                mount_path="/opt/agent-cli/claude",
                read_only=False,
            )
        ]

    def test_unknown_key_names_x_modal_volumes(self) -> None:
        """An unexpected key raises a ValueError naming x-modal.volumes."""
        config = self._config(
            [
                {
                    "name": "agent-cli-claude-2-1-205",
                    "mount_path": "/opt/agent-cli/claude",
                    "readonly": True,  # typo for read_only
                }
            ]
        )

        with patch("inspect_sandboxes.modal._compose.modal.Image"):
            with pytest.raises(ValueError, match="x-modal.volumes"):
                convert_compose_to_modal_params(config, None)

    def test_docker_shorthand_string_names_x_modal_volumes(self) -> None:
        """Docker Compose's short volume syntax raises a TypeError naming x-modal.volumes."""
        config = self._config(["agent-cli-claude-2-1-205:/opt/agent-cli/claude:ro"])

        with patch("inspect_sandboxes.modal._compose.modal.Image"):
            with pytest.raises(TypeError, match="x-modal.volumes"):
                convert_compose_to_modal_params(config, None)

    def test_missing_required_key_names_x_modal_volumes(self) -> None:
        """A missing required key raises a ValueError naming x-modal.volumes."""
        config = self._config([{"name": "agent-cli-claude-2-1-205"}])

        with patch("inspect_sandboxes.modal._compose.modal.Image"):
            with pytest.raises(ValueError, match="x-modal.volumes"):
                convert_compose_to_modal_params(config, None)


def test_service_ports_translated_to_unencrypted_ports() -> None:
    """service.ports container side defaults into unencrypted_ports."""
    params: dict[str, Any] = {}
    _apply_service_ports(params, ComposeService(image="x", ports=["8080:80", "443"]))
    assert params["unencrypted_ports"] == [80, 443]


def test_service_ports_differing_host_port_warns_and_translates_container(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A differing host port can't be honored; warn but still expose container."""
    params: dict[str, Any] = {}
    with caplog.at_level("WARNING"):
        _apply_service_ports(params, ComposeService(image="x", ports=["9090:80"]))
    assert params["unencrypted_ports"] == [80]
    assert any("host port" in r.message for r in caplog.records)


def test_service_ports_udp_and_range_skipped_with_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """UDP and port ranges can't be tunnels; skip and warn, keep the rest."""
    params: dict[str, Any] = {}
    with caplog.at_level("WARNING"):
        _apply_service_ports(
            params,
            ComposeService(image="x", ports=["53:53/udp", "8000-8005:8000-8005", "80"]),
        )
    assert params["unencrypted_ports"] == [80]
    messages = " ".join(r.message for r in caplog.records)
    assert "UDP" in messages
    assert "range" in messages


def test_service_ports_malformed_value_warns_neutrally(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A malformed (non-range) entry warns without mislabelling it a range."""
    params: dict[str, Any] = {}
    with caplog.at_level("WARNING"):
        _apply_service_ports(params, ComposeService(image="x", ports=["notaport"]))
    assert "unencrypted_ports" not in params
    messages = " ".join(r.message for r in caplog.records)
    assert "notaport" in messages
    assert "port range or malformed" in messages


def test_expose_warns_and_is_not_translated(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Expose is host-private: warn, never translate."""
    params: dict[str, Any] = {}
    with caplog.at_level("WARNING"):
        _apply_service_ports(params, ComposeService(image="x", expose=["5432"]))
    assert "unencrypted_ports" not in params
    assert any("expose" in r.message for r in caplog.records)


def test_x_modal_ports_override_translated_service_ports() -> None:
    """Any x-modal.*_ports replaces the ports translated from service.ports."""
    config = ComposeConfig(
        services={"default": ComposeService(image="python:3.12", ports=["8080:80"])},
        **{"x-modal": {"encrypted_ports": [443]}},
    )
    with patch("inspect_sandboxes.modal._compose.modal.Image") as mock_image:
        mock_image.from_registry.side_effect = lambda x: f"registry:{x}"
        result = convert_compose_to_modal_params(config, None)
    # service.ports would have produced unencrypted_ports=[80]; the explicit
    # encrypted_ports extension overrides it entirely.
    assert "unencrypted_ports" not in result.kwargs
    assert result.kwargs["encrypted_ports"] == [443]


def test_service_ports_without_extension_survive() -> None:
    """With no x-modal port override, translated service.ports are applied."""
    config = ComposeConfig(
        services={"default": ComposeService(image="python:3.12", ports=["80:80"])},
    )
    with patch("inspect_sandboxes.modal._compose.modal.Image") as mock_image:
        mock_image.from_registry.side_effect = lambda x: f"registry:{x}"
        result = convert_compose_to_modal_params(config, None)
    assert result.kwargs["unencrypted_ports"] == [80]


@pytest.mark.parametrize(
    ("command", "expected_command"),
    [
        # String command is split via shlex
        ("echo hello world", ["echo", "hello", "world"]),
        # List command is passed through
        (["echo", "hello world"], ["echo", "hello world"]),
        # No command
        (None, []),
    ],
)
def test_convert_compose_command(
    command: str | list[str] | None,
    expected_command: list[str],
) -> None:
    """Test that service command is parsed into ModalSandboxParams.command."""
    service_kwargs: dict[str, Any] = {"image": "python:3.12"}
    if command is not None:
        service_kwargs["command"] = command
    service = ComposeService(**service_kwargs)
    config = ComposeConfig(services={"default": service})

    with patch("inspect_sandboxes.modal._compose.modal.Image") as mock_image:
        mock_image.from_registry.side_effect = lambda x: f"registry:{x}"
        result = convert_compose_to_modal_params(config, None)

    assert result.command == expected_command


class TestImageRegistrySecret:
    """A private registry needs credentials at image-PULL time.

    Without them Modal reports `RemoteError('Image build for im-... failed')`, which reads
    as a build problem rather than an auth problem, so the cause is easy to misdiagnose.
    `x-modal.secrets` does NOT cover this: it injects env vars into the running sandbox
    and has no effect on the pull.
    """

    @staticmethod
    def _config(tmp_path: Path, x_modal: str) -> ComposeConfig:
        compose = tmp_path / "compose.yaml"
        _ = compose.write_text(
            "services:\n  default:\n    image: ghcr.io/private/app:tag\n" + x_modal,
            encoding="utf-8",
        )
        return parse_compose_yaml(str(compose), multiple_services=False)

    def test_registry_secret_is_passed_to_from_registry(self, tmp_path: Path) -> None:
        """The named Modal secret authenticates `Image.from_registry`."""
        config = self._config(
            tmp_path, "x-modal:\n  image_registry_secret: ghcr-secret\n"
        )

        with patch(
            "inspect_sandboxes.modal._compose.modal.Image.from_registry"
        ) as from_registry:
            with patch(
                "inspect_sandboxes.modal._compose.modal.Secret.from_name"
            ) as from_name:
                from_name.return_value = "SENTINEL_SECRET"
                convert_compose_to_modal_params(config, None)

        from_name.assert_called_once_with("ghcr-secret")
        assert from_registry.call_args.kwargs.get("secret") == "SENTINEL_SECRET"

    def test_no_secret_declared_keeps_the_anonymous_pull(self, tmp_path: Path) -> None:
        """Without the key, the pull stays anonymous (no `secret` kwarg)."""
        config = self._config(tmp_path, "")

        with patch(
            "inspect_sandboxes.modal._compose.modal.Image.from_registry"
        ) as from_registry:
            convert_compose_to_modal_params(config, None)

        assert "secret" not in from_registry.call_args.kwargs

    def test_sandbox_secrets_do_not_authenticate_the_pull(self, tmp_path: Path) -> None:
        """`secrets` must not be mistaken for registry credentials."""
        config = self._config(tmp_path, "x-modal:\n  secrets: some-env-secret\n")

        with patch(
            "inspect_sandboxes.modal._compose.modal.Image.from_registry"
        ) as from_registry:
            with patch("inspect_sandboxes.modal._compose.modal.Secret.from_name"):
                convert_compose_to_modal_params(config, None)

        assert "secret" not in from_registry.call_args.kwargs

    def test_registry_secret_with_build_fails_loudly(self, tmp_path: Path) -> None:
        """`build:` cannot use the pull secret; raise rather than silently ignore it."""
        compose = tmp_path / "compose.yaml"
        (tmp_path / "Dockerfile").write_text("FROM python:3.12\n", encoding="utf-8")
        _ = compose.write_text(
            "services:\n  default:\n    build: .\n"
            "x-modal:\n  image_registry_secret: ghcr-secret\n",
            encoding="utf-8",
        )
        config = parse_compose_yaml(str(compose), multiple_services=False)

        with pytest.raises(ValueError, match="not supported with build:"):
            convert_compose_to_modal_params(config, None)

    def test_non_string_secret_name_fails_loudly(self, tmp_path: Path) -> None:
        """A non-string secret name raises rather than being ignored."""
        config = self._config(
            tmp_path, "x-modal:\n  image_registry_secret: [ghcr-secret]\n"
        )

        with pytest.raises(TypeError):
            convert_compose_to_modal_params(config, None)
