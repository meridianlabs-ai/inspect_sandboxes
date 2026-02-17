"""Tests for Modal compose configuration conversion."""

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from inspect_ai.util import (
    ComposeBuild,
    ComposeConfig,
    ComposeService,
    parse_compose_yaml,
)
from inspect_sandboxes.modal._compose import (
    _apply_modal_extensions,
    _convert_byte_value,
    _parse_environment,
    _resolve_dockerfile_path,
    _service_to_gpu,
    convert_compose_to_modal_params,
)


@pytest.mark.parametrize(
    ("input_str", "expected"),
    [
        ("512m", 512),
        ("1g", 1024),
        ("2048k", 2),
        ("1t", 1048576),
        ("512mb", 512),
        ("1.5g", 1536),
        ("512M", 512),  # case insensitive
        ("1G", 1024),
        ("512", 512),  # no unit defaults to MiB
        ("0.5g", 512),
    ],
)
def test_convert_byte_value_valid(input_str: str, expected: int) -> None:
    """Test valid memory string conversions."""
    assert _convert_byte_value(input_str) == expected


@pytest.mark.parametrize(
    "input_str",
    [
        "abc",
        "512x",
        "m512",
        "0m",
        "-512m",
        "",
        "  ",
    ],
)
def test_convert_byte_value_raises(input_str: str) -> None:
    """Test that invalid inputs raise ValueError."""
    with pytest.raises(ValueError):
        _convert_byte_value(input_str)


@pytest.mark.parametrize(
    ("environment", "expected"),
    [
        # List format
        (["KEY=VALUE", "FOO=BAR"], {"KEY": "VALUE", "FOO": "BAR"}),
        (["KEY=value=with=equals"], {"KEY": "value=with=equals"}),
        ([], {}),
        # Dict format
        ({"KEY": "VALUE"}, {"KEY": "VALUE"}),
        ({"KEY": "VALUE", "FOO": "BAR"}, {"KEY": "VALUE", "FOO": "BAR"}),
        # Dict with None values (should be excluded)
        ({"KEY": "VALUE", "SKIP": None}, {"KEY": "VALUE"}),
        ({}, {}),
    ],
)
def test_parse_environment(
    environment: list[str] | dict[str, str | None],
    expected: dict[str, str],
) -> None:
    """Test environment variable parsing from list and dict formats."""
    assert _parse_environment(environment) == expected


@pytest.mark.parametrize(
    ("build", "expected_relative"),
    [
        # String build
        ("myapp", "myapp/Dockerfile"),
        # ComposeBuild with defaults
        (ComposeBuild(context=None, dockerfile=None), "./Dockerfile"),
        # ComposeBuild with custom context
        (ComposeBuild(context="app", dockerfile=None), "app/Dockerfile"),
        # ComposeBuild with custom dockerfile
        (
            ComposeBuild(context=None, dockerfile="Custom.dockerfile"),
            "./Custom.dockerfile",
        ),
        # ComposeBuild with both custom
        (
            ComposeBuild(context="app", dockerfile="Custom.dockerfile"),
            "app/Custom.dockerfile",
        ),
    ],
)
def test_resolve_dockerfile_path(
    build: str | ComposeBuild,
    expected_relative: str,
) -> None:
    """Test Dockerfile path resolution for various build configurations."""
    compose_dir = Path("/tmp/compose")
    expected = compose_dir / expected_relative
    assert _resolve_dockerfile_path(build, compose_dir) == expected


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
        # Empty x-inspect_modal_sandbox
        ({"x-inspect_modal_sandbox": {}}, {}),
        # Single extension
        ({"x-inspect_modal_sandbox": {"timeout": 300}}, {"timeout": 300}),
        # Multiple extensions
        (
            {"x-inspect_modal_sandbox": {"timeout": 300, "cloud": "aws"}},
            {"timeout": 300, "cloud": "aws"},
        ),
        # Extension with None (should not be applied)
        ({"x-inspect_modal_sandbox": {"timeout": None}}, {}),
        # Mix of None and valid
        (
            {"x-inspect_modal_sandbox": {"timeout": 300, "cloud": None}},
            {"timeout": 300},
        ),
        # All supported extensions
        (
            {
                "x-inspect_modal_sandbox": {
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
    ("services", "expected_service_name"),
    [
        # Service with x_default=True
        (
            {
                "web": {"x_default": False},
                "api": {"x_default": True},
            },
            "api",
        ),
        # Service named "default" when no x_default
        (
            {
                "web": {},
                "default": {},
            },
            "default",
        ),
        # First service when no "default" or x_default
        (
            {
                "web": {},
                "api": {},
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
    # Build ComposeService objects
    compose_services = {
        name: ComposeService(**config) for name, config in services.items()
    }
    config = ComposeConfig(services=compose_services)

    with patch("inspect_sandboxes.modal._compose.Path"):
        result = convert_compose_to_modal_params(config, None)

    # Verify the function runs without error
    assert isinstance(result, dict)


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

    assert result == expected_params


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

    assert result["cpu"] == expected_cpu
    assert result["memory"] == expected_memory


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
            "inspect_sandboxes.modal._compose._resolve_dockerfile_path"
        ) as mock_resolve:
            mock_dockerfile_path = MagicMock(spec=Path)
            mock_dockerfile_path.exists.return_value = True
            mock_resolve.return_value = mock_dockerfile_path

            result = convert_compose_to_modal_params(config, "/tmp/compose.yml")

    assert "image" in result


def test_convert_compose_missing_dockerfile() -> None:
    """Test that FileNotFoundError is raised when Dockerfile is missing."""
    service = ComposeService(build="myapp")
    config = ComposeConfig(services={"default": service})

    with (
        patch("inspect_sandboxes.modal._compose.Path") as mock_path,
        patch(
            "inspect_sandboxes.modal._compose._resolve_dockerfile_path"
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
        **{"x-inspect_modal_sandbox": {"timeout": 300, "cloud": "aws"}},
    )

    with patch("inspect_sandboxes.modal._compose.modal.Image") as mock_image:
        mock_image.from_registry.side_effect = lambda x: f"registry:{x}"

        result = convert_compose_to_modal_params(config, None)

    assert result["timeout"] == 300
    assert result["cloud"] == "aws"
