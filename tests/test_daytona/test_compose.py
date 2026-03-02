"""Tests for Daytona compose configuration conversion."""

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from daytona_sdk import Image
from inspect_ai.util import (
    ComposeConfig,
    ComposeService,
    parse_compose_yaml,
)
from inspect_sandboxes.daytona._compose import (
    _apply_daytona_extensions,
    _service_to_resources,
    _to_gib,
    convert_compose_to_daytona_params,
)


@pytest.mark.parametrize(
    ("mem_str", "expected_gib"),
    [
        ("512m", 1),  # 0.5 GiB → rounds up to 1 GiB
        ("1g", 1),  # exactly 1 GiB
        ("1536m", 2),  # 1.5 GiB → rounds up to 2 GiB
        ("2g", 2),  # exactly 2 GiB
        ("100m", 1),  # tiny → minimum 1 GiB
        ("4g", 4),  # 4 GiB
    ],
)
def test_to_gib(mem_str: str, expected_gib: int) -> None:
    """Test memory string to GiB conversion with ceiling rounding and minimum 1."""
    assert _to_gib(mem_str) == expected_gib


@pytest.mark.parametrize(
    ("extensions", "expected_params"),
    [
        ({}, {}),
        ({"x-daytona": {}}, {}),
        ({"x-daytona": {"auto_stop_interval": 10}}, {"auto_stop_interval": 10}),
        (
            {
                "x-daytona": {
                    "network_block_all": True,
                    "network_allow_list": "10.0.0.0/8",
                }
            },
            {"network_block_all": True, "network_allow_list": "10.0.0.0/8"},
        ),
        # None values should not be applied
        ({"x-daytona": {"auto_stop_interval": None}}, {}),
        # All simple supported extensions
        (
            {
                "x-daytona": {
                    "auto_stop_interval": 30,
                    "auto_archive_interval": 60,
                    "auto_delete_interval": 120,
                    "network_block_all": False,
                    "network_allow_list": "192.168.0.0/16",
                    "language": "python",
                    "os_user": "root",
                    "public": False,
                    "ephemeral": False,
                    "timeout": 60.0,
                }
            },
            {
                "auto_stop_interval": 30,
                "auto_archive_interval": 60,
                "auto_delete_interval": 120,
                "network_block_all": False,
                "network_allow_list": "192.168.0.0/16",
                "language": "python",
                "os_user": "root",
                "public": False,
                "ephemeral": False,
                "timeout": 60.0,
            },
        ),
    ],
)
def test_apply_daytona_extensions(
    extensions: dict[str, Any],
    expected_params: dict[str, Any],
) -> None:
    """Test Daytona extensions are correctly applied to params dict."""
    params: dict[str, Any] = {}
    _apply_daytona_extensions(params, extensions)
    assert params == expected_params


@pytest.mark.parametrize(
    ("service_config", "expected_cpu", "expected_memory_gib", "expected_gpu"),
    [
        # No resources
        ({}, None, None, None),
        # CPU from deploy.resources.limits
        (
            {"deploy": {"resources": {"limits": {"cpus": "2.0"}}}},
            2,
            None,
            None,
        ),
        # CPU rounds up (0.5 → 1)
        (
            {"deploy": {"resources": {"limits": {"cpus": "0.5"}}}},
            1,
            None,
            None,
        ),
        # Memory from deploy.resources.limits ("2g" → 2 GiB)
        (
            {"deploy": {"resources": {"limits": {"memory": "2g"}}}},
            None,
            2,
            None,
        ),
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
            None,
            None,
            2,
        ),
        # GPU with no count → defaults to 1
        (
            {
                "deploy": {
                    "resources": {
                        "reservations": {"devices": [{"capabilities": ["gpu"]}]}
                    }
                }
            },
            None,
            None,
            1,
        ),
        # limits wins over reservations when both present
        (
            {
                "deploy": {
                    "resources": {
                        "limits": {"cpus": "2.0", "memory": "2g"},
                        "reservations": {"cpus": "8.0", "memory": "8g"},
                    }
                }
            },
            2,
            2,
            None,
        ),
        # CPU from service-level cpus field
        ({"cpus": 4.0}, 4, None, None),
        # CPU rounds up from service-level (1.5 → 2)
        ({"cpus": 1.5}, 2, None, None),
    ],
)
def test_service_to_resources(
    service_config: dict[str, Any],
    expected_cpu: int | None,
    expected_memory_gib: int | None,
    expected_gpu: int | None,
) -> None:
    """Test resource extraction from compose service."""
    service = ComposeService(**service_config)
    result = _service_to_resources(service)

    if expected_cpu is None and expected_memory_gib is None and expected_gpu is None:
        assert result is None
    else:
        assert result is not None
        assert result.cpu == expected_cpu
        assert result.memory == expected_memory_gib
        assert result.gpu == expected_gpu


@pytest.mark.parametrize(
    ("service_config", "compose_path", "expected_image_type"),
    [
        # Image from registry
        ({"image": "python:3.12"}, None, str),
        # Build from Dockerfile
        ({"build": "myapp"}, None, Image),
    ],
)
def test_convert_compose_to_daytona_params_image_type(
    service_config: dict[str, Any],
    compose_path: str | None,
    expected_image_type: type,
    tmp_path: Path,
) -> None:
    """Test that image is returned as the correct type."""
    service = ComposeService(**service_config)
    config = ComposeConfig(services={"default": service})

    with (
        patch("inspect_sandboxes.daytona._compose.Path") as mock_path_cls,
        patch("inspect_sandboxes.daytona._compose.Image") as mock_image,
    ):
        mock_dockerfile = MagicMock()
        mock_dockerfile.exists.return_value = True
        mock_path_cls.return_value.parent.__truediv__ = MagicMock(
            return_value=mock_dockerfile
        )

        if expected_image_type is str:
            image, resources, sandbox_params = convert_compose_to_daytona_params(
                config, compose_path
            )
            assert isinstance(image, str)
            assert image == "python:3.12"
        else:
            mock_image.from_dockerfile.return_value = MagicMock(spec=Image)
            image, resources, sandbox_params = convert_compose_to_daytona_params(
                config, compose_path
            )
            mock_image.from_dockerfile.assert_called_once()


@pytest.mark.parametrize(
    ("environment", "expected_env_vars"),
    [
        ({"MY_VAR": "value", "OTHER": "data"}, {"MY_VAR": "value", "OTHER": "data"}),
        (["KEY=VALUE", "FOO=BAR"], {"KEY": "VALUE", "FOO": "BAR"}),
    ],
)
def test_convert_compose_env_vars(
    environment: dict[str, str | None] | list[str],
    expected_env_vars: dict[str, str],
) -> None:
    """Test environment variables in dict and list formats are parsed into env_vars."""
    config = ComposeConfig(
        services={
            "default": ComposeService(image="python:3.12", environment=environment)
        }
    )

    _, _, sandbox_params = convert_compose_to_daytona_params(config, None)

    assert sandbox_params.get("env_vars") == expected_env_vars


def test_convert_compose_with_extensions() -> None:
    """Test that x-daytona extensions are applied to sandbox_params."""
    config = ComposeConfig(
        services={"default": ComposeService(image="python:3.12")},
        **{"x-daytona": {"network_block_all": True, "ephemeral": True}},
    )

    image, resources, sandbox_params = convert_compose_to_daytona_params(config, None)

    assert sandbox_params.get("network_block_all") is True
    assert sandbox_params.get("ephemeral") is True


def test_convert_compose_os_user_from_service() -> None:
    """Test that os_user is extracted from the service user field."""
    config = ComposeConfig(
        services={"default": ComposeService(image="python:3.12", user="ubuntu")}
    )

    _, _, sandbox_params = convert_compose_to_daytona_params(config, None)

    assert sandbox_params.get("os_user") == "ubuntu"


def test_convert_compose_os_user_extension_overrides_service() -> None:
    """Test that x-daytona os_user overrides the service-level user field."""
    config = ComposeConfig(
        services={"default": ComposeService(image="python:3.12", user="ubuntu")},
        **{"x-daytona": {"os_user": "root"}},
    )

    _, _, sandbox_params = convert_compose_to_daytona_params(config, None)

    assert sandbox_params.get("os_user") == "root"


def test_convert_compose_env_vars_merge_with_x_daytona() -> None:
    """Test that x-daytona env_vars merge with service environment, x-daytona wins."""
    config = ComposeConfig(
        services={
            "default": ComposeService(
                image="python:3.12",
                environment={"FROM_SERVICE": "service_val", "SHARED": "service"},
            )
        },
        **{"x-daytona": {"env_vars": {"FROM_EXT": "ext_val", "SHARED": "ext"}}},
    )

    _, _, sandbox_params = convert_compose_to_daytona_params(config, None)

    assert sandbox_params["env_vars"] == {
        "FROM_SERVICE": "service_val",
        "FROM_EXT": "ext_val",
        "SHARED": "ext",  # x-daytona wins
    }


def test_convert_compose_labels_from_extension() -> None:
    """Test that x-daytona labels are stored in sandbox_params for caller to merge."""
    config = ComposeConfig(
        services={"default": ComposeService(image="python:3.12")},
        **{"x-daytona": {"labels": {"team": "ml", "project": "evals"}}},
    )

    _, _, sandbox_params = convert_compose_to_daytona_params(config, None)

    assert sandbox_params.get("labels") == {"team": "ml", "project": "evals"}


def test_convert_compose_missing_image_and_build() -> None:
    """Test ValueError when service has neither image nor build."""
    config = ComposeConfig(
        services={"default": ComposeService()}  # No image or build
    )

    with pytest.raises(ValueError, match="must specify either 'image' or 'build'"):
        convert_compose_to_daytona_params(config, None)


def test_convert_compose_missing_dockerfile() -> None:
    """Test FileNotFoundError when Dockerfile is missing."""
    service = ComposeService(build="nonexistent")
    config = ComposeConfig(services={"default": service})

    with patch(
        "inspect_sandboxes.daytona._compose.resolve_dockerfile_path"
    ) as mock_resolve:
        mock_path = MagicMock(spec=Path)
        mock_path.exists.return_value = False
        mock_resolve.return_value = mock_path

        with pytest.raises(FileNotFoundError, match="Dockerfile not found"):
            convert_compose_to_daytona_params(config, "/tmp/compose.yml")


def test_convert_compose_service_selection_x_default() -> None:
    """Test x_default service is preferred."""
    config = ComposeConfig(
        services={
            "web": ComposeService(image="nginx:latest"),
            "api": ComposeService(**{"image": "python:3.12", "x-default": True}),
        }
    )

    image, _, _ = convert_compose_to_daytona_params(config, None)
    assert image == "python:3.12"


def test_convert_compose_service_selection_default_name() -> None:
    """Test 'default' service is preferred when no x_default."""
    config = ComposeConfig(
        services={
            "web": ComposeService(image="nginx:latest"),
            "default": ComposeService(image="python:3.12"),
        }
    )

    image, _, _ = convert_compose_to_daytona_params(config, None)
    assert image == "python:3.12"


def test_convert_compose_from_yaml_file(tmp_path: Path) -> None:
    """Test end-to-end conversion from a YAML file parses resources correctly."""
    compose_file = tmp_path / "compose.yaml"
    compose_file.write_text("""
services:
  default:
    image: python:3.12
    deploy:
      resources:
        limits:
          cpus: "4.0"
          memory: 4g
""")

    config = parse_compose_yaml(str(compose_file), multiple_services=False)
    image, resources, _ = convert_compose_to_daytona_params(config, str(compose_file))

    assert image == "python:3.12"
    assert resources is not None
    assert resources.cpu == 4
    assert resources.memory == 4
