"""Tests for Daytona compose configuration conversion."""

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from daytona_sdk import (
    CreateSandboxFromImageParams,
    CreateSandboxFromSnapshotParams,
    Image,
)
from inspect_ai.util import (
    ComposeConfig,
    ComposeService,
    parse_compose_yaml,
)
from inspect_ai.util._sandbox.compose import (
    ComposeDeploy,
    ComposeResourceConfig,
    ComposeResources,
)
from inspect_sandboxes.daytona._compose import (
    _service_to_resources,
    _to_gib,
    aggregate_resources,
    apply_daytona_extensions,
    create_single_service_params,
    extract_daytona_timeout,
)

STUB_LABELS = {"created_by": "test"}


@pytest.mark.parametrize(
    ("mem_str", "expected_gib"),
    [
        ("512m", 1),  # 0.5 GiB -> rounds up to 1 GiB
        ("1g", 1),  # exactly 1 GiB
        ("1536m", 2),  # 1.5 GiB -> rounds up to 2 GiB
        ("2g", 2),  # exactly 2 GiB
        ("100m", 1),  # tiny -> minimum 1 GiB
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
        # All simple supported extensions.
        # `timeout` is intentionally omitted from the expected params dict — it's
        # handled via extract_daytona_timeout and forwarded to client.create(),
        # not applied as a sandbox params field (see extract_daytona_timeout test).
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
    apply_daytona_extensions(params, extensions)
    assert params == expected_params


@pytest.mark.parametrize(
    ("extensions", "expected"),
    [
        ({}, None),
        ({"x-daytona": {}}, None),
        ({"x-daytona": {"timeout": 120.0}}, 120.0),
        # `timeout` must come through even when other x-daytona keys are present
        ({"x-daytona": {"timeout": 30, "ephemeral": True}}, 30),
        # Numeric strings (what YAML produces from quoted values) must coerce.
        ({"x-daytona": {"timeout": "45"}}, 45.0),
        ({"x-daytona": {"timeout": "45.5"}}, 45.5),
        # int values are fine (YAML unquoted ints).
        ({"x-daytona": {"timeout": 0}}, 0.0),
    ],
)
def test_extract_daytona_timeout(
    extensions: dict[str, Any], expected: float | None
) -> None:
    """Test that x-daytona.timeout is extracted independently of the params dict."""
    result = extract_daytona_timeout(extensions)
    assert result == expected
    if result is not None:
        assert isinstance(result, float)


@pytest.mark.parametrize(
    "bad_value",
    [
        "not-a-number",
        "",
        "30s",  # common mistake: adding a unit suffix
        [],
        {},
    ],
)
def test_extract_daytona_timeout_rejects_non_numeric(bad_value: Any) -> None:
    """Values that can't coerce to float must raise ValueError with context."""
    with pytest.raises(ValueError, match="x-daytona.timeout must be a number"):
        extract_daytona_timeout({"x-daytona": {"timeout": bad_value}})


def test_apply_daytona_extensions_does_not_set_timeout() -> None:
    """Regression: apply_daytona_extensions must NOT place `timeout` in params.

    Unpacking `timeout` into the Pydantic sandbox params model would silently
    drop it (the field doesn't exist), so timeout is handled out-of-band via
    extract_daytona_timeout and forwarded to AsyncDaytona.create().
    """
    params: dict[str, Any] = {}
    apply_daytona_extensions(params, {"x-daytona": {"timeout": 45.0}})
    assert "timeout" not in params


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
        # Memory from deploy.resources.limits ("2g" -> 2 GiB)
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
        # GPU with no count -> defaults to 1
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
        # CPU from service-level cpus field
        ({"cpus": 4.0}, 4, None, None),
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
def test_create_single_service_params_image_type(
    service_config: dict[str, Any],
    compose_path: str | None,
    expected_image_type: type,
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
            result = create_single_service_params(config, compose_path, STUB_LABELS)
            assert isinstance(result, CreateSandboxFromImageParams)
            assert isinstance(result.image, str)
            assert result.image == "python:3.12"
        else:
            mock_image.from_dockerfile.return_value = MagicMock(spec=Image)
            create_single_service_params(config, compose_path, STUB_LABELS)
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

    result = create_single_service_params(config, None, STUB_LABELS)

    assert result.env_vars == expected_env_vars


def test_convert_compose_with_extensions() -> None:
    """Test that x-daytona extensions are applied to sandbox_params."""
    config = ComposeConfig(
        services={"default": ComposeService(image="python:3.12")},
        **{"x-daytona": {"network_block_all": True, "ephemeral": True}},
    )

    result = create_single_service_params(config, None, STUB_LABELS)

    assert result.network_block_all is True
    assert result.ephemeral is True


def test_convert_compose_os_user_from_service() -> None:
    """Test that os_user is extracted from the service user field."""
    config = ComposeConfig(
        services={"default": ComposeService(image="python:3.12", user="ubuntu")}
    )

    result = create_single_service_params(config, None, STUB_LABELS)

    assert result.os_user == "ubuntu"


def test_convert_compose_os_user_extension_overrides_service() -> None:
    """Test that x-daytona os_user overrides the service-level user field."""
    config = ComposeConfig(
        services={"default": ComposeService(image="python:3.12", user="ubuntu")},
        **{"x-daytona": {"os_user": "root"}},
    )

    result = create_single_service_params(config, None, STUB_LABELS)

    assert result.os_user == "root"


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

    result = create_single_service_params(config, None, STUB_LABELS)

    assert result.env_vars == {
        "FROM_SERVICE": "service_val",
        "FROM_EXT": "ext_val",
        "SHARED": "ext",  # x-daytona wins
    }


def test_convert_compose_labels_from_extension() -> None:
    """Test that x-daytona labels are merged with run labels."""
    config = ComposeConfig(
        services={"default": ComposeService(image="python:3.12")},
        **{"x-daytona": {"labels": {"team": "ml", "project": "evals"}}},
    )

    result = create_single_service_params(config, None, STUB_LABELS)

    # x-daytona labels merged with run labels; run labels take precedence
    assert result.labels is not None
    assert result.labels["team"] == "ml"
    assert result.labels["project"] == "evals"
    assert result.labels["created_by"] == "test"


def test_network_mode_none_sets_block_all() -> None:
    """Test that network_mode='none' translates to network_block_all=True."""
    config = ComposeConfig(
        services={"default": ComposeService(image="python:3.12", network_mode="none")}
    )

    result = create_single_service_params(config, None, STUB_LABELS)

    assert result.network_block_all is True


def test_network_mode_bridge_allows_network() -> None:
    """Test that network_mode='bridge' translates to network_block_all=False."""
    config = ComposeConfig(
        services={"default": ComposeService(image="python:3.12", network_mode="bridge")}
    )

    result = create_single_service_params(config, None, STUB_LABELS)

    assert result.network_block_all is False


def test_x_daytona_overrides_network_mode() -> None:
    """Test that x-daytona network_block_all overrides service network_mode."""
    config = ComposeConfig(
        services={"default": ComposeService(image="python:3.12", network_mode="none")},
        **{"x-daytona": {"network_block_all": False}},
    )

    result = create_single_service_params(config, None, STUB_LABELS)

    assert result.network_block_all is False


def test_convert_compose_missing_image_and_build() -> None:
    """Test ValueError when service has neither image nor build."""
    config = ComposeConfig(
        services={"default": ComposeService()}  # No image or build
    )

    with pytest.raises(ValueError, match="must specify either 'image' or 'build'"):
        create_single_service_params(config, None, STUB_LABELS)


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
            create_single_service_params(config, "/tmp/compose.yml", STUB_LABELS)


def test_convert_compose_service_selection_x_default() -> None:
    """Test x_default service is preferred."""
    config = ComposeConfig(
        services={
            "web": ComposeService(image="nginx:latest"),
            "api": ComposeService(**{"image": "python:3.12", "x-default": True}),  # type: ignore[arg-type]
        }
    )

    result = create_single_service_params(config, None, STUB_LABELS)
    assert isinstance(result, CreateSandboxFromImageParams)
    assert result.image == "python:3.12"


def test_convert_compose_service_selection_default_name() -> None:
    """Test 'default' service is preferred when no x_default."""
    config = ComposeConfig(
        services={
            "web": ComposeService(image="nginx:latest"),
            "default": ComposeService(image="python:3.12"),
        }
    )

    result = create_single_service_params(config, None, STUB_LABELS)
    assert isinstance(result, CreateSandboxFromImageParams)
    assert result.image == "python:3.12"


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
    result = create_single_service_params(config, str(compose_file), STUB_LABELS)

    assert isinstance(result, CreateSandboxFromImageParams)
    assert result.image == "python:3.12"
    assert result.resources is not None
    assert result.resources.cpu == 4
    assert result.resources.memory == 4


def test_auto_stop_interval_defaults_to_zero() -> None:
    """Test that auto_stop_interval defaults to 0 when not set."""
    config = ComposeConfig(services={"default": ComposeService(image="python:3.12")})

    result = create_single_service_params(config, None, STUB_LABELS)

    assert result.auto_stop_interval == 0


def test_auto_stop_interval_from_extension() -> None:
    """Test that x-daytona auto_stop_interval overrides the default."""
    config = ComposeConfig(
        services={"default": ComposeService(image="python:3.12")},
        **{"x-daytona": {"auto_stop_interval": 30}},
    )

    result = create_single_service_params(config, None, STUB_LABELS)

    assert result.auto_stop_interval == 30


def test_aggregate_resources_sums_with_overhead() -> None:
    """Test aggregate_resources sums per-service resources and adds daemon overhead."""
    config = ComposeConfig(
        services={
            "web": ComposeService(
                image="python:3.12",
                deploy=ComposeDeploy(
                    resources=ComposeResourceConfig(
                        limits=ComposeResources(cpus="2", memory="2g")
                    )
                ),
            ),
            "db": ComposeService(
                image="postgres:16",
                deploy=ComposeDeploy(
                    resources=ComposeResourceConfig(
                        limits=ComposeResources(cpus="1", memory="1g")
                    )
                ),
            ),
        }
    )

    result = aggregate_resources(config)

    assert result is not None
    assert result.cpu == 4  # 2 + 1 + 1 overhead
    assert result.memory == 4  # 2 + 1 + 1 overhead
    assert result.gpu is None


def test_aggregate_resources_returns_none_when_no_resources() -> None:
    """Test aggregate_resources returns None when no services have resources."""
    config = ComposeConfig(
        services={
            "web": ComposeService(image="python:3.12"),
            "db": ComposeService(image="postgres:16"),
        }
    )

    result = aggregate_resources(config)

    assert result is None


def test_create_single_service_params_with_snapshot() -> None:
    """Test that x-daytona.snapshot returns CreateSandboxFromSnapshotParams."""
    config = ComposeConfig(
        services={"default": ComposeService(image="python:3.12")},
        **{"x-daytona": {"snapshot": "my-snapshot"}},
    )

    result = create_single_service_params(config, None, STUB_LABELS)

    assert isinstance(result, CreateSandboxFromSnapshotParams)
    assert result.snapshot == "my-snapshot"


def test_create_single_service_params_with_resources_override() -> None:
    """Test that x-daytona.resources overrides service-level resources."""
    config = ComposeConfig(
        services={"default": ComposeService(image="python:3.12")},
        **{"x-daytona": {"resources": {"cpu": 4, "memory": 8}}},
    )

    result = create_single_service_params(config, None, STUB_LABELS)

    assert isinstance(result, CreateSandboxFromImageParams)
    assert result.resources is not None
    assert result.resources.cpu == 4
    assert result.resources.memory == 8


def test_create_single_service_params_forwards_name_to_image_params() -> None:
    """``name`` kwarg reaches CreateSandboxFromImageParams on the image path."""
    config = ComposeConfig(services={"default": ComposeService(image="python:3.12")})
    result = create_single_service_params(
        config, None, STUB_LABELS, name="inspect-foo-1-abcdef12"
    )
    assert isinstance(result, CreateSandboxFromImageParams)
    assert result.name == "inspect-foo-1-abcdef12"


def test_create_single_service_params_forwards_name_to_snapshot_params() -> None:
    """``name`` kwarg reaches CreateSandboxFromSnapshotParams on the snapshot path."""
    config = ComposeConfig(
        services={"default": ComposeService(image="python:3.12")},
        **{"x-daytona": {"snapshot": "my-snap"}},
    )
    result = create_single_service_params(
        config, None, STUB_LABELS, name="inspect-foo-1-abcdef12"
    )
    assert isinstance(result, CreateSandboxFromSnapshotParams)
    assert result.name == "inspect-foo-1-abcdef12"


def test_create_single_service_params_name_defaults_to_none() -> None:
    """Backward-compat: omitting ``name`` must not break existing callers."""
    config = ComposeConfig(services={"default": ComposeService(image="python:3.12")})
    result = create_single_service_params(config, None, STUB_LABELS)
    assert result.name is None
