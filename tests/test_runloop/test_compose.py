"""Tests for Runloop compose configuration parsing."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from inspect_ai.util import ComposeConfig, ComposeService
from inspect_sandboxes.runloop._compose import (
    extract_runloop_timeout,
    resolve_single_service_params,
)


@pytest.mark.parametrize(
    ("extensions", "expected"),
    [
        ({}, None),
        ({"x-runloop": {}}, None),
        ({"x-runloop": {"timeout": 1800}}, 1800.0),
        # `timeout` must come through even when other x-runloop keys are present
        ({"x-runloop": {"timeout": 30, "blueprint_name": "foo"}}, 30),
        # Numeric strings (what YAML produces from quoted values) must coerce.
        ({"x-runloop": {"timeout": "45"}}, 45.0),
        ({"x-runloop": {"timeout": "45.5"}}, 45.5),
        # int values are fine (YAML unquoted ints).
        ({"x-runloop": {"timeout": 0}}, 0.0),
    ],
)
def test_extract_runloop_timeout(
    extensions: dict[str, Any], expected: float | None
) -> None:
    """Test that x-runloop.timeout is extracted independently of the params dict."""
    result = extract_runloop_timeout(extensions)
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
def test_extract_runloop_timeout_rejects_non_numeric(bad_value: Any) -> None:
    """Values that can't coerce to float must raise ValueError with context."""
    with pytest.raises(ValueError, match="x-runloop.timeout must be a number"):
        extract_runloop_timeout({"x-runloop": {"timeout": bad_value}})


def test_image_only_uses_image_template_path() -> None:
    config = ComposeConfig(services={"default": ComposeService(image="python:3.12")})
    result = resolve_single_service_params(config, None)
    assert result.image == "python:3.12"
    assert result.dockerfile_path is None
    assert result.blueprint_id is None


def test_build_resolves_dockerfile(tmp_path: Path) -> None:
    df = tmp_path / "Dockerfile"
    df.write_text("FROM python:3.12\n")
    config = ComposeConfig(services={"default": ComposeService(build=".")})
    result = resolve_single_service_params(config, str(tmp_path / "compose.yaml"))
    assert result.dockerfile_path == str(df)
    assert result.image is None


def test_x_runloop_blueprint_name_skips_build() -> None:
    config = ComposeConfig(
        services={"default": ComposeService(image="alpine")},
        **{"x-runloop": {"blueprint_name": "my-prebuilt-blueprint"}},  # type: ignore[arg-type]
    )
    result = resolve_single_service_params(config, None)
    assert result.blueprint_name == "my-prebuilt-blueprint"
    assert result.dockerfile_path is None
    assert result.image is None


def test_x_runloop_blueprint_id_skips_build() -> None:
    config = ComposeConfig(
        services={"default": ComposeService(image="alpine")},
        **{"x-runloop": {"blueprint_id": "bp_123"}},  # type: ignore[arg-type]
    )
    result = resolve_single_service_params(config, None)
    assert result.blueprint_id == "bp_123"


def test_x_runloop_snapshot_id_skips_build() -> None:
    config = ComposeConfig(
        services={"default": ComposeService(image="alpine")},
        **{"x-runloop": {"snapshot_id": "snap_123"}},  # type: ignore[arg-type]
    )
    result = resolve_single_service_params(config, None)
    assert result.snapshot_id == "snap_123"


def test_environment_passthrough() -> None:
    config = ComposeConfig(
        services={
            "default": ComposeService(
                image="alpine", environment=["FOO=bar", "BAZ=qux"]
            )
        }
    )
    result = resolve_single_service_params(config, None)
    assert result.environment_variables == {"FOO": "bar", "BAZ": "qux"}


def test_x_runloop_environment_variables_extend() -> None:
    config = ComposeConfig(
        services={"default": ComposeService(image="alpine", environment=["FOO=bar"])},
        **{"x-runloop": {"environment_variables": {"EXTRA": "value"}}},  # type: ignore[arg-type]
    )
    result = resolve_single_service_params(config, None)
    assert result.environment_variables == {"FOO": "bar", "EXTRA": "value"}


def test_x_runloop_metadata() -> None:
    config = ComposeConfig(
        services={"default": ComposeService(image="alpine")},
        **{"x-runloop": {"metadata": {"project": "alpha"}}},  # type: ignore[arg-type]
    )
    result = resolve_single_service_params(config, None)
    assert result.metadata == {"project": "alpha"}


def test_convert_compose_missing_image_and_build() -> None:
    """Test ValueError when service has neither image nor build."""
    config = ComposeConfig(services={"default": ComposeService()})
    with pytest.raises(ValueError, match="image"):
        resolve_single_service_params(config, None)


def test_convert_compose_missing_dockerfile(tmp_path: Path) -> None:
    """Test FileNotFoundError when Dockerfile is missing."""
    config = ComposeConfig(services={"default": ComposeService(build="nonexistent")})
    with pytest.raises(FileNotFoundError, match="Dockerfile not found"):
        resolve_single_service_params(config, str(tmp_path / "compose.yaml"))


def test_no_default_resources() -> None:
    config = ComposeConfig(services={"default": ComposeService(image="alpine")})
    result = resolve_single_service_params(config, None)
    assert result.launch_parameters is None


def test_deploy_resources_to_launch_parameters() -> None:
    config = ComposeConfig(
        services={
            "default": ComposeService(
                image="alpine",
                deploy={  # type: ignore[arg-type]
                    "resources": {"limits": {"cpus": "4", "memory": "2g"}}
                },
            )
        }
    )
    result = resolve_single_service_params(config, None)
    assert result.launch_parameters is not None
    assert result.launch_parameters.get("custom_cpu_cores") == 4
    assert result.launch_parameters.get("custom_gb_memory") == 2
    assert result.launch_parameters.get("resource_size_request") == "CUSTOM_SIZE"


def test_service_level_cpus_mem() -> None:
    config = ComposeConfig(
        services={"default": ComposeService(image="alpine", cpus=2.5, mem_limit="4g")}
    )
    result = resolve_single_service_params(config, None)
    assert result.launch_parameters is not None
    assert result.launch_parameters.get("custom_cpu_cores") == 3  # ceil(2.5)
    assert result.launch_parameters.get("custom_gb_memory") == 4


def test_x_runloop_launch_parameters_take_precedence() -> None:
    config = ComposeConfig(
        services={
            "default": ComposeService(
                image="alpine",
                deploy={  # type: ignore[arg-type]
                    "resources": {"limits": {"cpus": "4", "memory": "2g"}}
                },
            )
        },
        **{"x-runloop": {"launch_parameters": {"custom_cpu_cores": 8}}},  # type: ignore[arg-type]
    )
    result = resolve_single_service_params(config, None)
    assert result.launch_parameters is not None
    assert result.launch_parameters.get("custom_cpu_cores") == 8
    # Memory from deploy.resources still passes through.
    assert result.launch_parameters.get("custom_gb_memory") == 2


def test_x_runloop_timeout_passthrough() -> None:
    config = ComposeConfig(
        services={"default": ComposeService(image="alpine")},
        **{"x-runloop": {"timeout": 1800}},  # type: ignore[arg-type]
    )
    result = resolve_single_service_params(config, None)
    assert result.timeout == 1800.0
