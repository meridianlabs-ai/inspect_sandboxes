"""Tests for Novita compose configuration conversion."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from inspect_ai.util import ComposeConfig, ComposeService
from inspect_sandboxes.novita._compose import (
    DEFAULT_CPU_COUNT,
    DEFAULT_MEMORY_MB,
    _service_to_resources,
    extract_novita_timeout,
    resolve_single_service_params,
    service_connection_ports,
)


@pytest.fixture(autouse=True)
def _reset_warn_once() -> None:
    """warn_once dedupes globally by message; clear it between tests."""
    from inspect_ai._util import logger as inspect_logger

    inspect_logger._warned.clear()


@pytest.mark.parametrize(
    ("extensions", "expected"),
    [
        ({}, None),
        ({"x-novita": {}}, None),
        ({"x-novita": {"timeout": 1800}}, 1800.0),
        # `timeout` must come through even when other x-novita keys are present
        ({"x-novita": {"timeout": 30, "template": "foo"}}, 30),
        # Numeric strings (what YAML produces from quoted values) must coerce.
        ({"x-novita": {"timeout": "45"}}, 45.0),
        ({"x-novita": {"timeout": "45.5"}}, 45.5),
        # int values are fine (YAML unquoted ints).
        ({"x-novita": {"timeout": 0}}, 0.0),
    ],
)
def test_extract_novita_timeout(
    extensions: dict[str, Any], expected: float | None
) -> None:
    """Test that x-novita.timeout is extracted independently of the params dict."""
    result = extract_novita_timeout(extensions)
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
def test_extract_novita_timeout_rejects_non_numeric(bad_value: Any) -> None:
    """Values that can't coerce to float must raise ValueError with context."""
    with pytest.raises(ValueError, match="x-novita.timeout must be a number"):
        extract_novita_timeout({"x-novita": {"timeout": bad_value}})


@pytest.mark.parametrize(
    ("service_config", "extensions", "expected_cpu", "expected_memory_mb"),
    [
        # Defaults (nothing set)
        ({"image": "alpine"}, {}, DEFAULT_CPU_COUNT, DEFAULT_MEMORY_MB),
        # x-novita overrides take precedence
        (
            {"image": "alpine"},
            {"cpu_count": 8, "memory_mb": 4096},
            8,
            4096,
        ),
        # deploy.resources.limits
        (
            {
                "image": "alpine",
                "deploy": {"resources": {"limits": {"cpus": "4", "memory": "2g"}}},
            },
            {},
            4,
            2 * 1024,
        ),
        # deploy.resources.reservations (fallback when limits absent)
        (
            {
                "image": "alpine",
                "deploy": {
                    "resources": {"reservations": {"cpus": "2", "memory": "512m"}}
                },
            },
            {},
            2,
            512,
        ),
        # service-level cpus/mem_limit
        (
            {"image": "alpine", "cpus": 2.5, "mem_limit": "1g"},
            {},
            3,  # ceil(2.5)
            1024,
        ),
        # x-novita overrides deploy.resources
        (
            {
                "image": "alpine",
                "deploy": {"resources": {"limits": {"cpus": "2", "memory": "1g"}}},
            },
            {"cpu_count": 8},
            8,
            1024,
        ),
    ],
)
def test_service_to_resources(
    service_config: dict[str, Any],
    extensions: dict[str, Any],
    expected_cpu: int,
    expected_memory_mb: int,
) -> None:
    """Test resource extraction from compose service."""
    service = ComposeService(**service_config)
    cpu, memory_mb = _service_to_resources(service, extensions)
    assert cpu == expected_cpu
    assert memory_mb == expected_memory_mb


def test_image_only_uses_image_template_path() -> None:
    config = ComposeConfig(services={"default": ComposeService(image="python:3.12")})
    result = resolve_single_service_params(config, None)
    assert result.image == "python:3.12"
    assert result.dockerfile_path is None
    assert result.template is None


def test_build_resolves_dockerfile(tmp_path: Path) -> None:
    df = tmp_path / "Dockerfile"
    df.write_text("FROM python:3.12\n")
    config = ComposeConfig(services={"default": ComposeService(build=".")})
    result = resolve_single_service_params(config, str(tmp_path / "compose.yaml"))
    assert result.dockerfile_path == str(df)
    assert result.image is None


def test_x_novita_template_skips_build() -> None:
    config = ComposeConfig(
        services={"default": ComposeService(image="alpine")},
        **{"x-novita": {"template": "my-prebuilt-template"}},  # type: ignore[arg-type]
    )
    result = resolve_single_service_params(config, None)
    assert result.template == "my-prebuilt-template"
    assert result.dockerfile_path is None


def test_environment_and_user_passthrough() -> None:
    config = ComposeConfig(
        services={
            "default": ComposeService(
                image="alpine",
                environment=["FOO=bar", "BAZ=qux"],
                user="nobody",
            )
        }
    )
    result = resolve_single_service_params(config, None)
    assert result.envs == {"FOO": "bar", "BAZ": "qux"}
    assert result.user == "nobody"


def test_x_novita_envs_extend_environment() -> None:
    config = ComposeConfig(
        services={"default": ComposeService(image="alpine", environment=["FOO=bar"])},
        **{"x-novita": {"envs": {"EXTRA": "value"}}},  # type: ignore[arg-type]
    )
    result = resolve_single_service_params(config, None)
    assert result.envs == {"FOO": "bar", "EXTRA": "value"}


def test_x_novita_user_overrides_service_user() -> None:
    config = ComposeConfig(
        services={"default": ComposeService(image="alpine", user="nobody")},
        **{"x-novita": {"user": "root"}},  # type: ignore[arg-type]
    )
    result = resolve_single_service_params(config, None)
    assert result.user == "root"


def test_x_novita_metadata() -> None:
    config = ComposeConfig(
        services={"default": ComposeService(image="alpine")},
        **{"x-novita": {"metadata": {"project": "alpha", "owner": "jj"}}},  # type: ignore[arg-type]
    )
    result = resolve_single_service_params(config, None)
    assert result.metadata == {"project": "alpha", "owner": "jj"}


def test_network_mode_none_blocks_network() -> None:
    """Test that network_mode='none' translates to allow_internet_access=False."""
    config = ComposeConfig(
        services={"default": ComposeService(image="python:3.12", network_mode="none")}
    )
    result = resolve_single_service_params(config, None)
    assert result.allow_internet_access is False


def test_network_mode_bridge_allows_network() -> None:
    """Test that network_mode='bridge' translates to allow_internet_access=True."""
    config = ComposeConfig(
        services={"default": ComposeService(image="python:3.12", network_mode="bridge")}
    )
    result = resolve_single_service_params(config, None)
    assert result.allow_internet_access is True


def test_x_novita_overrides_network_mode() -> None:
    """Test that x-novita allow_internet_access overrides service network_mode."""
    config = ComposeConfig(
        services={"default": ComposeService(image="python:3.12", network_mode="none")},
        **{"x-novita": {"allow_internet_access": True}},  # type: ignore[arg-type]
    )
    result = resolve_single_service_params(config, None)
    assert result.allow_internet_access is True


def test_convert_compose_missing_image_and_build() -> None:
    """Test ValueError when service has neither image nor build."""
    config = ComposeConfig(services={"default": ComposeService()})
    with pytest.raises(ValueError, match="image"):
        resolve_single_service_params(config, None)


def test_convert_compose_missing_dockerfile() -> None:
    """Test FileNotFoundError when Dockerfile is missing."""
    config = ComposeConfig(services={"default": ComposeService(build="nonexistent")})

    with patch(
        "inspect_sandboxes.novita._compose.resolve_dockerfile_path"
    ) as mock_resolve:
        mock_path = MagicMock(spec=Path)
        mock_path.exists.return_value = False
        mock_resolve.return_value = mock_path

        with pytest.raises(FileNotFoundError, match="Dockerfile not found"):
            resolve_single_service_params(config, "/tmp/compose.yml")


def test_default_resources() -> None:
    config = ComposeConfig(services={"default": ComposeService(image="alpine")})
    result = resolve_single_service_params(config, None)
    assert result.cpu_count == DEFAULT_CPU_COUNT
    assert result.memory_mb == DEFAULT_MEMORY_MB


def test_deploy_resources_limits() -> None:
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
    assert result.cpu_count == 4
    assert result.memory_mb == 2 * 1024


def test_service_level_cpus_mem() -> None:
    config = ComposeConfig(
        services={"default": ComposeService(image="alpine", cpus=2.5, mem_limit="512m")}
    )
    result = resolve_single_service_params(config, None)
    assert result.cpu_count == 3  # ceil(2.5)
    assert result.memory_mb == 512


def test_x_novita_resources_take_precedence() -> None:
    config = ComposeConfig(
        services={
            "default": ComposeService(
                image="alpine",
                deploy={  # type: ignore[arg-type]
                    "resources": {"limits": {"cpus": "4", "memory": "2g"}}
                },
            )
        },
        **{"x-novita": {"cpu_count": 8, "memory_mb": 4096}},  # type: ignore[arg-type]
    )
    result = resolve_single_service_params(config, None)
    assert result.cpu_count == 8
    assert result.memory_mb == 4096


def test_service_connection_ports_returns_container_ports() -> None:
    """Ports container side is recorded for connection(); host side dropped."""
    service = ComposeService(image="x", ports=["8080:80", "443"])
    assert service_connection_ports(service) == [80, 443]


def test_service_connection_ports_skips_range_and_udp(
    caplog: pytest.LogCaptureFixture,
) -> None:
    service = ComposeService(
        image="x", ports=["53:53/udp", "8000-8005:8000-8005", "80"]
    )
    with caplog.at_level("WARNING"):
        ports = service_connection_ports(service)
    assert ports == [80]
    messages = " ".join(r.message for r in caplog.records)
    assert "range" in messages
    assert "UDP" in messages


def test_service_connection_ports_malformed_value_warns_neutrally(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A malformed (non-range) entry warns without mislabelling it a range."""
    service = ComposeService(image="x", ports=["notaport"])
    with caplog.at_level("WARNING"):
        ports = service_connection_ports(service)
    assert ports == []
    messages = " ".join(r.message for r in caplog.records)
    assert "notaport" in messages
    assert "port range or malformed" in messages


def test_service_connection_ports_expose_warns_not_surfaced(
    caplog: pytest.LogCaptureFixture,
) -> None:
    service = ComposeService(image="x", expose=["5432"])
    with caplog.at_level("WARNING"):
        ports = service_connection_ports(service)
    assert ports == []
    assert any("expose" in r.message for r in caplog.records)
