"""Tests for _util/compose.py utility functions."""

from pathlib import Path

import pytest
from inspect_ai.util import ComposeBuild
from inspect_sandboxes._util.compose import (
    parse_environment,
    parse_memory,
    resolve_dockerfile_path,
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
def test_parse_memory_valid(input_str: str, expected: int) -> None:
    """Test valid memory string conversions to MiB."""
    assert parse_memory(input_str) == expected


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
def test_parse_memory_invalid(input_str: str) -> None:
    """Test that invalid inputs raise ValueError."""
    with pytest.raises(ValueError):
        parse_memory(input_str)


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
    assert parse_environment(environment) == expected


@pytest.mark.parametrize(
    ("build", "expected_relative"),
    [
        # String build context
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
    assert (
        resolve_dockerfile_path(build, compose_dir) == compose_dir / expected_relative
    )
