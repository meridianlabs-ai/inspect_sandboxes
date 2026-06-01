"""Tests for E2B template build + content-hash caching."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from inspect_sandboxes.e2b._template import (
    TEMPLATE_NAME_PREFIX,
    build_template_for_dockerfile,
    build_template_for_image,
    template_name_for_dockerfile,
    template_name_for_image,
)


def test_dockerfile_name_is_content_derived(tmp_path: Any) -> None:
    """Identical content + resources → identical name; differing content → different."""
    df1 = tmp_path / "Dockerfile1"
    df1.write_text("FROM python:3.12\n")
    df2 = tmp_path / "Dockerfile2"
    df2.write_text("FROM python:3.12\n")  # same content
    df3 = tmp_path / "Dockerfile3"
    df3.write_text("FROM python:3.11\n")  # different content

    n1 = template_name_for_dockerfile(str(df1))
    n2 = template_name_for_dockerfile(str(df2))
    n3 = template_name_for_dockerfile(str(df3))

    assert n1 == n2
    assert n1 != n3
    assert n1.startswith(TEMPLATE_NAME_PREFIX)
    # 12-char hash + prefix
    assert len(n1) == len(TEMPLATE_NAME_PREFIX) + 12


def test_dockerfile_name_includes_resources(tmp_path: Any) -> None:
    """Different cpu_count or memory_mb → different name."""
    df = tmp_path / "Dockerfile"
    df.write_text("FROM python:3.12\n")

    base = template_name_for_dockerfile(str(df), cpu_count=2, memory_mb=1024)
    diff_cpu = template_name_for_dockerfile(str(df), cpu_count=4, memory_mb=1024)
    diff_mem = template_name_for_dockerfile(str(df), cpu_count=2, memory_mb=2048)

    assert base != diff_cpu
    assert base != diff_mem
    assert diff_cpu != diff_mem


def test_image_name_is_image_derived() -> None:
    n1 = template_name_for_image("python:3.12")
    n2 = template_name_for_image("python:3.12")
    n3 = template_name_for_image("python:3.11")

    assert n1 == n2
    assert n1 != n3


@pytest.mark.asyncio
async def test_dockerfile_build_invokes_sdk(tmp_path: Any) -> None:
    df = tmp_path / "Dockerfile"
    df.write_text("FROM python:3.12\nRUN echo hi\n")

    with patch("inspect_sandboxes.e2b._template.AsyncTemplate") as mock_cls:
        mock_cls.build = AsyncMock()
        builder = MagicMock()
        instance = MagicMock()
        instance.from_dockerfile = MagicMock(return_value=builder)
        mock_cls.return_value = instance

        name = await build_template_for_dockerfile(str(df), cpu_count=4, memory_mb=2048)

        instance.from_dockerfile.assert_called_once_with(str(df))
        mock_cls.build.assert_awaited_once()
        assert mock_cls.build.await_args is not None
        kwargs = mock_cls.build.await_args.kwargs
        assert kwargs["name"] == name
        assert kwargs["cpu_count"] == 4
        assert kwargs["memory_mb"] == 2048


@pytest.mark.asyncio
async def test_dockerfile_build_passes_file_context_path(tmp_path: Any) -> None:
    """COPY/ADD paths in the Dockerfile must resolve against the Dockerfile's own directory."""
    df = tmp_path / "Dockerfile"
    df.write_text("FROM python:3.12\nCOPY workspace/ /app/\n")

    with patch("inspect_sandboxes.e2b._template.AsyncTemplate") as mock_cls:
        mock_cls.build = AsyncMock()
        instance = MagicMock()
        instance.from_dockerfile = MagicMock(return_value=MagicMock())
        mock_cls.return_value = instance

        await build_template_for_dockerfile(str(df))

        # AsyncTemplate(...) must be called with file_context_path set to the
        # Dockerfile's parent directory.
        mock_cls.assert_called_once()
        ctor_kwargs = mock_cls.call_args.kwargs
        assert "file_context_path" in ctor_kwargs
        assert str(ctor_kwargs["file_context_path"]) == str(tmp_path)


@pytest.mark.asyncio
async def test_image_build_short_circuits_when_cached() -> None:
    with patch("inspect_sandboxes.e2b._template.AsyncTemplate") as mock_cls:
        mock_cls.exists = AsyncMock(return_value=True)
        mock_cls.build = AsyncMock()

        name = await build_template_for_image("python:3.12")

        mock_cls.exists.assert_awaited_once_with(name)
        mock_cls.build.assert_not_called()


@pytest.mark.asyncio
async def test_image_build_runs_when_not_cached() -> None:
    with patch("inspect_sandboxes.e2b._template.AsyncTemplate") as mock_cls:
        mock_cls.exists = AsyncMock(return_value=False)
        mock_cls.build = AsyncMock()
        builder = MagicMock()
        instance = MagicMock()
        instance.from_image = MagicMock(return_value=builder)
        mock_cls.return_value = instance

        name = await build_template_for_image("python:3.12")

        instance.from_image.assert_called_once_with("python:3.12")
        mock_cls.build.assert_awaited_once()
        assert mock_cls.build.await_args is not None
        assert mock_cls.build.await_args.kwargs["name"] == name
