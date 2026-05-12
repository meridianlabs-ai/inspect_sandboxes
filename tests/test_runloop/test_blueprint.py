"""Tests for Runloop blueprint naming + build helpers."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from inspect_sandboxes.runloop._blueprint import (
    BLUEPRINT_NAME_PREFIX,
    blueprint_name_for_dockerfile,
    blueprint_name_for_image,
    build_blueprint_for_dockerfile,
    build_blueprint_for_image,
)


def _make_paginator(items: list[Any]) -> MagicMock:
    """Create a mock async paginator over the given items."""

    async def _aiter():
        for item in items:
            yield item

    paginator = MagicMock()
    paginator.__aiter__ = lambda self: _aiter()
    return paginator


def _make_client(
    list_items: list[Any] | None = None,
    upload_object_id: str = "obj_test_123",
    created_blueprint_id: str = "bp_test_created",
) -> MagicMock:
    """Mock AsyncRunloop client wired for blueprint + build-context calls."""
    client = MagicMock()
    client.blueprints = MagicMock()
    client.blueprints.list = MagicMock(return_value=_make_paginator(list_items or []))
    created_blueprint = MagicMock()
    created_blueprint.id = created_blueprint_id
    client.blueprints.create = AsyncMock(return_value=created_blueprint)
    client.blueprints.await_build_complete = AsyncMock()
    # ``with_options(max_retries=0)`` returns a clone of the client. Make it
    # return self so .blueprints.create is the same mock as the top-level one.
    client.with_options = MagicMock(return_value=client)

    client.objects = MagicMock()
    created_object = MagicMock()
    created_object.id = upload_object_id
    created_object.upload_url = "https://upload.example/test"
    client.objects.create = AsyncMock(return_value=created_object)
    client.objects.complete = AsyncMock()
    client.objects.delete = AsyncMock()
    return client


def _patch_httpx_put() -> Any:
    """Patch ``httpx.AsyncClient`` so the build-context PUT succeeds without I/O."""
    response = MagicMock()
    response.raise_for_status = MagicMock()

    http_client = MagicMock()
    http_client.put = AsyncMock(return_value=response)
    http_client.__aenter__ = AsyncMock(return_value=http_client)
    http_client.__aexit__ = AsyncMock(return_value=None)
    return patch(
        "inspect_sandboxes.runloop._blueprint.httpx.AsyncClient",
        return_value=http_client,
    )


def test_dockerfile_name_is_content_derived(tmp_path: Any) -> None:
    """Identical content + launch params → identical name; differing content → different."""
    df1 = tmp_path / "Dockerfile1"
    df1.write_text("FROM python:3.12\n")
    df2 = tmp_path / "Dockerfile2"
    df2.write_text("FROM python:3.12\n")  # same content
    df3 = tmp_path / "Dockerfile3"
    df3.write_text("FROM python:3.11\n")  # different content

    n1 = blueprint_name_for_dockerfile(str(df1))
    n2 = blueprint_name_for_dockerfile(str(df2))
    n3 = blueprint_name_for_dockerfile(str(df3))

    assert n1 == n2
    assert n1 != n3
    assert n1.startswith(BLUEPRINT_NAME_PREFIX)
    assert len(n1) == len(BLUEPRINT_NAME_PREFIX) + 12


def test_dockerfile_name_includes_launch_parameters(tmp_path: Any) -> None:
    """Different launch parameters → different name."""
    df = tmp_path / "Dockerfile"
    df.write_text("FROM python:3.12\n")

    base = blueprint_name_for_dockerfile(str(df))
    diff_cpu = blueprint_name_for_dockerfile(
        str(df), launch_parameters={"custom_cpu_cores": 4}
    )
    diff_mem = blueprint_name_for_dockerfile(
        str(df), launch_parameters={"custom_gb_memory": 8}
    )
    same_as_base = blueprint_name_for_dockerfile(str(df), launch_parameters=None)

    assert base != diff_cpu
    assert base != diff_mem
    assert diff_cpu != diff_mem
    assert base == same_as_base


def test_image_name_is_image_derived() -> None:
    n1 = blueprint_name_for_image("python:3.12")
    n2 = blueprint_name_for_image("python:3.12")
    n3 = blueprint_name_for_image("python:3.11")

    assert n1 == n2
    assert n1 != n3


def test_dockerfile_and_image_names_differ_for_same_content(tmp_path: Any) -> None:
    """A Dockerfile reading ``FROM python:3.12`` should not collide with the image-only path."""
    df = tmp_path / "Dockerfile"
    df.write_text("FROM python:3.12\n")
    assert blueprint_name_for_dockerfile(str(df)) != blueprint_name_for_image(
        "python:3.12"
    )


@pytest.mark.asyncio
async def test_dockerfile_build_short_circuits_when_cached(tmp_path: Any) -> None:
    df = tmp_path / "Dockerfile"
    df.write_text("FROM python:3.12\n")
    cached = MagicMock()
    cached.id = "bp_cached"
    cached.status = "build_complete"
    client = _make_client(list_items=[cached])

    name = await build_blueprint_for_dockerfile(client, str(df))

    assert name.startswith(BLUEPRINT_NAME_PREFIX)
    client.blueprints.list.assert_called_once()
    client.blueprints.create.assert_not_called()


@pytest.mark.asyncio
async def test_dockerfile_build_runs_when_not_cached(tmp_path: Any) -> None:
    df = tmp_path / "Dockerfile"
    df.write_text("FROM python:3.12\nRUN echo hi\n")
    client = _make_client(list_items=[])

    with _patch_httpx_put():
        name = await build_blueprint_for_dockerfile(
            client,
            str(df),
            launch_parameters={"custom_cpu_cores": 4, "custom_gb_memory": 8},
        )

    client.blueprints.create.assert_awaited_once()
    kwargs = client.blueprints.create.await_args.kwargs
    assert kwargs["name"] == name
    assert kwargs["dockerfile"] == "FROM python:3.12\nRUN echo hi\n"
    assert kwargs["build_context"] == {"object_id": "obj_test_123", "type": "object"}
    assert kwargs["launch_parameters"]["custom_cpu_cores"] == 4
    assert kwargs["launch_parameters"]["custom_gb_memory"] == 8
    # Idempotency key dedupes SDK retries and concurrent samples.
    assert kwargs["idempotency_key"] == name


@pytest.mark.asyncio
async def test_dockerfile_build_uploads_context(tmp_path: Any) -> None:
    """COPY-able siblings get tarred + uploaded as a Runloop Object."""
    df = tmp_path / "Dockerfile"
    df.write_text("FROM python:3.12\nCOPY requirements.txt .\n")
    (tmp_path / "requirements.txt").write_text("numpy==2.0\n")
    client = _make_client(list_items=[])

    with _patch_httpx_put() as http_cls:
        await build_blueprint_for_dockerfile(client, str(df))

    client.objects.create.assert_awaited_once()
    create_kwargs = client.objects.create.await_args.kwargs
    assert create_kwargs["content_type"] == "tgz"
    assert create_kwargs["name"].startswith("inspect-context-")

    http_cls.return_value.put.assert_awaited_once()
    put_args = http_cls.return_value.put.await_args
    assert put_args.args[0] == "https://upload.example/test"

    client.objects.complete.assert_awaited_once_with("obj_test_123")
    # Context Object is deleted after the build so we don't blow Runloop's
    # free-tier Object cap.
    client.objects.delete.assert_awaited_once_with("obj_test_123")


def test_dockerfile_name_changes_when_context_file_changes(tmp_path: Any) -> None:
    """Editing a COPY-target invalidates the cached blueprint name."""
    df = tmp_path / "Dockerfile"
    df.write_text("FROM python:3.12\nCOPY requirements.txt .\n")
    req = tmp_path / "requirements.txt"

    req.write_text("numpy==2.0\n")
    n1 = blueprint_name_for_dockerfile(str(df))
    req.write_text("numpy==2.1\n")
    n2 = blueprint_name_for_dockerfile(str(df))

    assert n1 != n2


@pytest.mark.asyncio
async def test_image_build_short_circuits_when_cached() -> None:
    cached = MagicMock()
    cached.id = "bp_cached"
    cached.status = "build_complete"
    client = _make_client(list_items=[cached])

    name = await build_blueprint_for_image(client, "python:3.12")

    assert name.startswith(BLUEPRINT_NAME_PREFIX)
    client.blueprints.create.assert_not_called()


@pytest.mark.asyncio
async def test_image_build_runs_when_not_cached() -> None:
    client = _make_client(list_items=[])

    name = await build_blueprint_for_image(client, "python:3.12")

    client.blueprints.create.assert_awaited_once()
    kwargs = client.blueprints.create.await_args.kwargs
    assert kwargs["name"] == name
    assert kwargs["dockerfile"] == "FROM python:3.12\n"
    # Idempotency key dedupes SDK retries and concurrent samples.
    assert kwargs["idempotency_key"] == name


@pytest.mark.asyncio
async def test_list_filtered_by_name(tmp_path: Any) -> None:
    """Blueprint lookup must filter by name (not status).

    We bucket statuses client-side so we can also wait on in-flight builds,
    not just ``build_complete``.
    """
    df = tmp_path / "Dockerfile"
    df.write_text("FROM python:3.12\n")
    client = _make_client(list_items=[])

    with _patch_httpx_put():
        await build_blueprint_for_dockerfile(client, str(df))

    client.blueprints.list.assert_called_once()
    list_kwargs = client.blueprints.list.call_args.kwargs
    assert list_kwargs["name"].startswith(BLUEPRINT_NAME_PREFIX)
    assert "status" not in list_kwargs


@pytest.mark.asyncio
async def test_dockerfile_build_waits_on_in_flight(tmp_path: Any) -> None:
    """A `provisioning` blueprint with our hash should be waited on, not duplicated."""
    df = tmp_path / "Dockerfile"
    df.write_text("FROM python:3.12\n")
    in_flight = MagicMock()
    in_flight.id = "bp_in_flight"
    in_flight.status = "provisioning"
    client = _make_client(list_items=[in_flight])

    name = await build_blueprint_for_dockerfile(client, str(df))

    assert name.startswith(BLUEPRINT_NAME_PREFIX)
    client.blueprints.await_build_complete.assert_awaited_once()
    awaited_id = client.blueprints.await_build_complete.await_args.args[0]
    assert awaited_id == "bp_in_flight"
    client.blueprints.create.assert_not_called()


@pytest.mark.asyncio
async def test_create_disables_sdk_retries(tmp_path: Any) -> None:
    """Verify we disable SDK retries on the create POST.

    The POST is not idempotent on Runloop's side, so auto-retries spawn
    duplicate blueprints (and blow the account cap). We override to
    ``max_retries=0`` for this one call.
    """
    df = tmp_path / "Dockerfile"
    df.write_text("FROM python:3.12\n")
    client = _make_client(list_items=[])

    with _patch_httpx_put():
        await build_blueprint_for_dockerfile(client, str(df))

    client.with_options.assert_called_once_with(max_retries=0)
    client.blueprints.create.assert_awaited_once()
