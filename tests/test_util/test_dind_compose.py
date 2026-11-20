"""Tests for shared DinD compose helpers."""

from __future__ import annotations

from pathlib import Path

import pytest
from inspect_ai.util import (
    ComposeBuild,
    ComposeConfig,
    ComposeService,
    parse_compose_yaml,
)
from inspect_sandboxes._util.dind_compose import (
    compute_healthcheck_timeout,
    discover_build_contexts,
    resolve_build_contexts,
    rewrite_compose_yaml,
)


def _write(p: Path, body: str) -> Path:
    p.write_text(body)
    return p


def test_compute_healthcheck_timeout_default_when_no_healthchecks(
    tmp_path: Path,
) -> None:
    p = _write(
        tmp_path / "compose.yaml",
        """
services:
  default:
    image: alpine
""",
    )
    cfg = parse_compose_yaml(str(p), multiple_services=True)
    assert compute_healthcheck_timeout(cfg.services, default=120) == 120


def test_compute_healthcheck_timeout_picks_max_across_services(tmp_path: Path) -> None:
    p = _write(
        tmp_path / "compose.yaml",
        """
services:
  fast:
    image: alpine
    healthcheck:
      test: ["CMD", "true"]
      interval: 1s
      timeout: 1s
      retries: 2
  slow:
    image: alpine
    healthcheck:
      test: ["CMD", "true"]
      interval: 5s
      timeout: 10s
      retries: 4
""",
    )
    cfg = parse_compose_yaml(str(p), multiple_services=True)
    # slow: 4 * (5 + 10) = 60
    assert compute_healthcheck_timeout(cfg.services) == 60


def test_discover_build_contexts_inside_compose_dir_no_rewrite(tmp_path: Path) -> None:
    (tmp_path / "Dockerfile").write_text("FROM ubuntu\n")
    p = _write(
        tmp_path / "compose.yaml",
        """
services:
  default:
    build: .
""",
    )
    cfg = parse_compose_yaml(str(p), multiple_services=True)
    context_map, needs_rewrite = discover_build_contexts(
        cfg, tmp_path, "/inspect/contexts"
    )
    assert needs_rewrite is False
    assert tmp_path.resolve() == Path(next(iter(context_map.keys()))).resolve()


def test_discover_build_contexts_external_triggers_rewrite(tmp_path: Path) -> None:
    external = tmp_path / "external"
    external.mkdir()
    (external / "Dockerfile").write_text("FROM ubuntu\n")
    compose_dir = tmp_path / "project"
    compose_dir.mkdir()
    p = _write(
        compose_dir / "compose.yaml",
        f"""
services:
  default:
    build:
      context: {external.resolve()}
""",
    )
    cfg = parse_compose_yaml(str(p), multiple_services=True)
    context_map, needs_rewrite = discover_build_contexts(
        cfg, compose_dir, "/inspect/contexts"
    )
    assert needs_rewrite is True
    assert any(Path(local) == external.resolve() for local in context_map)


def test_rewrite_compose_yaml_replaces_external_context(tmp_path: Path) -> None:
    external = tmp_path / "external"
    external.mkdir()
    (external / "Dockerfile").write_text("FROM ubuntu\n")
    compose_dir = tmp_path / "project"
    compose_dir.mkdir()
    p = _write(
        compose_dir / "compose.yaml",
        f"""
services:
  default:
    build:
      context: {external.resolve()}
""",
    )
    cfg = parse_compose_yaml(str(p), multiple_services=True)
    context_map = {str(external.resolve()): "/inspect/contexts/default"}
    rewritten = rewrite_compose_yaml(cfg, compose_dir, context_map)
    assert "/inspect/contexts/default" in rewritten
    assert str(external.resolve()) not in rewritten


@pytest.mark.parametrize(
    "remote",
    [
        "https://github.com/docker/getting-started.git#main",
        "git://example.com/repo.git",
        "ssh://git@example.com/repo.git",
        "git@github.com:org/repo.git",
        "github.com/org/repo",
        "docker-image://alpine",
    ],
)
def test_resolve_build_contexts_leaves_remote_contexts_unchanged(
    tmp_path: Path, remote: str
) -> None:
    config = ComposeConfig(
        services={
            "as_string": ComposeService(build=remote),
            "as_object": ComposeService(build=ComposeBuild(context=remote)),
            "local": ComposeService(build="app"),
        }
    )
    resolved = resolve_build_contexts(config, tmp_path)
    assert resolved.services["as_string"].build == remote
    as_object = resolved.services["as_object"].build
    assert isinstance(as_object, ComposeBuild)
    assert as_object.context == remote
    assert resolved.services["local"].build == str(tmp_path / "app")
