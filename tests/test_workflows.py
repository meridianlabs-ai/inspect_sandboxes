"""Tests for the permissions of .github/workflows/build.yaml.

An agent may land build and dependency configuration (pyproject.toml,
uv.lock: the tier-2 opt-in, meridianlabs-ai/agents
design/executed-paths-residual.md), and Build runs it on every same-repo PR.
So no Build job that runs anything from the checkout holds a write
permission or leaves the job token in .git/config; the coverage writes run
where nothing from the checkout does; the required check names stay as the
`main` ruleset lists them; and py-build refuses a uv.lock that installs from
anywhere but PyPI or this repository.

The Claude stubs opt in to tier 2 on every job that calls a reusable workflow
declaring `allow_build_config`; the reviewer's takes no such input.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).parent.parent
WORKFLOWS = ROOT / ".github" / "workflows"
BUILD = WORKFLOWS / "build.yaml"
LOCK_STEP = "Check uv.lock package sources"

# The `main` ruleset's required checks from build.yaml (the other one,
# `lint / lint`, is pr-title-lint.yml's).
REQUIRED_CHECKS = {
    f"{job} ({python})"
    for job in ("py-lint", "py-type", "py-test", "py-build")
    for python in ("3.12", "3.13")
}


def _load(path: Path) -> dict[Any, Any]:
    workflow: dict[Any, Any] = yaml.safe_load(path.read_text())
    return workflow


def _runs_checkout_code(job: dict[str, Any]) -> bool:
    return any(
        "run" in step or str(step.get("uses", "")).startswith("./")
        for step in job.get("steps", [])
    )


def _writes(permissions: dict[str, str]) -> set[str]:
    return {scope for scope, level in permissions.items() if level == "write"}


def test_build_jobs_that_run_the_checkout_hold_no_write_permission() -> None:
    """No job that runs the checkout's code can write, or keeps the token."""
    workflow = _load(BUILD)
    assert workflow["permissions"] == {"contents": "read"}
    writers = {}
    for name, job in workflow["jobs"].items():
        permissions = job.get("permissions", workflow["permissions"])
        checkouts = [
            s
            for s in job["steps"]
            if str(s.get("uses", "")).startswith("actions/checkout@")
        ]
        if _runs_checkout_code(job):
            assert not _writes(permissions), name
            for checkout in checkouts:
                assert checkout["with"]["persist-credentials"] is False, name
        elif _writes(permissions):
            writers[name] = _writes(permissions)
    assert writers == {"coverage": {"contents"}}


def test_build_keeps_the_required_check_names() -> None:
    workflow = _load(BUILD)
    checks = {
        f"{name} ({python})"
        for name, job in workflow["jobs"].items()
        for python in job.get("strategy", {})
        .get("matrix", {})
        .get("python-version", [])
    }
    assert checks == REQUIRED_CHECKS
    assert set(workflow["jobs"]) == {
        "py-lint",
        "py-type",
        "py-test",
        "py-build",
        "coverage",
    }


def test_coverage_writes_run_nothing_from_the_pr() -> None:
    """py-test computes the comment read-only; the writes run no PR code.

    The workflow_run workflow posts the stored comment without a checkout,
    and main's data branch is written by a job that runs only the coverage
    action.
    """
    build = _load(BUILD)
    py_test = {s.get("id", s.get("name")): s for s in build["jobs"]["py-test"]["steps"]}
    assert py_test["coverage_comment"]["if"] == (
        "matrix.python-version == '3.12' && github.event_name == 'pull_request'"
    )
    stored = py_test["Store the coverage comment"]["with"]
    assert stored["name"] == "python-coverage-comment-action"
    assert stored["path"] == "python-coverage-comment-action.txt"

    coverage = build["jobs"]["coverage"]
    assert coverage["needs"] == "py-test"
    assert coverage["if"] == (
        "github.event_name == 'push' && github.ref == 'refs/heads/main'"
    )
    assert [s["uses"].split("@")[0] for s in coverage["steps"]] == [
        "actions/checkout",
        "actions/download-artifact",
        "py-cov-action/python-coverage-comment-action",
    ]
    assert (
        coverage["steps"][1]["with"]["name"]
        == py_test["Store the coverage data"]["with"]["name"]
    )

    comment = _load(WORKFLOWS / "coverage-comment.yml")
    # PyYAML reads the `on:` key as True.
    assert comment[True]["workflow_run"]["workflows"] == [build["name"]]
    assert comment["permissions"] == {}
    (job,) = comment["jobs"].values()
    assert "head_repository.full_name == github.repository" in job["if"]
    assert _writes(job["permissions"]) == {"pull-requests"}
    (step,) = job["steps"]
    assert step["uses"].startswith("py-cov-action/python-coverage-comment-action@")
    assert step["with"]["GITHUB_PR_RUN_ID"] == "${{ github.event.workflow_run.id }}"


def _check_lock(tmp_path: Path, lock: str) -> subprocess.CompletedProcess[str]:
    steps = {s.get("name"): s for s in _load(BUILD)["jobs"]["py-build"]["steps"]}
    step = steps[LOCK_STEP]
    assert step["shell"] == "python3 {0}"
    (tmp_path / "uv.lock").write_text(lock)
    return subprocess.run(
        [sys.executable, "-c", step["run"]],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )


def test_lock_source_check_runs_before_the_checkout_code() -> None:
    steps = _load(BUILD)["jobs"]["py-build"]["steps"]
    assert steps[0]["uses"].startswith("actions/checkout@")
    assert steps[1]["name"] == LOCK_STEP


def test_lock_source_check_passes_this_repository(tmp_path: Path) -> None:
    result = _check_lock(tmp_path, (ROOT / "uv.lock").read_text())

    assert result.returncode == 0, result.stdout + result.stderr


PYPI_PACKAGE = """
[[package]]
name = "six"
version = "1.17.0"
source = { registry = "https://pypi.org/simple" }
sdist = { url = "https://files.pythonhosted.org/packages/six-1.17.0.tar.gz", hash = "sha256:ff" }
wheels = [
    { url = "https://files.pythonhosted.org/packages/six-1.17.0-py2.py3-none-any.whl", hash = "sha256:4f" },
]
"""


@pytest.mark.parametrize(
    "package",
    [
        'source = { registry = "https://example.com/simple" }',
        'source = { git = "https://github.com/x/y?rev=main#abc" }',
        'source = { url = "https://example.com/y-1.0.tar.gz" }',
        'source = { path = "vendor/y-1.0.whl" }',
        'source = { directory = "vendor/y" }',
        'source = { editable = "vendor/y" }',
        'source = { virtual = "." }',
        "",
        'source = { registry = "https://pypi.org/simple" }\n'
        'wheels = [{ url = "https://example.com/y-1.0-py3-none-any.whl", hash = "sha256:00" }]',
        'source = { registry = "https://pypi.org/simple" }\n'
        'sdist = { path = "y-1.0.tar.gz", hash = "sha256:00" }',
    ],
)
def test_lock_source_check_refuses_other_sources(tmp_path: Path, package: str) -> None:
    lock = f'version = 1\n{PYPI_PACKAGE}\n[[package]]\nname = "y"\nversion = "1.0"\n{package}\n'

    result = _check_lock(tmp_path, lock)

    assert result.returncode == 1
    assert result.stdout.count("::error file=uv.lock::y 1.0: ") == 1, result.stdout
    assert "six" not in result.stdout


# The agents reusable workflows that declare `allow_build_config`.
OPT_IN_WORKFLOWS = {"claude.yml", "claude-auto.yml", "claude-auto-review.yml"}


def test_claude_stubs_opt_in_to_tier_2() -> None:
    opted_in = {}
    for stub in ("claude.yml", "claude-auto.yml", "claude-review.yml"):
        for name, job in _load(WORKFLOWS / stub)["jobs"].items():
            uses = str(job.get("uses", ""))
            if uses.startswith("meridianlabs-ai/agents/.github/workflows/"):
                called = uses.split("/")[-1].split("@")[0]
                setting = job.get("with", {}).get("allow_build_config")
                assert setting is (True if called in OPT_IN_WORKFLOWS else None), name
                opted_in[f"{stub}:{name}"] = called
    assert opted_in == {
        "claude.yml:claude": "claude.yml",
        "claude.yml:claude-auto": "claude.yml",
        "claude-auto.yml:ci-fix": "claude-auto.yml",
        "claude-auto.yml:review-fix": "claude-auto-review.yml",
        "claude-review.yml:review": "claude-review.yml",
    }
