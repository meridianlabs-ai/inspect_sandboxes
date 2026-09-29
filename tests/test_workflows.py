"""Tests for the permissions of .github/workflows/build.yaml.

An agent may land build and dependency configuration (pyproject.toml,
uv.lock: the tier-2 opt-in, meridianlabs-ai/agents
design/executed-paths-residual.md), and Build runs it on every same-repo PR.
So no Build job that runs anything from the checkout holds a write
permission or leaves the job token in .git/config; the coverage writes run
where nothing from the checkout does; and the required check names stay as
the `main` ruleset lists them. The main-only `coverage` job holds a write
token, so it reads no coverage configuration from the checkout (which could
name a plugin to import) and takes nothing from py-test's artifact but a
single regular `.coverage` file (which could otherwise carry `.git/config`).

The Claude stubs opt in to tier 2 on every job that calls a reusable workflow
declaring `allow_build_config`; the reviewer's takes no such input.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).parent.parent
WORKFLOWS = ROOT / ".github" / "workflows"
BUILD = WORKFLOWS / "build.yaml"
COVERAGE_ACTION = "py-cov-action/python-coverage-comment-action@"
TAKE_STEP = "Take only the coverage data file"

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
    """Whether a job runs anything the checkout or a producer job controls.

    A `run` step counts unless it is the artifact filter, whose behaviour the
    tests below pin; the coverage action counts unless coverage reads no
    configuration (the checkout's could name a plugin to import).
    """
    for step in job.get("steps", []):
        uses = str(step.get("uses", ""))
        if uses.startswith("./"):
            return True
        if "run" in step and step.get("name") != TAKE_STEP:
            return True
        if uses.startswith(COVERAGE_ACTION) and (
            step.get("env", {}).get("COVERAGE_RCFILE") != "/dev/null"
        ):
            return True
    return False


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
    action, on a file taken from outside the checkout.
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
    assert [s.get("uses", s.get("name")).split("@")[0] for s in coverage["steps"]] == [
        "actions/checkout",
        "actions/download-artifact",
        TAKE_STEP,
        "py-cov-action/python-coverage-comment-action",
    ]
    download = coverage["steps"][1]["with"]
    assert download["name"] == py_test["Store the coverage data"]["with"]["name"]
    assert download["path"] == "${{ runner.temp }}/coverage-data"
    assert coverage["steps"][2]["env"]["ARTIFACT"] == download["path"]
    assert coverage["steps"][3]["env"] == {"COVERAGE_RCFILE": "/dev/null"}

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


def _coverage_step() -> dict[str, Any]:
    (step,) = [
        s
        for s in _load(BUILD)["jobs"]["coverage"]["steps"]
        if str(s.get("uses", "")).startswith(COVERAGE_ACTION)
    ]
    return step


HOSTILE_PLUGIN = """
import os
import pathlib

pathlib.Path(os.environ["PROBE_MARKER"]).write_text("imported")


def coverage_init(reg, options):
    pass
"""


def _hostile_checkout(tmp_path: Path) -> Path:
    """A checkout whose coverage configuration imports a planted plugin."""
    from coverage import CoverageData

    checkout = tmp_path / "checkout"
    checkout.mkdir()
    (checkout / "pyproject.toml").write_text(
        '[tool.coverage.run]\nplugins = ["review_probe"]\n'
    )
    (checkout / "review_probe.py").write_text(HOSTILE_PLUGIN)
    (checkout / "mod.py").write_text("x = 1\n")
    data = CoverageData(basename=str(checkout / ".coverage"))
    data.add_lines({"mod.py": [1]})
    data.write()
    return checkout


# The coverage commands the action runs in the checkout (v3's coverage.py).
ACTION_COVERAGE_COMMANDS = [
    ["json", "-o", "-"],
    ["html", "--skip-empty", "--directory", "htmlcov"],
    ["report", "--format=markdown", "--show-missing"],
]


def _coverage(
    checkout: Path, args: list[str], env: dict[str, str]
) -> subprocess.CompletedProcess[str]:
    # The console script, as the action calls it: `coverage ...` in the checkout.
    return subprocess.run(
        [str(Path(sys.executable).parent / "coverage"), *args],
        cwd=checkout,
        env={**os.environ, **env},
        capture_output=True,
        text=True,
    )


def test_a_planted_coverage_plugin_would_load_without_the_override(
    tmp_path: Path,
) -> None:
    """The control: the checkout's configuration alone imports the plugin."""
    checkout = _hostile_checkout(tmp_path)
    marker = tmp_path / "marker"

    result = _coverage(checkout, ["json", "-o", "-"], {"PROBE_MARKER": str(marker)})

    assert result.returncode == 0, result.stderr
    assert marker.exists()


@pytest.mark.parametrize("args", ACTION_COVERAGE_COMMANDS, ids=lambda a: a[0])
def test_the_coverage_writer_imports_no_planted_plugin(
    tmp_path: Path, args: list[str]
) -> None:
    checkout = _hostile_checkout(tmp_path)
    marker = tmp_path / "marker"
    env = {**_coverage_step()["env"], "PROBE_MARKER": str(marker)}

    result = _coverage(checkout, args, env)

    assert result.returncode == 0, result.stderr
    assert not marker.exists()
    if args[0] == "json":
        assert "mod.py" in result.stdout


def _take(
    tmp_path: Path, build_artifact: Any
) -> tuple[subprocess.CompletedProcess[str], Path]:
    steps = {s.get("name"): s for s in _load(BUILD)["jobs"]["coverage"]["steps"]}
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    workspace = tmp_path / "workspace"
    (workspace / ".git").mkdir(parents=True)
    (workspace / ".git" / "config").write_text("[core]\n")
    build_artifact(artifact, workspace)
    result = subprocess.run(
        ["bash", "-e", "-c", steps[TAKE_STEP]["run"]],
        cwd=workspace,
        env={**os.environ, "ARTIFACT": str(artifact)},
        capture_output=True,
        text=True,
    )
    return result, workspace


def test_the_coverage_writer_takes_a_lone_coverage_file(tmp_path: Path) -> None:
    def build(artifact: Path, workspace: Path) -> None:
        (artifact / ".coverage").write_bytes(b"SQLite format 3\x00data")

    result, workspace = _take(tmp_path, build)

    assert result.returncode == 0, result.stdout + result.stderr
    assert (workspace / ".coverage").read_bytes() == b"SQLite format 3\x00data"
    assert (workspace / ".git" / "config").read_text() == "[core]\n"


def test_the_coverage_writer_replaces_a_link_in_the_checkout(tmp_path: Path) -> None:
    def build(artifact: Path, workspace: Path) -> None:
        (artifact / ".coverage").write_bytes(b"data")
        (workspace / ".coverage").symlink_to(".git/config")

    result, workspace = _take(tmp_path, build)

    assert result.returncode == 0, result.stdout + result.stderr
    assert not (workspace / ".coverage").is_symlink()
    assert (workspace / ".coverage").read_bytes() == b"data"
    assert (workspace / ".git" / "config").read_text() == "[core]\n"


def _extra_git_config(artifact: Path, workspace: Path) -> None:
    (artifact / ".coverage").write_bytes(b"data")
    (artifact / ".git").mkdir()
    (artifact / ".git" / "config").write_text(
        "[core]\n\tfsmonitor = sh ./review-monitor.sh\n"
    )
    (artifact / "review-monitor.sh").write_text("touch pwned\n")


def _extra_file(artifact: Path, workspace: Path) -> None:
    (artifact / ".coverage").write_bytes(b"data")
    (artifact / "review-monitor.sh").write_text("touch pwned\n")


def _directory(artifact: Path, workspace: Path) -> None:
    (artifact / ".coverage").mkdir()
    (artifact / ".coverage" / "config").write_text("[core]\n")


def _link(artifact: Path, workspace: Path) -> None:
    (artifact / "data").write_bytes(b"data")
    (artifact / ".coverage").symlink_to("data")


def _link_alone(artifact: Path, workspace: Path) -> None:
    (artifact / ".coverage").symlink_to(workspace / ".git" / "config")


def _empty(artifact: Path, workspace: Path) -> None:
    pass


def _other_name(artifact: Path, workspace: Path) -> None:
    (artifact / "coverage.xml").write_text("<coverage/>")


@pytest.mark.parametrize(
    "build",
    [
        _extra_git_config,
        _extra_file,
        _directory,
        _link,
        _link_alone,
        _empty,
        _other_name,
    ],
    ids=lambda f: f.__name__.lstrip("_"),
)
def test_the_coverage_writer_refuses_any_other_artifact(
    tmp_path: Path, build: Any
) -> None:
    result, workspace = _take(tmp_path, build)

    assert result.returncode == 1
    assert "coverage-data must hold only a regular .coverage file" in result.stdout
    assert not (workspace / ".coverage").exists()
    assert not (workspace / "review-monitor.sh").exists()
    assert (workspace / ".git" / "config").read_text() == "[core]\n"


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
