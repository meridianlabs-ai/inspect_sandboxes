# Contributing to Inspect Sandboxes

Thanks for contributing to Inspect Sandboxes, the collection of cloud sandbox environments for [Inspect AI](https://inspect.aisi.org.uk/).

## Development setup

This project uses [uv](https://docs.astral.sh/uv/) 0.13.4 or later and targets
Python 3.12+.

```bash
git clone https://github.com/meridianlabs-ai/inspect_sandboxes.git
cd inspect_sandboxes
make install
```

`make install` runs `uv sync` to create the virtual environment and install the
dev dependencies, then installs the pre-commit hooks. The dev tools are in the
`dev` extra. The `dev` dependency group includes that extra, so a plain
`uv sync` or `uv run` installs them. `uv sync --extra dev` does the same.

### Dependencies and the cooldown

`uv.lock` pins every Python dependency except the build toolchain. CI's lint,
type-check, test and build jobs install their dependencies from it, and fail
when `uv.lock` is out of date with `pyproject.toml`. CI's build job and the
release build use the build toolchain (build, hatchling, hatch-vcs) pinned with
hashes in `scripts/build-dist/requirements.txt`. Dependabot updates that file
weekly, and `.github/workflows/publish.yaml` gives the command to update it by
hand. When uv builds the project itself, for example for the editable install
in `uv sync`, it resolves the build backend, still within the cooldown. When
you change dependencies in `pyproject.toml`, run `uv lock` and commit both
files.

`pyproject.toml` sets `exclude-newer = "7 days"` under `[tool.uv]`, so
`uv lock`, `uv sync` and `uv pip` only pick releases that have been on PyPI for
at least seven days. A compromised release is usually found and removed within
that time, before it reaches `uv.lock`, CI or your machine. A few packages wait
6 hours instead, so Inspect Sandboxes can adopt a new release the day it ships:
`inspect-ai`, and the `e2b` and `modal` SDKs. They are listed with `"6 hours"`
in `[tool.uv.exclude-newer-package]`. Only a direct dependency (a package in
`dependencies` or the `dev` extra in `pyproject.toml`) that is a sandbox
provider SDK or an Inspect package gets this exception. Every other package,
the dependencies of these packages included, keeps the seven-day cooldown.
Thus `daytona` is not on the list: each release pins its `daytona-*` client
packages to the same version, and they wait seven days, so a 6-hour entry for
`daytona` alone has no effect. Dependabot's `uv` entry in `.github/dependabot.yml` repeats the
cooldown and the list. When you add a `"6 hours"` entry, also add the package
to `cooldown.exclude` and the `six-hour-cooldown` group there.
`tests/test_dependency_cooldown.py` checks that each entry is a direct
dependency, that dated entries (below) are less than seven days old, and that
the two files match. Whether a package is a provider SDK or an Inspect package
is up to review.

Dependabot proposes `uv.lock` updates daily, once a release has cleared its
cooldown. To adopt a release sooner, run `uv lock --upgrade-package <package>`
and commit `uv.lock`. A release still inside its cooldown can't be locked; wait
for it rather than overriding `exclude-newer`, unless it's an emergency
(below).

In an emergency, for example a security fix, you can adopt a release that is
less than seven days old with a one-time dated entry:

1. Under `[tool.uv.exclude-newer-package]` in `pyproject.toml`, add
   `<package> = "<timestamp>"`. The package must be a direct dependency. Use an
   RFC 3339 timestamp just after the upload time of the fixed release. Don't
   add a dated entry to `.github/dependabot.yml`.
2. Run `uv lock --upgrade-package <package>` and commit both files.
3. Within seven days, once the normal cooldown includes the release, delete the
   entry, run `uv lock --upgrade-package <package>`, and commit both files. A
   plain `uv lock` doesn't reliably remove the entry from `uv.lock`, and CI
   then fails. A test fails when a dated entry is seven days old or older.

Testing against the newest release of every dependency happens in the nightly
canary in [meridianlabs-ai/canaries](https://github.com/meridianlabs-ai/canaries),
so Inspect Sandboxes' own CI doesn't need to.

## Checks and tests

```bash
# Lint, format, and type-check (ruff + pyright)
make check

# Run tests (skips integration tests)
make test

# Run all tests, including integration tests
# (requires real Modal/Daytona/E2B resources)
make test-all
```

Integration tests are marked with the `integration` marker and are skipped by
default; run them explicitly with `make test-integration`.

## Docs

The docs live in `docs/` and are built with [Quarto](https://quarto.org). The
docs tools are in the `doc` dependency group, which `uv sync` and `uv run` leave
out, so ask for it:

```bash
# Preview the docs locally
uv run --group doc quarto preview docs

# Publish to GitHub Pages
uv run --group doc quarto publish gh-pages docs
```

## Commit messages and releases

We use [Conventional Commits](https://www.conventionalcommits.org/). Because we
squash-merge, **the PR title becomes the commit message** — so the title is what
matters. Format it as `<type>: <description>`.

Releases are automated with [Release Please](https://github.com/googleapis/release-please):
**don't edit `CHANGELOG.md` or bump the version by hand.** Release Please reads the
merged commit types, opens a release PR that updates the changelog and version, and
merging that PR tags the release; the publish then runs once a maintainer approves
the deployment.

Choose the type deliberately — `feat:` and `fix:` drive the version bump and
headline the release notes:

| Type | Effect |
| --- | --- |
| `feat:` | a user-facing feature — bumps the minor version |
| `fix:` | a user-facing bug fix — bumps the patch version |
| `perf:`, `revert:` | appear in the release notes (no bump on their own) |
| `docs:`, `refactor:`, `chore:`, `build:`, `ci:`, `test:`, `style:` | hidden from the release notes |

Anything that isn't a user-facing feature or fix should avoid `feat:`/`fix:` so it
stays out of the headline sections.

## Reporting issues

Found a bug or have a feature request? Please open an issue on the
[GitHub issue tracker](https://github.com/meridianlabs-ai/inspect_sandboxes/issues).
