# Contributing to Inspect Sandboxes

Thanks for contributing to Inspect Sandboxes, the collection of cloud sandbox environments for [Inspect AI](https://inspect.aisi.org.uk/).

## Development setup

This project uses [uv](https://docs.astral.sh/uv/) 0.13.4 or later and targets
Python 3.12+.

```bash
git clone https://github.com/meridianlabs-ai/inspect_sandboxes.git
cd inspect_sandboxes
uv sync --locked
uv run --no-sync pre-commit install
```

`uv sync --locked` creates the virtual environment with the exact versions in
`uv.lock` and installs the dev dependencies. It fails if `uv.lock` is out of
date with `pyproject.toml`. `pre-commit install` sets up the pre-commit hooks.
The dev tools are in the
`dev` extra. The `dev` dependency group includes that extra, so a plain
`uv sync` or `uv run` installs them. `uv sync --extra dev` does the same.

### Dependencies

[DEPENDENCIES.md](https://github.com/meridianlabs-ai/actions/blob/main/DEPENDENCIES.md)
in meridianlabs-ai/actions tells how the Inspect repositories install and change
dependencies. It covers the lock file, the cooldown, Dependabot, the canaries,
the release build tools, and the procedures to add or update a dependency.
Follow it. It is the same for every Inspect repository, so this file does not
repeat it.

For this repository:

- Inspect Sandboxes uses only released inspect_ai: `inspect-ai` comes from PyPI.
- The 6-hour packages are in `[tool.uv.exclude-newer-package]` in
  `pyproject.toml`. Here a direct dependency that is a sandbox provider SDK
  (`e2b`, `modal`) can also wait 6 hours.
- `daytona` is not on the 6-hour list: each release pins its `daytona-*` client
  packages to the same version, and they wait seven days, so a 6-hour entry for
  `daytona` alone has no effect. An emergency entry does not shorten the
  cooldown of a package's own dependencies. Thus it does not work for
  `daytona`, whose `daytona-*` client packages are released with it and are not
  direct dependencies.
- When uv builds the project itself, for example for the editable install in
  `uv sync`, it resolves the build backend, still within the cooldown.

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
