# Contributing to Inspect Sandboxes

Thanks for contributing to Inspect Sandboxes, the collection of cloud sandbox environments for [Inspect AI](https://inspect.aisi.org.uk/).

## Development setup

This project uses [uv](https://docs.astral.sh/uv/) and targets Python 3.12+.

```bash
git clone https://github.com/meridianlabs-ai/inspect_sandboxes.git
cd inspect_sandboxes
make install
```

`make install` runs `uv sync` to create the virtual environment and install the
dev dependencies, then installs the pre-commit hooks.

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

## Commit messages and releases

We use [Conventional Commits](https://www.conventionalcommits.org/). Because we
squash-merge, **the PR title becomes the commit message** — so the title is what
matters. Format it as `<type>: <description>`.

Releases are automated with [Release Please](https://github.com/googleapis/release-please):
**don't edit `CHANGELOG.md` or bump the version by hand.** Release Please reads the
merged commit types, opens a release PR that updates the changelog and version, and
merging that PR tags and publishes the release.

Choose the type deliberately — only `feat:` and `fix:` appear in the release notes
and drive the version bump:

| Type | Use for |
| --- | --- |
| `feat:` | a user-facing feature |
| `fix:` | a user-facing bug fix |
| `docs:`, `refactor:`, `perf:`, `test:`, `build:`, `chore:`, `ci:` | everything else — excluded from the release notes |

Anything that isn't a user-facing feature or fix should avoid `feat:`/`fix:` so it
stays out of the release notes.

## Reporting issues

Found a bug or have a feature request? Please open an issue on the
[GitHub issue tracker](https://github.com/meridianlabs-ai/inspect_sandboxes/issues).
