# AGENTS.md

## Pull Requests

- Title PRs as Conventional Commits (`<type>: <description>`)—we squash-merge, so the PR title becomes the commit message that drives releases; `pr-title-lint` enforces it
- `feat:`/`fix:` are for user-facing changes only: they headline the release notes and bump the version. `perf:`/`revert:` also appear in the notes (no bump); `docs:`, `refactor:`, `chore:`, `build:`, `ci:`, `test:`, `style:` are hidden
- Body lines starting with `<type>:` are parsed as extra changelog entries—don't begin description lines with a conventional-commit prefix unless that's intended
- Never edit `CHANGELOG.md`, version numbers, or `.release-please-manifest.json`—Release Please owns them
- See [CONTRIBUTING.md](CONTRIBUTING.md) for full guidelines

## Code Review

- Before opening a non-trivial PR, run at least one code review pass in a fresh context—a reviewer that hasn't seen the authoring conversation (e.g. `/code-review` in Claude Code, or a subagent)—on a strong (frontier-class) model; small fast-tier models rarely surface real issues
- Fix or explicitly dismiss each finding before opening; prefer multiple fresh-context passes for large changes, since each tends to catch what the last missed
- This pre-open pass is separate from CI: `claude-review.yml` auto-reviews same-repo, non-draft PRs when they open (fork PRs need a maintainer to trigger it with a comment containing the review keyword)—don't count that CI run as your pass, and don't write the trigger token in PR comments unless you mean to invoke it
- Include an `### Agent review` section in every PR description: reviewer model/tool, whether it ran in a fresh context and/or on a different model from the author, pass count, and findings—how many fixed, how many dismissed with a one-line reason each
- If no review pass was run, say so. Never report a review that didn't happen—"reviewed, looks good" is worse than disclosing none

Example:

```markdown
### Agent review
- Reviewer: Claude Fable 5.1 via /code-review (fresh context), 2 passes
- Findings: 3—2 fixed, 1 dismissed (flagged a missing None check that is guarded upstream)
```
