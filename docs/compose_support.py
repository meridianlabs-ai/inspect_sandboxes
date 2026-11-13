"""Regenerate the "Compose support" section of each provider's documentation.

Usage::

    uv run python docs/compose_support.py

The tables are rendered from the providers' ``ComposeSupport`` declarations
(``MODAL_COMPOSE_SUPPORT`` and friends), which are the single source of truth
for how each native converter treats every Compose field. The committed docs
are a rendered copy so the site stays plain Markdown;
``tests/test_util/test_compose_support.py`` fails whenever a copy is stale and
points here.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from inspect_sandboxes._util.compose_support import ComposeSupport, FieldSupport
from inspect_sandboxes.daytona._compose import DAYTONA_COMPOSE_SUPPORT
from inspect_sandboxes.e2b._compose import E2B_COMPOSE_SUPPORT
from inspect_sandboxes.modal._compose import MODAL_COMPOSE_SUPPORT

DOCS_DIR = Path(__file__).resolve().parent

PROVIDERS: dict[str, ComposeSupport] = {
    "modal": MODAL_COMPOSE_SUPPORT,
    "daytona": DAYTONA_COMPOSE_SUPPORT,
    "e2b": E2B_COMPOSE_SUPPORT,
}

SECTION_START = "## Compose support"
SECTION_END = "## Finding sandboxes"

_NOT_STARTED = (
    "The Compose `command` and `entrypoint` are never started on {provider}: the "
    "sandbox only runs what you pass to `exec()`, so a service that is expected "
    "to serve something (for example `command: python server.py` plus `ports`) "
    "will not."
)

_SINGLE_SERVICE_ONLY = (
    "For single-service Compose files the {provider} converter maps the service "
    "onto {target}, and every populated field is checked against the tables "
    "below before the sandbox is created. Multi-service files run under "
    "Docker-in-Docker, where Compose itself handles every field, so this check "
    "does not apply to them. " + _NOT_STARTED
)

INTRO: dict[str, str] = {
    "modal": (
        "The Modal converter maps one Compose service onto `modal.Sandbox.create()`. "
        "Before a sandbox is created, every populated field of the Compose file is "
        "checked against the tables below:"
    ),
    "daytona": _SINGLE_SERVICE_ONLY.format(
        provider="Daytona", target="Daytona sandbox parameters"
    ),
    "e2b": _SINGLE_SERVICE_ONLY.format(
        provider="E2B", target="an E2B template and sandbox"
    ),
}


def _table(fields: dict[str, FieldSupport]) -> str:
    rows = ["| Field | Support | Notes |", "|---|---|---|"]
    for name, field in fields.items():
        shown = name.replace("x_default", "x-default")
        rows.append(f"| `{shown}` | {field.level.value} | {field.note} |")
    body = "\n".join(rows)
    return f'::: {{tbl-colwidths="[18,12,70]"}}\n{body}\n:::'


def render_section(key: str) -> str:
    """The full ``## Compose support`` section for provider *key*."""
    support = PROVIDERS[key]
    ext = support.extension
    return f"""{SECTION_START} {{#{key}-compose-support}}

{INTRO[key]}

- **supported**: translated.
- **partial**: translated with caveats; the note says what is dropped (some conversions warn about it, some do not).
- **ignored**: dropped with a warning (once per process).
- **rejected**: `ValueError` before sandbox creation, because dropping the field would change the sandbox's identity, isolation or data.

A populated field missing from these tables (for example one added by a newer `inspect_ai`) is warned about rather than silently dropped, and unknown `{ext}` keys warn too. The `{ext}` settings themselves are listed under [Configuration](#configuration).

### Service fields

{_table(dict(support.service))}

### Top-level fields

{_table(dict(support.top_level))}

"""


def current_section(text: str) -> str | None:
    """The committed section of a doc, or None if it has none."""
    match = re.search(
        rf"{re.escape(SECTION_START)}.*?(?={re.escape(SECTION_END)})", text, re.S
    )
    return match.group(0) if match else None


def doc_path(key: str) -> Path:
    """The provider doc that carries the section for *key*."""
    return DOCS_DIR / f"{key}.qmd"


def update(key: str) -> bool:
    """Rewrite the section in the provider doc; returns whether it changed."""
    path = doc_path(key)
    text = path.read_text()
    if SECTION_END not in text:
        raise SystemExit(f"{path}: no '{SECTION_END}' heading to anchor the section")
    rendered = render_section(key)
    existing = current_section(text)
    if existing == rendered:
        return False
    if existing is not None:
        text = text.replace(existing, rendered, 1)
    else:
        text = text.replace(SECTION_END, rendered + SECTION_END, 1)
    path.write_text(text)
    return True


def main() -> int:
    """Rewrite every provider doc whose section is stale."""
    changed = [key for key in PROVIDERS if update(key)]
    for key in changed:
        print(f"updated {doc_path(key).relative_to(DOCS_DIR.parent)}")
    if not changed:
        print("docs already up to date")
    return 0


if __name__ == "__main__":
    sys.exit(main())
