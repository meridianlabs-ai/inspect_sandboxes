"""Tests for the shared Compose field-support validator and provider declarations."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
from inspect_ai.util import ComposeConfig, ComposeService
from inspect_sandboxes._util.compose_support import (
    ComposeSupport,
    Support,
    ignored,
    partial,
    rejected,
    supported,
    validate_compose_support,
)
from inspect_sandboxes.daytona._compose import DAYTONA_COMPOSE_SUPPORT
from inspect_sandboxes.e2b._compose import E2B_COMPOSE_SUPPORT
from inspect_sandboxes.modal._compose import MODAL_COMPOSE_SUPPORT

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def _reset_warn_once() -> None:
    """warn_once dedupes globally by message; clear it between tests."""
    from inspect_ai._util import logger as inspect_logger

    inspect_logger._warned.clear()


def _support(**service: Any) -> ComposeSupport:
    """A throwaway provider declaration for exercising the validator."""
    return ComposeSupport(
        provider="Testbox",
        service={"image": supported(), **service},
        top_level={
            "volumes": ignored("Top-level volumes have no Testbox mapping."),
            "networks": ignored("Top-level networks have no Testbox mapping."),
        },
        extension="x-testbox",
        extension_keys=frozenset({"timeout", "region"}),
    )


def _config(**service: Any) -> ComposeConfig:
    return ComposeConfig(
        services={"default": ComposeService(image="python:3.12", **service)}
    )


def _messages(caplog: pytest.LogCaptureFixture) -> str:
    return " ".join(record.message for record in caplog.records)


# ---------------------------------------------------------------------------
# Validator behavior
# ---------------------------------------------------------------------------


def test_rejected_field_raises_naming_path_provider_and_note() -> None:
    support = _support(volumes=rejected("Use x-testbox.volumes instead."))

    with pytest.raises(ValueError) as excinfo:
        validate_compose_support(_config(volumes=["./data:/data"]), support)

    message = str(excinfo.value)
    assert "services.default.volumes" in message
    assert "Testbox" in message
    assert "Use x-testbox.volumes instead." in message


def test_all_rejected_fields_are_reported_in_one_error() -> None:
    support = _support(volumes=rejected("no volumes"), user=rejected("no user"))

    with pytest.raises(ValueError) as excinfo:
        validate_compose_support(_config(volumes=["a:/a"], user="agent"), support)

    message = str(excinfo.value)
    assert "services.default.volumes" in message
    assert "services.default.user" in message


def test_ignored_field_warns_with_note_and_continues(
    caplog: pytest.LogCaptureFixture,
) -> None:
    support = _support(init=ignored("Testbox always runs an init."))

    with caplog.at_level("WARNING"):
        validate_compose_support(_config(init=True), support)

    messages = _messages(caplog)
    assert "services.default.init" in messages
    assert "ignored" in messages
    assert "Testbox always runs an init." in messages


def test_supported_and_partial_fields_are_silent(
    caplog: pytest.LogCaptureFixture,
) -> None:
    support = _support(ports=partial("container ports only"), command=supported())

    with caplog.at_level("WARNING"):
        validate_compose_support(
            _config(ports=["8080:80"], command=["sleep", "infinity"]), support
        )

    assert caplog.records == []


def test_unclassified_field_warns(caplog: pytest.LogCaptureFixture) -> None:
    """Default deny: a populated field the provider never classified is flagged."""
    support = _support()  # no entry for `hostname`

    with caplog.at_level("WARNING"):
        validate_compose_support(_config(hostname="box"), support)

    messages = _messages(caplog)
    assert "services.default.hostname" in messages
    assert "not classified" in messages


def test_paths_use_the_compose_key_not_the_field_name(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """`x-default` is stored as `x_default`; the report must say what the user wrote."""
    support = _support()  # no entry for x_default
    config = ComposeConfig.model_validate(
        {"services": {"default": {"image": "a", "x-default": True}}}
    )

    with caplog.at_level("WARNING"):
        validate_compose_support(config, support)

    messages = _messages(caplog)
    assert "services.default.x-default" in messages
    assert "x_default" not in messages


@pytest.mark.parametrize(
    "service",
    [
        {"volumes": None},
        {"volumes": []},
        {"environment": {}},
        {"privileged": False},
        {"command": ""},
        {"restart": "no"},  # Compose's default, spelled out
    ],
)
def test_unpopulated_values_are_not_reported(
    service: dict[str, Any], caplog: pytest.LogCaptureFixture
) -> None:
    """None, empty containers, False and explicit defaults are not real settings."""
    support = _support(
        volumes=rejected("no"),
        environment=rejected("no"),
        privileged=rejected("no"),
        command=rejected("no"),
        restart=rejected("no"),
    )

    with caplog.at_level("WARNING"):
        validate_compose_support(_config(**service), support)

    assert caplog.records == []


@pytest.mark.parametrize(
    "service",
    [
        {"cpus": 0.0},
        {"restart": "always"},
        {"deploy": {}},  # an empty nested model is still a setting
    ],
)
def test_other_values_are_populated(service: dict[str, Any]) -> None:
    support = _support(
        cpus=rejected("no"), restart=rejected("no"), deploy=rejected("no")
    )

    with pytest.raises(ValueError):
        validate_compose_support(_config(**service), support)


def test_every_service_is_validated_under_its_own_path() -> None:
    support = _support(volumes=rejected("no volumes"))
    config = ComposeConfig(
        services={
            "default": ComposeService(image="a"),
            "helper": ComposeService(image="b", volumes=["x:/x"]),
        }
    )

    with pytest.raises(ValueError, match=r"services\.helper\.volumes"):
        validate_compose_support(config, support)


def test_services_filter_restricts_validation_to_the_named_services() -> None:
    support = _support(volumes=rejected("no volumes"))
    config = ComposeConfig(
        services={
            "default": ComposeService(image="a"),
            "helper": ComposeService(image="b", volumes=["x:/x"]),
        }
    )

    validate_compose_support(config, support, services=["default"])
    with pytest.raises(ValueError, match=r"services\.helper\.volumes"):
        validate_compose_support(config, support, services=["helper"])


def test_top_level_fields_are_validated(caplog: pytest.LogCaptureFixture) -> None:
    config = ComposeConfig(
        services={"default": ComposeService(image="a")},
        volumes={"data": {}},
    )

    with caplog.at_level("WARNING"):
        validate_compose_support(config, _support())

    messages = _messages(caplog)
    assert "volumes" in messages
    assert "Top-level volumes have no Testbox mapping." in messages


def test_unknown_extension_key_warns_and_lists_known_keys(
    caplog: pytest.LogCaptureFixture,
) -> None:
    config = ComposeConfig.model_validate(
        {
            "services": {"default": {"image": "a"}},
            "x-testbox": {"timeout": 30, "regoin": "eu", "unset": None},
        }
    )

    with caplog.at_level("WARNING"):
        validate_compose_support(config, _support())

    messages = _messages(caplog)
    assert "x-testbox.regoin" in messages
    assert "region" in messages  # the known keys are listed as a hint
    assert "x-testbox.timeout" not in messages
    assert "x-testbox.unset" not in messages  # null values are not settings


def test_non_mapping_extension_block_warns(caplog: pytest.LogCaptureFixture) -> None:
    config = ComposeConfig.model_validate(
        {"services": {"default": {"image": "a"}}, "x-testbox": ["timeout"]}
    )

    with caplog.at_level("WARNING"):
        validate_compose_support(config, _support())

    assert "x-testbox must be a mapping" in _messages(caplog)


def test_service_level_extension_block_warns(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Provider settings nested under a service are never read; say so."""
    config = ComposeConfig.model_validate(
        {"services": {"default": {"image": "a", "x-testbox": {"timeout": 30}}}}
    )

    with caplog.at_level("WARNING"):
        validate_compose_support(config, _support())

    messages = _messages(caplog)
    assert "services.default.x-testbox" in messages
    assert "top level" in messages


def test_other_providers_extensions_are_not_validated(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A portable file may carry x-modal, x-daytona, ... side by side."""
    config = ComposeConfig.model_validate(
        {
            "services": {"default": {"image": "a", "x-otherbox": {"k": 1}}},
            "x-otherbox": {"anything": True},
        }
    )

    with caplog.at_level("WARNING"):
        validate_compose_support(config, _support())

    assert caplog.records == []


def test_helper_constructors_carry_level_and_note() -> None:
    assert supported().level is Support.SUPPORTED
    assert partial("p").level is Support.PARTIAL
    assert ignored("i").level is Support.IGNORED
    assert rejected("r") == rejected("r")
    assert rejected("r").note == "r"


# ---------------------------------------------------------------------------
# Provider declarations: guard against silent no-ops as the schema grows
# ---------------------------------------------------------------------------

PROVIDERS = [
    pytest.param(MODAL_COMPOSE_SUPPORT, "modal.qmd", id="modal"),
    pytest.param(DAYTONA_COMPOSE_SUPPORT, "daytona.qmd", id="daytona"),
    pytest.param(E2B_COMPOSE_SUPPORT, "e2b.qmd", id="e2b"),
]


@pytest.mark.parametrize(("support", "doc"), PROVIDERS)
def test_declaration_classifies_every_service_field(
    support: ComposeSupport, doc: str
) -> None:
    """A field added to ComposeService must be classified before it ships."""
    assert set(support.service) == set(ComposeService.model_fields)


@pytest.mark.parametrize(("support", "doc"), PROVIDERS)
def test_declaration_classifies_every_top_level_field(
    support: ComposeSupport, doc: str
) -> None:
    expected = set(ComposeConfig.model_fields) - {"services"}
    assert set(support.top_level) == expected


@pytest.mark.parametrize(("support", "doc"), PROVIDERS)
def test_non_supported_fields_explain_themselves(
    support: ComposeSupport, doc: str
) -> None:
    for name, field in {**support.service, **support.top_level}.items():
        if field.level is not Support.SUPPORTED:
            assert field.note, f"{support.provider}: {name} needs a note"


@pytest.mark.parametrize(("support", "doc"), PROVIDERS)
def test_shared_severities_are_consistent_across_providers(
    support: ComposeSupport, doc: str
) -> None:
    """Identity/isolation/data fields fail everywhere; selection always works."""
    assert support.service["image"].level is Support.SUPPORTED
    assert support.service["x_default"].level is Support.SUPPORTED
    for name in ("volumes", "cap_drop", "security_opt"):
        assert support.service[name].level is Support.REJECTED, name
    # `none` blocks network access on every provider; other values allow it.
    assert support.service["network_mode"].level is Support.PARTIAL
    assert "`none`" in support.service["network_mode"].note


def _load_docs_generator() -> Any:
    """Import docs/compose_support.py (outside the package) by path."""
    import importlib.util

    path = REPO_ROOT / "docs" / "compose_support.py"
    spec = importlib.util.spec_from_file_location("compose_support_docs", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(("support", "doc"), PROVIDERS)
def test_docs_support_section_is_generated_from_declaration(
    support: ComposeSupport, doc: str
) -> None:
    """The committed docs section must equal what the generator renders now.

    Levels, notes and prose all come from the declarations via
    docs/compose_support.py; a stale copy fails here with the command to run.
    """
    generator = _load_docs_generator()
    key = doc.removesuffix(".qmd")
    assert generator.PROVIDERS[key] is support
    text = (REPO_ROOT / "docs" / doc).read_text()
    assert generator.current_section(text) == generator.render_section(key), (
        f"docs/{doc} is out of date; run `uv run python docs/compose_support.py`"
    )


def _docs_matrix(doc: str, heading: str) -> dict[str, tuple[str, str]]:
    """Parse ``| `field` | level | note |`` rows under a ### heading of the docs."""
    text = (REPO_ROOT / "docs" / doc).read_text()
    section = text.split("## Compose support", 1)[1].split("\n## ", 1)[0]
    table = section.split(f"### {heading}", 1)[1].split("\n### ", 1)[0]
    rows = re.findall(r"^\| `([^`]+)` \| ([a-z]+) \| (.*) \|$", table, flags=re.M)
    return {field: (level, note) for field, level, note in rows}


@pytest.mark.parametrize(("support", "doc"), PROVIDERS)
def test_docs_support_matrix_lists_every_declared_field(
    support: ComposeSupport, doc: str
) -> None:
    """The rendered tables are parseable and carry one row per declared field."""
    documented = _docs_matrix(doc, "Service fields")
    declared = {
        name.replace("x_default", "x-default"): (field.level.value, field.note)
        for name, field in support.service.items()
    }
    assert documented == declared
    documented_top = _docs_matrix(doc, "Top-level fields")
    declared_top = {
        name: (field.level.value, field.note)
        for name, field in support.top_level.items()
    }
    assert documented_top == declared_top
