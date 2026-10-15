"""Default-deny validation of Compose fields for native converters.

Providers that translate a Compose service into their own SDK parameters
(Modal, and the Daytona / E2B single-service paths) can only honor part of
the Compose schema. Rather than letting each converter silently drop whatever
it does not read, every provider declares how it treats each ``ComposeService``
and ``ComposeConfig`` field in a :class:`ComposeSupport`, and
:func:`validate_compose_support` checks a config's *populated* fields against
that declaration before any sandbox is created:

- ``SUPPORTED`` fields are translated.
- ``PARTIAL`` fields are translated with caveats; the note says what is
  dropped (some conversions warn about it, some do not).
- ``IGNORED`` fields are dropped with a ``warn_once``.
- ``REJECTED`` fields raise ``ValueError``, because dropping them would change
  the sandbox's identity, isolation or data.
- A populated field the provider has not classified at all is warned about,
  so a newly modeled Compose field can never become a silent no-op.

Docker-in-Docker paths run the Compose project as-is and must not use this.
The declarations double as the source of truth for the support matrix in each
provider's documentation (a test keeps the two in sync).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import Enum
from logging import getLogger
from typing import Any

from inspect_ai.util import ComposeConfig, ComposeService, warn_once
from pydantic import BaseModel

logger = getLogger(__name__)


class Support(str, Enum):
    """How a provider's native converter treats a populated Compose field."""

    SUPPORTED = "supported"
    """Translated faithfully."""

    PARTIAL = "partial"
    """Translated with caveats; the note says what is dropped."""

    IGNORED = "ignored"
    """Dropped with a warning: the sandbox is still what the file describes."""

    REJECTED = "rejected"
    """Raises: dropping it would change the sandbox's identity, isolation or data."""


@dataclass(frozen=True)
class FieldSupport:
    """A field's support level plus the note shown in warnings, errors and docs."""

    level: Support
    note: str = ""


def supported(note: str = "") -> FieldSupport:
    return FieldSupport(Support.SUPPORTED, note)


def partial(note: str) -> FieldSupport:
    return FieldSupport(Support.PARTIAL, note)


def ignored(note: str = "") -> FieldSupport:
    return FieldSupport(Support.IGNORED, note)


def rejected(note: str) -> FieldSupport:
    return FieldSupport(Support.REJECTED, note)


@dataclass(frozen=True)
class ComposeSupport:
    """A provider's declaration of its Compose field support.

    Attributes:
        provider: Display name used in messages (e.g. ``"Modal"``).
        service: Support for every ``ComposeService`` field, by field name.
        top_level: Support for every ``ComposeConfig`` field other than
            ``services``, by field name.
        extension: The provider's own extension block (e.g. ``"x-modal"``).
            Other providers' ``x-*`` blocks are left alone so one file can
            carry settings for several providers.
        extension_keys: The keys the provider reads from that block; any other
            populated key is warned about.
    """

    provider: str
    service: Mapping[str, FieldSupport]
    top_level: Mapping[str, FieldSupport]
    extension: str
    extension_keys: frozenset[str]


# Values a Compose file may spell out that mean "the default, do nothing"; they
# are not reported even though they are set (YAML `restart: no` parses to the
# string "no").
_EXPLICIT_DEFAULTS: dict[str, Any] = {"restart": "no"}


def validate_compose_support(
    config: ComposeConfig,
    support: ComposeSupport,
    *,
    services: Iterable[str] | None = None,
) -> None:
    """Check every populated field of *config* against *support*.

    Warns once per ignored or unclassified field and per unknown extension
    key, then raises a single ``ValueError`` listing every rejected field.

    Args:
        config: The parsed Compose configuration.
        support: The provider's declaration.
        services: Names of the services to validate; all of them by default.
            A single-service provider passes the one service it will run so
            settings on a sibling it drops anyway are not reported.

    Raises:
        ValueError: If any populated field is ``REJECTED`` by the provider.
    """
    rejections: list[str] = []
    selected = set(services) if services is not None else None

    for service_name, service in config.services.items():
        if selected is not None and service_name not in selected:
            continue
        for field_name, compose_name in _populated_fields(service):
            _check(
                f"services.{service_name}.{compose_name}",
                support.service.get(field_name),
                support.provider,
                rejections,
            )
        _check_service_extension(service_name, service, support)

    for field_name, compose_name in _populated_fields(config):
        if field_name == "services":
            continue
        _check(
            compose_name,
            support.top_level.get(field_name),
            support.provider,
            rejections,
        )

    _check_extension_block(config, support)

    if rejections:
        details = "\n".join(f"  - {rejection}" for rejection in rejections)
        raise ValueError(
            f"{support.provider} cannot honor the following Compose settings; "
            f"remove them or use a {support.extension} equivalent:\n{details}"
        )


def _check(
    path: str, field: FieldSupport | None, provider: str, rejections: list[str]
) -> None:
    if field is None:
        warn_once(
            logger,
            f"Compose field {path} is not classified for {provider} and is ignored; "
            "please report this so it can be mapped or rejected explicitly.",
        )
    elif field.level is Support.REJECTED:
        rejections.append(f"{path}: {field.note}")
    elif field.level is Support.IGNORED:
        note = f" {field.note}" if field.note else ""
        warn_once(
            logger,
            f"Compose field {path} is not supported by {provider} and is ignored.{note}",
        )


def _check_extension_block(config: ComposeConfig, support: ComposeSupport) -> None:
    block = config.extensions.get(support.extension)
    if block is None:
        return
    if not isinstance(block, dict):
        warn_once(
            logger,
            f"{support.extension} must be a mapping of settings, got "
            f"{type(block).__name__}; ignoring it.",
        )
        return
    known = ", ".join(sorted(support.extension_keys))
    for key, value in block.items():
        if key not in support.extension_keys and _is_populated(value):
            warn_once(
                logger,
                f"{support.extension}.{key} is not a recognised {support.provider} "
                f"setting and is ignored. Known {support.extension} keys: {known}.",
            )


def _check_service_extension(
    service_name: str, service: ComposeService, support: ComposeSupport
) -> None:
    if support.extension in service.extensions:
        warn_once(
            logger,
            f"services.{service_name}.{support.extension} is ignored: "
            f"{support.extension} settings belong at the top level of the Compose file.",
        )


def _is_populated(value: Any) -> bool:
    """Whether a value carries a real setting (not None, False or empty)."""
    if value is None or value is False:
        return False
    if isinstance(value, (list, dict, str)) and len(value) == 0:
        return False
    return True


def _populated_fields(model: BaseModel) -> list[tuple[str, str]]:
    """``(field_name, compose_name)`` for every populated field of *model*.

    ``compose_name`` is the key as it appears in the file (the pydantic alias,
    e.g. ``x-default``) so reported paths match what the user wrote.
    """
    populated: list[tuple[str, str]] = []
    for name, info in type(model).model_fields.items():
        value = getattr(model, name)
        if not _is_populated(value):
            continue
        if name in _EXPLICIT_DEFAULTS and value == _EXPLICIT_DEFAULTS[name]:
            continue
        populated.append((name, info.alias or name))
    return populated
