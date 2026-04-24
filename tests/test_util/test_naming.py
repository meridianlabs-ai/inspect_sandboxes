"""Tests for :func:`inspect_sandboxes._util.naming.make_sandbox_name`."""

from __future__ import annotations

import re
from typing import Any

import pytest
from inspect_sandboxes._util.naming import make_sandbox_name

_HEX_SUFFIX = re.compile(r"-[0-9a-f]{8}$")


def test_both_task_and_sample_id_present() -> None:
    name = make_sandbox_name("my_eval", {"__sample_id__": 42})
    assert name.startswith("inspect-my_eval-42-")
    assert _HEX_SUFFIX.search(name)


def test_only_task_name() -> None:
    name = make_sandbox_name("my_eval", {})
    assert name.startswith("inspect-my_eval-")
    assert _HEX_SUFFIX.search(name)
    # segments: "inspect" / "my_eval" / hex — exactly 2 hyphens
    assert name.count("-") == 2


def test_only_sample_id() -> None:
    name = make_sandbox_name(None, {"__sample_id__": "abc_123"})
    assert name.startswith("inspect-abc_123-")
    assert _HEX_SUFFIX.search(name)


def test_neither() -> None:
    name = make_sandbox_name(None, {})
    assert name.startswith("inspect-")
    # exactly one hyphen separating "inspect" from the hex suffix
    assert name.count("-") == 1
    assert _HEX_SUFFIX.search(name)


def test_empty_task_name_treated_as_missing() -> None:
    name = make_sandbox_name("", {"__sample_id__": 7})
    assert name.startswith("inspect-7-")


def test_all_nonalnum_task_slugs_to_empty_and_is_dropped() -> None:
    # Task name that slugifies to "" should be dropped entirely, not leave
    # double-hyphens or empty segments.
    name = make_sandbox_name("!!!", {"__sample_id__": 5})
    assert name.startswith("inspect-5-")
    assert "--" not in name


def test_sample_id_zero_preserved() -> None:
    # `0` is a legitimate sample id and must not be treated as "missing".
    name = make_sandbox_name("eval", {"__sample_id__": 0})
    assert name.startswith("inspect-eval-0-")


def test_sample_id_empty_string_dropped() -> None:
    # Empty-string slugs to "" and should be skipped rather than leaving a
    # dangling hyphen.
    name = make_sandbox_name("eval", {"__sample_id__": ""})
    assert name.startswith("inspect-eval-")
    assert "--" not in name


@pytest.mark.parametrize(
    ("raw", "expected_segment"),
    [
        ("Hello World", "hello-world"),
        ("sample 42", "sample-42"),
        ("_leading_and_trailing_", "leading_and_trailing"),  # underscores preserved
        ("my_eval", "my_eval"),  # internal underscore preserved
        ("UPPERCASE", "uppercase"),
        ("样本 42", "42"),  # non-ASCII stripped, ASCII part preserved
    ],
)
def test_slug_rules(raw: str, expected_segment: str) -> None:
    name = make_sandbox_name("t", {"__sample_id__": raw})
    assert f"-{expected_segment}-" in name


def test_slug_clamped_to_40_chars() -> None:
    long_task = "a" * 100
    name = make_sandbox_name(long_task, {})
    # Extract the task segment (between "inspect-" and the hex suffix)
    core = name[len("inspect-") :]
    task_segment = core.rsplit("-", 1)[0]
    assert len(task_segment) == 40


def test_uniqueness_across_calls() -> None:
    # Two consecutive calls with identical inputs must produce different names.
    a = make_sandbox_name("eval", {"__sample_id__": 1})
    b = make_sandbox_name("eval", {"__sample_id__": 1})
    assert a != b


def test_returns_str_not_bytes() -> None:
    assert isinstance(make_sandbox_name("eval", {"__sample_id__": 1}), str)


def test_unknown_metadata_keys_ignored() -> None:
    # Noise in the metadata dict shouldn't leak into the name.
    name = make_sandbox_name(
        "eval",
        {"__sample_id__": 1, "user_key": "private", "foo": "bar"},
    )
    assert "private" not in name
    assert "foo" not in name
    assert "user-key" not in name


def test_int_sample_id_stringified() -> None:
    name = make_sandbox_name("eval", {"__sample_id__": 12345})
    assert "-12345-" in name


def test_metadata_value_with_path_chars() -> None:
    # Slashes / dots / colons commonly appear in derived sample IDs.
    md: dict[str, Any] = {"__sample_id__": "bench/task_3.1:variant_a"}
    name = make_sandbox_name("eval", md)
    # underscores preserved; `/`, `.`, `:` collapsed to `-`.
    assert "-bench-task_3-1-variant_a-" in name
