"""Tests for the shared tolerant JSON extraction helper."""

from __future__ import annotations

from app.utils.json_parse import extract_json_object


def test_plain_json_object() -> None:
    assert extract_json_object('{"a": 1}') == {"a": 1}


def test_json_with_code_fence() -> None:
    assert extract_json_object('```json\n{"a": 1}\n```') == {"a": 1}


def test_json_with_plain_fence() -> None:
    assert extract_json_object('```\n{"a": 1}\n```') == {"a": 1}


def test_json_surrounded_by_prose() -> None:
    raw = 'Sure! Here is the result:\n{"title": "x"}\nHope that helps.'
    assert extract_json_object(raw) == {"title": "x"}


def test_nested_object_returns_outermost() -> None:
    assert extract_json_object('{"a": {"b": 2}}') == {"a": {"b": 2}}


def test_invalid_json_returns_none() -> None:
    assert extract_json_object("{not valid json}") is None


def test_no_object_returns_none() -> None:
    assert extract_json_object("just some text") is None


def test_empty_returns_none() -> None:
    assert extract_json_object("") is None
    assert extract_json_object(None) is None  # type: ignore[arg-type]


def test_non_dict_json_returns_none() -> None:
    assert extract_json_object("[1, 2, 3]") is None
