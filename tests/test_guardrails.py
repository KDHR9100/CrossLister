"""Tests for the LLM compliance checker's parse-failure handling (T4).

The checker's verdict gates the regeneration loop: an unparseable reply must
be retried once, and if it still cannot be parsed the review is skipped
(fail-open) with an explicit warning instead of silently passing.
"""

from __future__ import annotations

import asyncio

from app.config import LLMMode, Settings
from app.guardrails import llm_checker


class _FakeLLMClient:
    """Test double returning queued replies and counting calls."""

    def __init__(self, replies: list[str]) -> None:
        self._replies = list(replies)
        self.calls = 0

    async def chat(
        self, system: str, user: str, temperature: float = 0.0, response_format=None
    ) -> str:
        self.calls += 1
        return self._replies.pop(0)


def _api_settings() -> Settings:
    return Settings(
        llm_mode=LLMMode.API,
        llm_api_base="https://llm.example.test/v1",
        llm_api_key="key-1",
    )


def test_checker_parses_valid_reply_single_call(monkeypatch) -> None:
    fake = _FakeLLMClient(
        ['{"passed": false, "violations": ["banned word"], "suggestions": ["remove it"]}']
    )
    monkeypatch.setattr(llm_checker, "LLMClient", lambda s: fake)

    result = asyncio.run(
        llm_checker.check_listing(
            "amazon", "Some title", ["b"], "desc", settings=_api_settings()
        )
    )

    assert result["passed"] is False
    assert result["violations"] == ["banned word"]
    assert result["suggestions"] == ["remove it"]
    assert not result.get("warnings")
    assert fake.calls == 1


def test_checker_retries_once_on_unparseable_reply(monkeypatch) -> None:
    fake = _FakeLLMClient(["sorry, I cannot...", '{"passed": true}'])
    monkeypatch.setattr(llm_checker, "LLMClient", lambda s: fake)

    result = asyncio.run(
        llm_checker.check_listing(
            "amazon", "Some title", ["b"], "desc", settings=_api_settings()
        )
    )

    assert result["passed"] is True
    assert not result.get("violations")
    assert not result.get("warnings")
    assert fake.calls == 2


def test_checker_fails_open_with_warning_after_two_failures(monkeypatch) -> None:
    fake = _FakeLLMClient(["garbage one", "garbage two"])
    monkeypatch.setattr(llm_checker, "LLMClient", lambda s: fake)

    result = asyncio.run(
        llm_checker.check_listing(
            "amazon", "Some title", ["b"], "desc", settings=_api_settings()
        )
    )

    # Fail-open so a flaky endpoint cannot wedge the pipeline…
    assert result["passed"] is True
    assert result["violations"] == []
    # …but the skipped review must be visible in the report.
    assert result["warnings"]
    assert fake.calls == 2


def test_mock_mode_passes_without_warning() -> None:
    settings = Settings(llm_mode=LLMMode.MOCK)
    result = asyncio.run(
        llm_checker.check_listing(
            "amazon", "Some title", ["b"], "desc", settings=settings
        )
    )
    assert result == {"passed": True, "violations": [], "suggestions": []}
