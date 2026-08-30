"""Tests for LLMClient request construction (T6/T7): JSON mode + token cap."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from app.config import LLMMode, Settings
from app.llm.client import LLMClient


class _FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeChoice:
    def __init__(self, content: str) -> None:
        self.message = _FakeMessage(content)


class _FakeUsage:
    total_tokens = 42


class _FakeResponse:
    def __init__(self, content: str) -> None:
        self.choices = [_FakeChoice(content)]
        self.usage = _FakeUsage()


class _RecordingCompletions:
    """Captures the kwargs a chat() call passes to the SDK."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> _FakeResponse:
        self.calls.append(kwargs)
        return _FakeResponse('{"ok": true}')


class _FakeSDKClient:
    def __init__(self) -> None:
        self.chat = type("_Chat", (), {})()
        self.chat.completions = _RecordingCompletions()


def _api_settings(**overrides: object) -> Settings:
    kwargs: dict = {
        "llm_mode": LLMMode.API,
        "llm_api_base": "https://llm.example.test/v1",
        "llm_api_key": "key-1",
        "llm_max_retries": 0,
    }
    kwargs.update(overrides)
    return Settings(**kwargs)


def _client_with_recorder(settings: Settings):
    client = LLMClient(settings)
    fake = _FakeSDKClient()
    client._client = fake  # bypass the shared cache entirely
    return client, fake.chat.completions


def test_chat_forwards_response_format_and_max_tokens() -> None:
    client, recorder = _client_with_recorder(_api_settings(llm_max_output_tokens=1024))

    out = asyncio.run(
        client.chat(
            "sys", "user", temperature=0.3, response_format={"type": "json_object"}
        )
    )

    assert json.loads(out) == {"ok": True}
    kwargs = recorder.calls[0]
    assert kwargs["response_format"] == {"type": "json_object"}
    assert kwargs["max_tokens"] == 1024
    assert kwargs["temperature"] == 0.3
    assert kwargs["model"] == "qwen3.6-flash"


def test_chat_omits_max_tokens_when_disabled() -> None:
    client, recorder = _client_with_recorder(_api_settings(llm_max_output_tokens=0))

    asyncio.run(client.chat("sys", "user"))

    assert "max_tokens" not in recorder.calls[0]
    assert "response_format" not in recorder.calls[0]


def test_chat_without_response_format_sends_no_response_format() -> None:
    client, recorder = _client_with_recorder(_api_settings())

    asyncio.run(client.chat("sys", "user", response_format=None))

    assert "response_format" not in recorder.calls[0]
