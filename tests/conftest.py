"""Shared pytest fixtures: force offline mock modes for the whole test run."""

from __future__ import annotations

import pytest

from app.config import EmbeddingMode, LLMMode, VisionMode, get_settings


@pytest.fixture(scope="session", autouse=True)
def force_mock_modes():
    """Ensure every test runs in mock mode regardless of a local .env file.

    The cached Settings instance is mutated in place so that modules which
    imported ``get_settings`` directly all observe the same values.
    """
    settings = get_settings()
    settings.vision_mode = VisionMode.MOCK
    settings.llm_mode = LLMMode.MOCK
    settings.embedding_mode = EmbeddingMode.MOCK
    yield
