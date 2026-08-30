"""Shared pytest fixtures: force offline mock modes for the whole test run."""

from __future__ import annotations

import pytest

from app.config import EmbeddingMode, LLMMode, VisionMode, get_settings


@pytest.fixture(scope="session", autouse=True)
def force_mock_modes(tmp_path_factory):
    """Ensure every test runs in mock mode regardless of a local .env file.

    The cached Settings instance is mutated in place so that modules which
    imported ``get_settings`` directly all observe the same values.
    """
    settings = get_settings()
    settings.vision_mode = VisionMode.MOCK
    settings.llm_mode = LLMMode.MOCK
    settings.embedding_mode = EmbeddingMode.MOCK
    # The optional API-key gate must never depend on a local .env file.
    settings.auth_api_key = ""
    # Keep test generations out of the real cold storage; history-specific
    # tests re-enable it against a throwaway directory.
    settings.history_enabled = False
    # Never let the mock embedder (384-dim) rebuild over the production
    # vector index (1024-dim) — that causes dimension-mismatch errors in the
    # live service. Tests build their own throwaway index instead.
    settings.chroma_persist_dir = tmp_path_factory.mktemp("chroma")
    yield
