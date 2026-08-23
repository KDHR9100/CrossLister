"""Translate node tests: Chinese-passthrough and mock translation branches.

Run offline (mock LLM mode enforced by conftest).
"""

from __future__ import annotations

import asyncio

from app.agents.nodes.translate_node import translate_node
from app.agents.state import AgentState


def _state(lang: str) -> AgentState:
    return {
        "target_lang": lang,
        "listing": {
            "title": "Stackable Storage Bins",
            "bullet_points": ["Durable PP plastic", "Saves space"],
            "description": "Keep your closet tidy.",
            "backend_keywords": ["storage", "bins"],
        },
    }


def test_translate_chinese_target_is_noop() -> None:
    """When the target language is Chinese the listing is reused verbatim."""
    result = asyncio.run(translate_node(_state("zh")))
    assert result["final_listing"]["title"] == "Stackable Storage Bins"
    assert result["title_zh"] == "Stackable Storage Bins"
    assert result["bullet_points_zh"] == ["Durable PP plastic", "Saves space"]
    assert result["description_zh"] == "Keep your closet tidy."


def test_translate_mock_non_chinese_adds_stub() -> None:
    """In mock mode non-Chinese targets get a deterministic Chinese stub."""
    result = asyncio.run(translate_node(_state("en")))
    # The original target-language listing is preserved.
    assert result["final_listing"]["title"] == "Stackable Storage Bins"
    # And a Chinese translation is produced alongside it.
    assert result["title_zh"].startswith("[中文翻译]")
    assert all(b.startswith("[中文翻译]") for b in result["bullet_points_zh"])
    assert result["description_zh"].startswith("[中文翻译]")


def test_translate_preserves_backend_keywords() -> None:
    result = asyncio.run(translate_node(_state("ja")))
    assert result["final_listing"]["backend_keywords"] == ["storage", "bins"]
