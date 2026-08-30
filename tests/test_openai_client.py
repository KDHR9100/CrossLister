"""Tests for the process-wide OpenAI client cache (T1 robustness fix).

Every pipeline node constructs its own ``LLMClient``/``VisionClient`` wrapper,
so the underlying AsyncOpenAI client must be shared process-wide instead of
being rebuilt (with a fresh connection pool) per wrapper instance.
"""

from __future__ import annotations

import asyncio

from app.config import LLMMode, Settings, VisionMode
from app.llm.client import LLMClient
from app.utils.openai_client import reset_openai_clients
from app.vision.client import VisionClient


def setup_function(_: object) -> None:
    reset_openai_clients()


def teardown_function(_: object) -> None:
    reset_openai_clients()


def _llm_settings(**overrides: object) -> Settings:
    kwargs: dict = {
        "llm_mode": LLMMode.API,
        "llm_api_base": "https://llm.example.test/v1",
        "llm_api_key": "key-1",
        "llm_timeout_s": 30.0,
        "vision_mode": VisionMode.MOCK,
    }
    kwargs.update(overrides)
    return Settings(**kwargs)


def _vision_settings(**overrides: object) -> Settings:
    kwargs: dict = {
        "vision_mode": VisionMode.API,
        "vision_api_base": "https://vision.example.test/v1",
        "vision_api_key": "key-1",
        "vision_timeout_s": 30.0,
        "llm_mode": LLMMode.MOCK,
    }
    kwargs.update(overrides)
    return Settings(**kwargs)


def test_llm_client_is_shared_across_instances() -> None:
    s = _llm_settings()
    c1 = LLMClient(s)._get_client()
    c2 = LLMClient(Settings(**s.model_dump()))._get_client()
    assert c1 is c2


def test_vision_client_is_shared_across_instances() -> None:
    s = _vision_settings()
    c1 = VisionClient(s)._get_client()
    c2 = VisionClient(Settings(**s.model_dump()))._get_client()
    assert c1 is c2


def test_llm_and_vision_share_pool_for_same_endpoint() -> None:
    endpoint = "https://shared.example.test/v1"
    llm = LLMClient(_llm_settings(llm_api_base=endpoint))._get_client()
    vision = VisionClient(_vision_settings(vision_api_base=endpoint))._get_client()
    assert llm is vision


def test_different_timeout_gets_a_separate_client() -> None:
    c1 = LLMClient(_llm_settings())._get_client()
    c2 = LLMClient(_llm_settings(llm_timeout_s=99.0))._get_client()
    assert c1 is not c2


def test_reset_forces_a_fresh_client() -> None:
    s = _llm_settings()
    c1 = LLMClient(s)._get_client()
    reset_openai_clients()
    c2 = LLMClient(s)._get_client()
    assert c1 is not c2


def test_mock_mode_wrapper_never_builds_a_client() -> None:
    """mock mode must not create any HTTP client (offline guarantee)."""
    wrapper = LLMClient(Settings(llm_mode=LLMMode.MOCK))
    assert wrapper.mode == LLMMode.MOCK
    assert wrapper._client is None


def test_shared_client_survives_across_event_loops_only_via_rebuild() -> None:
    """Sanity: the cached object is a plain singleton; callers drive it.

    A full cross-loop reuse test would need real HTTP; here we only assert
    that concurrent wrapper constructions all resolve to one client.
    """

    async def _grab() -> object:
        return LLMClient(_llm_settings())._get_client()

    async def _main() -> list[object]:
        return await asyncio.gather(_grab(), _grab(), _grab())

    results = asyncio.run(_main())
    assert results[0] is results[1] is results[2]
