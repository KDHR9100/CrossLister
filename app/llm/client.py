"""Shared OpenAI-compatible chat-completions client for the text LLM.

The generation, compliance-check and translation nodes all speak to the same
kind of endpoint (OpenAI chat-completions protocol), so they share this
single client. In ``mock`` mode no network call is made: callers are
expected to branch on :attr:`LLMClient.mode` and produce deterministic
outputs themselves, keeping the whole pipeline runnable offline.
"""

from __future__ import annotations

import time

from app.config import LLMMode, Settings, get_settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


class LLMClient:
    """Thin async wrapper around an OpenAI-compatible chat-completions API.

    Args:
        settings: Optional settings override; defaults to global settings.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

    @property
    def mode(self) -> LLMMode:
        """Currently configured LLM invocation mode."""
        return self._settings.llm_mode

    @property
    def model(self) -> str:
        """Model name reported in metadata / logs."""
        if self._settings.llm_mode == LLMMode.MOCK:
            return "mock"
        return self._settings.llm_model

    async def chat(
        self,
        system: str,
        user: str,
        temperature: float = 0.2,
    ) -> str:
        """Send a single-turn chat request and return the assistant text.

        Args:
            system: System prompt constraining the model behaviour.
            user: User prompt carrying the concrete task payload.
            temperature: Sampling temperature.

        Returns:
            The assistant message content as a string.

        Raises:
            RuntimeError: If called while the client is in mock mode;
                mock behaviour must be handled by the calling node so that
                no accidental network dependency is hidden here.
        """
        if self._settings.llm_mode == LLMMode.MOCK:
            raise RuntimeError(
                "LLMClient.chat() called in mock mode; nodes must handle the "
                "mock branch themselves."
            )

        # Lazy import so mock mode works without the `openai` package.
        from openai import AsyncOpenAI

        s = self._settings
        client = AsyncOpenAI(
            base_url=s.llm_api_base,
            api_key=s.llm_api_key or "EMPTY",
            timeout=s.llm_timeout_s,
        )

        started = time.perf_counter()
        logger.info("llm.request", model=s.llm_model)
        response = await client.chat.completions.create(
            model=s.llm_model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
        )
        latency_ms = int((time.perf_counter() - started) * 1000)
        logger.info("llm.response", latency_ms=latency_ms)
        return response.choices[0].message.content or ""
