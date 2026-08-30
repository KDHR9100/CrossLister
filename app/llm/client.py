"""Shared OpenAI-compatible chat-completions client for the text LLM.

The generation, compliance-check and translation nodes all speak to the same
kind of endpoint (OpenAI chat-completions protocol), so they share this
single client. In ``mock`` mode no network call is made: callers are
expected to branch on :attr:`LLMClient.mode` and produce deterministic
outputs themselves, keeping the whole pipeline runnable offline.
"""

from __future__ import annotations

import asyncio
import time

from app.config import LLMMode, Settings, get_settings
from app.utils.logger import get_logger
from app.utils.openai_client import get_openai_client
from app.utils.retry import is_retryable
from app.utils.usage import add_usage

logger = get_logger(__name__)


class LLMClient:
    """Thin async wrapper around an OpenAI-compatible chat-completions API.

    Args:
        settings: Optional settings override; defaults to global settings.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        # Lazily-created and cached OpenAI client so the HTTP connection pool
        # is reused across calls instead of being rebuilt every request.
        self._client = None

    def _get_client(self):
        """Return the process-wide cached AsyncOpenAI client for this endpoint.

        The actual cache lives in :mod:`app.utils.openai_client` and is keyed
        by endpoint settings, so every LLMClient instance (one is created per
        pipeline-node call) reuses the same connection pool instead of paying
        a fresh TCP/TLS handshake per request.
        """
        if self._client is None:
            s = self._settings
            self._client = get_openai_client(
                base_url=s.llm_api_base,
                api_key=s.llm_api_key,
                timeout_s=s.llm_timeout_s,
            )
        return self._client

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
        response_format: dict | None = None,
    ) -> str:
        """Send a single-turn chat request and return the assistant text.

        Args:
            system: System prompt constraining the model behaviour.
            user: User prompt carrying the concrete task payload.
            temperature: Sampling temperature.
            response_format: Optional OpenAI response_format (e.g.
                ``{"type": "json_object"}``) to force structured output;
                callers keep a tolerant parser as fallback.

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

        s = self._settings
        client = self._get_client()
        max_retries = max(0, s.llm_max_retries)

        request_kwargs: dict = {
            "model": s.llm_model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
        }
        if response_format is not None:
            request_kwargs["response_format"] = response_format
        # Upper bound on generated tokens so a runaway completion cannot burn
        # unbounded quota; 0 disables the cap.
        max_output_tokens = int(getattr(s, "llm_max_output_tokens", 0) or 0)
        if max_output_tokens > 0:
            request_kwargs["max_tokens"] = max_output_tokens

        last_exc: Exception | None = None
        for attempt in range(max_retries + 1):
            started = time.perf_counter()
            logger.info("llm.request", model=s.llm_model, attempt=attempt + 1)
            try:
                response = await client.chat.completions.create(**request_kwargs)
                latency_ms = int((time.perf_counter() - started) * 1000)
                logger.info("llm.response", latency_ms=latency_ms)
                add_usage(getattr(response, "usage", None))
                return response.choices[0].message.content or ""
            except Exception as exc:  # noqa: BLE001 - classify below
                # Only retry transient errors, and only while attempts remain.
                if not is_retryable(exc) or attempt >= max_retries:
                    logger.error("llm.request_failed", error=str(exc))
                    raise
                last_exc = exc
                backoff_s = min(2 ** attempt, 8)  # 1s, 2s, 4s, capped at 8s
                logger.warning(
                    "llm.retry",
                    attempt=attempt + 1,
                    backoff_s=backoff_s,
                    error=str(exc),
                )
                await asyncio.sleep(backoff_s)

        # Unreachable: the loop either returns or re-raises, but keeps the
        # type-checker satisfied about a guaranteed return path.
        raise last_exc  # pragma: no cover
