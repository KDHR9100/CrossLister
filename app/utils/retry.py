"""Shared helpers for deciding whether a model-call error is worth retrying,
plus a common retry loop used by both the LLM and vision clients."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import TypeVar

T = TypeVar("T")


def is_retryable(exc: Exception) -> bool:
    """Return True for transient errors worth retrying.

    Connection drops, timeouts, rate limits and 5xx server errors are
    transient; 4xx client errors (bad auth, invalid request) are not.
    """
    try:
        import openai
    except ImportError:  # pragma: no cover - openai always present in prod
        return False
    if isinstance(
        exc, (openai.APIConnectionError, openai.APITimeoutError, openai.RateLimitError)
    ):
        return True
    if isinstance(exc, openai.APIStatusError):
        return getattr(exc, "status_code", 0) >= 500
    return False


async def with_retries(
    run: Callable[[], Awaitable[T]],
    *,
    max_retries: int,
    what: str,
    on_retry: Callable[[int, float, Exception], None] | None = None,
) -> T:
    """Run ``run()`` with bounded exponential backoff on transient errors.

    Args:
        run: Zero-arg callable performing one attempt.
        max_retries: Number of retries *after* the first attempt.
        what: Label used in exception context (the caller logs its own events).
        on_retry: Optional callback receiving (attempt_number, backoff_s, exc)
            right before each backoff sleep, so callers keep their own logs.

    Returns:
        Whatever ``run()`` returns on the first successful attempt.
    """
    for attempt in range(max(0, max_retries) + 1):
        try:
            return await run()
        except Exception as exc:  # noqa: BLE001 - classified by is_retryable
            if not is_retryable(exc) or attempt >= max_retries:
                raise
            backoff_s = min(2**attempt, 8)  # 1s, 2s, 4s, capped at 8s
            if on_retry is not None:
                on_retry(attempt + 1, backoff_s, exc)
            await asyncio.sleep(backoff_s)
    raise RuntimeError(f"unreachable retry loop for {what}")  # pragma: no cover
