"""Shared helpers for deciding whether a model-call error is worth retrying."""

from __future__ import annotations


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
