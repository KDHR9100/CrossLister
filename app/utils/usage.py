"""Per-request LLM token usage accounting.

A ContextVar accumulates the total tokens consumed by every model call that
belongs to a single listing-generation task. Each batch product runs in its
own asyncio Task (which copies the surrounding context), so the counters stay
isolated per product, and the single-product endpoint gets a clean count too.
"""

from __future__ import annotations

import contextvars

_total_tokens: contextvars.ContextVar[int] = contextvars.ContextVar(
    "listing_usage_total_tokens", default=0
)


def reset_usage() -> None:
    """Start a fresh accounting window for the current task."""
    _total_tokens.set(0)


def add_usage(usage) -> None:
    """Add token counts from an OpenAI-style ``usage`` object, if present.

    Args:
        usage: The ``response.usage`` attribute; may be None for providers
            that omit it, in which case this is a no-op.
    """
    if usage is None:
        return
    total = getattr(usage, "total_tokens", None)
    if total is None:
        prompt = getattr(usage, "prompt_tokens", 0) or 0
        completion = getattr(usage, "completion_tokens", 0) or 0
        total = prompt + completion
    if total:
        _total_tokens.set(_total_tokens.get() + int(total))


def get_total_tokens() -> int:
    """Return the tokens accumulated so far in the current task."""
    return _total_tokens.get()
