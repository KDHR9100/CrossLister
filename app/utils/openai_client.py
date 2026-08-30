"""Process-wide cache of OpenAI-compatible HTTP clients.

``LLMClient`` and ``VisionClient`` are instantiated per pipeline-node call, so
caching the underlying :class:`AsyncOpenAI` at instance level would rebuild the
HTTP connection pool (and re-run the TLS handshake) for every request. This
module owns the real cache, keyed by endpoint settings, so every caller that
points at the same endpoint reuses one client and its connection pool.

The client deliberately disables proxy inheritance (``trust_env=False``): model
requests must connect direct — local clash-style proxies drop large/slow
payloads mid-response.
"""

from __future__ import annotations

import threading

# Cache key: (base_url, api_key, timeout). A change in any of these yields a
# separate client, so callers with different settings never share a pool.
_CACHE: dict[tuple[str, str, float], object] = {}
_LOCK = threading.Lock()


def get_openai_client(*, base_url: str, api_key: str, timeout_s: float):
    """Return the cached AsyncOpenAI client for the given endpoint settings.

    Builds the client on first use; concurrent callers are guarded by a lock so
    only one initialisation happens.
    """
    key = (base_url, api_key or "EMPTY", float(timeout_s))
    cached = _CACHE.get(key)
    if cached is not None:
        return cached

    with _LOCK:
        # Re-check inside the lock (another thread may have built it).
        cached = _CACHE.get(key)
        if cached is not None:
            return cached

        import openai._base_client as _bc
        from openai import AsyncOpenAI

        # The SDK's own httpx module (may be the renamed httpx2 build); reusing
        # it guarantees http_client type compatibility.
        _httpx = getattr(_bc, "httpx", None) or getattr(_bc, "httpx2")
        client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key or "EMPTY",
            timeout=timeout_s,
            http_client=_httpx.AsyncClient(trust_env=False),
        )
        _CACHE[key] = client
        return client


def reset_openai_clients() -> None:
    """Drop every cached client.

    Intended for test isolation so a stale cached client never leaks between
    test cases. In-flight requests keep their reference to the old client;
    new calls simply build a fresh one.
    """
    with _LOCK:
        _CACHE.clear()
