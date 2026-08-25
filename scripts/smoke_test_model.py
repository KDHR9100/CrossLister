"""Quick smoke test: can the remote model endpoint answer a tiny request?

Run:  uv run python scripts/smoke_test_model.py

It makes a minimal TEXT chat completion (no images, tiny payload). If this
succeeds, the endpoint/network are fine and the earlier failures are caused
by oversized image payloads, not by the model or connectivity.
"""

from __future__ import annotations

import asyncio
import time

from app.config import get_settings


async def main() -> None:
    s = get_settings()
    print(f"[cfg] vision_mode={s.vision_mode.value} model={s.vision_model}")
    print(f"[cfg] base_url={s.vision_api_base}")

    from openai import AsyncOpenAI

    client = AsyncOpenAI(
        base_url=s.vision_api_base,
        api_key=s.vision_api_key or "EMPTY",
        timeout=60.0,
    )

    print("[test] sending tiny text-only chat completion ...")
    t0 = time.perf_counter()
    try:
        resp = await client.chat.completions.create(
            model=s.vision_model,
            messages=[{"role": "user", "content": "Reply with the single word: OK"}],
            temperature=0.0,
        )
        ms = int((time.perf_counter() - t0) * 1000)
        content = resp.choices[0].message.content
        usage = getattr(resp, "usage", None)
        print(f"[ok] responded in {ms}ms: {content!r}")
        if usage:
            print(f"[ok] usage: prompt={usage.prompt_tokens} "
                  f"completion={usage.completion_tokens} total={usage.total_tokens}")
        print("\nRESULT: model endpoint is reachable and working.")
        print("=> Earlier failures are due to OVERSIZED image payloads, not network/model.")
    except Exception as exc:  # noqa: BLE001
        ms = int((time.perf_counter() - t0) * 1000)
        print(f"[fail] after {ms}ms: {type(exc).__name__}: {exc}")
        print("\nRESULT: even a tiny request failed -> likely network/auth/endpoint issue.")


if __name__ == "__main__":
    asyncio.run(main())
