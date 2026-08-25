"""Vision client turning product images into a structured VisualAnalysis.

Three invocation modes are selected via ``Settings.vision_mode``:
  - mock:  deterministic stub for development and tests (no network, no deps).
  - local: self-hosted vLLM server exposing an OpenAI-compatible API.
  - api:   any cloud endpoint compatible with the OpenAI Vision protocol.

The local and api modes both speak the OpenAI Vision protocol, so they share
a single code path that only differs in ``base_url`` / ``api_key`` / model.
"""

from __future__ import annotations

import asyncio
import base64
import time
from typing import Any

from app.config import Settings, VisionMode, get_settings
from app.models.listing import VisualAnalysis
from app.utils.logger import get_logger
from app.utils.retry import is_retryable
from app.utils.usage import add_usage
from app.vision.prompts import build_vision_messages, parse_vision_json

logger = get_logger(__name__)

# Image magic-byte signatures used to infer the MIME type.
_SIGNATURES: tuple[tuple[bytes, str], ...] = (
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
    (b"RIFF", "image/webp"),  # refined below: RIFF....WEBP
)


def encode_image(data: bytes) -> str:
    """Base64-encode raw image bytes for the OpenAI Vision payload."""
    return base64.b64encode(data).decode("utf-8")


def guess_mime(data: bytes) -> str:
    """Best-effort MIME detection from image magic bytes; defaults to JPEG."""
    if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return "image/webp"
    for sig, mime in _SIGNATURES:
        if data.startswith(sig):
            return mime
    return "image/jpeg"


def preprocess_image(data: bytes, max_side: int, quality: int) -> bytes:
    """Downscale and re-compress an image so the request body stays small.

    Large original photos (several MB each) base64-encoded can exceed the
    remote gateway's request-body limit, causing 413 errors or dropped
    connections. This resizes the image so its longest side is at most
    ``max_side`` pixels and re-encodes it as JPEG at ``quality``.

    Args:
        data: Raw image bytes.
        max_side: Maximum length of the longest side; 0 disables downscaling.
        quality: JPEG quality (1-95).

    Returns:
        Compressed JPEG bytes, or the original bytes if preprocessing is
        disabled or fails (we never want to break generation over this).
    """
    if max_side <= 0:
        return data
    try:
        import io

        from PIL import Image

        img = Image.open(io.BytesIO(data))
        # Normalize orientation from EXIF so resizing matches what we see.
        img = ImageOps_exif_transpose(img)
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")

        w, h = img.size
        longest = max(w, h)
        if longest > max_side:
            scale = max_side / float(longest)
            img = img.resize(
                (max(1, int(w * scale)), max(1, int(h * scale))),
                Image.LANCZOS,
            )

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality, optimize=True)
        return buf.getvalue()
    except Exception as exc:  # noqa: BLE001 - fall back to the original bytes
        logger.warning("vision.preprocess_failed", error=str(exc))
        return data


def ImageOps_exif_transpose(img):
    """Apply EXIF orientation and return the corrected image (safe wrapper)."""
    try:
        from PIL import ImageOps

        return ImageOps.exif_transpose(img)
    except Exception:  # noqa: BLE001 - older Pillow or no EXIF; keep as-is
        return img


class VisionClient:
    """High-level wrapper that analyzes product images into a VisualAnalysis."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        # Lazily-created and cached OpenAI client so the HTTP connection pool
        # is reused across calls instead of being rebuilt every request.
        self._client = None

    def _get_client(self):
        """Return a cached AsyncOpenAI client, building it on first use."""
        if self._client is None:
            from openai import AsyncOpenAI

            s = self._settings
            self._client = AsyncOpenAI(
                base_url=s.vision_api_base,
                api_key=s.vision_api_key or "EMPTY",
                timeout=s.vision_timeout_s,
            )
        return self._client

    async def analyze(
        self,
        images: list[bytes],
        category_hint: str = "",
        extra_info: dict[str, Any] | None = None,
    ) -> VisualAnalysis:
        """Run multimodal understanding over 1-5 product images.

        Args:
            images: Raw image bytes (already read from the uploaded files).
            category_hint: Seller-declared category, used to disambiguate.
            extra_info: Optional seller attributes (brand/material/price...).

        Returns:
            A populated VisualAnalysis. Falls back to an empty-but-valid
            structure carrying the raw model text if JSON parsing fails.
        """
        limit = self._settings.vision_max_images
        if len(images) > limit:
            logger.warning("vision.too_many_images", got=len(images), limit=limit)
            images = images[:limit]

        if self._settings.vision_mode == VisionMode.MOCK:
            return self._mock_analyze(category_hint)
        return await self._remote_analyze(images, category_hint, extra_info)

    # -- Mock mode -----------------------------------------------------
    def _mock_analyze(self, category_hint: str) -> VisualAnalysis:
        """Return a deterministic, realistic sample for dev and tests."""
        logger.info("vision.mock_analyze", category_hint=category_hint)
        return VisualAnalysis(
            detected_category=category_hint or "storage organizer",
            colors=["white", "transparent"],
            materials=["pp plastic"],
            selling_points=["stackable", "dust-proof", "space-saving"],
            scenes=["closet", "bedroom", "office"],
            raw_description=(
                "A set of stackable, transparent storage bins made of PP "
                "plastic, designed to organize closets, bedrooms and offices "
                "while keeping contents dust-proof."
            ),
        )

    # -- Local / API mode ----------------------------------------------
    async def _remote_analyze(
        self,
        images: list[bytes],
        category_hint: str,
        extra_info: dict[str, Any] | None,
    ) -> VisualAnalysis:
        """Call an OpenAI-Vision compatible endpoint and parse its reply."""
        s = self._settings
        client = self._get_client()

        # Downscale/compress each image so the base64 payload stays under the
        # remote gateway's request-body limit (avoids 413 / dropped uploads).
        images = [
            preprocess_image(img, s.vision_max_image_side, s.vision_jpeg_quality)
            for img in images
        ]
        images_b64 = [encode_image(img) for img in images]
        mimes = [guess_mime(img) for img in images]
        messages = build_vision_messages(images_b64, category_hint, extra_info, mimes)

        started = time.perf_counter()
        logger.info(
            "vision.request",
            mode=s.vision_mode.value,
            model=s.vision_model,
            num_images=len(images),
        )

        # Retry transient errors (connection drops, rate limits, 5xx). The
        # vision call is the first step of the pipeline, so a single dropped
        # connection here would otherwise fail the whole product immediately.
        max_retries = max(0, s.llm_max_retries)
        response = None
        for attempt in range(max_retries + 1):
            try:
                response = await client.chat.completions.create(
                    model=s.vision_model,
                    messages=messages,
                    temperature=0.1,
                )
                break
            except Exception as exc:  # noqa: BLE001 - classify below
                if not is_retryable(exc) or attempt >= max_retries:
                    logger.error("vision.request_failed", error=str(exc))
                    raise
                backoff_s = min(2 ** attempt, 8)  # 1s, 2s, 4s, capped at 8s
                logger.warning(
                    "vision.retry",
                    attempt=attempt + 1,
                    backoff_s=backoff_s,
                    error=str(exc),
                )
                await asyncio.sleep(backoff_s)

        latency_ms = int((time.perf_counter() - started) * 1000)

        raw = response.choices[0].message.content or ""
        logger.info("vision.response", latency_ms=latency_ms)
        add_usage(getattr(response, "usage", None))

        parsed = parse_vision_json(raw)
        if not parsed:
            # Parsing failed: keep the raw text so callers still get something.
            return VisualAnalysis(raw_description=raw)
        return VisualAnalysis(**parsed)
