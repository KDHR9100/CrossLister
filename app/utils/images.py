"""Shared image helpers (compression/downscaling).

Used by both the vision client (to keep remote request bodies small) and the
history store (to persist compact image copies). Kept in utils so neither
module depends on the other.
"""

from __future__ import annotations

from app.utils.logger import get_logger

logger = get_logger(__name__)


def pillow_available() -> bool:
    """Return True when Pillow is importable in the current environment.

    ``preprocess_image`` degrades gracefully (returns the original bytes) when
    Pillow is missing, which silently disables request-body compression and
    history thumbnails. Startup checks and ``/api/v1/diag`` use this helper to
    surface that condition loudly instead of letting it hide until a 413.
    """
    try:
        import PIL  # noqa: F401
        return True
    except ImportError:
        return False


def ImageOps_exif_transpose(img):
    """Apply EXIF orientation and return the corrected image (safe wrapper)."""
    try:
        from PIL import ImageOps

        return ImageOps.exif_transpose(img)
    except Exception:  # noqa: BLE001 - older Pillow or no EXIF; keep as-is
        return img


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
        logger.warning("image.preprocess_failed", error=str(exc))
        return data
