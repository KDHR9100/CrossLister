"""Prompt templates and message builders for the vision understanding step.

The vision model is asked to inspect one or more product photos and return a
strict JSON object describing the product. The same prompt is reused by both
the local vLLM mode and any OpenAI-Vision compatible cloud endpoint.
"""

from __future__ import annotations

from typing import Any

from app.utils.json_parse import extract_json_object
from app.utils.logger import get_logger

logger = get_logger(__name__)

# Fields we expect back from the vision model. Kept in sync with
# ``app.models.listing.VisualAnalysis``.
_EXPECTED_FIELDS = (
    "detected_category",
    "colors",
    "materials",
    "selling_points",
    "scenes",
    "raw_description",
)

SYSTEM_PROMPT = (
    "You are a precise product-analysis engine for cross-border e-commerce. "
    "You inspect product photos and extract structured, factual attributes. "
    "You always reply with a single valid JSON object and nothing else: no "
    "markdown, no code fences, no commentary. Every value must be grounded in "
    "what is actually visible or clearly implied by the images; never invent "
    "features. Use concise lowercase English terms for lists."
)

_USER_INSTRUCTION = (
    "Analyze the attached product image(s) and return a JSON object with "
    "exactly these keys:\n"
    '- "detected_category": short English category of the product '
    "(e.g. 'storage organizer').\n"
    '- "colors": list of the dominant visible colors.\n'
    '- "materials": list of likely materials (e.g. "pp plastic").\n'
    '- "selling_points": list of concrete, visible selling features '
    "(e.g. 'stackable', 'dust-proof').\n"
    '- "scenes": list of usage scenes implied by the images '
    "(e.g. 'closet', 'office').\n"
    '- "raw_description": 1-3 sentence factual description of the product.\n'
)


def _context_block(
    category_hint: str, extra_info: dict[str, Any] | None
) -> str:
    """Build an optional context hint appended to the user instruction."""
    parts: list[str] = []
    if category_hint:
        parts.append(f"The seller lists this under the category: {category_hint}.")
    if extra_info:
        rendered = ", ".join(f"{k}={v}" for k, v in extra_info.items())
        parts.append(f"Seller-provided context: {rendered}.")
    if not parts:
        return ""
    return "\nContext (use only to disambiguate, images take precedence):\n" + " ".join(
        parts
    )


def build_vision_messages(
    images_b64: list[str],
    category_hint: str = "",
    extra_info: dict[str, Any] | None = None,
    mime_types: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Assemble an OpenAI-Vision style chat message list.

    Args:
        images_b64: Base64-encoded image payloads.
        category_hint: Optional seller-declared category for disambiguation.
        extra_info: Optional seller-provided attributes (brand/material/price).
        mime_types: MIME type per image; defaults to image/jpeg for each.

    Returns:
        A list of message dicts ready for ``chat.completions.create``.
    """
    mimes = mime_types or ["image/jpeg"] * len(images_b64)
    user_text = _USER_INSTRUCTION + _context_block(category_hint, extra_info)

    content: list[dict[str, Any]] = [{"type": "text", "text": user_text}]
    for b64, mime in zip(images_b64, mimes):
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{b64}"},
            }
        )

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]


def parse_vision_json(raw: str) -> dict[str, Any]:
    """Parse the model's raw reply into a dict, tolerating common noise.

    Handles markdown code fences, stray prose around the JSON object and
    unknown keys. Returns an empty dict when no valid JSON can be recovered.

    Args:
        raw: Raw text content returned by the vision model.
    """
    data = extract_json_object(raw)
    if data is None:
        logger.warning("vision.parse_failed", preview=(raw or "")[:300])
        return {}

    # Keep only the known fields so downstream validation never breaks.
    return {k: v for k, v in data.items() if k in _EXPECTED_FIELDS}
