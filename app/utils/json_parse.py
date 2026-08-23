"""Shared helper for recovering a JSON object from raw LLM output.

The generation, compliance-check, translation and vision nodes all ask the
model for a JSON object, but models frequently wrap it in markdown fences or
surround it with prose. This module centralises the tolerant extraction so the
same noise-handling logic isn't duplicated at every call site.
"""

from __future__ import annotations

import json
from typing import Any


def extract_json_object(raw: str) -> dict[str, Any] | None:
    """Extract the outermost JSON object from raw model text.

    Handles common noise: leading/trailing whitespace, ``` code fences
    (optionally tagged ``json``), and stray prose before/after the object.

    Args:
        raw: Raw text returned by the model.

    Returns:
        The parsed dict, or None when no valid JSON object can be recovered.
    """
    if not raw:
        return None

    text = raw.strip()

    # Strip ```json ... ``` fences if present.
    if text.startswith("```"):
        text = text.strip("`").strip()
        if text.lower().startswith("json"):
            text = text[4:].strip()

    # Isolate the outermost JSON object.
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return None

    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None

    if not isinstance(data, dict):
        return None
    return data
