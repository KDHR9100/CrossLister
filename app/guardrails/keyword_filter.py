"""Deterministic keyword-based compliance filter.

Each supported platform ships a small list of banned / risky phrases derived
from the bundled rule documents (see ``data/platform_rules``). Matching is
case-insensitive over the whole listing text. This is the fast, cheap first
line of defence; the LLM checker adds semantic review on top.
"""

from __future__ import annotations

import re

# Per-platform banned phrases. Keep in sync with data/platform_rules/*.md.
BANNED_KEYWORDS: dict[str, list[str]] = {
    "amazon": [
        "best seller",
        "best-selling",
        "#1",
        "free shipping",
        "free gift",
        "guarantee",
        "money-back",
        "fda approved",
        "cures",
        "top rated",
    ],
    "shopee": [
        "cheapest",
        "lowest price",
        "1:1",
        "aaa replica",
        "first copy",
        "miracle",
        "instant cure",
        "contact us outside",
        "whatsapp",
        "wechat",
    ],
    "temu": [
        "hot sale",
        "limited time",
        "clearance",
        "clickbait",
        "best",
        "cheapest",
        "guaranteed",
        "miracle",
        "exclusive offer",
        "vip",
    ],
}


def scan_listing(
    platform: str,
    title: str,
    bullet_points: list[str],
    description: str,
    backend_keywords: list[str] | None = None,
) -> list[str]:
    """Scan listing fields for platform-specific banned phrases.

    Args:
        platform: Canonical platform key (amazon / shopee / temu).
        title: Listing title.
        bullet_points: Listing bullet points.
        description: Listing description text.
        backend_keywords: Optional backend search keywords.

    Returns:
        A list of violation strings, one per banned phrase found. Empty when
        the listing is clean or the platform has no keyword list.
    """
    banned = BANNED_KEYWORDS.get(platform, [])
    if not banned:
        return []

    haystack = "\n".join(
        [title, description, *(backend_keywords or []), *bullet_points]
    ).lower()

    violations: list[str] = []
    for phrase in banned:
        # Word-boundary match keeps "best" from firing inside "bestow" etc.
        if re.search(rf"(?<![a-z0-9]){re.escape(phrase.lower())}(?![a-z0-9])", haystack):
            violations.append(
                f"Banned phrase '{phrase}' violates {platform} listing policy."
            )
    return violations
