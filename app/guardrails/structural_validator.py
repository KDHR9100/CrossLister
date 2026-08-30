"""Deterministic structural validation of generated listings.

Hard platform limits (lengths, counts, byte budgets) and forbidden content
patterns (URLs, emojis, price mentions) are enforced in pure code — zero model
cost, zero hallucination — while the LLM checker focuses on semantics.

Findings split into two severities:
  - violations: hard rule breaks. They flow into ``ComplianceResult.violations``
    and drive the generate -> compliance rewrite loop.
  - warnings: stylistic/policy-gray findings (ALL-CAPS shouting, Amazon's
    decorative characters, over-long bullets). They surface in the report but
    never trigger a rewrite, so a false positive cannot burn retry budget.

Rules mirror ``data/platform_rules/*.md`` and carry the matching rule ids.
"""

from __future__ import annotations

import re

# -- Per-platform structural limits (from data/platform_rules/*.md) ---------

# Title length: (min_chars, max_chars); None = no minimum.
TITLE_LIMITS: dict[str, tuple[int | None, int]] = {
    "amazon": (None, 200),  # AMZ-TITLE-01
    "shopee": (None, 120),  # SPE-TITLE-01
    "temu": (20, 120),  # TEMU-TITLE-01
}

AMAZON_MAX_BULLETS = 5  # AMZ-BULLET-01
AMAZON_MAX_BULLET_CHARS = 500  # AMZ-BULLET-01 (hard)
AMAZON_BULLET_SUGGESTED_CHARS = 250  # AMZ-BULLET-01 (soft target -> warning)
AMAZON_MAX_DESC_CHARS = 2000  # AMZ-DESC-01
AMAZON_MAX_KEYWORDS_BYTES = 249  # AMZ-KW-01

# Emoji / decorative-symbol ranges (Shopee & Temu reject these in titles).
_EMOJI_RE = re.compile(
    "["
    "\U0001F000-\U0001FAFF"  # pictographs / transport / activity / objects
    "\U00002600-\U000027BF"  # misc symbols + dingbats (★ ✓ ✂ …)
    "\U0001F1E6-\U0001F1FF"  # regional indicators (flag emojis)
    "\U00002B00-\U00002BFF"  # arrows / stars
    "\U0001F900-\U0001F9FF"  # supplemental symbols
    "]+"
)

# External links / other-marketplace references (all platforms reject them).
_URL_RE = re.compile(
    r"(?:https?://|www\.)\S+|[\w.-]+\.(?:com|net|org|cn|co|io|shop|store|site|online)\b",
    re.IGNORECASE,
)

# Price mentions in content (Temu: pricing is fully platform-controlled).
_PRICE_RE = re.compile(r"[$€£¥￥]|\b(?:usd|eur|gbp|cny|jpy)\b", re.IGNORECASE)

# ALL-CAPS "shouting" candidates: words of 3+ ASCII letters, fully uppercase.
_ALLCAPS_WORD_RE = re.compile(r"\b[A-Za-z]{3,}\b")

# Uppercase tokens that are legitimate product acronyms, never shouting.
_ACRONYM_ALLOWLIST = {
    "ABS", "BPA", "CE", "FCC", "HDMI", "LCD", "LED", "PET", "PP", "PVC",
    "ROHS", "SPF", "USB", "UV", "TPU", "TPE", "PE", "EVA", "NYL", "PC",
}

# Amazon titles: avoid decorative characters (AMZ-TITLE-02). Warning-level:
# "?" or "!" can appear in legitimately quoted phrases, so we flag, not fail.
_AMAZON_DECOR_RE = re.compile(r"[!*$?]")

# Rule id applied per platform when a URL is found anywhere in the copy.
_URL_RULE_IDS = {
    "amazon": "AMZ-DESC-01",
    "shopee": "SPE-CONTENT-01",
    "temu": "TEMU-CONTENT-02",
}


def scan_structure(
    platform: str,
    title: str,
    bullet_points: list[str],
    description: str,
    backend_keywords: list[str] | None = None,
) -> tuple[list[str], list[str]]:
    """Validate a listing's structure against the platform's hard limits.

    Args:
        platform: Canonical platform key (amazon / shopee / temu).
        title: Listing title.
        bullet_points: Listing bullet points.
        description: Listing description text.
        backend_keywords: Optional backend search keywords.

    Returns:
        ``(violations, warnings)`` — violations trigger a rewrite loop, warnings
        are report-only. Unknown platforms yield no structural findings.
    """
    violations: list[str] = []
    warnings: list[str] = []
    if platform not in TITLE_LIMITS:
        return violations, warnings

    _check_title(platform, title, violations, warnings)
    _check_bullets(platform, bullet_points, violations, warnings)
    _check_description(platform, description, violations)
    _check_keywords(platform, backend_keywords or [], violations)
    _check_external_links(platform, title, bullet_points, description, backend_keywords or [], violations)
    _check_price(platform, bullet_points, description, violations)

    return violations, warnings


# -- Checks ------------------------------------------------------------------


def _check_title(platform: str, title: str, violations: list[str], warnings: list[str]) -> None:
    limits = TITLE_LIMITS[platform]
    min_chars, max_chars = limits
    rule_id = {"amazon": "AMZ-TITLE-01", "shopee": "SPE-TITLE-01", "temu": "TEMU-TITLE-01"}[platform]
    n = len(title)
    if n > max_chars:
        violations.append(
            f"[{rule_id}] Title is {n} characters; {platform} allows at most {max_chars}."
        )
    elif min_chars is not None and n < min_chars:
        violations.append(
            f"[{rule_id}] Title is only {n} characters; {platform} requires at least {min_chars}."
        )

    # Emojis / decorative symbols in titles (hard reject on shopee & temu).
    if platform in ("shopee", "temu") and _EMOJI_RE.search(title):
        rid = "SPE-TITLE-01" if platform == "shopee" else "TEMU-TITLE-02"
        violations.append(f"[{rid}] Title must not contain emojis or decorative symbols.")

    # ALL-CAPS shouting: warning-only, acronyms like USB/HDMI are normal.
    if platform in ("amazon", "temu"):
        shouting = [
            w
            for w in _ALLCAPS_WORD_RE.findall(title)
            if w.isupper() and w not in _ACRONYM_ALLOWLIST
        ]
        if shouting:
            rid = "AMZ-TITLE-02" if platform == "amazon" else "TEMU-TITLE-02"
            warnings.append(
                f"[{rid}] Title contains ALL-CAPS words ({', '.join(shouting[:5])}); "
                f"avoid shouting unless part of the brand name."
            )

    # Amazon decorative characters (!, *, $, ?) — warning-level.
    if platform == "amazon" and _AMAZON_DECOR_RE.search(title):
        warnings.append(
            "[AMZ-TITLE-02] Title contains decorative characters (!, *, $, ?); "
            "Amazon recommends removing them."
        )


def _check_bullets(
    platform: str, bullet_points: list[str], violations: list[str], warnings: list[str]
) -> None:
    if platform != "amazon":
        return
    if len(bullet_points) > AMAZON_MAX_BULLETS:
        violations.append(
            f"[AMZ-BULLET-01] {len(bullet_points)} bullet points provided; "
            f"Amazon allows at most {AMAZON_MAX_BULLETS}."
        )
    for idx, bullet in enumerate(bullet_points, start=1):
        n = len(bullet)
        if n > AMAZON_MAX_BULLET_CHARS:
            violations.append(
                f"[AMZ-BULLET-01] Bullet {idx} is {n} characters; "
                f"Amazon allows at most {AMAZON_MAX_BULLET_CHARS}."
            )
        elif n > AMAZON_BULLET_SUGGESTED_CHARS:
            warnings.append(
                f"[AMZ-BULLET-01] Bullet {idx} is {n} characters; aim for under "
                f"{AMAZON_BULLET_SUGGESTED_CHARS} to avoid mobile truncation."
            )


def _check_description(platform: str, description: str, violations: list[str]) -> None:
    if platform == "amazon" and len(description) > AMAZON_MAX_DESC_CHARS:
        violations.append(
            f"[AMZ-DESC-01] Description is {len(description)} characters; "
            f"Amazon allows at most {AMAZON_MAX_DESC_CHARS}."
        )


def _check_keywords(
    platform: str, backend_keywords: list[str], violations: list[str]
) -> None:
    if platform != "amazon" or not backend_keywords:
        return
    # Amazon counts the backend search-terms field in bytes, not characters.
    total_bytes = len(" ".join(backend_keywords).encode("utf-8"))
    if total_bytes > AMAZON_MAX_KEYWORDS_BYTES:
        violations.append(
            f"[AMZ-KW-01] Backend keywords total {total_bytes} bytes; "
            f"Amazon allows at most {AMAZON_MAX_KEYWORDS_BYTES}."
        )


def _check_external_links(
    platform: str,
    title: str,
    bullet_points: list[str],
    description: str,
    backend_keywords: list[str],
    violations: list[str],
) -> None:
    haystack = "\n".join([title, description, *backend_keywords, *bullet_points])
    match = _URL_RE.search(haystack)
    if match:
        violations.append(
            f"[{_URL_RULE_IDS[platform]}] External links or website references "
            f"are not allowed (found near '{match.group(0)[:40]}')."
        )


def _check_price(
    platform: str,
    bullet_points: list[str],
    description: str,
    violations: list[str],
) -> None:
    if platform != "temu":
        return
    haystack = "\n".join([description, *bullet_points])
    match = _PRICE_RE.search(haystack)
    if match:
        violations.append(
            f"[TEMU-PRICE-01] Content must not mention prices or currency "
            f"(found '{match.group(0)}'); pricing is controlled by the platform."
        )
