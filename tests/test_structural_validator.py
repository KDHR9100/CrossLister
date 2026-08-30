"""Tests for the deterministic structural compliance validator (T5).

Each rule gets a positive (violation/warning fired) and negative (clean)
case. Mock-mode pipeline output must naturally pass, which the agent-graph
tests cover end to end.
"""

from __future__ import annotations

from app.guardrails.structural_validator import scan_structure


def _clean(platform: str) -> dict:
    """A listing that passes every structural rule on any platform."""
    return {
        "title": "Stackable Storage Organizer Bins for Closet",
        "bullet_points": [
            "Made of durable PP plastic for long-lasting use",
            "Stackable design saves space and stays organized",
        ],
        "description": "Keep your essentials tidy with these dust-proof storage bins.",
        "backend_keywords": ["storage bins", "stackable organizer"],
    }


# -- Title length ------------------------------------------------------------


def test_amazon_title_over_200_chars_violates() -> None:
    data = _clean("amazon")
    data["title"] = "x" * 201
    violations, warnings = scan_structure("amazon", **_fields(data))
    assert any("AMZ-TITLE-01" in v and "201" in v for v in violations)
    assert not warnings


def test_amazon_title_at_limit_is_clean() -> None:
    data = _clean("amazon")
    data["title"] = "x" * 200
    violations, _ = scan_structure("amazon", **_fields(data))
    assert violations == []


def test_shopee_title_over_120_chars_violates() -> None:
    data = _clean("shopee")
    data["title"] = "x" * 121
    violations, _ = scan_structure("shopee", **_fields(data))
    assert any("SPE-TITLE-01" in v for v in violations)


def test_temu_title_min_length_enforced() -> None:
    data = _clean("temu")
    data["title"] = "short title"
    violations, _ = scan_structure("temu", **_fields(data))
    assert any("TEMU-TITLE-01" in v and "at least 20" in v for v in violations)


def test_temu_title_over_120_chars_violates() -> None:
    data = _clean("temu")
    data["title"] = "x" * 121
    violations, _ = scan_structure("temu", **_fields(data))
    assert any("TEMU-TITLE-01" in v for v in violations)


# -- Amazon bullets / description / keywords ---------------------------------


def test_amazon_too_many_bullets_violates() -> None:
    data = _clean("amazon")
    data["bullet_points"] = ["short bullet"] * 6
    violations, _ = scan_structure("amazon", **_fields(data))
    assert any("AMZ-BULLET-01" in v and "at most 5" in v for v in violations)


def test_amazon_overlong_bullet_violates_but_250_plus_warns() -> None:
    data = _clean("amazon")
    data["bullet_points"] = ["b" * 501]
    violations, _ = scan_structure("amazon", **_fields(data))
    assert any("AMZ-BULLET-01" in v and "501" in v for v in violations)

    data["bullet_points"] = ["b" * 300]
    violations, warnings = scan_structure("amazon", **_fields(data))
    assert violations == []
    assert any("AMZ-BULLET-01" in w for w in warnings)


def test_amazon_long_description_violates() -> None:
    data = _clean("amazon")
    data["description"] = "d" * 2001
    violations, _ = scan_structure("amazon", **_fields(data))
    assert any("AMZ-DESC-01" in v for v in violations)


def test_amazon_keywords_over_249_bytes_violate() -> None:
    data = _clean("amazon")
    # 250 single-byte characters joined into one field.
    data["backend_keywords"] = ["k" * 250]
    violations, _ = scan_structure("amazon", **_fields(data))
    assert any("AMZ-KW-01" in v and "250 bytes" in v for v in violations)


def test_amazon_multibyte_keywords_counted_in_bytes() -> None:
    data = _clean("amazon")
    # 100 CJK characters = 300 UTF-8 bytes, though only 100 chars.
    data["backend_keywords"] = ["收" * 100]
    violations, _ = scan_structure("amazon", **_fields(data))
    assert any("AMZ-KW-01" in v for v in violations)


# -- Emoji / ALL-CAPS / decorative characters --------------------------------


def test_emoji_in_shopee_and_temu_titles_violate() -> None:
    for platform, rid in (("shopee", "SPE-TITLE-01"), ("temu", "TEMU-TITLE-02")):
        data = _clean(platform)
        data["title"] = "Great Product 🎉 Value Pack"
        violations, _ = scan_structure(platform, **_fields(data))
        assert any(rid in v for v in violations), platform


def test_all_caps_title_word_warns_but_does_not_violate() -> None:
    data = _clean("amazon")
    data["title"] = "AMAZING Storage Organizer Bins for Closet"
    violations, warnings = scan_structure("amazon", **_fields(data))
    assert violations == []
    assert any("AMAZING" in w for w in warnings)


def test_acronyms_in_title_do_not_warn() -> None:
    data = _clean("amazon")
    data["title"] = "USB LED HDMI Storage Organizer Bins for Closet"
    violations, warnings = scan_structure("amazon", **_fields(data))
    assert violations == []
    assert warnings == []


def test_amazon_decorative_characters_warn() -> None:
    data = _clean("amazon")
    data["title"] = "Amazing! Storage Organizer Bins $ Deal?"
    violations, warnings = scan_structure("amazon", **_fields(data))
    assert violations == []
    assert any("AMZ-TITLE-02" in w for w in warnings)


# -- External links / prices -------------------------------------------------


def test_url_anywhere_violates_on_every_platform() -> None:
    cases = {
        "amazon": "AMZ-DESC-01",
        "shopee": "SPE-CONTENT-01",
        "temu": "TEMU-CONTENT-02",
    }
    for platform, rid in cases.items():
        data = _clean(platform)
        data["description"] = f"Buy more at https://shop.example.com/deal today."
        violations, _ = scan_structure(platform, **_fields(data))
        assert any(rid in v for v in violations), platform


def test_bare_domain_reference_violates() -> None:
    data = _clean("amazon")
    data["description"] = "Visit www.storagepro for more sizes."
    violations, _ = scan_structure("amazon", **_fields(data))
    assert any("AMZ-DESC-01" in v for v in violations)


def test_price_mention_violates_on_temu_only() -> None:
    data = _clean("temu")
    data["description"] = "Only $9.99 for a limited pack of bins."
    violations, _ = scan_structure("temu", **_fields(data))
    assert any("TEMU-PRICE-01" in v for v in violations)

    data = _clean("amazon")
    data["description"] = "Only $9.99 for a limited pack of bins."
    violations, _ = scan_structure("amazon", **_fields(data))
    assert violations == []


def test_clean_listing_passes_everywhere() -> None:
    for platform in ("amazon", "shopee", "temu"):
        violations, warnings = scan_structure(platform, **_fields(_clean(platform)))
        assert violations == [], platform
        assert warnings == [], platform


def test_unknown_platform_is_a_noop() -> None:
    violations, warnings = scan_structure("etsy", **_fields(_clean("etsy")))
    assert violations == [] and warnings == []


# -- Compliance node integration ---------------------------------------------


def test_compliance_node_flags_structural_violation() -> None:
    """A structurally invalid draft must fail compliance and reach the loop."""
    import asyncio

    from app.agents.nodes import compliance_node

    state = {
        "platform": "amazon",
        "listing": {
            "title": "x" * 250,
            "bullet_points": ["ok bullet"],
            "description": "fine description",
            "backend_keywords": ["kw"],
        },
        "retrieved_rules": [],
    }
    result = asyncio.run(compliance_node(state))
    compliance = result["compliance"]
    assert compliance.passed is False
    assert any("AMZ-TITLE-01" in v for v in compliance.violations)


def _fields(data: dict) -> dict:
    """Adapt a listing dict to scan_structure kwargs (no backend keywords key)."""
    return {
        "title": data["title"],
        "bullet_points": data["bullet_points"],
        "description": data["description"],
        "backend_keywords": data.get("backend_keywords"),
    }
