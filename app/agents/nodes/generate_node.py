"""Generation node: draft a platform-compliant listing with the text LLM.

In ``mock`` mode the node produces a deterministic, realistic listing so the
whole pipeline runs offline. Setting ``extra_info["force_violation"] = true``
makes the *first* draft contain a platform-banned phrase, which exercises the
compliance -> regeneration loop end to end.
"""

from __future__ import annotations

import json
from typing import Any

from app.agents.state import AgentState, GeneratedListing
from app.config import LLMMode, get_settings
from app.llm.client import LLMClient
from app.models.compliance import ComplianceResult
from app.utils.json_parse import extract_json_object
from app.utils.logger import get_logger

logger = get_logger(__name__)

_SYSTEM_PROMPT = (
    "You are a senior cross-border e-commerce copywriter. Write a product "
    "listing that strictly follows the provided platform rules. Respond with "
    "ONLY a JSON object with keys: title, bullet_points (list of 3-5 "
    "strings), description, backend_keywords (list of 5-10 short strings). "
    "Never use banned or promotional phrases listed in the rules."
)

_USER_TEMPLATE = """Product category: {category}
Target language: {target_lang}
Extra seller info: {extra_info}

Visual analysis of the product images:
{visual}

Platform rules to obey (most relevant first):
{rules}
{feedback}
Write the listing now."""

# Banned phrase injected per platform when force_violation is requested,
# mirroring the keyword tables in app/guardrails/keyword_filter.py.
_FORCE_VIOLATION_TITLE: dict[str, str] = {
    "amazon": "Best Seller Stackable Storage Bins",
    "shopee": "Cheapest Stackable Storage Bins",
    "temu": "Hot Sale Stackable Storage Bins",
}


def _format_visual(state: AgentState) -> str:
    """Render the VisualAnalysis as a compact prompt block."""
    analysis = state.get("visual_analysis")
    if analysis is None:
        return "(no visual analysis available)"
    return (
        f"Category: {analysis.detected_category}\n"
        f"Colors: {', '.join(analysis.colors)}\n"
        f"Materials: {', '.join(analysis.materials)}\n"
        f"Selling points: {', '.join(analysis.selling_points)}\n"
        f"Scenes: {', '.join(analysis.scenes)}\n"
        f"Description: {analysis.raw_description}"
    )


def _format_rules(state: AgentState) -> str:
    """Render retrieved rule chunks as a compact prompt block."""
    rules = state.get("retrieved_rules") or []
    if not rules:
        return "(no platform rules retrieved)"
    return "\n".join(f"- [{r.rule_id}] {r.text}" for r in rules)


def _format_feedback(compliance: ComplianceResult | None) -> str:
    """Render previous compliance violations as rewrite instructions."""
    if compliance is None or compliance.passed:
        return ""
    parts = [
        "",
        "Your previous draft FAILED compliance. Fix ALL of these issues:",
    ]
    parts.extend(f"- {v}" for v in compliance.violations)
    parts.extend(f"Suggestion: {s}" for s in compliance.suggestions)
    return "\n".join(parts)


async def generate_node(state: AgentState) -> dict:
    """Draft (or re-draft) the listing based on vision, RAG and feedback.

    Args:
        state: Current pipeline state.

    Returns:
        A partial state update containing ``listing``.
    """
    settings = get_settings()
    attempts = int(state.get("attempts", 0))

    if settings.llm_mode == LLMMode.MOCK:
        listing = _mock_generate(state, attempts)
    else:
        listing = await _llm_generate(state)

    logger.info("node.generate.done", attempts=attempts, title=listing["title"])
    return {"listing": listing}


def _mock_generate(state: AgentState, attempts: int) -> GeneratedListing:
    """Deterministic listing draft used in mock mode."""
    platform = state.get("platform", "")
    category = state.get("category", "product")
    extra: dict[str, Any] = state.get("extra_info") or {}
    analysis = state.get("visual_analysis")

    selling_points = (
        list(analysis.selling_points)
        if analysis is not None
        else ["durable", "practical"]
    )
    materials = list(analysis.materials) if analysis is not None else []

    title = f"Stackable Storage Organizer Bins for {category.title()}"
    if extra.get("force_violation") and attempts == 0:
        # Deliberately violate on the first draft to exercise the loop.
        title = _FORCE_VIOLATION_TITLE.get(
            platform, f"Best Seller Storage Bins for {category.title()}"
        )

    bullets = [
        f"Made of {', '.join(materials) or 'premium material'} for long-lasting use",
        *(f"{sp.title()} design saves space and stays organized" for sp in selling_points[:2]),
        "Easy to assemble, clean and stack; fits closets, bedrooms and offices",
        "Ideal for cross-border sellers: neutral packaging, no brand claims",
    ]
    description = (
        f"Keep your {category} essentials tidy with these stackable, "
        "dust-proof storage bins. "
        + " ".join(f"{sp.title()}." for sp in selling_points)
        + " Built for everyday home organization."
    )
    keywords = [
        "storage bins",
        "stackable organizer",
        category.lower(),
        *(sp.lower() for sp in selling_points[:3]),
        "home organization",
    ]
    return {
        "title": title,
        "bullet_points": bullets,
        "description": description,
        "backend_keywords": keywords,
    }


async def _llm_generate(state: AgentState) -> GeneratedListing:
    """Call the text LLM and parse its JSON listing draft."""
    client = LLMClient()
    user_prompt = _USER_TEMPLATE.format(
        category=state.get("category", ""),
        target_lang=state.get("target_lang", "en"),
        extra_info=json.dumps(state.get("extra_info") or {}, ensure_ascii=False),
        visual=_format_visual(state),
        rules=_format_rules(state),
        feedback=_format_feedback(state.get("compliance")),
    )
    raw = await client.chat(_SYSTEM_PROMPT, user_prompt, temperature=0.4)
    return _parse_listing_json(raw)


def _parse_listing_json(raw: str) -> GeneratedListing:
    """Parse the LLM reply into a GeneratedListing, tolerant of noise."""
    data = extract_json_object(raw)
    if data is None:
        text = raw.strip()
        logger.warning("node.generate.parse_failed")
        return {
            "title": text[:120] or "Untitled product",
            "bullet_points": [],
            "description": text,
            "backend_keywords": [],
        }
    return {
        "title": str(data.get("title", "")).strip() or "Untitled product",
        "bullet_points": [str(b) for b in data.get("bullet_points", [])],
        "description": str(data.get("description", "")).strip(),
        "backend_keywords": [str(k) for k in data.get("backend_keywords", [])],
    }
