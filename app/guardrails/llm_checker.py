"""Semantic compliance check backed by the text LLM.

The keyword filter catches known banned phrases; this checker asks the LLM to
review the listing against the retrieved platform rules for subtler issues
(unsupported claims, category mismatches, tone problems). In ``mock`` mode it
always passes, keeping the pipeline fully offline.
"""

from __future__ import annotations

import json
from typing import Any

from app.config import LLMMode, Settings, get_settings
from app.llm.client import LLMClient
from app.utils.logger import get_logger

logger = get_logger(__name__)

_SYSTEM_PROMPT = (
    "You are an e-commerce listing compliance reviewer. Given a product "
    "listing and the platform rules that apply to it, decide whether the "
    "listing violates any rule. Respond with ONLY a JSON object of shape: "
    '{"passed": true|false, "violations": [str], "suggestions": [str]}'
)

_USER_TEMPLATE = """Platform: {platform}

Retrieved platform rules:
{rules}

Listing under review:
Title: {title}
Bullet points:
{bullets}
Description: {description}
"""


def _format_rules(rules: list[dict[str, str]]) -> str:
    """Render retrieved rule chunks into a compact prompt block."""
    if not rules:
        return "(no rules retrieved)"
    lines = []
    for rule in rules:
        lines.append(f"- [{rule.get('rule_id', '?')}] {rule.get('text', '')}")
    return "\n".join(lines)


async def check_listing(
    platform: str,
    title: str,
    bullet_points: list[str],
    description: str,
    retrieved_rules: list[dict[str, str]] | None = None,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Run the semantic compliance check over a generated listing.

    Args:
        platform: Canonical platform key (amazon / shopee / temu).
        title: Listing title.
        bullet_points: Listing bullet points.
        description: Listing description text.
        retrieved_rules: Rule chunks (dicts with rule_id/title/text) from RAG.
        settings: Optional settings override.

    Returns:
        A dict with keys ``passed`` (bool), ``violations`` (list[str]) and
        ``suggestions`` (list[str]).
    """
    s = settings or get_settings()

    if s.llm_mode == LLMMode.MOCK:
        logger.info("guardrails.llm_checker.mock_pass", platform=platform)
        return {"passed": True, "violations": [], "suggestions": []}

    client = LLMClient(s)
    user_prompt = _USER_TEMPLATE.format(
        platform=platform,
        rules=_format_rules(retrieved_rules or []),
        title=title,
        bullets="\n".join(f"- {b}" for b in bullet_points) or "(none)",
        description=description,
    )
    raw = await client.chat(_SYSTEM_PROMPT, user_prompt, temperature=0.0)
    return _parse_checker_json(raw)


def _parse_checker_json(raw: str) -> dict[str, Any]:
    """Parse the checker reply, tolerating fenced or noisy JSON output."""
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        # Drop a possible leading language tag line.
        if "\n" in text:
            first_line, rest = text.split("\n", 1)
            if ":" not in first_line and not first_line.startswith("{"):
                text = rest
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        logger.warning("guardrails.llm_checker.parse_failed")
        return {"passed": True, "violations": [], "suggestions": []}
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        logger.warning("guardrails.llm_checker.parse_failed")
        return {"passed": True, "violations": [], "suggestions": []}
    return {
        "passed": bool(data.get("passed", True)),
        "violations": [str(v) for v in data.get("violations", [])],
        "suggestions": [str(v) for v in data.get("suggestions", [])],
    }
