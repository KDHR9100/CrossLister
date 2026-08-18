"""Translation node: localize the approved listing into the target language.

In ``mock`` mode the approved draft is passed through unchanged. In ``api``
mode the text LLM rewrites title / bullets / description into the target
language while preserving structure.
"""

from __future__ import annotations

import json

from app.agents.state import AgentState, GeneratedListing
from app.config import LLMMode, get_settings
from app.llm.client import LLMClient
from app.utils.logger import get_logger

logger = get_logger(__name__)

_SYSTEM_PROMPT = (
    "You are a professional e-commerce translator. Translate the given "
    "listing into the target language. Keep marketing tone and structure, do "
    "not add or remove claims. Respond with ONLY a JSON object with keys: "
    "title, bullet_points (list of strings), description."
)

_USER_TEMPLATE = """Target language: {target_lang}

Title: {title}
Bullet points:
{bullets}
Description: {description}
"""


async def translate_node(state: AgentState) -> dict:
    """Produce the final localized listing.

    Args:
        state: Current pipeline state with a compliance-approved listing.

    Returns:
        A partial state update containing ``final_listing``.
    """
    settings = get_settings()
    listing = state.get("listing") or {}
    target_lang = state.get("target_lang", "en")

    final_listing: GeneratedListing = {
        "title": str(listing.get("title", "")),
        "bullet_points": [str(b) for b in listing.get("bullet_points", [])],
        "description": str(listing.get("description", "")),
        "backend_keywords": [str(k) for k in listing.get("backend_keywords", [])],
    }

    if settings.llm_mode == LLMMode.MOCK or target_lang.lower() in ("en", ""):
        logger.info("node.translate.passthrough", target_lang=target_lang)
        return {"final_listing": final_listing}

    client = LLMClient()
    user_prompt = _USER_TEMPLATE.format(
        target_lang=target_lang,
        title=final_listing["title"],
        bullets="\n".join(f"- {b}" for b in final_listing["bullet_points"]),
        description=final_listing["description"],
    )
    raw = await client.chat(_SYSTEM_PROMPT, user_prompt, temperature=0.1)
    translated = _parse_translation_json(raw)
    if translated:
        final_listing["title"] = translated.get("title", final_listing["title"])
        final_listing["bullet_points"] = translated.get(
            "bullet_points", final_listing["bullet_points"]
        )
        final_listing["description"] = translated.get(
            "description", final_listing["description"]
        )

    logger.info("node.translate.done", target_lang=target_lang)
    return {"final_listing": final_listing}


def _parse_translation_json(raw: str) -> dict | None:
    """Parse the translator reply; return None when unparseable."""
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        logger.warning("node.translate.parse_failed")
        return None
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        logger.warning("node.translate.parse_failed")
        return None
    return {
        "title": str(data.get("title", "")),
        "bullet_points": [str(b) for b in data.get("bullet_points", [])],
        "description": str(data.get("description", "")),
    }
