"""Translation node: produce target-language listing + Chinese translation.

Flow: the generate node already produces content in the target language.
This node keeps that content as ``final_listing`` and additionally produces
a strict Chinese translation (``title_zh``, ``bullet_points_zh``,
``description_zh``).

In ``mock`` mode the Chinese fields are filled with deterministic stubs so
the whole pipeline runs offline.
"""

from __future__ import annotations

from app.agents.state import AgentState, GeneratedListing
from app.config import LLMMode, get_settings
from app.llm.client import LLMClient
from app.utils.json_parse import extract_json_object
from app.utils.logger import get_logger

logger = get_logger(__name__)

_SYSTEM_PROMPT = (
    "You are a professional e-commerce translator. Translate the following "
    "product listing into Chinese (简体中文). Provide a strict, accurate "
    "translation — do not add or remove claims. Keep marketing tone. "
    "Respond with ONLY a JSON object with keys: "
    "title, bullet_points (list of strings), description."
)

_USER_TEMPLATE = """Target language of the original listing: {target_lang}

Title: {title}
Bullet points:
{bullets}
Description: {description}

Translate the above into Chinese now."""


async def translate_node(state: AgentState) -> dict:
    """Produce the final listing (target lang) and its Chinese translation.

    Args:
        state: Current pipeline state with a compliance-approved listing.

    Returns:
        A partial state update containing ``final_listing``, ``title_zh``,
        ``bullet_points_zh``, ``description_zh``.
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

    # If target language is Chinese, no translation needed
    if target_lang.lower() in ("zh", "zh-cn", "zh-tw"):
        logger.info("node.translate.target_is_chinese")
        return {
            "final_listing": final_listing,
            "title_zh": final_listing["title"],
            "bullet_points_zh": list(final_listing["bullet_points"]),
            "description_zh": final_listing["description"],
        }

    # Mock mode: fill Chinese fields with deterministic stubs
    if settings.llm_mode == LLMMode.MOCK:
        zh_title = f"[中文翻译] {final_listing['title']}"
        zh_bullets = [f"[中文翻译] {b}" for b in final_listing["bullet_points"]]
        zh_desc = f"[中文翻译] {final_listing['description']}"
        logger.info("node.translate.mock_chinese", target_lang=target_lang)
        return {
            "final_listing": final_listing,
            "title_zh": zh_title,
            "bullet_points_zh": zh_bullets,
            "description_zh": zh_desc,
        }

    # API/local mode: call LLM to translate into Chinese
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
        zh_title = translated.get("title", final_listing["title"])
        zh_bullets = translated.get("bullet_points", final_listing["bullet_points"])
        zh_desc = translated.get("description", final_listing["description"])
    else:
        # Fallback: use original content if translation parsing failed
        zh_title = final_listing["title"]
        zh_bullets = list(final_listing["bullet_points"])
        zh_desc = final_listing["description"]

    logger.info("node.translate.done", target_lang=target_lang)
    return {
        "final_listing": final_listing,
        "title_zh": zh_title,
        "bullet_points_zh": zh_bullets,
        "description_zh": zh_desc,
    }


def _parse_translation_json(raw: str) -> dict | None:
    """Parse the translator reply; return None when unparseable."""
    data = extract_json_object(raw)
    if data is None:
        logger.warning("node.translate.parse_failed")
        return None
    return {
        "title": str(data.get("title", "")),
        "bullet_points": [str(b) for b in data.get("bullet_points", [])],
        "description": str(data.get("description", "")),
    }
