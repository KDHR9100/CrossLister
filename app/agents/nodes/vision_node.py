"""Vision node: turn uploaded product images into a VisualAnalysis."""

from __future__ import annotations

from app.agents.state import AgentState
from app.utils.logger import get_logger
from app.vision.client import VisionClient

logger = get_logger(__name__)


async def vision_node(state: AgentState) -> dict:
    """Analyze the product images with the configured vision model.

    Args:
        state: Current pipeline state carrying images/category/extra_info.

    Returns:
        A partial state update containing ``visual_analysis``.
    """
    client = VisionClient()
    analysis = await client.analyze(
        images=state.get("images", []),
        category_hint=state.get("category", ""),
        extra_info=state.get("extra_info"),
    )
    logger.info(
        "node.vision.done",
        detected_category=analysis.detected_category,
        selling_points=len(analysis.selling_points),
    )
    return {"visual_analysis": analysis}
