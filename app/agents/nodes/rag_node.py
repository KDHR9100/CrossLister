"""RAG node: retrieve the most relevant platform rules for the product."""

from __future__ import annotations

from fastapi.concurrency import run_in_threadpool

from app.agents.state import AgentState
from app.rag.retriever import RuleRetriever, build_query
from app.utils.logger import get_logger

logger = get_logger(__name__)


async def rag_node(state: AgentState) -> dict:
    """Retrieve top-k platform rule chunks for the detected product.

    The query combines the seller-declared category with the selling points
    detected by the vision model, so image-derived signals steer retrieval.

    The underlying retrieval (Chroma query + embedding) is synchronous and is
    run in a threadpool so it never blocks the async event loop.

    Args:
        state: Current pipeline state with visual_analysis populated.

    Returns:
        A partial state update containing ``retrieved_rules``.
    """
    platform = state.get("platform", "")
    category = state.get("category", "")
    analysis = state.get("visual_analysis")

    extra_terms: list[str] = []
    if analysis is not None:
        extra_terms = list(analysis.selling_points) + list(analysis.materials)

    query = build_query(category, extra_terms)
    retriever = RuleRetriever()
    rules = await run_in_threadpool(retriever.retrieve, platform=platform, query=query)

    logger.info("node.rag.done", platform=platform, query=query, rules=len(rules))
    return {"retrieved_rules": rules}
