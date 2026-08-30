"""LangGraph orchestration of the listing pipeline.

Flow: vision -> rag -> generate -> guardrails ->(loop <= N)-> translate.

The guardrails router sends the state back to ``generate`` when the draft
failed the guardrails, bounded by ``Settings.max_compliance_retries``. Once
the budget is exhausted the last draft proceeds anyway, with the violations
reported in the response so callers can decide what to do.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from typing import Any

from langgraph.graph import END, START, StateGraph

from app.agents.nodes import (
    compliance_node,
    generate_node,
    rag_node,
    translate_node,
    vision_node,
)
from app.agents.state import AgentState
from app.config import LLMMode, Settings, get_settings
from app.models.compliance import ComplianceResult
from app.models.listing import ListingMetadata, ListingResponse, VisualAnalysis
from app.utils.logger import get_logger
from app.utils.usage import get_total_tokens, reset_usage

logger = get_logger(__name__)


def _route_after_guardrails(state: AgentState) -> str:
    """Decide whether to regenerate the draft or move on to translation.

    Args:
        state: State right after the compliance node.

    Returns:
        ``"regenerate"`` to loop back into generation, ``"translate"`` to
        finish with the current draft.
    """
    settings = get_settings()
    compliance = state.get("compliance")
    attempts = int(state.get("attempts", 0))

    if compliance is not None and compliance.passed:
        return "translate"
    if attempts >= settings.max_compliance_retries:
        logger.warning("graph.compliance.budget_exhausted", attempts=attempts)
        return "translate"
    return "regenerate"


def build_graph():
    """Construct and compile the listing StateGraph.

    Returns:
        A compiled LangGraph runnable over :class:`AgentState`.
    """
    graph = StateGraph(AgentState)
    graph.add_node("vision", vision_node)
    graph.add_node("rag", rag_node)
    graph.add_node("generate", generate_node)
    graph.add_node("guardrails", compliance_node)
    graph.add_node("translate", translate_node)

    graph.add_edge(START, "vision")
    graph.add_edge("vision", "rag")
    graph.add_edge("rag", "generate")
    graph.add_edge("generate", "guardrails")
    graph.add_conditional_edges(
        "guardrails",
        _route_after_guardrails,
        {"regenerate": "generate", "translate": "translate"},
    )
    graph.add_edge("translate", END)
    return graph.compile()


_compiled_graph = None


def get_graph():
    """Return the process-wide compiled graph, building it on first use."""
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph


async def generate_listing(
    images: list[bytes],
    category: str,
    platform: str,
    target_lang: str = "en",
    extra_info: dict[str, Any] | None = None,
    settings: Settings | None = None,
) -> ListingResponse:
    """Run the full agent pipeline for one product.

    Args:
        images: Raw bytes of the uploaded product images (1..max_images).
        category: Seller-declared product category.
        platform: Target marketplace key (amazon / shopee / temu).
        target_lang: BCP-47 style target language code for the final copy.
        extra_info: Optional seller attributes (brand/material/price...).
        settings: Optional settings override (metadata only; nodes read the
            global settings).

    Returns:
        The final ListingResponse including compliance report and metadata.
    """
    async for event in stream_listing(
        images=images,
        category=category,
        platform=platform,
        target_lang=target_lang,
        extra_info=extra_info,
        settings=settings,
    ):
        if event["type"] == "product_done":
            return event["listing"]
    raise RuntimeError("pipeline produced no result")  # pragma: no cover


async def stream_listing(
    images: list[bytes],
    category: str,
    platform: str,
    target_lang: str = "en",
    extra_info: dict[str, Any] | None = None,
    settings: Settings | None = None,
) -> AsyncIterator[dict[str, Any]]:
    """Run the pipeline, yielding progress events and the final listing.

    Events (dicts, JSON-serializable):
      - ``{"type": "node", "node": <name>, "info": {...}}``
        once per completed graph node; ``info`` is a compact JSON-safe summary.
      - ``{"type": "product_done", "listing": <ListingResponse>}`` as the
        final event on success (the raw model; JSON serialization is the
        caller's job). Errors propagate as exceptions so callers can attach
        their own terminal event.
    """
    s = settings or get_settings()
    graph = get_graph()

    initial_state: AgentState = {
        "images": images,
        "category": category,
        "platform": platform,
        "target_lang": target_lang,
        "extra_info": extra_info or {},
    }

    # Fresh token accounting window for this task (isolated per async task).
    reset_usage()

    started = time.perf_counter()
    result: AgentState = {}

    async for chunk in graph.astream(initial_state, stream_mode="updates"):
        for node_name, delta in chunk.items():
            if not isinstance(delta, dict):
                continue
            result.update(delta)
            yield {
                "type": "node",
                "node": node_name,
                "info": _summarize_node(node_name, delta),
            }

    latency_ms = int((time.perf_counter() - started) * 1000)
    total_tokens = get_total_tokens()
    logger.info(
        "graph.completed",
        platform=platform,
        latency_ms=latency_ms,
        attempts=result.get("attempts", 0),
        total_tokens=total_tokens,
    )
    response = _to_response(result, latency_ms, s, total_tokens)
    yield {"type": "product_done", "listing": response}


def _summarize_node(node_name: str, delta: dict[str, Any]) -> dict[str, Any]:
    """Compact progress info for a node event (safe for SSE payloads)."""
    if node_name == "vision" and "visual_analysis" in delta:
        analysis = delta["visual_analysis"]
        return {
            "detected_category": analysis.detected_category,
            "selling_points": len(analysis.selling_points),
        }
    if node_name == "rag":
        return {"rules": len(delta.get("retrieved_rules", []))}
    if node_name == "generate":
        listing = delta.get("listing") or {}
        return {"title": str(listing.get("title", ""))[:80]}
    if node_name == "guardrails" and "compliance" in delta:
        compliance = delta["compliance"]
        return {
            "passed": compliance.passed,
            "attempts": compliance.attempts,
            "violations": len(compliance.violations),
        }
    return {}


def _to_response(
    result: AgentState, latency_ms: int, settings: Settings, total_tokens: int = 0
) -> ListingResponse:
    """Convert the terminal graph state into the API response model."""
    final = result.get("final_listing") or result.get("listing") or {}
    compliance = result.get("compliance") or ComplianceResult(
        passed=False, violations=["pipeline produced no compliance result"]
    )
    visual = result.get("visual_analysis") or VisualAnalysis()
    rules = result.get("retrieved_rules") or []

    model_used = (
        settings.llm_model if settings.llm_mode == LLMMode.API else "mock"
    )

    return ListingResponse(
        title=str(final.get("title", "")),
        bullet_points=[str(b) for b in final.get("bullet_points", [])],
        description=str(final.get("description", "")),
        backend_keywords=[str(k) for k in final.get("backend_keywords", [])],
        compliance=compliance,
        visual_analysis=visual,
        metadata=ListingMetadata(
            model_used=model_used,
            latency_ms=latency_ms,
            rag_chunks_used=len(rules),
            total_tokens=total_tokens,
        ),
        title_zh=str(result.get("title_zh", "")),
        bullet_points_zh=[str(b) for b in result.get("bullet_points_zh", [])],
        description_zh=str(result.get("description_zh", "")),
    )
