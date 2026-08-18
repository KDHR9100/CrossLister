"""Typed state carried through the LangGraph listing pipeline.

Each node reads the fields it needs and returns a partial dict that LangGraph
merges into the state. ``total=False`` keeps every field optional so nodes can
be reasoned about in isolation.
"""

from __future__ import annotations

from typing import Any, TypedDict

from app.models.compliance import ComplianceResult
from app.models.listing import VisualAnalysis
from app.rag.retriever import RetrievedRule


class GeneratedListing(TypedDict, total=False):
    """Intermediate listing produced by the generation node."""

    title: str
    bullet_points: list[str]
    description: str
    backend_keywords: list[str]


class AgentState(TypedDict, total=False):
    """Full state flowing through vision -> rag -> generate -> compliance.

    Input fields are populated up front; the rest are filled by each node.
    """

    # -- Inputs -------------------------------------------------------
    images: list[bytes]
    category: str
    platform: str
    target_lang: str
    extra_info: dict[str, Any]

    # -- Vision node output -------------------------------------------
    visual_analysis: VisualAnalysis

    # -- RAG node output ----------------------------------------------
    retrieved_rules: list[RetrievedRule]

    # -- Generation node output ---------------------------------------
    listing: GeneratedListing

    # -- Compliance node output ---------------------------------------
    compliance: ComplianceResult
    attempts: int

    # -- Translate node output (final) --------------------------------
    final_listing: GeneratedListing

    # -- Bookkeeping ---------------------------------------------------
    error: str | None
