"""Phase 4 tests: the LangGraph pipeline and guardrails, all in mock mode.

The tests exercise the compiled graph end to end, including the bounded
compliance -> regeneration loop, without any network access.
"""

from __future__ import annotations

import asyncio

import pytest

from app.agents.graph import generate_listing, get_graph
from app.agents.state import AgentState
from app.config import get_settings
from app.guardrails import keyword_filter, llm_checker
from app.models.compliance import ComplianceResult
from app.rag.indexer import build_index


@pytest.fixture(scope="module", autouse=True)
def ensure_index():
    """Make sure the default Chroma index exists before graph tests run."""
    build_index(get_settings())
    yield


def test_graph_compiles_with_expected_nodes() -> None:
    graph = get_graph()
    names = set(graph.get_graph().nodes.keys())
    assert {"vision", "rag", "generate", "guardrails", "translate"} <= names


def test_graph_happy_path() -> None:
    response = asyncio.run(
        generate_listing([b"fake-image"], "storage organizer", "amazon", "en")
    )
    assert response.title
    assert response.bullet_points
    assert response.description
    assert response.backend_keywords
    assert response.compliance.passed is True
    assert response.compliance.attempts == 1
    assert response.metadata.model_used == "mock"
    assert response.metadata.rag_chunks_used > 0
    assert response.visual_analysis.detected_category == "storage organizer"


def test_graph_compliance_loop_on_forced_violation() -> None:
    response = asyncio.run(
        generate_listing(
            [b"fake-image"],
            "storage organizer",
            "amazon",
            "en",
            extra_info={"force_violation": True},
        )
    )
    # First draft is rejected, regeneration succeeds on the second attempt.
    assert response.compliance.passed is True
    assert response.compliance.attempts == 2
    assert "best seller" not in response.title.lower()


def test_route_budget_exhausted_proceeds_to_translate() -> None:
    from app.agents.graph import _route_after_guardrails

    max_attempts = get_settings().max_compliance_retries
    state: AgentState = {
        "compliance": ComplianceResult(passed=False, violations=["bad"]),
        "attempts": max_attempts,
    }
    assert _route_after_guardrails(state) == "translate"


def test_route_loops_back_on_failure() -> None:
    from app.agents.graph import _route_after_guardrails

    state: AgentState = {
        "compliance": ComplianceResult(passed=False, violations=["bad"]),
        "attempts": 1,
    }
    assert _route_after_guardrails(state) == "regenerate"


def test_keyword_filter_flags_banned_phrase() -> None:
    violations = keyword_filter.scan_listing(
        platform="amazon",
        title="Best Seller Storage Bins",
        bullet_points=[],
        description="great bins",
    )
    assert violations and "best seller" in violations[0].lower()


def test_keyword_filter_passes_clean_listing() -> None:
    violations = keyword_filter.scan_listing(
        platform="amazon",
        title="Stackable Storage Organizer Bins",
        bullet_points=["Durable PP plastic"],
        description="Keeps closets tidy.",
        backend_keywords=["storage bins"],
    )
    assert violations == []


def test_keyword_filter_unknown_platform_is_noop() -> None:
    assert (
        keyword_filter.scan_listing("ebay", "anything", [], "best seller") == []
    )


def test_llm_checker_mock_passes() -> None:
    result = asyncio.run(
        llm_checker.check_listing(
            platform="amazon",
            title="Stackable Storage Bins",
            bullet_points=["durable"],
            description="tidy closet",
        )
    )
    assert result["passed"] is True
    assert result["violations"] == []
