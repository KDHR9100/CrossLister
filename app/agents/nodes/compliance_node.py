"""Compliance node: guardrails check with bounded regeneration.

Combines the deterministic keyword filter with the semantic LLM checker and
counts how many generate->compliance attempts have been consumed. The graph
router uses the resulting ComplianceResult plus ``attempts`` to decide whether
to loop back to generation or continue to translation.
"""

from __future__ import annotations

from app.agents.state import AgentState
from app.guardrails import keyword_filter, llm_checker, structural_validator
from app.models.compliance import ComplianceResult
from app.utils.logger import get_logger

logger = get_logger(__name__)


async def compliance_node(state: AgentState) -> dict:
    """Check the current listing draft against platform rules.

    Args:
        state: Current pipeline state with ``listing`` populated.

    Returns:
        A partial state update containing ``compliance`` and ``attempts``.
    """
    platform = state.get("platform", "")
    listing = state.get("listing") or {}
    attempts = int(state.get("attempts", 0)) + 1

    title = str(listing.get("title", ""))
    bullets = [str(b) for b in listing.get("bullet_points", [])]
    description = str(listing.get("description", ""))
    keywords = [str(k) for k in listing.get("backend_keywords", [])]

    # 1) Fast deterministic checks.
    violations = keyword_filter.scan_listing(
        platform=platform,
        title=title,
        bullet_points=bullets,
        description=description,
        backend_keywords=keywords,
    )

    # 1b) Deterministic structural limits (lengths/counts/URLs/prices...).
    # Warnings are report-only; violations join the rewrite loop below.
    structural_violations, structural_warnings = structural_validator.scan_structure(
        platform=platform,
        title=title,
        bullet_points=bullets,
        description=description,
        backend_keywords=keywords,
    )
    violations.extend(structural_violations)
    warnings = list(structural_warnings)

    # 2) Semantic LLM review against the retrieved rules.
    rules_payload = [
        {"rule_id": r.rule_id, "title": r.title, "text": r.text}
        for r in (state.get("retrieved_rules") or [])
    ]
    llm_result = await llm_checker.check_listing(
        platform=platform,
        title=title,
        bullet_points=bullets,
        description=description,
        retrieved_rules=rules_payload,
    )
    violations.extend(llm_result.get("violations", []))
    suggestions = list(llm_result.get("suggestions", []))
    warnings.extend(llm_result.get("warnings", []))

    passed = not violations
    if not passed:
        suggestions.extend(
            f"Remove the banned/forbidden phrasing flagged above "
            f"and rewrite the affected fields for {platform}."
        )

    result = ComplianceResult(
        passed=passed,
        warnings=warnings,
        violations=violations,
        suggestions=suggestions,
        attempts=attempts,
    )
    logger.info(
        "node.compliance.done",
        attempts=attempts,
        passed=passed,
        violations=len(violations),
    )
    return {"compliance": result, "attempts": attempts}
