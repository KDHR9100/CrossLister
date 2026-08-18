"""Compliance-related Pydantic models."""

from pydantic import BaseModel, Field


class ComplianceResult(BaseModel):
    """Outcome of the guardrails compliance check."""

    passed: bool = Field(description="Whether the listing passed all checks")
    warnings: list[str] = Field(
        default_factory=list,
        description="Non-blocking issues worth attention",
    )
    violations: list[str] = Field(
        default_factory=list,
        description="Blocking rule violations found in the listing",
    )
    suggestions: list[str] = Field(
        default_factory=list,
        description="Actionable rewrite suggestions fed back to the generator",
    )
    attempts: int = Field(
        default=1,
        description="Number of generate->compliance attempts consumed",
    )
