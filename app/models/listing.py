"""Listing-related Pydantic models."""

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from app.models.compliance import ComplianceResult


class Platform(str, Enum):
    """Supported e-commerce target platforms."""

    AMAZON = "amazon"
    SHOPEE = "shopee"
    TEMU = "temu"


class ListingRequest(BaseModel):
    """Form fields of POST /api/v1/listing/generate.

    Images are received separately as multipart files; the remaining form
    fields are validated by this model.
    """

    category: str = Field(description="Product category, e.g. '家居收纳'")
    platform: Platform = Field(
        default=Platform.AMAZON, description="Target marketplace"
    )
    target_lang: str = Field(
        default="en", description="BCP-47 style target language code"
    )
    extra_info: dict[str, Any] | None = Field(
        default=None,
        description="Optional extra context such as brand / material / price",
    )


class VisualAnalysis(BaseModel):
    """Structured output produced by the vision node."""

    detected_category: str = Field(
        default="", description="Category detected from the product images"
    )
    colors: list[str] = Field(default_factory=list)
    materials: list[str] = Field(default_factory=list)
    selling_points: list[str] = Field(default_factory=list)
    scenes: list[str] = Field(
        default_factory=list, description="Usage scenes implied by the images"
    )
    raw_description: str = Field(
        default="", description="Free-form description from the vision model"
    )


class ListingMetadata(BaseModel):
    """Generation metadata returned to the caller."""

    model_used: str = ""
    latency_ms: int = 0
    rag_chunks_used: int = 0
    total_tokens: int = 0


class ListingResponse(BaseModel):
    """Final structured listing returned by the API."""

    title: str
    bullet_points: list[str] = Field(default_factory=list)
    description: str = ""
    backend_keywords: list[str] = Field(default_factory=list)
    compliance: ComplianceResult
    visual_analysis: VisualAnalysis = Field(default_factory=VisualAnalysis)
    metadata: ListingMetadata = Field(default_factory=ListingMetadata)
    # Chinese translation fields (always provided alongside target language)
    title_zh: str = ""
    bullet_points_zh: list[str] = Field(default_factory=list)
    description_zh: str = ""
