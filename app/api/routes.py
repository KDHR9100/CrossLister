"""HTTP API routes: listing generation and RAG index maintenance."""

from __future__ import annotations

import json

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from app.agents.graph import generate_listing
from app.config import get_settings
from app.models.listing import ListingResponse, Platform
from app.rag.indexer import build_index
from app.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1")


# --------------- Response models ---------------

class PlatformInfo(BaseModel):
    """A single supported platform."""
    id: str
    name: str
    description: str


class PlatformsResponse(BaseModel):
    """List of supported platforms."""
    platforms: list[PlatformInfo]


class ErrorResponse(BaseModel):
    """Structured error response."""
    error: str
    detail: str = ""
    code: str = ""


class SupportedLanguages(BaseModel):
    """Supported target languages."""
    languages: list[dict[str, str]] = Field(
        description="List of {code, name} language objects"
    )


# --------------- Constants ---------------

SUPPORTED_LANGUAGES = [
    {"code": "en", "name": "English"},
    {"code": "zh", "name": "中文"},
    {"code": "ja", "name": "日本語"},
    {"code": "ko", "name": "한국어"},
    {"code": "es", "name": "Español"},
    {"code": "fr", "name": "Français"},
    {"code": "de", "name": "Deutsch"},
    {"code": "pt", "name": "Português"},
    {"code": "th", "name": "ภาษาไทย"},
    {"code": "vi", "name": "Tiếng Việt"},
    {"code": "id", "name": "Bahasa Indonesia"},
    {"code": "ms", "name": "Bahasa Melayu"},
]

PLATFORM_INFO = {
    Platform.AMAZON: PlatformInfo(
        id="amazon", name="Amazon",
        description="全球最大电商平台，覆盖北美/欧洲/日本等站点",
    ),
    Platform.SHOPEE: PlatformInfo(
        id="shopee", name="Shopee",
        description="东南亚及台湾地区领先电商平台",
    ),
    Platform.TEMU: PlatformInfo(
        id="temu", name="Temu",
        description="拼多多旗下跨境电商平台，覆盖北美/欧洲",
    ),
}


# --------------- Routes ---------------

@router.get("/platforms", response_model=PlatformsResponse, tags=["meta"])
async def list_platforms() -> PlatformsResponse:
    """Return all supported marketplace platforms."""
    return PlatformsResponse(platforms=list(PLATFORM_INFO.values()))


@router.get("/languages", response_model=SupportedLanguages, tags=["meta"])
async def list_languages() -> SupportedLanguages:
    """Return supported target languages for listing translation."""
    return SupportedLanguages(languages=SUPPORTED_LANGUAGES)


@router.post("/listing/generate", response_model=ListingResponse, tags=["listing"])
async def generate(
    images: list[UploadFile] = File(description="1..5 product images"),
    category: str = Form(description="Product category, e.g. '家居收纳'"),
    platform: Platform = Form(default=Platform.AMAZON),
    target_lang: str = Form(default="en"),
    extra_info: str | None = Form(
        default=None,
        description="Optional JSON object with extra seller context",
    ),
) -> ListingResponse:
    """Generate a compliance-checked, localized listing from product images."""
    settings = get_settings()

    if not images:
        raise HTTPException(status_code=400, detail="At least one image is required.")
    if len(images) > settings.vision_max_images:
        raise HTTPException(
            status_code=400,
            detail=f"At most {settings.vision_max_images} images are allowed.",
        )

    parsed_extra: dict | None = None
    if extra_info:
        try:
            parsed_extra = json.loads(extra_info)
        except json.JSONDecodeError as exc:
            raise HTTPException(
                status_code=400, detail=f"extra_info is not valid JSON: {exc}"
            ) from exc
        if not isinstance(parsed_extra, dict):
            raise HTTPException(
                status_code=400, detail="extra_info must be a JSON object."
            )

    image_bytes = [await f.read() for f in images]
    if any(not data for data in image_bytes):
        raise HTTPException(status_code=400, detail="Empty image upload detected.")

    logger.info(
        "api.generate.request",
        category=category,
        platform=platform.value,
        num_images=len(images),
    )

    try:
        return await generate_listing(
            images=image_bytes,
            category=category,
            platform=platform.value,
            target_lang=target_lang,
            extra_info=parsed_extra,
            settings=settings,
        )
    except Exception as exc:
        logger.error("api.generate.failed", error=str(exc))
        raise HTTPException(
            status_code=500,
            detail=f"Listing generation failed: {exc}",
        ) from exc


@router.post("/rag/rebuild", tags=["rag"])
async def rebuild_rag() -> dict:
    """Rebuild the per-platform rule index from the bundled rule documents."""
    settings = get_settings()
    logger.info("api.rag_rebuild.request")
    stats = await run_in_threadpool(build_index, settings)
    return {
        "status": "ok",
        "platforms": stats.platforms,
        "total_chunks": stats.total_chunks,
    }
