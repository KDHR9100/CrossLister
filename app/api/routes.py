"""HTTP API routes: listing generation and RAG index maintenance."""

from __future__ import annotations

import asyncio
import base64
import json

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import Response
from pydantic import BaseModel, Field

from app.agents.graph import generate_listing
from app.api.batch_import import (
    ParseResult,
    generate_csv_template,
    parse_csv,
    parse_excel,
)
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


class BatchProductItem(BaseModel):
    """A single product in a batch generation request."""
    product_index: int = Field(description="0-based product index")
    images_base64: list[str] = Field(
        description="List of base64-encoded image data (with or without data URI prefix)"
    )
    category: str = Field(description="Product category")
    platform: str = Field(default="amazon", description="Target platform")
    target_lang: str = Field(default="en", description="Target language code")
    extra_info: str | None = Field(
        default=None,
        description="Optional extra info: JSON string or natural language text",
    )


class BatchGenerateRequest(BaseModel):
    """Request body for batch listing generation."""
    products: list[BatchProductItem]


class BatchProductResult(BaseModel):
    """Result for a single product in a batch response."""
    product_index: int
    listing: ListingResponse | None = None
    error: str | None = None


class BatchGenerateResponse(BaseModel):
    """Response for batch listing generation."""
    results: list[BatchProductResult]


class ImportParsedProduct(BaseModel):
    """A single parsed product from batch import."""
    row_number: int
    category: str
    platform: str
    target_lang: str
    extra_info: str
    errors: list[str] = []
    is_valid: bool


class ImportParseResponse(BaseModel):
    """Response from parsing a batch import file."""
    total_rows: int
    valid_count: int
    error_count: int
    file_errors: list[str] = []
    products: list[ImportParsedProduct]


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


# --------------- Helpers ---------------

def _parse_extra_info(extra_info: str | None) -> dict | None:
    """Parse extra_info: try JSON first, fall back to natural language.

    Returns:
        A dict suitable for passing to the pipeline. If the input is not
        valid JSON, it is wrapped under ``natural_language_description``.
    """
    if not extra_info:
        return None
    text = extra_info.strip()
    if not text:
        return None
    # Try JSON first
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
        # JSON but not an object — wrap it
        return {"natural_language_description": text}
    except (json.JSONDecodeError, ValueError):
        # Not JSON — treat as natural language
        return {"natural_language_description": text}


def _decode_base64_image(data: str) -> bytes:
    """Decode a base64 image, stripping data URI prefix if present."""
    if "," in data and data.startswith("data:"):
        data = data.split(",", 1)[1]
    return base64.b64decode(data)


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
        description="Optional JSON object or natural language text with extra seller context",
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

    parsed_extra = _parse_extra_info(extra_info)

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


@router.post(
    "/listing/batch_generate",
    response_model=BatchGenerateResponse,
    tags=["listing"],
)
async def batch_generate(
    request: BatchGenerateRequest,
) -> BatchGenerateResponse:
    """Generate listings for multiple products concurrently.

    Each product carries its own images (base64-encoded), category, platform,
    target language, and optional extra info. All products are processed in
    parallel via ``asyncio.gather``.
    """
    settings = get_settings()

    if not request.products:
        raise HTTPException(status_code=400, detail="At least one product is required.")

    async def _process_one(item: BatchProductItem) -> BatchProductResult:
        try:
            # Validate platform
            try:
                plat = Platform(item.platform)
            except ValueError:
                return BatchProductResult(
                    product_index=item.product_index,
                    error=f"Unsupported platform: {item.platform}",
                )

            # Decode images
            if not item.images_base64:
                return BatchProductResult(
                    product_index=item.product_index,
                    error="At least one image is required.",
                )
            if len(item.images_base64) > settings.vision_max_images:
                return BatchProductResult(
                    product_index=item.product_index,
                    error=f"At most {settings.vision_max_images} images are allowed.",
                )

            try:
                image_bytes = [_decode_base64_image(img) for img in item.images_base64]
            except Exception as exc:
                return BatchProductResult(
                    product_index=item.product_index,
                    error=f"Failed to decode image: {exc}",
                )

            if any(not data for data in image_bytes):
                return BatchProductResult(
                    product_index=item.product_index,
                    error="Empty image data detected.",
                )

            parsed_extra = _parse_extra_info(item.extra_info)

            result = await generate_listing(
                images=image_bytes,
                category=item.category,
                platform=plat.value,
                target_lang=item.target_lang,
                extra_info=parsed_extra,
                settings=settings,
            )
            return BatchProductResult(
                product_index=item.product_index,
                listing=result,
            )
        except Exception as exc:
            logger.error(
                "api.batch_generate.product_failed",
                product_index=item.product_index,
                error=str(exc),
            )
            return BatchProductResult(
                product_index=item.product_index,
                error=str(exc),
            )

    logger.info(
        "api.batch_generate.request",
        num_products=len(request.products),
    )

    results = await asyncio.gather(
        *[_process_one(item) for item in request.products]
    )
    return BatchGenerateResponse(results=list(results))


@router.get("/import/template", tags=["import"])
async def download_import_template() -> Response:
    """Download a CSV template for batch product import."""
    template_content = generate_csv_template()
    # Add BOM for Excel compatibility with Chinese characters
    content = "\ufeff" + template_content
    return Response(
        content=content.encode("utf-8"),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=product_import_template.csv"},
    )


@router.post("/import/parse", response_model=ImportParseResponse, tags=["import"])
async def parse_import_file(
    file: UploadFile = File(description="CSV or Excel file for batch import"),
) -> ImportParseResponse:
    """Parse and validate a CSV/Excel file for batch product import.

    Returns parsed product rows with validation errors for each row.
    """
    filename = file.filename or ""
    content_type = file.content_type or ""

    logger.info("api.import.parse.request", filename=filename, content_type=content_type)

    # Determine file type and parse
    if filename.endswith(".xlsx") or filename.endswith(".xls") or "spreadsheet" in content_type:
        logger.info("api.import.parse.detected_excel", filename=filename)
        content = await file.read()
        logger.info("api.import.parse.file_read", filename=filename, size_bytes=len(content))
        result = await run_in_threadpool(parse_excel, content)
    elif filename.endswith(".csv") or "csv" in content_type or "text" in content_type:
        logger.info("api.import.parse.detected_csv", filename=filename)
        raw = await file.read()
        logger.info("api.import.parse.file_read", filename=filename, size_bytes=len(raw))
        # Try to decode as UTF-8, handle BOM
        try:
            text = raw.decode("utf-8-sig")
            logger.debug("api.import.parse.decoded_utf8", filename=filename)
        except UnicodeDecodeError:
            try:
                text = raw.decode("gbk")
                logger.debug("api.import.parse.decoded_gbk", filename=filename)
            except UnicodeDecodeError:
                logger.error("api.import.parse.decode_failed", filename=filename)
                raise HTTPException(
                    status_code=400,
                    detail="文件编码不支持，请使用 UTF-8 或 GBK 编码的 CSV 文件",
                )
        result = await run_in_threadpool(parse_csv, text)
    else:
        logger.error("api.import.parse.unsupported_format", filename=filename, content_type=content_type)
        raise HTTPException(
            status_code=400,
            detail="不支持的文件格式，请上传 CSV 或 Excel (.xlsx) 文件",
        )

    logger.info(
        "api.import.parse.done",
        filename=filename,
        total_rows=result.total_rows,
        valid_count=result.valid_count,
        error_count=result.error_count,
        file_errors=result.errors,
    )

    return ImportParseResponse(
        total_rows=result.total_rows,
        valid_count=result.valid_count,
        error_count=result.error_count,
        file_errors=result.errors,
        products=[
            ImportParsedProduct(
                row_number=p.row_number,
                category=p.category,
                platform=p.platform,
                target_lang=p.target_lang,
                extra_info=p.extra_info,
                errors=p.errors,
                is_valid=p.is_valid,
            )
            for p in result.products
        ],
    )


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
