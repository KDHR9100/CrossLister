"""HTTP API routes: listing generation and RAG index maintenance."""

from __future__ import annotations

import asyncio
import json
import re
import time
from collections.abc import AsyncIterator
from datetime import datetime

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field

from app.agents.graph import generate_listing, stream_listing
from app.api.batch_import import (
    ParseResult,
    generate_csv_template,
    parse_csv,
    parse_excel,
)
from app.config import Settings, get_settings
from app.history import store as history_store
from app.models.listing import ListingResponse, Platform
from app.rag.indexer import build_index
from app.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1")

# Captured when this module is imported (i.e. when the server starts or a
# --reload worker spawns). Exposed via /api/v1/diag so operators can confirm
# a running server actually loaded the latest code after a restart.
_MODULE_STARTED_AT = datetime.now().astimezone().isoformat(timespec="seconds")


# --------------- Upload limits & sanitisation ---------------

# Per-image and per-import-file size caps (bytes). Generous for real product
# photos but bounded so a malicious oversized upload can't exhaust memory.
MAX_IMAGE_BYTES = 20 * 1024 * 1024  # 20 MB per image
MAX_IMPORT_FILE_BYTES = 20 * 1024 * 1024  # 20 MB per CSV/Excel file

# Magic-byte signatures used to confirm an upload is really an image.
_IMAGE_SIGNATURES: tuple[tuple[bytes, str], ...] = (
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
    (b"BM", "image/bmp"),
)

# Secret-looking patterns redacted from error messages before they reach the
# client. Operational context stays, credentials never do.
_SK_KEY_RE = re.compile(r"sk-[A-Za-z0-9_\-]{8,}")
_BEARER_RE = re.compile(r"(?i)bearer\s+[A-Za-z0-9_\-\.=]{8,}")
_CRED_RE = re.compile(
    r"(?i)(api[_-]?key|token|secret|password|authorization)\s*[:=]\s*[^\s,;\"']{6,}"
)


def _is_image_bytes(data: bytes) -> bool:
    """Best-effort check that raw bytes start with a known image signature."""
    if data.startswith(b"RIFF") and len(data) >= 12 and data[8:12] == b"WEBP":
        return True
    return any(data.startswith(sig) for sig, _ in _IMAGE_SIGNATURES)


def sanitize_error(message: str) -> str:
    """Redact credentials from an error string while keeping it reportable.

    Ops staff can still screenshot and share the message with engineers, but
    API keys / bearer tokens never leak to the client.
    """
    text = _SK_KEY_RE.sub("[REDACTED_KEY]", message)
    text = _BEARER_RE.sub("Bearer [REDACTED_KEY]", text)
    text = _CRED_RE.sub("[REDACTED_CREDENTIAL]", text)
    return text


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


class BatchProductMeta(BaseModel):
    """Metadata for one product in a multipart batch generation request.

    Images travel as separate multipart file parts; ``image_count`` declares
    how many of the (ordered) ``images`` parts belong to this product.
    """
    product_index: int = Field(description="0-based product index")
    category: str = Field(description="Product category")
    platform: str = Field(default="amazon", description="Target platform")
    target_lang: str = Field(default="en", description="Target language code")
    extra_info: str | None = Field(
        default=None,
        description="Optional extra info: JSON string or natural language text",
    )
    image_count: int = Field(ge=0, description="Number of images for this product")


class BatchProductResult(BaseModel):
    """Result for a single product in a batch response."""
    product_index: int
    listing: ListingResponse | None = None
    error: str | None = None
    # Wall-clock time spent on this product (covers success and failure).
    elapsed_ms: int = 0


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

def _parse_batch_request(
    products: str,
    images: list[UploadFile],
    settings: Settings,
) -> tuple[list[BatchProductMeta], list[list[UploadFile]]]:
    """Parse and validate the shared batch request shape.

    Both ``/listing/batch_generate`` and the SSE stream variant accept the
    same multipart body: a ``products`` JSON array plus ordered ``images``
    parts sliced per product by their declared ``image_count``.

    Returns:
        (product metadata list, per-product image file groups).

    Raises:
        HTTPException: 400 on malformed JSON, empty products, or image count
            mismatch between metadata and uploads.
    """
    try:
        raw_products = json.loads(products)
    except (json.JSONDecodeError, ValueError) as exc:
        raise HTTPException(
            status_code=400, detail=f"'products' is not valid JSON: {exc}"
        ) from exc
    if not isinstance(raw_products, list) or not raw_products:
        raise HTTPException(status_code=400, detail="At least one product is required.")
    try:
        metas = [BatchProductMeta(**item) for item in raw_products]
    except Exception as exc:  # noqa: BLE001 - pydantic ValidationError & co.
        raise HTTPException(
            status_code=400, detail=sanitize_error(f"Invalid product metadata: {exc}")
        ) from exc

    expected = sum(m.image_count for m in metas)
    if expected != len(images):
        raise HTTPException(
            status_code=400,
            detail=(
                f"Image count mismatch: metadata declares {expected} images "
                f"but {len(images)} were uploaded."
            ),
        )
    grouped: list[list[UploadFile]] = []
    offset = 0
    for m in metas:
        grouped.append(images[offset : offset + m.image_count])
        offset += m.image_count
    return metas, grouped

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


def _declared_size(f: UploadFile) -> int | None:
    """Return the upload's declared byte size, or None when unknown.

    Starlette's multipart parser tracks the byte count per part, so the size
    is available *before* the body is read — letting us reject oversized
    uploads without ever pulling them into memory.
    """
    return getattr(f, "size", None)


async def _read_and_validate(f: UploadFile) -> bytes:
    """Read one uploaded image and validate size and format.

    Raises:
        ValueError: When the file is empty, oversized, or not an image.
    """
    # Reject oversize uploads before reading: when the declared size is known
    # we never pull the body into memory at all. The post-read check below is
    # the fallback for transports that do not report a size.
    declared = _declared_size(f)
    if declared is not None and declared > MAX_IMAGE_BYTES:
        raise ValueError(
            f"Image '{f.filename}' exceeds the "
            f"{MAX_IMAGE_BYTES // (1024 * 1024)} MB limit."
        )
    data = await f.read()
    if not data:
        raise ValueError(f"Image '{f.filename}' is empty.")
    if len(data) > MAX_IMAGE_BYTES:
        raise ValueError(
            f"Image '{f.filename}' exceeds the "
            f"{MAX_IMAGE_BYTES // (1024 * 1024)} MB limit."
        )
    if not _is_image_bytes(data):
        raise ValueError(f"Image '{f.filename}' is not a recognised image format.")
    return data


def _record_history(
    api: str,
    products_payload: list[dict],
    collected_images: dict[int, list[bytes]],
) -> None:
    """Persist a generation record to cold storage, fire-and-forget.

    Called only after generation completes (success or failure). The write runs
    in a background task off the event loop, and every error is swallowed and
    logged — history can never block, delay, or break the request.
    """
    settings = get_settings()
    if not settings.history_enabled:
        return
    task = asyncio.create_task(
        history_store.save_record_async(
            settings.history_dir,
            api,
            products_payload,
            collected_images,
            save_images=settings.history_save_images,
            max_records=settings.history_max_records,
            max_image_side=settings.vision_max_image_side,
            jpeg_quality=settings.vision_jpeg_quality,
        )
    )

    def _log_failure(t: asyncio.Task) -> None:
        exc = t.exception()
        if exc is not None:
            logger.error("history.task_failed", error=str(exc))

    task.add_done_callback(_log_failure)


# --------------- Routes ---------------

@router.get("/platforms", response_model=PlatformsResponse, tags=["meta"])
async def list_platforms() -> PlatformsResponse:
    """Return all supported marketplace platforms."""
    return PlatformsResponse(platforms=list(PLATFORM_INFO.values()))


@router.get("/languages", response_model=SupportedLanguages, tags=["meta"])
async def list_languages() -> SupportedLanguages:
    """Return supported target languages for listing translation."""
    return SupportedLanguages(languages=SUPPORTED_LANGUAGES)


@router.get("/diag", tags=["meta"])
async def diagnostics() -> dict:
    """Expose the loaded config and module start time.

    Lets operators confirm a running server actually picked up new code after
    a restart: if ``module_started_at`` predates the last code change or
    restart, the server is stale and must be restarted.
    """
    s = get_settings()
    return {
        "module_started_at": _MODULE_STARTED_AT,
        "vision": {
            "mode": s.vision_mode.value,
            "model": s.vision_model,
            "max_images": s.vision_max_images,
            "max_image_side": s.vision_max_image_side,
            "jpeg_quality": s.vision_jpeg_quality,
        },
        "batch": {
            "max_concurrency": s.batch_max_concurrency,
            "product_timeout_s": s.batch_product_timeout_s,
        },
        "limits": {
            "max_image_bytes": MAX_IMAGE_BYTES,
            "max_import_file_bytes": MAX_IMPORT_FILE_BYTES,
        },
    }


@router.post("/listing/generate", response_model=ListingResponse, tags=["listing"])
async def generate(
    images: list[UploadFile] = File(
        description="Product images; the per-product maximum is controlled by VISION_MAX_IMAGES"
    ),
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

    image_bytes: list[bytes] = []
    for f in images:
        try:
            image_bytes.append(await _read_and_validate(f))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    logger.info(
        "api.generate.request",
        category=category,
        platform=platform.value,
        num_images=len(images),
    )

    t0 = time.perf_counter()
    try:
        result = await generate_listing(
            images=image_bytes,
            category=category,
            platform=platform.value,
            target_lang=target_lang,
            extra_info=parsed_extra,
            settings=settings,
        )
    except Exception as exc:
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        logger.error("api.generate.failed", error=str(exc))
        # History: record the failed generation too (fire-and-forget, after
        # the pipeline finished — it never touches the response path).
        _record_history(
            "listing/generate",
            [
                {
                    "product_index": 0,
                    "category": category,
                    "platform": platform.value,
                    "target_lang": target_lang,
                    "status": "failed",
                    "listing": None,
                    "error": sanitize_error(str(exc)),
                    "elapsed_ms": elapsed_ms,
                }
            ],
            {0: image_bytes},
        )
        # Keep the message useful for ops screenshots but strip credentials.
        raise HTTPException(
            status_code=500,
            detail=sanitize_error(f"Listing generation failed: {exc}"),
        ) from exc

    # History: persisted only after generation succeeded; response is already
    # built, so the background write cannot affect it.
    _record_history(
        "listing/generate",
        [
            {
                "product_index": 0,
                "category": category,
                "platform": platform.value,
                "target_lang": target_lang,
                "status": "success",
                "listing": result.model_dump(),
                "error": None,
                "elapsed_ms": int((time.perf_counter() - t0) * 1000),
            }
        ],
        {0: image_bytes},
    )
    return result


@router.post(
    "/listing/batch_generate",
    response_model=BatchGenerateResponse,
    tags=["listing"],
)
async def batch_generate(
    products: str = Form(
        description="JSON array of product metadata; each item declares image_count"
    ),
    images: list[UploadFile] = File(
        default=[],
        description="All product images, appended in product order",
    ),
) -> BatchGenerateResponse:
    """Generate listings for multiple products concurrently.

    Multipart request: the ``products`` form field holds a JSON array of
    product metadata (category, platform, target language, extra info and
    ``image_count``), and every image is uploaded as an ``images`` file part.
    Multipart preserves part order, so images are sliced per product by their
    declared ``image_count``.
    """
    settings = get_settings()
    metas, grouped = _parse_batch_request(products, images, settings)

    # Bound concurrent generations so we don't flood the remote LLM endpoint
    # (which otherwise triggers rate limits / connection resets).
    max_concurrency = max(1, settings.batch_max_concurrency)
    sem = asyncio.Semaphore(max_concurrency)

    # Raw image bytes per product index, kept only for the post-generation
    # history write (cold storage). Populated while validating uploads.
    collected_images: dict[int, list[bytes]] = {}

    async def _process_one(
        item: BatchProductMeta, files: list[UploadFile]
    ) -> BatchProductResult:
        async with sem:
            t0 = time.perf_counter()

            def _elapsed() -> int:
                return int((time.perf_counter() - t0) * 1000)

            try:
                # Validate platform
                try:
                    plat = Platform(item.platform)
                except ValueError:
                    return BatchProductResult(
                        product_index=item.product_index,
                        error=f"Unsupported platform: {item.platform}",
                        elapsed_ms=_elapsed(),
                    )

                # Read and validate this product's images
                if not files:
                    return BatchProductResult(
                        product_index=item.product_index,
                        error="At least one image is required.",
                        elapsed_ms=_elapsed(),
                    )
                if len(files) > settings.vision_max_images:
                    return BatchProductResult(
                        product_index=item.product_index,
                        error=f"At most {settings.vision_max_images} images are allowed.",
                        elapsed_ms=_elapsed(),
                    )

                try:
                    image_bytes = [await _read_and_validate(f) for f in files]
                except Exception as exc:  # noqa: BLE001 - report per product
                    return BatchProductResult(
                        product_index=item.product_index,
                        error=sanitize_error(str(exc)),
                        elapsed_ms=_elapsed(),
                    )
                collected_images[item.product_index] = image_bytes

                parsed_extra = _parse_extra_info(item.extra_info)

                try:
                    result = await asyncio.wait_for(
                        generate_listing(
                            images=image_bytes,
                            category=item.category,
                            platform=plat.value,
                            target_lang=item.target_lang,
                            extra_info=parsed_extra,
                            settings=settings,
                        ),
                        timeout=settings.batch_product_timeout_s,
                    )
                except asyncio.TimeoutError:
                    return BatchProductResult(
                        product_index=item.product_index,
                        error=f"生成超时（超过 {int(settings.batch_product_timeout_s)} 秒），请重试或减少图片数量。",
                        elapsed_ms=_elapsed(),
                    )
                return BatchProductResult(
                    product_index=item.product_index,
                    listing=result,
                    elapsed_ms=_elapsed(),
                )
            except Exception as exc:
                logger.error(
                    "api.batch_generate.product_failed",
                    product_index=item.product_index,
                    error=str(exc),
                )
                return BatchProductResult(
                    product_index=item.product_index,
                    error=sanitize_error(str(exc)),
                    elapsed_ms=_elapsed(),
                )

    logger.info(
        "api.batch_generate.request",
        num_products=len(metas),
        num_images=len(images),
    )

    results = await asyncio.gather(
        *[_process_one(m, files) for m, files in zip(metas, grouped)]
    )

    # History (cold storage): written only after every product has finished.
    # Fire-and-forget — the response below is already built and returns
    # immediately; the background write cannot block or alter it.
    _record_history(
        "listing/batch_generate",
        [
            {
                "product_index": m.product_index,
                "category": m.category,
                "platform": m.platform,
                "target_lang": m.target_lang,
                "status": "success" if r.listing else "failed",
                "listing": r.listing.model_dump() if r.listing else None,
                "error": r.error,
                "elapsed_ms": r.elapsed_ms,
            }
            for m, r in zip(metas, results)
        ],
        collected_images,
    )
    return BatchGenerateResponse(results=list(results))


# --------------- Streaming batch generation (SSE) ---------------

def _sse(payload: dict) -> str:
    """Format one server-sent event frame."""
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


@router.post(
    "/listing/batch_generate_stream",
    response_model=None,
    tags=["listing"],
)
async def batch_generate_stream(
    products: str = Form(
        description="JSON array of product metadata; each item declares image_count"
    ),
    images: list[UploadFile] = File(
        default=[],
        description="All product images, appended in product order",
    ),
) -> StreamingResponse:
    """Generate listings for multiple products, streaming progress via SSE.

    Same request shape as ``/listing/batch_generate``. Emits ``data:`` frames
    carrying JSON events:
      - ``{"type": "product_start", "product_index": i}``
      - ``{"type": "node", "product_index": i, "node": ..., "info": {...}}``
        after each pipeline node of product i completes
      - ``{"type": "product_done", "product_index": i, "listing": {...}}`` or
        ``{"type": "product_done", "product_index": i, "error": "..."}``
      - ``{"type": "done"}`` when every product has finished
    """
    settings = get_settings()
    metas, grouped = _parse_batch_request(products, images, settings)

    max_concurrency = max(1, settings.batch_max_concurrency)
    semaphore = asyncio.Semaphore(max_concurrency)
    queue: asyncio.Queue = asyncio.Queue()
    collected_images: dict[int, list[bytes]] = {}

    async def _produce(idx: int, item: BatchProductMeta, files: list[UploadFile]) -> None:
        """Run one product in its own task, pushing events into the queue.

        Every terminal path emits exactly one ``product_done`` event, and the
        ``finally`` block pushes a sentinel so the consumer knows this
        producer finished.
        """
        started = time.perf_counter()

        def _done_payload(listing: dict | None = None, error: str | None = None) -> dict:
            return {
                "type": "product_done",
                "product_index": idx,
                "listing": listing,
                "error": error,
                "elapsed_ms": int((time.perf_counter() - started) * 1000),
            }

        try:
            async with semaphore:
                try:
                    plat = Platform(item.platform)
                except ValueError:
                    await queue.put(
                        _done_payload(error=f"Unsupported platform: {item.platform}")
                    )
                    return

                if not files:
                    await queue.put(
                        _done_payload(error="At least one image is required.")
                    )
                    return
                if len(files) > settings.vision_max_images:
                    await queue.put(
                        _done_payload(
                            error=f"At most {settings.vision_max_images} images are allowed."
                        )
                    )
                    return

                try:
                    image_bytes = [await _read_and_validate(f) for f in files]
                except Exception as exc:  # noqa: BLE001 - report per product
                    await queue.put(_done_payload(error=sanitize_error(str(exc))))
                    return
                collected_images[idx] = image_bytes

                await queue.put({"type": "product_start", "product_index": idx})

                try:
                    async with asyncio.timeout(settings.batch_product_timeout_s):
                        async for event in stream_listing(
                            images=image_bytes,
                            category=item.category,
                            platform=plat.value,
                            target_lang=item.target_lang,
                            extra_info=_parse_extra_info(item.extra_info),
                            settings=settings,
                        ):
                            if event.get("type") == "product_done" and event.get(
                                "listing"
                            ) is not None:
                                # JSON-ready dict for SSE frames and history.
                                event["listing"] = event["listing"].model_dump(
                                    mode="json"
                                )
                            event["product_index"] = idx
                            await queue.put(event)
                except TimeoutError:
                    await queue.put(
                        _done_payload(
                            error=(
                                f"生成超时（超过 {int(settings.batch_product_timeout_s)} 秒），"
                                f"请重试或减少图片数量。"
                            )
                        )
                    )
        except Exception as exc:  # noqa: BLE001 - one product must not kill the stream
            logger.error(
                "api.batch_generate_stream.product_failed",
                product_index=idx,
                error=str(exc),
            )
            await queue.put(_done_payload(error=sanitize_error(str(exc))))
        finally:
            await queue.put(None)  # sentinel: producer finished

    async def _event_stream() -> AsyncIterator[str]:
        producers = [
            asyncio.create_task(_produce(idx, item, files))
            for idx, (item, files) in enumerate(zip(metas, grouped))
        ]
        history_payloads: list[dict] = []
        finished = 0
        try:
            while finished < len(producers):
                event = await queue.get()
                if event is None:
                    finished += 1
                    continue
                if event.get("type") == "product_done":
                    idx = event["product_index"]
                    history_payloads.append(
                        {
                            "product_index": idx,
                            "category": metas[idx].category,
                            "platform": metas[idx].platform,
                            "target_lang": metas[idx].target_lang,
                            "status": "success" if event.get("listing") else "failed",
                            "listing": event.get("listing"),
                            "error": event.get("error"),
                            "elapsed_ms": event.get("elapsed_ms", 0),
                        }
                    )
                yield _sse(event)
            # History (cold storage): fire-and-forget after all products end.
            history_payloads.sort(key=lambda p: p["product_index"])
            _record_history(
                "listing/batch_generate_stream",
                history_payloads,
                collected_images,
            )
            yield _sse({"type": "done"})
        finally:
            # Client disconnected: stop the remaining producer tasks.
            for task in producers:
                if not task.done():
                    task.cancel()

    logger.info(
        "api.batch_generate_stream.request",
        num_products=len(metas),
        num_images=len(images),
    )
    return StreamingResponse(
        _event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


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

    # Reject oversize imports before reading the body into memory (falls back
    # to the post-read checks below when the size is not declared).
    declared = _declared_size(file)
    if declared is not None and declared > MAX_IMPORT_FILE_BYTES:
        raise HTTPException(
            status_code=400,
            detail=f"文件超过 {MAX_IMPORT_FILE_BYTES // (1024 * 1024)} MB 限制",
        )

    # Determine file type and parse
    if filename.endswith(".xlsx") or filename.endswith(".xls") or "spreadsheet" in content_type:
        logger.info("api.import.parse.detected_excel", filename=filename)
        content = await file.read()
        if len(content) > MAX_IMPORT_FILE_BYTES:
            raise HTTPException(
                status_code=400,
                detail=f"文件超过 {MAX_IMPORT_FILE_BYTES // (1024 * 1024)} MB 限制",
            )
        logger.info("api.import.parse.file_read", filename=filename, size_bytes=len(content))
        result = await run_in_threadpool(parse_excel, content)
    elif filename.endswith(".csv") or "csv" in content_type or "text" in content_type:
        logger.info("api.import.parse.detected_csv", filename=filename)
        raw = await file.read()
        if len(raw) > MAX_IMPORT_FILE_BYTES:
            raise HTTPException(
                status_code=400,
                detail=f"文件超过 {MAX_IMPORT_FILE_BYTES // (1024 * 1024)} MB 限制",
            )
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
