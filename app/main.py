"""FastAPI application entry point."""

import hmac
import os
from contextlib import asynccontextmanager
from pathlib import Path

# Bypass HTTP proxy for local requests to avoid connection issues
os.environ.setdefault("NO_PROXY", "*")
os.environ.setdefault("no_proxy", "*")

from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

from app import __version__
from app.api.history import router as history_router
from app.api.routes import router as api_router
from app.config import get_settings, BASE_DIR
from app.utils.images import pillow_available
from app.utils.logger import get_logger, setup_logging

logger = get_logger(__name__)

STATIC_DIR = BASE_DIR / "static"


def create_app() -> FastAPI:
    """Build and configure the FastAPI application."""
    settings = get_settings()
    setup_logging(settings.debug)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logger.info(
            "app.startup",
            app_name=settings.app_name,
            version=__version__,
            vision_mode=settings.vision_mode.value,
            llm_mode=settings.llm_mode.value,
            embedding_mode=settings.embedding_mode.value,
        )
        await _check_rag_index()
        _check_image_pipeline()
        yield
        logger.info("app.shutdown")

    def _check_image_pipeline() -> None:
        """Fail loudly (in logs) when Pillow is missing.

        ``preprocess_image`` degrades to pass-through without Pillow, so a
        missing dependency would silently disable request compression (413
        risk returns) and history thumbnails. Surface it once at startup.
        """
        if pillow_available():
            return
        logger.error(
            "image_pipeline.pillow_missing",
            impact=(
                "Pillow 未安装：发给视觉模型的图片不会压缩（413/连接中断风险回归），"
                "历史记录将保存原图。请在当前运行环境安装 pillow，"
                "或改用 `uv run uvicorn app.main:app` 以使用 uv.lock 管理的完整依赖。"
            ),
        )

    async def _check_rag_index() -> None:
        """Warn when a platform-rule collection is missing/empty at startup.

        Generation silently proceeds without rules when retrieval finds no
        collection, so a forgotten ``build_index.py`` run would go unnoticed.
        With ``rag_autobuild_on_startup`` enabled, the index is rebuilt in the
        background instead of just warning.
        """
        from app.models.listing import Platform
        from app.rag.indexer import build_index, collection_name, get_chroma_client

        try:
            client = get_chroma_client(settings)
            missing = []
            for platform in Platform:
                name = collection_name(platform.value)
                try:
                    if client.get_collection(name=name).count() == 0:
                        missing.append(name)
                except Exception:  # noqa: BLE001 - collection absent
                    missing.append(name)
        except Exception as exc:  # noqa: BLE001 - never block startup
            logger.warning("rag.startup_check_failed", error=str(exc))
            return

        if not missing:
            return
        logger.warning(
            "rag.index_missing",
            collections=missing,
            hint="run scripts/build_index.py or POST /api/v1/rag/rebuild",
        )
        if settings.rag_autobuild_on_startup:
            logger.info("rag.autobuild.start", collections=missing)
            try:
                stats = await run_in_threadpool(build_index, settings)
                logger.info("rag.autobuild.done", total_chunks=stats.total_chunks)
            except Exception as exc:  # noqa: BLE001
                logger.error("rag.autobuild.failed", error=str(exc))

    app = FastAPI(
        title=settings.app_name,
        version=__version__,
        description=(
            "Open-source multimodal agent for cross-border e-commerce "
            "listing generation: vision analysis -> RAG rules -> "
            "compliance guardrails -> multilingual listing."
        ),
        lifespan=lifespan,
    )

    app.include_router(api_router)
    app.include_router(history_router)

    @app.middleware("http")
    async def require_api_key(request: Request, call_next):
        """Optional shared-secret gate for the API.

        Enabled only when ``auth_api_key`` is set in the environment: every
        ``/api/*`` call except the health probe must present the key in the
        ``X-API-Key`` header. The static UI stays open so the page can render
        and prompt for the key client-side.
        """
        expected = settings.auth_api_key
        if expected:
            path = request.url.path
            if path.startswith("/api/") and path != "/api/v1/health":
                provided = request.headers.get("X-API-Key", "")
                if not hmac.compare_digest(provided, expected):
                    return JSONResponse(
                        {
                            "error": "unauthorized",
                            "detail": "Missing or invalid X-API-Key header.",
                        },
                        status_code=401,
                    )
        return await call_next(request)

    @app.get("/api/v1/health", tags=["system"])
    async def health() -> dict:
        """Liveness probe returning service status."""
        return {
            "status": "ok",
            "version": __version__,
            "vision_mode": settings.vision_mode.value,
            "llm_mode": settings.llm_mode.value,
            "embedding_mode": settings.embedding_mode.value,
        }

    # Serve static frontend files
    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

        @app.get("/", include_in_schema=False)
        async def serve_index():
            return FileResponse(STATIC_DIR / "index.html")

    return app


app = create_app()
