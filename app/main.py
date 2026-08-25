"""FastAPI application entry point."""

import os
from contextlib import asynccontextmanager
from pathlib import Path

# Bypass HTTP proxy for local requests to avoid connection issues
os.environ.setdefault("NO_PROXY", "*")
os.environ.setdefault("no_proxy", "*")

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from app import __version__
from app.api.history import router as history_router
from app.api.routes import router as api_router
from app.config import get_settings, BASE_DIR
from app.utils.logger import get_logger, setup_logging

STATIC_DIR = BASE_DIR / "static"


def create_app() -> FastAPI:
    """Build and configure the FastAPI application."""
    settings = get_settings()
    setup_logging(settings.debug)
    logger = get_logger(__name__)

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
        yield
        logger.info("app.shutdown")

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
