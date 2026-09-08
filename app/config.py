"""Application configuration backed by Pydantic Settings.

All settings can be overridden via environment variables or a local `.env`
file (see `.env.example`).
"""

from enum import Enum
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Repository root (parent of the `app/` package).
BASE_DIR = Path(__file__).resolve().parent.parent


class VisionMode(str, Enum):
    """Vision model invocation mode.

    local: self-hosted vLLM server exposing the OpenAI-compatible API.
    api:   any cloud endpoint compatible with the OpenAI Vision protocol.
    mock:  deterministic stub used for development and tests.
    """

    LOCAL = "local"
    API = "api"
    MOCK = "mock"


class EmbeddingMode(str, Enum):
    """Embedding generation mode for the RAG index.

    local: load a local Hugging Face model via sentence_transformers (default).
    api:   kept for backward compatibility, now also routes to local model.
    mock:  deterministic offline token-hashing embedder for dev and tests.
    """

    LOCAL = "local"
    API = "api"
    MOCK = "mock"


class LLMMode(str, Enum):
    """Text LLM invocation mode for generation / compliance / translation.

    api:  call an OpenAI-compatible chat-completions endpoint.
    mock: deterministic stub used for development and tests.
    """

    API = "api"
    MOCK = "mock"


class VideoMode(str, Enum):
    """Image-to-video generation mode.

    api:  call the DashScope-style async video-synthesis endpoint on the
          same MaaS gateway as the vision/LLM models.
    mock: deterministic stub used for development and tests (no network).
    """

    API = "api"
    MOCK = "mock"


class Settings(BaseSettings):
    """Global application settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # -- Application ---------------------------------------------------
    app_name: str = "CrossLister"
    debug: bool = False

    # -- Vision model --------------------------------------------------
    vision_mode: VisionMode = VisionMode.MOCK
    vision_api_base: str = "https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
    vision_api_key: str = "EMPTY"
    vision_model: str = "qwen3.6-flash"
    vision_max_images: int = 20
    vision_timeout_s: float = 120.0
    # Downscale each image so its longest side is at most this many pixels,
    # then re-encode as JPEG at the given quality. Keeps the request body small
    # enough for the remote gateway (avoids 413 / connection resets on upload).
    # Set vision_max_image_side to 0 to disable downscaling.
    vision_max_image_side: int = 1280
    vision_jpeg_quality: int = 85

    # -- Text LLM (listing generation / compliance check) --------------
    llm_mode: LLMMode = LLMMode.MOCK
    llm_api_base: str = "https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
    llm_api_key: str = "EMPTY"
    llm_model: str = "qwen3.6-flash"
    llm_timeout_s: float = 120.0
    # Transient-error retries for LLM calls (connection drops, rate limits, 5xx).
    llm_max_retries: int = 3
    # Upper bound on generated tokens per LLM call (0 disables the cap).
    llm_max_output_tokens: int = 2048

    # -- Batch generation ------------------------------------------------
    # Cap on how many products are generated concurrently. Keeps pressure off
    # the remote LLM endpoint (avoids rate-limit / connection-reset storms).
    batch_max_concurrency: int = 8
    # Per-product timeout (seconds). A single product that exceeds this is
    # marked as failed so it cannot block the rest of the batch forever.
    batch_product_timeout_s: float = 300.0
    # Hard caps on one batch request. Without them a runaway client could
    # submit thousands of multipart parts, exhausting memory/CPU during
    # parsing long before any rate limit kicks in. 100+ images in one batch
    # is comfortably inside these defaults.
    batch_max_products: int = 200
    batch_max_images: int = 1000

    # -- Video generation (image-to-video) -------------------------------
    # Side feature: turn the first product image into a short marketing
    # video. Uses the DashScope-style async task API on the same gateway:
    #   POST {video_api_base}/api/v1/services/aigc/video-generation/video-synthesis
    #   GET  {video_api_base}/api/v1/tasks/{task_id}
    # Note video_api_base is the gateway ROOT (no /compatible-mode suffix).
    video_mode: VideoMode = VideoMode.MOCK
    video_api_base: str = "https://token-plan.cn-beijing.maas.aliyuncs.com"
    # Empty key falls back to vision_api_key so one credential serves all.
    video_api_key: str = ""
    video_model: str = "happyhorse-1.1-i2v"
    video_duration_s: int = 5
    video_resolution: str = "720P"
    # Wall-clock budget per video covering create + poll + download.
    video_timeout_s: float = 900.0
    video_poll_interval_s: float = 5.0
    # Videos are heavy: bound how many generate at once (separate from the
    # listing batch semaphore so slow videos never block listing slots).
    video_max_concurrency: int = 3
    # Finished MP4s are downloaded here (remote URLs expire after 24h) and
    # served by GET /api/v1/video/{filename}. Oldest files are pruned beyond
    # this count to bound disk usage.
    video_output_dir: Path = BASE_DIR / "data" / "videos"
    video_max_output_files: int = 400

    # -- RAG -----------------------------------------------------------
    platform_rules_dir: Path = BASE_DIR / "data" / "platform_rules"
    chroma_persist_dir: Path = BASE_DIR / "data" / "chroma"
    embedding_mode: EmbeddingMode = EmbeddingMode.LOCAL
    embedding_model: str = "Qwen/Qwen3-Embedding-0.6B"
    embedding_local_model_path: str = "Qwen/Qwen3-Embedding-0.6B"
    embedding_api_base: str = "http://localhost:8000/v1"
    embedding_api_key: str = "EMPTY"
    rag_top_k: int = 4
    # Drop retrieved rules whose similarity falls below this floor
    # (cosine similarity mapped to [-1, 1]; default 0 keeps everything that is
    # not anti-correlated, so only clearly unrelated chunks are filtered).
    rag_min_score: float = 0.0
    # When true, a missing/empty platform-rule collection detected at startup
    # is rebuilt automatically in the background (embedding model permitting).
    rag_autobuild_on_startup: bool = False

    # -- Compliance guardrails ------------------------------------------
    max_compliance_retries: int = 3

    # -- Generation history (cold storage) -------------------------------
    # Records are written to disk *after* generation completes, in a
    # fire-and-forget task; the history store never participates in the
    # live generation pipeline.
    history_enabled: bool = True
    history_dir: Path = BASE_DIR / "data" / "history"
    # Save (compressed) product images alongside the text. Disable to store
    # text-only records and minimize disk usage.
    history_save_images: bool = True
    # Oldest records are pruned once the count exceeds this cap.
    history_max_records: int = 200

    # -- Security ----------------------------------------------------------
    # When set, every /api/* call (except /api/v1/health) must present this
    # shared secret in the X-API-Key header. Empty string disables auth,
    # keeping the zero-config single-user experience.
    auth_api_key: str = ""


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance."""
    return Settings()
