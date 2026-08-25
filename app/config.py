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
    vision_api_base: str = "http://localhost:8000/v1"
    vision_api_key: str = "EMPTY"
    vision_model: str = "Qwen/Qwen2.5-VL-7B-Instruct"
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
    llm_api_base: str = "http://localhost:8000/v1"
    llm_api_key: str = "EMPTY"
    llm_model: str = "Qwen/Qwen2.5-7B-Instruct"
    llm_timeout_s: float = 120.0
    # Transient-error retries for LLM calls (connection drops, rate limits, 5xx).
    llm_max_retries: int = 3

    # -- Batch generation ------------------------------------------------
    # Cap on how many products are generated concurrently. Keeps pressure off
    # the remote LLM endpoint (avoids rate-limit / connection-reset storms).
    batch_max_concurrency: int = 8
    # Per-product timeout (seconds). A single product that exceeds this is
    # marked as failed so it cannot block the rest of the batch forever.
    batch_product_timeout_s: float = 300.0

    # -- RAG -----------------------------------------------------------
    platform_rules_dir: Path = BASE_DIR / "data" / "platform_rules"
    chroma_persist_dir: Path = BASE_DIR / "data" / "chroma"
    embedding_mode: EmbeddingMode = EmbeddingMode.LOCAL
    embedding_model: str = "Qwen/Qwen3-Embedding-0.6B"
    embedding_local_model_path: str = "Qwen/Qwen3-Embedding-0.6B"
    embedding_api_base: str = "http://localhost:8000/v1"
    embedding_api_key: str = "EMPTY"
    rag_top_k: int = 4

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


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance."""
    return Settings()
