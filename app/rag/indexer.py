"""ChromaDB index construction for platform rule chunks.

Per spec: one Chroma collection per platform and one vector per rule entry.
Embeddings are produced by a pluggable :class:`Embedder` so that:
  - ``local`` mode (default) loads a local Hugging Face model via
    sentence_transformers with lazy initialisation (singleton).
  - ``mock`` mode runs fully offline with a deterministic token-hashing
    embedder, ideal for development and tests.

Embeddings are computed here and passed to Chroma explicitly, so the same
embedder is guaranteed to be used at both index time and query time.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass

from app.config import EmbeddingMode, Settings, get_settings
from app.rag.loader import RuleChunk, load_all_rules
from app.utils.logger import get_logger

logger = get_logger(__name__)

# Collection naming: one collection per platform.
COLLECTION_PREFIX = "platform_rules"

_TOKEN_RE = re.compile(r"[a-z0-9\u4e00-\u9fff]+")
_MOCK_DIM = 384

# -- Singleton lazy-loaded sentence-transformers model ----------------------
_st_model = None


def _get_st_model(model_path: str):
    """Return a cached SentenceTransformer instance (loaded on first call)."""
    global _st_model
    if _st_model is None:
        import torch
        from sentence_transformers import SentenceTransformer

        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("embedding.loading_model", path=model_path, device=device)
        _st_model = SentenceTransformer(model_path, device=device)
        logger.info("embedding.model_loaded", path=model_path, device=device)
    return _st_model


def collection_name(platform: str) -> str:
    """Return the Chroma collection name for a platform key."""
    return f"{COLLECTION_PREFIX}_{platform}"


class Embedder:
    """Turns text into fixed-size vectors.

    Args:
        settings: Settings override controlling mode, endpoint and model.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts into a list of vectors.

        For ``local`` and ``api`` modes the call is routed to the local
        sentence-transformers model (API mode is kept for backward
        compatibility but no longer makes HTTP requests).  ``mock`` mode
        uses the deterministic token-hashing embedder.
        """
        if not texts:
            return []
        if self._settings.embedding_mode == EmbeddingMode.MOCK:
            return [self._embed_mock(t) for t in texts]
        return self._embed_local(texts)

    def embed_one(self, text: str) -> list[float]:
        """Embed a single text and return its vector."""
        return self.embed([text])[0]

    # -- Mock (offline, deterministic) ---------------------------------
    @staticmethod
    def _embed_mock(text: str) -> list[float]:
        """Bag-of-token hashing embedder: deterministic and dependency-free.

        Not semantically strong, but overlapping tokens produce overlapping
        vectors, which is enough to exercise retrieval in dev and tests.
        """
        vector = [0.0] * _MOCK_DIM
        for token in _TOKEN_RE.findall(text.lower()):
            digest = hashlib.md5(token.encode("utf-8")).hexdigest()
            vector[int(digest, 16) % _MOCK_DIM] += 1.0
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]

    # -- Local (sentence-transformers, lazy singleton) -----------------
    def _embed_local(self, texts: list[str]) -> list[list[float]]:
        """Embed texts using a local Hugging Face model via sentence-transformers."""
        model = _get_st_model(self._settings.embedding_local_model_path)
        embeddings = model.encode(texts, normalize_embeddings=True)
        return embeddings.tolist()


@dataclass
class IndexStats:
    """Summary returned after building the index."""

    platforms: dict[str, int]
    total_chunks: int


def get_chroma_client(settings: Settings):
    """Open an embedded persistent Chroma client rooted at the data dir.

    Telemetry is explicitly disabled, which suits a self-hosted tool and also
    avoids noisy posthog errors in restricted environments.
    """
    import chromadb
    from chromadb.config import Settings as ChromaSettings

    settings.chroma_persist_dir.mkdir(parents=True, exist_ok=True)
    chroma_settings = ChromaSettings(anonymized_telemetry=False)
    return chromadb.PersistentClient(
        path=str(settings.chroma_persist_dir), settings=chroma_settings
    )


def build_index(
    settings: Settings | None = None,
    chunks: list[RuleChunk] | None = None,
    rebuild: bool = True,
) -> IndexStats:
    """Load rule documents, embed them and upsert into per-platform collections.

    Args:
        settings: Optional settings override.
        chunks: Pre-loaded chunks; when None they are loaded from disk.
        rebuild: When True, existing platform collections are dropped first so
            the index mirrors the current documents exactly.

    Returns:
        An IndexStats summary of what was written.
    """
    settings = settings or get_settings()
    if chunks is None:
        chunks = load_all_rules(settings)
    if not chunks:
        logger.warning("index.no_chunks")
        return IndexStats(platforms={}, total_chunks=0)

    client = get_chroma_client(settings)
    embedder = Embedder(settings)

    by_platform: dict[str, list[RuleChunk]] = {}
    for chunk in chunks:
        by_platform.setdefault(chunk.platform, []).append(chunk)

    stats_platforms: dict[str, int] = {}
    for platform, platform_chunks in by_platform.items():
        name = collection_name(platform)
        if rebuild:
            try:
                client.delete_collection(name)
            except Exception:  # pragma: no cover - collection may not exist yet
                pass

        collection = client.get_or_create_collection(
            name=name, metadata={"hnsw:space": "cosine"}
        )

        texts = [c.full_text for c in platform_chunks]
        embeddings = embedder.embed(texts)
        collection.upsert(
            ids=[c.chunk_id for c in platform_chunks],
            embeddings=embeddings,
            documents=texts,
            metadatas=[c.metadata for c in platform_chunks],
        )
        stats_platforms[platform] = len(platform_chunks)
        logger.info("index.collection_written", collection=name, chunks=len(platform_chunks))

    total = sum(stats_platforms.values())
    logger.info("index.complete", total_chunks=total, platforms=stats_platforms)
    return IndexStats(platforms=stats_platforms, total_chunks=total)
