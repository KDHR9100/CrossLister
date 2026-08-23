"""Retrieval interface over the per-platform rule collections.

Given a target platform and a natural-language query (typically built from the
product category plus detected selling points), returns the top-k most relevant
rule chunks. The query is embedded with the same :class:`Embedder` used at
index time so vectors are directly comparable.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.config import Settings, get_settings
from app.rag.indexer import Embedder, collection_name, get_chroma_client
from app.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class RetrievedRule:
    """A single rule chunk returned by retrieval, with its relevance score."""

    rule_id: str
    title: str
    text: str
    platform: str
    source: str
    score: float
    metadata: dict[str, str] = field(default_factory=dict)


def build_query(category: str, extra_terms: list[str] | None = None) -> str:
    """Compose a retrieval query from the category and optional extra terms.

    Args:
        category: The product category, e.g. "家居收纳".
        extra_terms: Optional extra signals such as detected selling points.

    Returns:
        A single query string used for embedding.
    """
    parts = [category] + [t for t in (extra_terms or []) if t]
    return " ".join(p.strip() for p in parts if p and p.strip())


class RuleRetriever:
    """Top-k rule retrieval against the Chroma platform collections.

    Args:
        settings: Optional settings override (client, embedder, top-k).
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._embedder = Embedder(self._settings)
        # Cache the persistent Chroma client so we don't reopen the store and
        # its file handles on every retrieve() call.
        self._client = get_chroma_client(self._settings)

    def retrieve(
        self,
        platform: str,
        query: str,
        top_k: int | None = None,
    ) -> list[RetrievedRule]:
        """Return the top-k most relevant rule chunks for a platform and query.

        Args:
            platform: Canonical platform key (amazon / shopee / temu).
            query: Natural-language query describing the product.
            top_k: Number of results; defaults to settings.rag_top_k.

        Returns:
            A list of RetrievedRule sorted by descending relevance. Empty when
            the collection is missing or has no rows.
        """
        k = top_k or self._settings.rag_top_k
        name = collection_name(platform)
        client = get_chroma_client(self._settings)

        try:
            collection = client.get_collection(name=name)
        except Exception:
            logger.warning("rag.collection_missing", collection=name)
            return []

        count = collection.count()
        if count == 0:
            logger.warning("rag.collection_empty", collection=name)
            return []

        k = max(1, min(k, count))
        query_embedding = self._embedder.embed_one(query)
        result = collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )

        rules: list[RetrievedRule] = []
        ids = result.get("ids", [[]])[0]
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]

        for idx, _id in enumerate(ids):
            metadata = metadatas[idx] if idx < len(metadatas) else {}
            distance = distances[idx] if idx < len(distances) else 1.0
            metadata = metadata or {}
            # Chroma cosine distance is in [0, 2]; map to similarity in [-1, 1].
            score = 1.0 - float(distance)
            rules.append(
                RetrievedRule(
                    rule_id=metadata.get("rule_id", _id),
                    title=_title_from_document(documents[idx] if idx < len(documents) else ""),
                    text=documents[idx] if idx < len(documents) else "",
                    platform=platform,
                    source=metadata.get("source", ""),
                    score=score,
                    metadata=dict(metadata),
                )
            )

        logger.info("rag.retrieved", collection=name, query=query, results=len(rules))
        return rules


def _title_from_document(document: str) -> str:
    """Recover the rule title (first line) from a stored document string."""
    if not document:
        return ""
    return document.splitlines()[0].strip()
