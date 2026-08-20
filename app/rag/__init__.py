"""RAG module: platform rule loading, indexing and retrieval."""

from app.rag.indexer import Embedder, IndexStats, build_index, collection_name
from app.rag.loader import RuleChunk, chunk_document, detect_platform, load_all_rules
from app.rag.retriever import RetrievedRule, RuleRetriever, build_query

__all__ = [
    "Embedder",
    "IndexStats",
    "build_index",
    "collection_name",
    "RuleChunk",
    "chunk_document",
    "detect_platform",
    "load_all_rules",
    "RetrievedRule",
    "RuleRetriever",
    "build_query",
]

