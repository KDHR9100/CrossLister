"""Phase 3 tests for the RAG loader, indexer and retriever (mock embeddings).

These tests run fully offline: they use the deterministic mock embedder and a
temporary Chroma persistence directory so nothing touches the real data store.
"""

from pathlib import Path

import pytest

from app.config import EmbeddingMode, Settings
from app.rag.indexer import Embedder, build_index, collection_name
from app.rag.loader import RuleChunk, chunk_document, detect_platform, load_all_rules
from app.rag.retriever import RuleRetriever, build_query

# -- Sample fixtures ---------------------------------------------------------

_AMAZON_DOC = """\
# Amazon Test Policy

Intro text that should be discarded because it precedes any rule heading.

## AMZ-TITLE-01: Maximum title length

Titles must not exceed 200 characters including spaces. Aim for 80 characters.

## AMZ-TITLE-02: Prohibited content in titles

Titles must not contain promotional phrases such as sale or best seller.
"""

_SHOPEE_DOC = """\
# Shopee Test Policy

## SPE-TITLE-01: Title format

Titles are limited to 120 characters and should follow brand then product name.
"""


@pytest.fixture()
def rules_dir(tmp_path: Path) -> Path:
    """Create a temporary rules directory with controlled sample documents."""
    d = tmp_path / "platform_rules"
    d.mkdir()
    (d / "amazon_listing_policy.md").write_text(_AMAZON_DOC, encoding="utf-8")
    (d / "shopee_prohibited_items.md").write_text(_SHOPEE_DOC, encoding="utf-8")
    # An unknown-platform file that must be skipped.
    (d / "ebay_rules.md").write_text("## EBAY-01: X\n\nbody", encoding="utf-8")
    # An unsupported extension that must be skipped.
    (d / "amazon_notes.docx").write_text("binary-ish", encoding="utf-8")
    return d


@pytest.fixture()
def settings(tmp_path: Path, rules_dir: Path) -> Settings:
    """Settings pinned to the temp dirs with the offline mock embedder."""
    return Settings(
        platform_rules_dir=rules_dir,
        chroma_persist_dir=tmp_path / "chroma",
        embedding_mode=EmbeddingMode.MOCK,
        rag_top_k=4,
    )


# -- Loader ------------------------------------------------------------------


def test_detect_platform_from_filename() -> None:
    assert detect_platform(Path("amazon_listing_policy.md")) == "amazon"
    assert detect_platform(Path("shopee_prohibited_items.md")) == "shopee"
    assert detect_platform(Path("temu_content_guidelines.md")) == "temu"
    assert detect_platform(Path("ebay_rules.md")) is None


def test_chunk_document_splits_per_rule_and_drops_intro(rules_dir: Path) -> None:
    chunks = chunk_document(rules_dir / "amazon_listing_policy.md", "amazon")
    assert [c.rule_id for c in chunks] == ["AMZ-TITLE-01", "AMZ-TITLE-02"]
    assert chunks[0].title == "Maximum title length"
    assert "200 characters" in chunks[0].text
    # Intro text must not leak into any chunk.
    assert all("Intro text" not in c.text for c in chunks)
    # full_text combines title and body for embedding.
    assert chunks[0].full_text.startswith("Maximum title length")


def test_load_all_rules_skips_unknown_and_unsupported(settings: Settings) -> None:
    chunks = load_all_rules(settings)
    platforms = {c.platform for c in chunks}
    assert platforms == {"amazon", "shopee"}
    assert len(chunks) == 3  # 2 amazon + 1 shopee; ebay + docx skipped
    assert all(c.chunk_id.startswith(c.platform + ":") for c in chunks)


def test_load_real_shipped_documents() -> None:
    """The bundled Amazon/Shopee/Temu documents must all parse into chunks."""
    chunks = load_all_rules()  # uses default settings -> data/platform_rules
    platforms = {c.platform for c in chunks}
    assert platforms == {"amazon", "shopee", "temu"}
    assert len(chunks) >= 10
    assert all(c.rule_id and c.title and c.text for c in chunks)


# -- Embedder ----------------------------------------------------------------


def test_mock_embedder_is_deterministic_and_normalized() -> None:
    embedder = Embedder(Settings(embedding_mode=EmbeddingMode.MOCK))
    a = embedder.embed_one("stackable storage bins")
    b = embedder.embed_one("stackable storage bins")
    assert a == b
    assert len(a) == 384
    norm = sum(v * v for v in a) ** 0.5
    assert abs(norm - 1.0) < 1e-6


def test_build_query_joins_category_and_terms() -> None:
    assert build_query("家居收纳", ["stackable", "dust-proof"]) == (
        "家居收纳 stackable dust-proof"
    )
    assert build_query("家居收纳") == "家居收纳"


# -- Indexer + Retriever (end to end, offline) ------------------------------


def test_build_index_and_retrieve_top_rule(settings: Settings) -> None:
    stats = build_index(settings, rebuild=True)
    assert stats.total_chunks == 3
    assert stats.platforms == {"amazon": 2, "shopee": 1}

    retriever = RuleRetriever(settings)
    results = retriever.retrieve("amazon", "Maximum title length", top_k=2)
    assert len(results) == 2
    # The rule whose title exactly matches the query must rank first.
    assert results[0].rule_id == "AMZ-TITLE-01"
    assert results[0].platform == "amazon"
    assert results[0].score >= results[1].score
    assert results[0].text  # document text round-trips through the store


def test_retrieve_respects_platform_isolation(settings: Settings) -> None:
    build_index(settings, rebuild=True)
    retriever = RuleRetriever(settings)
    shopee_results = retriever.retrieve("shopee", "Title format", top_k=5)
    assert {r.rule_id for r in shopee_results} == {"SPE-TITLE-01"}
    # Amazon rules must not leak into the Shopee collection.
    assert all(r.platform == "shopee" for r in shopee_results)


def test_retrieve_missing_collection_returns_empty(settings: Settings) -> None:
    # No index built at all -> collection does not exist.
    retriever = RuleRetriever(settings)
    assert retriever.retrieve("temu", "anything") == []


def test_collection_naming() -> None:
    assert collection_name("amazon") == "platform_rules_amazon"
    assert collection_name("temu") == "platform_rules_temu"


def test_build_index_with_no_chunks(settings: Settings, tmp_path: Path) -> None:
    empty_dir = tmp_path / "empty_rules"
    empty_dir.mkdir()
    empty_settings = Settings(
        platform_rules_dir=empty_dir,
        chroma_persist_dir=tmp_path / "chroma_empty",
        embedding_mode=EmbeddingMode.MOCK,
    )
    stats = build_index(empty_settings)
    assert stats.total_chunks == 0
    assert stats.platforms == {}
