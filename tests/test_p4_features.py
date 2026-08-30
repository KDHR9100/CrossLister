"""Tests for P4 features: RAG score threshold, history management, API auth."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import EmbeddingMode, Settings, get_settings
from app.history import store as history_store
from app.main import app
from app.rag.indexer import build_index
from app.rag.retriever import RuleRetriever

client = TestClient(app)

# -- T11: rag_min_score ------------------------------------------------------

_AMAZON_DOC = """\
# Amazon Test Policy

## AMZ-TITLE-01: Maximum title length

Titles must not exceed 200 characters including spaces. Aim for 80 characters.

## AMZ-DESC-01: Description limits

Descriptions are limited to 2000 characters.
"""


@pytest.fixture()
def rag_env(tmp_path: Path) -> Settings:
    rules_dir = tmp_path / "platform_rules"
    rules_dir.mkdir()
    (rules_dir / "amazon_listing_policy.md").write_text(_AMAZON_DOC, encoding="utf-8")
    settings = Settings(
        platform_rules_dir=rules_dir,
        chroma_persist_dir=tmp_path / "chroma",
        embedding_mode=EmbeddingMode.MOCK,
    )
    build_index(settings)
    return settings


def test_min_score_zero_keeps_relevant_rules(rag_env: Settings) -> None:
    rules = RuleRetriever(rag_env).retrieve("amazon", "title length characters")
    assert rules
    assert all(r.score >= 0.0 for r in rules)


def test_min_score_above_best_match_drops_everything(rag_env: Settings) -> None:
    rules = RuleRetriever(rag_env).retrieve("amazon", "title length characters")
    floor = max(r.score for r in rules) + 0.01
    strict = Settings(**{**rag_env.model_dump(), "rag_min_score": floor})
    assert RuleRetriever(strict).retrieve("amazon", "title length characters") == []


# -- T12: history delete / rebuild / filters ---------------------------------


def _seed_records(history_dir: Path, n: int = 3) -> list[str]:
    ids = []
    for i in range(n):
        rid = history_store.save_record(
            history_dir,
            api="listing/generate",
            products=[
                {
                    "product_index": 0,
                    "category": "cat",
                    "platform": "amazon" if i % 2 == 0 else "shopee",
                    "target_lang": "en",
                    "status": "success",
                    "listing": {"title": f"Product {i}"},
                    "error": None,
                    "elapsed_ms": 10,
                }
            ],
            images=None,
        )
        ids.append(rid)
    return ids


def test_delete_record_removes_dir_and_index_entry(tmp_path: Path) -> None:
    history_dir = tmp_path / "history"
    rid = _seed_records(history_dir)[0]

    assert history_store.delete_record(history_dir, rid) is True
    assert not (history_dir / rid).exists()
    remaining = history_store.list_records(history_dir)
    assert all(r["record_id"] != rid for r in remaining)


def test_delete_record_rejects_bad_ids(tmp_path: Path) -> None:
    assert history_store.delete_record(tmp_path, "../../etc") is False
    assert history_store.delete_record(tmp_path, "not-an-id") is False


def test_rebuild_index_recovers_summaries_from_record_dirs(tmp_path: Path) -> None:
    history_dir = tmp_path / "history"
    ids = _seed_records(history_dir, n=3)
    (history_dir / "index.json").unlink()  # simulate corruption/loss

    count = history_store.rebuild_index(history_dir)
    assert count == 3
    listed = history_store.list_records(history_dir)
    assert [r["record_id"] for r in listed] == sorted(ids, reverse=True)


def test_corrupt_index_is_rebuilt_on_list(tmp_path: Path) -> None:
    history_dir = tmp_path / "history"
    _seed_records(history_dir, n=2)
    (history_dir / "index.json").write_text("{not json", encoding="utf-8")

    records = history_store.list_records(history_dir)
    assert len(records) == 2


def test_list_records_filters_by_platform_and_status(tmp_path: Path) -> None:
    history_dir = tmp_path / "history"
    _seed_records(history_dir, n=4)

    amazon_only = history_store.list_records(history_dir, platform="amazon")
    assert amazon_only and all(
        "amazon" in r["platforms"] for r in amazon_only
    )

    ok_only = history_store.list_records(history_dir, status="success")
    assert len(ok_only) == 4  # every seeded product succeeded

    failed_only = history_store.list_records(history_dir, status="failed")
    assert failed_only == []


def test_delete_api_endpoint(tmp_path: Path, monkeypatch) -> None:
    settings = get_settings()
    monkeypatch.setattr(settings, "history_enabled", True)
    monkeypatch.setattr(settings, "history_dir", tmp_path / "history")
    rid = _seed_records(settings.history_dir)[0]

    resp = client.delete(f"/api/v1/history/{rid}")
    assert resp.status_code == 200
    assert resp.json()["record_id"] == rid
    assert client.delete(f"/api/v1/history/{rid}").status_code == 404
    assert client.delete("/api/v1/history/../../etc").status_code == 404


def test_list_api_endpoint_supports_filters(tmp_path: Path, monkeypatch) -> None:
    settings = get_settings()
    monkeypatch.setattr(settings, "history_enabled", True)
    monkeypatch.setattr(settings, "history_dir", tmp_path / "history")
    _seed_records(settings.history_dir, n=2)

    resp = client.get("/api/v1/history", params={"platform": "shopee"})
    assert resp.status_code == 200
    records = resp.json()["records"]
    assert records and all("shopee" in r["platforms"] for r in records)

    bad = client.get("/api/v1/history", params={"status": "bogus"})
    assert bad.status_code == 422


# -- T13: optional API-key auth ----------------------------------------------


def test_auth_disabled_by_default() -> None:
    assert client.get("/api/v1/platforms").status_code == 200


def test_auth_enforced_when_key_configured(monkeypatch) -> None:
    settings = get_settings()
    monkeypatch.setattr(settings, "auth_api_key", "secret-key-123")

    # API endpoints require the key; the health probe and static UI stay open.
    assert client.get("/api/v1/platforms").status_code == 401
    assert client.get("/api/v1/health").status_code == 200
    assert client.get("/api/v1/platforms", headers={"X-API-Key": "wrong"}).status_code == 401
    ok = client.get("/api/v1/platforms", headers={"X-API-Key": "secret-key-123"})
    assert ok.status_code == 200

    # Unknown paths under /api/ are gated too.
    assert client.get("/api/v1/nope", headers={"X-API-Key": "secret-key-123"}).status_code == 404
    assert client.get("/api/v1/nope").status_code == 401


def test_health_payload_shape_unchanged() -> None:
    body = client.get("/api/v1/health").json()
    assert body["status"] == "ok"
    assert set(body) >= {"status", "version", "vision_mode", "llm_mode", "embedding_mode"}
