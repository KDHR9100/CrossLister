"""Tests for the generation history cold storage (store + read-only API).

The history system is deliberately decoupled from the generation pipeline,
so these tests exercise the store directly and pre-populate records before
hitting the read-only endpoints — no dependency on generation timing.
"""

from __future__ import annotations

import asyncio
import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.config import get_settings
from app.history import store as history_store
from app.main import app

client = TestClient(app)


def _jpeg_bytes(color: tuple[int, int, int] = (200, 30, 30)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (16, 16), color).save(buf, format="JPEG")
    return buf.getvalue()


def _product(index: int = 0, status: str = "success") -> dict:
    return {
        "product_index": index,
        "category": "storage organizer",
        "platform": "amazon",
        "target_lang": "en",
        "status": status,
        "listing": None
        if status == "failed"
        else {"title": f"Test Listing {index}", "bullet_points": ["b1"]},
        "error": None if status == "success" else "boom",
        "elapsed_ms": 1234,
    }


@pytest.fixture()
def history_dir(tmp_path):
    """Point the app at a throwaway history directory for one test."""
    settings = get_settings()
    original_dir, original_enabled = settings.history_dir, settings.history_enabled
    settings.history_dir = tmp_path
    settings.history_enabled = True
    yield tmp_path
    settings.history_dir, settings.history_enabled = original_dir, original_enabled


# ---------------- Store: write path ----------------


def test_save_record_writes_text_and_images(history_dir):
    record_id = history_store.save_record(
        history_dir,
        "listing/batch_generate",
        [_product(0), _product(1)],
        images={0: [_jpeg_bytes()], 1: [_jpeg_bytes(), _jpeg_bytes((0, 90, 200))]},
        save_images=True,
    )
    rdir = history_dir / record_id
    assert (rdir / "record.json").exists()
    assert (rdir / "images" / "p0_0.jpg").exists()
    assert (rdir / "images" / "p1_0.jpg").exists()
    assert (rdir / "images" / "p1_1.jpg").exists()
    # Stored copies are re-encoded JPEGs, not the raw input.
    assert (rdir / "images" / "p0_0.jpg").read_bytes()[:3] == b"\xff\xd8\xff"

    record = history_store.get_record(history_dir, record_id)
    assert record is not None
    assert record["api"] == "listing/batch_generate"
    assert record["products"][0]["images"] == ["p0_0.jpg"]
    assert record["products"][0]["image_count"] == 1
    assert record["products"][1]["image_count"] == 2


def test_save_record_text_only_skips_images(history_dir):
    record_id = history_store.save_record(
        history_dir,
        "listing/generate",
        [_product(0)],
        images={0: [_jpeg_bytes()]},
        save_images=False,
    )
    assert not (history_dir / record_id / "images").exists()
    record = history_store.get_record(history_dir, record_id)
    assert record["products"][0]["images"] == []
    # The count still reflects what was generated, even if not stored.
    assert record["products"][0]["image_count"] == 1


def test_save_prunes_oldest_records(history_dir):
    ids = [
        history_store.save_record(
            history_dir, "listing/generate", [_product(i)], max_records=2
        )
        for i in range(3)
    ]
    # Oldest record directory is removed; index keeps only the newest two.
    assert not (history_dir / ids[0]).exists()
    assert (history_dir / ids[1] / "record.json").exists()
    assert (history_dir / ids[2] / "record.json").exists()
    summaries = history_store.list_records(history_dir)
    assert [s["record_id"] for s in summaries] == [ids[2], ids[1]]


def test_save_record_async_returns_id_and_never_raises(history_dir):
    record_id = asyncio.run(
        history_store.save_record_async(
            history_dir, "listing/generate", [_product(0)], {0: [_jpeg_bytes()]}
        )
    )
    assert record_id is not None
    assert (history_dir / record_id / "record.json").exists()

    # Corrupt image bytes degrade gracefully (stored as-is), no exception.
    degraded = asyncio.run(
        history_store.save_record_async(
            history_dir, "listing/generate", [_product(0)], {0: [b"not-an-image"]}
        )
    )
    assert degraded is not None

    # A genuinely broken payload (non-JSON-serializable) must not propagate.
    bad = asyncio.run(
        history_store.save_record_async(
            history_dir,
            "listing/generate",
            [{"product_index": 0, "listing": {"bad": object()}}],
        )
    )
    assert bad is None


# ---------------- Store: read path & traversal defence ----------------


def test_list_records_newest_first_and_limit(history_dir):
    ids = [
        history_store.save_record(history_dir, "listing/generate", [_product(i)])
        for i in range(3)
    ]
    summaries = history_store.list_records(history_dir, limit=2)
    assert len(summaries) == 2
    assert summaries[0]["record_id"] == ids[-1]
    assert summaries[0]["product_count"] == 1
    assert summaries[0]["platforms"] == ["amazon"]


def test_get_record_rejects_invalid_ids(history_dir):
    history_store.save_record(history_dir, "listing/generate", [_product(0)])
    assert history_store.get_record(history_dir, "../etc/passwd") is None
    assert history_store.get_record(history_dir, "not-a-record-id") is None
    assert history_store.get_record(history_dir, "99999999-999999-abcdef") is None


def test_get_image_path_rejects_traversal(history_dir):
    record_id = history_store.save_record(
        history_dir,
        "listing/generate",
        [_product(0)],
        images={0: [_jpeg_bytes()]},
        save_images=True,
    )
    ok = history_store.get_image_path(history_dir, record_id, "p0_0.jpg")
    assert ok is not None and ok.exists()
    assert history_store.get_image_path(history_dir, record_id, "../record.json") is None
    assert history_store.get_image_path(history_dir, record_id, "p0_0.png") is None
    assert history_store.get_image_path(history_dir, "../secret", "p0_0.jpg") is None


# ---------------- Read-only HTTP API ----------------


def test_history_list_endpoint(history_dir):
    history_store.save_record(history_dir, "listing/batch_generate", [_product(0)])
    resp = client.get("/api/v1/history")
    assert resp.status_code == 200
    body = resp.json()
    assert body["enabled"] is True
    assert len(body["records"]) == 1
    assert body["records"][0]["api"] == "listing/batch_generate"


def test_history_list_endpoint_disabled(history_dir):
    get_settings().history_enabled = False
    resp = client.get("/api/v1/history")
    assert resp.status_code == 200
    assert resp.json() == {"enabled": False, "records": []}


def test_history_detail_and_image_endpoints(history_dir):
    record_id = history_store.save_record(
        history_dir,
        "listing/generate",
        [_product(0)],
        images={0: [_jpeg_bytes()]},
        save_images=True,
    )
    detail = client.get(f"/api/v1/history/{record_id}")
    assert detail.status_code == 200
    assert detail.json()["record_id"] == record_id

    img = client.get(f"/api/v1/history/{record_id}/images/p0_0.jpg")
    assert img.status_code == 200
    assert img.headers["content-type"] == "image/jpeg"
    assert img.content[:3] == b"\xff\xd8\xff"

    # Missing record / image and traversal attempts all map to 404.
    assert client.get("/api/v1/history/99999999-999999-abcdef").status_code == 404
    assert client.get(f"/api/v1/history/{record_id}/images/p9_9.jpg").status_code == 404
    assert client.get("/api/v1/history/..%2fsecret").status_code == 404
