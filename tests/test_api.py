"""API tests: health probe plus end-to-end listing generation in mock mode.

Everything runs offline (mock vision / LLM / embeddings) against the real
Chroma index built from the bundled rule documents.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.config import get_settings
from app.main import app
from app.rag.indexer import build_index

client = TestClient(app)

# Minimal valid 1x1 transparent PNG used for multipart uploads.
_PNG_1X1 = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01"
    b"\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
)


@pytest.fixture(scope="module", autouse=True)
def ensure_index():
    """Build the default rule index once before the API tests run."""
    build_index(get_settings())
    yield


def test_health_returns_ok() -> None:
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "version" in body
    assert body["vision_mode"] in {"local", "api", "mock"}
    assert body["llm_mode"] in {"api", "mock"}
    assert body["embedding_mode"] in {"api", "mock"}


def test_generate_listing_end_to_end() -> None:
    resp = client.post(
        "/api/v1/listing/generate",
        files=[("images", ("product.png", _PNG_1X1, "image/png"))],
        data={"category": "storage organizer", "platform": "amazon"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["title"]
    assert body["bullet_points"]
    assert body["compliance"]["passed"] is True
    assert body["compliance"]["attempts"] == 1
    assert body["metadata"]["rag_chunks_used"] > 0
    assert body["visual_analysis"]["detected_category"] == "storage organizer"


def test_generate_listing_compliance_loop() -> None:
    resp = client.post(
        "/api/v1/listing/generate",
        files=[("images", ("product.png", _PNG_1X1, "image/png"))],
        data={
            "category": "storage organizer",
            "platform": "amazon",
            "extra_info": '{"force_violation": true}',
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["compliance"]["passed"] is True
    assert body["compliance"]["attempts"] == 2


def test_generate_rejects_too_many_images() -> None:
    files = [
        ("images", (f"img{i}.png", _PNG_1X1, "image/png")) for i in range(6)
    ]
    resp = client.post(
        "/api/v1/listing/generate",
        files=files,
        data={"category": "storage organizer"},
    )
    assert resp.status_code == 400


def test_generate_rejects_invalid_extra_info() -> None:
    resp = client.post(
        "/api/v1/listing/generate",
        files=[("images", ("product.png", _PNG_1X1, "image/png"))],
        data={"category": "storage organizer", "extra_info": "not-json"},
    )
    assert resp.status_code == 400


def test_generate_requires_images() -> None:
    resp = client.post(
        "/api/v1/listing/generate",
        data={"category": "storage organizer"},
    )
    assert resp.status_code == 422


def test_generate_rejects_unknown_platform() -> None:
    resp = client.post(
        "/api/v1/listing/generate",
        files=[("images", ("product.png", _PNG_1X1, "image/png"))],
        data={"category": "storage organizer", "platform": "ebay"},
    )
    assert resp.status_code == 422


def test_rag_rebuild() -> None:
    resp = client.post("/api/v1/rag/rebuild")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert set(body["platforms"]) == {"amazon", "shopee", "temu"}
    assert body["total_chunks"] == 30
