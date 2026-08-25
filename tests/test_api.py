"""API tests: health probe plus end-to-end listing generation in mock mode.

Everything runs offline (mock vision / LLM / embeddings) against the real
Chroma index built from the bundled rule documents.
"""

from __future__ import annotations

import json

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
    # Token accounting is present (0 in mock mode, real counts in API mode).
    assert body["metadata"]["total_tokens"] >= 0
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
    settings = get_settings()
    original = settings.vision_max_images
    settings.vision_max_images = 2
    try:
        files = [
            ("images", (f"img{i}.png", _PNG_1X1, "image/png")) for i in range(3)
        ]
        resp = client.post(
            "/api/v1/listing/generate",
            files=files,
            data={"category": "storage organizer"},
        )
        assert resp.status_code == 400
    finally:
        settings.vision_max_images = original


def test_generate_accepts_natural_language_extra_info() -> None:
    """Non-JSON extra_info is now treated as natural language, not rejected."""
    resp = client.post(
        "/api/v1/listing/generate",
        files=[("images", ("product.png", _PNG_1X1, "image/png"))],
        data={"category": "storage organizer", "extra_info": "not-json"},
    )
    assert resp.status_code == 200


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


# --------------- Batch generation (multipart) ---------------

def _png_part(name: str = "a.png"):
    """One multipart file tuple carrying the minimal valid PNG."""
    return ("images", (name, _PNG_1X1, "image/png"))


def test_batch_generate_multiple_products() -> None:
    meta = [
        {
            "product_index": 0,
            "category": "storage organizer",
            "platform": "amazon",
            "target_lang": "en",
            "image_count": 1,
        },
        {
            "product_index": 1,
            "category": "kitchen tools",
            "platform": "shopee",
            "target_lang": "en",
            "image_count": 1,
        },
    ]
    resp = client.post(
        "/api/v1/listing/batch_generate",
        data={"products": json.dumps(meta)},
        files=[_png_part("p0.png"), _png_part("p1.png")],
    )
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["results"]) == 2
    for r in body["results"]:
        assert r["error"] is None
        assert r["listing"]["title"]
        assert r["listing"]["compliance"]["passed"] is True
        # Per-product wall-clock time and token accounting are present.
        assert r["elapsed_ms"] >= 0
        assert r["listing"]["metadata"]["total_tokens"] >= 0


def test_batch_generate_requires_products() -> None:
    resp = client.post(
        "/api/v1/listing/batch_generate",
        data={"products": json.dumps([])},
    )
    assert resp.status_code == 400


def test_batch_generate_reports_invalid_platform() -> None:
    meta = [
        {
            "product_index": 0,
            "category": "storage organizer",
            "platform": "ebay",
            "target_lang": "en",
            "image_count": 1,
        }
    ]
    resp = client.post(
        "/api/v1/listing/batch_generate",
        data={"products": json.dumps(meta)},
        files=[_png_part()],
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["results"][0]["listing"] is None
    assert "platform" in body["results"][0]["error"].lower()


def test_batch_generate_reports_missing_images() -> None:
    meta = [
        {
            "product_index": 0,
            "category": "storage organizer",
            "platform": "amazon",
            "target_lang": "en",
            "image_count": 0,
        }
    ]
    resp = client.post(
        "/api/v1/listing/batch_generate",
        data={"products": json.dumps(meta)},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["results"][0]["listing"] is None
    assert body["results"][0]["error"]


def test_batch_generate_rejects_non_image_upload() -> None:
    # Plain text bytes are not a recognised image signature.
    meta = [
        {
            "product_index": 0,
            "category": "storage organizer",
            "platform": "amazon",
            "target_lang": "en",
            "image_count": 1,
        }
    ]
    resp = client.post(
        "/api/v1/listing/batch_generate",
        data={"products": json.dumps(meta)},
        files=[("images", ("bad.txt", b"this is definitely not an image", "text/plain"))],
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["results"][0]["listing"] is None
    assert body["results"][0]["error"]


def test_batch_generate_rejects_image_count_mismatch() -> None:
    # Metadata declares two images but only one file part is uploaded.
    meta = [
        {
            "product_index": 0,
            "category": "storage organizer",
            "platform": "amazon",
            "target_lang": "en",
            "image_count": 2,
        }
    ]
    resp = client.post(
        "/api/v1/listing/batch_generate",
        data={"products": json.dumps(meta)},
        files=[_png_part()],
    )
    assert resp.status_code == 400
    assert "mismatch" in resp.json()["detail"].lower()


def test_batch_generate_rejects_invalid_products_json() -> None:
    resp = client.post(
        "/api/v1/listing/batch_generate",
        data={"products": "not-json"},
        files=[_png_part()],
    )
    assert resp.status_code == 400


def test_generate_rejects_non_image_upload() -> None:
    resp = client.post(
        "/api/v1/listing/generate",
        files=[("images", ("evil.png", b"not-an-image-payload", "image/png"))],
        data={"category": "storage organizer"},
    )
    assert resp.status_code == 400


# --------------- Batch import ---------------

def test_import_template_download() -> None:
    resp = client.get("/api/v1/import/template")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/csv")
    text = resp.content.decode("utf-8")
    # BOM for Excel compatibility + the required header columns.
    assert text.startswith("\ufeff")
    for col in ("商品类目", "目标平台", "目标语言"):
        assert col in text


def test_import_parse_csv_valid() -> None:
    csv_text = "商品类目,目标平台,目标语言,补充信息\n收纳盒,amazon,en,brand:ACME\n水杯,shopee,zh,\n"
    resp = client.post(
        "/api/v1/import/parse",
        files=[("file", ("products.csv", csv_text.encode("utf-8"), "text/csv"))],
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["total_rows"] == 2
    assert body["valid_count"] == 2
    assert body["error_count"] == 0
    assert body["products"][0]["category"] == "收纳盒"
    assert body["products"][1]["platform"] == "shopee"


def test_import_parse_csv_reports_row_errors() -> None:
    csv_text = "商品类目,目标平台,目标语言,补充信息\n,amazon,en,\n水杯,ebay,xx,\n"
    resp = client.post(
        "/api/v1/import/parse",
        files=[("file", ("products.csv", csv_text.encode("utf-8"), "text/csv"))],
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["total_rows"] == 2
    assert body["valid_count"] == 0
    assert body["error_count"] == 2
    # Row 2: empty category; Row 3: bad platform + bad language.
    assert body["products"][0]["is_valid"] is False
    assert body["products"][0]["errors"]
    assert body["products"][1]["is_valid"] is False
    assert len(body["products"][1]["errors"]) >= 2


def test_import_parse_unsupported_format() -> None:
    resp = client.post(
        "/api/v1/import/parse",
        files=[("file", ("data.bin", b"\x00\x01\x02", "application/octet-stream"))],
    )
    assert resp.status_code == 400


def test_import_parse_text_file_treated_as_csv_reports_missing_columns() -> None:
    # text/* uploads fall back to CSV parsing and report missing columns
    # rather than being hard-rejected, so users get actionable feedback.
    resp = client.post(
        "/api/v1/import/parse",
        files=[("file", ("data.txt", b"plain text", "text/plain"))],
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["total_rows"] == 0
    assert body["file_errors"]
