"""Video generation tests: client mock mode, API wiring, and file serving.

Everything runs offline — ``video_mode=mock`` makes the VideoClient write a
placeholder MP4 instead of calling the remote gateway, which is enough to
cover the request parsing, the SSE event flow, and the serving route.
"""

from __future__ import annotations

import asyncio
import json
import re

import pytest
from fastapi.testclient import TestClient

from app.config import VideoMode, get_settings
from app.main import app
from app.video.client import VideoClient, VideoResult

from test_api import _PNG_1X1, _png_part, ensure_index  # noqa: F401

client = TestClient(app)

_VIDEO_URL_RE = re.compile(r"^/api/v1/video/[0-9a-f]{32}\.mp4$")


@pytest.fixture(autouse=True)
def mock_video_mode(tmp_path):
    """Force mock video mode with a throwaway output directory."""
    settings = get_settings()
    old = (settings.video_mode, settings.video_output_dir)
    settings.video_mode = VideoMode.MOCK
    settings.video_output_dir = tmp_path / "videos"
    yield settings
    settings.video_mode, settings.video_output_dir = old


def test_video_client_mock_writes_file(tmp_path) -> None:
    settings = get_settings()
    result = asyncio.run(VideoClient(settings).generate(_PNG_1X1, "rotate"))
    assert isinstance(result, VideoResult)
    assert _VIDEO_URL_RE.match(result.url)
    assert result.path.is_file()
    assert result.path.parent == settings.video_output_dir
    assert b"MOCK_VIDEO_PLACEHOLDER" in result.path.read_bytes()


def test_video_client_mock_prunes_old_files() -> None:
    settings = get_settings()
    settings.video_output_dir.mkdir(parents=True, exist_ok=True)
    # max_output_files=0 should prune everything on the next generation;
    # the important part is that nothing raises and no stale files stay.
    settings.video_max_output_files = 0
    try:
        asyncio.run(VideoClient(settings).generate(_PNG_1X1, ""))
        remaining = list(settings.video_output_dir.glob("*.mp4"))
        assert len(remaining) <= 1
    finally:
        settings.video_max_output_files = 400


def test_single_generate_with_video() -> None:
    resp = client.post(
        "/api/v1/listing/generate",
        files=[("images", ("product.png", _PNG_1X1, "image/png"))],
        data={
            "category": "storage organizer",
            "platform": "amazon",
            "generate_video": "true",
            "video_prompt": "the product slowly rotates",
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["title"]
    assert _VIDEO_URL_RE.match(body["video_url"])
    assert body["video_error"] is None

    # The referenced file is servable and rejects traversal elsewhere.
    video_path = body["video_url"]
    got = client.get(video_path)
    assert got.status_code == 200
    assert got.headers["content-type"].startswith("video/mp4")
    assert client.get("/api/v1/video/..%2f..%2fetc%2fpasswd").status_code in (
        400, 404,
    )


def test_single_generate_without_video() -> None:
    resp = client.post(
        "/api/v1/listing/generate",
        files=[("images", ("product.png", _PNG_1X1, "image/png"))],
        data={"category": "storage organizer"},
    )
    assert resp.status_code == 200
    assert resp.json()["video_url"] is None


def test_batch_generate_with_video() -> None:
    meta = [
        {
            "product_index": 0,
            "category": "storage organizer",
            "platform": "amazon",
            "image_count": 1,
            "generate_video": True,
            "video_prompt": "rotate",
        },
        {
            "product_index": 1,
            "category": "kitchen tools",
            "platform": "shopee",
            "image_count": 1,
        },
    ]
    resp = client.post(
        "/api/v1/listing/batch_generate",
        data={"products": json.dumps(meta)},
        files=[_png_part("p0.png"), _png_part("p1.png")],
    )
    assert resp.status_code == 200
    results = resp.json()["results"]
    assert results[0]["listing"]["video_url"]
    assert results[0]["listing"]["video_error"] is None
    # Video was never requested for the second product.
    assert results[1]["listing"]["video_url"] is None


def test_batch_stream_emits_video_events() -> None:
    meta = [
        {
            "product_index": 0,
            "category": "storage organizer",
            "platform": "amazon",
            "image_count": 1,
            "generate_video": True,
            "video_prompt": "rotate",
        }
    ]
    with client.stream(
        "POST",
        "/api/v1/listing/batch_generate_stream",
        data={"products": json.dumps(meta)},
        files=[_png_part("p0.png")],
    ) as resp:
        assert resp.status_code == 200
        events = []
        for line in resp.iter_lines():
            if line.startswith("data: "):
                events.append(json.loads(line[len("data: "):]))

    types = [e["type"] for e in events]
    assert types[0] == "video_start"
    assert "product_start" in types
    assert "product_done" in types
    assert "video_done" in types
    assert types[-1] == "done"

    video_done = next(e for e in events if e["type"] == "video_done")
    assert video_done["product_index"] == 0
    assert _VIDEO_URL_RE.match(video_done["video_url"])
    product_done = next(e for e in events if e["type"] == "product_done")
    assert product_done["listing"]["title"]


def test_video_route_rejects_bad_names() -> None:
    for bad in ["evil.mp4", "abc.mp4", "sub/dir/x.mp4", "x.mp4.exe"]:
        assert client.get(f"/api/v1/video/{bad}").status_code in (400, 404)


def test_batch_caps_reject_oversized_requests() -> None:
    settings = get_settings()
    old_products = settings.batch_max_products
    settings.batch_max_products = 2
    try:
        meta = [
            {
                "product_index": i,
                "category": "c",
                "platform": "amazon",
                "image_count": 1,
            }
            for i in range(3)
        ]
        resp = client.post(
            "/api/v1/listing/batch_generate",
            data={"products": json.dumps(meta)},
            files=[_png_part(f"p{i}.png") for i in range(3)],
        )
        assert resp.status_code == 400
        assert "products" in resp.json()["detail"]
    finally:
        settings.batch_max_products = old_products
