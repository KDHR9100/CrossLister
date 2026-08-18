"""Phase 2 unit tests for the vision module (mock mode focused)."""

import asyncio

from app.config import Settings, VisionMode
from app.models.listing import VisualAnalysis
from app.vision.client import VisionClient, encode_image, guess_mime
from app.vision.prompts import build_vision_messages, parse_vision_json


def _mock_client() -> VisionClient:
    return VisionClient(Settings(vision_mode=VisionMode.MOCK))


# -- VisionClient (mock mode) --------------------------------------------


def test_mock_analyze_returns_full_structure() -> None:
    client = _mock_client()
    result = asyncio.run(client.analyze([b"fake-image-bytes"]))

    assert isinstance(result, VisualAnalysis)
    assert result.detected_category
    assert isinstance(result.colors, list) and result.colors
    assert isinstance(result.materials, list) and result.materials
    assert isinstance(result.selling_points, list) and result.selling_points
    assert isinstance(result.scenes, list) and result.scenes
    assert result.raw_description


def test_mock_analyze_uses_category_hint() -> None:
    client = _mock_client()
    result = asyncio.run(client.analyze([b"x"], category_hint="家居收纳"))
    assert result.detected_category == "家居收纳"


def test_mock_analyze_defaults_category_when_no_hint() -> None:
    client = _mock_client()
    result = asyncio.run(client.analyze([b"x"]))
    assert result.detected_category == "storage organizer"


def test_analyze_truncates_images_to_limit() -> None:
    settings = Settings(vision_mode=VisionMode.MOCK, vision_max_images=2)
    client = VisionClient(settings)
    # Should not raise even when more than the limit is provided.
    result = asyncio.run(client.analyze([b"a", b"b", b"c", b"d"]))
    assert isinstance(result, VisualAnalysis)


# -- Helpers -------------------------------------------------------------


def test_encode_image_is_base64() -> None:
    assert encode_image(b"abc") == "YWJj"


def test_guess_mime_detects_png_and_defaults_to_jpeg() -> None:
    png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 8
    assert guess_mime(png) == "image/png"
    assert guess_mime(b"\xff\xd8\xff\xe0") == "image/jpeg"
    assert guess_mime(b"random-bytes") == "image/jpeg"


# -- Prompts -------------------------------------------------------------


def test_build_vision_messages_shape() -> None:
    messages = build_vision_messages(
        [encode_image(b"img")],
        category_hint="家居收纳",
        extra_info={"brand": "HomeBox"},
        mime_types=["image/png"],
    )
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"

    content = messages[1]["content"]
    assert content[0]["type"] == "text"
    assert "家居收纳" in content[0]["text"]
    assert "HomeBox" in content[0]["text"]

    image_part = content[1]
    assert image_part["type"] == "image_url"
    assert image_part["image_url"]["url"].startswith("data:image/png;base64,")


def test_parse_vision_json_strips_fences_and_unknown_keys() -> None:
    raw = (
        'Here is the result:\n'
        '```json\n'
        '{"detected_category": "storage organizer", "colors": ["white"], '
        '"unknown_key": 123}\n'
        '```\n'
    )
    parsed = parse_vision_json(raw)
    assert parsed["detected_category"] == "storage organizer"
    assert parsed["colors"] == ["white"]
    assert "unknown_key" not in parsed


def test_parse_vision_json_returns_empty_on_garbage() -> None:
    assert parse_vision_json("no json here") == {}
    assert parse_vision_json("") == {}
