"""Cold-storage history store for completed generations.

Design constraints:
  - Non-intrusive: records are written only *after* generation completes and
    the response is built; callers use the async fire-and-forget wrapper.
  - Cold storage: plain files on disk (one directory per record + an index
    file). No database, no background process, no participation in the live
    generation pipeline.
  - Decoupled: this module receives only plain dicts and raw bytes; it never
    imports from ``app.agents`` or ``app.api``.
"""

from __future__ import annotations

import asyncio
import json
import re
import shutil
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.utils.images import preprocess_image
from app.utils.logger import get_logger

logger = get_logger(__name__)

# Record ids look like 20260825-153900-a1b2c3. Strict patterns block any
# path-traversal attempts through the lookup APIs.
_RECORD_ID_RE = re.compile(r"^\d{8}-\d{6}-[0-9a-f]{6}$")
_IMAGE_NAME_RE = re.compile(r"^p\d+_\d+\.jpg$")

_INDEX_FILE = "index.json"

# Serializes index.json read-modify-write cycles plus pruning. The sync store
# may run in threadpool workers, so a threading lock is the correct primitive.
_fs_lock = threading.Lock()


def _new_record_id() -> str:
    now = datetime.now()
    return f"{now.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"


def _build_summary(record: dict[str, Any]) -> dict[str, Any]:
    """Compact index entry used by the list view (avoids reading every record)."""
    products = record.get("products", [])
    success = sum(1 for p in products if p.get("status") == "success")
    titles = [
        (p.get("listing") or {}).get("title", "")
        for p in products
        if p.get("listing")
    ]
    return {
        "record_id": record["record_id"],
        "created_at": record["created_at"],
        "api": record.get("api", ""),
        "product_count": len(products),
        "success_count": success,
        "platforms": sorted({p.get("platform", "") for p in products}),
        "title_preview": titles[0][:60] if titles and titles[0] else "",
    }


def _update_index(history_dir: Path, summary: dict[str, Any], max_records: int) -> None:
    """Prepend the new summary, persist the index, and prune overflow records."""
    with _fs_lock:
        index_path = history_dir / _INDEX_FILE
        records: list[dict[str, Any]] = []
        if index_path.exists():
            try:
                records = json.loads(index_path.read_text(encoding="utf-8")).get(
                    "records", []
                )
            except Exception:  # noqa: BLE001 - corrupt index: start fresh
                logger.warning("history.index_corrupt", path=str(index_path))
                records = []
        records.insert(0, summary)
        pruned = records[max_records:] if len(records) > max_records else []
        records = records[:max_records]
        index_path.write_text(
            json.dumps({"records": records}, ensure_ascii=False), encoding="utf-8"
        )
    for s in pruned:
        shutil.rmtree(history_dir / s["record_id"], ignore_errors=True)
    if pruned:
        logger.info("history.pruned", count=len(pruned))


def save_record(
    history_dir: Path,
    api: str,
    products: list[dict[str, Any]],
    images: dict[int, list[bytes]] | None = None,
    *,
    save_images: bool = True,
    max_records: int = 200,
    max_image_side: int = 1280,
    jpeg_quality: int = 85,
) -> str:
    """Persist one generation record synchronously; returns the record id.

    Args:
        history_dir: Root directory of the history cold storage.
        api: Which endpoint produced the record (e.g. "batch_generate").
        products: Per-product dicts carrying category/platform/target_lang/
            status/listing/error/elapsed_ms etc. (plain JSON-serializable data).
        images: Optional {product_index: [image bytes]} used when save_images.
        save_images: Store compressed image copies alongside the text.
        max_records: Keep at most this many records (oldest pruned).
        max_image_side / jpeg_quality: Compression for stored image copies.
    """
    history_dir.mkdir(parents=True, exist_ok=True)
    record_id = _new_record_id()
    rdir = history_dir / record_id
    rdir.mkdir(parents=True)

    images = images or {}
    products_out: list[dict[str, Any]] = []
    for p in products:
        pi = int(p.get("product_index", 0))
        entry = dict(p)
        raw_list = images.get(pi, [])
        entry["image_count"] = len(raw_list)
        saved_names: list[str] = []
        if save_images and raw_list:
            img_dir = rdir / "images"
            img_dir.mkdir(exist_ok=True)
            for idx, raw in enumerate(raw_list):
                # Store the same compact JPEG the model saw, not the original
                # upload, keeping per-record disk usage bounded.
                name = f"p{pi}_{idx}.jpg"
                (img_dir / name).write_bytes(
                    preprocess_image(raw, max_image_side, jpeg_quality)
                )
                saved_names.append(name)
        entry["images"] = saved_names
        products_out.append(entry)

    record = {
        "record_id": record_id,
        "created_at": datetime.now(timezone.utc)
        .astimezone()
        .isoformat(timespec="seconds"),
        "api": api,
        "products": products_out,
    }
    (rdir / "record.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    _update_index(history_dir, _build_summary(record), max_records)
    logger.info(
        "history.saved",
        record_id=record_id,
        products=len(products_out),
        save_images=save_images,
    )
    return record_id


async def save_record_async(
    history_dir: Path,
    api: str,
    products: list[dict[str, Any]],
    images: dict[int, list[bytes]] | None = None,
    *,
    save_images: bool = True,
    max_records: int = 200,
    max_image_side: int = 1280,
    jpeg_quality: int = 85,
) -> str | None:
    """Fire-and-forget friendly wrapper: runs off the event loop and never
    raises, so history persistence can neither block nor break generation."""
    try:
        return await asyncio.to_thread(
            save_record,
            history_dir,
            api,
            products,
            images,
            save_images=save_images,
            max_records=max_records,
            max_image_side=max_image_side,
            jpeg_quality=jpeg_quality,
        )
    except Exception as exc:  # noqa: BLE001 - history must never break the app
        logger.error("history.save_failed", error=str(exc))
        return None


def list_records(history_dir: Path, limit: int = 100) -> list[dict[str, Any]]:
    """Return index summaries, newest first."""
    index_path = history_dir / _INDEX_FILE
    if not index_path.exists():
        return []
    try:
        records = json.loads(index_path.read_text(encoding="utf-8")).get("records", [])
    except Exception:  # noqa: BLE001 - corrupt index reads as empty
        logger.warning("history.index_corrupt", path=str(index_path))
        return []
    return records[: max(0, limit)]


def get_record(history_dir: Path, record_id: str) -> dict[str, Any] | None:
    """Return one full record, or None when the id is invalid/missing."""
    if not _RECORD_ID_RE.match(record_id):
        return None
    path = history_dir / record_id / "record.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        logger.warning("history.record_corrupt", record_id=record_id)
        return None


def get_image_path(history_dir: Path, record_id: str, name: str) -> Path | None:
    """Resolve a stored image file, or None when invalid/missing."""
    if not _RECORD_ID_RE.match(record_id) or not _IMAGE_NAME_RE.match(name):
        return None
    path = history_dir / record_id / "images" / name
    return path if path.exists() else None
