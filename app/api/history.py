"""Read-only API over the generation history cold storage."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse

from app.config import get_settings
from app.history import store as history_store

router = APIRouter(prefix="/api/v1/history", tags=["history"])


@router.get("")
async def list_history(limit: int = Query(default=100, ge=1, le=500)) -> dict:
    """Return recent generation records (summaries), newest first."""
    settings = get_settings()
    if not settings.history_enabled:
        return {"enabled": False, "records": []}
    records = history_store.list_records(settings.history_dir, limit=limit)
    return {"enabled": True, "records": records}


@router.get("/{record_id}")
async def get_history_record(record_id: str) -> dict:
    """Return one full record including per-product listings and image names."""
    settings = get_settings()
    record = history_store.get_record(settings.history_dir, record_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Record not found.")
    return record


@router.get("/{record_id}/images/{name}")
async def get_history_image(record_id: str, name: str):
    """Serve a stored (compressed) product image."""
    settings = get_settings()
    path = history_store.get_image_path(settings.history_dir, record_id, name)
    if path is None:
        raise HTTPException(status_code=404, detail="Image not found.")
    return FileResponse(path, media_type="image/jpeg")
