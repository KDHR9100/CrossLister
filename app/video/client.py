"""Video client turning one product image into a short MP4 clip.

Talks to the DashScope-style *async task* API exposed by the same MaaS
gateway as the vision/LLM models (``Settings.video_api_base``):

  1. ``POST /api/v1/services/aigc/video-generation/video-synthesis``
     with ``X-DashScope-Async: enable`` → returns ``output.task_id``.
     The input image travels as a base64 data URI inside
     ``input.media[0].url`` (verified against the gateway: data URIs are
     accepted, so no public image URL is ever needed).
  2. ``GET /api/v1/tasks/{task_id}`` polled until ``task_status`` is
     ``SUCCEEDED``/``FAILED`` → ``output.video_url`` (remote, expires in
     24h).
  3. The MP4 is downloaded immediately into ``Settings.video_output_dir``
     and served locally, because the remote URL is short-lived.

``mock`` mode writes a placeholder file so the whole flow (API wiring,
frontend, history) can be exercised offline in dev and tests.
"""

from __future__ import annotations

import asyncio
import base64
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from app.config import Settings, VideoMode, get_settings
from app.utils.images import preprocess_image
from app.utils.logger import get_logger
from app.vision.client import guess_mime

logger = get_logger(__name__)

# Cap the image handed to the video model: generous enough for i2v quality,
# small enough that the base64 payload stays well under gateway limits.
_VIDEO_MAX_IMAGE_SIDE = 1280


def _httpx_module():
    """Return the httpx module vendored by the installed openai SDK.

    Reusing the SDK's own httpx build guarantees client compatibility with
    the rest of the app (same trick as app/utils/openai_client.py).
    """
    import openai._base_client as _bc

    return getattr(_bc, "httpx", None) or getattr(_bc, "httpx2")


def _is_retryable_status(status: int) -> bool:
    return status >= 500 or status in (408, 429)


@dataclass
class VideoResult:
    """Outcome of one video generation."""

    # Serving path for the browser, e.g. "/api/v1/video/abc123.mp4".
    url: str
    # Absolute path of the stored MP4 on disk.
    path: Path
    # Model that produced the clip ("mock" in mock mode).
    model: str


class VideoError(RuntimeError):
    """Raised when video generation fails or exceeds its time budget."""


class VideoClient:
    """High-level wrapper around the async video-synthesis API."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._client = None

    # -- HTTP ------------------------------------------------------------
    def _get_client(self):
        """Lazily build one cached async HTTP client (shared per process)."""
        if self._client is None:
            httpx = _httpx_module()
            self._client = httpx.AsyncClient(
                base_url=self._settings.video_api_base,
                headers={
                    "Authorization": f"Bearer {self._effective_api_key()}",
                },
                timeout=self._settings.video_timeout_s,
                trust_env=False,
            )
        return self._client

    def _effective_api_key(self) -> str:
        s = self._settings
        return s.video_api_key or s.vision_api_key or "EMPTY"

    async def _aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    # -- Public API -------------------------------------------------------
    async def generate(self, image: bytes, prompt: str = "") -> VideoResult:
        """Turn ``image`` (raw bytes) into a short clip and store it locally.

        Raises:
            VideoError: on API rejection, task failure, or timeout.
        """
        s = self._settings
        prompt = (prompt or "").strip()
        if s.video_mode == VideoMode.MOCK:
            return self._mock_generate(prompt)
        return await self._remote_generate(image, prompt)

    # -- Mock mode ---------------------------------------------------------
    def _mock_generate(self, prompt: str) -> VideoResult:
        s = self._settings
        out_dir = Path(s.video_output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        name = f"{uuid.uuid4().hex}.mp4"
        path = out_dir / name
        path.write_bytes(b"MOCK_VIDEO_PLACEHOLDER:" + prompt.encode("utf-8")[:200])
        _prune_old_files(out_dir, s.video_max_output_files)
        logger.info("video.mock_generated", file=name)
        return VideoResult(url=f"/api/v1/video/{name}", path=path, model="mock")

    # -- API mode ----------------------------------------------------------
    async def _remote_generate(self, image: bytes, prompt: str) -> VideoResult:
        s = self._settings
        client = self._get_client()

        # Keep the upload small: downscale + JPEG-recompress like the vision
        # client does (CPU-bound, off the event loop).
        image = await asyncio.to_thread(
            preprocess_image, image, _VIDEO_MAX_IMAGE_SIDE, 90
        )
        data_uri = (
            "data:" + guess_mime(image) + ";base64,"
            + base64.b64encode(image).decode("ascii")
        )

        body = {
            "model": s.video_model,
            "input": {
                "media": [{"type": "first_frame", "url": data_uri}],
            },
            "parameters": {
                "resolution": s.video_resolution,
                "duration": s.video_duration_s,
            },
        }
        if prompt:
            body["input"]["prompt"] = prompt

        started = time.perf_counter()
        logger.info("video.request", model=s.video_model, duration=s.video_duration_s)

        task_id = await self._create_task(client, body)
        video_url = await self._poll_task(
            client, task_id, deadline=started + s.video_timeout_s
        )
        result = await self._download(
            client, video_url, deadline=started + s.video_timeout_s
        )

        logger.info(
            "video.completed",
            task_id=task_id,
            latency_ms=int((time.perf_counter() - started) * 1000),
        )
        return result

    async def _create_task(self, client, body: dict) -> str:
        """Submit the synthesis task, retrying transient errors."""
        s = self._settings
        max_retries = 2
        last_error = ""
        for attempt in range(max_retries + 1):
            try:
                resp = await client.post(
                    "/api/v1/services/aigc/video-generation/video-synthesis",
                    json=body,
                    headers={"X-DashScope-Async": "enable"},
                )
                if resp.status_code == 200:
                    data = resp.json()
                    task_id = (data.get("output") or {}).get("task_id", "")
                    if task_id:
                        return task_id
                    raise VideoError(f"video create returned no task_id: {data}")
                if _is_retryable_status(resp.status_code) and attempt < max_retries:
                    last_error = f"HTTP {resp.status_code}: {resp.text[:300]}"
                    logger.warning(
                        "video.create_retry", attempt=attempt + 1, error=last_error
                    )
                    await asyncio.sleep(2 ** (attempt + 1))
                    continue
                raise VideoError(
                    f"video create failed: HTTP {resp.status_code}: {resp.text[:300]}"
                )
            except VideoError:
                raise
            except Exception as exc:  # noqa: BLE001 - transport errors
                if attempt < max_retries:
                    last_error = str(exc)
                    logger.warning(
                        "video.create_retry", attempt=attempt + 1, error=last_error
                    )
                    await asyncio.sleep(2 ** (attempt + 1))
                    continue
                raise VideoError(f"video create request failed: {exc}") from exc
        raise VideoError(f"video create failed after retries: {last_error}")

    async def _poll_task(self, client, task_id: str, *, deadline: float) -> str:
        """Poll the task until completion; return the remote video URL."""
        s = self._settings
        while time.perf_counter() < deadline:
            try:
                resp = await client.get(f"/api/v1/tasks/{task_id}")
            except Exception as exc:  # noqa: BLE001 - transient transport error
                logger.warning("video.poll_retry", error=str(exc))
                await asyncio.sleep(s.video_poll_interval_s)
                continue
            if resp.status_code != 200:
                # Brief server hiccups shouldn't kill a 5-minute job.
                logger.warning(
                    "video.poll_non_200", status=resp.status_code, body=resp.text[:200]
                )
                await asyncio.sleep(s.video_poll_interval_s)
                continue
            output = (resp.json().get("output") or {})
            status = str(output.get("task_status", "")).upper()
            if status == "SUCCEEDED":
                # Wan-family tasks put the link at output.video_url; some
                # variants nest it in video_metrics.video_url.
                metrics = output.get("video_metrics") or {}
                url = output.get("video_url") or (
                    metrics.get("video_url") if isinstance(metrics, dict) else None
                )
                if not url:
                    raise VideoError(f"task succeeded but no video_url: {output}")
                return str(url)
            if status in {"FAILED", "CANCELED", "UNKNOWN"}:
                raise VideoError(
                    f"video task {status}: {output.get('message', '') or output}"
                )
            await asyncio.sleep(s.video_poll_interval_s)
        raise VideoError(
            f"video generation timed out after {int(self._settings.video_timeout_s)}s"
        )

    async def _download(self, client, remote_url: str, *, deadline: float) -> VideoResult:
        """Download the finished MP4 into the local output dir."""
        s = self._settings
        out_dir = Path(s.video_output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        name = f"{uuid.uuid4().hex}.mp4"
        path = out_dir / name

        remaining = deadline - time.perf_counter()
        if remaining <= 5:
            raise VideoError("video generation timed out before download")
        try:
            async with asyncio.timeout(remaining):
                async with client.stream("GET", remote_url) as resp:
                    if resp.status_code != 200:
                        body = (await resp.aread())[:300]
                        raise VideoError(
                            f"video download failed: HTTP {resp.status_code}: {body!r}"
                        )
                    with path.open("wb") as fh:
                        async for chunk in resp.aiter_bytes(64 * 1024):
                            fh.write(chunk)
        except VideoError:
            path.unlink(missing_ok=True)
            raise
        except TimeoutError as exc:
            path.unlink(missing_ok=True)
            raise VideoError("video download timed out") from exc
        except Exception as exc:  # noqa: BLE001 - transport errors
            path.unlink(missing_ok=True)
            raise VideoError(f"video download failed: {exc}") from exc

        _prune_old_files(out_dir, s.video_max_output_files)
        return VideoResult(
            url=f"/api/v1/video/{name}", path=path, model=s.video_model
        )


def _prune_old_files(out_dir: Path, max_files: int) -> None:
    """Delete oldest MP4s once the directory exceeds ``max_files``."""
    try:
        files = sorted(
            (p for p in out_dir.glob("*.mp4") if p.is_file()),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for stale in files[max_files:]:
            stale.unlink(missing_ok=True)
    except Exception as exc:  # noqa: BLE001 - pruning must never break a result
        logger.warning("video.prune_failed", error=str(exc))
