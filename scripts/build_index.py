"""One-click RAG index builder.

Reads every platform rule document under ``data/platform_rules``, chunks it
per rule entry and upserts the vectors into the per-platform Chroma
collections. Run from the repository root:

    uv run python scripts/build_index.py

The script is idempotent: collections are rebuilt from scratch on each run so
the index always mirrors the current documents on disk.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure the repository root is importable when run as a plain script.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from app.config import get_settings  # noqa: E402
from app.rag.indexer import build_index  # noqa: E402
from app.utils.logger import get_logger, setup_logging  # noqa: E402


def main() -> int:
    """Build the index and print a short summary."""
    settings = get_settings()
    setup_logging(settings.debug)
    logger = get_logger("scripts.build_index")

    logger.info(
        "build_index.start",
        rules_dir=str(settings.platform_rules_dir),
        chroma_dir=str(settings.chroma_persist_dir),
        embedding_mode=settings.embedding_mode.value,
        embedding_model=settings.embedding_model,
    )

    stats = build_index(settings, rebuild=True)

    logger.info("build_index.done", total_chunks=stats.total_chunks, platforms=stats.platforms)
    print(f"Indexed {stats.total_chunks} rule chunks across {len(stats.platforms)} platform(s):")
    for platform, count in sorted(stats.platforms.items()):
        print(f"  - {platform}: {count} rules")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
