"""Platform rule document loading and chunking.

Documents live in ``data/platform_rules`` and are named with a platform
prefix (``amazon_*``, ``shopee_*``, ``temu_*``). Supported formats are
Markdown, plain text and PDF.

Chunking strategy (per spec): each rule entry becomes exactly one chunk.
A rule entry is a level-2 Markdown section whose heading starts with a
rule id, e.g. ``## AMZ-TITLE-01: Maximum title length``. Content before
the first rule heading (front-matter / intro) is discarded.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from app.config import Settings, get_settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

# Map a filename prefix to the canonical platform key used for collections.
_PREFIX_TO_PLATFORM: dict[str, str] = {
    "amazon": "amazon",
    "shopee": "shopee",
    "temu": "temu",
}

_SUPPORTED_SUFFIXES = {".md", ".markdown", ".txt", ".pdf"}

# Matches a rule heading such as "## AMZ-TITLE-01: Maximum title length".
# Captures the rule id and the human-readable title that follows the colon.
_RULE_HEADING = re.compile(r"^##\s+(?P<rule_id>[A-Z0-9][A-Z0-9\-]*):\s*(?P<title>.+?)\s*$")


@dataclass
class RuleChunk:
    """A single rule entry ready for embedding and indexing."""

    rule_id: str
    title: str
    text: str
    platform: str
    source: str
    metadata: dict[str, str] = field(default_factory=dict)

    @property
    def full_text(self) -> str:
        """Text actually embedded and searched: title plus body."""
        return f"{self.title}\n{self.text}".strip()

    @property
    def chunk_id(self) -> str:
        """Stable, unique id for the vector store."""
        return f"{self.platform}:{self.rule_id}"


def detect_platform(path: Path) -> str | None:
    """Infer the platform key from a filename prefix, or None if unknown."""
    name = path.name.lower()
    for prefix, platform in _PREFIX_TO_PLATFORM.items():
        if name.startswith(prefix):
            return platform
    return None


def _extract_text(path: Path) -> str:
    """Read raw text from a supported document."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _read_pdf(path)
    return path.read_text(encoding="utf-8", errors="replace")


def _read_pdf(path: Path) -> str:
    """Extract text from a PDF without pulling in a heavy mandatory dep.

    Uses pypdf when available; otherwise falls back to an empty string with a
    warning so that a missing optional dependency never breaks indexing.
    """
    try:
        from pypdf import PdfReader  # type: ignore
    except ImportError:  # pragma: no cover - depends on optional dep
        logger.warning("rag.pdf_skipped_missing_pypdf", file=path.name)
        return ""
    reader = PdfReader(str(path))
    return "\n".join((page.extract_text() or "") for page in reader.pages)


def chunk_document(path: Path, platform: str) -> list[RuleChunk]:
    """Split one rule document into per-rule chunks.

    Args:
        path: Path to the source document.
        platform: Canonical platform key for this document.

    Returns:
        A list of RuleChunk objects, one per rule heading.
    """
    raw = _extract_text(path)
    if not raw.strip():
        logger.warning("rag.document_empty", file=path.name)
        return []

    chunks: list[RuleChunk] = []
    lines = raw.splitlines()

    current_id: str | None = None
    current_title: str | None = None
    buffer: list[str] = []

    def flush() -> None:
        if current_id is not None and current_title is not None:
            body = "\n".join(buffer).strip()
            if body:
                chunks.append(
                    RuleChunk(
                        rule_id=current_id,
                        title=current_title,
                        text=body,
                        platform=platform,
                        source=path.name,
                        metadata={"source": path.name, "rule_id": current_id},
                    )
                )

    for line in lines:
        match = _RULE_HEADING.match(line.strip())
        if match:
            flush()
            current_id = match.group("rule_id")
            current_title = match.group("title")
            buffer = []
        elif current_id is not None:
            buffer.append(line)
    flush()

    logger.info("rag.document_chunked", file=path.name, platform=platform, chunks=len(chunks))
    return chunks


def load_all_rules(settings: Settings | None = None) -> list[RuleChunk]:
    """Load and chunk every rule document found in the rules directory.

    Files that do not match a known platform prefix or supported suffix are
    skipped with a warning rather than failing the whole load.

    Args:
        settings: Optional settings override (defaults to get_settings()).

    Returns:
        All rule chunks across all platform documents.
    """
    settings = settings or get_settings()
    rules_dir = Path(settings.platform_rules_dir)
    if not rules_dir.is_dir():
        logger.error("rag.rules_dir_missing", dir=str(rules_dir))
        return []

    chunks: list[RuleChunk] = []
    for path in sorted(rules_dir.iterdir()):
        if not path.is_file():
            continue
        if path.suffix.lower() not in _SUPPORTED_SUFFIXES:
            logger.debug("rag.file_skipped_unsupported", file=path.name)
            continue
        platform = detect_platform(path)
        if platform is None:
            logger.warning("rag.file_skipped_unknown_platform", file=path.name)
            continue
        chunks.extend(chunk_document(path, platform))

    logger.info("rag.load_complete", total_chunks=len(chunks))
    return chunks
