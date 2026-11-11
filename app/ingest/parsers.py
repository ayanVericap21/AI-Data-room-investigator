"""File -> list of pages of text. Page numbers are 1-based; text/markdown/csv are a single page."""
from __future__ import annotations

import csv
from pathlib import Path

from pydantic import BaseModel


class PageText(BaseModel):
    page: int
    text: str


class ParseError(Exception):
    """Deterministic failure (bad file, no text). Not retried."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def _parse_pdf(path: Path) -> list[PageText]:
    from pypdf import PdfReader

    try:
        reader = PdfReader(str(path))
        pages = [PageText(page=i + 1, text=(p.extract_text() or "")) for i, p in enumerate(reader.pages)]
    except Exception as e:  # pypdf raises many different types for corrupt files
        raise ParseError("PARSE_ERROR", f"could not read PDF: {e}") from e
    return pages


def _parse_csv(path: Path) -> list[PageText]:
    # Render each row as "header=value; ..." in its own paragraph so column meaning stays attached to values.
    with path.open(newline="", encoding="utf-8", errors="replace") as f:
        rows = list(csv.reader(f))
    if not rows:
        return []
    header, body = rows[0], rows[1:]
    paras = ["; ".join(f"{h}={v}" for h, v in zip(header, r)) for r in body if any(c.strip() for c in r)]
    return [PageText(page=1, text="\n\n".join(paras))]


def parse_file(path: Path) -> list[PageText]:
    ext = path.suffix.lower()
    if ext == ".pdf":
        pages = _parse_pdf(path)
    elif ext in {".txt", ".md"}:
        pages = [PageText(page=1, text=path.read_text(encoding="utf-8", errors="replace"))]
    elif ext == ".csv":
        pages = _parse_csv(path)
    else:
        raise ParseError("UNSUPPORTED_TYPE", f"unsupported file type {ext}")
    if not any(p.text.strip() for p in pages):
        # e.g. a scanned PDF. OCR is not implemented; say so rather than silently producing nothing.
        raise ParseError("EMPTY_TEXT", "no extractable text (scanned document? OCR is not implemented)")
    return pages
