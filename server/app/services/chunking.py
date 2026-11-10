"""Turns extracted units into storable chunks. Deterministic.

Rules:
- One extracted unit (paragraph, heading, table, spreadsheet row) is one chunk, so the
  locator stays exact.
- A unit longer than ``max_chars`` is split on sentence boundaries (text) or row
  boundaries (tables). Every piece keeps the unit's locator; ``metadata.part`` /
  ``metadata.parts`` record the split.
- The embedding text is the chunk text prefixed with its filename and heading, so short
  chunks still carry their context. The stored ``content`` is never altered.
"""
from __future__ import annotations

import re

from pydantic import BaseModel, Field

from app.schemas.extraction import ExtractedUnit

DEFAULT_MAX_CHARS = 1500
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


class ChunkDraft(BaseModel):
    chunk_index: int
    content: str
    content_type: str
    page_number: int | None = None
    sheet_name: str | None = None
    cell_reference: str | None = None
    heading: str | None = None
    metadata: dict = Field(default_factory=dict)
    filename: str = ""


def _pack(pieces: list[str], max_chars: int, joiner: str) -> list[str]:
    """Greedily pack pieces into groups no longer than max_chars (a single oversized
    piece is hard-split so nothing exceeds the limit)."""
    groups: list[str] = []
    current = ""
    for piece in pieces:
        while len(piece) > max_chars:
            if current:
                groups.append(current)
                current = ""
            groups.append(piece[:max_chars])
            piece = piece[max_chars:]
        candidate = piece if not current else current + joiner + piece
        if len(candidate) > max_chars and current:
            groups.append(current)
            current = piece
        else:
            current = candidate
    if current:
        groups.append(current)
    return groups


def split_text(text: str, content_type: str, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    if content_type in ("table", "table_row"):
        return _pack(text.split("\n"), max_chars, "\n")
    return _pack([s for s in _SENTENCE_END.split(text) if s], max_chars, " ")


def chunk_units(units: list[ExtractedUnit], max_chars: int = DEFAULT_MAX_CHARS) -> list[ChunkDraft]:
    drafts: list[ChunkDraft] = []
    for unit in units:
        parts = split_text(unit.content, unit.content_type, max_chars)
        for i, part in enumerate(parts, start=1):
            metadata = dict(unit.metadata)
            metadata["locator"] = unit.locator.model_dump(exclude_none=True)
            if len(parts) > 1:
                metadata.update(part=i, parts=len(parts))
            drafts.append(ChunkDraft(
                chunk_index=len(drafts),
                content=part,
                content_type=unit.content_type,
                page_number=unit.locator.page,
                sheet_name=unit.locator.sheet_name,
                cell_reference=unit.locator.cell_reference,
                heading=unit.locator.heading,
                metadata=metadata,
                filename=unit.filename,
            ))
    return drafts


def embedding_text(filename: str, heading: str | None, content: str) -> str:
    prefix = f"Document: {filename}"
    if heading:
        prefix += f" | Section: {heading}"
    return f"{prefix}\n{content}"
