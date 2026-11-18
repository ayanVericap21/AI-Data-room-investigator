"""Turns extracted units into storable chunks. Deterministic.

Rules:
- Without merging, one extracted unit (paragraph, heading, table, spreadsheet row) is one chunk.
- A unit longer than ``max_chars`` is split on sentence boundaries (text) or row
  boundaries (tables). Every piece keeps the unit's locator; ``metadata.part`` /
  ``metadata.parts`` record the split.
- Merge (on by default, see MERGE_BY_DEFAULT): small neighbouring units of the
  same page / sheet / heading section are joined up to ``target_chars`` (1000). The chunk
  keeps the first unit's page, sheet and heading; a merged spreadsheet chunk gets the
  bounding cell range, and every row inside keeps its own ``[Sheet!A5:F5]`` tag. Tables
  are never merged.
- The embedding text is the chunk text prefixed with its filename and heading, so short
  chunks still carry their context. The stored ``content`` is never altered.
"""
from __future__ import annotations

import re

from openpyxl.utils import get_column_letter
from openpyxl.utils.cell import range_boundaries
from pydantic import BaseModel, Field

from app.schemas.extraction import ExtractedUnit

DEFAULT_MAX_CHARS = 1000  # hard limit for any chunk (the local embedding model reads about 1,000 characters)
DEFAULT_TARGET_CHARS = 1000  # small neighbouring units are merged up to this size
DEFAULT_OVERLAP_CHARS = 150  # when a unit is split, each part after the first starts with up to this many characters from the end of the part before it (whole sentences or rows)
MERGE_BY_DEFAULT = True  # small neighbouring units are merged; False means one unit is one chunk
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


def _overlap_tail(part: str, overlap_chars: int, joiner: str) -> str:
    """The end of ``part`` to repeat at the start of the next part: whole sentences (text) or whole rows (tables) that fit in ``overlap_chars``;
    when even the last one is longer, its last words."""
    pieces = part.split("\n") if joiner == "\n" else [s for s in _SENTENCE_END.split(part) if s]
    tail = ""
    for piece in reversed(pieces):
        candidate = piece if not tail else piece + joiner + tail
        if len(candidate) > overlap_chars:
            break
        tail = candidate
    if not tail and joiner != "\n":
        words = pieces[-1].split()
        for word in reversed(words):
            candidate = word if not tail else word + " " + tail
            if len(candidate) > overlap_chars:
                break
            tail = candidate
    return tail


def split_text(text: str, content_type: str, max_chars: int, overlap_chars: int = 0) -> list[str]:
    """Split a unit that is longer than ``max_chars``. With ``overlap_chars`` each part after the first starts with the end of the part before it
    (so a fact cut at the boundary is whole in one of them); no part is longer than ``max_chars``."""
    if len(text) <= max_chars:
        return [text]
    overlap_chars = max(0, min(overlap_chars, max_chars // 2))
    room = max_chars - overlap_chars
    if content_type in ("table", "table_row"):
        joiner, parts = "\n", _pack(text.split("\n"), room, "\n")
    else:
        joiner, parts = " ", _pack([s for s in _SENTENCE_END.split(text) if s], room, " ")
    if overlap_chars and len(parts) > 1:
        parts = [parts[0]] + [(_overlap_tail(prev, overlap_chars, joiner) + joiner + cur).lstrip(joiner) for prev, cur in zip(parts, parts[1:])]
    return parts


_MERGEABLE = {"paragraph", "heading", "table_row"}  # tables stay as their own chunk


def _section_key(unit: ExtractedUnit) -> tuple:
    """Units are only merged within the same page (PDF), sheet (XLSX) or heading section (DOCX)."""
    loc = unit.locator
    if loc.sheet_name:
        return ("sheet", loc.sheet_name)
    if loc.page is not None:
        return ("page", loc.page)
    return ("heading", loc.heading)


def _merged_cell_range(refs: list[str | None]) -> str | None:
    """Bounding range of several cell references, e.g. A5:F5 + A6:H6 -> A5:H6."""
    boxes = []
    for ref in refs:
        if ref:
            try:
                boxes.append(range_boundaries(ref))
            except ValueError:
                return refs[0]
    if not boxes:
        return None
    lo_c, lo_r = min(b[0] for b in boxes), min(b[1] for b in boxes)
    hi_c, hi_r = max(b[2] for b in boxes), max(b[3] for b in boxes)
    return f"{get_column_letter(lo_c)}{lo_r}:{get_column_letter(hi_c)}{hi_r}"


def _draft_from_group(group: list[tuple[ExtractedUnit, str, int, int]], index: int) -> ChunkDraft:
    first = group[0][0]
    if len(group) == 1:
        unit, text, part, parts = group[0]
        metadata = dict(unit.metadata)
        metadata["locator"] = unit.locator.model_dump(exclude_none=True)
        if parts > 1:
            metadata.update(part=part, parts=parts)
            if part > 1:
                metadata["overlaps_previous_part"] = True  # its first sentences (or rows) repeat the end of the part before
        return ChunkDraft(
            chunk_index=index, content=text, content_type=unit.content_type, page_number=unit.locator.page,
            sheet_name=unit.locator.sheet_name, cell_reference=unit.locator.cell_reference,
            heading=unit.locator.heading, metadata=metadata, filename=unit.filename,
        )
    units = [g[0] for g in group]
    types = {u.content_type for u in units}
    content_type = next(iter(types)) if len(types) == 1 else ("paragraph" if types <= {"heading", "paragraph"} else "mixed")
    metadata: dict = {
        "locator": first.locator.model_dump(exclude_none=True), "merged_units": len(units),
        "merged_from": [u.locator.model_dump(exclude_none=True) for u in units],
    }
    if any("cells" in u.metadata for u in units):  # spreadsheet rows keep their cell detail
        metadata["cells"] = [c for u in units for c in u.metadata.get("cells", [])]
        metadata["header_row"] = first.metadata.get("header_row")
        metadata["is_header_row"] = any(u.metadata.get("is_header_row") for u in units)
    return ChunkDraft(
        chunk_index=index, content="\n".join(g[1] for g in group), content_type=content_type,
        page_number=first.locator.page, sheet_name=first.locator.sheet_name,
        cell_reference=_merged_cell_range([u.locator.cell_reference for u in units]),
        heading=first.locator.heading, metadata=metadata, filename=first.filename,
    )


def chunk_units(units: list[ExtractedUnit], max_chars: int = DEFAULT_MAX_CHARS,
                target_chars: int = DEFAULT_TARGET_CHARS, merge: bool | None = None,
                overlap_chars: int = DEFAULT_OVERLAP_CHARS) -> list[ChunkDraft]:
    """Units -> chunks. Oversized units are split. With ``merge``, neighbouring small units of the
    same page, sheet or heading are joined (newline-separated, in reading order) until a chunk
    reaches ``target_chars``, so a slide of ten text boxes becomes one chunk, not ten fragments.
    Nothing is reworded: every unit's text appears in the chunk exactly as extracted.
    ``merge=None`` follows MERGE_BY_DEFAULT (on)."""
    if merge is None:
        merge = MERGE_BY_DEFAULT
    target_chars = min(target_chars, max_chars)
    groups: list[list[tuple[ExtractedUnit, str, int, int]]] = []
    group_chars = 0
    for unit in units:
        parts = split_text(unit.content, unit.content_type, max_chars, overlap_chars)
        for i, part in enumerate(parts, start=1):
            piece = (unit, part, i, len(parts))
            if (merge and groups and len(parts) == 1 and groups[-1][-1][3] == 1
                    and unit.content_type in _MERGEABLE and groups[-1][-1][0].content_type in _MERGEABLE
                    and _section_key(unit) == _section_key(groups[-1][-1][0])
                    and group_chars + 1 + len(part) <= target_chars):
                groups[-1].append(piece)
                group_chars += 1 + len(part)
            else:
                groups.append([piece])
                group_chars = len(part)
    return [_draft_from_group(g, i) for i, g in enumerate(groups)]


def embedding_text(filename: str, heading: str | None, content: str) -> str:
    prefix = f"Document: {filename}"
    if heading:
        prefix += f" | Section: {heading}"
    return f"{prefix}\n{content}"
