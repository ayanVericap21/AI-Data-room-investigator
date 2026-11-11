"""Common internal structure for extracted document content (all formats)."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

SourceType = Literal["pdf", "docx", "xlsx"]
ContentType = Literal["paragraph", "heading", "table", "table_row"]


class Locator(BaseModel):
    """Where in the source a unit came from. Only the relevant fields are set."""

    page: int | None = None  # PDF, 1-based
    paragraph_index: int | None = None  # DOCX body paragraph number, 1-based
    table_index: int | None = None  # DOCX/PDF table number (per document/page), 1-based
    heading: str | None = None  # nearest preceding heading, where known
    sheet_name: str | None = None  # XLSX
    cell_reference: str | None = None  # XLSX cell or range, e.g. "B2" or "A5:F5"


class ExtractedUnit(BaseModel):
    """One extracted passage with its locator, before chunking/persistence."""

    document_id: str
    filename: str
    source_type: SourceType
    locator: Locator
    content: str
    content_type: ContentType
    metadata: dict[str, Any] = Field(default_factory=dict)


class XlsxCell(BaseModel):
    """Cell-level detail kept in unit metadata so formulas are never flattened to text."""

    ref: str
    column_header: str | None = None
    value: Any = None  # cached value (None if the file has no cached result)
    formula: str | None = None  # e.g. "=B2*C2" when the cell holds a formula
    cached_value_available: bool = True


class ExtractionReport(BaseModel):
    """Side information from one extraction: milliseconds per part, and plain-language notes about
    anything that was not captured or was deliberately skipped."""

    timings_ms: dict[str, int] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)
