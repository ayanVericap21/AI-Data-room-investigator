"""API schemas for documents. No absolute server paths are ever exposed."""
from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict


class DocumentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    filename: str  # relative to the data room root
    file_type: str
    file_size: int
    processing_status: str
    last_modified_at: datetime
    # Pages (PDF) / sheets (XLSX); null for DOCX or unreadable files.
    unit_count: int | None = None


class DocumentListOut(BaseModel):
    items: list[DocumentOut]
    total: int


class ScanResultOut(BaseModel):
    discovered: int
    added: int
    updated: int
    unchanged: int
    missing: int
    skipped_duplicates: list[str]
    unsupported: list[str]
    documents: list[DocumentOut]
