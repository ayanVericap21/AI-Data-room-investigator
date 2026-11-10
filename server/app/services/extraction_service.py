"""Dispatches extraction by file type into the common ExtractedUnit structure."""
from __future__ import annotations

from pathlib import Path

from app.core.exceptions import ExtractionError, UnsupportedDocumentError
from app.schemas.extraction import ExtractedUnit
from app.services.extractors.docx import extract_docx
from app.services.extractors.pdf import count_pages, extract_pdf
from app.services.extractors.xlsx import count_sheets, extract_xlsx

SUPPORTED_EXTENSIONS = {".pdf": "pdf", ".docx": "docx", ".xlsx": "xlsx"}


def file_type_for(path: Path) -> str | None:
    return SUPPORTED_EXTENSIONS.get(path.suffix.lower())


def unit_count(path: Path, file_type: str) -> int | None:
    """Pages (PDF) or sheets (XLSX); None where not applicable or unreadable."""
    if file_type == "pdf":
        return count_pages(path)
    if file_type == "xlsx":
        return count_sheets(path)
    return None


def extract_document(path: Path, document_id: str, filename: str) -> list[ExtractedUnit]:
    """Extract all content units. Raises ExtractionError for corrupt/empty files."""
    file_type = file_type_for(path)
    if file_type is None:
        raise UnsupportedDocumentError(f"Unsupported file type: {path.suffix}")
    extractor = {"pdf": extract_pdf, "docx": extract_docx, "xlsx": extract_xlsx}[file_type]
    units = extractor(path, document_id, filename)
    if not units:
        raise ExtractionError(f"No extractable content in '{filename}'")
    return units
