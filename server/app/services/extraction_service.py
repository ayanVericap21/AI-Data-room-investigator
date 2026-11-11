"""Dispatches extraction by file type into the common ExtractedUnit structure."""
from __future__ import annotations

from pathlib import Path

from app.core.exceptions import ExtractionError, UnsupportedDocumentError
from app.schemas.extraction import ExtractedUnit, ExtractionReport
from app.services.extractors.docx import extract_docx
from app.services.extractors.pdf import count_pages, extract_pdf
from app.services.extractors.xlsx import count_sheets, extract_xlsx

# .xlsm (macro-enabled workbook) is read exactly like .xlsx: cells and formulas only. openpyxl
# never runs or keeps macros, so it is stored with the same "xlsx" file type.
SUPPORTED_EXTENSIONS = {".pdf": "pdf", ".docx": "docx", ".xlsx": "xlsx", ".xlsm": "xlsx"}


def file_type_for(path: Path) -> str | None:
    return SUPPORTED_EXTENSIONS.get(path.suffix.lower())


def unit_count(path: Path, file_type: str) -> int | None:
    """Pages (PDF) or sheets (XLSX); None where not applicable or unreadable."""
    if file_type == "pdf":
        return count_pages(path)
    if file_type == "xlsx":
        return count_sheets(path)
    return None


def extract_document(path: Path, document_id: str, filename: str,
                     report: ExtractionReport | None = None) -> list[ExtractedUnit]:
    """Extract all content units. Raises ExtractionError for corrupt/empty files.
    ``report`` (optional) receives per-part timings and notes for PDF and XLSX (see those extractors)."""
    file_type = file_type_for(path)
    if file_type is None:
        raise UnsupportedDocumentError(f"Unsupported file type: {path.suffix}")
    if file_type == "docx":
        units = extract_docx(path, document_id, filename)
    else:
        units = {"pdf": extract_pdf, "xlsx": extract_xlsx}[file_type](path, document_id, filename, report)
    if not units:
        raise ExtractionError(f"No extractable content in '{filename}'")
    return units
