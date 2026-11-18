"""Dispatches extraction by file type into the common ExtractedUnit structure."""
from __future__ import annotations

from pathlib import Path

from app.core.exceptions import ExtractionError, UnsupportedDocumentError
from app.llm.llama_extract import LlamaExtractError
from app.llm.llama_parse import LlamaParseClient, OcrConfig, OcrResult
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


def ocr_function(config: OcrConfig):
    """OCR of picture-only PDF pages through LlamaParse. A failed job (no credits, network, the service) fails the document with the reason."""
    def run(path: Path, pages: list[int]) -> OcrResult:
        try:
            return LlamaParseClient(config.api_key, config.project_id, tier=config.tier).parse_pages(path, pages)
        except LlamaExtractError as exc:
            raise ExtractionError(f"OCR of '{path.name}' failed: {exc}") from exc
    return run


def extract_document(path: Path, document_id: str, filename: str,
                     report: ExtractionReport | None = None, ocr: OcrConfig | None = None) -> list[ExtractedUnit]:
    """Extract all content units. Raises ExtractionError for corrupt/empty files.
    ``report`` (optional) receives per-part timings and notes for PDF and XLSX (see those extractors). ``ocr`` (PDF only): read picture-only pages by OCR."""
    file_type = file_type_for(path)
    if file_type is None:
        raise UnsupportedDocumentError(f"Unsupported file type: {path.suffix}")
    if file_type == "docx":
        units = extract_docx(path, document_id, filename)
    elif file_type == "pdf":
        units = extract_pdf(path, document_id, filename, report, ocr_function(ocr) if ocr is not None else None)
    else:
        units = extract_xlsx(path, document_id, filename, report)
    if not units:
        raise ExtractionError(f"No extractable content in '{filename}'")
    return units
