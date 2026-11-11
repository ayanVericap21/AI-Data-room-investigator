"""Scanned PDFs (no text layer) are left out of the run, not failed, and never retried."""
from __future__ import annotations

from pathlib import Path

import pymupdf as fitz
import pytest
from sqlalchemy import func, select

from app.core.exceptions import ScannedDocumentError
from app.llm.embeddings import MockEmbeddingProvider
from app.models import Document, DocumentChunk
from app.services.document_service import DocumentService
from app.services.extraction_service import extract_document
from app.services.extractors.pdf import check_pdf_text
from app.services.processing_service import ProcessingService

TEXT = "Annual production capacity is 30000 tonnes at the plant."


def make_mixed_pdf(path: Path, kinds: list[str]) -> Path:
    """kinds: 'text' pages carry a text layer; 'scan' pages are a picture only."""
    doc = fitz.open()
    picture = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 40, 40))
    picture.clear_with(200)
    for kind in kinds:
        page = doc.new_page()
        if kind == "text":
            page.insert_text((72, 100), TEXT)
        else:
            page.insert_image(page.rect, pixmap=picture)
    doc.save(path)
    doc.close()
    return path


def process(session, settings):
    svc = DocumentService(session, settings)
    svc.scan()
    return {r.filename: r for r in ProcessingService(session, svc, MockEmbeddingProvider()).process_pending()}


def test_check_reports_pages_without_a_text_layer(data_room):
    path = make_mixed_pdf(data_room / "m.pdf", ["text", "scan", "text", "scan"])
    check = check_pdf_text(path)
    assert check.pages == 4 and check.pages_without_text == [2, 4] and check.is_scanned()  # 50% is scanned
    assert not check_pdf_text(make_mixed_pdf(data_room / "t.pdf", ["text", "text", "scan"])).is_scanned()
    assert check_pdf_text(data_room / "missing.pdf") is None


def test_short_text_or_blank_pages_without_an_image_are_not_scans(data_room):
    doc = fitz.open()
    doc.new_page().insert_text((72, 100), "Thank you")  # a divider slide: 9 characters, no picture
    doc.new_page()  # a blank page
    doc.new_page().insert_text((72, 100), "Appendix")
    doc.save(data_room / "dividers.pdf")
    doc.close()
    check = check_pdf_text(data_room / "dividers.pdf")
    assert check.pages_without_text == [] and not check.is_scanned()


@pytest.mark.parametrize("kinds", [["scan"], ["scan", "scan", "scan"], ["text", "scan", "scan"]])
def test_extraction_refuses_a_scanned_pdf(data_room, kinds):
    path = make_mixed_pdf(data_room / "scan.pdf", kinds)
    with pytest.raises(ScannedDocumentError) as info:
        extract_document(path, "doc_x", "scan.pdf")
    assert "scanned" in info.value.message and "OCR" in info.value.message


def test_scanned_pdf_is_skipped_not_failed_and_has_no_chunks(session, settings, data_room):
    make_mixed_pdf(data_room / "scan.pdf", ["scan", "scan"])
    make_mixed_pdf(data_room / "real.pdf", ["text"])
    results = process(session, settings)

    assert results["scan.pdf"].status == "skipped" and not results["scan.pdf"].error
    assert "scanned" in results["scan.pdf"].note
    assert results["real.pdf"].status == "processed" and results["real.pdf"].chunks > 0
    scan_id = session.scalar(select(Document.id).where(Document.filename == "scan.pdf"))
    assert session.scalar(select(func.count()).select_from(DocumentChunk).where(DocumentChunk.document_id == scan_id)) == 0


def test_a_skipped_scan_is_not_retried_on_the_next_run(session, settings, data_room):
    make_mixed_pdf(data_room / "scan.pdf", ["scan"])
    process(session, settings)
    again = ProcessingService(session, DocumentService(session, settings), MockEmbeddingProvider()).process_pending()
    assert again == []  # skipped is not pending, so nothing is attempted


def test_a_mostly_text_pdf_is_processed_and_names_the_pages_it_could_not_read(session, settings, data_room):
    make_mixed_pdf(data_room / "deck.pdf", ["text", "text", "text", "scan"])
    result = process(session, settings)["deck.pdf"]
    assert result.status == "processed" and result.chunks > 0
    assert "[4]" in result.note and "no text layer" in result.note


def test_a_fully_text_pdf_has_no_note(session, settings, data_room):
    make_mixed_pdf(data_room / "ok.pdf", ["text", "text"])
    assert process(session, settings)["ok.pdf"].note is None


def test_listing_shows_the_skipped_status(session, settings, data_room):
    make_mixed_pdf(data_room / "scan.pdf", ["scan"])
    process(session, settings)
    assert [d.processing_status for d in DocumentService(session, settings).list_documents()] == ["skipped"]


# ---- per-part latency recording ----------------------------------------------
def test_each_part_of_processing_is_timed_for_a_pdf(session, settings, data_room):
    make_mixed_pdf(data_room / "deck.pdf", ["text", "text"])
    result = process(session, settings)["deck.pdf"]
    t = result.timings_ms
    assert {"extract", "chunk", "save", "embed"} <= set(t)
    assert {"extract.open", "extract.scan_check", "extract.tables", "extract.text"} <= set(t)
    assert "extract.recheck" not in t  # picture-only pages are reported by the extractor itself, no second pass
    assert all(v >= 0 for v in t.values())


def test_xlsx_parts_are_timed_and_a_skipped_scan_still_reports_its_time(session, settings, data_room):
    from tests.conftest import make_xlsx

    make_xlsx(data_room / "m.xlsx")
    make_mixed_pdf(data_room / "scan.pdf", ["scan"])
    results = process(session, settings)
    assert {"extract.load_formulas", "extract.load_values", "extract.read_rows"} <= set(results["m.xlsx"].timings_ms)
    assert results["scan.pdf"].status == "skipped" and "extract" in results["scan.pdf"].timings_ms
    assert "extract.scan_check" in results["scan.pdf"].timings_ms
