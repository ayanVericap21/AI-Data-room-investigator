"""Parallel parsing gives the same results as one at a time, and artwork pages skip table detection."""
from __future__ import annotations

import pymupdf as fitz
import pytest

from app.llm.embeddings import MockEmbeddingProvider
from app.schemas.extraction import ExtractionReport
from app.services.document_service import DocumentService
from app.services.extraction_service import extract_document
from app.services.extractors import pdf as pdf_extractor
from app.services.processing_service import ProcessingService
from tests.conftest import make_docx, make_xlsx
from tests.test_scanned_pdf import make_mixed_pdf


# ---- artwork pages skip table detection --------------------------------------
def make_artwork_pdf(path, shapes: int):
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 100), "Plant capacity is 120,000 tonnes per year at the Pune site.")
    for i in range(shapes):
        page.draw_rect(fitz.Rect(10 + i, 200, 14 + i, 204), color=(0, 0, 0))
    doc.save(path)
    doc.close()
    return path


def test_a_page_with_more_drawings_than_the_limit_skips_table_detection_but_keeps_its_text(data_room, monkeypatch):
    monkeypatch.setattr(pdf_extractor, "MAX_DRAWINGS_FOR_TABLES", 20)
    path = make_artwork_pdf(data_room / "map.pdf", shapes=50)
    report = ExtractionReport()
    units = extract_document(path, "d", "map.pdf", report)
    assert any("Plant capacity is 120,000 tonnes" in u.content for u in units)  # the text is still extracted
    assert len(report.notes) == 1 and "Table detection skipped on 1 artwork-heavy pages" in report.notes[0] and "[1]" in report.notes[0]
    assert "tables" in report.timings_ms


def test_a_normal_page_is_still_checked_for_tables(data_room, monkeypatch):
    monkeypatch.setattr(pdf_extractor, "MAX_DRAWINGS_FOR_TABLES", 20)
    report = ExtractionReport()
    extract_document(make_artwork_pdf(data_room / "plain.pdf", shapes=5), "d", "plain.pdf", report)
    assert report.notes == []


def test_the_default_limit_is_ten_thousand_paths():
    assert pdf_extractor.MAX_DRAWINGS_FOR_TABLES == 10_000


# ---- parallel == sequential --------------------------------------------------
def build_room(data_room):
    make_mixed_pdf(data_room / "a_text.pdf", ["text", "text"])
    make_mixed_pdf(data_room / "b_scan.pdf", ["scan", "scan"])
    make_mixed_pdf(data_room / "c_partial.pdf", ["text", "text", "text", "scan"])
    make_xlsx(data_room / "d_model.xlsx")
    make_docx(data_room / "e_ops.docx")
    (data_room / "f_corrupt.pdf").write_bytes(b"garbage")


def run(session_factory, settings, workers):
    with session_factory() as session:
        svc = DocumentService(session, settings)
        svc.scan()
        processing = ProcessingService(session, svc, MockEmbeddingProvider(), extract_workers=workers)
        results = {r.filename: r for r in processing.process_pending()}
        return results, processing.extract_phase_ms


def summary(results):
    return {name: (r.status, r.chunks, r.note, r.error) for name, r in results.items()}


def test_parallel_parsing_gives_identical_results_to_one_at_a_time(session_factory, settings, data_room):
    build_room(data_room)
    sequential, seq_phase = run(session_factory, settings, workers=1)
    assert seq_phase == 0
    assert {r.status for r in sequential.values()} == {"processed", "skipped", "failed"}

    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy.pool import StaticPool
    from app.models import Base

    engine = create_engine("sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    parallel, par_phase = run(sessionmaker(bind=engine, expire_on_commit=False), settings, workers=3)
    engine.dispose()

    assert summary(parallel) == summary(sequential)  # same status, chunk count, note and error for every file
    assert par_phase > 0
    assert parallel["b_scan.pdf"].status == "skipped" and "scanned" in parallel["b_scan.pdf"].note
    assert parallel["f_corrupt.pdf"].status == "failed" and parallel["f_corrupt.pdf"].error
    assert "[4]" in parallel["c_partial.pdf"].note
    assert {"extract", "chunk", "save", "embed"} <= set(parallel["a_text.pdf"].timings_ms)  # per-part times still recorded


def test_if_the_worker_pool_fails_documents_are_parsed_one_at_a_time(session_factory, settings, data_room, monkeypatch):
    class BrokenPool:
        def __init__(self, *a, **k):
            raise RuntimeError("no processes available")

    monkeypatch.setattr("app.services.processing_service.ProcessPoolExecutor", BrokenPool)
    build_room(data_room)
    results, phase = run(session_factory, settings, workers=4)
    assert phase == 0 and results["a_text.pdf"].status == "processed" and results["b_scan.pdf"].status == "skipped"


def test_a_single_document_is_not_sent_to_a_pool(session_factory, settings, data_room, monkeypatch):
    class NoPool:
        def __init__(self, *a, **k):
            raise AssertionError("a pool should not be started for one document")

    monkeypatch.setattr("app.services.processing_service.ProcessPoolExecutor", NoPool)
    make_mixed_pdf(data_room / "only.pdf", ["text"])
    results, phase = run(session_factory, settings, workers=4)
    assert results["only.pdf"].status == "processed" and phase == 0


def test_workers_setting_defaults_to_four_and_reads_the_environment(monkeypatch):
    from app.core.config import Settings, load_settings

    assert Settings().extract_workers == 4
    monkeypatch.setenv("EXTRACT_WORKERS", "2")
    assert load_settings().extract_workers == 2
    with pytest.raises(ValueError):
        Settings(extract_workers=0)
