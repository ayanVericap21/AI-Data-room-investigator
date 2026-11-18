"""OCR of picture-only PDF pages through LlamaParse: the text comes back first, then the chunks for those pages are made from it."""
from __future__ import annotations

import json

import httpx
import pytest
from sqlalchemy import select

from app.core.config import Settings
from app.core.exceptions import ExtractionError
from app.llm.embeddings import MockEmbeddingProvider
from app.llm.llama_extract import BASE_URL, LlamaExtractError
from app.llm.llama_parse import LlamaParseClient, OcrConfig, OcrResult, page_ranges
from app.models import Document, DocumentChunk
from app.schemas.extraction import ExtractionReport
from app.services.document_service import DocumentService
from app.services.extraction_service import extract_document
from app.services.extractors.pdf import extract_pdf
from app.services.processing_service import ProcessingService
from tests.test_scanned_pdf import TEXT, make_mixed_pdf

SCAN_TEXT = "Plant lease runs 25 years from 1 March 2025.\n\n| Item | Rent |\n| --- | --- |\n| Base | 120,000 |"


def fake_ocr(texts: dict[int, str], credits: float = 6.0):
    calls = []

    def run(path, pages):
        calls.append(list(pages))
        return OcrResult(pages={p: texts.get(p, "") for p in pages}, credits=credits)
    run.calls = calls
    return run


def test_page_ranges_are_written_the_way_llamaparse_reads_them():
    assert page_ranges([1, 2, 3, 5, 7, 8]) == "1-3,5,7-8"
    assert page_ranges([4]) == "4" and page_ranges([9, 2, 2]) == "2,9"


def test_the_text_of_picture_only_pages_becomes_units_on_their_own_pages_in_order(data_room):
    path = make_mixed_pdf(data_room / "m.pdf", ["text", "scan", "text", "scan"])
    ocr, report = fake_ocr({2: SCAN_TEXT, 4: "Closing balance 9,500."}), ExtractionReport()
    units = extract_pdf(path, "d1", "m.pdf", report, ocr)
    assert ocr.calls == [[2, 4]]  # only the picture-only pages are sent
    assert [(u.locator.page, u.content_type) for u in units] == [(1, "paragraph"), (2, "paragraph"), (2, "table"), (3, "paragraph"), (4, "paragraph")]
    assert all(u.metadata.get("ocr") for u in units if u.locator.page in (2, 4)) and not units[0].metadata.get("ocr")
    assert units[2].content.startswith("| Item | Rent |") and "Closing balance 9,500." == units[4].content
    assert "ocr" in report.timings_ms and "OCR read 2 of 2 picture-only pages (6 credits)" in report.notes[0]


def test_a_fully_scanned_pdf_is_read_instead_of_refused_and_pages_with_no_text_back_are_reported(data_room):
    path = make_mixed_pdf(data_room / "s.pdf", ["scan", "scan"])
    report = ExtractionReport()
    units = extract_pdf(path, "d1", "s.pdf", report, fake_ocr({1: "Only page one was readable."}))
    assert [u.locator.page for u in units] == [1]
    assert "no text came back for pages [2]" in report.notes[0]


def test_without_ocr_a_mostly_scanned_pdf_is_still_refused(data_room):
    from app.core.exceptions import ScannedDocumentError
    with pytest.raises(ScannedDocumentError):
        extract_pdf(make_mixed_pdf(data_room / "s.pdf", ["scan", "scan"]), "d1", "s.pdf")


def test_a_failed_ocr_job_fails_the_document_with_the_reason(data_room, monkeypatch):
    def fail(self, path, pages):
        raise LlamaExtractError("LlamaExtract POST /api/v2/parse -> 402 (out of credits)", 402)
    monkeypatch.setattr(LlamaParseClient, "parse_pages", fail)
    with pytest.raises(ExtractionError, match="OCR of 's.pdf' failed.*out of credits"):
        extract_document(make_mixed_pdf(data_room / "s.pdf", ["scan"]), "d1", "s.pdf", ocr=OcrConfig(api_key="k"))


def client_with(handler) -> LlamaParseClient:
    return LlamaParseClient("secret-key", poll_seconds=0, http=httpx.Client(base_url=BASE_URL, transport=httpx.MockTransport(handler)))


def parse_handler(pages: list[dict], status: str = "COMPLETED", seen: list | None = None):
    def handler(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append((request.method, request.url.path, request.read()))
        assert request.headers["authorization"] == "Bearer secret-key"
        if request.url.path == "/api/v1/beta/files":
            return httpx.Response(200, json={"id": "file_1"})
        if request.method == "POST" and request.url.path == "/api/v2/parse":
            return httpx.Response(200, json={"id": "job_1", "status": "PENDING"})
        return httpx.Response(200, json={"job": {"id": "job_1", "status": status, "error_message": "bad file" if status == "FAILED" else None,
                                                 "usage": {"credits": 9.0}}, "markdown": {"pages": pages}})
    return handler


def test_parse_pages_uploads_for_parsing_asks_for_only_the_named_pages_and_returns_their_text(data_room):
    seen: list = []
    pages = [{"page_number": 1, "success": True, "markdown": "first scan"}, {"page_number": 2, "success": True, "markdown": " second scan "},
             {"page_number": 3, "success": True, "markdown": "third scan"}]
    result = client_with(parse_handler(pages, seen=seen)).parse_pages(make_mixed_pdf(data_room / "s.pdf", ["scan"] * 3), [2, 3, 5])
    assert result.pages == {2: "first scan", 3: "second scan", 5: "third scan"}  # one entry per requested page: taken in order
    assert result.credits == 9.0
    upload, create = seen[0], seen[1]
    assert b'name="purpose"' in upload[2] and b"parse" in upload[2]
    assert json.loads(create[2]) == {"file_id": "file_1", "tier": "cost_effective", "version": "latest", "page_ranges": {"target_pages": "2-3,5"}}


def test_a_failed_page_gives_no_text_and_an_unfinished_job_is_an_error(data_room):
    path = make_mixed_pdf(data_room / "s.pdf", ["scan", "scan"])
    pages = [{"page_number": 1, "success": True, "markdown": "ok"}, {"page_number": 2, "success": False, "error": "x"}]
    result = client_with(parse_handler(pages)).parse_pages(path, [1, 2])
    assert result.pages == {1: "ok", 2: ""} and result.failed_pages == [2]
    with pytest.raises(LlamaExtractError, match="ended as FAILED: bad file"):
        client_with(parse_handler([], status="FAILED")).parse_pages(path, [1])


def test_results_that_do_not_match_the_request_are_matched_by_page_number_or_refused(data_room):
    path = make_mixed_pdf(data_room / "s.pdf", ["scan"] * 4)
    extra = [{"page_number": 1, "markdown": "a"}, {"page_number": 2, "markdown": "b"}, {"page_number": 3, "markdown": "c"}]
    assert client_with(parse_handler(extra)).parse_pages(path, [2, 3]).pages == {2: "b", 3: "c"}
    with pytest.raises(LlamaExtractError, match="none of them is numbered as requested"):
        client_with(parse_handler([{"page_number": 9, "markdown": "z"}] * 3)).parse_pages(path, [1, 2])


def test_a_document_skipped_as_scanned_is_read_once_ocr_is_available_and_its_chunks_carry_the_page(session, settings, data_room, monkeypatch):
    make_mixed_pdf(data_room / "scan.pdf", ["scan", "scan"])
    svc = DocumentService(session, settings)
    svc.scan()
    first = {r.filename: r for r in ProcessingService(session, svc, MockEmbeddingProvider()).process_pending()}
    assert first["scan.pdf"].status == "skipped"

    monkeypatch.setattr(LlamaParseClient, "parse_pages", lambda self, path, pages: OcrResult(pages={p: f"Rent is 120,000 per year (page {p})." for p in pages}, credits=6))
    second = {r.filename: r for r in ProcessingService(session, svc, MockEmbeddingProvider(), ocr=OcrConfig(api_key="k")).process_pending()}
    assert second["scan.pdf"].status == "processed" and "OCR read 2 of 2" in second["scan.pdf"].note
    doc = session.scalars(select(Document)).one()
    chunks = list(session.scalars(select(DocumentChunk).where(DocumentChunk.document_id == doc.id).order_by(DocumentChunk.chunk_index)))
    assert [c.page_number for c in chunks] == [1, 2] and "page 2" in chunks[1].content


def test_ocr_is_on_only_with_a_key_and_when_third_party_apis_are_allowed():
    assert Settings().ocr_config is None  # no key
    on = Settings(llama_cloud_api_key="k", llama_cloud_project_id="p")
    assert on.ocr_config == OcrConfig(api_key="k", project_id="p", tier="cost_effective")
    assert Settings(llama_cloud_api_key="k", ocr_enabled=False).ocr_config is None
    assert Settings(llama_cloud_api_key="k", allow_third_party_apis=False).ocr_config is None
    assert TEXT  # the shared fixture text is used by the scanned-PDF tests
