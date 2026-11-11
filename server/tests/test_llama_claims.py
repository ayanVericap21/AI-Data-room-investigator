"""LlamaExtract claims are tied to stored chunks and verified like any other claim. The service is replaced by a fake client."""
from __future__ import annotations

from pathlib import Path

import httpx
import pytest
from sqlalchemy import select

from app.llm.llama_extract import LlamaExtractClient, LlamaExtractError, LlamaJob
from app.models import Claim
from app.services.claim_service import ExtractionSummary
from app.services.llama_claims import LlamaClaimExtractor
from tests.test_tools import room  # noqa: F401  (fixture)


class FakeClient:
    def __init__(self, result=None, metadata=None, error: Exception | None = None, status="COMPLETED", credits=8.0):
        self.job = LlamaJob(job_id="ext-1", status=status, result=result, metadata=metadata or {}, credits=credits)
        self.error, self.calls = error, []

    def extract(self, path, schema, **kwargs):
        self.calls.append((Path(path).name, kwargs))
        if self.error:
            raise self.error
        return self.job


def citation(page, text):
    return {"citation": [{"page": page, "matching_text": text}]}


def extractor(session, room, client):  # noqa: F811
    return LlamaClaimExtractor(session, client, Path("/data_room"))


def claim(**kw):
    base = dict(subject="plant", metric="capacity", value="120,000", unit="tonnes", claim_text="capacity is 120,000 tonnes per year")
    base.update(kw)
    return base


def test_a_claim_is_tied_to_the_chunk_on_its_cited_page_and_stored(session, room):  # noqa: F811
    meta = {"field_metadata": {"document_metadata": {"claims": [{"value": citation(1, "120,000"), "claim_text": citation(1, "x")}]}}}
    client = FakeClient({"claims": [claim()]}, meta)
    summary = ExtractionSummary()
    extractor(session, room, client).extract_document(room.deck, summary)
    row = session.scalars(select(Claim)).one()
    assert row.chunk_id == room.deck_p1 and row.value == "120,000" and row.document_id == room.deck.id
    assert (summary.claims_created, summary.documents_processed, summary.external_calls, summary.external_credits) == (1, 1, 1, 8.0)
    assert client.calls[0][0] == "deck.pdf" and "system_prompt" in client.calls[0][1]


def test_a_value_that_is_in_no_chunk_is_rejected_and_nothing_invented_is_stored(session, room):  # noqa: F811
    client = FakeClient({"claims": [claim(value="999,999", claim_text="made up"), claim(value="18%", metric="EBITDA margin",
                                                                                      claim_text="EBITDA margin for 2024 was 18% of revenue.")]})
    summary = ExtractionSummary()
    extractor(session, room, client).extract_document(room.deck, summary)
    rows = session.scalars(select(Claim)).all()
    assert [r.value for r in rows] == ["18%"] and rows[0].chunk_id == room.deck_p2  # found without any citation, by its text
    assert summary.claims_rejected == 1 and "999,999" in summary.rejections[0]


def test_documents_without_pages_are_matched_by_text(session, room):  # noqa: F811
    client = FakeClient({"claims": [claim(value="5%", metric="rent escalation", claim_text="Rent escalates 5% each year.")]},
                        {"field_metadata": {"document_metadata": {"claims": [{"value": citation(1, "5%")}]}}})  # a page that docx chunks do not have
    extractor(session, room, client).extract_document(room.lease, ExtractionSummary())
    assert session.scalars(select(Claim)).one().chunk_id == room.rent_2


def test_a_failed_document_keeps_no_claims_and_running_out_of_credits_stops_the_run(session, room):  # noqa: F811
    summary = ExtractionSummary()
    ex = extractor(session, room, FakeClient(error=LlamaExtractError("boom", 500)))
    ex.extract_document(room.deck, summary)
    assert (summary.documents_failed, summary.stopped) == (1, False) and "boom" in summary.notes[0]
    ex = extractor(session, room, FakeClient(error=LlamaExtractError("no credits", 402)))
    ex.extract_document(room.deck, summary)
    assert summary.stopped and summary.stop_reason == "LlamaExtract is out of credits"
    assert session.scalars(select(Claim)).all() == []


def test_a_job_that_did_not_complete_is_a_failed_document(session, room):  # noqa: F811
    summary = ExtractionSummary()
    extractor(session, room, FakeClient({"claims": []}, status="FAILED")).extract_document(room.deck, summary)
    assert summary.documents_failed == 1 and "FAILED" in summary.notes[0]


def test_the_client_uploads_creates_polls_and_never_leaks_the_key(tmp_path):
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.method, request.url.path, request.headers.get("authorization")))
        if request.url.path == "/api/v1/beta/files":
            return httpx.Response(200, json={"id": "dfl-1"})
        if request.method == "POST":
            return httpx.Response(200, json={"id": "ext-1"})
        status = "COMPLETED" if len([s for s in seen if s[0] == "GET"]) > 1 else "RUNNING"
        return httpx.Response(200, json={"id": "ext-1", "status": status, "extract_result": {"claims": []},
                                         "usage": {"credits": 5.0}, "metadata": {"usage": {"num_pages_billed": 1}}})

    client = LlamaExtractClient("secret-key", http=httpx.Client(base_url="https://x", transport=httpx.MockTransport(handler)), poll_seconds=0)
    pdf = tmp_path / "a.pdf"
    pdf.write_bytes(b"%PDF")
    job = client.extract(pdf, {"type": "object"})
    assert [(m, p) for m, p, _ in seen[:2]] == [("POST", "/api/v1/beta/files"), ("POST", "/api/v2/extract")]
    assert job.status == "COMPLETED" and job.credits == 5.0 and job.pages_billed == 1
    assert all(auth == "Bearer secret-key" for _, _, auth in seen)

    def deny(request):
        return httpx.Response(402, text="payment required")

    broke = LlamaExtractClient("secret-key", http=httpx.Client(base_url="https://x", transport=httpx.MockTransport(deny)))
    with pytest.raises(LlamaExtractError) as err:
        broke.upload(pdf)
    assert err.value.code == 402 and "secret-key" not in str(err.value) and "out of credits" in str(err.value)
    with pytest.raises(LlamaExtractError):
        LlamaExtractClient("")
