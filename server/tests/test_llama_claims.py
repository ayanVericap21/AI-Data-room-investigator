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


def test_the_profile_the_extraction_returns_is_kept_and_null_fields_are_tolerated(session, room):  # noqa: F811
    result = {"claims": [claim()], "profile": {"main_entity": " Pune plant ", "document_type": "presentation", "purpose": None, "date": None, "status": None, "scope": "capacity"}}
    ex = extractor(session, room, FakeClient(result))
    ex.extract_document(room.deck, ExtractionSummary())
    name, profile = ex.profiles[room.deck.id]
    assert name == "deck.pdf" and profile.main_entity == "Pune plant" and profile.document_type == "presentation" and profile.purpose == "" and profile.scope == "capacity"


def test_an_empty_or_missing_profile_is_not_kept(session, room):  # noqa: F811
    ex = extractor(session, room, FakeClient({"claims": [claim()], "profile": {"main_entity": None}}))
    ex.extract_document(room.deck, ExtractionSummary())
    assert ex.profiles == {}
    ex = extractor(session, room, FakeClient({"claims": [claim()]}))
    ex.extract_document(room.deck, ExtractionSummary())
    assert ex.profiles == {}


def test_the_extraction_prompt_stays_under_the_service_limit_and_asks_for_the_profile():
    from app.services.llama_claims import SCHEMA, SYSTEM_PROMPT
    assert len(SYSTEM_PROMPT) < 10_000 and "profile" in SYSTEM_PROMPT and set(SCHEMA["properties"]) == {"claims", "references", "profile"}


def test_every_documents_job_is_started_ahead_and_each_result_is_still_stored_in_order(session, room):  # noqa: F811
    import threading
    import time

    class Slow(FakeClient):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self.in_flight = self.peak = 0
            self._lock = threading.Lock()

        def extract(self, path, schema, **kwargs):
            with self._lock:
                self.in_flight += 1
                self.peak = max(self.peak, self.in_flight)
            time.sleep(0.1)
            with self._lock:
                self.in_flight -= 1
            return super().extract(path, schema, **kwargs)

    client = Slow({"claims": [claim()]})
    ex = extractor(session, room, client)
    ex.prefetch([room.deck, room.capex, room.lease], workers=3)
    summary = ExtractionSummary()
    ex.extract_document(room.deck, summary)  # the jobs of the other documents are already running
    assert client.peak >= 2 and len(client.calls) <= 3
    ex.extract_document(room.capex, summary)
    ex.extract_document(room.lease, summary)
    ex.close()
    assert len(client.calls) == 3 and summary.external_calls == 3  # no document was extracted twice


def test_one_worker_reads_the_documents_one_by_one(session, room):  # noqa: F811
    client = FakeClient({"claims": [claim()]})
    ex = extractor(session, room, client)
    ex.prefetch([room.deck, room.capex], workers=1)
    assert client.calls == []
    ex.extract_document(room.deck, ExtractionSummary())
    assert len(client.calls) == 1


class RangeClient:
    """Answers each page range with the claim stated on those pages, and records what was uploaded and which ranges were asked for."""

    def __init__(self, by_pages):
        self.by_pages, self.uploads, self.ranges = by_pages, [], []

    def upload(self, path):
        self.uploads.append(Path(path).name)
        return "file-1"

    def extract(self, path, schema, file_id=None, **kwargs):
        pages = kwargs["extra_configuration"]["target_pages"]
        self.ranges.append((file_id, pages))
        value, text, page = self.by_pages[pages]
        return LlamaJob(job_id=f"job-{pages}", status="COMPLETED", credits=4.0, result={"claims": [claim(value=value, claim_text=text)]},
                        metadata={"field_metadata": {"document_metadata": {"claims": [{"value": citation(page, value)}]}}})


def test_a_long_pdf_is_read_in_page_ranges_uploaded_once_and_each_claim_keeps_its_own_page(session, room):  # noqa: F811
    client = RangeClient({"1-1": ("120,000", "capacity is 120,000 tonnes per year", 1), "2-2": ("18%", "EBITDA margin for 2024 was 18% of revenue.", 2)})
    ex = LlamaClaimExtractor(session, client, Path("/data_room"), pages_per_job=1)
    ex.prefetch([room.deck], workers=4)
    summary = ExtractionSummary()
    ex.extract_document(room.deck, summary)
    ex.close()
    rows = {r.value: r.chunk_id for r in session.scalars(select(Claim))}
    assert client.uploads == ["deck.pdf"] and sorted(client.ranges) == [("file-1", "1-1"), ("file-1", "2-2")]
    assert rows == {"120,000": room.deck_p1, "18%": room.deck_p2} and summary.external_credits == 8.0


def test_a_failed_page_range_fails_the_whole_document():
    from app.services.llama_claims import merge_jobs
    ok = LlamaJob(job_id="a", status="COMPLETED", result={"claims": []})
    bad = LlamaJob(job_id="b", status="FAILED", error="boom")
    assert merge_jobs([ok, bad]) is bad and merge_jobs([ok]) is ok


def test_a_short_pdf_and_a_document_that_is_not_a_pdf_are_read_in_one_job(session, room):  # noqa: F811
    ex = LlamaClaimExtractor(session, FakeClient({"claims": []}), Path("/data_room"), pages_per_job=5)
    assert ex._page_ranges(room.deck) == [] and ex._page_ranges(room.capex) == []
    assert LlamaClaimExtractor(session, FakeClient({}), Path("/data_room"), pages_per_job=1)._page_ranges(room.deck) == ["1-1", "2-2"]


# ---- what the extraction call says beyond the value --------------------------------------------------------------------------------------
def test_the_tags_the_extraction_call_returns_are_stored_with_the_claim(session, room):  # noqa: F811
    client = FakeClient({"claims": [claim(basis="planned", precision="approximate", anchor="event start", complete=True, about="document", quantifier="none")]})
    extractor(session, room, client).extract_document(room.deck, ExtractionSummary())
    assert session.scalars(select(Claim)).one().assumptions == ["@about: document", "@basis: planned", "@precision: approximate", "@quantifier: none", "@anchor: event start", "@complete: list"]


def test_a_claim_with_only_defaults_stores_no_tags(session, room):  # noqa: F811
    extractor(session, room, FakeClient({"claims": [claim(about="content", precision="exact")]})).extract_document(room.deck, ExtractionSummary())
    assert not session.scalars(select(Claim)).one().assumptions


def test_a_year_that_is_part_of_the_subjects_name_is_not_a_date(session, room):  # noqa: F811
    summary = ExtractionSummary()
    extractor(session, room, FakeClient({"claims": [claim(subject="Plant Forum 2024", metric="event date", value="2024", value_type="date", unit=None,
                                                          claim_text="EBITDA margin for 2024 was 18% of revenue.")]})).extract_document(room.deck, summary)
    assert session.scalar(select(Claim)) is None and summary.claims_skipped_by_rule == 1


@pytest.mark.parametrize("metric,value,text", [
    ("page number", "Page 6 of 6", "Ashby Mill Lofts | Financial Model Summary | Page 6 of 6"),
    ("total page count", "17 of 18", "17 of 18"),
    ("document page number", "22", "Operating Agreement Excerpt | 22"),
    ("total pages", "34 page", "34 page"),
    ("page", "Page 4", "Page 4"),
])
def test_a_page_number_printed_on_the_page_is_skipped_and_counted(session, room, metric, value, text):  # noqa: F811
    summary = ExtractionSummary()
    extractor(session, room, FakeClient({"claims": [claim(subject="Financial Model Summary", metric=metric, value=value, unit=None, claim_text=text, value_type="statement")]})).extract_document(room.deck, summary)
    assert session.scalar(select(Claim)) is None and summary.claims_skipped_page_numbers == 1


def test_a_title_a_version_a_date_and_a_page_count_stated_in_a_sentence_are_still_claims(session, room):  # noqa: F811
    from app.services.llama_claims import LlamaClaim, _is_page_number

    kept = [
        dict(subject="Financial Model Summary", metric="document title", value="Financial Model Summary", claim_text="Ashby Mill Lofts | Financial Model Summary"),
        dict(subject="Financial Model Summary", metric="document version date", value="March 1, 2025", claim_text="Version dated March 1, 2025"),
        dict(subject="agreement", metric="revision", value="Revised Draft", claim_text="Revised Draft"),
        dict(subject="report", metric="total pages", value="24", claim_text="The full report has 24 pages and is available on request from the sponsor."),
        dict(subject="plant", metric="page rate", value="12", claim_text="Pages are printed at 12 per minute"),
    ]
    assert not any(_is_page_number(LlamaClaim(**k)) for k in kept)


def test_the_extraction_prompt_drops_page_numbers_but_keeps_titles_versions_and_dates():
    from app.services.llama_claims import SYSTEM_PROMPT

    assert "PAGE NUMBERS" in SYSTEM_PROMPT and "return no claim for it" in SYSTEM_PROMPT
    assert "title, its version or revision and its dates are still claims" in SYSTEM_PROMPT


def test_a_disclaimer_sentence_gives_neither_a_claim_nor_a_reference(session, room):  # noqa: F811
    sentence = "This report does not replace the audited accounts, the venue contract or the delegate feedback forms."
    client = FakeClient({"claims": [claim(subject="report", metric="replacement", value="does not replace", unit=None, claim_text=sentence, value_type="statement")],
                         "references": [{"item": "audited accounts", "quote": sentence}]})
    ex = extractor(session, room, client)
    summary = ExtractionSummary()
    ex.extract_document(room.deck, summary)
    assert session.scalar(select(Claim)) is None and ex.references == [] and summary.claims_skipped_by_rule == 1


def test_the_issuer_goes_into_the_purpose_and_the_main_entity_stays_what_the_document_is_about():
    from app.services.llama_claims import LlamaProfile
    profile = LlamaProfile(main_entity="Lakeside Health Forum 2025", purpose="Reports the costs.", issuer="Norvale Medical Society").to_profile()
    assert profile.main_entity == "Lakeside Health Forum 2025" and profile.purpose == "Reports the costs. Issued by Norvale Medical Society."


def test_tags_never_make_two_claims_differ_in_context():
    from app.schemas.claim import ClaimView
    from app.services.candidate_matching import compare_attributes
    a = ClaimView(id="a", document_id="d1", chunk_id="c1", subject="x", metric="m", value="1", assumptions=["@basis: actual"])
    b = ClaimView(id="b", document_id="d2", chunk_id="c2", subject="x", metric="m", value="2", assumptions=[])
    assert compare_attributes(a, b).assumptions_differ is False


def test_the_same_claim_returned_twice_for_a_document_is_kept_once_and_a_rule_flag_is_not_lost(session, room):  # noqa: F811
    client = FakeClient({"claims": [claim(), claim(), claim(is_rule=True), claim(metric="capacity at year end"),
                                    claim(claim_text="capacity is 120,000 tonnes per year, as stated again")]})
    summary = ExtractionSummary()
    extractor(session, room, client).extract_document(room.deck, summary)
    rows = session.scalars(select(Claim)).all()
    assert len(rows) == 2 and summary.claims_skipped_duplicates == 3 and summary.claims_created == 2  # the same sentence with another wording of the metric is a repeat; another sentence is not
    assert rows[0].assumptions == ["@rule: yes"]  # the repeat carried the flag
