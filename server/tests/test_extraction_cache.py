"""A LlamaExtract response is kept by (file hash, schema hash, prompt hash, tier, pages) and read back instead of paying for the job again."""
from __future__ import annotations

from pathlib import Path

from sqlalchemy import func, select

from app.llm.llama_extract import LlamaJob
from app.models import Claim, ExtractionCacheEntry
from app.services.claim_service import ExtractionSummary
from app.services.extraction_cache import ExtractionCache, schema_digest
from app.services.llama_claims import SCHEMA, SYSTEM_PROMPT, LlamaClaimExtractor
from tests.test_llama_claims import FakeClient, claim, extractor  # noqa: F401
from tests.test_tools import room  # noqa: F401  (fixture)

RESULT = {"claims": [claim()], "references": [], "profile": None}
META = {"field_metadata": {"document_metadata": {"claims": [{"value": {"citation": [{"page": 1, "matching_text": "120,000"}]}}]}}}


def make_cache(session_factory, **kw):
    return ExtractionCache(session_factory, SCHEMA, SYSTEM_PROMPT, "cost_effective", **kw)


def run(session, room, client, cache, document=None):  # noqa: F811
    summary = ExtractionSummary()
    LlamaClaimExtractor(session, client, Path("/data_room"), cache=cache).extract_document(document or room.deck, summary)
    return summary


def test_the_second_run_on_an_unchanged_file_reads_the_stored_response_and_calls_nothing(session, session_factory, room):  # noqa: F811
    cache = make_cache(session_factory)
    first, second = FakeClient(RESULT, META, credits=48.0), FakeClient(RESULT, META, credits=48.0)
    s1 = run(session, room, first, cache)
    session.query(Claim).delete()
    session.commit()
    s2 = run(session, room, second, cache)
    assert len(first.calls) == 1 and second.calls == []  # nothing was sent the second time
    assert (s1.external_calls, s1.external_credits, s1.documents_from_cache) == (1, 48.0, 0)
    assert (s2.external_calls, s2.external_credits, s2.documents_from_cache, s2.extraction_credits_saved) == (0, 0.0, 1, 48.0)
    assert session.scalars(select(Claim)).one().value == "120,000"  # the same claim, made from the stored response


def test_the_checks_after_extraction_still_run_on_a_stored_response(session, session_factory, room):  # noqa: F811
    cache = make_cache(session_factory)
    page_number = claim(subject="Deck", metric="page number", value="Page 1 of 2", unit=None, claim_text="Page 1 of 2", value_type="statement")
    both = {"claims": [claim(), page_number], "references": [], "profile": None}
    run(session, room, FakeClient(both, {}), cache)
    session.query(Claim).delete()
    session.commit()
    s2 = run(session, room, FakeClient({}, {}), cache)  # read from the cache: the filter of today still applies
    assert s2.claims_skipped_page_numbers == 1 and session.scalars(select(Claim)).one().value == "120,000"


def test_a_changed_prompt_schema_tier_or_file_is_a_different_key(session_factory):
    base = make_cache(session_factory)
    job = LlamaJob(job_id="j", status="COMPLETED", result=RESULT, metadata=META, credits=8.0)
    base.put("hash-a", "", job)
    assert base.get("hash-a") is not None
    assert ExtractionCache(session_factory, SCHEMA, SYSTEM_PROMPT + " changed", "cost_effective").get("hash-a") is None
    assert ExtractionCache(session_factory, {**SCHEMA, "x": 1}, SYSTEM_PROMPT, "cost_effective").get("hash-a") is None
    assert ExtractionCache(session_factory, SCHEMA, SYSTEM_PROMPT, "agentic").get("hash-a") is None
    assert base.get("hash-b") is None and base.get("hash-a", "1-4") is None  # another file, another page range


def test_a_failed_or_empty_job_is_never_stored(session_factory):
    cache = make_cache(session_factory)
    cache.put("h", "", LlamaJob(job_id="j", status="FAILED", result=None))
    cache.put("h", "", LlamaJob(job_id="j", status="COMPLETED", result=None))
    with session_factory() as s:
        assert s.scalar(select(func.count()).select_from(ExtractionCacheEntry)) == 0


def test_refresh_reads_nothing_and_replaces_what_is_stored(session, session_factory, room):  # noqa: F811
    run(session, room, FakeClient(RESULT, META, credits=48.0), make_cache(session_factory))
    session.query(Claim).delete()
    session.commit()
    fresh = FakeClient({"claims": [claim(value="999")], "references": [], "profile": None}, META, credits=50.0)
    s = run(session, room, fresh, make_cache(session_factory, refresh=True))
    assert len(fresh.calls) == 1 and s.documents_from_cache == 0
    with session_factory() as db:
        row = db.scalar(select(ExtractionCacheEntry))
        assert db.scalar(select(func.count()).select_from(ExtractionCacheEntry)) == 1 and row.credits == 50.0 and row.result["claims"][0]["value"] == "999"


def test_a_database_error_is_reported_and_the_run_goes_on_without_the_cache(session, room):  # noqa: F811
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    broken = sessionmaker(bind=create_engine("sqlite://"))  # an empty database: the extraction_cache table does not exist
    cache = make_cache(broken)
    messages = []
    client = FakeClient(RESULT, META, credits=8.0)
    ex = LlamaClaimExtractor(session, client, Path("/data_room"), cache=cache, on_progress=lambda m, d: messages.append(m))
    summary = ExtractionSummary()
    ex.extract_document(room.deck, summary)
    assert len(client.calls) == 1 and summary.documents_processed == 1 and cache.error
    assert any("extraction cache is not working" in m for m in messages)


def test_the_schema_hash_does_not_depend_on_key_order():
    assert schema_digest({"a": 1, "b": {"c": 2, "d": 3}}) == schema_digest({"b": {"d": 3, "c": 2}, "a": 1})


def test_a_long_pdf_reads_only_the_page_ranges_that_are_not_stored(session, session_factory, room):  # noqa: F811
    from app.models import DocumentChunk

    for n in (5, 6, 7):  # the deck has chunks on pages 1 and 2: pad it to 7 pages so that 4-page ranges give 1-4 and 5-7
        session.add(DocumentChunk(document_id=room.deck.id, chunk_index=10 + n, content=f"page {n}", content_type="paragraph", page_number=n))
    session.commit()
    cache = make_cache(session_factory)
    cache.put(room.deck.file_hash, "1-4", LlamaJob(job_id="old", status="COMPLETED", result=RESULT, metadata=META, credits=32.0))
    client = FakeClient({"claims": [], "references": [], "profile": None}, {}, credits=24.0)
    client.upload = lambda path: "file-1"
    ex = LlamaClaimExtractor(session, client, Path("/data_room"), cache=cache, pages_per_job=4)
    ex.prefetch([room.deck], workers=2)
    summary = ExtractionSummary()
    ex.extract_document(room.deck, summary)
    ex.close()
    assert [kw.get("extra_configuration") for _name, kw in client.calls] == [{"target_pages": "5-7"}]  # only the missing range was read
    assert summary.external_credits == 24.0 and summary.extraction_credits_saved == 32.0 and summary.documents_from_cache == 0
