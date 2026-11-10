from decimal import Decimal

import pytest
from pydantic import ValidationError

from app.llm.errors import LLMTransientError
from app.llm.mock import MockLLMClient
from app.models import Claim, Document, DocumentChunk
from app.schemas.claim import ClaimExtractionResult, ClaimView, ExtractedClaim
from app.services.candidate_matching import (compare_attributes, generate_candidate_pairs, normalize_period,
                                              normalize_unit, relation_hint)
from app.services.claim_service import ClaimService
from app.services.numbers import numbers_in_text, parse_number, value_supported_by_text


# ---- numbers ---------------------------------------------------------------
@pytest.mark.parametrize("raw,expected", [(30000, "30000"), ("30,000", "30000"), ("$1,250.50", "1250.50"),
                                          ("12%", "12"), (2.5, "2.5"), ("abc", None), (None, None), (True, None)])
def test_parse_number(raw, expected):
    assert parse_number(raw) == (Decimal(expected) if expected else None)


def test_value_support_checks_source_text():
    text = "Capacity is 30,000 tonnes; price is $5.50 per tonne."
    assert Decimal("30000") in numbers_in_text(text)
    assert value_supported_by_text(30000, text)
    assert value_supported_by_text("5.50", text)
    assert not value_supported_by_text(25000, text)
    assert value_supported_by_text(None, text)
    assert value_supported_by_text("tonnes", text)


# ---- claim schema ----------------------------------------------------------
def test_claim_schema_nulls_and_validation():
    c = ExtractedClaim(chunk_id="c", claim_text="x", subject="  ", unit="")
    assert c.subject is None and c.unit is None and c.assumptions == []
    with pytest.raises(ValidationError):
        ExtractedClaim(chunk_id="c", claim_text="")
    with pytest.raises(ValidationError):
        ExtractedClaim(chunk_id="c", claim_text="x", confidence=1.5)


# ---- extraction service ----------------------------------------------------
@pytest.fixture
def doc_with_chunks(session):
    doc = Document(filename="a.pdf", file_hash="h" * 64, file_type="pdf", file_size=1,
                   last_modified_at=__import__("datetime").datetime.now(__import__("datetime").timezone.utc),
                   processing_status="processed")
    session.add(doc)
    session.flush()
    chunks = [DocumentChunk(document_id=doc.id, chunk_index=i, content=t, content_type="paragraph", page_number=1,
                            metadata_={}) for i, t in enumerate(
        ["Annual production capacity is 30,000 tonnes under optimized operations.", "Unrelated boilerplate text."])]
    session.add_all(chunks)
    session.commit()
    return doc, chunks


def claim(chunk_id, **kw):
    base = dict(chunk_id=chunk_id, subject="Production", metric="Annual capacity", value=30000, unit="tonnes",
                period="annual", conditions="optimized operations", claim_text="Capacity is 30,000 tonnes.",
                claim_category="capacity", confidence=0.9)
    base.update(kw)
    return base


def test_extraction_persists_valid_claims_with_source(session, doc_with_chunks):
    doc, chunks = doc_with_chunks
    llm = MockLLMClient([{"claims": [claim(chunks[0].id)]}])
    summary = ClaimService(session, llm).extract_all()
    assert (summary.claims_created, summary.documents_processed, summary.llm_calls) == (1, 1, 1)
    items, total = ClaimService(session, llm).list_claims()
    assert total == 1 and items[0].source.page_number == 1 and items[0].source.filename == "a.pdf"
    assert items[0].conditions == "optimized operations"


def test_invented_chunk_and_invented_number_are_rejected(session, doc_with_chunks):
    doc, chunks = doc_with_chunks
    llm = MockLLMClient([{"claims": [claim("chk_fake"), claim(chunks[0].id, value=99999), claim(chunks[0].id)]}])
    summary = ClaimService(session, llm).extract_all()
    assert summary.claims_created == 1 and summary.claims_rejected == 2
    assert any("unknown chunk_id" in r for r in summary.rejections)
    assert any("not found in cited chunk" in r for r in summary.rejections)


def test_unknown_attributes_stay_null(session, doc_with_chunks):
    _, chunks = doc_with_chunks
    llm = MockLLMClient([{"claims": [{"chunk_id": chunks[0].id, "claim_text": "Capacity mentioned."}]}])
    ClaimService(session, llm).extract_all()
    row = session.query(Claim).one()
    assert row.value is None and row.unit is None and row.period is None and row.assumptions is None


def test_idempotent_and_force(session, doc_with_chunks):
    _, chunks = doc_with_chunks
    llm = MockLLMClient([{"claims": [claim(chunks[0].id)]}, {"claims": [claim(chunks[0].id)]}])
    ClaimService(session, llm).extract_all()
    second = ClaimService(session, llm).extract_all()
    assert second.documents_skipped == 1 and second.llm_calls == 0
    ClaimService(session, llm).extract_all(force=True)
    assert session.query(Claim).count() == 1  # replaced, not duplicated


def test_provider_failure_leaves_no_partial_claims_and_can_rerun(session, doc_with_chunks):
    _, chunks = doc_with_chunks
    llm = MockLLMClient([LLMTransientError("429")] * 3, max_attempts=3)
    summary = ClaimService(session, llm).extract_all()
    assert summary.documents_failed == 1 and session.query(Claim).count() == 0
    ok = MockLLMClient([{"claims": [claim(chunks[0].id)]}])
    assert ClaimService(session, ok).extract_all().claims_created == 1


def test_invalid_llm_output_fails_document_cleanly(session, doc_with_chunks):
    llm = MockLLMClient(["garbage"] * 3, max_attempts=3)
    summary = ClaimService(session, llm).extract_all()
    assert summary.documents_failed == 1 and session.query(Claim).count() == 0


def test_stop_request_and_call_limit_prevent_llm_calls(session, doc_with_chunks):
    llm = MockLLMClient([])
    s = ClaimService(session, llm, should_stop=lambda: True).extract_all()
    assert s.stopped and s.llm_calls == 0 and llm.prompts == []
    s = ClaimService(session, MockLLMClient([]), max_llm_calls=0).extract_all()
    assert s.stopped and s.stop_reason == "LLM call limit reached" and s.llm_calls == 0


def test_batching_splits_by_size(session, doc_with_chunks):
    _, chunks = doc_with_chunks
    llm = MockLLMClient([{"claims": []}, {"claims": []}])
    ClaimService(session, llm, max_chars_per_call=40).extract_all()
    assert len(llm.prompts) == 2


# ---- candidate matching ----------------------------------------------------
def view(id, doc="d1", chunk=None, **kw):
    base = dict(subject="Production", metric="Annual capacity", value=30000, unit="tonnes", period="annual")
    base.update(kw)
    return ClaimView(id=id, document_id=doc, chunk_id=chunk or f"chk_{id}", **base)


def test_normalisation():
    assert normalize_unit("Tonnes") == normalize_unit("t") == "tonne"
    assert normalize_period("Per Year") == normalize_period("annually") == "annual"
    assert normalize_unit(" ") is None


def test_partial_match_example_from_spec_is_contextual_not_finding():
    a = view("a", doc="eng", conditions="optimized operations")
    b = view("b", doc="sup", value=25000, conditions="normal conditions")
    comp = compare_attributes(a, b)
    assert comp.unit_match and comp.period_match and comp.value_relation == "different" and comp.conditions_differ
    assert relation_hint(comp) == "contextual"
    pairs = generate_candidate_pairs([a, b])
    assert len(pairs) == 1 and pairs[0].relation_hint == "contextual"


def test_hints_exact_conflicting_partial():
    assert generate_candidate_pairs([view("a"), view("b")])[0].relation_hint == "exact"
    assert generate_candidate_pairs([view("a"), view("b", value=25000)])[0].relation_hint == "conflicting"
    assert generate_candidate_pairs([view("a", unit=None), view("b", value=25000, unit=None)])[0].relation_hint == "conflicting"
    assert generate_candidate_pairs([view("a"), view("b", value=None)])[0].relation_hint == "partial"
    assert generate_candidate_pairs([view("a", unit="kg"), view("b", value=25000)])[0].relation_hint == "partial"


def test_unrelated_claims_are_not_paired():
    a = view("a")
    b = view("b", subject="Office lease", metric="Monthly rent", value=5000, unit="USD", period="monthly")
    assert generate_candidate_pairs([a, b]) == []


def test_same_chunk_and_self_not_paired_and_pairs_deduplicated():
    a, b = view("a", chunk="same"), view("b", chunk="same")
    assert generate_candidate_pairs([a, b]) == []
    pairs = generate_candidate_pairs([view("a"), view("b"), view("c")])
    keys = {frozenset((p.claim_a_id, p.claim_b_id)) for p in pairs}
    assert len(pairs) == len(keys) == 3


def test_partner_cap_limits_pairs_per_claim():
    claims = [view(f"c{i}") for i in range(10)]
    pairs = generate_candidate_pairs(claims, max_partners_per_claim=2)
    from collections import Counter
    counts = Counter(x for p in pairs for x in (p.claim_a_id, p.claim_b_id))
    assert max(counts.values()) <= 2  # hard cap per claim
    assert len(pairs) <= 10
    assert generate_candidate_pairs(claims, max_pairs=3).__len__() == 3


def test_missing_attributes_fall_back_to_available_field():
    a = view("a", subject=None)
    b = view("b", subject=None, value=25000)
    assert len(generate_candidate_pairs([a, b])) == 1
    assert generate_candidate_pairs([view("x", metric=None, subject=None), view("y", metric=None, subject=None)]) == []
