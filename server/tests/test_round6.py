"""Round 6: ellipsis restored, no-claim documents flagged, references searched in the text, names and limits in the judge, identifier reconciliation."""
from pathlib import Path

from sqlalchemy import select

from app.llm.mock import MockLLMClient
from app.models import Claim, Evidence, Finding
from app.services.claim_service import ExtractionSummary
from app.services.llama_claims import SCHEMA, SYSTEM_PROMPT, LlamaClaimExtractor, expand_ellipsis
from app.services.missing_items import DocReference, MissingItemChecker
from app.services.pair_judge import PairJudge
from app.services.record_checks import IdentifierChecker, mentions_in
from app.services.value_rules import bound_direction, comparison_note, one_contains_other, within_limit
from app.tools import build_registry
from tests.test_llama_claims import FakeClient
from tests.test_pair_judge import pair
from tests.test_tools import add_doc, room  # noqa: F401  (fixtures and helper)


def registry(session, room):  # noqa: F811
    return build_registry(session, None, room.investigation.id)


# ---- 1: an ellipsis in a value is restored from the source; a document with values and no claim is flagged ---------------------
def test_a_value_shortened_with_an_ellipsis_is_restored_from_the_source_text(session, room):  # noqa: F811
    doc, chunks = add_doc(session, "permit.pdf", "pdf", [dict(content="Condition 1. NO biological sampling\nor collection of any kind is permitted in a Level 1 zone.", page_number=1)])
    assert expand_ellipsis("NO biological sampling ... is permitted", chunks)[1] == "NO biological sampling\nor collection of any kind is permitted"
    assert expand_ellipsis("is permitted ... NO biological sampling", chunks) is None  # pieces out of order
    assert expand_ellipsis("NO biological sampling", chunks) is None  # nothing to restore
    summary = ExtractionSummary()
    claim = dict(subject="Charlie", metric="restriction", value="NO biological sampling ... is permitted", claim_text="NO biological sampling or collection of any kind is permitted", value_type="statement")
    LlamaClaimExtractor(session, FakeClient({"claims": [claim]}), Path("/data_room")).extract_document(doc, summary)
    stored = session.scalars(select(Claim)).one()
    assert "..." not in str(stored.value) and str(stored.value).endswith("is permitted") and summary.claims_ellipsis_expanded == 1 and summary.claims_rejected == 0
    assert "ellipsis" in SYSTEM_PROMPT


def test_a_document_with_numbers_and_no_claims_is_flagged(session, room):  # noqa: F811
    summary = ExtractionSummary()
    LlamaClaimExtractor(session, FakeClient({"claims": []}), Path("/data_room")).extract_document(room.deck, summary)
    assert summary.documents_without_claims == ["deck.pdf"] and any("no claims" in n for n in summary.notes)


def test_a_flagged_document_silences_the_missing_items_check(session, room):  # noqa: F811
    refs = [DocReference(room.lease.id, room.rent_1, "Schedule A", "Annual rent is INR 1.2 crore.")]
    stats = MissingItemChecker(session, registry(session, room)).run(refs, not_fully_read=["temperature.xlsx"])
    assert stats.saved == 0 and "temperature.xlsx" in stats.skipped_reason


# ---- 2: references are looked up in the text of the room, and an item is reported once ------------------------------------------
def test_items_named_by_a_caption_a_heading_or_a_row_label_inside_a_document_are_present(session, room):  # noqa: F811
    add_doc(session, "report.pdf", "pdf", [dict(content="Table 3. Water quality by site\nsome rows follow", page_number=2),
                                          dict(content="Annex B - Laboratory methods\nmethods text", page_number=3)])
    add_doc(session, "register.xlsx", "xlsx", [dict(content="[Log!A2:C2] ID: DR-01; Date: 2026-10-08; Event: gear loss", sheet_name="Log", cell_reference="A2:C2", content_type="table_row")])
    quote = "Annual rent is INR 1.2 crore."
    items = ["Table 3", "Annex B", "DR-01", "Table 9"]
    stats = MissingItemChecker(session, registry(session, room)).run([DocReference(room.lease.id, room.rent_1, i, quote) for i in items])
    assert stats.present == 3 and stats.missing_items == ["Table 9"]


def test_a_sentence_that_only_mentions_an_item_does_not_prove_it_exists(session, room):  # noqa: F811
    add_doc(session, "notes.pdf", "pdf", [dict(content="Please see the Dive Safety Plan for the rules that apply.", page_number=1)])
    stats = MissingItemChecker(session, registry(session, room)).run([DocReference(room.lease.id, room.rent_1, "Dive Safety Plan", "Annual rent is INR 1.2 crore.")])
    assert stats.present == 0 and stats.saved == 1


def test_an_item_named_in_two_documents_is_one_finding_with_both_sentences(session, room):  # noqa: F811
    other, rows = add_doc(session, "other.pdf", "pdf", [dict(content="The MIMS Dive Safety Plan was not provided.", page_number=1)])
    refs = [DocReference(room.lease.id, room.rent_1, "Dive Safety Plan", "Annual rent is INR 1.2 crore."),
            DocReference(other.id, rows[0].id, "MIMS Dive Safety Plan", "The MIMS Dive Safety Plan was not provided.")]
    stats = MissingItemChecker(session, registry(session, room)).run(refs)
    finding = session.scalars(select(Finding)).one()
    assert stats.saved == 1 and stats.duplicates == 1 and "other.pdf" in finding.description
    assert len(session.scalars(select(Evidence).where(Evidence.finding_id == finding.id)).all()) == 2


# ---- 3: wording-only differences and limits --------------------------------------------------------------------------------------
def test_one_name_containing_the_other_is_wording_not_a_difference():
    assert one_contains_other("Technical Officer (instrumentation)", "Technical Officer")
    assert one_contains_other("MIMS Marine Operations", "MIMS")
    assert not one_contains_other("Technical Officer", "Technical Officer")  # equal, nothing to say
    assert not one_contains_other("Chief Scientist", "Technical Officer")
    assert not one_contains_other("12", "120")  # numbers are never contained in one another
    assert "wording only" in comparison_note("MIMS Marine Operations", None, "MIMS", None, "name", "name")


def test_a_number_inside_a_limit_is_compliant_and_one_outside_is_not():
    assert bound_direction("Maximum samples per site") == "upper" and bound_direction("Minimum crew") == "lower" and bound_direction("Samples collected") is None
    assert within_limit("8", None, "upper", "6", None) is True and within_limit("8", None, "upper", "9", None) is False
    assert within_limit("3", None, "lower", "4", None) is True and within_limit("many", None, "upper", "6", None) is None


def test_the_judge_is_told_when_a_value_is_inside_a_limit(session, room):  # noqa: F811
    da, ca = add_doc(session, "limits.pdf", "pdf", [dict(content="The permit allows a maximum of 8 samples per site.", page_number=1)])
    db, cb = add_doc(session, "log.pdf", "pdf", [dict(content="At most 6 samples were taken per site in practice.", page_number=1)])
    a = Claim(document_id=da.id, chunk_id=ca[0].id, subject="site", metric="maximum samples per site", value="8", claim_text="a maximum of 8 samples per site")
    b = Claim(document_id=db.id, chunk_id=cb[0].id, subject="site", metric="samples taken per site", value="6", claim_text="6 samples were taken per site")
    session.add_all([a, b])
    session.commit()
    prompt = PairJudge._prompt([pair(a, b)], {a.id: a, b.id: b}, {ca[0].id: ca[0], cb[0].id: cb[0]}, {da.id: "limits.pdf", db.id: "log.pdf"})
    assert "Code check (limit)" in prompt and "inside a limit is compliant" in prompt


def test_a_contradiction_between_two_names_where_one_contains_the_other_is_saved_as_consistent(session, room):  # noqa: F811
    da, ca = add_doc(session, "one.pdf", "pdf", [dict(content="The Technical Officer (instrumentation) joined the survey.", page_number=1)])
    db, cb = add_doc(session, "two.pdf", "pdf", [dict(content="The Technical Officer joined the survey in October.", page_number=1)])
    a = Claim(document_id=da.id, chunk_id=ca[0].id, subject="surveyor", metric="role", value="Technical Officer (instrumentation)", claim_category="name", claim_text="Technical Officer (instrumentation) joined")
    b = Claim(document_id=db.id, chunk_id=cb[0].id, subject="surveyor", metric="role", value="Technical Officer", claim_category="name", claim_text="Technical Officer joined")
    session.add_all([a, b])
    session.commit()
    p = pair(a, b)
    quotes = {a.id: "Technical Officer (instrumentation) joined the survey", b.id: "Technical Officer joined the survey in October"}
    verdict = dict(pair_index=0, same_attribute=True, classification="contradiction", severity="high", confidence=0.9, title="Role differs", description="d",
                   quote_a=quotes[p.claim_a_id], quote_b=quotes[p.claim_b_id])
    stats = PairJudge(session, MockLLMClient([{"verdicts": [verdict]}]), registry(session, room)).run([p])
    assert [f.classification for f in session.scalars(select(Finding))] == ["consistent"] and stats.reclassified_names == 1


# ---- 4: composite row subjects are asked for ---------------------------------------------------------------------------------------
def test_extraction_asks_for_the_subject_to_name_every_identifying_column():
    assert "names all of them" in SYSTEM_PROMPT and "names all of them" in SCHEMA["properties"]["claims"]["items"]["properties"]["subject"]["description"]


# ---- 5: identifier reconciliation ---------------------------------------------------------------------------------------------------
def lists(session):
    field, _ = add_doc(session, "field_register.xlsx", "xlsx", [dict(content=f"[Log!A{i}:B{i}] Sample: SPL-{i:03d}; Site: A", sheet_name="Log", cell_reference=f"A{i}:B{i}", content_type="table_row") for i in range(1, 9)])
    lab, lab_chunks = add_doc(session, "lab_report.pdf", "pdf", [dict(content="Sample SPL-003 was analysed.\nSample SPL-044-B was received without field metadata.", page_number=1)])
    return field, lab


def test_an_identifier_outside_the_range_of_another_documents_list_is_raised_with_both_quotes(session, room):  # noqa: F811
    lists(session)
    stats = IdentifierChecker(session, registry(session, room)).run()
    finding = session.scalars(select(Finding).where(Finding.title.like("SPL-044-B%"))).one()
    assert finding.classification == "missing_information" and stats.saved == 1 and stats.raised == ["SPL-044-B"]
    quotes = [e.quoted_text for e in session.scalars(select(Evidence).where(Evidence.finding_id == finding.id))]
    assert any("SPL-044-B" in q for q in quotes) and any("SPL-008" in q for q in quotes)  # the record and the end of the list


def test_an_identifier_inside_the_range_or_a_year_or_a_section_label_is_not_raised(session, room):  # noqa: F811
    add_doc(session, "field_register.xlsx", "xlsx", [dict(content=f"[Log!A{i}:B{i}] Sample: SPL-{i:03d}", sheet_name="Log", cell_reference=f"A{i}:B{i}", content_type="table_row") for i in range(1, 9)])
    add_doc(session, "lab_report.pdf", "pdf", [dict(content="Sample SPL-005 was analysed in 2026. See Table-12 and IR-2025 for details.", page_number=1)])
    stats = IdentifierChecker(session, registry(session, room)).run()
    assert stats.saved == 0 and stats.candidates == 0
    assert [m.ident for _k, m in mentions_in("Rev-02 and Fig-10 and PO-1032 and X-2025", "c")] == ["PO-1032"]


def test_identifier_reconciliation_is_silent_when_a_document_was_not_read(session, room):  # noqa: F811
    lists(session)
    stats = IdentifierChecker(session, registry(session, room)).run(not_fully_read=["temperature.xlsx"])
    assert stats.saved == 0 and "temperature.xlsx" in stats.skipped_reason


# ---- LlamaExtract refuses a system prompt of 10,000 characters or more (a 422 on every call): keep every prompt under it -----------------
def test_every_system_prompt_stays_under_the_llamaextract_limit():
    from app.services.llm_pairing import SYSTEM as GROUPING
    from app.services.pair_judge import SYSTEM as JUDGE
    assert max(len(JUDGE), len(GROUPING), len(SYSTEM_PROMPT)) < 10_000
