"""Single-claim missing information: an item a document refers to that nothing in the data room is named like."""
from sqlalchemy import select

from app.models import Evidence, Finding
from app.services.llama_claims import SCHEMA, SYSTEM_PROMPT, LlamaClaims
from app.services.missing_items import DocReference, MissingItemChecker, _item_keys
from app.tools import build_registry
from tests.test_tools import add_doc, room  # noqa: F401  (fixtures and helper)


def checker(session, room):
    return MissingItemChecker(session, build_registry(session, None, room.investigation.id))


def reference(room, item, quote="Annual rent is INR 1.2 crore."):
    return DocReference(document_id=room.lease.id, chunk_id=room.rent_1, item=item, quote=quote)


def test_an_item_that_is_nowhere_in_the_room_becomes_a_missing_information_finding_with_its_quote(session, room):  # noqa: F811
    stats = checker(session, room).run([reference(room, "Schedule A")])
    assert (stats.references, stats.present, stats.saved, stats.rejected) == (1, 0, 1, 0)
    finding = session.scalars(select(Finding)).one()
    assert finding.classification == "missing_information" and "Schedule A" in finding.title and finding.verification_status == "verified"
    evidence = session.scalars(select(Evidence).where(Evidence.finding_id == finding.id)).one()
    assert evidence.quoted_text == "Annual rent is INR 1.2 crore." and evidence.chunk_id == room.rent_1


def test_an_item_named_by_a_file_a_sheet_or_a_heading_is_present_and_a_different_letter_is_not(session, room):  # noqa: F811
    add_doc(session, "Schedule A - Prices.xlsx", "xlsx", [dict(content="x | 1", sheet_name="Prices", content_type="table_row")])
    add_doc(session, "annexes.docx", "docx", [dict(content="Schedule B", heading="Schedule B")])
    session.commit()
    stats = checker(session, room).run([reference(room, "Schedule A"), reference(room, "Schedule B (price list)"), reference(room, "Schedule C")])
    assert (stats.present, stats.saved) == (2, 1) and stats.missing_items == ["Schedule C"]  # a bracketed description does not hide a match


def test_the_referring_chunk_itself_never_proves_the_item_exists(session, room):  # noqa: F811
    stats = checker(session, room).run([reference(room, "Rent", quote="Annual rent is INR 1.2 crore.")])
    assert stats.present == 1  # the lease's other chunk is headed "Rent"
    stats = checker(session, room).run([DocReference(room.lease.id, room.term, "Term", "The lease term is 9 years.")])
    assert stats.present == 0 and stats.saved == 1  # only its own chunk was headed "Term"


def test_nothing_is_raised_unless_every_document_was_read(session, room):  # noqa: F811
    add_doc(session, "scan.pdf", "pdf", [dict(content="x", page_number=1)], status="skipped")
    session.commit()
    stats = checker(session, room).run([reference(room, "Schedule A")])
    assert stats.saved == 0 and "scan.pdf" in stats.skipped_reason
    assert session.scalars(select(Finding)).all() == []


def test_the_same_item_in_the_same_document_is_reported_once(session, room):  # noqa: F811
    stats = checker(session, room).run([reference(room, "Schedule A"), reference(room, "schedule a", quote="Payable quarterly in advance.")])
    assert (stats.saved, stats.duplicates) == (1, 1)


def test_extraction_asks_for_references_and_keeps_them(session, room):  # noqa: F811
    assert "references" in SCHEMA["properties"] and "references" in SYSTEM_PROMPT
    parsed = LlamaClaims.model_validate({"claims": [], "references": [{"item": "Annex 2", "quote": "see Annex 2 for prices"}]})
    assert parsed.references[0].item == "Annex 2"
    assert _item_keys("Schedule A (wafer specification)") == [frozenset({"schedule", "a", "wafer", "specification"}), frozenset({"schedule", "a"})]
