"""A limit a line gives is not a record; claims about different records are never grouped."""
from __future__ import annotations

from sqlalchemy import select

from app.models import Finding
from app.schemas.claim import ClaimView
from app.services.claim_matching import different_records, comparable_subjects
from app.services.identifiers import ceiling_idents, entity_ids
from app.services.record_checks import IdentifierChecker, mentions_in
from app.tools import build_registry
from tests.test_tools import add_doc, room  # noqa: F401  (fixture and helper)

PERMIT = "1. Artifact Ceiling: authorized to recover up to 60 diagnostic artifacts (sequential identifiers ART-101 through a maximum of ART-160) and up to 10 primary organic samples (RC-01 through RC-10) for AMS radiocarbon dating."


def test_identifiers_a_line_gives_as_a_limit_are_found_but_not_read_as_records():
    assert ceiling_idents(PERMIT) == {"ART-101", "ART-160", "RC-01", "RC-10"}
    assert ceiling_idents("Samples RC-01 to RC-08 were collected.") == set()  # no limit words: a range of records
    assert ceiling_idents("No more than RC-12 may be reported.") == {"RC-12"}
    assert mentions_in(PERMIT, "c") == []
    assert [m.ident for _k, m in mentions_in("Sample RC-12-B arrived without field metadata. Up to 10 samples (RC-01 through RC-10).", "c")] == ["RC-12-B"]


def test_a_permit_ceiling_is_not_raised_as_a_missing_record_but_a_real_missing_record_still_is(session, room):  # noqa: F811
    add_doc(session, "log.xlsx", "xlsx", [dict(content=f"[Log!A{i}:B{i}] Sample: RC-{i:02d}", sheet_name="Log", cell_reference=f"A{i}:B{i}", content_type="table_row") for i in range(1, 9)])
    add_doc(session, "permit.docx", "docx", [dict(content=PERMIT, heading="Conditions")])
    add_doc(session, "lab.pdf", "pdf", [dict(content="Sample RC-12-B arrived without field metadata.", page_number=1)])
    stats = IdentifierChecker(session, build_registry(session, None, room.investigation.id)).run()
    assert stats.raised == ["RC-12-B"] and stats.saved == 1
    assert not list(session.scalars(select(Finding).where(Finding.title.like("%RC-10%"))))


def view(id, subject, doc="d1"):
    return ClaimView(id=id, document_id=doc, chunk_id=f"c{id}", subject=subject, metric="calibrated age", value="1", claim_text=f"claim {id}")


def test_a_subject_names_its_records_by_letters_and_number_ignoring_the_suffix_and_years():
    assert entity_ids("Trench A Stratum I, Sample RC-02") == {("RC", 2)} and entity_ids("RC-12-B and RC-12") == {("RC", 12)}
    assert entity_ids("Dive on 2026-05-14 (Table-12, X-2025)") == set() and entity_ids(None) == set()


def test_claims_about_different_records_never_candidate_and_unlabelled_or_shared_records_still_do():
    assert different_records(view("a", "Trench A Stratum I, Sample RC-02"), view("b", "RC-04"))
    assert not comparable_subjects(view("a", "Trench A Stratum I, Sample RC-02"), view("b", "RC-03", "d2"))
    assert comparable_subjects(view("a", "Trench A Stratum I, Sample RC-02"), view("b", "RC-02", "d2"))  # the same record
    assert comparable_subjects(view("a", "RC-12-B"), view("b", "RC-12", "d2"))  # a sub-sample of the same record: the judge decides
    assert comparable_subjects(view("a", "Sample RC-02"), view("b", "Trench A radiocarbon results", "d2"))  # one side names no record
    assert comparable_subjects(view("a", "EX-04, RC-07"), view("b", "RC-07", "d2"))  # they share a record
