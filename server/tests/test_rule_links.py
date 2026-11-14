"""Rules and the records under them: the model's links, the dated windows code adds, the shared-record route and how a rule group is judged."""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import select

from app.llm.mock import MockLLMClient
from app.models import Claim, Finding
from app.schemas.claim import ClaimView
from app.services.claim_group_judge import RULE_SYSTEM, ClaimGroupJudge
from app.services.claim_grouping import ClaimGrouper
from app.services.rule_links import RuleLinker
from app.services.temporal_links import parse_dates, windows_and_records
from tests.test_claim_group_judge import answer, candidate, judge, verdict, warehouse_claims
from tests.test_tools import add_doc, room  # noqa: F401  (fixtures and helper)


def view(id, doc="d1", chunk=None, subject=None, metric=None, value=None, period=None, text="", kind="statement"):
    return ClaimView(id=id, document_id=doc, chunk_id=chunk or f"chunk-{id}", subject=subject, metric=metric, value=value, unit=None, period=period, conditions=None,
                     assumptions=[], claim_text=text, value_type=kind, table_key=None)


# ---- dates read by code ---------------------------------------------------------------------------------------------------------------------
def test_dates_are_read_in_iso_and_written_forms_with_a_time_and_a_default_year():
    assert parse_dates("from May 13 at 15:30 until May 15 at 07:30", 2026) == [datetime(2026, 5, 13, 15, 30), datetime(2026, 5, 15, 7, 30)]
    assert parse_dates("2026-05-14 and 13 June 2025 and May 13, 2026", None) == [datetime(2026, 5, 14), datetime(2025, 6, 13), datetime(2026, 5, 13)]
    assert parse_dates("May 13", None) == []  # no year anywhere: not used


def test_a_claim_dated_inside_a_restriction_window_is_a_record_under_it_and_one_outside_is_not():
    storm = view("storm", chunk="c-storm", subject="All site operations", metric="status", value="suspended", period="from May 13 at 15:30 until May 15 at 07:30")
    other = view("year", chunk="c-storm", subject="the storm", metric="date", value="May 13, 2026")  # gives the year, same place as the window
    inside = view("row", doc="d2", chunk="c-log", subject="2026-05-14, EX-04", metric="Max_Depth_cm", value="240", period="2026-05-14")
    outside = view("later", doc="d2", chunk="c-log2", subject="2026-05-18, EX-05", metric="Max_Depth_cm", value="290", period="2026-05-18")
    validity = view("permit", doc="d2", chunk="c-permit", subject="Permit", metric="validity", value="valid", period="from 2026-04-14 to 2026-06-30")  # not a restriction
    found = windows_and_records([storm, other, inside, outside, validity])
    assert [(c.claim_id, c.candidate_id, c.matched_by) for c in found] == [("storm", "row", "rule")]


# ---- the model's links ----------------------------------------------------------------------------------------------------------------------
def test_the_model_links_a_rule_to_its_records_and_unknown_refs_are_ignored(session, room):  # noqa: F811
    rule, rec_a, rec_b = warehouse_claims(session, ["no storage allowed", 500, 450])
    llm = MockLLMClient([{"links": [{"rule": "c1", "records": ["c2", "c3", "c9", "c1"], "topic": "storage ban"}, {"rule": "c8", "records": ["c1"]}]}])
    found, stats = RuleLinker(session, llm).run()
    assert stats.links == 1 and stats.unknown_refs == 2 and len(found) == 2 and all(c.matched_by == "rule" for c in found)
    assert {c.candidate_id for c in found} == {rec_a.id, rec_b.id} and len({c.claim_id for c in found}) == 1


# ---- the shared-record route ----------------------------------------------------------------------------------------------------------------
def test_dates_that_name_the_same_record_are_candidates_whatever_their_labels_are():
    a = view("sum", chunk="c1", subject="Trench B Stratum II", metric="dating", value="approximately 850-750 cal BCE", text="Sample RC-04 dates the layer to 850-750 cal BCE", kind="date")
    b = view("lab", doc="d2", chunk="c2", subject="RC-04 (Trench B, Stratum II)", metric="Calibrated", value="1685-1535 cal BCE", text="RC-04: 1685-1535 cal BCE", kind="date")
    c = view("other", doc="d2", chunk="c3", subject="RC-02", metric="Calibrated", value="1885-1740 cal BCE", text="RC-02: 1885-1740 cal BCE", kind="date")
    found: dict = {}
    ClaimGrouper(None, None)._add_shared_identifiers([a, b, c], found)
    assert list(found) == [("lab", "sum")] and found[("lab", "sum")].matched_by == "identifier"


# ---- how a rule group is judged -------------------------------------------------------------------------------------------------------------
def rule_group(session):
    rule, breaking, fine = warehouse_claims(session, ["no storage allowed", 500, 450])
    return rule, breaking, fine, [candidate(rule, breaking, how="rule"), candidate(rule, fine, how="rule")]


def test_a_rule_and_its_records_are_a_group_of_their_own_with_the_rule_first():
    from tests.test_claim_group_judge import Claim as _C  # noqa: F401
    cands = [candidate(type("X", (), {"id": "r"}), type("Y", (), {"id": "a"}), how="rule"), candidate(type("X", (), {"id": "r"}), type("Y", (), {"id": "b"}), how="rule"),
             candidate(type("X", (), {"id": "q"}), type("Y", (), {"id": "a"}), how="rule")]
    groups = ClaimGroupJudge._form_groups(cands)
    assert [(g.kind, g.claim_ids) for g in groups] == [("rule", ["r", "a", "b"]), ("rule", ["q", "a"])]  # two rules sharing a record are two groups


def test_a_rule_group_is_judged_with_the_rules_prompt_and_a_break_becomes_a_finding(session, room):  # noqa: F811
    rule, breaking, fine, cands = rule_group(session)
    llm = MockLLMClient([answer(verdict([rule, breaking, fine], aside={3}, title="A breaks the rule", description="The rule bans storage; A stores 500."))])
    stats = judge(session, room, llm).run(cands)
    finding = session.scalars(select(Finding)).one()
    assert "RULE GROUP" in llm.prompts[0] and "Code checks:" not in llm.prompts[0]
    assert finding.classification == "contradiction" and stats.groups_series == 0 and stats.groups_agreed == 0


def test_the_rules_prompt_is_under_the_limit_and_says_how_a_break_is_found():
    assert len(RULE_SYSTEM) < 10_000
    for must in ("BREAKS the rule", "contradiction", "set_aside", "evidence:", "ONE verdict per group"):
        assert must in RULE_SYSTEM, must
