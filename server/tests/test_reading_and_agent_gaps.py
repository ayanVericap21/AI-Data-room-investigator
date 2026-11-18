"""The gaps closed in reading the documents and in the investigation agent: chunk overlap, PDF headings, coverage of unreadable documents, search minimum score,
duplicate and same-number findings, compare_values, stored calculations, claim links, full step logging, rejected findings, cost, and the investigation API."""
from __future__ import annotations

from types import SimpleNamespace

import pymupdf as fitz
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from app.agent import run_investigation
from app.api.dependencies import get_llm, get_optional_embedder, get_session_factory
from app.llm.mock import MockLLMClient
from app.main import create_app
from app.models import Claim, Document, Evidence, FindingClaim, Investigation, InvestigationEvent
from app.schemas.extraction import ExtractedUnit, Locator
from app.services.chunking import chunk_units, split_text
from app.services.extractors.pdf import extract_pdf
from app.tools import ToolBudget, build_registry
from tests.test_agent import call, finish
from tests.test_tools import add_doc, finding_args, registry, room  # noqa: F401  (fixtures)


def agent_registry(session, room, **kw):  # noqa: F811
    return build_registry(session, None, room.investigation.id, ToolBudget(), agent_checks=True, **kw)


# ---- chunk overlap ------------------------------------------------------------
def test_a_split_paragraph_repeats_the_end_of_each_part_at_the_start_of_the_next():
    text = " ".join(f"Sentence number {i} is here." for i in range(40))
    unit = ExtractedUnit(document_id="d", filename="f.pdf", source_type="pdf", locator=Locator(page=3), content=text, content_type="paragraph")
    drafts = chunk_units([unit], max_chars=200, overlap_chars=60)
    assert len(drafts) > 2 and all(len(d.content) <= 200 for d in drafts)
    for prev, cur in zip(drafts, drafts[1:]):
        last_sentence = prev.content.split(". ")[-1] if ". " in prev.content else prev.content
        assert last_sentence.rstrip(".") in cur.content[:80]  # the sentence cut at the boundary is whole at the start of the next part
    assert drafts[1].metadata["overlaps_previous_part"] and "overlaps_previous_part" not in drafts[0].metadata
    assert text.startswith(drafts[0].content) and drafts[-1].content.endswith(text[-20:])  # nothing lost


def test_a_split_table_repeats_whole_rows_and_zero_overlap_changes_nothing():
    rows = "\n".join(f"row{i} | {i}" for i in range(100))
    parts = split_text(rows, "table", 100, overlap_chars=20)
    assert all(len(p) <= 100 for p in parts) and parts[0].split("\n")[-1] in parts[1].split("\n")[:3]
    assert "\n".join(split_text(rows, "table", 100)) == rows


# ---- PDF headings -------------------------------------------------------------
def test_a_pdf_without_an_outline_gets_headings_from_font_size_and_carries_them_to_later_pages(tmp_path):
    path = tmp_path / "a.pdf"
    d = fitz.open()
    p = d.new_page()
    p.insert_text((72, 80), "Operations Overview", fontsize=22)
    p.insert_text((72, 120), "The plant produced 120,000 tonnes in 2024 and ran at full capacity all year.", fontsize=10)
    d.new_page().insert_text((72, 100), "Maintenance was carried out in the third quarter of the year.", fontsize=10)
    d.save(path)
    units = extract_pdf(path, "d", "a.pdf")
    assert [(u.content_type, u.content) for u in units if u.content_type == "heading"] == [("heading", "Operations Overview")]
    body = [u for u in units if u.content_type == "paragraph"]
    assert {u.locator.heading for u in body} == {"Operations Overview"} and [u.locator.page for u in body] == [1, 2]


def test_a_pdf_with_an_outline_uses_it_for_every_page(tmp_path):
    path = tmp_path / "b.pdf"
    d = fitz.open()
    for text in ("First page text about revenue of 5 million.", "Second page text about costs of 2 million.", "Third page text about staff of 40 people."):
        d.new_page().insert_text((72, 100), text, fontsize=11)
    d.set_toc([[1, "Revenue", 1], [1, "Costs", 2]])
    d.save(path)
    units = extract_pdf(path, "d", "b.pdf")
    assert [(u.locator.page, u.locator.heading) for u in units] == [(1, "Revenue"), (2, "Costs"), (3, "Costs")]


# ---- coverage: unreadable documents -------------------------------------------
def test_a_document_with_no_chunks_is_reported_as_unreadable_and_the_run_is_not_complete(session, room):  # noqa: F811
    add_doc(session, "scan.pdf", "pdf", [], status="failed")
    session.commit()
    cov = build_registry(session, None, room.investigation.id).coverage()
    assert cov.unreadable == ["scan.pdf"] and cov.total == 3 and "scan.pdf" not in cov.not_opened
    llm = MockLLMClient([call("list_documents"), finish()])
    result = run_investigation(session, llm, None, name="gaps", max_finish_refusals=0)
    assert result.documents_unreadable == ["scan.pdf"]
    investigation = session.get(Investigation, result.investigation_id)
    assert investigation.status == "partially_completed" and investigation.execution_config["coverage"]["unreadable"] == ["scan.pdf"]
    assert "scan.pdf" in next(s for s in result.steps if s.action == "finish").prompt  # the model is told which documents it cannot open


def test_finishing_with_every_document_opened_is_complete(session, room):  # noqa: F811
    session.add_all([])
    ids = [room.deck.id, room.capex.id, room.lease.id]
    llm = MockLLMClient([*[call("get_outline", {"document_id": i}) for i in ids], finish()])
    result = run_investigation(session, llm, None, name="all", max_finish_refusals=0)
    assert session.get(Investigation, result.investigation_id).status == "completed"


# ---- search minimum score -----------------------------------------------------
class RecordingSearch:
    def __init__(self):
        self.min_scores = []

    def search(self, query, top_k=5, document_ids=None, exclude_chunk_ids=None, exclude_document_ids=None, min_score=None):
        self.min_scores.append(min_score)
        return []


def test_search_uses_the_run_default_minimum_score_and_a_search_can_set_its_own(session, room):  # noqa: F811
    backend = RecordingSearch()
    reg = build_registry(session, backend, room.investigation.id, search_min_score=0.4)
    assert reg.execute("search", {"query": "capacity"}).ok and reg.execute("search", {"query": "capacity", "min_score": 0.7}).ok
    assert backend.min_scores == [0.4, 0.7]
    assert not reg.execute("search", {"query": "x", "min_score": 1.5}).ok


# ---- duplicate findings and same-number contradictions ------------------------
def test_the_agent_cannot_submit_the_same_finding_twice(session, room):  # noqa: F811
    reg = agent_registry(session, room)
    assert reg.execute("submit_finding", finding_args(room)).ok
    again = reg.execute("submit_finding", finding_args(room, title="worded differently"))
    assert not again.ok and "repeats" in again.error.message
    other_class = reg.execute("submit_finding", finding_args(room, classification="contextual_difference", title="same passages, other class"))
    assert other_class.ok  # a different classification of the same passages is a different finding
    assert build_registry(session, None, room.investigation.id).execute("submit_finding", finding_args(room, title="judge path")).ok  # the judge flow is unchanged


def test_a_contradiction_whose_quotes_state_the_same_numbers_is_refused_for_the_agent(session, room):  # noqa: F811
    same = finding_args(room, evidence=[
        dict(chunk_id=room.deck_p1, quoted_text="capacity is 120,000 tonnes per year", role="contradicting", relevance_reason="x", support_assessment="y"),
        dict(chunk_id=room.capex_expansion, quoted_text="Capacity after phase 2 | 150,000 tonnes (2027)", role="contradicting", relevance_reason="x", support_assessment="y")])
    assert agent_registry(session, room).execute("submit_finding", same).ok  # different numbers: accepted
    _twin, twin_rows = add_doc(session, "twin.pdf", "pdf", [dict(content="Annual production capacity is 120,000 tonnes per year.", page_number=1)])
    session.commit()
    twin_chunk = twin_rows[0].id
    refused = agent_registry(session, room).execute("submit_finding", finding_args(room, evidence=[
        dict(chunk_id=room.deck_p1, quoted_text="capacity is 120,000 tonnes per year", role="contradicting", relevance_reason="x", support_assessment="y"),
        dict(chunk_id=twin_chunk, quoted_text="capacity is 120,000 tonnes per year", role="contradicting", relevance_reason="x", support_assessment="y")]))
    assert not refused.ok and "same numbers" in refused.error.message and "compare_values" in refused.error.message


# ---- compare_values -----------------------------------------------------------
def test_compare_values_normalises_scale_units_and_dates(registry):  # noqa: F811
    assert registry.execute("compare_values", {"a": "120,000", "b": "120 thousand"}).output.relation == "equal"
    assert registry.execute("compare_values", {"a": "2", "unit_a": "tonnes", "b": "2000", "unit_b": "kg"}).output.relation == "equal"
    assert registry.execute("compare_values", {"a": "100,000", "b": "150,000"}).output.relation == "different"
    assert registry.execute("compare_values", {"a": "2025-03-01", "b": "1 March 2025", "value_type": "date"}).output.relation == "equal"


# ---- calculations and claim links are stored with the finding -----------------
def test_a_saved_finding_keeps_its_calculations_and_links_the_claims_its_quotes_state(session, room):  # noqa: F811
    claim = Claim(document_id=room.deck.id, chunk_id=room.deck_p1, subject="plant", metric="capacity", value="120,000", unit="tonnes", claim_text="capacity is 120,000 tonnes per year",
                  claim_category="quantity")
    session.add(claim)
    session.commit()
    reg = agent_registry(session, room)
    assert reg.execute("calculate", {"expression": "120000 - 100000", "label": "gap"}).ok
    saved = reg.execute("submit_finding", finding_args(room))
    assert saved.ok and saved.output.evidence[0].claim_ids == [claim.id]
    event = session.scalars(select(InvestigationEvent).where(InvestigationEvent.event_type == "finding_saved")).one()
    assert event.details["calculations"] == [{"expression": "120000 - 100000", "result": "20000", "label": "gap"}] and event.details["claim_ids"] == [claim.id]
    link = session.scalars(select(FindingClaim)).one()
    assert link.claim_id == claim.id and link.relationship_type == "contradicts"
    assert session.scalars(select(Evidence).where(Evidence.chunk_id == room.deck_p1)).one().claim_id == claim.id
    assert reg.context.calculations == []  # the next finding starts clean


# ---- full step logging, rejected findings, cost -------------------------------
def test_every_step_stores_what_the_model_saw_answered_and_got_back_and_a_rejected_finding_keeps_its_payload(session, room):  # noqa: F811
    bad = finding_args(room)
    bad["evidence"][0]["quoted_text"] = "capacity is 999 tonnes per year"
    llm = MockLLMClient([call("list_documents"), call("submit_finding", bad), finish()])
    result = run_investigation(session, llm, None, name="log", max_finish_refusals=0, prices_per_mtok=(1.0, 2.0))
    events = session.scalars(select(InvestigationEvent).where(InvestigationEvent.investigation_id == result.investigation_id)).all()
    first = next(e for e in events if e.event_type == "agent_step" and e.message.startswith("Step 1:"))
    assert "## Task" in first.details["prompt"] and first.details["action"]["tool"] == "list_documents" and first.details["output"]["documents"]
    assert first.details["llm_attempts"] and "prompt_tokens" in first.details["llm_attempts"][0] and first.details["prompt_notes"] == {"shortened_steps": [], "truncated": []}
    rejected = next(e for e in events if e.event_type == "finding_rejected")
    assert rejected.details["submitted"]["evidence"][0]["quoted_text"] == "capacity is 999 tonnes per year" and "not in chunk" in rejected.details["error_message"]
    investigation = session.get(Investigation, result.investigation_id)
    expected = (result.token_usage.prompt_tokens * 1.0 + result.token_usage.completion_tokens * 2.0) / 1_000_000
    assert float(investigation.estimated_cost) == pytest.approx(expected) and investigation.execution_config["cost"]["note"] is None
    assert "prompt" not in result.steps[0].model_dump()  # the large fields stay out of the run result


def test_what_was_cut_from_the_prompt_is_recorded(session, room):  # noqa: F811
    llm = MockLLMClient([call("list_documents")] * 5 + [finish()])
    result = run_investigation(session, llm, None, name="cut", max_finish_refusals=0)
    notes = [s.prompt_notes for s in result.steps]
    assert notes[3]["shortened_steps"] == [] and notes[4]["shortened_steps"] == [1] and notes[5]["shortened_steps"] == [1, 2]


def test_without_prices_the_cost_is_not_calculated_and_the_run_says_so(session, room):  # noqa: F811
    result = run_investigation(session, MockLLMClient([finish()]), None, name="nocost", max_finish_refusals=0)
    cost = session.get(Investigation, result.investigation_id).execution_config["cost"]
    assert cost["estimated"] is None and "no token prices" in cost["note"]


# ---- the investigation API ----------------------------------------------------
def test_an_investigation_is_started_followed_and_stopped_through_the_api(session_factory, room):  # noqa: F811
    llm = MockLLMClient([call("list_documents"), finish("done"), finish("done"), finish("done")])  # the first two finishes are refused while documents are unopened
    app = create_app()
    app.dependency_overrides[get_llm] = lambda: llm
    app.dependency_overrides[get_optional_embedder] = lambda: None
    app.dependency_overrides[get_session_factory] = lambda: session_factory
    from app.api.dependencies import get_job_manager
    from app.workers.jobs import JobManager
    jobs = JobManager()
    app.dependency_overrides[get_job_manager] = lambda: jobs
    client = TestClient(app)
    assert client.get("/api/investigations/status").json()["state"] == "idle"
    started = client.post("/api/investigations/start", json={"focus": "capacity", "max_steps": 5})
    assert started.status_code == 202
    jobs.wait("investigation", 30)
    status = client.get("/api/investigations/status").json()
    assert status["state"] == "completed" and status["result"]["termination"] == "finished" and status["result"]["investigation_id"]
    assert client.post("/api/investigations/stop").status_code == 200
