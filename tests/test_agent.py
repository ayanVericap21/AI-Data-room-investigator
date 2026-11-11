import pytest
from fastapi.testclient import TestClient

from app import jobs
from app.agent import service, store
from app.agent.runner import InvestigationRunner
from app.agent.schemas import (BaselineOut, CitationOut, ClaimOut, ClaimsOut, ConclusionOut, HypothesesOut,
                               HypothesisOut, StepOut, SupportOut)
from app.agent.verify import check_citation_rule, final_status, heuristic_candidates
from app.api.main import create_app
from app.baseline import run_baseline
from app.documents import save_upload
from app.llm import LLMError, UsageTracker
from app.mcp_server.client import LocalToolClient
from app.models import Citation, Claim, Classification as C, DocSummary
from app.tools import DataRoomTools

SUP = "The supplier guarantees only 25,000 tonnes per year."
MOD = "Revenue assumes 30,000 tonnes of annual production."


class FakeLLM(UsageTracker):
    """Scripted per-purpose responses (consumed in order). A callable gets the prompt and returns the model."""

    def __init__(self, script):
        super().__init__()
        self.script = {k: list(v) for k, v in script.items()}
        self.prompts = []   # (purpose, full prompt text = prefix blocks + user suffix)
        self.calls = []     # (purpose, prefix blocks, user suffix) - to assert on the caching layout

    def structured(self, *, system, user, schema, purpose, prefix=None):
        full = "\n\n".join([*(b.text for b in (prefix or [])), user])
        self.prompts.append((purpose, full))
        self.calls.append((purpose, list(prefix or []), user))
        user = full
        q = self.script.get(purpose)
        if not q:
            raise LLMError(f"no scripted response for {purpose}")
        item = q.pop(0) if len(q) > 1 else q[0]
        if isinstance(item, Exception):
            raise item
        item = item(user) if callable(item) else item
        self.record(purpose, 100, 50, 0.01)
        return item


def cid(db, frag):
    return db.one("SELECT id FROM chunks WHERE text LIKE ?", (f"%{frag}%",))["id"]


@pytest.fixture
def room(db, settings):
    a, _ = save_upload(db, settings, "supplier.txt", SUP.encode())
    b, _ = save_upload(db, settings, "model.txt", MOD.encode())
    j, _ = jobs.submit_job(db, [a.id, b.id], "k")
    jobs.run_job(db, settings, j.id)
    return [a.id, b.id]


def claims_out(db):
    s, m = cid(db, "supplier"), cid(db, "Revenue")
    return ClaimsOut(claims=[
        ClaimOut(chunk_id=s, subject="annual production capacity", value="25,000", unit="tonnes", quote="guarantees only 25,000 tonnes"),
        ClaimOut(chunk_id=m, subject="annual production capacity", value="30,000", unit="tonnes", quote="assumes 30,000 tonnes of annual production"),
        ClaimOut(chunk_id=m, subject="annual production capacity", value="99,999", unit="tonnes", quote="THIS WAS NEVER WRITTEN IN THE DOC"),  # fabricated
    ])


def conclusion(db, cls=C.CONTRADICTION, cites=None, conf=0.9):
    cites = cites if cites is not None else [
        CitationOut(chunk_id=cid(db, "supplier"), quote="guarantees only 25,000 tonnes"),
        CitationOut(chunk_id=cid(db, "Revenue"), quote="assumes 30,000 tonnes of annual production")]
    return ConclusionOut(classification=cls, title="Capacity mismatch", explanation="25k vs 30k", citations=cites, confidence=conf)


def step(action="conclude", belief=C.CONTRADICTION, concl=None, **kw):
    return StepOut(belief=belief, reasoning="r", action=action, conclusion=concl, **kw)


def run(db, settings, room, script, max_steps=5):
    settings.max_steps_per_hypothesis = max_steps
    inv = service.start_investigation(db, room)
    llm = FakeLLM(script)
    InvestigationRunner(db, settings, llm, LocalToolClient(DataRoomTools(db, room)), inv.id).run()
    return inv.id, llm


def base_script(db, steps, support=SupportOut(supported=True)):
    return {"extract_claims": [lambda u: claims_out(db)],
            "hypothesize": [HypothesesOut(hypotheses=[HypothesisOut(claim_ids=[1, 2], question="Capacity conflict?", rationale="same subject")])],
            "investigate_step": steps, "support_check": [support]}


# ---------- deterministic rules ----------
def _cit(i, ok=True):
    return Citation(chunk_id=i, quote="q", verified=ok)


def test_citation_rule():
    assert check_citation_rule(C.CONTRADICTION, [_cit(1), _cit(2)])[0] == C.CONTRADICTION
    cls, notes = check_citation_rule(C.CONTRADICTION, [_cit(1), _cit(1)])  # same passage twice
    assert cls == C.INSUFFICIENT_EVIDENCE and any("downgraded" in n for n in notes)
    assert check_citation_rule(C.CONTRADICTION, [_cit(1), _cit(2, ok=False)])[0] == C.INSUFFICIENT_EVIDENCE
    assert check_citation_rule(C.INSUFFICIENT_EVIDENCE, [])[0] == C.INSUFFICIENT_EVIDENCE


def test_final_status():
    assert final_status(C.CONTRADICTION, 0.9, 0.5, False) == "established"
    assert final_status(C.CONTRADICTION, 0.3, 0.5, False) == "unresolved"
    assert final_status(C.CONTRADICTION, 0.9, 0.5, True) == "unresolved"
    assert final_status(C.INSUFFICIENT_EVIDENCE, 0.9, 0.5, False) == "unresolved"


def test_heuristic_candidates():
    mk = lambda i, d, subj, v: Claim(id=i, investigation_id=1, document_id=d, chunk_id=i, subject=subj, value=v, quote="q")
    cl = [mk(1, 1, "annual production capacity", "25"), mk(2, 2, "production capacity annual", "30"),
          mk(3, 3, "annual production capacity", "25"), mk(4, 2, "office rent", "5"), mk(5, 2, "annual production capacity", "40")]
    pairs = heuristic_candidates(cl)
    assert [1, 2] in pairs and [1, 3] not in pairs          # identical values are not candidates
    assert not any(4 in p for p in pairs)                    # unrelated subject
    assert [2, 5] not in pairs                               # same document


# ---------- agent runs ----------
def test_happy_path_search_then_conclude(db, settings, room):
    s = [step("search", belief=C.CONTRADICTION, query="tonnes"), step("conclude", concl=conclusion(db))]
    inv_id, llm = run(db, settings, room, base_script(db, s))
    inv = store.get(db, inv_id)
    assert inv.status == "completed" and inv.phase == "done"
    assert len(store.claims(db, inv_id)) == 2                  # fabricated claim was dropped
    f = store.findings(db, inv_id)[0]
    assert f.classification == C.CONTRADICTION and f.status == "established"
    assert all(c.verified and c.filename for c in f.citations) and {c.filename for c in f.citations} == {"supplier.txt", "model.txt"}
    kinds = [e.kind for e in store.events(db, inv_id)]
    assert "tool_call" in kinds and "finding" in kinds and "claims_extracted" in kinds
    assert inv.usage["calls"] == llm.usage.calls and inv.usage["cost_usd"] >= 0


def test_fabricated_citation_is_downgraded(db, settings, room):
    bad = conclusion(db, cites=[CitationOut(chunk_id=cid(db, "supplier"), quote="guarantees only 25,000 tonnes"),
                                CitationOut(chunk_id=cid(db, "Revenue"), quote="assumes 50,000 tonnes of production")])
    inv_id, _ = run(db, settings, room, base_script(db, [step("conclude", concl=bad)]))
    f = store.findings(db, inv_id)[0]
    assert f.classification == C.INSUFFICIENT_EVIDENCE and f.status == "unresolved"
    assert any(not c.verified for c in f.citations) and any("could not be verified" in n for n in f.verification_notes)
    assert any("originally concluded" in u for u in f.uncertainties)


def test_support_check_failure_downgrades(db, settings, room):
    sup = SupportOut(supported=False, issues=["quotes describe different scopes"])
    inv_id, _ = run(db, settings, room, base_script(db, [step("conclude", concl=conclusion(db))], support=sup))
    f = store.findings(db, inv_id)[0]
    assert f.classification == C.INSUFFICIENT_EVIDENCE and f.status == "unresolved"
    assert any("support check failed" in n for n in f.verification_notes)


def test_low_confidence_not_established(db, settings, room):
    inv_id, _ = run(db, settings, room, base_script(db, [step("conclude", concl=conclusion(db, conf=0.2))]))
    assert store.findings(db, inv_id)[0].status == "unresolved"


def test_step_limit_forces_insufficient_evidence(db, settings, room):
    loop = step("search", query="anything")  # model never concludes
    inv_id, llm = run(db, settings, room, base_script(db, [loop]), max_steps=2)
    f = store.findings(db, inv_id)[0]
    assert f.classification == C.INSUFFICIENT_EVIDENCE and f.status == "unresolved"
    assert "step limit" in f.explanation.lower()
    assert sum(1 for p, _ in llm.prompts if p == "investigate_step") == 3  # 2 tool steps + the forced one
    assert any(e.kind == "step_limit" for e in store.events(db, inv_id))
    # duplicate searches are skipped, not re-executed
    assert sum(1 for e in store.events(db, inv_id) if e.kind == "tool_call" and e.data.get("duplicate")) == 1


def test_belief_revision_is_recorded(db, settings, room):
    s = [step("search", belief=C.CONTRADICTION, query="tonnes"),
         step("conclude", belief=C.CONTEXTUAL_DIFFERENCE, concl=conclusion(db, cls=C.CONTEXTUAL_DIFFERENCE))]
    inv_id, _ = run(db, settings, room, base_script(db, s))
    ev = [e for e in store.events(db, inv_id) if e.kind == "belief_changed"]
    assert ev and "contradiction -> contextual_difference" in ev[0].message
    assert store.findings(db, inv_id)[0].classification == C.CONTEXTUAL_DIFFERENCE


def test_llm_failure_gives_partial_with_unresolved_finding(db, settings, room):
    inv_id, _ = run(db, settings, room, base_script(db, [LLMError("boom")]))
    assert store.get(db, inv_id).status == "partial"
    f = store.findings(db, inv_id)[0]
    assert f.status == "unresolved" and "failed" in f.explanation
    # resume re-opens the failed hypothesis
    service.prepare_resume(db, inv_id)
    assert store.hypotheses(db, inv_id)[0].status == "pending" and store.get(db, inv_id).phase == "investigate"


def test_extraction_failure_everywhere_fails_investigation(db, settings, room):
    inv_id, _ = run(db, settings, room, {"extract_claims": [LLMError("down")]})
    inv = store.get(db, inv_id)
    assert inv.status == "failed" and "extraction" in inv.error


def test_hypothesis_generation_failure_falls_back_to_heuristics(db, settings, room):
    script = base_script(db, [step("conclude", concl=conclusion(db))])
    script["hypothesize"] = [LLMError("down")]
    inv_id, _ = run(db, settings, room, script)
    h = store.hypotheses(db, inv_id)
    assert len(h) == 1 and h[0].rationale.startswith("heuristic")
    assert store.get(db, inv_id).status == "completed"


def test_invalid_hypothesis_ids_and_dedupe_are_filtered(db, settings, room):
    script = base_script(db, [step("conclude", concl=conclusion(db))])
    script["hypothesize"] = [HypothesesOut(hypotheses=[
        HypothesisOut(claim_ids=[1, 999], question="bad", rationale=""),   # only one valid id -> dropped
        HypothesisOut(claim_ids=[1, 2], question="ok", rationale=""),
        HypothesisOut(claim_ids=[2, 1], question="dup", rationale="")])]
    inv_id, _ = run(db, settings, room, script)
    assert [h.question for h in store.hypotheses(db, inv_id)] == ["ok"]


def test_followup_with_clarification_reinvestigates_and_supersedes(db, settings, room):
    inv_id, _ = run(db, settings, room, base_script(db, [step("conclude", concl=conclusion(db))]))
    f1 = store.findings(db, inv_id)[0]
    hid = service.request_followup(db, inv_id, finding_id=f1.id, note="25k is the guaranteed floor, 30k is nameplate")
    assert store.get(db, inv_id).status == "queued"
    llm = FakeLLM({"investigate_step": [step("conclude", belief=C.CONTEXTUAL_DIFFERENCE, concl=conclusion(db, cls=C.CONTEXTUAL_DIFFERENCE))],
                   "support_check": [SupportOut(supported=True)]})
    InvestigationRunner(db, settings, llm, LocalToolClient(DataRoomTools(db, room)), inv_id).run()
    assert any("guaranteed floor" in u for p, u in llm.prompts if p == "investigate_step")  # clarification reached the agent
    live = store.findings(db, inv_id)
    assert [f.classification for f in live] == [C.CONTEXTUAL_DIFFERENCE]
    assert len(store.findings(db, inv_id, include_superseded=True)) == 2   # history preserved
    assert store.hypotheses(db, inv_id)[0].id == hid
    assert any(e.kind == "user_feedback" for e in store.events(db, inv_id))


def test_followup_new_question_and_validation(db, settings, room):
    inv_id, _ = run(db, settings, room, base_script(db, [step("conclude", concl=conclusion(db))]))
    hid = service.request_followup(db, inv_id, question="Is there a permit for the plant?")
    assert store.get_hypothesis(db, hid).claim_ids == []
    with pytest.raises(service.InvestigationError):
        service.request_followup(db, inv_id)
    with pytest.raises(service.InvestigationError):
        service.request_followup(db, inv_id, hypothesis_id=9999, note="x")
    with pytest.raises(service.InvestigationError):
        service.request_followup(db, inv_id, finding_id=9999)


def test_start_validation(db, settings):
    with pytest.raises(service.InvestigationError):
        service.start_investigation(db, None)  # nothing ingested
    d, _ = save_upload(db, settings, "a.txt", b"not processed yet at all")
    with pytest.raises(service.InvestigationError):
        service.start_investigation(db, [d.id])


def test_interrupted_run_is_resumable(db, settings, room):
    inv = service.start_investigation(db, room)
    store.update(db, inv.id, status="running")
    assert service.recover_interrupted(db) == 1
    assert store.get(db, inv.id).status == "interrupted"
    service.prepare_resume(db, inv.id)
    assert store.get(db, inv.id).status == "queued"


def test_no_claims_means_completed_without_findings(db, settings, room):
    script = {"extract_claims": [ClaimsOut(claims=[])]}
    inv_id, _ = run(db, settings, room, script)
    assert store.get(db, inv_id).status == "completed" and store.findings(db, inv_id) == []


# ---------- baseline ----------
def test_baseline_flags_unverifiable_citations(db, settings, room):
    out = BaselineOut(findings=[conclusion(db), conclusion(db, cites=[CitationOut(chunk_id=1, quote="totally invented quote text")])])
    res = run_baseline(db, settings, FakeLLM({"baseline": [out]}), room)
    assert len(res.findings) == 2 and res.usage.calls == 1
    assert all(c.verified for c in res.findings[0].citations)
    assert not any(c.verified for c in res.findings[1].citations)


def test_baseline_llm_error_is_reported(db, settings, room):
    res = run_baseline(db, settings, FakeLLM({"baseline": [LLMError("x")]}), room)
    assert res.findings == [] and res.error


# ---------- API end to end ----------
def test_api_investigation_flow(settings, db):
    llm_holder, n_calls = {}, {"n": 0}

    def llm_factory():
        n_calls["n"] += 1
        if n_calls["n"] == 1:  # the ingest job asks for an LLM first: it only summarises documents
            return FakeLLM({"summarize": [DocSummary(overview="Doc summary text.", key_topics=["capacity"])]})
        llm_holder["llm"] = FakeLLM(base_script(db, [step("conclude", concl=conclusion(db))]))
        return llm_holder["llm"]

    app = create_app(settings, llm_factory=llm_factory,
                     tool_client_factory=lambda ids: LocalToolClient(DataRoomTools(db, ids)), run_inline=True)
    c = TestClient(app)
    c.app.state.ctx.db = db  # share the fixture db with the script's chunk lookups
    up = c.post("/api/v1/documents", files=[("files", (n, t.encode(), "text/plain"))
                                            for n, t in [("supplier.txt", SUP), ("model.txt", MOD)]]).json()
    ids = [u["document"]["id"] for u in up]
    job = c.get(f"/api/v1/jobs/{up[0]['job_id']}").json()  # preprocessing started automatically by the upload
    assert job["status"] == "completed" and job["usage"]["calls"] == 2 and job["results"][0]["summary_status"] == "done"
    assert c.get(f"/api/v1/documents/{ids[0]}").json()["summary"]["overview"] == "Doc summary text."
    r = c.post("/api/v1/investigations", json={})
    assert r.status_code == 202
    inv_id = r.json()["id"]
    p = c.get(f"/api/v1/investigations/{inv_id}").json()
    assert p["investigation"]["status"] == "completed" and p["findings"] == 1 and p["claims"] == 2
    f = c.get(f"/api/v1/investigations/{inv_id}/findings").json()[0]
    assert f["classification"] == "contradiction" and f["citations"][0]["verified"]
    assert len(c.get(f"/api/v1/investigations/{inv_id}/events").json()) > 3
    assert c.get(f"/api/v1/investigations/{inv_id}/hypotheses").json()[0]["status"] == "done"
    fb = c.post(f"/api/v1/investigations/{inv_id}/feedback", json={"finding_id": f["id"], "note": "clarify"})
    assert fb.status_code == 202
    assert c.get("/api/v1/investigations/999").status_code == 404
    assert c.post(f"/api/v1/investigations/{inv_id}/feedback", json={}).status_code == 409
    assert c.post(f"/api/v1/investigations/{inv_id}/resume").status_code == 409  # completed runs cannot be resumed


def test_api_start_without_documents_is_409(settings):
    c = TestClient(create_app(settings, run_inline=True))
    assert c.post("/api/v1/investigations", json={}).status_code == 409


def test_api_missing_api_key_fails_cleanly(settings, db, room, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    app = create_app(settings, tool_client_factory=lambda ids: LocalToolClient(DataRoomTools(db, ids)), run_inline=True)
    c = TestClient(app)
    c.app.state.ctx.db = db
    inv = c.post("/api/v1/investigations", json={}).json()
    assert c.get(f"/api/v1/investigations/{inv['id']}").json()["investigation"]["status"] == "failed"
    assert "ANTHROPIC_API_KEY" in c.get(f"/api/v1/investigations/{inv['id']}").json()["investigation"]["error"]
