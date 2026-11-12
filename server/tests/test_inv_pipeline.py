"""The whole pipeline on a small room with a scripted model: tables, facts, case model, plan, workers, verifier, consolidation, saving, report, score."""
import json
import re
from pathlib import Path

import pytest
from sqlalchemy import select

from app.evaluation.scoring import AnswerKey, load_key, render as render_score, score
from app.investigator.case_model import CaseModelDraft
from app.investigator.models import ConsolidationPlan, ReferenceKinds, VerifierBatch, WorkerOutcome, WorkerPlan
from app.investigator.pipeline import CoverageGateError, InvestigatorPipeline, Options, RunStopped
from app.investigator.report import RunInfo, findings_rows, metrics_dict, render
from app.llm.llama_extract import LlamaJob
from app.models import Finding, Investigation
from app.services.llm_pairing import ClaimGroups
from app.tools import build_registry
from tests.inv_support import RoutedLLM, dive_room
from tests.test_inv_case_tools_workers import draft

WINDOW = "Sampling is permitted between 3 March 2025 and 21 March 2025."
LIMIT = "No more than 8 fragments may be collected per site."
LOG = "The decontamination log must be submitted with the final report."
MEAN = "Mean sample count per dive was 5.7."


class FactsClient:
    def __init__(self, by_file):
        self.by_file, self.calls = by_file, []

    def extract(self, path, schema, **kw):
        name = Path(path).name
        self.calls.append(name)
        return LlamaJob(job_id="x", status="COMPLETED", result={"facts": self.by_file.get(name, [])}, metadata={}, credits=3.0)


def facts_by_file():
    return {
        "permit.docx": [
            dict(type="constraint", subject="sampling", attribute="permitted period", operator="between", value="3 March 2025 to 21 March 2025", modality="must", quote=WINDOW),
            dict(type="constraint", subject="site", attribute="fragments collected", operator="<=", value="8", unit="fragments", modality="must", quote=LIMIT),
            dict(type="reference", subject="permit", attribute="requires", value="decontamination log", quote=LOG)],
        "lab.pdf": [dict(type="observation", subject="dives", attribute="mean sample count per dive", value="5.7", aggregate_of="samples per dive", value_type="quantity", quote=MEAN)],
    }


def worker_plan(prompt):
    table = re.search(r"(t\d+): register\.xlsx", prompt).group(1)
    return {"note": "look at the register", "calls": [{"task_id": t, "tool": "sql", "sql": f'SELECT dive, date, samples, _ref FROM "{table}"'}
                                                      for t in re.findall(r"### Task (\w+)", prompt)[:1]]}


def worker_outcome(prompt):
    table = re.search(r"(t\d+): register\.xlsx", prompt).group(1)
    results = []
    for task in re.findall(r"### Task (\w+) \(family (\w)", prompt):
        tid, family = task
        block = prompt.split(f"### Task {tid}")[1].split("### Task")[0]
        if family == "A" and "3 March 2025" in block:
            results.append({"task_id": tid, "status": "finding", "candidates": [{
                "type": "constraint_violation", "title": "Dive D-03 took place after the permit window", "description": "D-03 is dated 22 March; the permit allows sampling until 21 March.",
                "severity": "high", "confidence": 0.8, "entities": ["D-03"], "computed": "dates after 2025-03-21: D-03",
                "evidence": [{"kind": "row", "table": table, "ref": "A4:D4", "shows": "dive D-03 on 2025-03-22", "role": "contradicting"},
                             {"kind": "quote", "document": "permit.docx", "quote": WINDOW, "shows": "the permitted window", "role": "supporting"}]}]})
        elif family == "A" and "8 fragments" in block:
            results.append({"task_id": tid, "status": "finding", "candidates": [{
                "type": "constraint_violation", "title": "Dive D-04 collected 9 samples against a limit of 8", "description": "9 samples were taken on D-04; the permit limits collection to 8.",
                "severity": "medium", "confidence": 0.7, "entities": ["D-04"],
                "evidence": [{"kind": "row", "table": table, "ref": "A5:D5", "shows": "dive D-04 with 9 samples", "role": "contradicting"},
                             {"kind": "quote", "document": "permit.docx", "quote": LIMIT, "shows": "the limit", "role": "supporting"}]}]})
        else:
            results.append({"task_id": tid, "status": "passed", "reason": "checked, matches"})
    return {"results": results}


def confirm_all(prompt):
    return {"verdicts": [{"candidate": int(n), "verdict": "confirmed", "reasons": "stands"} for n in re.findall(r"### Candidate (\d+)", prompt)]}


def llm_for_room(extra=None):
    handlers = {CaseModelDraft: draft(), WorkerPlan: worker_plan, WorkerOutcome: worker_outcome, VerifierBatch: confirm_all, ConsolidationPlan: {"groups": []},
                ReferenceKinds: {"items": [{"number": 0, "kind": "expected_artifact", "reason": "a log the permit requires"}]}, ClaimGroups: {"groups": []}}
    handlers.update(extra or {})
    return RoutedLLM(handlers)


def pipeline(session, tmp_path, llm, client, options=None, should_stop=None):
    inv = Investigation(name="t")
    session.add(inv)
    session.commit()
    registry = build_registry(session, None, inv.id)
    return InvestigatorPipeline(session, llm, None, registry, inv.id, tmp_path, tmp_path / "tables.sqlite", client, options or Options(), should_stop=should_stop,
                                case_path=tmp_path / "case.json"), inv


def test_the_pipeline_finds_the_planted_violations_and_the_missing_log_and_reports_them(session, tmp_path):
    dive_room(session)
    llm = llm_for_room()
    p, inv = pipeline(session, tmp_path, llm, FactsClient(facts_by_file()))
    out = p.run()
    titles = sorted(f.title for f in session.scalars(select(Finding).where(Finding.investigation_id == inv.id)))
    assert titles == ["[constraint violation] Dive D-03 took place after the permit window", "[constraint violation] Dive D-04 collected 9 samples against a limit of 8",
                      "[missing evidence] Referenced item not in the data room: decontamination log"]
    assert len(out.saved) == 3 and len(out.facts) == 4 and out.coverage and all(c.ok for c in out.coverage.values())
    assert {r.family for r in out.records} >= {"A", "D", "F", "G"} and any(r.status == "passed" and r.family == "D" for r in out.records)
    assert (tmp_path / "case.json").exists() and set(out.stage_ms) >= {"1. load tables (SQLite), keep cell addresses", "6. verify adversarially"}
    assert llm.count(VerifierBatch) >= 1 and out.verify.stats.by_verdict["confirmed"] == 3


def test_the_report_the_findings_file_and_the_score_describe_the_run(session, tmp_path):
    dive_room(session)
    p, inv = pipeline(session, tmp_path, llm_for_room(), FactsClient(facts_by_file()))
    out = p.run()
    info = RunInfo("20260101-000000", "t", "db", {"workers": 1}, "2026-01-01 00:00:00", 12.0, out, p.llm, inv.id, session)
    text = render(info)
    for heading in ("AT A GLANCE", "COVERAGE GATE", "CASE MODEL", "FINDINGS", "CHECK PLAN AND AUDIT LOG", "VERIFIER", "CONSOLIDATION", "TIMING AND LLM USE PER STAGE"):
        assert heading in text
    rows = findings_rows(info)
    assert len(rows) == 3 and rows[0]["taxonomy"] in ("constraint_violation", "missing_evidence") and rows[0]["family"] in ("A", "G") and rows[0]["evidence"]
    assert json.loads(json.dumps(metrics_dict(info)))["findings"] == 3
    key = AnswerKey.model_validate({"room": "t", "issues": [
        {"id": "T1", "title": "dive after window", "match": {"all": ["d-03"], "any": ["window", "march"]}},
        {"id": "T2", "title": "limit broken", "match": {"all": ["d-04", "8"]}},
        {"id": "T3", "title": "log missing", "match": {"all": ["decontamination"]}, "classes": ["missing_information"]},
        {"id": "T4", "title": "never found", "match": {"all": ["zebra"]}}, {"id": "T5", "title": "by hand"}],
        "traps": [{"id": "L1", "title": "limit is a trap", "match": {"all": ["d-04"]}, "flag_classes": ["insufficient_evidence"]}]})
    s = score(key, rows)
    assert [r.hit for r in s.counted] == [True, True, True, False] and s.recall == 0.75 and len(s.issues) == 5 and s.unmatched == []
    assert "MISSED" in render_score(s) and "[manual] T5" in render_score(s)


def test_the_real_answer_keys_load():
    keys = Path(__file__).resolve().parents[2] / "test_results" / "answer_keys"
    for name in ("dive_room.json", "reef_room.json"):
        key = load_key(keys / name)
        assert key.issues and key.traps


def test_a_document_that_produced_no_facts_and_no_tables_fails_the_gate_unless_told_to_warn(session, tmp_path):
    dive_room(session)
    with pytest.raises(CoverageGateError) as caught:
        pipeline(session, tmp_path, llm_for_room(), FactsClient({}))[0].run()
    assert "lab.pdf" in str(caught.value) and "permit.docx" in str(caught.value) and "register.xlsx" not in str(caught.value)
    (tmp_path / "w").mkdir()
    p, _ = pipeline(session, tmp_path / "w", llm_for_room(), FactsClient({}), Options(gate="warn"))
    out = p.run()
    assert {c.filename for c in out.gate_failures} == {"lab.pdf", "permit.docx"}


def test_a_stop_request_ends_the_run_before_the_next_stage(session, tmp_path):
    dive_room(session)
    p, _ = pipeline(session, tmp_path, llm_for_room(), FactsClient(facts_by_file()), should_stop=lambda: True)
    with pytest.raises(RunStopped):
        p.run()


def test_without_the_verifier_every_candidate_is_reported_and_families_can_be_chosen(session, tmp_path):
    dive_room(session)
    llm = llm_for_room()
    p, inv = pipeline(session, tmp_path, llm, FactsClient(facts_by_file()), Options(verify=False, families="A"))
    out = p.run()
    assert llm.count(VerifierBatch) == 0 and len(out.saved) == 2 and {r.family for r in out.records} == {"A"}


def test_the_documents_are_read_once_each_and_only_the_selected_ones(session, tmp_path):
    dive_room(session)
    client = FactsClient(facts_by_file())
    p, _ = pipeline(session, tmp_path, llm_for_room(), client, Options(families="A", verify=False))
    p.run({"permit.docx", "register.xlsx"})
    assert sorted(client.calls) == ["permit.docx", "register.xlsx"]


def test_one_results_file_carries_the_metrics_findings_case_model_and_trace(session, tmp_path):
    import json
    from app.evaluation.scoring import load_findings
    from app.services.run_metrics import extract_block, find_metrics
    dive_room(session)
    p, inv = pipeline(session, tmp_path, llm_for_room(), FactsClient(facts_by_file()))
    out = p.run()
    results = tmp_path / "TEST-RUN-x.txt"
    info = RunInfo("20260101-000000", "t", "db", {"gate": "fail", "max_tasks": 60}, "2026-01-01T00:00:00", 12.0, out, p.llm, inv.id, session, llm_model="m", embedding_model="e",
                   results_path=str(results), finished="2026-01-01T00:00:12")
    results.write_text(render(info), encoding="utf-8")
    assert [x.name for x in tmp_path.glob("TEST-RUN-*")] == ["TEST-RUN-x.txt"]  # nothing is written next to it
    text = results.read_text(encoding="utf-8")
    for name in ("RUN_METRICS", "FINDINGS", "CASE_MODEL", "STATS", "WORKER_TRACE"):
        assert json.loads(extract_block(text, name)) is not None
    found = find_metrics(inv.id, tmp_path)  # the run page's metrics panel reads the block
    assert found is not None and found.counts.findings_saved == 3 and found.counts.documents == 3 and found.llm.credits >= 9.0 and found.stages
    assert len(load_findings(results)) == 3 and json.loads(extract_block(text, "CASE_MODEL"))["case_summary"] == "A sampling survey."
    assert json.loads(extract_block(text, "WORKER_TRACE"))[0]["outcome"]["results"]
