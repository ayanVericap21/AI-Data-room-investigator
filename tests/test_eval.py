from pathlib import Path

import pytest

from app.agent.schemas import (BaselineOut, CitationOut, ClaimOut, ClaimsOut, ConclusionOut, HypothesesOut, HypothesisOut,
                               StepOut, SupportOut)
from app.models import Citation, Classification as C, DocSummary, Finding
from eval.run_eval import evaluate, render_markdown
from eval.scoring import GroundTruth, GoldItem, Pred, aggregate, repeatability, score_run, to_pred
from tests.test_agent import FakeLLM

CASE = Path(__file__).resolve().parents[1] / "eval" / "cases" / "sample_plant"
TRUTH = GroundTruth.model_validate_json((CASE / "ground_truth.json").read_text())


def pred(cls, docs, text, verified=None):
    verified = docs if verified is None else verified
    return Pred(effective_class=cls, text=text.lower(), cited_all=set(docs), cited_verified=set(verified),
                n_citations=len(docs), n_verified=len(verified))


def test_perfect_run_scores_perfectly():
    preds = [
        pred(C.CONTRADICTION, ["supplier_quotation.txt", "financial_model.txt"], "supplier guarantees 25,000 tonnes"),
        pred(C.CONTRADICTION, ["feasibility_study.txt", "engineering_report.txt"], "commissioning 2026 vs 2027"),
        pred(C.CONTEXTUAL_DIFFERENCE, ["feasibility_study.txt", "engineering_report.txt"], "only at 92% availability, 27,500"),
        pred(C.CONTEXTUAL_DIFFERENCE, ["financial_model.txt", "investor_memo.txt"], "revenue 10.5 vs 21.0"),
        pred(C.CONSISTENT, ["feasibility_study.txt", "financial_model.txt"], "capex 48 million"),
        pred(C.INSUFFICIENT_EVIDENCE, ["feasibility_study.txt"], "permits claimed secured, no permit documents"),
    ]
    s = score_run(preds, TRUTH)
    assert s["classification_accuracy"] == 1.0 and s["false_positive_contradictions"] == 0
    assert s["evidence_recall"] == 1.0 and s["citation_validity"] == 1.0 and s["n_unmatched"] == 0


def test_false_positive_and_overclaim_and_missing():
    preds = [
        # contextual difference mislabelled as contradiction -> false positive
        pred(C.CONTRADICTION, ["financial_model.txt", "investor_memo.txt"], "revenue 10.5 vs 21.0"),
        # unverifiable permit claim presented as a contradiction -> overclaim
        pred(C.CONTRADICTION, ["feasibility_study.txt"], "permit situation"),
        # contradiction about nothing in the gold set
        pred(C.CONTRADICTION, ["investor_memo.txt", "financial_model.txt"], "irr dispute"),
    ]
    s = score_run(preds, TRUTH)
    assert s["false_positive_contradictions"] == 3
    assert s["overclaimed_unverifiable"] == 1
    assert s["contradiction_recall"] == 0.0 and s["found_rate"] < 1.0
    assert s["n_unmatched"] == 1 and s["contextual_accuracy"] == 0.0


def test_alt_docs_match_and_unverified_citations_earn_no_evidence_credit():
    p = pred(C.CONTRADICTION, ["supplier_quotation.txt", "feasibility_study.txt"], "25,000 vs 30,000 tonnes", verified=[])
    s = score_run([p], TRUTH)
    assert s["items"]["G1"] == "contradiction"
    assert s["evidence_recall"] == 0.0 and s["citation_validity"] == 0.0


def test_empty_run():
    s = score_run([], TRUTH)
    assert s["found_rate"] == 0 and s["citation_validity"] is None and s["false_positive_contradictions"] == 0


def test_to_pred_treats_unresolved_agent_findings_as_insufficient():
    f = Finding(classification=C.CONTRADICTION, title="t", explanation="e", status="unresolved",
                citations=[Citation(chunk_id=1, quote="q", verified=True, filename="a.txt")])
    assert to_pred(f, {1: "a.txt"}, "agent").effective_class == C.INSUFFICIENT_EVIDENCE
    assert to_pred(f, {1: "a.txt"}, "baseline").effective_class == C.CONTRADICTION


def test_repeatability_and_aggregate():
    a = {"items": {"G1": "contradiction", "G2": None}}
    b = {"items": {"G1": "contradiction", "G2": "consistent"}}
    assert repeatability([a, a]) == 1.0 and repeatability([a, b]) == 0.5 and repeatability([a]) is None
    assert aggregate([{"found_rate": 1.0, "items": a["items"]}, {"found_rate": 0.5, "items": a["items"]}])["found_rate"] == 0.75


def test_evaluate_plumbing_with_fake_llm(tmp_path, settings):
    """Runs both approaches end-to-end against the sample case with a scripted LLM (checks wiring, not model quality)."""
    import re

    def factory():
        # chunk ids are only known after ingest; resolve them from the prompt text instead
        class Lazy(FakeLLM):
            def structured(self, *, system, user, schema, purpose, prefix=None):
                user = "\n\n".join([*(b.text for b in (prefix or [])), user])
                self.prompts.append((purpose, user))
                self.record(purpose, 10, 5, 0.0)
                if purpose == "summarize":
                    return DocSummary(overview="Summary of a document.", key_topics=["t"])
                if purpose == "baseline":
                    def pid(name, frag):
                        m = re.search(r"\[passage (\d+) \| " + name + r" \| p\.\d+\]\n[^\[]*?" + re.escape(frag), user)
                        return int(m.group(1))
                    return BaselineOut(findings=[ConclusionOut(
                        classification=C.CONTRADICTION, title="Capacity", explanation="25,000 vs 30,000 tonnes", confidence=0.9,
                        citations=[CitationOut(chunk_id=pid("supplier_quotation.txt", "25,000 tonnes"), quote="guarantees an annual output of 25,000 tonnes"),
                                   CitationOut(chunk_id=pid("financial_model.txt", "30,000 tonnes"), quote="assumed to be 30,000 tonnes")])])
                if purpose == "extract_claims":
                    return ClaimsOut(claims=[])
                raise AssertionError(purpose)
        return Lazy({})

    res = evaluate(CASE, factory, runs=2, approaches=["baseline", "agent"], settings=settings, workdir=tmp_path / "w", use_mcp=False)
    b = res["approaches"]["baseline"]
    assert b["aggregate"]["contradiction_recall"] == 0.5 and b["aggregate"]["repeatability"] == 1.0
    assert b["runs"][0]["score"]["citation_validity"] == 1.0
    a = res["approaches"]["agent"]
    assert a["runs"][0]["status"] == "completed" and a["aggregate"]["found_rate"] == 0.0  # no claims -> no findings
    md = render_markdown(res)
    assert "| Metric | baseline | agent |" in md and "G1" in md
