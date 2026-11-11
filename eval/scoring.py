"""Deterministic scoring of findings against ground truth. Identical for both approaches.

Matching rule: a finding matches a gold item if the documents it cites (resolved from passage ids) include all of
the item's required documents (or one of its alt_docs sets) AND its text mentions at least one gold keyword. A
finding that matches no gold item is "unmatched" (reported for manual review; only unmatched *contradictions* count
as false positives).
"""
from __future__ import annotations

from itertools import combinations

from pydantic import BaseModel

from app.models import Classification, Finding

C = Classification


class GoldItem(BaseModel):
    id: str
    classification: Classification
    docs: list[str]
    alt_docs: list[list[str]] = []
    keywords: list[str]
    description: str = ""

    def doc_sets(self) -> list[set[str]]:
        return [set(self.docs), *[set(a) for a in self.alt_docs]]


class GroundTruth(BaseModel):
    case: str
    items: list[GoldItem]
    note: str = ""


class Pred(BaseModel):
    effective_class: Classification
    text: str
    cited_all: set[str]
    cited_verified: set[str]
    n_citations: int
    n_verified: int
    title: str = ""


def to_pred(f: Finding, chunk_doc: dict[int, str], approach: str) -> Pred:
    # Agent findings that are not "established" are open questions, i.e. effectively insufficient_evidence.
    eff = C.INSUFFICIENT_EVIDENCE if (approach == "agent" and f.status != "established") else f.classification
    text = " ".join([f.title, f.explanation, *[c.quote for c in f.citations]]).lower()
    return Pred(effective_class=eff, text=text, title=f.title,
                cited_all={chunk_doc[c.chunk_id] for c in f.citations if c.chunk_id in chunk_doc},
                cited_verified={c.filename for c in f.citations if c.verified and c.filename},
                n_citations=len(f.citations), n_verified=sum(1 for c in f.citations if c.verified))


def _hits(p: Pred, g: GoldItem) -> int:
    if not any(s <= p.cited_all for s in g.doc_sets()):
        return 0
    return sum(1 for k in g.keywords if k.lower() in p.text)


def match(preds: list[Pred], gold: list[GoldItem]) -> tuple[dict[str, Pred | None], list[int]]:
    assigned: dict[str, list[Pred]] = {g.id: [] for g in gold}
    unmatched: list[int] = []
    for i, p in enumerate(preds):
        scored = [(_hits(p, g), g.classification == p.effective_class, g) for g in gold]
        scored = [s for s in scored if s[0] > 0]
        if not scored:
            unmatched.append(i)
            continue
        assigned[max(scored, key=lambda s: (s[0], s[1]))[2].id].append(p)
    best = {}
    gmap = {g.id: g for g in gold}
    for gid, ps in assigned.items():
        same = [p for p in ps if p.effective_class == gmap[gid].classification]
        best[gid] = (same or ps or [None])[0]
    return best, unmatched


def score_run(preds: list[Pred], truth: GroundTruth) -> dict:
    best, unmatched = match(preds, truth.items)
    items = {g.id: (best[g.id].effective_class.value if best[g.id] else None) for g in truth.items}

    def acc(cls: Classification) -> float | None:
        sel = [g for g in truth.items if g.classification == cls]
        return round(sum(items[g.id] == cls.value for g in sel) / len(sel), 3) if sel else None

    gold_contra = {g.id for g in truth.items if g.classification == C.CONTRADICTION}
    # false positives: predicted contradictions that are not the best match of a gold contradiction
    matched_good = {id(best[gid]) for gid in gold_contra if best[gid] and best[gid].effective_class == C.CONTRADICTION}
    false_pos = sum(1 for p in preds if p.effective_class == C.CONTRADICTION and id(p) not in matched_good)
    overclaim = [g.id for g in truth.items if g.classification == C.INSUFFICIENT_EVIDENCE
                 and items[g.id] in (C.CONTRADICTION.value, C.CONTEXTUAL_DIFFERENCE.value, C.CONSISTENT.value)]
    ev = []
    for g in truth.items:
        p = best[g.id]
        if p:
            ev.append(max(len(s & p.cited_verified) / len(s) for s in g.doc_sets()))
    n_cit = sum(p.n_citations for p in preds)
    return {
        "items": items,
        "found_rate": round(sum(v is not None for v in items.values()) / len(items), 3),
        "classification_accuracy": round(sum(items[g.id] == g.classification.value for g in truth.items) / len(truth.items), 3),
        "contradiction_recall": acc(C.CONTRADICTION),
        "contextual_accuracy": acc(C.CONTEXTUAL_DIFFERENCE),
        "consistent_accuracy": acc(C.CONSISTENT),
        "uncertainty_accuracy": acc(C.INSUFFICIENT_EVIDENCE),
        "false_positive_contradictions": false_pos,
        "overclaimed_unverifiable": len(overclaim),
        "evidence_recall": round(sum(ev) / len(ev), 3) if ev else None,
        "citation_validity": round(sum(p.n_verified for p in preds) / n_cit, 3) if n_cit else None,
        "n_findings": len(preds),
        "n_unmatched": len(unmatched),
        "unmatched_titles": [preds[i].title for i in unmatched],
    }


def repeatability(runs: list[dict]) -> float | None:
    """Mean over item and run-pair of 'same predicted class'. 1.0 = identical results every run."""
    if len(runs) < 2:
        return None
    ids = list(runs[0]["items"])
    agree = [runs[a]["items"][i] == runs[b]["items"][i] for a, b in combinations(range(len(runs)), 2) for i in ids]
    return round(sum(agree) / len(agree), 3)


NUMERIC = ["found_rate", "classification_accuracy", "contradiction_recall", "contextual_accuracy", "consistent_accuracy",
           "uncertainty_accuracy", "false_positive_contradictions", "overclaimed_unverifiable", "evidence_recall",
           "citation_validity", "n_findings", "n_unmatched"]


def aggregate(runs: list[dict]) -> dict:
    out = {}
    for k in NUMERIC:
        vals = [r[k] for r in runs if r.get(k) is not None]
        out[k] = round(sum(vals) / len(vals), 3) if vals else None
    out["repeatability"] = repeatability(runs)
    return out
