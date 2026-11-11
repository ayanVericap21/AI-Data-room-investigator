"""LangGraph investigation workflow.

AI reasoning decides: which claims are worth comparing, what evidence to look for next, whether the evidence
resolves the hypothesis (and revising belief), and whether the cited quotes support the conclusion.
Deterministic code decides: phase routing/resume, step and hypothesis limits, loop/duplicate-call guards, tool
execution, quote existence checks, the >=2 verified passage rule, downgrade-to-unresolved, persistence.
"""
from __future__ import annotations

import logging
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from ..config import Settings
from ..db import Database
from ..llm import LLM, Block, LLMError
from ..mcp_server.client import ToolClient, ToolError
from ..models import Classification, Finding
from ..tools import normalize
from . import store
from .prompts import (EXTRACT_SYSTEM, HYPOTHESIZE_SYSTEM, INVESTIGATE_SYSTEM, SUPPORT_SYSTEM)
from .schemas import ClaimsOut, ConclusionOut, HypothesesOut, StepOut, SupportOut
from .verify import check_citation_rule, final_status, heuristic_candidates, verify_citations

log = logging.getLogger(__name__)


class S(TypedDict, total=False):
    phase: str
    hyp_id: int | None
    steps: int
    scratch: list[str]
    belief: str | None
    pending: dict | None
    seen: list[str]


def _fmt_chunk(c: dict, limit: int = 700) -> str:
    t = c["text"] if len(c["text"]) <= limit else c["text"][:limit] + " ..."
    return f"[passage {c['chunk_id']} | {c['filename']} p.{c['page']}]\n{t}"


class InvestigationRunner:
    def __init__(self, db: Database, settings: Settings, llm: LLM, tools: ToolClient, inv_id: int):
        self.db, self.settings, self.llm, self.tools, self.inv_id = db, settings, llm, tools, inv_id
        self._overview_text: str | None = None
        self.graph = self._build()

    # ---------- helpers ----------
    def log(self, kind: str, msg: str, **data: Any) -> None:
        store.log_event(self.db, self.inv_id, kind, msg, data)

    def _save_usage(self) -> None:
        u = self.llm.usage.model_dump()
        u["cost_usd"] = self.llm.usage.cost_usd(self.settings)
        u["cache_hit_ratio"] = self.llm.usage.cache_hit_ratio
        store.update(self.db, self.inv_id, usage=u)

    def _set_phase(self, phase: str) -> None:
        store.update(self.db, self.inv_id, phase=phase)

    def _overview(self) -> str:
        """Document summaries, fetched once through the MCP `list_documents` tool. Orientation only - not evidence."""
        if self._overview_text is None:
            lines = []
            try:
                for d in self.tools.call("list_documents", {}).get("documents", []):
                    s = d.get("summary")
                    if s:
                        lines.append(f"[document {d['id']}] {d['filename']}: {s['overview']}"
                                     + (f" Topics: {', '.join(s['key_topics'])}." if s.get("key_topics") else "")
                                     + (f" Key figures: {'; '.join(s['key_figures'])}." if s.get("key_figures") else ""))
            except ToolError as e:
                self.log("error", f"could not fetch document summaries: {e}")
            self._overview_text = ("Data room overview (auto-generated summaries: an orientation aid for choosing what to search; "
                                   "NOT evidence - never quote them):\n" + "\n".join(lines)) if lines else ""
        return self._overview_text

    def _claim_lines(self, ids: list[int]) -> str:
        by_id = {c.id: c for c in store.claims(self.db, self.inv_id)}
        names = {r["id"]: r["filename"] for r in self.db.query("SELECT id, filename FROM documents")}
        lines = []
        for i in ids:
            c = by_id.get(i)
            if c:
                lines.append(f"[claim {c.id}] {names.get(c.document_id)} passage {c.chunk_id}: {c.subject} = {c.value}"
                             f"{' ' + c.unit if c.unit else ''}; conditions: {c.conditions or 'none stated'}; quote: \"{c.quote}\"")
        return "\n".join(lines)

    # ---------- graph ----------
    def _build(self):
        g = StateGraph(S)
        for name, fn in [("extract_claims", self.n_extract), ("hypothesize", self.n_hypothesize), ("pick_next", self.n_pick),
                         ("decide", self.n_decide), ("run_tool", self.n_tool), ("finalize", self.n_finalize),
                         ("complete", self.n_complete)]:
            g.add_node(name, fn)
        g.add_conditional_edges(START, lambda s: {"extract": "extract_claims", "hypothesize": "hypothesize",
                                                  "investigate": "pick_next"}.get(s["phase"], "complete"))
        g.add_conditional_edges("extract_claims", lambda s: "hypothesize" if s["phase"] == "hypothesize" else "complete")
        g.add_conditional_edges("hypothesize", lambda s: "pick_next" if s["phase"] == "investigate" else "complete")
        g.add_conditional_edges("pick_next", lambda s: "decide" if s["phase"] == "investigate" else "complete")
        g.add_conditional_edges("decide", self._route_decide)
        g.add_edge("run_tool", "decide")
        g.add_edge("finalize", "pick_next")
        g.add_edge("complete", END)
        return g.compile()

    @staticmethod
    def _route_decide(s: S) -> str:
        a = (s.get("pending") or {}).get("action")
        return {"conclude": "finalize", "failed": "pick_next"}.get(a, "run_tool")

    def run(self) -> None:
        inv = store.get(self.db, self.inv_id)
        for h in store.hypotheses(self.db, self.inv_id):  # resume: a hypothesis left 'running' by a crash is retried
            if h.status == "running":
                store.set_hyp_status(self.db, h.id, "pending")
        store.update(self.db, self.inv_id, status="running", error=None)
        self.log("run_started", f"Run started at phase '{inv.phase}'")
        try:
            self.graph.invoke({"phase": inv.phase}, {"recursion_limit": 10_000})
        except Exception as e:  # unexpected: record and surface, never crash silently
            log.exception("investigation %s crashed", self.inv_id)
            store.update(self.db, self.inv_id, status="failed", error=f"{type(e).__name__}: {e}")
            self.log("error", f"Investigation crashed: {type(e).__name__}: {e}")
        self._save_usage()

    # ---------- nodes ----------
    def n_extract(self, s: S) -> S:
        self.log("phase", "Extracting claims from documents")
        total = dropped = failures = 0
        names = {r["id"]: r["filename"] for r in self.db.query("SELECT id, filename FROM documents")}
        for did in store.get(self.db, self.inv_id).document_ids:
            rows = self.db.query("SELECT id, page, text FROM chunks WHERE document_id=? ORDER BY ordinal", (did,))
            batches, cur, size = [], [], 0
            for r in rows:
                if cur and size + len(r["text"]) > self.settings.chars_per_extraction_batch:
                    batches.append(cur)
                    cur, size = [], 0
                cur.append(r)
                size += len(r["text"])
            if cur:
                batches.append(cur)
            seen: set[tuple[int, str]] = set()
            for batch in batches:
                valid = {r["id"] for r in batch}
                text = f"Document: {names.get(did)}\n\n" + "\n\n".join(f"[passage {r['id']} | page {r['page']}]\n{r['text']}" for r in batch)
                try:
                    out = self.llm.structured(system=EXTRACT_SYSTEM, user=text, schema=ClaimsOut, purpose="extract_claims")
                except LLMError as e:
                    failures += 1
                    self.log("error", f"Claim extraction failed for {names.get(did)}: {e}", document_id=did)
                    continue
                for c in out.claims:
                    if c.chunk_id not in valid:
                        dropped += 1
                        continue
                    res = self.tools.call("verify_quote", {"chunk_id": c.chunk_id, "quote": c.quote})
                    if not res.get("found"):
                        dropped += 1  # fabricated/misquoted claim: never enters the pipeline
                        continue
                    cid = res["chunk_id"]
                    key = (cid, normalize(c.quote))
                    if key in seen:
                        continue
                    seen.add(key)
                    store.add_claim(self.db, self.inv_id, did, cid, c.subject, c.value, c.unit, c.conditions, c.quote)
                    total += 1
        self.log("claims_extracted", f"{total} verified claims extracted ({dropped} dropped as unverifiable)",
                 claims=total, dropped=dropped, failed_batches=failures)
        self._save_usage()
        if total == 0 and failures:
            store.update(self.db, self.inv_id, status="failed", error="claim extraction failed for all documents", phase="done")
            return {"phase": "failed"}
        nxt = "hypothesize" if total >= 2 else "done"
        self._set_phase(nxt)
        return {"phase": nxt}

    def n_hypothesize(self, s: S) -> S:
        cl = store.claims(self.db, self.inv_id)
        known = {c.id for c in cl}
        seen: set[frozenset[int]] = {frozenset(h.claim_ids) for h in store.hypotheses(self.db, self.inv_id)}
        n = len(seen)
        proposed = 0
        try:
            out = self.llm.structured(system=HYPOTHESIZE_SYSTEM, purpose="hypothesize", schema=HypothesesOut,
                                      user=(f"{self._overview()}\n\n" if self._overview() else "") + "Claims:\n"
                                      + self._claim_lines([c.id for c in cl[: self.settings.max_claims_in_prompt]]))
        except LLMError as e:  # degrade gracefully: heuristic candidates still run
            self.log("error", f"Hypothesis generation failed, falling back to heuristics only: {e}")
            out = HypothesesOut(hypotheses=[])
        for h in out.hypotheses:
            ids = [i for i in dict.fromkeys(h.claim_ids) if i in known]
            key = frozenset(ids)
            if len(ids) < 2 or key in seen or n >= self.settings.max_hypotheses:
                continue
            seen.add(key)
            store.add_hypothesis(self.db, self.inv_id, ids, h.question, h.rationale)
            n += 1
            proposed += 1
        by_id = {c.id: c for c in cl}
        heur = 0
        for ids in heuristic_candidates(cl):
            key = frozenset(ids)
            if key in seen or n >= self.settings.max_hypotheses:
                continue
            seen.add(key)
            a, b = by_id[ids[0]], by_id[ids[1]]
            store.add_hypothesis(self.db, self.inv_id, ids, f"Do the statements about '{a.subject}' and '{b.subject}' conflict?",
                                 "heuristic: overlapping subject across documents with different values")
            n += 1
            heur += 1
        self.log("hypotheses", f"{proposed} hypotheses from the model, {heur} from heuristics", model=proposed, heuristic=heur)
        self._save_usage()
        nxt = "investigate" if n else "done"
        self._set_phase(nxt)
        return {"phase": nxt}

    def n_pick(self, s: S) -> S:
        h = store.next_pending(self.db, self.inv_id)
        if not h:
            self._set_phase("done")
            return {"phase": "done", "hyp_id": None}
        store.set_hyp_status(self.db, h.id, "running", bump_attempt=True)
        self.log("hypothesis_started", h.question, hypothesis_id=h.id, claim_ids=h.claim_ids, attempt=h.attempts + 1)
        scratch = []
        if h.claim_ids:
            scratch.append("Claims under comparison:\n" + self._claim_lines(h.claim_ids))
        return {"phase": "investigate", "hyp_id": h.id, "steps": 0, "scratch": scratch, "belief": None, "seen": [], "pending": None}

    def n_decide(self, s: S) -> S:
        h = store.get_hypothesis(self.db, s["hyp_id"])
        steps, limit = s["steps"], self.settings.max_steps_per_hypothesis
        force = steps >= limit
        # Prompt layout for caching: stable -> volatile. The data-room overview is identical for every step of every
        # hypothesis (breakpoint 1); the hypothesis header and the append-only evidence log grow step by step, so each
        # step re-reads the previous step's prefix (breakpoint 2 on the last evidence block). Only the short
        # "steps used" line changes every call, and it sits after the last breakpoint.
        header = (f"Hypothesis: {h.question}\nRationale: {h.rationale or 'n/a'}\n"
                  + (f"User clarifications to take into account:\n- " + "\n- ".join(h.clarifications) + "\n" if h.clarifications else "")
                  + "\nEvidence so far:")
        prefix = ([Block(text=self._overview(), cache=True)] if self._overview() else [])
        prefix += [Block(text=header)] + [Block(text=x) for x in s["scratch"]]
        prefix[-1] = Block(text=prefix[-1].text, cache=True)
        user = (f"Steps used: {steps}/{limit}."
                + (" You have no steps left: you MUST conclude now (insufficient_evidence if unresolved)." if force else ""))
        try:
            step = self.llm.structured(system=INVESTIGATE_SYSTEM, user=user, schema=StepOut, purpose="investigate_step",
                                       prefix=prefix)
        except LLMError as e:
            self.log("error", f"Analysis failed for hypothesis {h.id}: {e}", hypothesis_id=h.id)
            f = Finding(classification=Classification.INSUFFICIENT_EVIDENCE, title=h.question[:120],
                        explanation="The analysis of this question failed (model error) and was not completed.",
                        uncertainties=["analysis failed; retry the investigation"], status="unresolved",
                        verification_notes=[str(e)[:300]])
            store.save_finding(self.db, self.inv_id, h.id, f)
            store.set_hyp_status(self.db, h.id, "failed")
            return {"pending": {"action": "failed"}}
        if s.get("belief") and step.belief.value != s["belief"]:
            self.log("belief_changed", f"Belief revised: {s['belief']} -> {step.belief.value}", hypothesis_id=h.id,
                     reasoning=step.reasoning)
        action = step.action
        if action == "conclude" and step.conclusion is None:
            action = "search" if not force else "conclude"
        pending = step.model_dump(mode="json")
        if force and (step.action != "conclude" or step.conclusion is None):
            pending["action"] = "conclude"
            pending["conclusion"] = ConclusionOut(
                classification=Classification.INSUFFICIENT_EVIDENCE, title=h.question[:120],
                explanation="Step limit reached before the evidence resolved this question.", citations=[],
                uncertainties=["investigation stopped at the step limit"], confidence=0.2).model_dump(mode="json")
            self.log("step_limit", "Step limit reached; concluding as insufficient evidence", hypothesis_id=h.id)
        else:
            pending["action"] = action
        self.log("step", step.reasoning, hypothesis_id=h.id, belief=step.belief.value, action=pending["action"], step=steps + 1)
        return {"pending": pending, "belief": step.belief.value}

    def n_tool(self, s: S) -> S:
        p = s["pending"]
        action, scratch, seen = p["action"], list(s["scratch"]), list(s["seen"])
        if action == "search" and p.get("query"):
            name, args = "search_chunks", {"query": p["query"], "limit": 5}
        elif action == "get_neighbors" and p.get("chunk_id") is not None:
            name, args = "get_neighbors", {"chunk_id": p["chunk_id"], "window": 1}
        else:
            scratch.append(f"(Invalid {action} request: missing arguments. No evidence gathered.)")
            return {"steps": s["steps"] + 1, "scratch": scratch}
        key = f"{name}:{sorted(args.items())}"
        if key in seen:  # loop guard
            scratch.append(f"(You already ran {name} with these arguments; no new information. Try something different or conclude.)")
            self.log("tool_call", f"duplicate {name} skipped", hypothesis_id=s["hyp_id"], tool=name, args=args, duplicate=True)
        else:
            seen.append(key)
            try:
                res = self.tools.call(name, args)
                items = res.get("results") or res.get("chunks") or ([res["chunk"]] if "chunk" in res else [])
                scratch.append(f"{name}({args}) returned {len(items)} passage(s):\n" + "\n\n".join(_fmt_chunk(c) for c in items))
                self.log("tool_call", f"{name} -> {len(items)} passages", hypothesis_id=s["hyp_id"], tool=name, args=args,
                         chunk_ids=[c["chunk_id"] for c in items])
            except ToolError as e:
                scratch.append(f"({name} failed: {e})")
                self.log("error", f"tool {name} failed: {e}", hypothesis_id=s["hyp_id"])
        return {"steps": s["steps"] + 1, "scratch": scratch, "seen": seen}

    def n_finalize(self, s: S) -> S:
        h = store.get_hypothesis(self.db, s["hyp_id"])
        concl = ConclusionOut.model_validate(s["pending"]["conclusion"])
        cites = verify_citations(self.tools, concl.citations)
        cls, notes = check_citation_rule(concl.classification, cites)
        downgraded = cls != concl.classification
        unchecked = False
        if not downgraded and cls != Classification.INSUFFICIENT_EVIDENCE:
            quotes = "\n".join(f"- ({c.filename} p.{c.page}) \"{c.quote}\"" for c in cites if c.verified)
            try:
                sup = self.llm.structured(system=SUPPORT_SYSTEM, purpose="support_check", schema=SupportOut,
                                          user=f"Conclusion: {cls.value}\nTitle: {concl.title}\nExplanation: {concl.explanation}\n\nQuotes:\n{quotes}")
                if not sup.supported:
                    cls, downgraded = Classification.INSUFFICIENT_EVIDENCE, True
                    notes.append("support check failed: " + "; ".join(sup.issues or ["quotes do not support the conclusion"]))
            except LLMError as e:
                unchecked = True
                notes.append(f"support check unavailable ({e}); not presented as established")
        status = final_status(cls, concl.confidence, self.settings.min_established_confidence, downgraded or unchecked)
        unc = list(concl.uncertainties)
        if downgraded:
            unc.append(f"Model originally concluded '{concl.classification.value}' but verification did not support it.")
        f = Finding(classification=cls, title=concl.title, explanation=concl.explanation, citations=cites, uncertainties=unc,
                    assumptions=concl.assumptions, confidence=concl.confidence, status=status, verification_notes=notes)
        f = store.save_finding(self.db, self.inv_id, h.id, f)
        store.set_hyp_status(self.db, h.id, "done")
        self.log("finding_downgraded" if downgraded else "finding", f"{cls.value} ({status}): {concl.title}",
                 hypothesis_id=h.id, finding_id=f.id, classification=cls.value, status=status)
        self._save_usage()
        return {}

    def n_complete(self, s: S) -> S:
        inv = store.get(self.db, self.inv_id)
        if inv.status == "failed":
            return {}
        failed = [h for h in store.hypotheses(self.db, self.inv_id) if h.status == "failed"]
        status = "partial" if failed else "completed"
        store.update(self.db, self.inv_id, status=status, phase="done")
        self.log("completed", f"Investigation {status}" + (f" ({len(failed)} hypotheses failed)" if failed else ""))
        return {}
