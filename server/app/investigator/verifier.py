"""The adversarial verifier: every candidate finding must survive an attempt to explain it away before it is reported.

The verifier is a separate call that sees the candidate and its raw evidence (exact quotes with their locations), not the worker's reasoning,
and passages from elsewhere in the room that might resolve or explain it. It answers one of: confirmed, contextual_difference, unsupported,
missing_evidence, rejected. Only the first, the explained difference, an absent-record finding and (for conclusions) an unsupported one are
reported; everything else goes to the audit log with the reason. A verifier call that fails leaves its candidates reported as unverified.
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable

from app.investigator.models import SEVERITIES, Candidate, VerifierBatch
from app.investigator.tables import TableStore

logger = logging.getLogger(__name__)

SYSTEM = (
    "You are an adversarial verifier in a data room investigation. Each numbered candidate finding was proposed by another analyst. You do not see their reasoning, and you must not "
    "assume they are right: your job is to try to explain each candidate away. For each one: "
    "(1) For every evidence quote, say whether it exists in the text shown and supports exactly what the candidate says it shows (supports true or false, with a note). "
    "(2) Look for a legitimate explanation: a different scope, period, metric, population or unit; plan against actual; a documented deviation; a progression over time; "
    "one name containing the other; a value inside a stated limit; a definition that makes the figures compatible. "
    "(3) Look in the passages from elsewhere in the room for evidence that resolves it, such as the approval or document that exists after all. "
    "(4) Decide: confirmed (the candidate stands as stated and no explanation was found), contextual_difference (a real difference that stated context fully explains), "
    "unsupported (a claim or conclusion that the evidence does not back, without being shown wrong), missing_evidence (something required is absent from the room), "
    "rejected (it is wrong, trivial, or its quotes do not support it). "
    "Give reasons in one or two sentences, name any explanation_found or resolving_evidence, and a severity only if the candidate's is wrong. "
    "Wording differences, containment, plan against actual and limit against actual are not contradictions. Return exactly one verdict per candidate, using its number.")


@dataclass
class VerifyStats:
    candidates: int = 0
    calls: int = 0
    by_verdict: dict[str, int] = field(default_factory=dict)
    unverified: int = 0
    errors: list[str] = field(default_factory=list)


@dataclass
class VerifyResult:
    kept: list[Candidate] = field(default_factory=list)
    rejected: list[Candidate] = field(default_factory=list)
    stats: VerifyStats = field(default_factory=VerifyStats)


class Verifier:
    def __init__(self, llm: Any, store: TableStore, workers: int = 1, batch_size: int = 4, should_stop: Callable[[], bool] | None = None,
                 on_progress: Callable[[str], None] | None = None):
        self.llm, self.store = llm, store
        self.workers = max(1, workers)
        self.batch_size = max(1, batch_size)
        self.should_stop = should_stop or (lambda: False)
        self.on_progress = on_progress or (lambda _m: None)

    def _prompt(self, batch: list[Candidate]) -> str:
        blocks = []
        for n, c in enumerate(batch, start=1):
            seen = {e.chunk_id for e in c.evidence}
            ev = "\n".join(f"  [{k}] ({e.role}) {e.filename}, {e.location}: \"{e.quote[:500]}\"\n      candidate says this shows: {e.shows or '-'}" for k, e in enumerate(c.evidence, start=1))
            hits = [h for h in self.store.search(" ".join([c.title, *c.entities[:4]]), limit=6) if h["chunk_id"] not in seen][:3]
            elsewhere = "\n".join(f"  - {h['filename']}, {h['location']}: {h['snippet'][:260]}" for h in hits) or "  (nothing found)"
            blocks.append(f"### Candidate {n} ({c.type.replace('_', ' ')}, severity {c.severity}, from check family {c.family})\nTitle: {c.title}\nDescription: {c.description}\n"
                          + (f"Computed: {c.computed}\n" if c.computed else "") + f"Evidence:\n{ev}\nPassages elsewhere in the room that may explain or resolve it:\n{elsewhere}")
        return "\n\n".join(blocks) + "\n\nVerify each candidate."

    def _one(self, batch: list[Candidate]) -> VerifierBatch | Exception:
        if self.should_stop():
            return RuntimeError("stop requested")
        try:
            return self.llm.generate_structured_output(self._prompt(batch), VerifierBatch, system=SYSTEM)
        except Exception as exc:
            return exc

    def verify(self, candidates: list[Candidate]) -> VerifyResult:
        result = VerifyResult()
        result.stats.candidates = len(candidates)
        batches = [candidates[i:i + self.batch_size] for i in range(0, len(candidates), self.batch_size)]
        if self.workers <= 1 or len(batches) <= 1:
            answers = [self._one(b) for b in batches]
        else:
            with ThreadPoolExecutor(max_workers=self.workers) as pool:
                answers = [f.result() for f in [pool.submit(self._one, b) for b in batches]]
        for i, (batch, answer) in enumerate(zip(batches, answers), start=1):
            result.stats.calls += 1
            if isinstance(answer, Exception):
                result.stats.errors.append(f"{type(answer).__name__}: {str(answer)[:140]}")
                for c in batch:
                    c.verdict, c.verifier_notes = "unverified", f"the verifier could not be reached ({type(answer).__name__})"
                    c.status = "needs_investigation"
                    c.confidence = min(c.confidence, 0.4)
                    result.stats.unverified += 1
                    result.kept.append(c)
                continue
            by_number = {v.candidate: v for v in answer.verdicts}
            for n, c in enumerate(batch, start=1):
                v = by_number.get(n)
                if v is None:
                    c.verdict, c.verifier_notes = "rejected", "the verifier returned no verdict for this candidate"
                    result.rejected.append(c)
                    result.stats.by_verdict["no verdict"] = result.stats.by_verdict.get("no verdict", 0) + 1
                    continue
                self._apply(c, v, result)
            self.on_progress(f"verify: batch {i} of {len(batches)} done ({len(result.kept)} kept, {len(result.rejected)} rejected so far)")
        return result

    @staticmethod
    def _apply(c: Candidate, v, result: VerifyResult) -> None:
        verdict = v.verdict
        if v.quotes and all(not q.supports for q in v.quotes) and verdict == "confirmed":
            verdict = "rejected"  # the verifier itself found that no quote supports the finding
            v.reasons = ("none of the quotes supports the finding. " + v.reasons).strip()
        result.stats.by_verdict[verdict] = result.stats.by_verdict.get(verdict, 0) + 1
        c.verdict = verdict
        c.verifier_notes = " ".join(x for x in (v.reasons.strip(), f"Explanation considered: {v.explanation_found.strip()}" if v.explanation_found.strip() else "",
                                                f"Possibly resolving evidence: {v.resolving_evidence.strip()}" if v.resolving_evidence.strip() else "") if x)
        if v.severity in SEVERITIES:
            c.severity = v.severity
        if verdict == "confirmed":
            result.kept.append(c)
        elif verdict == "contextual_difference":
            c.type = "contextual_difference"
            c.severity = "low" if c.severity not in ("informational", "low") else c.severity
            result.kept.append(c)
        elif verdict == "missing_evidence":
            if c.type not in ("missing_evidence", "reconciliation_gap"):
                c.type = "missing_evidence"
            c.status = "needs_investigation"
            result.kept.append(c)
        elif verdict == "unsupported" and c.type == "unsupported_conclusion":
            result.kept.append(c)
        else:
            result.rejected.append(c)
