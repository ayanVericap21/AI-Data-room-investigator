"""Worker agents: investigate a batch of tasks with tools and return candidate findings.

LlamaExtract is the only model, and each call to it can take a minute or more, so a worker is not a long chat. For a batch of related tasks it
runs rounds of: (1) PLAN, one call in which the model writes up to six tool calls (usually SQL it composes against the loaded tables);
(2) code runs them; (3) CONCLUDE, one call in which the model reads the results and decides each task: finding, passed, untestable (with the
reason), or needs_more (another round, at most ``max_rounds``). A task that cannot be settled is reported as untestable, never guessed.
Candidate evidence is resolved by code into exact chunk text; a candidate whose evidence cannot be found is dropped and logged.
"""
from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable

from app.investigator.models import (FAMILIES, SEVERITIES, TAXONOMY, Candidate, TaskRecord, ToolCall, WorkerOutcome, WorkerPlan)
from app.investigator.tables import FUNCTIONS_HELP, TableStore
from app.investigator.tools import ToolBox

logger = logging.getLogger(__name__)

TAXONOMY_HELP = (
    "direct_contradiction (two statements or values about the same thing cannot both be true), contextual_difference (a difference a stated scope, period, definition or plan-versus-actual explains), "
    "temporal_inconsistency (the order or timing of events is impossible or conflicts), constraint_violation (a record breaks a rule, limit, window, deadline or allowed list), "
    "reconciliation_gap (identifiers or quantities that should reconcile across documents do not), missing_evidence (a required or referenced document, record or approval is not in the room), "
    "unsupported_conclusion (a conclusion the evidence does not support), data_integrity_issue (the data itself is implausible, duplicated or reused).")

PLAN_SYSTEM = (
    "You are a worker in a data room investigation. You test checks with tools against tables loaded from the documents (SQLite) and the text of the documents. "
    "Tools: sql(sql): one read-only SELECT on the tables listed (quote identifiers with double quotes; every table also has _ref (the row address), _chunk and _line); "
    "search(query, document): passages that match words; read(document, location): chunks of a document (location: part of its location text, or a chunk id); "
    "find_entity(entity): every mention of an identifier or alias; timeline(entity, window_start, window_end): dated events; get_facts(query): extracted facts. "
    + FUNCTIONS_HELP + " "
    "Return a plan: at most six tool calls, each tagged with the task_id it serves. Prefer one precise query to many searches. In SQL, select the columns you will cite, "
    "and always _ref, so the rows can be used as evidence. Use only the tables and columns shown. Small tables are shown in full with their _ref; larger ones only as a sample, "
    "so never decide a task from a sample: query the table. Every task needs at least one call unless the data it needs is already shown in full or in the results so far.")

CONCLUDE_SYSTEM = (
    "You are a worker in a data room investigation. Read the tool results and decide each task: finding (give candidates), passed (checked, and the records comply), "
    "untestable (say exactly why: which record is missing from the room; never because only some rows were shown, which is needs_more), or needs_more (when one more round of tool calls would settle it). "
    "A candidate has: type, one of " + TAXONOMY_HELP + " title; description (the values, which document states each and where, and why they conflict); "
    "severity (critical, high, medium, low, informational) chosen by materiality, not by type; confidence 0 to 1; "
    "status (definitive when the records in the room settle it, needs_investigation otherwise, with evidence_needed naming what would resolve it); "
    "evidence: each item either kind row with table and _ref copied from your SQL results, or kind quote with document and a quote copied exactly from a result; "
    "give at least two items when two statements or a rule and a record are compared, say what each shows, and its role (supporting, contradicting, contextual, insufficient); "
    "computed: the query or arithmetic behind the finding and what it returned; entities: the identifiers and names involved; root_cause: a short phrase for the underlying cause. "
    "Use only what the results show; never invent a value, row or quote. These are not contradictions: wording differences or one name containing the other, a limit against an actual "
    "inside the limit, plan against actual, different scopes, periods or populations. When a rule requires an approval, record or document, check that the room holds it before calling it missing.")


@dataclass
class WorkerTask:
    task_id: str
    family: str
    title: str
    instructions: str
    context: str = ""
    tables: set[str] | None = None  # tables to show with the task (None: all, up to a size cap)

    def text(self) -> str:
        return f"### Task {self.task_id} (family {self.family}, {FAMILIES.get(self.family, '')})\n{self.title}\n{self.instructions}" + (f"\n{self.context}" if self.context else "")


@dataclass
class WorkerStats:
    batches: int = 0
    llm_calls: int = 0
    tool_calls: int = 0
    candidates: int = 0
    dropped_evidence: int = 0
    dropped_candidates: int = 0
    dropped_log: list[str] = field(default_factory=list)  # each dropped candidate and the evidence that could not be found
    errors: list[str] = field(default_factory=list)


class WorkerRunner:
    def __init__(self, llm: Any, store: TableStore, toolbox: ToolBox, workers: int = 1, max_rounds: int = 2, max_calls: int = 6,
                 batch_chars: int = 14000, tasks_per_batch: dict[str, int] | None = None, should_stop: Callable[[], bool] | None = None,
                 on_progress: Callable[[str], None] | None = None):
        self.llm, self.store, self.toolbox = llm, store, toolbox
        self.workers = max(1, workers)
        self.max_rounds = max(1, max_rounds)
        self.max_calls = max_calls
        self.batch_chars = batch_chars
        self.tasks_per_batch = tasks_per_batch or {"A": 5, "B": 2, "C": 3, "D": 4, "E": 4, "F": 3, "H": 3}
        self.should_stop = should_stop or (lambda: False)
        self.on_progress = on_progress or (lambda _m: None)
        self.stats = WorkerStats()
        self.trace: list[dict] = []  # per batch and round: the plan, the tool results and what the model concluded (written next to the report)
        self._lock = threading.Lock()  # batches run on several threads and share the counters

    # -- batching --------------------------------------------------------------------------------------------------------------------
    def batches(self, tasks: list[WorkerTask]) -> list[list[WorkerTask]]:
        out: list[list[WorkerTask]] = []
        for family in sorted({t.family for t in tasks}):
            limit = self.tasks_per_batch.get(family, 3)
            current: list[WorkerTask] = []
            size = 0
            for t in (x for x in tasks if x.family == family):
                n = len(t.text())
                if current and (len(current) >= limit or size + n > self.batch_chars):
                    out.append(current)
                    current, size = [], 0
                current.append(t)
                size += n
            if current:
                out.append(current)
        return out

    def run(self, tasks: list[WorkerTask]) -> tuple[list[Candidate], list[TaskRecord]]:
        batches = self.batches(tasks)
        self.stats.batches = len(batches)
        results: list[tuple[list[Candidate], list[TaskRecord]]] = []
        if self.workers <= 1 or len(batches) <= 1:
            for i, b in enumerate(batches, start=1):
                results.append(self._batch(b))
                self.on_progress(f"investigate: batch {i} of {len(batches)} done ({len(b)} tasks, {sum(len(r[0]) for r in results)} candidates so far)")
        else:
            with ThreadPoolExecutor(max_workers=self.workers) as pool:
                futures = [pool.submit(self._batch, b) for b in batches]
                for i, f in enumerate(futures, start=1):
                    results.append(f.result())
                    self.on_progress(f"investigate: batch {i} of {len(batches)} done ({sum(len(r[0]) for r in results)} candidates so far)")
        candidates = [c for r in results for c in r[0]]
        records = [r for rs in results for r in rs[1]]
        self.stats.candidates = len(candidates)
        return candidates, records

    # -- one batch ------------------------------------------------------------------------------------------------------------------
    def _schema_for(self, tasks: list[WorkerTask]) -> str:
        wanted = {t for task in tasks if task.tables for t in task.tables}
        return self.store.schema_text(max_chars=7000, only=wanted or None)

    def _prompt(self, tasks: list[WorkerTask], schema: str, history: list[tuple[str, str]], conclude: bool) -> str:
        parts = ["Tasks:\n\n" + "\n\n".join(t.text() for t in tasks), "Tables:\n" + schema]
        if history:
            parts.append("Tool results so far:\n" + "\n\n".join(f"[{label}]\n{text}" for label, text in history))
        parts.append("Decide each task now." if conclude else "Plan the tool calls.")
        return "\n\n".join(parts)

    def _batch(self, tasks: list[WorkerTask]) -> tuple[list[Candidate], list[TaskRecord]]:
        records = {t.task_id: TaskRecord(t.task_id, t.family, t.title) for t in tasks}
        by_id = {t.task_id: t for t in tasks}
        if self.should_stop():
            for r in records.values():
                r.status, r.reason = "skipped", "stop requested"
            return [], list(records.values())
        schema = self._schema_for(tasks)
        history: list[tuple[str, str]] = []
        pending = list(tasks)
        candidates: list[Candidate] = []
        try:
            for round_no in range(1, self.max_rounds + 1):
                if self.should_stop():
                    break
                plan = self.llm.generate_structured_output(self._prompt(pending, schema, history, conclude=False), WorkerPlan, system=PLAN_SYSTEM)
                with self._lock:
                    self.stats.llm_calls += 1
                for call in plan.calls[:self.max_calls]:
                    text = self.toolbox.run(call)
                    with self._lock:
                        self.stats.tool_calls += 1
                    records[call.task_id if call.task_id in records else pending[0].task_id].calls += 1
                    history.append((f"round {round_no}, task {call.task_id}, {_describe(call)}", text))
                last_round = round_no == self.max_rounds
                outcome = self.llm.generate_structured_output(self._prompt(pending, schema, history, conclude=True), WorkerOutcome, system=CONCLUDE_SYSTEM)
                with self._lock:
                    self.stats.llm_calls += 1
                with self._lock:
                    self.trace.append({"tasks": [t.task_id for t in tasks], "round": round_no, "plan": [c.model_dump() for c in plan.calls],
                                       "tool_results": [{"call": label, "result": text[:700]} for label, text in history if label.startswith(f"round {round_no},")],
                                       "outcome": outcome.model_dump()})
                still: list[WorkerTask] = []
                answered = {r.task_id for r in outcome.results}
                for res in outcome.results:
                    task, rec = by_id.get(res.task_id), records.get(res.task_id)
                    if task is None or rec is None:
                        continue
                    if res.status == "needs_more" and not last_round:
                        still.append(task)
                        continue
                    rec.reason = res.reason[:300]
                    made = [c for c in (self._candidate(task, c) for c in res.candidates) if c is not None]
                    candidates += made
                    rec.candidates += len(made)
                    rec.status = "finding" if made else "passed" if res.status == "passed" else "untestable"
                    if res.status == "finding" and not made:
                        rec.reason = (rec.reason + " (its candidates had no evidence that could be found in the documents)").strip()
                for t in pending:  # a task the model did not answer is not a pass: it is asked again while rounds remain
                    if t.task_id not in answered:
                        if not last_round:
                            still.append(t)
                        else:
                            records[t.task_id].status, records[t.task_id].reason = "untestable", "the worker returned no result for this task"
                pending = still
                if not pending:
                    break
        except Exception as exc:
            with self._lock:
                self.trace.append({"tasks": [t.task_id for t in tasks], "error": f"{type(exc).__name__}: {str(exc)[:600]}"})
            self.stats.errors.append(f"{tasks[0].family} batch: {type(exc).__name__}: {str(exc)[:300]}")
            for t in pending:
                if records[t.task_id].status == "pending":
                    records[t.task_id].status, records[t.task_id].reason = "error", f"{type(exc).__name__}: {str(exc)[:160]}"
        for r in records.values():
            if r.status == "pending":
                r.status, r.reason = "untestable", r.reason or "no result after the allowed rounds"
        return candidates, list(records.values())

    # -- candidates ---------------------------------------------------------------------------------------------------------------------
    def _candidate(self, task: WorkerTask, out) -> Candidate | None:
        if out.type not in TAXONOMY:
            out = out.model_copy(update={"type": "direct_contradiction"})
        title, description = out.title.strip() or out.description.strip()[:120], out.description.strip() or out.title.strip()
        if not title:
            with self._lock:
                self.stats.dropped_candidates += 1
                self.stats.dropped_log.append(f"{task.task_id}: a candidate with no title and no description ({len(out.evidence)} evidence items)")
            return None
        out = out.model_copy(update={"title": title, "description": description})
        resolved, missing = [], []
        for ref in out.evidence:
            r = self.toolbox.resolve(ref)
            if r is None:
                with self._lock:
                    self.stats.dropped_evidence += 1
                missing.append((ref.quote or f"{ref.table} {ref.ref}" or "?")[:90])
                continue
            if all((r.chunk_id, r.quote) != (x.chunk_id, x.quote) for x in resolved):
                resolved.append(r)
        if not resolved:
            with self._lock:
                self.stats.dropped_candidates += 1
                self.stats.dropped_log.append(f"{task.task_id}: {out.title[:90]} | evidence not found: {'; '.join(missing) or 'none given'}")
            return None
        severity = out.severity if out.severity in SEVERITIES else "medium"
        status = out.status if out.status in ("definitive", "needs_investigation") else "definitive"
        return Candidate(family=task.family, type=out.type, title=out.title.strip()[:260], description=out.description.strip(), severity=severity, confidence=out.confidence,
                         status=status, evidence_needed=list(out.evidence_needed), evidence=resolved[:10], computed=out.computed, entities=list(out.entities),
                         root_cause=out.root_cause, task_id=task.task_id)


def _describe(call: ToolCall) -> str:
    detail = call.sql or call.query or call.entity or call.location or ""
    return f"{call.tool}({' '.join(detail.split())[:160]})"
