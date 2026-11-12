"""The investigator pipeline: tables and a coverage gate, typed facts, the case model, a plan of checks, workers, the verifier, consolidation, saving.

Stage 0 (scanning and parsing the files) is done before this runs, by the existing document service. Everything here reads the stored chunks.
"""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.investigator import families as fam
from app.investigator.case_model import CaseModelBuilder, CaseResult
from app.investigator.facts import FactExtractor, FactStats
from app.investigator.models import Candidate, Fact, ResolvedEvidence, TaskRecord
from app.investigator.output import Consolidator, ConsolidationStats, Saver, SavedItem
from app.investigator.tables import DocCoverage, TableStore
from app.investigator.tools import ToolBox, build_timeline
from app.investigator.verifier import VerifyResult, Verifier
from app.investigator.workers import WorkerRunner, WorkerStats
from app.models import Document, Evidence, Finding
from app.models.base import ProcessingStatus
from app.services.llm_pairing import LLMPairer
from app.services.missing_items import MissingItemChecker
from app.services.pair_judge import PairJudge
from app.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

I_CLASSES = {"contradiction": "direct_contradiction", "contextual_difference": "contextual_difference", "missing_information": "missing_evidence"}


class RunStopped(Exception):
    pass


class CoverageGateError(Exception):
    def __init__(self, failures: list[DocCoverage]):
        super().__init__("; ".join(f"{c.filename}: {c.reason()}" for c in failures))
        self.failures = failures


@dataclass
class Options:
    workers: int = 1
    max_tasks: int = 60  # per family
    families: str = "ABCDEFGHI"
    verify: bool = True
    gate: str = "fail"  # fail | warn
    worker_rounds: int = 2
    pairing_batch_chars: int = 60000
    judge_batch_chars: int = 12000
    verifier_batch: int = 4


@dataclass
class RunOutcome:
    store: TableStore | None = None
    coverage: dict[str, DocCoverage] = field(default_factory=dict)
    fact_stats: FactStats = field(default_factory=FactStats)
    facts: list[Fact] = field(default_factory=list)
    case: CaseResult | None = None
    events: int = 0
    plan: fam.Plan = field(default_factory=fam.Plan)
    records: list[TaskRecord] = field(default_factory=list)
    worker_stats: WorkerStats = field(default_factory=WorkerStats)
    pairing_note: str = ""
    pairs: int = 0
    judge_stats: Any = None
    pairing_dropped: list[str] = field(default_factory=list)
    candidates: list[Candidate] = field(default_factory=list)
    verify: VerifyResult = field(default_factory=VerifyResult)
    consolidation: ConsolidationStats = field(default_factory=ConsolidationStats)
    saved: list[SavedItem] = field(default_factory=list)
    save_rejections: list[str] = field(default_factory=list)
    stage_ms: dict[str, int] = field(default_factory=dict)
    llm_marks: dict[str, tuple[int, int]] = field(default_factory=dict)
    gate_failures: list[DocCoverage] = field(default_factory=list)
    case_path: Path | None = None
    worker_trace: list = field(default_factory=list)


class InvestigatorPipeline:
    def __init__(self, session: Session, llm: Any, embedder: Any, registry: ToolRegistry, investigation_id: str, data_room_dir: Path, store_path: Path,
                 extract_client: Any, options: Options | None = None, on_progress: Callable[[str], None] | None = None,
                 should_stop: Callable[[], bool] | None = None, case_path: Path | None = None):
        self.session, self.llm, self.embedder, self.registry = session, llm, embedder, registry
        self.investigation_id = investigation_id
        self.root = data_room_dir
        self.store_path = store_path
        self.client = extract_client
        self.opt = options or Options()
        self.say = on_progress or (lambda _m: None)
        self.stopped = should_stop or (lambda: False)
        self.case_path = case_path
        self.out = RunOutcome()

    # -- helpers ---------------------------------------------------------------------------------------------------------------------
    def _check(self) -> None:
        if self.stopped():
            raise RunStopped()

    def _timed(self, name: str):
        pipeline = self

        class _Timer:
            def __enter__(self_inner):
                self_inner.t = time.perf_counter()
                self_inner.m = len(pipeline.llm.call_metrics)
                return self_inner

            def __exit__(self_inner, *exc):
                pipeline.out.stage_ms[name] = pipeline.out.stage_ms.get(name, 0) + round((time.perf_counter() - self_inner.t) * 1000)
                a, b = pipeline.out.llm_marks.get(name, (self_inner.m, self_inner.m))
                pipeline.out.llm_marks[name] = (min(a, self_inner.m), len(pipeline.llm.call_metrics))
                return False

        return _Timer()

    # -- the run ---------------------------------------------------------------------------------------------------------------------
    def run(self, selected: set[str] | None = None) -> RunOutcome:
        out = self.out
        docs = [d for d in self.session.scalars(select(Document).order_by(Document.filename))
                if d.processing_status == ProcessingStatus.processed.value and (selected is None or d.filename in selected)]
        names = {d.id: d.filename for d in docs}

        # 1. tables
        self._check()
        with self._timed("1. load tables (SQLite), keep cell addresses"):
            store = out.store = TableStore(self.store_path)
            out.coverage = store.build(self.session, only=set(names.values()))
        self.say(f"tables: {len(store.table_list())} tables, {sum(t.rows for t in store.table_list())} rows from {len(docs)} documents")

        # 2. typed facts
        self._check()
        with self._timed("2. extract typed facts"):
            extractor = FactExtractor(self.session, self.client, self.root, store, self.llm, workers=self.opt.workers, on_progress=lambda m: self.say("facts: " + m),
                                      should_stop=self.stopped)
            out.fact_stats = extractor.run(docs, out.coverage)
            out.facts = extractor.facts
        failures = [c for c in out.coverage.values() if not c.ok]
        for f in out.fact_stats.failed:
            self.say("facts: FAILED " + f)
        if failures:
            out.gate_failures = failures
            for c in failures:
                self.say(f"GATE: {c.filename} produced no facts and no tables although {c.reason()}")
            if self.opt.gate == "fail":
                raise CoverageGateError(failures)
        self.say(f"facts: {len(out.facts)} facts ({dict(out.fact_stats.by_type)}), {out.fact_stats.claims} also stored as claims")

        # 3. case model
        self._check()
        with self._timed("3. case model, identifiers, aliases, timeline"):
            builder = CaseModelBuilder(store, self.llm, on_progress=lambda m: self.say("case: " + m))
            out.case = builder.build()
            mentions = builder.scan(out.case)
            if self.case_path is not None:
                builder.save(out.case, self.case_path)
                out.case_path = self.case_path
            out.events = build_timeline(store, out.facts)
        self.say(f"case model: {len(out.case.model.roles)} roles, {len(out.case.patterns)} identifier kinds, {len(out.case.model.aliases)} alias groups, "
                 f"{mentions} mentions, {out.events} dated events")
        toolbox = ToolBox(store, out.facts)

        # 4. plan
        self._check()
        with self._timed("4. plan the checks"):
            plan = out.plan = self._plan(store, out, docs, names)
        counts = {f: sum(t.family == f for t in plan.tasks) for f in sorted({t.family for t in plan.tasks})}
        self.say(f"plan: {len(plan.tasks)} worker tasks {counts}, {len(plan.candidates)} candidates from code checks")

        # 5. investigate
        self._check()
        with self._timed("5a. workers investigate"):
            runner = WorkerRunner(self.llm, store, toolbox, workers=self.opt.workers, max_rounds=self.opt.worker_rounds, should_stop=self.stopped,
                                  on_progress=lambda m: self.say(m))
            worked, records = runner.run(plan.tasks)
            out.worker_stats = runner.stats
            out.worker_trace = runner.trace
        out.records = plan.records + records
        candidates = plan.candidates + worked
        if "I" in self.opt.families:
            self._check()
            with self._timed("5b. semantic value comparison (family I)"):
                candidates += self._family_i(store)
        out.candidates = candidates
        self.say(f"investigate: {len(candidates)} candidates ({len(plan.candidates)} from code, {len(worked)} from workers)")

        # 6. verify
        self._check()
        with self._timed("6. verify adversarially"):
            if self.opt.verify and candidates:
                out.verify = Verifier(self.llm, store, workers=self.opt.workers, batch_size=self.opt.verifier_batch, should_stop=self.stopped,
                                      on_progress=lambda m: self.say(m)).verify(candidates)
            else:
                out.verify = VerifyResult(kept=list(candidates))
        for c in out.verify.rejected:
            if c.finding_id:
                Saver(self.session, self.registry).remove(c.finding_id)
        self.say(f"verify: {len(out.verify.kept)} kept, {len(out.verify.rejected)} rejected {out.verify.stats.by_verdict}")

        # 7. consolidate and save
        self._check()
        with self._timed("7. consolidate by root cause and save"):
            final, out.consolidation = Consolidator(self.llm, on_progress=lambda m: self.say(m)).run(out.verify.kept)
            saver = Saver(self.session, self.registry)
            out.saved = saver.save(final)
            out.save_rejections = saver.rejected
        self.say(f"findings: {len(out.saved)} saved ({out.consolidation.before} candidates, {out.consolidation.groups} merged groups, {len(out.save_rejections)} rejected on saving)")
        return out

    # -- planning ----------------------------------------------------------------------------------------------------------------------
    def _plan(self, store: TableStore, out: RunOutcome, docs: list[Document], names: dict[str, str]) -> fam.Plan:
        plan, fs, n = fam.Plan(), self.opt.families, self.opt.max_tasks
        if "A" in fs:
            fam.family_a(out.facts, store, plan, n)
        if "B" in fs:
            fam.family_b(store, out.case, plan, n)
        if "C" in fs:
            fam.family_c(out.facts, store, plan, n)
        if "D" in fs:
            fam.family_d(out.facts, store, plan, n)
        if "E" in fs:
            fam.family_e(out.facts, store, plan, n)
        if "F" in fs:
            fam.family_f(store, plan, n)
        if "G" in fs:
            pool = MissingItemChecker(self.session, self.registry)._pool(names)
            fam.family_g(out.facts, store, out.case, pool, self.llm, plan, n)
        if "H" in fs:
            fam.family_h(out.facts, store, plan, n)
        return plan

    # -- family I: the existing pairing and judging, over observation and assertion facts stored as claims ------------------------------------
    def _family_i(self, store: TableStore) -> list[Candidate]:
        out = self.out
        pairs, pstats = LLMPairer(self.session, self.llm, max_chars_per_call=self.opt.pairing_batch_chars, should_stop=self.stopped, embedder=self.embedder,
                                  workers=self.opt.workers, on_progress=lambda m: self.say("match: " + m)).run()
        out.pairs = len(pairs)
        out.pairing_note = (f"{pstats.claims} claims in {pstats.batches} batch(es), {pstats.groups} groups covering {pstats.claims_grouped} claims, "
                            f"{pstats.pairs} pairs ({pstats.neighbor_pairs} from nearest neighbours)")
        self._check()
        judge = PairJudge(self.session, self.llm, self.registry, max_chars_per_call=self.opt.judge_batch_chars, should_stop=self.stopped, workers=self.opt.workers,
                          on_progress=lambda m, d: self.say("judge: " + m))
        out.judge_stats = judge.run(pairs)
        found: list[Candidate] = []
        for f in self.session.scalars(select(Finding).where(Finding.investigation_id == self.investigation_id)):
            kind = I_CLASSES.get(f.classification)
            if kind is None:
                out.pairing_dropped.append(f"{f.classification}: {f.title[:90]}")
                Saver(self.session, self.registry).remove(f.id)
                continue
            ev = []
            for e in self.session.scalars(select(Evidence).where(Evidence.finding_id == f.id)):
                info = store.chunk(e.chunk_id)
                if info:
                    ev.append(ResolvedEvidence(e.chunk_id, info[0], info[1], info[2], e.quoted_text, e.evidence_role, e.relevance_reason))
            if not ev:
                continue
            found.append(Candidate("I", kind, f.title, f.description, f.severity, f.confidence, "definitive", [], ev, source="pairing", finding_id=f.id, task_id="I"))
        out.records.append(TaskRecord("I", "I", f"pairing and judging of claims: {out.pairing_note}", "finding" if found else "passed",
                                      f"{len(found)} candidates; {len(out.pairing_dropped)} consistent or insufficient-evidence results not reported", 0, len(found)))
        return found
