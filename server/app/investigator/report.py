"""The run report (text), the findings file (JSON, read by the scoring harness) and the metrics file (JSON)."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.investigator.models import FAMILIES, TAXONOMY
from app.investigator.pipeline import RunOutcome
from app.models import Evidence, Finding

BAR = "=" * 78


@dataclass
class RunInfo:
    run_id: str
    config: str
    db_name: str
    args: dict[str, Any]
    started: str
    wall_s: float
    outcome: RunOutcome
    llm: Any
    investigation_id: str
    session: Session
    error: str | None = None
    stopped: bool = False
    gate_failed: bool = False
    scan_note: str = ""
    llm_model: str = ""
    embedding_model: str = ""
    results_path: str = ""
    finished: str = ""
    embedder_metrics: list = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def clip(text: Any, n: int) -> str:
    s = " ".join(str(text).split())
    return s if len(s) <= n else s[: n - 1] + "…"


def table(headers: list[str], rows: list[list[Any]], max_width: int = 90) -> list[str]:
    cells = [[clip(c, max_width) for c in r] for r in rows]
    widths = [max([len(h)] + [len(r[i]) for r in cells]) for i, h in enumerate(headers)]
    line = lambda r: "  ".join(c.ljust(w) for c, w in zip(r, widths)).rstrip()
    return [line(headers), line(["-" * w for w in widths])] + [line(r) for r in cells]


def _llm_stage(info: RunInfo, name: str) -> tuple[int, int, float, float]:
    a, b = info.outcome.llm_marks.get(name, (0, 0))
    calls = info.llm.call_metrics[a:b]
    return len(calls), sum(not m.ok for m in calls), round(sum(m.latency_ms for m in calls) / 1000, 1), 0.0


def findings_rows(info: RunInfo) -> list[dict]:
    """Every saved finding with its evidence, taxonomy, family and status (what the scoring harness reads)."""
    out = info.outcome
    by_id = {s.finding_id: s.candidate for s in out.saved}
    rows = []
    for f in info.session.scalars(select(Finding).where(Finding.investigation_id == info.investigation_id).order_by(Finding.created_at)):
        c = by_id.get(f.id)
        ev = [{"quote": e.quoted_text, "role": e.evidence_role, "chunk_id": e.chunk_id, "reason": e.relevance_reason}
              for e in info.session.scalars(select(Evidence).where(Evidence.finding_id == f.id))]
        rows.append({"id": f.id, "title": f.title, "classification": f.classification, "taxonomy": c.type if c else None, "family": c.family if c else None,
                     "severity": f.severity, "confidence": f.confidence, "status": c.status if c else None, "description": f.description,
                     "unresolved_questions": list(f.unresolved_questions or []), "evidence": ev, "verdict": c.verdict if c else None})
    return rows


def metrics_dict(info: RunInfo) -> dict:
    out = info.outcome
    calls = info.llm.call_metrics
    findings = findings_rows(info)
    return {
        "run_id": info.run_id, "config": info.config, "wall_seconds": round(info.wall_s, 1), "args": info.args,
        "documents": len(out.coverage), "tables": len(out.store.table_list()) if out.store else 0, "table_rows": sum(t.rows for t in out.store.table_list()) if out.store else 0,
        "facts": len(out.facts), "facts_by_type": dict(out.fact_stats.by_type), "claims": out.fact_stats.claims, "facts_rejected": out.fact_stats.rejected,
        "worker_tasks": len(out.plan.tasks), "task_status": dict(Counter(r.status for r in out.records)),
        "candidates": len(out.candidates), "verdicts": out.verify.stats.by_verdict, "rejected_by_verifier": len(out.verify.rejected),
        "findings": len(findings), "findings_by_taxonomy": dict(Counter(f["taxonomy"] or "other" for f in findings)),
        "findings_by_family": dict(Counter(f["family"] or "?" for f in findings)), "findings_by_severity": dict(Counter(f["severity"] for f in findings)),
        "llm": {"attempts": len(calls), "failed": sum(not m.ok for m in calls), "seconds": round(sum(m.latency_ms for m in calls) / 1000, 1),
                "credits": round(getattr(info.llm, "credits", 0.0), 1)},
        "stages_seconds": {k: round(v / 1000, 1) for k, v in out.stage_ms.items()},
        "gate_failures": [c.filename for c in out.gate_failures],
    }


def render(info: RunInfo) -> str:
    out = info.outcome
    L: list[str] = [f"INVESTIGATOR RUN  {info.run_id}  {info.config}", BAR, ""]
    status = "RUN ERROR" if info.error else "COVERAGE GATE FAILED" if info.gate_failed else "STOPPED" if info.stopped else "completed"
    L += [f"Status        : {status}", f"Started       : {info.started}   wall time {info.wall_s:.0f}s", f"Database      : {info.db_name}", f"Pipeline      : investigator v2 (tables, typed facts, case model, check families, adversarial verifier)",
          f"Options       : {info.args}", f"Documents     : {len(out.coverage)} ({info.scan_note})"]
    findings = findings_rows(info)
    L += ["", BAR, "AT A GLANCE", BAR,
          f"Tables        : {len(out.store.table_list()) if out.store else 0} tables, {sum(t.rows for t in out.store.table_list()) if out.store else 0} rows",
          f"Facts         : {len(out.facts)} {dict(out.fact_stats.by_type)}; {out.fact_stats.rejected} rejected (quote not found); {out.fact_stats.claims} stored as claims for family I",
          f"Case model    : " + (f"{len(out.case.model.roles)} document roles, {len(out.case.patterns)} identifier kinds, {len(out.case.model.aliases)} alias groups, {out.case.mentions} mentions, {out.events} dated events" if out.case else "not built"),
          f"Plan          : {len(out.plan.tasks)} worker tasks, {len(out.plan.candidates)} candidates from code checks; task outcomes {dict(Counter(r.status for r in out.records))}",
          f"Candidates    : {len(out.candidates)}; verifier verdicts {out.verify.stats.by_verdict}; {len(out.verify.rejected)} rejected, {out.verify.stats.unverified} unverified",
          f"Findings      : {len(findings)} saved; by type {dict(Counter(f['taxonomy'] or 'other' for f in findings))}; by family {dict(Counter(f['family'] or '?' for f in findings))}; by severity {dict(Counter(f['severity'] for f in findings))}",
          f"LLM calls     : {len(info.llm.call_metrics)} attempts, {sum(not m.ok for m in info.llm.call_metrics)} failed, {sum(m.latency_ms for m in info.llm.call_metrics) / 1000:.0f}s in total, {getattr(info.llm, 'credits', 0.0):.0f} credits"]
    if out.gate_failures:
        L.append("GATE          : " + "; ".join(f"{c.filename} ({c.reason()})" for c in out.gate_failures))
    L += ["", BAR, "COVERAGE GATE (every document must produce facts or tables, or have no numbers)", BAR]
    L += table(["document", "chunks", "with numbers", "tables", "table rows", "facts", "claims", "gate"],
               [[c.filename, c.chunks, c.chunks_with_numbers, c.tables, c.table_rows, c.facts, c.claims, "ok" if c.ok else "FAILED"] for c in out.coverage.values()])
    if out.fact_stats.failed:
        L += ["", "Extraction failures:"] + [f"  {x}" for x in out.fact_stats.failed]
    if out.case:
        m = out.case.model
        L += ["", BAR, "CASE MODEL", BAR, f"Summary: {m.case_summary or '(none)'}", "", "Document roles:"]
        L += table(["document", "role", "summary"], [[r.document, r.role, r.summary] for r in m.roles])
        L += ["", "Identifier kinds (pattern, distinct identifiers found):"]
        with out.store._lock:
            counts = dict(out.store.conn.execute("SELECT entity_type, COUNT(DISTINCT ident) FROM _mentions GROUP BY entity_type").fetchall())
        L += table(["kind", "pattern", "distinct ids"], [[p.entity_type, p.regex, counts.get(p.entity_type, 0)] for p, _ in out.case.patterns])
        if m.aliases:
            L += ["", "Aliases:"] + [f"  {a.canonical} = {', '.join(a.aliases)}" for a in m.aliases[:30]]
        if m.definitions:
            L += ["", "Definitions:"] + [f"  {d.metric}: {clip(d.definition, 140)} ({d.document})" for d in m.definitions[:30]]
        if out.case.notes:
            L += ["", "Notes:"] + [f"  {n}" for n in out.case.notes]
    L += ["", BAR, "FINDINGS", BAR]
    if not findings:
        L.append("(none)")
    for i, f in enumerate(findings, start=1):
        L += ["", f"--- Finding {i} of {len(findings)} " + "-" * 40, f"title          : {f['title']}", f"classification : {f['classification']}   type: {f['taxonomy']}   family: {f['family']} {FAMILIES.get(f['family'] or '', '')}",
              f"severity       : {f['severity']}   confidence {f['confidence']:.2f}   status: {f['status']}   verifier: {f['verdict']}", "description    : " + f["description"].replace("\n", "\n                 ")]
        for q in f["unresolved_questions"]:
            L.append(f"needs          : {q}")
        for e in f["evidence"]:
            L.append(f"evidence ({e['role']}): {clip(e['quote'], 300)}   [{e['chunk_id']}]")
    L += ["", BAR, "CHECK PLAN AND AUDIT LOG (every task and how it ended)", BAR]
    by_family = Counter((r.family, r.status) for r in out.records)
    L += table(["family", "name", "finding", "passed", "untestable", "error", "skipped"],
               [[f, FAMILIES[f], *(by_family.get((f, s), 0) for s in ("finding", "passed", "untestable", "error", "skipped"))] for f in FAMILIES if any(k[0] == f for k in by_family)])
    L += [""] + table(["task", "fam", "status", "calls", "cand", "what", "why"], [[r.task_id, r.family, r.status, r.calls, r.candidates, r.title, r.reason] for r in out.records], 70)
    if out.plan.quiet:
        L += ["", "Found and deliberately not reported:"] + [f"  {q}" for q in out.plan.quiet[:60]]
    L += ["", BAR, "VERIFIER", BAR, f"{out.verify.stats.calls} calls; verdicts {out.verify.stats.by_verdict}; errors {out.verify.stats.errors}"]
    for c in out.verify.rejected:
        L.append(f"  rejected [{c.family}] {clip(c.title, 90)}: {clip(c.verifier_notes, 220)}")
    L += ["", BAR, "CONSOLIDATION", BAR, f"{out.consolidation.before} candidates became {out.consolidation.after} findings ({out.consolidation.exact_duplicates} exact duplicates, {out.consolidation.groups} root-cause groups)"]
    L += [f"  {x}" for x in out.consolidation.log]
    if out.pairing_note:
        L += ["", BAR, "FAMILY I (semantic value comparison)", BAR, out.pairing_note, f"not reported: {len(out.pairing_dropped)} consistent or insufficient-evidence results"]
        L += [f"  {x}" for x in out.pairing_dropped[:30]]
    L += ["", BAR, "WORKERS", BAR, f"{out.worker_stats.batches} batches, {out.worker_stats.llm_calls} model calls, {out.worker_stats.tool_calls} tool calls; "
          f"{out.worker_stats.dropped_candidates} candidates dropped (evidence not found or empty), {out.worker_stats.dropped_evidence} evidence items dropped"]
    L += [f"  {x}" for x in out.worker_stats.errors]
    if out.worker_stats.dropped_log:
        L += ["Candidates dropped because none of their evidence could be found in the documents:"] + [f"  {x}" for x in out.worker_stats.dropped_log[:40]]
    if out.save_rejections:
        L += ["", "Rejected when saving:"] + [f"  {x}" for x in out.save_rejections]
    L += ["", BAR, "TIMING AND LLM USE PER STAGE", BAR]
    rows = []
    for name, ms in out.stage_ms.items():
        n, failed, secs, _ = _llm_stage(info, name)
        rows.append([name, f"{ms / 1000:.1f}s", n, failed, f"{secs:.0f}s"])
    L += table(["stage", "wall", "llm attempts", "failed", "llm time"], rows)
    failed = [m for m in info.llm.call_metrics if not m.ok]
    if failed:
        L += ["", f"Failed LLM attempts ({len(failed)}):"] + [f"  {clip(m.error, 160)}" for m in failed[:20]]
    if info.error:
        L += ["", BAR, "Run error", BAR, info.error]
    L += _appendices(info, findings)
    return "\n".join(L) + "\n"


def _block(title: str, name: str, data: Any) -> list[str]:
    """A section holding JSON between <<<NAME and NAME>>> lines, so one results file carries everything and tools can read it back."""
    import json
    return ["", BAR, title, BAR, f"<<<{name}", json.dumps(data, indent=1, ensure_ascii=False, default=str), f"{name}>>>"]


def _appendices(info: RunInfo, findings: list[dict]) -> list[str]:
    out = info.outcome
    case = None
    if out.case is not None:
        case = out.case.model.model_dump()
        case["shapes_seen"], case["notes"] = out.case.shapes, out.case.notes
    L = _block("APPENDIX A: RUN METRICS (read by the run page)", "RUN_METRICS", run_metrics_model(info, info.results_path, info.finished or info.started).model_dump())
    L += _block("APPENDIX B: FINDINGS AS DATA (read by scripts/score_run.py)", "FINDINGS", findings)
    L += _block("APPENDIX C: CASE MODEL (what the run assumed about the room)", "CASE_MODEL", case)
    L += _block("APPENDIX D: RUN STATISTICS", "STATS", metrics_dict(info))
    L += _block("APPENDIX E: WORKER TRACE (each batch: the plan, the tool results, what the model concluded)", "WORKER_TRACE", out.worker_trace)
    return L


# ---- the metrics file the UI panel reads (same schema as the classic run) ----------------------------------------------------------------------
def run_metrics_model(info: RunInfo, results_path: Any, finished: str):
    import statistics
    from app.schemas.run_metrics import LLMTotals, PhaseMetrics, RunCounts, RunMetrics, RunSettings, StageTiming
    out = info.outcome
    calls = info.llm.call_metrics
    ok = [m for m in calls if m.ok]
    lat = info.llm.latency_summary()
    tokens = info.llm.get_token_usage()
    phases = []
    for name in out.stage_ms:
        a, b = out.llm_marks.get(name, (0, 0))
        items = calls[a:b]
        if not items:
            continue
        good = [m for m in items if m.ok]
        latencies = sorted(m.latency_ms for m in good)
        phases.append(PhaseMetrics(
            phase=name, attempts=len(items), failed=len(items) - len(good), prompt_tokens=sum(m.prompt_tokens for m in good), completion_tokens=sum(m.completion_tokens for m in good),
            thinking_tokens=sum(m.thinking_tokens for m in good), avg_context_tokens=round(statistics.mean(m.prompt_tokens for m in good)) if good else 0,
            max_context_tokens=max((m.prompt_tokens for m in good), default=0), avg_ms=round(statistics.mean(latencies)) if latencies else 0,
            p95_ms=latencies[min(len(latencies) - 1, int(.95 * len(latencies)))] if latencies else 0, total_s=round(sum(m.latency_ms for m in items) / 1000, 1)))
    pt = [m.prompt_tokens for m in ok]
    js = out.judge_stats
    emb = getattr(info, "embedder_metrics", []) or []
    counts = RunCounts(
        documents=len(out.coverage), documents_fully_read=sum(c.ok for c in out.coverage.values()), chunks=sum(c.chunks for c in out.coverage.values()),
        claims_recorded=out.fact_stats.claims, claims_rejected=out.fact_stats.rejected, candidate_pairs=out.pairs,
        extractor_credits=round(out.fact_stats.credits, 1), pairs_exact=getattr(js, "pairs_exact", 0), pairs_judged=getattr(js, "pairs_judged", 0),
        verdicts=dict(getattr(js, "verdict_counts", {}) or {}), findings_saved=len(out.saved), findings_rejected=len(out.save_rejections) + len(out.verify.rejected),
        embedding_calls=len(emb), embedding_texts=sum(getattr(m, "items", 0) for m in emb))
    a = info.args
    wall = info.wall_s or 1.0
    return RunMetrics(
        investigation_id=info.investigation_id, config_name=info.config, state="failed" if info.error else "stopped" if info.stopped else "completed",
        started_at=info.started, finished_at=finished, wall_s=round(info.wall_s, 1), results_file=str(results_path),
        settings=RunSettings(extractor="llamaextract (typed facts)", pairing="llm (family I) + check families", vocabulary="off", coverage_check=f"gate {a.get('gate', 'fail')}",
                             batch_chunks=None, batch_chars=int(a.get("max_tasks", 0) or 0), llm_model=getattr(info, "llm_model", ""), embedding_model=getattr(info, "embedding_model", "")),
        counts=counts,
        llm=LLMTotals(attempts=len(calls), failed=sum(not m.ok for m in calls), retries=getattr(info.llm, "retry_count", 0), total_s=round(sum(m.latency_ms for m in calls) / 1000, 1),
                      avg_ms=lat.avg_ms, p95_ms=lat.p95_ms, max_ms=lat.max_ms, prompt_tokens=tokens.prompt_tokens, completion_tokens=tokens.completion_tokens,
                      thinking_tokens=tokens.thinking_tokens, cached_tokens=tokens.cached_tokens, total_tokens=tokens.total_tokens,
                      credits=round(getattr(info.llm, "credits", 0.0) + out.fact_stats.credits, 1), min_context_tokens=min(pt, default=0),
                      avg_context_tokens=round(statistics.mean(pt)) if pt else 0, max_context_tokens=max(pt, default=0)),
        phases=phases, stages=[StageTiming(stage=k, seconds=round(v / 1000, 1), percent=round(100 * v / (wall * 1000), 1)) for k, v in out.stage_ms.items()])
