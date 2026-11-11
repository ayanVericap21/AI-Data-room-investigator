"""Compare Approach A (baseline) with Approach B (agentic system) on the same cases.

  ANTHROPIC_API_KEY=... python -m eval.run_eval --case eval/cases/sample_plant --runs 3 --out eval/results
  (Gemini: DRI_LLM_PROVIDER=gemini GEMINI_API_KEY=... DRI_MODEL=<gemini model> [DRI_PRICE_IN/OUT])

Both approaches get the same ingested data room, the same model, and the same scoring (eval/scoring.py).
Cost uses DRI_PRICE_IN / DRI_PRICE_OUT (USD per 1M tokens) plus the prompt-cache write/read multipliers; verify the
defaults match the model you run. Postgres runs use a throw-away schema; without DRI_DATABASE_URL a temp SQLite file.
"""
from __future__ import annotations

import argparse
import json
import tempfile
import time
import uuid
from pathlib import Path
from typing import Callable

from app import jobs
from app.agent import service, store
from app.agent.runner import InvestigationRunner
from app.baseline import run_baseline
from app.config import Settings, get_settings
from app.db import Database
from app.documents import save_upload
from app.llm import LLM, LLMError, create_llm
from app.mcp_server.client import LocalToolClient, MCPToolClient
from app.tools import DataRoomTools
from .scoring import GroundTruth, aggregate, score_run, to_pred


def isolated_db(settings: Settings, workdir: Path) -> Database:
    """A throw-away database: own Postgres schema when DRI_DATABASE_URL points at Postgres (so real data is never
    touched), otherwise a SQLite file in the work dir."""
    if settings.database_url and settings.database_url.startswith(("postgresql://", "postgres://")):
        return Database(settings.database_url, f"eval_{uuid.uuid4().hex[:12]}")
    return Database(f"sqlite:///{workdir / 'eval.sqlite3'}")


def prepare_room(case_dir: Path, workdir: Path, settings: Settings, db: Database,
                 llm_factory: Callable[[], LLM] | None = None) -> tuple[Settings, list[int], dict]:
    """Upload + ingest (incl. summaries when an LLM is given). Returns (settings, document ids, ingest usage)."""
    s = settings.model_copy(update={"data_dir": workdir})
    ids = [save_upload(db, s, p.name, p.read_bytes())[0].id for p in sorted((case_dir / "docs").iterdir()) if p.is_file()]
    job, _ = jobs.submit_job(db, ids, f"eval-{case_dir.name}")
    done = jobs.run_job(db, s, job.id, llm_factory)
    if done.status.value != "completed":
        raise RuntimeError(f"ingest failed: {done.error}")
    return s, ids, done.usage


def _usage(llm: LLM, s: Settings) -> dict:
    u = llm.usage
    return {"calls": u.calls, "input_tokens": u.input_tokens, "output_tokens": u.output_tokens,
            "cache_write_tokens": u.cache_write_tokens, "cache_read_tokens": u.cache_read_tokens,
            "cache_hit_ratio": u.cache_hit_ratio, "cost_usd": u.cost_usd(s)}


def evaluate(case_dir: Path, llm_factory: Callable[[], LLM], runs: int, approaches: list[str], settings: Settings,
             workdir: Path | None = None, use_mcp: bool = True) -> dict:
    truth = GroundTruth.model_validate_json((case_dir / "ground_truth.json").read_text())
    tmp = None if workdir else tempfile.TemporaryDirectory()
    work = workdir or Path(tmp.name)
    work.mkdir(parents=True, exist_ok=True)
    db = isolated_db(settings, work)
    try:
        s, ids, ingest_usage = prepare_room(case_dir, work, settings, db, llm_factory)
        chunk_doc = {r["id"]: r["filename"] for r in db.query("SELECT c.id, d.filename FROM chunks c JOIN documents d ON d.id=c.document_id")}
        result: dict = {"case": truth.case, "runs": runs, "model": s.model, "approaches": {},
                        "ingest_usage": ingest_usage}  # one-off summarisation cost; only the agent consumes summaries
        for approach in approaches:
            per_run = []
            for n in range(runs):
                llm = llm_factory()
                t0 = time.time()
                status, error, findings = "completed", None, []
                try:
                    if approach == "baseline":
                        res = run_baseline(db, s, llm, ids)
                        findings, error = res.findings, res.error
                        status = "failed" if res.error else "completed"
                    else:
                        inv = service.start_investigation(db, ids)
                        tools = MCPToolClient(db.url, db.schema, ids) if use_mcp else LocalToolClient(DataRoomTools(db, ids))
                        try:
                            InvestigationRunner(db, s, llm, tools, inv.id).run()
                        finally:
                            tools.close()
                        final = store.get(db, inv.id)
                        status, error = final.status, final.error
                        findings = store.findings(db, inv.id)
                except Exception as e:  # a crashed run is a result, not a reason to lose the others
                    status, error = "crashed", f"{type(e).__name__}: {e}"
                preds = [to_pred(f, chunk_doc, approach) for f in findings]
                run = {"run": n + 1, "status": status, "error": error, "wall_s": round(time.time() - t0, 1),
                       "usage": _usage(llm, s), "score": score_run(preds, truth),
                       "findings": [f.model_dump(mode="json") for f in findings]}
                per_run.append(run)
            result["approaches"][approach] = {
                "runs": per_run, "aggregate": aggregate([r["score"] for r in per_run]),
                "mean_wall_s": round(sum(r["wall_s"] for r in per_run) / len(per_run), 1),
                "mean_cost_usd": round(sum(r["usage"]["cost_usd"] for r in per_run) / len(per_run), 4),
                "mean_calls": round(sum(r["usage"]["calls"] for r in per_run) / len(per_run), 1),
                "mean_tokens": round(sum(r["usage"]["input_tokens"] + r["usage"]["output_tokens"] + r["usage"]["cache_write_tokens"]
                                         + r["usage"]["cache_read_tokens"] for r in per_run) / len(per_run)),
                "mean_cache_hit_ratio": round(sum(r["usage"]["cache_hit_ratio"] or 0 for r in per_run) / len(per_run), 3),
                "failed_runs": [r["run"] for r in per_run if r["status"] not in {"completed"}],
            }
        return result
    finally:
        db.drop_schema()  # Postgres: remove the throw-away schema
        if tmp:
            tmp.cleanup()


ROWS = [("Gold items found", "found_rate"), ("Classification accuracy", "classification_accuracy"),
        ("Contradiction recall", "contradiction_recall"), ("Contextual-difference accuracy", "contextual_accuracy"),
        ("Consistent accuracy", "consistent_accuracy"), ("Uncertainty handling (insufficient evidence)", "uncertainty_accuracy"),
        ("False-positive contradictions (count)", "false_positive_contradictions"),
        ("Overclaimed unverifiable items (count)", "overclaimed_unverifiable"),
        ("Evidence recall (verified citations)", "evidence_recall"), ("Citation validity", "citation_validity"),
        ("Findings reported (count)", "n_findings"), ("Unmatched findings (count, review manually)", "n_unmatched"),
        ("Repeatability across runs", "repeatability")]


def render_markdown(result: dict) -> str:
    names = list(result["approaches"])
    out = [f"# Baseline vs agentic system - case `{result['case']}`", "", f"Model: `{result['model']}`; runs per approach: {result['runs']}.",
           "Cost figures use the configured per-token rates and cache multipliers (see DRI_PRICE_IN/OUT); approximate.", "",
           "| Metric | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    for label, key in ROWS:
        out.append(f"| {label} | " + " | ".join("n/a" if result["approaches"][n]["aggregate"].get(key) is None
                                                  else str(result["approaches"][n]["aggregate"][key]) for n in names) + " |")
    for label, key in [("Mean latency per run (s)", "mean_wall_s"), ("Mean LLM calls per run", "mean_calls"),
                       ("Mean tokens per run (incl. cache)", "mean_tokens"),
                       ("Mean prompt-cache hit ratio", "mean_cache_hit_ratio"),
                       ("Mean cost per run (USD, approx.)", "mean_cost_usd")]:
        out.append(f"| {label} | " + " | ".join(str(result["approaches"][n][key]) for n in names) + " |")
    iu = result.get("ingest_usage") or {}
    if iu:
        out += ["", f"One-off ingest (document summaries, used only by the agent): {iu.get('calls', 0)} LLM calls, "
                    f"≈ ${iu.get('cost_usd', 0)} - not included in the per-run costs above."]
    out += ["", "## Failures and unmatched findings"]
    for n in names:
        a = result["approaches"][n]
        out.append(f"- **{n}**: failed/crashed runs: {a['failed_runs'] or 'none'}")
        for r in a["runs"]:
            if r["error"]:
                out.append(f"  - run {r['run']} error: {r['error']}")
            if r["score"]["unmatched_titles"]:
                out.append(f"  - run {r['run']} unmatched findings: {r['score']['unmatched_titles']}")
    out += ["", "## Per-item predictions (run 1)", "| Gold item | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    for gid in result["approaches"][names[0]]["runs"][0]["score"]["items"]:
        out.append(f"| {gid} | " + " | ".join(str(result["approaches"][n]["runs"][0]["score"]["items"].get(gid)) for n in names) + " |")
    return "\n".join(out) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True, type=Path)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--approaches", default="baseline,agent")
    ap.add_argument("--out", type=Path, default=Path("eval/results"))
    ap.add_argument("--local-tools", action="store_true", help="in-process tools instead of the MCP server")
    a = ap.parse_args()
    settings = get_settings()
    try:
        create_llm(settings)  # fail fast with a clear message (missing key / model) before any work
    except LLMError as e:
        raise SystemExit(f"cannot run a live evaluation: {e}")
    res = evaluate(a.case, lambda: create_llm(settings), a.runs, a.approaches.split(","), settings, use_mcp=not a.local_tools)
    a.out.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    (a.out / f"{a.case.name}-{stamp}.json").write_text(json.dumps(res, indent=2))
    md = render_markdown(res)
    (a.out / f"{a.case.name}-{stamp}.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
