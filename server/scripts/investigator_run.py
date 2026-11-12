"""Investigator run (pipeline v2): documents are loaded as tables and typed facts, a case model links them, families of checks are run by workers
and code, and an adversarial verifier decides what is reported.

    scripts/investigator_run.py [--only FILE ...] [--judge-workers N] [--stop-file PATH]

Writes ONE file, ../test_results/TEST-RUN-<date>-<time>-<config>.txt: the report, with the metrics, findings, case model, statistics and worker trace as JSON blocks at its end.
Exit code: 0 completed or stopped, 1 error, 3 the coverage gate failed (a document produced no facts and no tables although it has numbers).
All model calls go through LlamaExtract (one provider); keys come from the environment only.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
import time
import traceback
from datetime import datetime
from pathlib import Path

SERVER_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = SERVER_DIR.parent
sys.path.insert(0, str(SERVER_DIR))

from sqlalchemy import create_engine, delete  # noqa: E402
from sqlalchemy.engine import make_url  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402

from app.core.config import get_settings  # noqa: E402
from app.investigator.pipeline import CoverageGateError, InvestigatorPipeline, Options, RunOutcome, RunStopped  # noqa: E402
from app.investigator.report import RunInfo, findings_rows, metrics_dict, render, run_metrics_model  # noqa: E402
from app.llm.factory import get_embedding_provider, get_llm_client  # noqa: E402
from app.llm.llama_extract import LlamaExtractClient  # noqa: E402
from app.llm.local import DEFAULT_LOCAL_MODEL  # noqa: E402
from app.models import Claim, Investigation  # noqa: E402
from app.models.base import InvestigationStatus, utcnow  # noqa: E402
from app.services.document_service import DocumentService  # noqa: E402
from app.services.run_metrics import write_metrics  # noqa: E402
from app.services.processing_service import ProcessingService  # noqa: E402
from app.tools import build_registry  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db-name", default=None, help="database to run in (default: the database the app is configured for)")
    ap.add_argument("--config", default="investigator")
    ap.add_argument("--only", action="append", default=None, help="read only this document (path relative to the data room); repeatable")
    ap.add_argument("--judge-workers", "--workers", dest="workers", type=int, default=1, help="model calls made at the same time (extraction, workers, verifier, pairing)")
    ap.add_argument("--max-tasks", type=int, default=60, help="worker tasks per check family")
    ap.add_argument("--families", default="ABCDEFGHI", help="check families to run, for example ABD")
    ap.add_argument("--no-verify", action="store_true", help="skip the adversarial verifier (every candidate is reported)")
    ap.add_argument("--gate", choices=["fail", "warn"], default="fail", help="fail = stop when a document produced no facts and no tables (default)")
    ap.add_argument("--worker-rounds", type=int, default=2)
    ap.add_argument("--llm-attempts", type=int, default=5)
    ap.add_argument("--stop-file", default=None, help="when this file exists the run stops at the next safe point and still writes its results")
    args = ap.parse_args()

    settings = get_settings().model_copy(update=dict(embedding_provider="local", embedding_model=DEFAULT_LOCAL_MODEL, embedding_dimensions=384))
    args.db_name = args.db_name or make_url(settings.database_url).database
    started = datetime.now()
    stamp = started.strftime("%Y%m%d-%H%M%S")
    config = re.sub(r"[^A-Za-z0-9._-]+", "-", args.config)
    path = REPO_DIR / "test_results" / f"TEST-RUN-{stamp}-{config}.txt"
    path.parent.mkdir(exist_ok=True)
    print(f"[run] pipeline=investigator v2\n[run] database={args.db_name}\n[run] results file={path}", flush=True)

    def stopped() -> bool:
        return bool(args.stop_file) and os.path.exists(args.stop_file)

    llm = get_llm_client(settings)
    embedder = get_embedding_provider(settings)
    llm.max_attempts = embedder.max_attempts = args.llm_attempts
    engine = create_engine(make_url(settings.database_url).set(database=args.db_name))
    session = sessionmaker(bind=engine, expire_on_commit=False)()
    cleared = session.execute(delete(Claim)).rowcount or 0
    session.commit()
    print(f"[run] deleted {cleared} existing claims; they will be rebuilt", flush=True)

    investigation = Investigation(name=f"investigator run {stamp}", status=InvestigationStatus.running.value, started_at=utcnow(),
                                  execution_config={"approach": "investigator_v2", "llm_model": settings.llm_model, "workers": args.workers, "families": args.families})
    session.add(investigation)
    session.commit()
    outcome: RunOutcome = RunOutcome()
    error, was_stopped, gate_failed = None, False, False
    scan_note = ""
    t_run = time.perf_counter()
    work_dir = Path(tempfile.mkdtemp(prefix="investigator_"))
    try:
        selected = None if args.only is None else {x.replace(chr(92), "/") for x in args.only}  # unselected files are not touched at all
        t = time.perf_counter()
        doc_service = DocumentService(session, settings)
        scan = doc_service.scan(only=selected)
        ProcessingService(session, doc_service, embedder, on_progress=lambda m, d: print(f"[process] {m}", flush=True)).process_pending(only=selected)
        scan_note = f"{scan.added} new, {scan.updated} updated, {scan.unchanged} unchanged; scan and parsing {time.perf_counter() - t:.0f}s"
        print(f"[run] scan: {scan_note}", flush=True)
        if not settings.allow_third_party_apis:
            raise RuntimeError("the investigator uploads documents to LlamaExtract (a third-party API), which is disabled (set ALLOW_THIRD_PARTY_APIS=true to allow it)")
        if not settings.llama_cloud_api_key:
            raise RuntimeError("the investigator needs LLAMA_CLOUD_API_KEY in the environment")
        client = LlamaExtractClient(settings.llama_cloud_api_key, project_id=settings.llama_cloud_project_id, tier="cost_effective")
        options = Options(workers=args.workers, max_tasks=args.max_tasks, families=args.families.upper(), verify=not args.no_verify, gate=args.gate, worker_rounds=args.worker_rounds)
        registry = build_registry(session, None, investigation.id)
        pipeline = InvestigatorPipeline(session, llm, embedder, registry, investigation.id, settings.data_room_dir, work_dir / "tables.sqlite", client, options,
                                        on_progress=lambda m: print(f"[{m.split(':', 1)[0]}] {m.split(':', 1)[1].strip() if ':' in m else m}", flush=True),
                                        should_stop=stopped, case_path=None)
        outcome = pipeline.out
        outcome = pipeline.run(selected)
    except RunStopped:
        was_stopped = True
        print("[run] STOP requested: writing the results with what was done so far", flush=True)
    except CoverageGateError as exc:
        gate_failed = True
        error = "COVERAGE GATE FAILED: " + str(exc)
        print("[run] " + error, flush=True)
    except Exception:
        error = traceback.format_exc()
        print("[run] ERROR\n" + error, flush=True)

    wall = time.perf_counter() - t_run
    try:
        inv = session.get(Investigation, investigation.id)
        inv.status = (InvestigationStatus.failed if error else InvestigationStatus.stopped if was_stopped else InvestigationStatus.completed).value
        inv.completed_at, inv.duration_ms = utcnow(), round(wall * 1000)
        session.commit()
    except Exception:
        session.rollback()
    info = RunInfo(run_id=stamp, config=config, db_name=args.db_name, args={k: v for k, v in vars(args).items() if k not in ("stop_file",)}, started=started.strftime("%Y-%m-%d %H:%M:%S"),
                   wall_s=wall, outcome=outcome, llm=llm, investigation_id=investigation.id, session=session, error=error, stopped=was_stopped, gate_failed=gate_failed, scan_note=scan_note, results_path=str(path),
                   finished=datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
                   llm_model=settings.llm_model, embedding_model=settings.embedding_model, embedder_metrics=list(embedder.call_metrics))
    try:
        path.write_text(render(info), encoding="utf-8")  # one file: the report, then its appendices (metrics, findings, case model, statistics, worker trace) as JSON blocks
    except Exception:
        print("[run] could not write the results files:\n" + traceback.format_exc(), flush=True)
    if outcome.store is not None:
        try:
            outcome.store.close()
        except Exception:
            pass
    session.close()
    engine.dispose()
    print(f"[run] DONE in {wall:.0f}s. Results: {path}", flush=True)
    return 3 if gate_failed else 1 if error else 0


if __name__ == "__main__":
    sys.exit(main())
