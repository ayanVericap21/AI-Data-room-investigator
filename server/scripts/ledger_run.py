"""Fact-ledger test run: every chunk is read by the LLM, claims go into the ledger (the claims table),
claims are matched across documents, conflicting pairs are judged with quotes verified in code, and every
document gets a verdict. Writes a NEW file: ../test_results/TEST-RUN-<date>-<time>-fact-ledger-....txt

Run from server/ on a database whose documents are already processed (see scripts/full_run.py):
    .venv/Scripts/python.exe -u scripts/ledger_run.py --db-name dataroom_run_20261111_011346 --llm-model gemini-3.5-flash-lite

Document text is sent to the configured LLM provider. No step, call or chunk limits are applied.
"""
from __future__ import annotations

import argparse
import fnmatch
import re
import statistics
import sys
import time
import traceback
from collections import Counter
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Literal

SERVER_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = SERVER_DIR.parent
sys.path.insert(0, str(SERVER_DIR))
sys.path.insert(0, str(SERVER_DIR / "scripts"))

from pydantic import BaseModel  # noqa: E402
from sqlalchemy import create_engine, delete, func, select, text  # noqa: E402
from sqlalchemy.engine import make_url  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402

import full_run as base  # noqa: E402  (shared report helpers)
from app.core.config import get_settings  # noqa: E402
from app.llm.factory import get_embedding_provider, get_llm_client  # noqa: E402
from app.llm.errors import LLMError  # noqa: E402
from app.llm.local import DEFAULT_LOCAL_MODEL  # noqa: E402
from app.models import Claim, Document, DocumentChunk, Finding, Investigation  # noqa: E402
from app.models.base import InvestigationStatus, ProcessingStatus, utcnow  # noqa: E402
from app.services.claim_service import EXCERPT_TYPES_SKIPPED, ClaimService, ExtractionSummary  # noqa: E402
from app.services.document_service import DocumentService  # noqa: E402
from app.services.processing_service import ProcessingService  # noqa: E402
from app.services.pair_judge import PairJudge  # noqa: E402
from app.tools import build_registry  # noqa: E402

table, clip, pct = base.table, base.clip, base.pct


class DocumentVerdict(BaseModel):
    verdict: Literal["consistent", "discrepancies_found", "partly_unverified", "nothing_to_compare"]
    reason: str


VERDICT_SYSTEM = """You write the verdict for one document after it was compared with the other documents of the same collection.
Use only the statistics and finding titles given. verdict is one of: consistent (its comparable claims agree with other documents), discrepancies_found (at least one discrepancy involves it), partly_unverified (a large share of its claims had nothing to compare against, or reading was incomplete), nothing_to_compare (no claims could be compared). reason is one short sentence."""


def phase_rows(R):
    """(phase, attempt) for every LLM provider attempt, in order."""
    bounds = R.llm_bounds
    names = ["extract claims", "judge pairs", "document verdicts"]
    for i, m in enumerate(R.llm.call_metrics):
        phase = names[0 if i < bounds[0] else 1 if i < bounds[1] else 2]
        yield phase, m


# ---- report sections ----------------------------------------------------------
def sec_how(R):
    a, s = R.args, R.settings
    return [
        f"1. Reading: every chunk of every processed document{' except ' + ', '.join(R.excluded) if R.excluded else ''} (database '{R.db_name}') was sent to the LLM in batches of up to {a.batch_chars:,} characters; each batch returns claims, kept only if their value appears in the cited chunk (existing claim extractor, no chunk or call limits).",
        f"2. Ledger: accepted claims are stored in the claims table; claims are matched across and within documents by wording and by local embeddings ({s.embedding_model}), without a fixed number of documents or claims assumed.",
        f"3. Judging: pairs that agree exactly count as corroboration; every other pair (conflicting, partial, contextual) goes to the LLM in batches of up to {a.judge_batch_chars:,} characters with both source excerpts.",
        "4. Findings: a discrepancy is saved only if its two quotes are exact substrings of their chunks (checked in code)"
        + ("; per-document verdicts were skipped by request." if a.skip_verdicts else "; then each document gets a verdict built from its counts."),
        f"5. Model: {s.llm_model}, temperature 0, up to {a.llm_attempts} attempts per call honouring Google's retry delay; no step, tool-call or finding limits.",
    ]


def sec_glance(R):
    ex, js, res = R.extraction, R.judge, R
    t = R.llm.get_token_usage()
    ok = [m for m in R.llm.call_metrics if m.ok]
    out = [f"Run started   : {R.started:%Y-%m-%d %H:%M:%S} (local)   finished: {R.finished:%H:%M:%S}   wall time: {R.wall_s:.1f}s",
           f"Config        : {R.config}", f"Database      : {R.db_name} (documents were already processed; reused)",
           f"Code version  : git {base.git_info()}   python {sys.version.split()[0]}"]
    if R.error:
        out.append("RUN ERROR     : see the end of this file")
    docs = list(R.doc_stats.values())
    if R.excluded:
        out.append(f"Excluded      : {', '.join(R.excluded)} (by request: not read, not compared)")
    out += [f"Documents     : {len(docs)}; fully read {sum(d['ok'] for d in docs)}, incomplete {sum(not d['ok'] for d in docs)}",
            f"Chunks        : {sum(d['chunks'] for d in docs)} total; sent to the LLM {sum(d['readable'] for d in docs if d['ok'])} "
            f"(headings are skipped by the extractor: {sum(d['chunks'] - d['readable'] for d in docs)})",
            f"Claims        : {ex.claims_created} recorded, {ex.claims_rejected} rejected (value not found in cited chunk or unknown chunk id)"]
    if R.pairs is not None:
        out.append(f"Pairs         : {len(R.pairs)} candidates; {js.pairs_exact} agreed exactly (not sent to the LLM); {js.pairs_judged} judged; "
                   f"verdicts {dict(js.verdict_counts)}")
        out.append(f"Findings      : {js.findings_saved} saved with verified quotes; {js.findings_rejected} rejected (quotes missing or not exact)")
    out += [f"LLM calls     : {len(R.llm.call_metrics)} provider attempts ({sum(not m.ok for m in R.llm.call_metrics)} failed, {R.llm.retry_count} retries incl. invalid-output retries), "
            f"{sum(m.latency_ms for m in R.llm.call_metrics) / 1000:.1f}s total, avg {R.llm.latency_summary().avg_ms}ms",
            f"Tokens        : prompt {t.prompt_tokens:,}  completion {t.completion_tokens:,}  thinking {t.thinking_tokens:,}  cached {t.cached_tokens:,}  total {t.total_tokens:,}   (cost not computed: no price per token configured)"]
    if ok:
        pt = [m.prompt_tokens for m in ok]
        out.append(f"Context/call  : prompt tokens min {min(pt):,}  avg {round(statistics.mean(pt)):,}  max {max(pt):,}")
    return out


def sec_new_files(R):
    sc = R.scan
    out = [f"claims deleted at the start of this run: {R.claims_cleared} (all claims in this run are newly extracted)",
           f"scan: discovered {sc.discovered}, new {sc.added}, updated {sc.updated}, unchanged {sc.unchanged}, missing {sc.missing}",
           f"      duplicates skipped (same content): {sc.skipped_duplicates or 'none'}", ""]
    if not R.processed:
        return out + ["no new or changed files had to be processed"]
    rows = [[r.filename[:70], r.status, r.chunks, f"{r.timings_ms.get('extract', 0) / 1000:.1f}", r.timings_ms.get("chunk", 0), r.timings_ms.get("save", 0),
             f"{r.timings_ms.get('embed', 0) / 1000:.1f}", clip(r.note or r.error or "", 60)] for r in R.processed]
    return out + table(["new or changed file", "status", "chunks", "extract s", "chunk ms", "save ms", "embed s", "note"], rows)


def sec_contradictions(R):
    s = R.session
    rows = list(s.scalars(select(Finding).where(Finding.investigation_id == R.result.investigation_id, Finding.classification == "contradiction")))
    if not rows:
        return ["No finding was classified as a contradiction in this run."]
    out = [f"{len(rows)} contradiction(s). Full evidence for each is in the findings report below (search for the finding id).", ""]
    for f in rows:
        out.append(f"- [{f.id}] severity {f.severity}, confidence {f.confidence}: {f.title}")
        out.append(f"    {clip(f.description, 220)}")
    return out


def sec_documents(R):
    rows = []
    for d in sorted(R.doc_stats.values(), key=lambda x: x["file"]):
        rows.append([d["file"][:58], d["chunks"], d["readable"] if d["ok"] else 0, "yes" if d["ok"] else "NO", d["claims"], d["calls"],
                     f"{d['total_tokens']:,}", f"{d['ms'] / 1000:.1f}", d["verdict"], clip(d["reason"], 90)])
    out = ["Coverage: 'read' = chunks the LLM was shown. A document counts as fully read only if every batch succeeded (all-or-nothing per document).", ""]
    out += table(["document", "chunks", "read", "fully read", "claims", "LLM calls", "tokens", "extract s", "verdict", "reason"], rows)
    return out


def sec_ledger(R):
    ex = R.extraction
    rows = []
    for d in sorted(R.doc_stats.values(), key=lambda x: x["file"]):
        rows.append([d["file"][:58], d["claims"], d["claims_paired"], d["claims_corroborated"], d["claims_unpaired"], d["pairs_judged"], d["findings"]])
    out = [f"claims recorded {ex.claims_created}, rejected {ex.claims_rejected}", "", "Per document (paired = has a comparable claim elsewhere; corroborated = same value stated elsewhere; unpaired = nothing comparable found):"]
    out += table(["document", "claims", "paired", "corroborated", "unpaired", "pairs judged", "findings"], rows)
    out += ["", "Rejected claims by reason (first 15):"]
    out += [f"  - {r}" for r in ex.rejections[:15]] or ["  none"]
    return out


def sec_pairs(R):
    js = R.judge
    out = [f"candidate pairs: {len(R.pairs)}", f"  by relation hint : {dict(Counter(p.relation_hint for p in R.pairs))}",
           f"  by match route   : {dict(Counter(p.matched_by for p in R.pairs))}",
           f"  across documents : {R.pairs_cross}   inside one document: {len(R.pairs) - R.pairs_cross}",
           f"exact agreements (corroboration, no LLM): {js.pairs_exact}", f"judged by the LLM: {js.pairs_judged} in {js.llm_calls} calls ({js.batches_failed} batches failed)",
           f"LLM verdicts: {dict(js.verdict_counts)}", f"findings saved {js.findings_saved}; rejected {js.findings_rejected}", "", "Rejected findings (first 15):"]
    out += [f"  - {r}" for r in js.rejections[:15]] or ["  none"]
    return out


def sec_llm(R):
    per_phase: dict[str, list] = {}
    for phase, m in phase_rows(R):
        per_phase.setdefault(phase, []).append(m)
    out = ["Per phase:"]
    rows = []
    for phase, ms_ in per_phase.items():
        ok = [m for m in ms_ if m.ok]
        lat = sorted(m.latency_ms for m in ok)
        rows.append([phase, len(ms_), len(ms_) - len(ok), f"{sum(m.prompt_tokens for m in ok):,}", f"{sum(m.completion_tokens for m in ok):,}",
                     f"{sum(m.thinking_tokens for m in ok):,}", f"{round(statistics.mean(m.prompt_tokens for m in ok)):,}" if ok else "-",
                     f"{max((m.prompt_tokens for m in ok), default=0):,}", round(statistics.mean(lat)) if lat else "-", lat[min(len(lat) - 1, int(.95 * len(lat)))] if lat else "-",
                     f"{sum(m.latency_ms for m in ms_) / 1000:.1f}"])
    out += table(["phase", "attempts", "failed", "prompt tok", "completion tok", "thinking tok", "avg ctx tok", "max ctx tok", "avg ms", "p95 ms", "total s"], rows)
    out += ["", "Every provider attempt (context = what was sent; tokens as counted by the API):"]
    rows = []
    for i, (phase, m) in enumerate(phase_rows(R), 1):
        rows.append([i, phase, f"{m.started_at - R.start_wall:.1f}", "ok" if m.ok else "FAIL", f"{m.prompt_chars:,}", f"{m.prompt_tokens:,}", f"{m.cached_tokens:,}",
                     m.completion_tokens, m.thinking_tokens, f"{m.total_tokens:,}", m.latency_ms, f"{m.response_chars:,}", clip(m.error or "", 60)])
    return out + table(["#", "phase", "at s", "result", "ctx chars", "ctx tokens", "cached", "completion", "thinking", "total tok", "latency ms", "resp chars", "error"], rows)


def sec_embeddings(R):
    calls = R.embedder.call_metrics
    rows = [[i, m.items, f"{m.prompt_chars:,}", m.latency_ms, "ok" if m.ok else "FAIL", clip(m.error or "", 60)] for i, m in enumerate(calls, 1)]
    return [f"calls {len(calls)}, failed {sum(not m.ok for m in calls)}, texts {sum(m.items for m in calls)} (claim topic texts embedded locally for matching)", ""] + \
        table(["#", "texts", "chars", "latency ms", "result", "error"], rows)


def sec_failures(R):
    out = ["LLM attempt failures:"]
    out += [f"  {phase}: {m.latency_ms}ms {clip(m.error, 160)}" for phase, m in phase_rows(R) if not m.ok] or ["  none"]
    out.append("Documents not fully read:")
    out += [f"  {d['file']}: {clip(d['error'] or 'a batch failed after all retries', 200)}" for d in R.doc_stats.values() if not d["ok"]] or ["  none"]
    out.append("Pair-judging problems:")
    out += [f"  {e}" for e in R.judge.errors[:10]] or ["  none"]
    return out


def sec_timing(R):
    rows = [[k, f"{v / 1000:.1f}s", pct(v, R.wall_s * 1000)] for k, v in R.stage.items()]
    return table(["stage", "duration", "share of wall time"], rows)


def sec_limits(R):
    return [
        "- Reading is all-or-nothing per document: if one batch fails after all retries, that document's claims are discarded and it counts as not read.",
        "- Candidate matching keeps at most 5 partners per claim (an existing default), so some comparable pairs are never judged.",
        "- The claim-extraction prompt lists business categories (capacity, supply, pricing, ...) and is not yet domain-agnostic.",
        "- Claims inside one document are also paired and judged; matching quality depends on how the model labels subject/metric/unit/period.",
        "- Exact agreements are trusted as corroboration without an LLM check.",
    ]


# ---- run ------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db-name", required=True, help="database with processed documents (e.g. from full_run.py)")
    ap.add_argument("--config", default=None)
    ap.add_argument("--llm-model", default=None)
    ap.add_argument("--embedding", choices=["local", "google"], default="local")
    ap.add_argument("--batch-chars", type=int, default=8000, help="characters of chunks per extraction call")
    ap.add_argument("--judge-batch-chars", type=int, default=12000, help="characters of excerpts per pair-judging call")
    ap.add_argument("--llm-attempts", type=int, default=5)
    ap.add_argument("--exclude", action="append", default=[],
                    help="skip documents whose file name matches this glob (case-insensitive), e.g. '*.xlsm'; repeatable")
    ap.add_argument("--skip-verdicts", action="store_true",
                    help="do not write per-document verdicts (saves one LLM call per document and the time they take)")
    ap.add_argument("--keep-claims", action="store_true",
                    help="reuse the claims already in the database (default: delete all claims and rebuild them, so every run uses new claims)")
    ap.add_argument("--copy-db-as", default=None,
                    help="first copy --db-name to this new database, clear its claims, and run there (the source stays intact)")
    args = ap.parse_args()

    overrides = {}
    if args.embedding == "local":
        overrides.update(embedding_provider="local", embedding_model=DEFAULT_LOCAL_MODEL, embedding_dimensions=384)
    if args.llm_model:
        overrides["llm_model"] = args.llm_model
    settings = get_settings().model_copy(update=overrides)
    started = datetime.now()
    stamp = started.strftime("%Y%m%d-%H%M%S")
    config = re.sub(r"[^A-Za-z0-9._-]+", "-", args.config or f"fact-ledger-fullread-{settings.llm_model}-embed-{args.embedding}")
    path = REPO_DIR / "test_results" / f"TEST-RUN-{stamp}-{config}.txt"
    path.parent.mkdir(exist_ok=True)

    llm = get_llm_client(settings)
    embedder = get_embedding_provider(settings)
    llm.max_attempts = embedder.max_attempts = args.llm_attempts
    source_db = args.db_name
    if args.copy_db_as:
        admin = create_engine(make_url(settings.database_url).set(database="postgres"), isolation_level="AUTOCOMMIT")
        with admin.connect() as conn:
            conn.execute(text("select pg_terminate_backend(pid) from pg_stat_activity where datname = :d and pid <> pg_backend_pid()"), {"d": source_db})
            conn.execute(text(f'CREATE DATABASE "{args.copy_db_as}" TEMPLATE "{source_db}"'))
        admin.dispose()
        args.db_name = args.copy_db_as
        print(f"[run] copied database '{source_db}' to '{args.db_name}'", flush=True)
    engine = create_engine(make_url(settings.database_url).set(database=args.db_name))
    session = sessionmaker(bind=engine, expire_on_commit=False)()
    R_cleared = 0
    if not args.keep_claims:
        R_cleared = session.execute(delete(Claim)).rowcount or 0  # every run rebuilds its claims from scratch
        session.commit()
        print(f"[run] deleted {R_cleared} existing claims; they will be rebuilt", flush=True)
    R = SimpleNamespace(args=args, settings=settings, started=started, config=config, db_name=args.db_name, llm=llm, embedder=embedder,
                        error=None, doc_stats={}, excluded=[], claims_cleared=0, scan=None, processed=[], extraction=ExtractionSummary(), pairs=None, pairs_cross=0, judge=None,
                        stage={}, llm_bounds=[0, 0], start_wall=time.time(), session=session)
    from app.services.pair_judge import JudgeStats
    R.judge = JudgeStats()
    t_run = time.perf_counter()
    print(f"[run] config={config}\n[run] database={args.db_name}\n[run] results file={path}", flush=True)

    investigation = Investigation(name=f"fact ledger run {stamp}", status=InvestigationStatus.running.value, started_at=utcnow(),
                                  execution_config={"approach": "fact_ledger_fullread", "batch_chars": args.batch_chars,
                                                    "judge_batch_chars": args.judge_batch_chars, "llm_model": settings.llm_model})
    R.claims_cleared = R_cleared
    try:
        session.add(investigation)
        session.commit()
        # ---- stage 0: register and process any new or changed files (local parsing + local embeddings)
        t = time.perf_counter()
        doc_service = DocumentService(session, settings)
        R.scan = doc_service.scan()
        R.processed = ProcessingService(session, doc_service, embedder, on_progress=lambda m, d: print(f"[process] {m}", flush=True)).process_pending()
        R.stage["0. scan files, process new or changed ones"] = round((time.perf_counter() - t) * 1000)
        print(f"[run] scan: {R.scan.added} new, {R.scan.updated} updated, {R.scan.unchanged} unchanged, {len(R.scan.skipped_duplicates)} duplicates", flush=True)
        service = ClaimService(session, llm, max_chars_per_call=args.batch_chars, on_progress=lambda m, d: print(f"[extract] {m}", flush=True))

        # ---- stage 1: read every chunk -> claims
        t = time.perf_counter()
        docs = list(session.scalars(select(Document).where(Document.processing_status == ProcessingStatus.processed.value).order_by(Document.filename)))
        R.excluded = [d.filename for d in docs if any(fnmatch.fnmatch(d.filename.lower(), p.lower()) for p in args.exclude)]
        docs = [d for d in docs if d.filename not in R.excluded]
        if R.excluded:
            print(f"[extract] excluded by request: {R.excluded}", flush=True)
        print(f"[extract] {len(docs)} documents to read", flush=True)
        for doc in docs:
            chunks_total = session.scalar(select(func.count()).select_from(DocumentChunk).where(DocumentChunk.document_id == doc.id)) or 0
            readable = session.scalar(select(func.count()).select_from(DocumentChunk).where(
                DocumentChunk.document_id == doc.id, DocumentChunk.content_type.not_in(EXCERPT_TYPES_SKIPPED))) or 0
            m0, c0, f0, t0 = len(llm.call_metrics), R.extraction.llm_calls, R.extraction.documents_failed, time.perf_counter()
            error = None
            try:
                service.extract_document(doc, R.extraction)
            except Exception as exc:  # one document must not end the run
                session.rollback()
                session.execute(delete(Claim).where(Claim.document_id == doc.id))
                session.commit()
                R.extraction.documents_failed += 1
                error = f"{exc.__class__.__name__}: {exc}"
                print(f"[extract] ERROR {doc.filename}: {error[:150]}", flush=True)
            attempts = llm.call_metrics[m0:]
            R.doc_stats[doc.id] = dict(
                id=doc.id, file=doc.filename, chunks=chunks_total, readable=readable, ok=R.extraction.documents_failed == f0 and error is None,
                claims=session.scalar(select(func.count()).select_from(Claim).where(Claim.document_id == doc.id)) or 0,
                calls=R.extraction.llm_calls - c0, total_tokens=sum(m.total_tokens for m in attempts if m.ok), ms=round((time.perf_counter() - t0) * 1000),
                error=error, verdict="-", reason="", claims_paired=0, claims_corroborated=0, claims_unpaired=0, pairs_judged=0, findings=0)
        R.stage["1. read every chunk, extract claims"] = round((time.perf_counter() - t) * 1000)
        R.llm_bounds[0] = len(llm.call_metrics)

        # ---- stage 2: match claims
        t = time.perf_counter()
        print("[match] matching claims across documents", flush=True)
        R.pairs = service.candidate_pairs(embedder)
        claim_doc = dict(session.execute(select(Claim.id, Claim.document_id)).all())
        R.pairs_cross = sum(claim_doc[p.claim_a_id] != claim_doc[p.claim_b_id] for p in R.pairs)
        R.stage["2. match claims (local embeddings)"] = round((time.perf_counter() - t) * 1000)
        print(f"[match] {len(R.pairs)} candidate pairs", flush=True)

        # ---- stage 3: judge pairs
        t = time.perf_counter()
        registry = build_registry(session, None, investigation.id)
        R.judge = PairJudge(session, llm, registry, max_chars_per_call=args.judge_batch_chars,
                            on_progress=lambda m, d: print(f"[judge] {m}", flush=True)).run(R.pairs)
        R.stage["3. judge pairs, verify quotes, save findings"] = round((time.perf_counter() - t) * 1000)
        R.llm_bounds[1] = len(llm.call_metrics)

        # ---- stage 4: per-document verdicts
        t = time.perf_counter()
        claims_by_doc: dict[str, set[str]] = {}
        for cid, did in claim_doc.items():
            claims_by_doc.setdefault(did, set()).add(cid)
        judged_by_doc = Counter()
        for p in R.pairs:
            if p.relation_hint != "exact":
                for did in {claim_doc[p.claim_a_id], claim_doc[p.claim_b_id]}:
                    judged_by_doc[did] += 1
        for d in R.doc_stats.values():
            mine = claims_by_doc.get(d["id"], set())
            d.update(claims_paired=len(mine & R.judge.paired_claims), claims_corroborated=len(mine & R.judge.corroborated_claims),
                     claims_unpaired=len(mine - R.judge.paired_claims), pairs_judged=judged_by_doc[d["id"]], findings=R.judge.findings_by_document[d["id"]])
            if not d["ok"]:
                d["verdict"], d["reason"] = "partly_unverified", "reading was incomplete, so its claims were not compared"
                continue
            if args.skip_verdicts:
                d["verdict"], d["reason"] = "skipped", "verdicts skipped by request (--skip-verdicts)"
                continue
            titles = R.judge.finding_titles_by_document.get(d["id"], [])
            prompt = (f"Document: {d['file']}\nChunks in the document: {d['chunks']}; chunks read by the model: {d['readable']}.\n"
                      f"Claims recorded: {d['claims']}; with a comparable claim elsewhere: {d['claims_paired']}; stated elsewhere with the same value: {d['claims_corroborated']}; "
                      f"with nothing comparable elsewhere: {d['claims_unpaired']}.\nPairs judged involving this document: {d['pairs_judged']}; discrepancies saved: {d['findings']}.\n"
                      + ("Discrepancy titles:\n" + "\n".join(f"- {x}" for x in titles) if titles else "No discrepancies were saved for this document."))
            try:
                v = llm.generate_structured_output(prompt, DocumentVerdict, system=VERDICT_SYSTEM)
                d["verdict"], d["reason"] = v.verdict, v.reason
            except LLMError as exc:
                d["verdict"], d["reason"] = "partly_unverified", f"verdict call failed: {str(exc)[:80]}"
            print(f"[verdict] {d['file']}: {d['verdict']}", flush=True)
        R.stage["4. per-document verdicts"] = round((time.perf_counter() - t) * 1000)
    except Exception:
        R.error = traceback.format_exc()
        print("[run] ERROR\n" + R.error, flush=True)

    R.finished = datetime.now()
    R.wall_s = time.perf_counter() - t_run
    try:
        inv = session.get(Investigation, investigation.id)
        inv.status = InvestigationStatus.failed.value if R.error else InvestigationStatus.completed.value
        inv.completed_at, inv.duration_ms = utcnow(), round(R.wall_s * 1000)
        inv.total_tokens = llm.get_token_usage().total_tokens
        session.commit()
    except Exception:
        session.rollback()
    R.result = SimpleNamespace(investigation_id=investigation.id)

    lines = [f"TEST RUN  {stamp}  {config}", "=" * 78, "", "HOW THIS RUN WAS ACHIEVED"]

    def section(title, fn):
        lines.extend(["", "=" * 78, title, "=" * 78])
        try:
            lines.extend(fn(R))
        except Exception as exc:
            lines.append(f"[section failed: {exc.__class__.__name__}: {exc}]")

    try:
        lines.extend(sec_how(R))
    except Exception as exc:
        lines.append(f"[failed: {exc}]")
    for title, fn in [("AT A GLANCE", sec_glance), ("NEW OR CHANGED FILES AND CLAIM RESET", sec_new_files),
                      ("CONTRADICTIONS ONLY (classification = contradiction)", sec_contradictions),
                      ("FINAL REPORT OF DISCREPANCIES (with evidence: document, chunk, location, quote)", base.sec_findings),
                      ("COVERAGE AND PER-DOCUMENT VERDICTS", sec_documents), ("CLAIM LEDGER", sec_ledger), ("PAIR MATCHING AND JUDGING", sec_pairs),
                      ("LLM CALLS: tokens, context size, latency, failures", sec_llm), ("EMBEDDING CALLS", sec_embeddings), ("FAILURES", sec_failures),
                      ("TIMING", sec_timing), ("KNOWN LIMITS OF THIS RUN", sec_limits)]:
        section(title, fn)
    if R.error:
        lines.extend(["", "=" * 78, "Run error", "=" * 78, R.error])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    session.close()
    engine.dispose()
    print(f"[run] DONE in {R.wall_s:.0f}s. Results: {path}", flush=True)


if __name__ == "__main__":
    main()
