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
import os
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
from app.models import Claim, Document, DocumentChunk, Evidence, Finding, Investigation  # noqa: E402
from app.tools.reading import locator_text  # noqa: E402
from app.schemas.run_metrics import LLMTotals, PhaseMetrics, RunCounts, RunMetrics, RunSettings, StageTiming  # noqa: E402
from app.services.run_metrics import write_metrics  # noqa: E402
from app.models.base import InvestigationStatus, ProcessingStatus, utcnow  # noqa: E402
from app.services.llm_pairing import LLMPairer  # noqa: E402
from app.services.llama_claims import LlamaClaimExtractor  # noqa: E402
from app.services.missing_items import MissingItemChecker  # noqa: E402
from app.llm.llama_extract import LlamaExtractClient  # noqa: E402
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
    names = ["extract claims", "group claims (LLM)", "judge pairs", "document verdicts"]
    for i, m in enumerate(R.llm.call_metrics):
        phase = names[0 if i < bounds[0] else 1 if i < bounds[1] else 2 if i < bounds[2] else 3]
        yield phase, m


# ---- report sections ----------------------------------------------------------
def sec_how(R):
    a, s = R.args, R.settings
    return [
        ("1. Reading: the documents selected for this run were uploaded to LlamaExtract (LlamaIndex Cloud, cost_effective tier), which returns claims with the page and text they came from; "
         "each claim is tied to a stored chunk and kept only if its value appears in that chunk text." if a.extractor == "llamaextract" else
         f"1. Reading: every chunk of every processed document{' except ' + ', '.join(R.excluded) if R.excluded else ''} (database '{R.db_name}') was sent to the LLM in batches of up to {a.batch_chars:,} characters; each batch returns claims, kept only if their value appears in the cited chunk (existing claim extractor, no chunk or call limits)."),
        (f"2. Ledger: accepted claims are stored in the claims table; the claims are sorted by label and sent to the LLM in batches of up to {a.pairing_batch_chars:,} characters, which groups the ones that state the same attribute of the same subject (no code pairing, no fixed number of documents or claims assumed)."
         if a.pairing == "llm" else
         f"2. Ledger: accepted claims are stored in the claims table; claims are matched across and within documents by wording and by local embeddings ({s.embedding_model}), without a fixed number of documents or claims assumed."),
        f"3. Judging: pairs that agree exactly count as corroboration; every other pair (conflicting, partial, contextual) goes to the LLM in batches of up to {a.judge_batch_chars:,} characters with both source excerpts.",
        "4. Findings: a discrepancy is saved only if its two quotes are exact substrings of their chunks (checked in code)"
        + ("; per-document verdicts were skipped by request." if a.skip_verdicts else "; then each document gets a verdict built from its counts."),
        f"5. Model: {s.llm_model}, {'through LlamaExtract' if s.llm_provider == 'llamaextract' else 'temperature 0'}, up to {a.llm_attempts} attempts per call honouring the provider's retry delay; no step, tool-call or finding limits.",
    ]


def sec_glance(R):
    ex, js, res = R.extraction, R.judge, R
    t = R.llm.get_token_usage()
    ok = [m for m in R.llm.call_metrics if m.ok]
    out = ([">>> THIS RUN WAS STOPPED BEFORE IT FINISHED: everything below covers only the part that ran. <<<"] if R.stopped else [])
    out += [f"Run started   : {R.started:%Y-%m-%d %H:%M:%S} (local)   finished: {R.finished:%H:%M:%S}   wall time: {R.wall_s:.1f}s",
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
            f"Claims        : {ex.claims_created} recorded, {ex.claims_rejected} rejected (value not found in cited chunk or unknown chunk id), "
            f"{ex.claims_skipped_as_labels} skipped because the value was only the label of its own row",
            f"Extraction    : up to {R.args.batch_chunks or 'any number of'} chunks per call (and {R.args.batch_chars:,} characters); running vocabulary {R.args.vocabulary}; "
            f"coverage check {R.args.coverage_check}"
            + (f": {ex.coverage_chunks_checked} chunks had {ex.coverage_numbers} numbers without a claim, {ex.coverage_calls} follow-up calls "
               f"({ex.coverage_calls_failed} failed), {ex.coverage_claims_added} claims added" if R.args.coverage_check == "on" else "")]
    if R.args.extractor == "llamaextract":
        out.append(f"Extractor     : LlamaExtract (cost_effective tier) instead of the LLM: {ex.external_calls} documents uploaded to LlamaIndex Cloud, "
                   f"{ex.external_credits:g} credits used, {ex.documents_failed} documents failed (the batch and coverage settings above do not apply)")
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
    out = [f"{len(rows)} contradiction(s). Each lists its evidence (document, location, exact quote); the full detail is in the findings report below (search for the finding id).",
           "Locations: 'page N' is the PDF page number (the first page of the file is 1), which can differ from a page number printed on the page itself.", ""]
    for f in rows:
        out.append(f"- [{f.id}] severity {f.severity}, confidence {f.confidence}: {f.title}")
        out.append(f"    {clip(f.description, 220)}")
        evidence = s.execute(select(Evidence, DocumentChunk, Document.filename).join(DocumentChunk, DocumentChunk.id == Evidence.chunk_id)
                             .join(Document, Document.id == DocumentChunk.document_id).where(Evidence.finding_id == f.id)).all()
        for ev, chunk, filename in evidence:
            out.append(f"    evidence: {filename} | {locator_text(chunk)} | chunk {chunk.id} | \"{clip(ev.quoted_text, 120)}\"")
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
    out = []
    if R.pairing_stats:
        ps = R.pairing_stats
        out += [f"pairing mode: LLM grouping of sorted claims: {ps.claims} claims in {ps.batches} batch(es), {ps.llm_calls} call(s) ({ps.batches_failed} failed), "
                f"{ps.groups} groups covering {ps.claims_grouped} claims (largest {ps.largest_group}), {ps.pairs} pairs ({ps.neighbor_pairs} added from nearest neighbours in other documents), {ps.unknown_refs} unknown refs ignored",
                f"  pairing every claim with every other in its group would have made {ps.pairs_all_combinations} pairs; "
                f"claims stating the same value are linked as agreement, and one claim per distinct value ({ps.distinct_values} values) is compared",
                "  note: claims in different batches are never grouped together" + (" (more than one batch in this run)" if ps.batches > 1 else " (one batch: no such gap in this run)")]
        out += [f"  pairing problem: {e}" for e in ps.errors]
    else:
        out.append("pairing mode: code (word overlap and local embeddings)")
    out += [f"candidate pairs: {len(R.pairs)}", f"  by relation hint : {dict(Counter(p.relation_hint for p in R.pairs))}",
           f"  by match route   : {dict(Counter(p.matched_by for p in R.pairs))}",
           f"  across documents : {R.pairs_cross}   inside one document: {len(R.pairs) - R.pairs_cross}",
           f"exact agreements (corroboration, no LLM): {js.pairs_exact}", f"judged by the LLM: {js.pairs_judged} in {js.llm_calls} calls ({js.batches_failed} batches failed)",
           f"LLM verdicts: {dict(js.verdict_counts)}", f"findings saved {js.findings_saved}; rejected {js.findings_rejected}",
           f"single-pair retries {js.retries}; verdicts that did not match their pair (echoed values) {js.misaligned}; verdicts without echoed values {js.echo_missing}; "
           f"quotes supplied by code for verdicts that gave none {js.quotes_from_code}",
           f"findings merged because they describe the same discrepancy {js.merged}; insufficient-evidence findings absorbed {js.absorbed}; "
           f"consistent findings grouped {js.grouped}", *[f"  - {m}" for m in js.merge_log[:30]],
           "", "Verdicts that did not match their pair (first 20):"]
    out += [f"  - {e}" for e in js.echo_log[:20]] or ["  none"]
    out += ["", "Rejected findings (first 15):"]
    out += [f"  - {r}" for r in js.rejections[:15]] or ["  none"]
    return out


def sec_pair_log(R):
    """Every candidate pair, with both claims (file | subject | metric | value) and what happened to it, so a missed discrepancy can be traced."""
    records = list(R.judge.records.values())
    outcomes = Counter(r.outcome.split(":")[0].split(",")[0] for r in records)
    rows = [[i, clip(r.claim_a, 70), clip(r.claim_b, 70), r.hint, r.route, clip(r.outcome, 160)] for i, r in enumerate(records, 1)]
    return [f"{len(records)} candidate pairs. Outcomes: {dict(outcomes)}", ""] + table(["#", "claim A", "claim B", "hint", "route", "what happened"], rows)


def sec_missing(R):
    m = getattr(R, "missing", None)
    if m is None:
        return ["not run (needs the LlamaExtract extractor, which returns the references)"]
    out = [f"references returned with the claims: {m.references}; found in the data room: {m.present}; "
           f"saved as missing information: {m.saved}; rejected by the quote check: {m.rejected}; repeats skipped: {m.duplicates}"]
    if m.skipped_reason:
        out.append(f"check skipped: {m.skipped_reason}")
    out += [f"  missing: {x}" for x in m.missing_items[:20]]
    out += [f"  rejected: {x}" for x in m.rejections[:10]]
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
def build_metrics(R, results_path) -> RunMetrics:
    """The run's numbers in the same terms as the results file, for the UI."""
    llm_calls = R.llm.call_metrics
    ok = [m for m in llm_calls if m.ok]
    lat = R.llm.latency_summary()
    tokens = R.llm.get_token_usage()
    pt = [m.prompt_tokens for m in ok]
    per_phase: dict[str, list] = {}
    for phase, m in phase_rows(R):
        per_phase.setdefault(phase, []).append(m)
    phases = []
    for phase, items in per_phase.items():
        good = [m for m in items if m.ok]
        latencies = sorted(m.latency_ms for m in good)
        phases.append(PhaseMetrics(
            phase=phase, attempts=len(items), failed=len(items) - len(good), prompt_tokens=sum(m.prompt_tokens for m in good),
            completion_tokens=sum(m.completion_tokens for m in good), thinking_tokens=sum(m.thinking_tokens for m in good),
            avg_context_tokens=round(statistics.mean(m.prompt_tokens for m in good)) if good else 0,
            max_context_tokens=max((m.prompt_tokens for m in good), default=0),
            avg_ms=round(statistics.mean(latencies)) if latencies else 0,
            p95_ms=latencies[min(len(latencies) - 1, int(.95 * len(latencies)))] if latencies else 0,
            total_s=round(sum(m.latency_ms for m in items) / 1000, 1)))
    docs = list(R.doc_stats.values())
    ex, js, emb = R.extraction, R.judge, R.embedder.call_metrics
    counts = RunCounts(
        documents=len(docs), documents_fully_read=sum(d["ok"] for d in docs), chunks=sum(d["chunks"] for d in docs),
        claims_recorded=ex.claims_created, claims_rejected=ex.claims_rejected, candidate_pairs=len(R.pairs or []),
        pairs_exact=js.pairs_exact, pairs_judged=js.pairs_judged, verdicts=dict(js.verdict_counts),
        findings_saved=js.findings_saved + (R.missing.saved if getattr(R, "missing", None) else 0),
        findings_rejected=js.findings_rejected + (R.missing.rejected if getattr(R, "missing", None) else 0), embedding_calls=len(emb),
        embedding_texts=sum(m.items for m in emb), extractor_credits=ex.external_credits, pairs_without_clustering=getattr(getattr(R, 'pairing_stats', None), 'pairs_all_combinations', 0),
        coverage_calls=ex.coverage_calls, coverage_claims_added=ex.coverage_claims_added)
    a, s = R.args, R.settings
    return RunMetrics(
        investigation_id=getattr(R.result, "investigation_id", None), config_name=R.config,
        state="failed" if R.error else "stopped" if R.stopped else "completed",
        started_at=R.started.isoformat(timespec="seconds"), finished_at=R.finished.isoformat(timespec="seconds"),
        wall_s=round(R.wall_s, 1), results_file=str(results_path),
        settings=RunSettings(extractor=a.extractor, pairing=a.pairing, vocabulary=a.vocabulary, coverage_check=a.coverage_check,
                             batch_chunks=a.batch_chunks or None, batch_chars=a.batch_chars, llm_model=s.llm_model,
                             embedding_model=s.embedding_model),
        counts=counts,
        llm=LLMTotals(
            attempts=len(llm_calls), failed=sum(not m.ok for m in llm_calls), retries=R.llm.retry_count,
            total_s=round(sum(m.latency_ms for m in llm_calls) / 1000, 1), avg_ms=lat.avg_ms, p95_ms=lat.p95_ms, max_ms=lat.max_ms,
            prompt_tokens=tokens.prompt_tokens, completion_tokens=tokens.completion_tokens, thinking_tokens=tokens.thinking_tokens,
            cached_tokens=tokens.cached_tokens, total_tokens=tokens.total_tokens, credits=round(getattr(R.llm, 'credits', 0.0), 1),
            min_context_tokens=min(pt, default=0),
            avg_context_tokens=round(statistics.mean(pt)) if pt else 0, max_context_tokens=max(pt, default=0)),
        phases=phases,
        stages=[StageTiming(stage=k, seconds=round(v / 1000, 1), percent=round(100 * v / (R.wall_s * 1000), 1) if R.wall_s else 0.0)
                for k, v in R.stage.items()])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db-name", default=None, help="database to run in (default: the database the app is configured for)")
    ap.add_argument("--config", default=None)
    ap.add_argument("--llm-model", default=None)
    ap.add_argument("--embedding", choices=["local", "google"], default="local")
    ap.add_argument("--batch-chars", type=int, default=8000, help="characters of chunks per extraction call")
    ap.add_argument("--judge-batch-chars", type=int, default=12000, help="characters of excerpts per pair-judging call")
    ap.add_argument("--judge-workers", type=int, default=1,
                    help="pair-judging calls made at the same time (1 = one after another, the default); the answers are still handled in order")
    ap.add_argument("--pairing", choices=["code", "llm"], default="llm",
                    help="llm = the LLM groups the sorted claims (default); code = candidate pairs from word overlap and embeddings")
    ap.add_argument("--pairing-batch-chars", type=int, default=60000, help="characters of claims per grouping call (--pairing llm)")
    ap.add_argument("--extractor", choices=["llm", "llamaextract"], default="llamaextract",
                    help="llamaextract = each selected document is uploaded to LlamaExtract (LlamaIndex Cloud; default); llm = the LLM reads the chunks")
    ap.add_argument("--batch-chunks", type=int, default=0, help="at most this many chunks per extraction call (default 0 = cut by characters only)")
    ap.add_argument("--vocabulary", choices=["on", "off"], default="off", help="show earlier subject/metric labels to each extraction call")
    ap.add_argument("--coverage-check", choices=["on", "off"], default="off",
                    help="one follow-up call per chunk that states numbers no claim accounts for")
    ap.add_argument("--llm-attempts", type=int, default=5)
    ap.add_argument("--only", action="append", default=None,
                    help="read only this document (path relative to the data room); repeatable. Without --only every document is read")
    ap.add_argument("--stop-file", default=None, help="when this file exists the run stops at the next safe point and still writes its results file")
    ap.add_argument("--verdicts", action="store_true", help="also write a verdict per document (one extra LLM call per document)")
    ap.add_argument("--exclude", action="append", default=[],
                    help="skip documents whose file name matches this glob (case-insensitive), e.g. '*.xlsm'; repeatable")
    ap.add_argument("--skip-verdicts", action="store_true",
                    help="do not write per-document verdicts (saves one LLM call per document and the time they take)")
    ap.add_argument("--keep-claims", action="store_true",
                    help="reuse the claims already in the database (default: delete all claims and rebuild them, so every run uses new claims)")
    ap.add_argument("--copy-db-as", default=None,
                    help="first copy --db-name to this new database, clear its claims, and run there (the source stays intact)")
    args = ap.parse_args()
    args.skip_verdicts = args.skip_verdicts or not args.verdicts  # verdicts are opt-in (--verdicts)

    overrides = {}
    if args.embedding == "local":
        overrides.update(embedding_provider="local", embedding_model=DEFAULT_LOCAL_MODEL, embedding_dimensions=384)
    if args.llm_model:
        overrides["llm_model"] = args.llm_model
    settings = get_settings().model_copy(update=overrides)
    args.db_name = args.db_name or make_url(settings.database_url).database
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
    def stopped() -> bool:
        return bool(args.stop_file) and os.path.exists(args.stop_file)

    R = SimpleNamespace(args=args, settings=settings, started=started, config=config, db_name=args.db_name, llm=llm, embedder=embedder,
                        error=None, stopped=False, doc_stats={}, excluded=[], claims_cleared=0, scan=None, processed=[], extraction=ExtractionSummary(), pairs=None, pairs_cross=0, judge=None,
                        stage={}, llm_bounds=[0, 0, 0], pairing_stats=None, start_wall=time.time(), session=session)
    from app.services.pair_judge import JudgeStats
    R.judge = JudgeStats()
    t_run = time.perf_counter()
    print(f"[run] config={config}\n[run] database={args.db_name}\n[run] results file={path}", flush=True)

    investigation = Investigation(name=f"fact ledger run {stamp}", status=InvestigationStatus.running.value, started_at=utcnow(),
                                  execution_config={"approach": "fact_ledger_fullread", "batch_chars": args.batch_chars,
                                                    "judge_batch_chars": args.judge_batch_chars, "llm_model": settings.llm_model})
    R.claims_cleared = R_cleared

    class RunStopped(Exception):
        pass

    def check_stop():
        if stopped():
            raise RunStopped()

    try:
        session.add(investigation)
        session.commit()
        # ---- stage 0: register and process any new or changed files (local parsing + local embeddings)
        t = time.perf_counter()
        doc_service = DocumentService(session, settings)
        selected = None if args.only is None else {x.replace(chr(92), '/') for x in args.only}  # unselected files are not touched at all
        R.scan = doc_service.scan(only=selected)
        R.processed = ProcessingService(session, doc_service, embedder, on_progress=lambda m, d: print(f"[process] {m}", flush=True)).process_pending(only=selected)
        R.stage["0. scan files, process new or changed ones"] = round((time.perf_counter() - t) * 1000)
        print(f"[run] scan: {R.scan.added} new, {R.scan.updated} updated, {R.scan.unchanged} unchanged, {len(R.scan.skipped_duplicates)} duplicates", flush=True)
        extractor = None
        if args.extractor == "llamaextract":
            if not settings.allow_third_party_apis:
                raise RuntimeError("--extractor llamaextract uploads documents to a third-party API, which is disabled (set ALLOW_THIRD_PARTY_APIS=true to allow it)")
            if not settings.llama_cloud_api_key:
                raise RuntimeError("--extractor llamaextract needs LLAMA_CLOUD_API_KEY in the environment")
            extractor = LlamaClaimExtractor(session, LlamaExtractClient(settings.llama_cloud_api_key, project_id=settings.llama_cloud_project_id,
                                                                         tier="cost_effective"), settings.data_room_dir,
                                            on_progress=lambda m, d: print(f"[extract] {m}", flush=True), should_stop=stopped)
        service = ClaimService(session, llm, max_chars_per_call=args.batch_chars, use_vocabulary=args.vocabulary == "on",
                               max_chunks_per_call=args.batch_chunks or None, should_stop=stopped, coverage_check=args.coverage_check == "on", on_progress=lambda m, d: print(f"[extract] {m}", flush=True))

        # ---- stage 1: read every chunk -> claims
        t = time.perf_counter()
        docs = list(session.scalars(select(Document).where(Document.processing_status == ProcessingStatus.processed.value).order_by(Document.filename)))
        wanted = None if args.only is None else {x.replace(chr(92), '/') for x in args.only}
        R.excluded = [d.filename for d in docs if (wanted is not None and d.filename.replace(chr(92), '/') not in wanted)
                      or any(fnmatch.fnmatch(d.filename.lower(), p.lower()) for p in args.exclude)]
        docs = [d for d in docs if d.filename not in R.excluded]
        if R.excluded:
            print(f"[extract] excluded by request: {R.excluded}", flush=True)
        print(f"[extract] {len(docs)} documents to read", flush=True)
        for doc in docs:
            check_stop()
            chunks_total = session.scalar(select(func.count()).select_from(DocumentChunk).where(DocumentChunk.document_id == doc.id)) or 0
            readable = session.scalar(select(func.count()).select_from(DocumentChunk).where(
                DocumentChunk.document_id == doc.id, DocumentChunk.content_type.not_in(EXCERPT_TYPES_SKIPPED))) or 0
            m0, c0, f0, t0 = len(llm.call_metrics), R.extraction.llm_calls, R.extraction.documents_failed, time.perf_counter()
            error = None
            try:
                (extractor or service).extract_document(doc, R.extraction)
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

        check_stop()
        # ---- stage 2: match claims
        t = time.perf_counter()
        print("[match] matching claims across documents", flush=True)
        if args.pairing == "llm":
            R.pairs, R.pairing_stats = LLMPairer(session, llm, max_chars_per_call=args.pairing_batch_chars, should_stop=stopped, embedder=embedder).run()
        else:
            R.pairs = service.candidate_pairs(embedder)
        R.llm_bounds[1] = len(llm.call_metrics)
        claim_doc = dict(session.execute(select(Claim.id, Claim.document_id)).all())
        R.pairs_cross = sum(claim_doc[p.claim_a_id] != claim_doc[p.claim_b_id] for p in R.pairs)
        R.stage["2. match claims (" + ("LLM grouping of sorted claims" if args.pairing == "llm" else "code + local embeddings") + ")"] = round((time.perf_counter() - t) * 1000)
        print(f"[match] {len(R.pairs)} candidate pairs", flush=True)

        check_stop()
        # ---- stage 3: judge pairs
        t = time.perf_counter()
        registry = build_registry(session, None, investigation.id)
        R.judge = PairJudge(session, llm, registry, max_chars_per_call=args.judge_batch_chars, should_stop=stopped, workers=args.judge_workers,
                            on_progress=lambda m, d: print(f"[judge] {m}", flush=True)).run(R.pairs)
        R.stage["3. judge pairs, verify quotes, save findings"] = round((time.perf_counter() - t) * 1000)
        R.llm_bounds[2] = len(llm.call_metrics)

        check_stop()
        # ---- stage 3b: items a document refers to that the data room does not hold (no LLM call; the references came with the claims)
        if extractor is not None:
            t = time.perf_counter()
            R.missing = MissingItemChecker(session, registry).run(extractor.references)
            R.stage["3b. referenced items absent from the data room"] = round((time.perf_counter() - t) * 1000)
            print(f"[missing] {R.missing.references} references, {R.missing.present} found in the room, {R.missing.saved} saved as missing information"
                  + (f" (check skipped: {R.missing.skipped_reason})" if R.missing.skipped_reason else ""), flush=True)

        check_stop()
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
    except (RunStopped, KeyboardInterrupt):
        R.stopped = True
        print("[run] STOP requested: writing the results file with what was done so far", flush=True)
    except Exception:
        R.error = traceback.format_exc()
        print("[run] ERROR\n" + R.error, flush=True)

    R.finished = datetime.now()
    R.wall_s = time.perf_counter() - t_run
    try:
        inv = session.get(Investigation, investigation.id)
        inv.status = (InvestigationStatus.failed if R.error else InvestigationStatus.stopped if R.stopped else InvestigationStatus.completed).value
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
                      ("COVERAGE AND PER-DOCUMENT VERDICTS", sec_documents), ("CLAIM LEDGER", sec_ledger), ("PAIR MATCHING AND JUDGING", sec_pairs), ("EVERY CANDIDATE PAIR AND WHAT HAPPENED TO IT", sec_pair_log),
                      ("REFERENCED ITEMS NOT IN THE DATA ROOM (single-claim missing information)", sec_missing),
                      ("LLM CALLS: tokens, context size, latency, failures", sec_llm), ("EMBEDDING CALLS", sec_embeddings), ("FAILURES", sec_failures),
                      ("TIMING", sec_timing), ("KNOWN LIMITS OF THIS RUN", sec_limits)]:
        section(title, fn)
    if R.error:
        lines.extend(["", "=" * 78, "Run error", "=" * 78, R.error])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    try:
        write_metrics(path, build_metrics(R, path))
    except Exception as exc:  # the metrics file is a convenience: never lose the results file over it
        print(f"[run] could not write the metrics file: {exc.__class__.__name__}: {exc}", flush=True)
    session.close()
    engine.dispose()
    print(f"[run] DONE in {R.wall_s:.0f}s. Results: {path}", flush=True)


if __name__ == "__main__":
    main()
