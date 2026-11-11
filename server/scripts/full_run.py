"""Full test run: scan -> process (extract, chunk, embed) -> agent investigation -> results file.

Run from server/:
    .venv/Scripts/python.exe scripts/full_run.py [--config LABEL] [--merge-chunks] [--max-steps 40]

Writes ../test_results/TEST-RUN-<YYYYMMDD>-<HHMMSS>-<config>.txt with every metric the run produced.
Each run uses its own Postgres database (named in the file) and leaves it in place so findings can be
inspected; reuse it with --db-name to retry only the investigation without re-embedding.
Real documents are sent to the configured provider (embeddings, and tool results to the LLM).
"""
from __future__ import annotations

import argparse
import inspect
import platform
import re
import statistics
import subprocess
import sys
import time
import traceback
from collections import Counter
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

SERVER_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = SERVER_DIR.parent
sys.path.insert(0, str(SERVER_DIR))

from alembic import command  # noqa: E402
from alembic.config import Config  # noqa: E402
from sqlalchemy import create_engine, func, select, text  # noqa: E402
from sqlalchemy.engine import make_url  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402

from app.agent import InvestigationAgent, run_investigation  # noqa: E402
from app.agent.prompts import SYSTEM, build_prompt  # noqa: E402
from app.core.config import get_settings  # noqa: E402
from app.llm.factory import get_embedding_provider, get_llm_client
from app.llm.local import DEFAULT_LOCAL_MODEL  # noqa: E402
from app.models import Document, DocumentChunk, Evidence, Finding  # noqa: E402
from app.services import chunking  # noqa: E402
from app.services.document_service import DocumentService  # noqa: E402
from app.services.extraction_service import unit_count  # noqa: E402
from app.services.extractors import pdf as pdf_extractor  # noqa: E402
from app.services.processing_service import ProcessingService  # noqa: E402
from app.services.retrieval_service import RetrievalService  # noqa: E402
from app.tools import ToolBudget, ToolContext, ToolRegistry  # noqa: E402
from app.tools.reading import MAX_SECTION_CHARS, locator_text  # noqa: E402

DEFAULT_FOCUS = (
    "Review the whole data room for discrepancies between documents. Prioritise figures that appear in more than "
    "one document (capacities, quantities, costs, revenues, prices, dates, timelines, locations, counts) and "
    "statements that one document makes and another contradicts or does not support. Compare like with like "
    "(same metric, unit, period, scope). Submit a finding for each material discrepancy with exact quotes from "
    "both sides, and use missing_information or insufficient_evidence where something is claimed but not evidenced.")

# Measured locally on 2026-11-11 on these same files (duplicates excluded). Not part of the run itself.
CHUNK_COMPARISON = [
    ("per-unit, old default", "pdf", 1300, 27, 53, 92, 1218, "92%"), ("per-unit, old default", "xlsx", 1594, 190, 383, 1174, 1500, "18%"),
    ("merge to 600", "pdf", 192, 324, 367, 590, 1218, "8%"), ("merge to 600", "xlsx", 894, 532, 683, 1500, 1500, "2%"),
    ("merge to 1000", "pdf", 143, 428, 494, 977, 1218, "6%"), ("merge to 1000", "xlsx", 651, 922, 939, 1500, 1500, "2%"),
    ("merge to 1500", "pdf", 126, 460, 560, 1193, 1490, "4%"), ("merge to 1500", "xlsx", 521, 1345, 1173, 1500, 1500, "2%"),
]


# ---- small helpers ------------------------------------------------------------
def clip(value, n: int) -> str:
    s = " ".join(str(value).split())
    return s if len(s) <= n else s[: n - 3] + "..."


def pct(a: float, b: float) -> str:
    return f"{100 * a / b:.0f}%" if b else "n/a"


def table(headers: list[str], rows: list[list], indent: str = "  ") -> list[str]:
    widths = [len(str(h)) for h in headers]
    for r in rows:
        for i, c in enumerate(r):
            widths[i] = max(widths[i], len(str(c)))
    fmt = lambda r: indent + "  ".join(str(c).ljust(widths[i]) for i, c in enumerate(r)).rstrip()
    return [fmt(headers), indent + "  ".join("-" * w for w in widths)] + [fmt(r) for r in rows]


def size_stats(sizes: list[int]) -> dict:
    if not sizes:
        return dict(n=0, total=0, mean=0, median=0, p90=0, max=0, lt100="n/a")
    q = statistics.quantiles(sizes, n=10) if len(sizes) >= 10 else [max(sizes)] * 9
    return dict(n=len(sizes), total=sum(sizes), mean=round(statistics.mean(sizes)), median=round(statistics.median(sizes)),
                p90=round(q[8]), max=max(sizes), lt100=pct(sum(s < 100 for s in sizes), len(sizes)))


def ms(seconds: float) -> int:
    return round(seconds * 1000)


def git_info() -> str:
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_DIR, capture_output=True, text=True, timeout=10).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=REPO_DIR, capture_output=True, text=True, timeout=10).stdout.strip()
        return f"{sha}{' + uncommitted changes' if dirty else ''}"
    except Exception:
        return "unknown"


def ensure_database(settings, name: str) -> tuple[str, bool]:
    admin_url = make_url(settings.database_url)
    admin = create_engine(admin_url.set(database="postgres"), isolation_level="AUTOCOMMIT")
    try:
        with admin.connect() as c:
            exists = bool(c.execute(text("select 1 from pg_database where datname = :n"), {"n": name}).scalar())
            if not exists:
                c.execute(text(f'CREATE DATABASE "{name}"'))
    finally:
        admin.dispose()
    url = admin_url.set(database=name).render_as_string(hide_password=False)
    if not exists:
        cfg = Config(str(SERVER_DIR / "alembic.ini"))
        cfg.set_main_option("script_location", str(SERVER_DIR / "alembic"))
        cfg.set_main_option("sqlalchemy.url", url)
        command.upgrade(cfg, "head")
    return url, not exists


# ---- report sections (each returns lines; a failing section never loses the others) --------------
def sec_how(R) -> list[str]:
    a, s = R.args, R.settings
    strategy = ("merged: small neighbouring units of the same page/sheet/heading joined up to "
                f"{chunking.DEFAULT_TARGET_CHARS} chars" if a.merge_chunks else "per-unit (old default): one extracted unit is one chunk")
    return [
        f"1. Documents: every supported file in server/data_room was scanned (SHA-256 identity, duplicates skipped) and parsed locally, "
        f"{a.extract_workers} documents at a time (PyMuPDF, python-docx, openpyxl; .xlsm read like .xlsx); scanned PDFs are skipped automatically.",
        f"2. Chunking: {strategy}; hard maximum {chunking.DEFAULT_MAX_CHARS} chars; embedding text is prefixed with file name and heading.",
        f"3. Embeddings: {s.embedding_model} at {s.embedding_dimensions} dims, "
        f"{'run locally on this machine (fastembed, no API)' if (s.embedding_provider or s.llm_provider) == 'local' else 'via the Google API'}, "
        f"batches of up to {R.embedder.MAX_BATCH}, stored with each chunk in Postgres/pgvector (HNSW cosine) in database '{R.db_name}'.",
        f"4. Investigation: a tool-using agent loop ({s.llm_model}, temperature 0, JSON action protocol, 7 tools, {a.max_steps or 'no'} step limit, "
        f"{a.tool_budget or 'no'} tool-call limit); the agent must open every document before it may finish; no claim extraction or candidate pairs were used.",
        f"5. Safeguards: every finding quote is checked in code as an exact substring of its stored chunk before saving; arithmetic only through "
        f"the calculate tool; provider retries up to {a.llm_attempts} attempts honouring Google's retry delay.",
    ]


def sec_glance(R) -> list[str]:
    res = R.result
    out = [f"Run started   : {R.started:%Y-%m-%d %H:%M:%S} (local)   finished: {R.finished:%H:%M:%S}   wall time: {R.wall_s:.1f}s",
           f"Config        : {R.config}",
           f"Database      : {R.db_name} ({'created for this run' if R.db_created else 'reused: processing was skipped if already done'})",
           f"Code version  : git {git_info()}   python {platform.python_version()}   {platform.system()}"]
    if R.error:
        out.append("RUN ERROR     : see section 'Run error' at the end of this file")
    if R.docs:
        counts = Counter(d["status"] for d in R.doc_rows)
        out.append(f"Documents     : {len(R.docs)} registered, {R.scan.skipped_duplicates and len(R.scan.skipped_duplicates) or 0} duplicates skipped; "
                   f"processing result: {dict(counts)}")
    out.append(f"Chunks        : {R.chunk_total} total, {R.chunk_chars:,} characters")
    ec = [m for m in R.embedder.call_metrics]
    out.append(f"Embedding     : {len(ec)} calls ({sum(not m.ok for m in ec)} failed), {sum(m.items for m in ec)} texts embedded, "
               f"{sum(m.latency_ms for m in ec) / 1000:.1f}s total")
    if res:
        t = res.token_usage
        out += [f"Investigation : {res.termination}, {len(res.steps)} agent steps, {len(res.finding_ids)} findings saved",
                f"LLM calls     : {len(res.llm_calls)} provider attempts ({sum(not c.ok for c in res.llm_calls)} failed, "
                f"{res.llm_retries} retries incl. invalid-output retries), {res.llm_latency.total_ms / 1000:.1f}s total, avg {res.llm_latency.avg_ms}ms",
                f"Tokens        : prompt {t.prompt_tokens:,}  completion {t.completion_tokens:,}  thinking {t.thinking_tokens:,}  "
                f"cached {t.cached_tokens:,}  total {t.total_tokens:,}   (cost not computed: no price per token configured)",
                f"Tool calls    : {len(res.tool_calls)} ({sum(not c.ok for c in res.tool_calls)} failed), {res.tool_latency.total_ms / 1000:.1f}s total"]
        out.append(f"Coverage      : {res.documents_opened} of {res.documents_total} documents opened by the agent; finish refused {res.finish_refusals}x"
                   + (f"; NOT opened: {', '.join(res.documents_not_opened)}" if res.documents_not_opened else ""))
        if res.error:
            out.append(f"Stopped early : {res.error[:300]}")
    return out


def sec_findings(R) -> list[str]:
    s = R.session
    findings = list(s.scalars(select(Finding).where(Finding.investigation_id == R.result.investigation_id).order_by(Finding.created_at))) if R.result else []
    if not findings:
        return ["No findings were saved in this run.", f"(termination: {R.result.termination if R.result else 'n/a'}; "
                "see the agent trace and failures sections for why)"]
    out = ["Fields follow the Finding schema: title, description, classification, severity, confidence, evidence_status, "
           "verification_status, materiality_reasoning, business_impact, unresolved_questions; evidence rows follow the Evidence schema.",
           f"Findings: {len(findings)}   by classification: {dict(Counter(f.classification for f in findings))}   "
           f"by severity: {dict(Counter(f.severity for f in findings))}", ""]
    for i, f in enumerate(findings, 1):
        out += [f"--- FINDING {i} of {len(findings)}  [{f.id}] " + "-" * 30,
                f"title                 : {f.title}",
                f"classification        : {f.classification}",
                f"severity              : {f.severity}",
                f"confidence            : {f.confidence}",
                f"evidence_status       : {f.evidence_status}",
                f"verification_status   : {f.verification_status}",
                f"description           : {f.description}",
                f"materiality_reasoning : {f.materiality_reasoning or '-'}",
                f"business_impact       : {f.business_impact or '-'}",
                f"unresolved_questions  : {f.unresolved_questions or '-'}",
                "evidence:"]
        rows = s.execute(select(Evidence, DocumentChunk, Document.filename).join(DocumentChunk, DocumentChunk.id == Evidence.chunk_id)
                         .join(Document, Document.id == DocumentChunk.document_id).where(Evidence.finding_id == f.id)).all()
        for j, (e, chunk, filename) in enumerate(rows, 1):
            out += [f"  [{j}] role={e.evidence_role}",
                    f"      document : {filename}",
                    f"      chunk    : {chunk.id}   (chunk_index {chunk.chunk_index})",
                    f"      location : {locator_text(chunk)}",
                    f"      quote    : \"{e.quoted_text}\"",
                    f"      relevance: {e.relevance_reason}",
                    f"      assessment: {e.support_assessment}"]
        out.append("")
    return out


def sec_config(R) -> list[str]:
    s, a = R.settings, R.args
    agent_defaults = inspect.signature(InvestigationAgent.__init__).parameters
    return [
        f"llm provider/model         : {s.llm_provider} / {s.llm_model} (temperature 0, JSON mime type for structured output)",
        f"embedding provider/model   : {s.embedding_provider or s.llm_provider} / {s.embedding_model}, {s.embedding_dimensions} dims"
        + (" (reads at most 256 word pieces, about 1,000 characters; longer text is cut by the model)" if (s.embedding_provider or s.llm_provider) == "local" else ", task types document/query"),
        f"retry attempts (llm+embed) : {a.llm_attempts} (backoff 1s, 2s, 4s...; Google retryDelay honoured; cap 60s per wait)",
        f"chunking                   : merge={'ON' if a.merge_chunks else 'OFF (one unit = one chunk)'}, target {chunking.DEFAULT_TARGET_CHARS}, hard max {chunking.DEFAULT_MAX_CHARS} chars",
        f"parsing                    : {a.extract_workers} parallel worker processes; PDF table detection skipped on pages with more than {pdf_extractor.MAX_DRAWINGS_FOR_TABLES:,} vector paths",
        f"scanned-PDF rule           : picture-only page = under {pdf_extractor.MIN_TEXT_CHARS} chars and image covers >= {pdf_extractor.MIN_IMAGE_COVERAGE:.0%}; "
        f"PDF skipped when >= {pdf_extractor.SCANNED_PAGE_RATIO:.0%} of pages are picture-only",
        f"agent limits               : max steps {a.max_steps or 'none'}, tool budget {a.tool_budget or 'none'}, max findings {a.max_findings or 'none'}; finishing is refused (up to 2 times) while documents are unopened",
        f"agent history window       : last {agent_defaults['full_results'].default} tool results in full, older ones shortened; each result capped at {agent_defaults['max_result_chars'].default:,} chars",
        f"tool limits                : search top_k <= 10; read_section <= 40 chunks / {MAX_SECTION_CHARS:,} chars; quotes >= 8 chars",
        f"focus given to the agent   : {a.focus}",
    ]


def sec_documents(R) -> list[str]:
    scan = R.scan
    out = [f"scan: discovered {scan.discovered}, added {scan.added}, unchanged {scan.unchanged}, missing {scan.missing}",
           f"      duplicates skipped (same content): {scan.skipped_duplicates or 'none'}",
           f"      unsupported types ignored        : {scan.unsupported or 'none'}", ""]
    rows = [[d["file"], d["type"], f"{d['kb']:,}", d["units"], d["status"], d["chunks"], f"{d['chars']:,}", d["median"], d["max"]] for d in R.doc_rows]
    out += table(["file", "type", "KB", "pages/sheets", "status", "chunks", "chars", "median", "max"], rows)

    def sec(d, key):
        return f"{d['t'][key] / 1000:.1f}" if key in d["t"] else "-"

    def msec(d, key):
        return str(d["t"][key]) if key in d["t"] else "-"

    out += ["", "LATENCY OF EACH PART, PER DOCUMENT (the old single 'process' time was all four parts added together):",
            "  extract = read the file into text (local, PyMuPDF/openpyxl)      chunk = split text into chunks (local pure Python)",
            "  save    = write chunks to Postgres (local)                       embed = Google embedding API calls (network)", ""]
    out += table(["file", "extract s", "chunk ms", "save ms", "embed s", "embed calls (failed)", "total s"],
                 [[d["file"], sec(d, "extract"), msec(d, "chunk"), msec(d, "save"), sec(d, "embed"),
                   f"{d['embed_calls']} ({d['embed_failed']})", f"{d['process_ms'] / 1000:.1f}"] for d in R.doc_rows])
    out += ["", "Where the extract time goes (seconds):"]
    detail = []
    for d in R.doc_rows:
        t = d["t"]
        if d["type"] == "pdf":
            parts = [("open", "open"), ("scan_check", "scanned-PDF check"), ("tables", "table detection"), ("text", "text blocks")]
        else:
            parts = [("load_formulas", "load (formulas)"), ("load_values", "load (cached values)"), ("read_rows", "read rows")]
        detail.append(f"  {d['file']}: " + ", ".join(f"{label} {t['extract.' + k] / 1000:.1f}" for k, label in parts if f"extract.{k}" in t))
    out += detail + [""]
    notes = [f"  - {d['file']}: {d['note']}" for d in R.doc_rows if d["note"]]
    errors = [f"  - {d['file']}: ERROR {d['error']}" for d in R.doc_rows if d["error"]]
    return out + ([""] + notes if notes else []) + errors


def sec_chunking(R) -> list[str]:
    s = R.session
    out = [f"Strategy used in this run: {'merged to 1000 chars' if R.args.merge_chunks else 'per-unit (old default)'}; "
           f"chunk size in characters (about 4 chars per token).", ""]
    rows = []
    for ft in ("pdf", "xlsx", "docx"):
        sizes = [n for (n, t) in R.chunk_sizes if t == ft]
        if sizes:
            st = size_stats(sizes)
            rows.append([ft, st["n"], f"{st['total']:,}", st["mean"], st["median"], st["p90"], st["max"], st["lt100"]])
    rows.append(["all", *(lambda st: [st["n"], f"{st['total']:,}", st["mean"], st["median"], st["p90"], st["max"], st["lt100"]])(
        size_stats([n for (n, _) in R.chunk_sizes]))])
    out += table(["type", "chunks", "chars", "mean", "median", "p90", "max", "under 100 chars"], rows)
    out += ["", "Chunking strategies compared (measured locally earlier today on these same files, not part of this run):"]
    out += table(["strategy", "type", "chunks", "median", "mean", "p90", "max", "under 100ch"], [list(r) for r in CHUNK_COMPARISON])
    out += ["", "Reading: the old default turns every PDF text box into its own chunk (median 27 chars), which is too small to embed or quote well.",
            "Merging to 1000 keeps a slide or page together (PDF median 428) and groups spreadsheet rows (median 922) without crossing a page, sheet or heading."]
    return out


def sec_embeddings(R) -> list[str]:
    calls = R.embedder.call_metrics
    n_proc = R.embed_calls_processing
    rows = [[i, "indexing" if i <= n_proc else "agent query", m.items, f"{m.prompt_chars:,}", m.latency_ms, "ok" if m.ok else "FAIL", clip(m.error or "", 60)]
            for i, m in enumerate(calls, 1)]
    out = [f"calls: {len(calls)} ({n_proc} indexing, {len(calls) - n_proc} agent searches), failed {sum(not m.ok for m in calls)}",
           f"texts embedded: {sum(m.items for m in calls)}, characters sent: {sum(m.prompt_chars for m in calls):,}", ""]
    return out + table(["#", "phase", "texts", "chars sent", "latency ms", "result", "error"], rows)


def sec_llm(R) -> list[str]:
    res = R.result
    if not res:
        return ["No investigation ran."]
    rows, n = [], 0
    for step in res.steps:
        for a in step.llm_attempts:
            n += 1
            rows.append([n, step.step, f"{a.started_at - R.agent_start:.1f}", "ok" if a.ok else "FAIL", f"{a.prompt_chars:,}", f"{a.prompt_tokens:,}",
                         f"{a.cached_tokens:,}", a.completion_tokens, a.thinking_tokens, f"{a.total_tokens:,}", a.latency_ms, f"{a.response_chars:,}",
                         clip(a.error or "", 70)])
    ok = [c for c in res.llm_calls if c.ok]
    t = res.token_usage
    lat = sorted(c.latency_ms for c in ok)
    out = [f"provider attempts      : {len(res.llm_calls)}  (ok {len(ok)}, failed {len(res.llm_calls) - len(ok)})",
           f"retries                : {res.llm_retries} (transient errors and invalid-output retries)",
           f"tokens (successful)    : prompt {t.prompt_tokens:,} | completion {t.completion_tokens:,} | thinking {t.thinking_tokens:,} | cached {t.cached_tokens:,} | total {t.total_tokens:,}",
           f"thinking share         : {pct(t.thinking_tokens, t.total_tokens)} of all tokens",
           f"latency (ok calls)     : avg {res.llm_latency.avg_ms}ms  median {lat[len(lat) // 2] if lat else 0}ms  p95 {res.llm_latency.p95_ms}ms  max {res.llm_latency.max_ms}ms  total {res.llm_latency.total_ms / 1000:.1f}s"]
    if ok:
        pc, pt = [c.prompt_chars for c in ok], [c.prompt_tokens for c in ok]
        out += ["", "CONTEXT SIZE of each API call (what was sent: system + prompt):",
                f"  characters : min {min(pc):,}  avg {round(statistics.mean(pc)):,}  max {max(pc):,}",
                f"  tokens     : min {min(pt):,}  avg {round(statistics.mean(pt)):,}  max {max(pt):,}   (as counted by the API)",
                f"  chars per token (measured): {sum(pc) / max(sum(pt), 1):.2f}"]
    out += ["", "Every provider attempt, in order (context = prompt chars / prompt tokens; thinking tokens are hidden reasoning):"]
    return out + table(["#", "step", "at s", "result", "ctx chars", "ctx tokens", "cached", "completion", "thinking", "total tok", "latency ms", "resp chars", "error"], rows)


def make_resolver(session):
    cache: dict = {}

    def chunk_info(chunk_id: str):
        if chunk_id not in cache:
            row = session.execute(select(DocumentChunk, Document.filename).join(Document, Document.id == DocumentChunk.document_id)
                                  .where(DocumentChunk.id == chunk_id)).first()
            cache[chunk_id] = (row[1], locator_text(row[0])) if row else (None, None)
        return cache[chunk_id]

    def doc_name(document_id: str):
        key = ("doc", document_id)
        if key not in cache:
            cache[key] = session.scalar(select(Document.filename).where(Document.id == document_id))
        return cache[key]

    def target(rec) -> tuple[str, str | None]:
        a = rec.arguments or {}
        if rec.tool in ("get_outline", "read_section"):
            name = doc_name(a.get("document_id", ""))
            sel = next((f"{k}={a[k]}" for k in ("page_number", "sheet_name", "heading") if a.get(k) is not None), "")
            return (f"{name or a.get('document_id')} {sel}".strip(), name)
        if rec.tool == "get_chunk":
            f, loc = chunk_info(a.get("chunk_id", ""))
            return (f"{f} @ {loc}" if f else str(a.get("chunk_id")), f)
        if rec.tool == "search":
            return (f"query: {a.get('query')}", None)
        if rec.tool == "calculate":
            return (f"{a.get('expression')}", None)
        if rec.tool == "submit_finding":
            return (f"{a.get('title')} ({len(a.get('evidence') or [])} quotes)", None)
        return ("all documents" if rec.tool == "list_documents" else "", None)

    return target


def sec_tools(R) -> list[str]:
    res = R.result
    if not res:
        return ["No investigation ran."]
    target = make_resolver(R.session)
    step_of = {s.tool_call_index: s.step for s in res.steps if s.tool_call_index}
    per_tool: dict[str, list] = {}
    for c in res.tool_calls:
        per_tool.setdefault(c.tool, []).append(c)
    out = [f"tool calls: {len(res.tool_calls)}  ok {sum(c.ok for c in res.tool_calls)}  failed {sum(not c.ok for c in res.tool_calls)}  "
           f"total {res.tool_latency.total_ms / 1000:.1f}s  (limit {R.args.tool_budget or 'none'})", "", "Per tool:"]
    out += table(["tool", "calls", "ok", "failed", "avg ms", "max ms", "total ms"],
                 [[t, len(cs), sum(c.ok for c in cs), sum(not c.ok for c in cs), round(statistics.mean(c.latency_ms for c in cs)),
                   max(c.latency_ms for c in cs), sum(c.latency_ms for c in cs)] for t, cs in per_tool.items()])
    docs = Counter()
    for c in res.tool_calls:
        _, name = target(c)
        if name:
            docs[name] += 1
    out += ["", "Tool calls that name a specific document (get_outline, read_section, get_chunk), by document:"]
    out += table(["document", "calls"], [[k, v] for k, v in docs.most_common()]) if docs else ["  none"]
    out += ["", "Every tool call, in order (target = the item the call was about):"]
    rows = [[c.index, step_of.get(c.index, "-"), c.tool, clip(target(c)[0], 80), "ok" if c.ok else "FAIL", c.error_code or "", c.latency_ms, clip(c.result_summary, 70)]
            for c in res.tool_calls]
    return out + table(["#", "step", "tool", "target", "result", "error", "ms", "result summary"], rows)


def sec_failures(R) -> list[str]:
    out = []
    res = R.result
    out.append("LLM attempt failures:")
    fails = [(s.step, a) for s in (res.steps if res else []) for a in s.llm_attempts if not a.ok]
    out += [f"  step {st}: {a.latency_ms}ms  {clip(a.error, 200)}" for st, a in fails] or ["  none"]
    out.append("Tool call failures:")
    tf = [c for c in (res.tool_calls if res else []) if not c.ok]
    out += [f"  call {c.index} {c.tool}: {c.error_code}: {clip(c.result_summary, 200)}" for c in tf] or ["  none"]
    out.append("Embedding failures:")
    ef = [(i, m) for i, m in enumerate(R.embedder.call_metrics, 1) if not m.ok]
    out += [f"  call {i}: {clip(m.error, 200)}" for i, m in ef] or ["  none"]
    out.append("Documents not processed:")
    bad = [d for d in R.doc_rows if d["status"] != "processed"]
    out += [f"  {d['file']}: {d['status']} {d['note'] or d['error'] or ''}" for d in bad] or ["  none"]
    if res and res.error:
        out.append(f"Run stopped early by: {res.error}")
    return out


def sec_timing(R) -> list[str]:
    res = R.result
    rows = [[k, f"{v / 1000:.1f}s", pct(v, R.wall_s * 1000)] for k, v in R.stage.items()]
    out = table(["stage", "duration", "share of wall time"], rows)
    if res:
        llm_ms, tool_ms = res.llm_latency.total_ms, res.tool_latency.total_ms
        q_ms = sum(m.latency_ms for m in R.embedder.call_metrics[R.embed_calls_processing:])
        other = max(res.duration_ms - llm_ms - tool_ms, 0)
        out += ["", f"Investigation phase {res.duration_ms / 1000:.1f}s split:",
                f"  LLM provider time   {llm_ms / 1000:.1f}s ({pct(llm_ms, res.duration_ms)})",
                f"  tool time           {tool_ms / 1000:.1f}s ({pct(tool_ms, res.duration_ms)}), of which query embeddings {q_ms / 1000:.1f}s",
                f"  waiting/overhead    {other / 1000:.1f}s ({pct(other, res.duration_ms)}), mostly retry backoff sleeps"]
    return out


def sec_cache(R) -> list[str]:
    res = R.result
    registry = ToolRegistry(ToolContext(session=R.session))
    static = len(SYSTEM) + len(build_prompt(R.args.focus, registry.describe(), [], R.args.max_steps)) - len("Nothing yet. This is your first turn.")
    out = ["Facts (Google docs, fetched 2026-11-11): implicit caching is on by default for Gemini 2.5 and newer; the minimum for gemini-3.6-flash is 4,096 tokens;",
           "  stable content should come first in the prompt; cached tokens are reported by the API (usageMetadata.cachedContentTokenCount).",
           "Our prompt layout: system instruction + task + tool list (stable) -> history (grows; older results get shortened as new ones arrive) -> step counter -> JSON schema.",
           "  Shortening an old result changes the middle of the prompt, so only the prefix before the first change can be a cache hit."]
    if res and res.llm_calls:
        ok = [c for c in res.llm_calls if c.ok]
        ratio = sum(c.prompt_chars for c in ok) / max(sum(c.prompt_tokens for c in ok), 1)
        t = res.token_usage
        out += ["", f"Estimated stable prefix: about {static:,} chars = about {round(static / ratio):,} tokens (using the measured {ratio:.2f} chars per token); minimum needed: 4,096.",
                f"Calls whose whole prompt reached 4,096 tokens: {sum(c.prompt_tokens >= 4096 for c in ok)} of {len(ok)}.",
                f"Measured: cached tokens {t.cached_tokens:,} of {t.prompt_tokens:,} prompt tokens ({pct(t.cached_tokens, t.prompt_tokens)}); calls with any cache hit: {sum(c.cached_tokens > 0 for c in ok)} of {len(ok)}."]
        out.append("Verdict: " + ("implicit caching is working and saved prompt tokens in this run." if t.cached_tokens
                                  else "no cache hits were reported in this run (prefix too small, or the field is not returned for this model)."))
    return out


def sec_trace(R) -> list[str]:
    res = R.result
    if not res:
        return ["No investigation ran."]
    out = [f"termination: {res.termination}   agent summary: {res.summary or '-'}", ""]
    for s in res.steps:
        what = s.tool if s.action == "call_tool" else s.action
        out.append(f"step {s.step:>2} [{what}] {'ok' if s.ok else 'FAIL ' + (s.error_code or '')}  {clip(s.result_summary, 90)}")
        if s.rationale:
            out.append(f"        why: {clip(s.rationale, 220)}")
    return out


# ---- the run --------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None, help="label for what this run tests (used in the file name)")
    ap.add_argument("--max-steps", type=int, default=None, help="default: no limit")
    ap.add_argument("--tool-budget", type=int, default=None, help="default: no limit")
    ap.add_argument("--max-findings", type=int, default=None, help="default: no limit")
    ap.add_argument("--llm-attempts", type=int, default=5)
    ap.add_argument("--focus", default=DEFAULT_FOCUS)
    ap.add_argument("--db-name", default=None)
    ap.add_argument("--merge-chunks", action=argparse.BooleanOptionalAction, default=True,
                    help="merge small neighbouring units into chunks of up to 1000 characters (default on)")
    ap.add_argument("--embedding", choices=["local", "google"], default="local",
                    help="local = all-MiniLM-L6-v2 on this machine via fastembed (default); google = the configured API")
    ap.add_argument("--llm-model", default=None, help="override LLM_MODEL for this run (e.g. a model with free quota left)")
    ap.add_argument("--extract-workers", type=int, default=None, help="documents parsed at once (default: EXTRACT_WORKERS setting, 4)")
    args = ap.parse_args()
    args.extract_workers = args.extract_workers or get_settings().extract_workers

    overrides = {}
    if args.embedding == "local":
        overrides.update(embedding_provider="local", embedding_model=DEFAULT_LOCAL_MODEL, embedding_dimensions=384)
    if args.llm_model:
        overrides["llm_model"] = args.llm_model
    settings = get_settings().model_copy(update=overrides)
    chunking.MERGE_BY_DEFAULT = args.merge_chunks
    started = datetime.now()
    stamp = started.strftime("%Y%m%d-%H%M%S")
    config = re.sub(r"[^A-Za-z0-9._-]+", "-", args.config or f"agent-{settings.llm_model}-embed-{args.embedding}-{'merged1000' if args.merge_chunks else 'perunit'}-steps{args.max_steps or 'unlimited'}")
    R = SimpleNamespace(args=args, settings=settings, started=started, config=config, error=None, result=None, docs=[], doc_rows=[],
                        stage={}, chunk_total=0, chunk_chars=0, chunk_sizes=[], embed_calls_processing=0, agent_start=0.0, db_created=False)
    out_dir = REPO_DIR / "test_results"
    out_dir.mkdir(exist_ok=True)
    path = out_dir / f"TEST-RUN-{stamp}-{config}.txt"
    t_run = time.perf_counter()

    llm = get_llm_client(settings)
    embedder = get_embedding_provider(settings)
    llm.max_attempts = embedder.max_attempts = args.llm_attempts
    R.embedder = embedder
    R.db_name = args.db_name or f"dataroom_run_{stamp.replace('-', '_')}"
    url, R.db_created = ensure_database(settings, R.db_name)
    engine = create_engine(url)
    session = sessionmaker(bind=engine, expire_on_commit=False)()
    R.session = session
    print(f"[run] config={config}\n[run] database={R.db_name}\n[run] results file={path}", flush=True)

    try:
        svc = DocumentService(session, settings)
        t = time.perf_counter()
        R.scan = svc.scan()
        R.stage["scan files"] = ms(time.perf_counter() - t)
        R.docs = list(session.scalars(select(Document).order_by(Document.filename)))
        print(f"[scan] {R.scan.discovered} files, {len(R.docs)} documents, duplicates skipped: {len(R.scan.skipped_duplicates)}", flush=True)

        t = time.perf_counter()
        processing = ProcessingService(session, svc, embedder, on_progress=lambda m, d: print(f"[process] {m}", flush=True),
                                       extract_workers=args.extract_workers)
        try:
            result_list = processing.process_pending()
        finally:
            R.embed_calls_processing = len(embedder.call_metrics)
        results = {r.document_id: r for r in result_list}
        timings = {r.document_id: (sum(r.timings_ms.get(k, 0) for k in ("extract", "chunk", "save", "embed")), r.embed_calls, 0)
                   for r in result_list}
        R.stage["process documents (parse, chunk, save, embed)"] = ms(time.perf_counter() - t)
        R.stage["   of which: parsing all documents in parallel"] = processing.extract_phase_ms
        R.embed_calls_processing = len(embedder.call_metrics)

        chunk_rows = session.execute(select(DocumentChunk.document_id, func.length(DocumentChunk.content))).all()
        by_doc: dict[str, list[int]] = {}
        for doc_id, n in chunk_rows:
            by_doc.setdefault(doc_id, []).append(n)
        for doc in session.scalars(select(Document).order_by(Document.filename)):
            sizes = by_doc.get(doc.id, [])
            st = size_stats(sizes)
            res = results.get(doc.id)
            tm = timings.get(doc.id, (0, 0, 0))
            try:
                units = unit_count(svc.resolve_path(doc), doc.file_type) or "-"
            except Exception:
                units = "-"
            R.doc_rows.append(dict(file=doc.filename, type=doc.file_type, kb=round(doc.file_size / 1024), units=units, status=doc.processing_status,
                                   chunks=st["n"], chars=st["total"], median=st["median"], max=st["max"], process_ms=tm[0],
                                   t=dict(getattr(res, "timings_ms", None) or {}), embed_calls=getattr(res, "embed_calls", 0),
                                   embed_failed=getattr(res, "embed_failed_attempts", 0),
                                   note=getattr(res, "note", None), error=getattr(res, "error", None)))
            R.chunk_sizes += [(n, doc.file_type) for n in sizes]
        R.chunk_total, R.chunk_chars = len(R.chunk_sizes), sum(n for n, _ in R.chunk_sizes)

        if R.chunk_total:
            retrieval = RetrievalService(session, embedder)
            R.agent_start = time.time()
            t = time.perf_counter()
            print(f"[agent] starting: {args.max_steps or 'no'} step limit", flush=True)

            def progress(rec):
                a = rec.llm_attempts[-1] if rec.llm_attempts else None
                ctx = f", context {a.prompt_tokens:,} tok" if a else ""
                print(f"[agent] step {rec.step}: {rec.tool or rec.action} {'ok' if rec.ok else 'FAIL ' + str(rec.error_code)}{ctx}, "
                      f"llm {sum(x.latency_ms for x in rec.llm_attempts)}ms", flush=True)

            R.result = run_investigation(session, llm, retrieval, name=f"full run {stamp}", focus=args.focus, max_steps=args.max_steps,
                                         budget=ToolBudget(max_tool_calls=args.tool_budget, max_findings=args.max_findings), on_progress=progress)
            R.stage["investigation (agent loop)"] = ms(time.perf_counter() - t)
        else:
            R.error = "No chunks were produced, so the investigation was not started."
    except Exception:
        R.error = traceback.format_exc()
        print("[run] ERROR\n" + R.error, flush=True)

    R.finished = datetime.now()
    R.wall_s = time.perf_counter() - t_run
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
    for title, fn in [("AT A GLANCE", sec_glance), ("FINAL REPORT OF DISCREPANCIES (with evidence: document, chunk, location, quote)", sec_findings),
                      ("CONFIGURATION", sec_config), ("DATA ROOM AND DOCUMENT PROCESSING", sec_documents), ("CHUNKING", sec_chunking),
                      ("EMBEDDING CALLS", sec_embeddings), ("LLM CALLS: tokens, context size, latency, failures", sec_llm),
                      ("TOOL CALLS: counts, targets, latency, failures", sec_tools), ("FAILURES", sec_failures), ("TIMING", sec_timing),
                      ("PROMPT CACHING", sec_cache), ("AGENT TRACE (operational summaries only)", sec_trace)]:
        section(title, fn)
    if R.error:
        lines.extend(["", "=" * 78, "Run error", "=" * 78, R.error])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    session.close()
    engine.dispose()
    print(f"[run] DONE in {R.wall_s:.0f}s. Results: {path}", flush=True)


if __name__ == "__main__":
    main()
