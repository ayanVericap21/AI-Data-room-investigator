"""Fact-ledger test run: every chunk is read by the LLM, claims go into the ledger (the claims table),
claims are matched across documents, claim groups are judged with quotes verified in code, and every
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
from app.services.claim_grouping import ClaimGrouper  # noqa: E402
from app.services.llama_claims import LlamaClaimExtractor  # noqa: E402
from app.services.missing_items import MissingItemChecker  # noqa: E402
from app.services.record_checks import IdentifierChecker  # noqa: E402
from app.llm.llama_extract import LlamaExtractClient  # noqa: E402
from app.services.claim_service import EXCERPT_TYPES_SKIPPED, ClaimService, ExtractionSummary  # noqa: E402
from app.services.document_service import DocumentService  # noqa: E402
from app.services.processing_service import ProcessingService  # noqa: E402
from app.services.claim_group_judge import ClaimGroupJudge  # noqa: E402
from app.services.doc_profiles import BATCH_CHARS, ProfileBuilder, ProfileStats, render_cards  # noqa: E402
from app.services.group_gates import apply_gates, decide_entity_groups  # noqa: E402
from app.services.internal_checks import InternalChecker  # noqa: E402
from app.services.rule_links import RuleLinker  # noqa: E402
from app.tools import build_registry  # noqa: E402

table, clip, pct = base.table, base.clip, base.pct

# Characters per call for (grouping claims, judging claim groups, describing documents). A hosted model with a large window takes big batches (fewest calls);
# Groq's key allows 8,000 tokens a minute AND per request, and the judge's own instructions are about 3,000 of them, so the batches are sized to fill one
# request as far as the limit allows: instructions + JSON schema + answer reserve + text must stay under 8,000 tokens (about 3 characters each).
DEFAULT_BATCH_CHARS = (300_000, 25_000, BATCH_CHARS)  # Gemini takes a million tokens: a whole room goes to the grouping in one or two calls
BATCH_CHARS_BY_PROVIDER = {"groq": (12_000, 6_500, 12_000)}


class DocumentVerdict(BaseModel):
    verdict: Literal["consistent", "discrepancies_found", "partly_unverified", "nothing_to_compare"]
    reason: str


VERDICT_SYSTEM = """You write the verdict for one document after it was compared with the other documents of the same collection.
Use only the statistics and finding titles given. verdict is one of: consistent (its comparable claims agree with other documents), discrepancies_found (at least one discrepancy involves it), partly_unverified (a large share of its claims had nothing to compare against, or reading was incomplete), nothing_to_compare (no claims could be compared). reason is one short sentence."""


def phase_rows(R):
    """(phase, attempt) for every LLM provider attempt, in order."""
    bounds = R.llm_bounds
    names = ["extract claims", "describe documents", "entity gate", "group claims (LLM)", "link rules to records (LLM)", "judge claim groups", "document verdicts"]
    for i, m in enumerate(R.llm.call_metrics):
        yield names[sum(i >= b for b in bounds)], m  # a call belongs to the first phase whose end it has not passed


# ---- report sections ----------------------------------------------------------
def sec_how(R):
    a, s = R.args, R.settings
    return [
        ("1. Reading: the documents selected for this run were uploaded to LlamaExtract (LlamaIndex Cloud, cost_effective tier), which returns claims with the page and text they came from; "
         "each claim is tied to a stored chunk and kept only if its value appears in that chunk text." if a.extractor == "llamaextract" else
         f"1. Reading: every chunk of every processed document{' except ' + ', '.join(R.excluded) if R.excluded else ''} (database '{R.db_name}') was sent to the LLM in batches of up to {a.batch_chars:,} characters; each batch returns claims, kept only if their value appears in the cited chunk (existing claim extractor, no chunk or call limits)."),
        (f"2. Ledger: accepted claims are stored in the claims table; the claims are sorted by label and sent to the LLM in batches of up to {a.grouping_batch_chars:,} characters, which groups the ones that state the same attribute of the same subject (no code matching, no fixed number of documents or claims assumed)."
         if a.grouping == "llm" else
         f"2. Ledger: accepted claims are stored in the claims table; claims are matched across and within documents by wording and by local embeddings ({s.embedding_model}), without a fixed number of documents or claims assumed."),
        f"3. Judging: a group whose claims all state the same value counts as corroboration; every other group goes to the LLM, several groups to a call, in batches of up to {a.judge_batch_chars:,} characters. The model gives one verdict per group: consistent when all its claims agree, a contradiction when any one differs.",
        "4. Findings: a discrepancy is saved only if its quotes are exact substrings of their chunks (checked in code)"
        + ("; per-document verdicts were skipped by request." if a.skip_verdicts else "; then each document gets a verdict built from its counts."),
        f"5. Model: {s.llm_model}, {'through LlamaExtract' if s.llm_provider == 'llamaextract' else 'temperature 0'}, up to {a.llm_attempts} attempts per call honouring the provider's retry delay; no step, tool-call or finding limits.",
    ]


def sec_glance(R):
    ex, js, res = R.extraction, R.judge, R
    t = R.llm.get_token_usage()
    ok = [m for m in R.llm.call_metrics if m.ok]
    out = ([">>> THIS RUN WAS STOPPED BEFORE IT FINISHED: everything below covers only the part that ran. <<<"] if R.stopped else [])
    lost = [f"{n} {what}" for n, what in (
        (getattr(js, "groups_unjudged", 0), "claim groups were never judged"), (js.batches_failed, "judging call(s) failed"),
        (getattr(getattr(R, "grouping_stats", None), "batches_failed", 0), "grouping call(s) failed"), (getattr(getattr(R, "rule_stats", None), "batches_failed", 0), "rule-linking call(s) failed"),
        (getattr(getattr(R, "internal", None), "failed", 0), "internal-check call(s) failed")) if n]
    if lost:
        out += [f">>> INVALID FOR SCORING: {'; '.join(lost)} (provider errors such as 429 or 503). Missing findings in this report may be outages, not logic errors: run it again. <<<"]
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
            f"{ex.claims_skipped_as_labels} skipped because the value was only the label of its own row, {ex.claims_skipped_by_rule} skipped as a year inside a name or a disclaimer sentence, "
            f"{ex.claims_ellipsis_expanded} values shortened by the extractor with an ellipsis restored from the source text",
            *([f"NO CLAIMS     : {', '.join(ex.documents_without_claims)} (text has values, extractor returned none: not treated as fully read)"] if ex.documents_without_claims else []),
            f"Extraction    : up to {R.args.batch_chunks or 'any number of'} chunks per call (and {R.args.batch_chars:,} characters); running vocabulary {R.args.vocabulary}; "
            f"coverage check {R.args.coverage_check}"
            + (f": {ex.coverage_chunks_checked} chunks had {ex.coverage_numbers} numbers without a claim, {ex.coverage_calls} follow-up calls "
               f"({ex.coverage_calls_failed} failed), {ex.coverage_claims_added} claims added" if R.args.coverage_check == "on" else "")]
    if R.args.extractor == "llamaextract":
        out.append(f"Extractor     : LlamaExtract (cost_effective tier) instead of the LLM: {ex.external_calls} documents uploaded to LlamaIndex Cloud, "
                   f"{ex.external_credits:g} credits used, {ex.documents_failed} documents failed (the batch and coverage settings above do not apply)")
    if R.candidates is not None:
        out.append(f"Claim groups  : {js.groups_total} groups from {len(R.candidates)} candidates; {js.groups_agreed} agreed (every claim states the same value, not sent to the LLM); "
                   f"{js.groups_judged} judged; verdicts {dict(js.verdict_counts)}")
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
    rows = [[r.filename, r.status, r.chunks, f"{r.timings_ms.get('extract', 0) / 1000:.1f}", r.timings_ms.get("chunk", 0), r.timings_ms.get("save", 0),
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
        rows.append([d["file"], d["chunks"], d["readable"] if d["ok"] else 0, "yes" if d["ok"] else "NO", d["claims"], d["calls"],
                     f"{d['total_tokens']:,}", f"{d['ms'] / 1000:.1f}", d["verdict"], clip(d["reason"], 90)])
    out = ["Coverage: 'read' = chunks the LLM was shown. A document counts as fully read only if every batch succeeded (all-or-nothing per document).", ""]
    out += table(["document", "chunks", "read", "fully read", "claims", "LLM calls", "tokens", "extract s", "verdict", "reason"], rows)
    return out


def profile_comparison(R):
    """Each document's profile as the describing calls made it and as the extraction call made it, field by field."""
    if not R.extract_profiles:
        return ["PROFILE FROM THE EXTRACTION CALL: none returned (the extractor is not LlamaExtract, or it left the profile empty)."]
    out = ["", f"PROFILE FROM THE EXTRACTION CALL (no extra call; {len(R.extract_profiles)} of {len(R.profiles) or len(R.extract_profiles)} documents) against the describing calls' profile:"]
    for doc_id, (name, ex) in sorted(R.extract_profiles.items(), key=lambda kv: kv[1][0]):
        mine = R.profiles.get(doc_id)
        out.append(f"  {name}")
        for field in ("main_entity", "document_type", "purpose", "date", "status", "scope"):
            theirs = getattr(mine[1], field, "") if mine and R.profiler is not None and mine[1] is not ex else None
            out.append(f"    {field:<14}: extraction: {clip(getattr(ex, field), 150) or '-'}")
            if theirs is not None:
                out.append(f"    {'':<14}  describing : {clip(theirs, 150) or '-'}")
    return out


def sec_profiles(R):
    pr = R.profiler
    if pr is None:
        return ["Profiles were not made by describing calls in this run." + ("" if R.profiles else " No profiles at all (--no-profiles).")] + (
            ["", "Document cards (shown to the grouping and judging calls):", render_cards(R.profiles)] if R.profiles else []) + profile_comparison(R)
    st: ProfileStats = pr.stats
    describe = [c for c in st.calls if c.kind == "describe"]
    out = [f"A profile describes each whole document. {len(st.records)} documents, {st.total_chars:,} characters of text, sent in {len(describe)} describing call(s) of up to "
           f"{st.budget:,} characters (several documents share a call; a longer document is cut into parts, and one extra call merges the parts of such documents)"
           f"; {len(st.calls)} calls in all.", ""]
    out += table(["call", "kind", "characters", "ok", "documents"], [[i, c.kind, f"{c.chars:,}", "yes" if c.ok else "NO", clip("; ".join(c.documents), 110)] for i, c in enumerate(st.calls, start=1)])
    out += [""] + table(["document", "characters", "parts", "described", "note"], [[r.filename, f"{r.chars:,}", r.parts, "yes" if r.ok else "NO", clip(r.note, 80)] for r in st.records])
    out += ["", "Document cards (shown to the grouping and judging calls):", render_cards(R.profiles) or "  none"]
    g = R.gate
    out += ["", "ENTITY GATE: " + ("off (--no-entity-gate or --no-profiles)" if g is None else f"{'applied' if g.applied else 'not applied'}; {g.note}; {g.calls} call")]
    if g is not None:
        for number, (entity, names) in enumerate(g.groups, start=1):
            out.append(f"  group {number}: {entity}: " + "; ".join(names))
        out += [f"  related groups: {a} and {b}" for a, b in g.related]
    gates = R.gates
    if gates is not None:
        out += ["", f"candidates dropped by the entity gate: {gates.entity_dropped}; dropped as list members (name or statement claims on one page of one document): {gates.list_dropped}"]
        out += [f"  - {x}" for x in gates.log]
    js = R.judge
    out += ["", f"contradictions withdrawn because their own reasoning says the excerpts do not establish one: {js.withdrawn}"]
    out += [f"  - {x}" for x in js.withdrawn_log[:20]]
    out += [f"consistent results counted but not saved as findings (use --save-consistent to keep them): {js.consistent_not_saved}"]
    return out + profile_comparison(R)


def sec_ledger(R):
    ex = R.extraction
    rows = []
    for d in sorted(R.doc_stats.values(), key=lambda x: x["file"]):
        rows.append([d["file"], d["claims"], d["claims_grouped"], d["claims_corroborated"], d["claims_alone"], d["groups_judged"], d["findings"]])
    out = [f"claims recorded {ex.claims_created}, rejected {ex.claims_rejected}", "", "Per document (grouped = is in a claim group; corroborated = same value stated elsewhere; alone = nothing comparable found):"]
    out += table(["document", "claims", "grouped", "corroborated", "alone", "groups judged", "findings"], rows)
    out += ["", "Rejected claims by reason (first 15):"]
    out += [f"  - {r}" for r in ex.rejections[:15]] or ["  none"]
    return out


def sec_groups(R):
    js = R.judge
    out = []
    if R.grouping_stats:
        gs = R.grouping_stats
        out += [f"grouping mode: LLM grouping of sorted claims: {gs.claims} claims in {gs.batches} batch(es), {gs.llm_calls} call(s) ({gs.batches_failed} failed), "
                f"{gs.claims_filtered} claims not sent (no other claim shares their subject or metric words); {gs.groups} groups covering {gs.claims_grouped} claims (largest {gs.largest_group}; {gs.distinct_values} distinct values in all), "
                f"{gs.candidates} candidates ({gs.neighbor_candidates} added from nearest neighbours in other documents), {gs.unknown_refs} unknown refs ignored",
                "  note: claims in different batches are never grouped together" + (" (more than one batch in this run)" if gs.batches > 1 else " (one batch: no such gap in this run)")]
        out += [f"  grouping problem: {e}" for e in gs.errors]
        out.append(f"  candidates by claims that mention the same record id and state the same kind of value: {gs.identifier_candidates}; by subject and metric words that largely coincide: {gs.lexical_candidates}; by a rare word of their labels shared across documents: {gs.label_candidates}; "
                   f"claims that say there is none of something, linked to the claims that mention it: {gs.polarity_candidates}")
    else:
        out.append("grouping mode: code (word overlap and local embeddings)")
    rs = getattr(R, "rule_stats", None)
    if rs is not None:
        out += [f"rule linking: {rs.claims} claims in {rs.batches} batch(es), {rs.llm_calls} call(s) ({rs.batches_failed} failed); {rs.links} rules with records under them, {rs.candidates} rule-to-record links ({rs.temporal_candidates} added by code from dates), {rs.unknown_refs} unknown refs ignored",
                *[f"  rule: {t}" for t in rs.topics[:30]], *[f"  rule linking problem: {e}" for e in rs.errors]]
    else:
        out.append("rule linking: off (--no-rule-links) or not run")
    out += [f"candidates: {len(R.candidates)}", f"  by relation hint : {dict(Counter(c.relation_hint for c in R.candidates))}",
           f"  by match route   : {dict(Counter(c.matched_by for c in R.candidates))}",
           f"  across documents : {R.candidates_cross}   inside one document: {len(R.candidates) - R.candidates_cross}",
           f"claim groups: {js.groups_total}; agreed (every claim states the same value, no LLM): {js.groups_agreed}; series (one value for each period, not compared): {js.groups_series}; document labels only (not compared): {js.groups_labels}; decided in code: {js.groups_basis} basis differences, {js.groups_superseded} superseded editions, {js.groups_currency} currency differences; judged by the LLM: {js.groups_judged} in {js.llm_calls} calls ({js.batches_failed} batches failed)",
           f"LLM verdicts: {dict(js.verdict_counts)}", f"findings saved {js.findings_saved}; rejected {js.findings_rejected}; claims left out of a finding because their quote did not show their value {js.claims_dropped}",
           f"single-group retries {js.retries}; verdicts whose values did not match their claims (echoed values) {js.misaligned}; verdicts without echoed values {js.echo_missing}; "
           f"quotes supplied by code for verdicts that gave none {js.quotes_from_code}",
           f"consistent findings grouped {js.grouped}; findings merged because they state one difference in several documents: {js.merged}", *[f"  - {m}" for m in js.merge_log[:30]],
           "", "Verdicts whose values did not match their claims (first 20):"]
    out += [f"  - {e}" for e in js.echo_log[:20]] or ["  none"]
    out += ["", "Rejected findings (first 15):"]
    out += [f"  - {r}" for r in js.rejections[:15]] or ["  none"]
    return out


def sec_group_log(R):
    """Every claim group (a fact and the claims that may state it), one row per claim with its candidates, and what happened to each, so a missed
    discrepancy can be traced."""
    js = R.judge
    records = js.records
    ids = {i for g in js.groups for i in g.claim_ids}
    claims = {c.id: c for c in R.session.scalars(select(Claim).where(Claim.id.in_(ids)))} if ids else {}
    chunks = {c.id: c for c in R.session.scalars(select(DocumentChunk).where(DocumentChunk.id.in_({c.chunk_id for c in claims.values()})))} if claims else {}
    names = dict(R.session.execute(select(Document.id, Document.filename)).all())

    def where(c):
        ch = chunks.get(c.chunk_id)
        if ch is None:
            return "-"
        return f"sheet {ch.sheet_name}, {ch.cell_reference}" if ch.sheet_name else (f"page {ch.page_number}" if ch.page_number else (ch.heading or ch.content_type))

    outcomes = Counter(r.outcome.split(":")[0].split(",")[0] for r in records.values())
    out = [f"{len(js.groups)} claim groups; {len(records)} candidates between their claims. What happened to the candidates: {dict(outcomes)}"]
    for n, g in enumerate(js.groups, start=1):
        number = {c: i for i, c in enumerate(g.claim_ids, start=1)}
        first = claims.get(g.claim_ids[0])
        status = "all claims state the same value, not sent to the LLM" if g.agreed else ("a rule (claim 1) and the records that may fall under it, judged together" if g.kind == "rule" else "judged together") + (f" ({g.folded} claims folded: they state a value another claim states)" if g.folded else "")
        out += ["", f"GROUP {n}: " + (f"RULE {clip(first.subject or '-', 50)}: {clip(first.metric or '-', 40)}" if g.kind == "rule" and first else clip((first.metric or first.subject or '-') if first else '-', 70)) + f"  ({len(g.claim_ids)} claims; {status})"]
        rows = []
        for c in g.claim_ids:
            cl = claims.get(c)
            mine = [e.candidate_id if e.claim_id == c else e.claim_id for e in g.edges if c in (e.claim_id, e.candidate_id)]
            rows.append([number[c], clip(names.get(cl.document_id, "?") if cl else "?", 34), clip(where(cl), 30) if cl else "-", clip(cl.subject or "-", 40) if cl else "-",
                         clip(cl.metric or "-", 30) if cl else "-", clip(str(cl.value) + (f" {cl.unit}" if cl and cl.unit else ""), 34) if cl else "-",
                         ", ".join(str(number[k]) for k in mine if k in number)])
        out += table(["claim", "document", "where", "subject", "metric", "value", "candidates"], rows)
        for e in g.edges:
            r = records[(e.claim_id, e.candidate_id)]
            out.append(f"  claim {number[e.claim_id]} with claim {number[e.candidate_id]} [{r.hint}, found by {r.route}]: {clip(r.outcome, 170)}")
    return out


def sec_all_claims(R):
    """Every claim of every document in the report, as extraction produced it (before any grouping)."""
    docs = {d.id: d.filename for d in R.session.scalars(select(Document))}
    rows = R.session.execute(select(Claim, DocumentChunk).join(DocumentChunk, DocumentChunk.id == Claim.chunk_id)
                             .order_by(DocumentChunk.document_id, DocumentChunk.chunk_index, Claim.created_at)).all()
    out, current, n = [], None, 0
    for c, chunk in rows:
        if c.document_id != current:
            current, n = c.document_id, 0
            out += ["", f"{docs.get(c.document_id, '?')}: {sum(1 for x, _ in rows if x.document_id == c.document_id)} claims"]
        n += 1
        extra = "".join(f"; {k}={v}" for k, v in (("unit", c.unit), ("period", c.period), ("conditions", c.conditions)) if v not in (None, ""))
        out += [f"  {n:>3}. [{c.claim_category or '-'}] subject={c.subject!r}; metric={c.metric!r}; value={c.value!r}{extra}", f"       sentence: {c.claim_text}"]
    return out or ["no claims"]


def sec_internal(R):
    st = getattr(R, "internal", None)
    if st is None:
        return ["not run (--no-internal-checks)"]
    return [f"claims read: {st.claims}; LLM calls: {st.calls} ({st.failed} failed); relations proposed by the model: {st.proposed}; checked in code (every value a plain number): {st.checked}; held: {st.held}; broken: {st.broken}",
            f"relations found by sentence patterns in code (part of a whole, average and total, a stated change): {st.patterns}; relative dates checked: {st.anchors_checked}; their event dated nowhere: {st.anchors_missing}; findings saved: {st.saved}; rejected: {st.rejected}",
            *[f"  - {x}" for x in st.log[:30]], *[f"  problem: {e}" for e in st.errors]]


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


def sec_records(R):
    r = getattr(R, "records", None)
    if r is None:
        return ["not run"]
    out = [f"identifier kinds with a list in one document and mentions in another: {r.kinds}; identifiers outside a list's range: {r.candidates}; saved: {r.saved}; rejected by the quote check: {r.rejected}"]
    if r.skipped_reason:
        out.append(f"check skipped: {r.skipped_reason}")
    out += [f"  raised: {x}" for x in r.raised[:20]]
    out += [f"  rejected: {x}" for x in r.rejections[:10]]
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
    out.append("Group-judging problems:")
    out += [f"  {e}" for e in R.judge.errors[:10]] or ["  none"]
    return out


def sec_timing(R):
    rows = [[k, f"{v / 1000:.1f}s", pct(v, R.wall_s * 1000)] for k, v in R.stage.items()]
    return table(["stage", "duration", "share of wall time"], rows)


def sec_limits(R):
    return [
        "- Reading is all-or-nothing per document: if one batch fails after all retries, that document's claims are discarded and it counts as not read.",
        "- Candidate matching keeps at most 5 partners per claim (an existing default), so some comparable claims are never judged.",
        "- The claim-extraction prompt lists business categories (capacity, supply, pricing, ...) and is not yet domain-agnostic.",
        "- Claims inside one document are also grouped and judged; matching quality depends on how the model labels subject/metric/unit/period.",
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
        claims_recorded=ex.claims_created, claims_rejected=ex.claims_rejected, candidates=len(R.candidates or []),
        claim_groups=js.groups_total, groups_agreed=js.groups_agreed, groups_judged=js.groups_judged, verdicts=dict(js.verdict_counts),
        findings_saved=js.findings_saved + (R.internal.saved if getattr(R, "internal", None) else 0) + (R.missing.saved if getattr(R, "missing", None) else 0) + (R.records.saved if getattr(R, "records", None) else 0),
        findings_rejected=js.findings_rejected + (R.missing.rejected if getattr(R, "missing", None) else 0) + (R.records.rejected if getattr(R, "records", None) else 0), embedding_calls=len(emb),
        embedding_texts=sum(m.items for m in emb), extractor_credits=ex.external_credits,
        coverage_calls=ex.coverage_calls, coverage_claims_added=ex.coverage_claims_added)
    a, s = R.args, R.settings
    return RunMetrics(
        investigation_id=getattr(R.result, "investigation_id", None), config_name=R.config,
        state="failed" if R.error else "stopped" if R.stopped else "completed",
        started_at=R.started.isoformat(timespec="seconds"), finished_at=R.finished.isoformat(timespec="seconds"),
        wall_s=round(R.wall_s, 1), results_file=str(results_path),
        settings=RunSettings(extractor=a.extractor, grouping=a.grouping, vocabulary=a.vocabulary, coverage_check=a.coverage_check,
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
    ap.add_argument("--judge-batch-chars", type=int, default=None, help="characters of excerpts per claim-group judging call (default: by provider)")
    ap.add_argument("--judge-workers", type=int, default=1,
                    help="grouping and judging calls made at the same time (1 = one after another, the default); the answers are still handled in order")
    ap.add_argument("--dump-claims", action="store_true", help="add every extracted claim of every document to the report (long for a large room)")
    ap.add_argument("--reuse-claims", action="store_true",
                    help="do not read the documents again: group and judge the claims already in the database (no LlamaExtract credits; the missing-items check, which needs the extractor, is skipped)")
    ap.add_argument("--extract-workers", type=int, default=8,
                    help="LlamaExtract jobs running at the same time, over all the documents (1 = one document after another); the results are stored in order")
    ap.add_argument("--pages-per-job", type=int, default=4,
                    help="a PDF longer than this many pages is read by LlamaExtract in page ranges, one job each, started together (0 = one job per document)")
    ap.add_argument("--claim-filter", action="store_true", help="send only the claims that the code's own matching finds a partner for to the grouping calls (off by default: it can withhold claims the LLM would group, such as the same sample worded differently)")
    ap.add_argument("--no-internal-checks", action="store_true", help="do not check arithmetic and part-of-whole relations inside each document, nor relative dates with no anchor")
    ap.add_argument("--no-rule-links", action="store_true", help="do not look for rules (what is prohibited, required, limited or suspended) and the records that fall under them")
    ap.add_argument("--grouping", choices=["code", "llm"], default="llm",
                    help="llm = the LLM groups the sorted claims (default); code = candidates from word overlap and embeddings")
    ap.add_argument("--grouping-batch-chars", type=int, default=None, help="characters of claims per grouping call (--grouping llm; default: by provider)")
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
    ap.add_argument("--profile-batch-chars", type=int, default=None,
                    help="characters of document text per call that describes the documents (several documents share a call; a longer document is cut into parts; default: by provider)")
    ap.add_argument("--profile-source", choices=["both", "extract", "llm"], default="extract",
                    help="where the document profiles come from: llm = calls that describe the documents; extract = the profile the extraction call returns (no describing calls, "
                         "LlamaExtract only; a document it did not describe falls back to a describing call); both = use the describing calls and list the extraction's profile beside them in the report")
    ap.add_argument("--no-profiles", action="store_true", help="do not describe the documents (no document cards, no entity gate)")
    ap.add_argument("--no-entity-gate", action="store_true", help="compare claims of all documents, even ones about unrelated subjects")
    ap.add_argument("--no-list-guard", action="store_true", help="also compare name or statement claims that sit on one page of one document (list items)")
    ap.add_argument("--save-consistent", action="store_true", help="also save consistent results as findings (default: only discrepancies are saved)")
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
    for name, size in zip(("grouping_batch_chars", "judge_batch_chars", "profile_batch_chars"), BATCH_CHARS_BY_PROVIDER.get(settings.llm_provider, DEFAULT_BATCH_CHARS)):
        if getattr(args, name) is None:
            setattr(args, name, size)
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
    if not args.keep_claims and not args.reuse_claims:
        R_cleared = session.execute(delete(Claim)).rowcount or 0  # every run rebuilds its claims from scratch
        session.commit()
        print(f"[run] deleted {R_cleared} existing claims; they will be rebuilt", flush=True)
    def stopped() -> bool:
        return bool(args.stop_file) and os.path.exists(args.stop_file)

    R = SimpleNamespace(args=args, settings=settings, started=started, config=config, db_name=args.db_name, llm=llm, embedder=embedder,
                        error=None, stopped=False, doc_stats={}, excluded=[], claims_cleared=0, scan=None, processed=[], extraction=ExtractionSummary(), candidates=None, candidates_cross=0, judge=None, profiler=None, profiles={}, extract_profiles={}, gate=None, gates=None,
                        stage={}, llm_bounds=[0, 0, 0, 0, 0, 0], grouping_stats=None, rule_stats=None, start_wall=time.time(), session=session)
    from app.services.claim_group_judge import JudgeStats
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
        if args.extractor == "llamaextract" and not args.reuse_claims:
            if not settings.allow_third_party_apis:
                raise RuntimeError("--extractor llamaextract uploads documents to a third-party API, which is disabled (set ALLOW_THIRD_PARTY_APIS=true to allow it)")
            if not settings.llama_cloud_api_key:
                raise RuntimeError("--extractor llamaextract needs LLAMA_CLOUD_API_KEY in the environment")
            extractor = LlamaClaimExtractor(session, LlamaExtractClient(settings.llama_cloud_api_key, project_id=settings.llama_cloud_project_id,
                                                                         tier="cost_effective"), settings.data_room_dir,
                                            on_progress=lambda m, d: print(f"[extract] {m}", flush=True), should_stop=stopped, pages_per_job=args.pages_per_job)
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
        if extractor is not None:
            extractor.prefetch(docs, args.extract_workers)  # every document's job starts now; the loop below stores the results in order
        for doc in docs:
            check_stop()
            chunks_total = session.scalar(select(func.count()).select_from(DocumentChunk).where(DocumentChunk.document_id == doc.id)) or 0
            readable = session.scalar(select(func.count()).select_from(DocumentChunk).where(
                DocumentChunk.document_id == doc.id, DocumentChunk.content_type.not_in(EXCERPT_TYPES_SKIPPED))) or 0
            m0, c0, f0, t0 = len(llm.call_metrics), R.extraction.llm_calls, R.extraction.documents_failed, time.perf_counter()
            error = None
            try:
                if not args.reuse_claims:
                    (extractor or service).extract_document(doc, R.extraction)
            except Exception as exc:  # one document must not end the run
                session.rollback()
                session.execute(delete(Claim).where(Claim.document_id == doc.id))
                session.commit()
                R.extraction.documents_failed += 1
                error = f"{exc.__class__.__name__}: {exc}"
                print(f"[extract] ERROR {doc.filename}: {error}", flush=True)
            attempts = llm.call_metrics[m0:]
            R.doc_stats[doc.id] = dict(
                id=doc.id, file=doc.filename, chunks=chunks_total, readable=readable, ok=R.extraction.documents_failed == f0 and error is None,
                claims=session.scalar(select(func.count()).select_from(Claim).where(Claim.document_id == doc.id)) or 0,
                calls=R.extraction.llm_calls - c0, total_tokens=sum(m.total_tokens for m in attempts if m.ok), ms=round((time.perf_counter() - t0) * 1000),
                error=error, verdict="-", reason="", claims_grouped=0, claims_corroborated=0, claims_alone=0, groups_judged=0, findings=0)
        if extractor is not None:
            extractor.close()
        R.stage["1. read every chunk, extract claims"] = round((time.perf_counter() - t) * 1000)
        R.llm_bounds[0] = R.llm_bounds[1] = R.llm_bounds[2] = len(llm.call_metrics)  # the end of each phase; those that do not run stay empty
        if not args.no_profiles:
            check_stop()
            t_profile = time.perf_counter()
            inputs = []  # (document id, file name, readable chunks): what each document says, for the calls that describe the documents
            for doc in docs:
                readable = [c for c in session.scalars(select(DocumentChunk).where(DocumentChunk.document_id == doc.id).order_by(DocumentChunk.chunk_index))
                            if c.content_type not in EXCERPT_TYPES_SKIPPED]
                inputs.append((doc.id, doc.filename, readable))
            R.extract_profiles = dict(getattr(extractor, "profiles", None) or {})  # described by the extraction calls (LlamaExtract only)
            if args.profile_source == "extract":
                R.profiles = {d: v for d, v in R.extract_profiles.items()}
                inputs = [i for i in inputs if i[0] not in R.profiles]  # a document the extraction did not describe is described by a call
                print(f"[profile] {len(R.profiles)} documents described by the extraction calls; {len(inputs)} left for describing calls", flush=True)
            if inputs:
                R.profiler = ProfileBuilder(llm, args.profile_batch_chars, workers=args.judge_workers, on_progress=lambda m: print(f"[profile] {m}", flush=True), should_stop=stopped)
                R.profiles.update(R.profiler.run(inputs))
                print(f"[profile] {len(inputs)} documents sent to describing calls: {len(R.profiler.stats.calls)} calls", flush=True)
            R.stage["1b. describe the documents (profiles)"] = round((time.perf_counter() - t_profile) * 1000)
            R.llm_bounds[1] = R.llm_bounds[2] = len(llm.call_metrics)
        if R.profiles and not args.no_entity_gate:
            check_stop()
            t_gate = time.perf_counter()
            R.gate = decide_entity_groups(llm, R.profiles)  # one call: which documents are about the same subject
            print(f"[gate] entity gate: {R.gate.note}", flush=True)
            R.stage["1c. entity gate (which documents are about the same subject)"] = round((time.perf_counter() - t_gate) * 1000)
            R.llm_bounds[2] = len(llm.call_metrics)

        check_stop()
        # ---- stage 2: match claims
        t = time.perf_counter()
        print("[match] matching claims across documents", flush=True)
        if args.grouping == "llm":
            R.candidates, R.grouping_stats = ClaimGrouper(session, llm, max_chars_per_call=args.grouping_batch_chars, should_stop=stopped, embedder=embedder, prefilter=args.claim_filter,
                                                workers=args.judge_workers, on_progress=lambda m: print(f"[match] {m}", flush=True),
                                                cards=render_cards(R.profiles) if R.profiles else "").run()
        else:
            R.candidates = service.candidates(embedder)
        R.llm_bounds[3] = len(llm.call_metrics)
        if not args.no_rule_links and args.grouping == "llm":
            check_stop()
            R.rule_candidates, R.rule_stats = RuleLinker(session, llm, max_chars_per_call=args.grouping_batch_chars, should_stop=stopped, workers=args.judge_workers,
                                                         on_progress=lambda m: print(f"[rules] {m}", flush=True), cards=render_cards(R.profiles) if R.profiles else "").run()
            R.candidates = list(R.candidates) + R.rule_candidates
            print(f"[rules] {R.rule_stats.links} rules with records under them, {R.rule_stats.candidates} rule-to-record links", flush=True)
        R.llm_bounds[4] = len(llm.call_metrics)
        R.gates = apply_gates(session, R.candidates, R.gate, list_guard=not args.no_list_guard)
        R.candidates = R.gates.candidates
        print(f"[gate] {R.gates.entity_dropped} candidates dropped by the entity gate, {R.gates.list_dropped} as list members; {len(R.candidates)} remain", flush=True)
        claim_doc = dict(session.execute(select(Claim.id, Claim.document_id)).all())
        R.candidates_cross = sum(claim_doc[c.claim_id] != claim_doc[c.candidate_id] for c in R.candidates)
        R.stage["2. match claims (" + ("LLM grouping of sorted claims" if args.grouping == "llm" else "code + local embeddings") + ")"] = round((time.perf_counter() - t) * 1000)
        print(f"[match] {len(ClaimGroupJudge._form_groups(R.candidates))} claim groups from {len(R.candidates)} candidates (all the claims of a group are judged together)", flush=True)

        check_stop()
        # ---- stage 3: judge claim groups
        t = time.perf_counter()
        registry = build_registry(session, None, investigation.id)
        R.judge = ClaimGroupJudge(session, llm, registry, max_chars_per_call=args.judge_batch_chars, should_stop=stopped, workers=args.judge_workers,
                            on_progress=lambda m, d: print(f"[judge] {m}", flush=True), save_consistent=args.save_consistent,
                            profiles=R.profiles).run(R.candidates)
        R.stage["3. judge claim groups, verify quotes, save findings"] = round((time.perf_counter() - t) * 1000)
        R.llm_bounds[5] = len(llm.call_metrics)

        check_stop()
        if not args.no_internal_checks:
            check_stop()
            t = time.perf_counter()
            R.internal = InternalChecker(session, llm, registry, max_chars_per_call=args.grouping_batch_chars, should_stop=stopped, workers=args.judge_workers,
                                         on_progress=lambda m: print(f"[internal] {m}", flush=True)).run()
            R.stage["3a. checks inside each document (arithmetic, part of whole, relative dates)"] = round((time.perf_counter() - t) * 1000)
            print(f"[internal] {R.internal.patterns} found by sentence patterns, {R.internal.proposed} relations proposed by the model, {R.internal.checked} checkable, {R.internal.broken} broken; "
                  f"{R.internal.anchors_missing} relative dates with no anchor; {R.internal.saved} saved", flush=True)
        # ---- stage 3b: items a document refers to that the data room does not hold (no LLM call; the references came with the claims)
        if extractor is not None:
            t = time.perf_counter()
            R.missing = MissingItemChecker(session, registry).run(extractor.references, not_fully_read=R.extraction.documents_without_claims)
            R.stage["3b. referenced items absent from the data room"] = round((time.perf_counter() - t) * 1000)
            print(f"[missing] {R.missing.references} references, {R.missing.present} found in the room, {R.missing.saved} saved as missing information"
                  + (f" (check skipped: {R.missing.skipped_reason})" if R.missing.skipped_reason else ""), flush=True)

        check_stop()
        # ---- stage 3c: identifiers named in one document that another document's list of the same kind does not hold (no LLM call)
        t = time.perf_counter()
        R.records = IdentifierChecker(session, registry).run(not_fully_read=R.extraction.documents_without_claims)
        R.stage["3c. identifier reconciliation across documents"] = round((time.perf_counter() - t) * 1000)
        print(f"[records] {R.records.candidates} identifiers outside the range of another document's list, {R.records.saved} saved"
              + (f" (check skipped: {R.records.skipped_reason})" if R.records.skipped_reason else ""), flush=True)

        check_stop()
        # ---- stage 4: per-document verdicts
        t = time.perf_counter()
        claims_by_doc: dict[str, set[str]] = {}
        for cid, did in claim_doc.items():
            claims_by_doc.setdefault(did, set()).add(cid)
        judged_by_doc = Counter()
        for g in R.judge.groups:
            if not g.agreed:
                for did in {claim_doc[i] for i in g.claim_ids}:
                    judged_by_doc[did] += 1
        for d in R.doc_stats.values():
            mine = claims_by_doc.get(d["id"], set())
            d.update(claims_grouped=len(mine & R.judge.grouped_claims), claims_corroborated=len(mine & R.judge.corroborated_claims),
                     claims_alone=len(mine - R.judge.grouped_claims), groups_judged=judged_by_doc[d["id"]], findings=R.judge.findings_by_document[d["id"]])
            if not d["ok"]:
                d["verdict"], d["reason"] = "partly_unverified", "reading was incomplete, so its claims were not compared"
                continue
            if args.skip_verdicts:
                d["verdict"], d["reason"] = "skipped", "verdicts skipped by request (--skip-verdicts)"
                continue
            titles = R.judge.finding_titles_by_document.get(d["id"], [])
            prompt = (f"Document: {d['file']}\nChunks in the document: {d['chunks']}; chunks read by the model: {d['readable']}.\n"
                      f"Claims recorded: {d['claims']}; in a claim group: {d['claims_grouped']}; stated elsewhere with the same value: {d['claims_corroborated']}; "
                      f"with nothing comparable elsewhere: {d['claims_alone']}.\nClaim groups judged that include this document: {d['groups_judged']}; discrepancies saved: {d['findings']}.\n"
                      + ("Discrepancy titles:\n" + "\n".join(f"- {x}" for x in titles) if titles else "No discrepancies were saved for this document."))
            try:
                v = llm.generate_structured_output(prompt, DocumentVerdict, system=VERDICT_SYSTEM)
                d["verdict"], d["reason"] = v.verdict, v.reason
            except LLMError as exc:
                d["verdict"], d["reason"] = "partly_unverified", f"verdict call failed: {str(exc)}"
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
                      ("COVERAGE AND PER-DOCUMENT VERDICTS", sec_documents), ("DOCUMENT PROFILES AND CANDIDATE GATES", sec_profiles), ("CLAIM LEDGER", sec_ledger), ("CLAIM GROUPING AND JUDGING", sec_groups), ("EVERY CLAIM GROUP, ITS CLAIMS AND WHAT HAPPENED TO THEIR CANDIDATES", sec_group_log),
                      ("CHECKS INSIDE EACH DOCUMENT (arithmetic, part of whole, relative dates)", sec_internal),
                      ("REFERENCED ITEMS NOT IN THE DATA ROOM (single-claim missing information)", sec_missing),
                      ("IDENTIFIER RECONCILIATION ACROSS DOCUMENTS", sec_records),
                      ("LLM CALLS: tokens, context size, latency, failures", sec_llm), ("EMBEDDING CALLS", sec_embeddings), ("FAILURES", sec_failures),
                      ("TIMING", sec_timing), ("KNOWN LIMITS OF THIS RUN", sec_limits),
                      *([("EVERY CLAIM EXTRACTED FROM EVERY DOCUMENT (what extraction produced, before grouping)", sec_all_claims)] if R.args.dump_claims else [])]:
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
