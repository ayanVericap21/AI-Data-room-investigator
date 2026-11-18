"""submit_finding: the only way the agent writes a finding, and where fabrication is stopped.

Every quote is checked here, in code, against the stored chunk text. A submission with any
unverifiable quote is rejected whole and nothing is saved, so every stored finding is fully
verified. The stored quote is always the exact source text, never the model's rendering.
"""
from __future__ import annotations

import logging
import re
import unicodedata

from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError

from app.models import Claim, Document, DocumentChunk, Evidence, Finding, FindingClaim, Investigation, InvestigationEvent
from app.models.base import Classification
from app.services.numbers import numbers_in_text, value_supported_by_text
from app.tools.base import ToolContext, ToolError
from app.tools.reading import locator_text
from app.tools.schemas import StoredEvidence, SubmitFindingInput, SubmitFindingOutput

logger = logging.getLogger(__name__)

# Classifications that compare two statements need at least two different quoted passages (from the same chunk or not).
NEEDS_TWO_PASSAGES = {Classification.contradiction, Classification.contextual_difference, Classification.consistent}


def find_source_span(content: str, quote: str) -> tuple[str, bool] | None:
    """Locate ``quote`` in ``content``. Returns (exact source text, matched_after_whitespace_normalisation).

    An exact substring match is preferred. Otherwise runs of whitespace are treated as
    equivalent (PDF and table text often differs from what a model reproduces only by line
    breaks), then so are the differences that parsing and copying introduce without changing the words: Unicode
    forms (ligatures, non-breaking spaces), curly vs straight quotes, dash variants, zero-width characters, a
    word hyphenated across a line break and the pipes and the HTML tags that separate table cells. The verbatim source span is returned.
    Any other difference fails."""
    quote = quote.strip()
    if not quote:
        return None
    if quote in content:
        return quote, False
    words = quote.split()
    match = re.search(r"\s+".join(re.escape(w) for w in words), content)
    if match:
        return match.group(0), True
    folded_content, positions = _fold(content)
    folded_quote, _ = _fold(quote)
    folded_quote = folded_quote.strip()
    if not folded_quote:
        return None
    at = folded_content.find(folded_quote)
    if at < 0:
        return None
    start, end = positions[at], positions[at + len(folded_quote) - 1] + 1
    return content[start:end], True


_QUOTE_AND_DASH_VARIANTS = str.maketrans({"‘": "'", "’": "'", "‚": "'", "‛": "'", "“": '"', "”": '"',
                                          "„": '"', "‟": '"', "‐": "-", "‑": "-", "‒": "-", "–": "-",
                                          "—": "-", "―": "-", "−": "-"})
_TAG = re.compile(r"</?[A-Za-z][^<>]*>")  # an HTML tag of a table chunk (<td>, </tr>): a cell separator like the pipe, not text the model would quote
_INVISIBLE = {"­", "​", "‌", "‍", "⁠", "﻿"}


def _fold(text: str) -> tuple[str, list[int]]:
    """``text`` with Unicode forms, quote and dash variants, zero-width characters and line-break hyphenation normalised and
    whitespace collapsed, plus, for each character of the result, the index of the original character it came from."""
    out: list[str] = []
    positions: list[int] = []
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if ch in _INVISIBLE:
            i += 1
            continue
        if ch == "<":
            tag = _TAG.match(text, i)
            if tag:
                if out and out[-1] != " ":
                    out.append(" ")
                    positions.append(i)
                i = tag.end()
                continue
        if ch in "-‐‑" and i + 1 < n and text[i + 1] in "\r\n" and out and out[-1].isalpha():  # "wa-\nfer" -> "wafer"
            j = i + 1
            while j < n and text[j].isspace():
                j += 1
            if j < n and text[j].isalpha():
                i = j
                continue
        for piece in unicodedata.normalize("NFKC", ch).translate(_QUOTE_AND_DASH_VARIANTS):
            if piece.isspace() or piece == "|":  # a table's cell separator: "Developer fee | $2,750,000" and "Developer fee $2,750,000" are the same words
                if out and out[-1] != " ":
                    out.append(" ")
                    positions.append(i)
            else:
                out.append(piece)
                positions.append(i)
        i += 1
    return "".join(out), positions


_CLAIM_RELATION = {"supporting": "supports", "contradicting": "contradicts", "contextual": "contextualizes", "insufficient": "contextualizes"}
MAX_CLAIMS_PER_EVIDENCE = 5


def _same_numbers_problem(verified: list, classification: Classification) -> str | None:
    """A contradiction whose quotes all state exactly the same numbers is not a numeric contradiction. Conservative: it only fires when every quote
    has numbers and the sets are identical, so a difference in scale words, units or dates is never rejected here (compare_values decides those)."""
    if classification != Classification.contradiction or len(verified) < 2:
        return None
    sets = [{n.normalize() for n in numbers_in_text(exact)} for (_i, _chunk, _filename, exact, _norm) in verified]
    if all(sets) and all(s == sets[0] for s in sets):
        shown = ", ".join(sorted(format(n, "f") for n in sets[0]))
        return (f"every quoted passage states the same numbers ({shown}), so the quotes do not show a numeric contradiction. "
                "If the difference is in units, scale, dates or wording, check it with compare_values and classify accordingly")
    return None


def _duplicate_of(ctx: ToolContext, classification: str, chunk_ids: set[str]) -> str | None:
    """The id of a finding already saved in this investigation with the same classification and the same evidence chunks."""
    rows = ctx.session.execute(
        select(Finding.id, Finding.classification, Evidence.chunk_id)
        .join(Evidence, Evidence.finding_id == Finding.id).where(Finding.investigation_id == ctx.investigation_id)
    ).all()
    chunks_of: dict[str, set[str]] = {}
    classes: dict[str, str] = {}
    for finding_id, cls, chunk_id in rows:
        chunks_of.setdefault(finding_id, set()).add(chunk_id)
        classes[finding_id] = cls
    return next((fid for fid, chunks in chunks_of.items() if classes[fid] == classification and chunks == chunk_ids), None)


def submit_finding(ctx: ToolContext, inp: SubmitFindingInput) -> SubmitFindingOutput:
    if ctx.investigation_id is None or ctx.session.get(Investigation, ctx.investigation_id) is None:
        raise ToolError("no_investigation", "No investigation is attached to this run, so findings cannot be saved.")
    if ctx.budget.max_findings is not None and len(ctx.finding_ids) >= ctx.budget.max_findings:
        raise ToolError("finding_budget_exhausted", f"This run may submit at most {ctx.budget.max_findings} findings.")

    chunk_ids = {e.chunk_id for e in inp.evidence}
    rows = {
        chunk.id: (chunk, filename)
        for chunk, filename in ctx.session.execute(
            select(DocumentChunk, Document.filename).join(Document, Document.id == DocumentChunk.document_id)
            .where(DocumentChunk.id.in_(chunk_ids))
        )
    }

    problems: list[str] = []
    verified: list[tuple[int, DocumentChunk, str, str, bool]] = []  # (index, chunk, filename, exact_text, normalised)
    for i, ev in enumerate(inp.evidence, start=1):
        if ev.chunk_id not in rows:
            problems.append(f"evidence {i}: chunk '{ev.chunk_id}' does not exist (use ids from search or read_section)")
            continue
        chunk, filename = rows[ev.chunk_id]
        span = find_source_span(chunk.content, ev.quoted_text)
        if span is None:
            problems.append(f"evidence {i}: the quote is not in chunk '{ev.chunk_id}' ({filename}, {locator_text(chunk)}). "
                            "Copy the text exactly from that chunk's content.")
            continue
        verified.append((i, chunk, filename, span[0], span[1]))
    distinct = {(e.chunk_id, " ".join(e.quoted_text.split())) for e in inp.evidence}
    if len(distinct) < 2 and inp.classification in NEEDS_TWO_PASSAGES:
        problems.append(f"a {inp.classification.value} compares statements, so at least two different quoted passages "
                        "are required")
    if not problems and ctx.agent_checks:
        same = _same_numbers_problem(verified, inp.classification)
        if same:
            problems.append(same)
        duplicate = _duplicate_of(ctx, inp.classification.value, chunk_ids)
        if duplicate:
            problems.append(f"this finding repeats {duplicate} (same classification, same evidence passages); submit a different finding or finish")
    if problems:
        raise ToolError("validation_failed", "Finding not saved. " + "; ".join(problems))

    finding = Finding(
        investigation_id=ctx.investigation_id, title=inp.title, description=inp.description,
        classification=inp.classification.value, severity=inp.severity.value, confidence=inp.confidence,
        materiality_reasoning=inp.materiality_reasoning, business_impact=inp.business_impact,
        evidence_status=inp.evidence_status.value, verification_status="verified",
        unresolved_questions=list(inp.unresolved_questions),
    )
    stored: list[StoredEvidence] = []
    notes: list[str] = []
    linked_claims: dict[str, str] = {}  # claim id -> relationship, in the order found
    try:
        ctx.session.add(finding)
        ctx.session.flush()
        for (i, chunk, filename, exact_text, normalised), ev in zip(verified, inp.evidence):
            row = Evidence(
                finding_id=finding.id, chunk_id=chunk.id, evidence_role=ev.role.value, quoted_text=exact_text,
                relevance_reason=ev.relevance_reason, support_assessment=ev.support_assessment,
            )
            matched = [c for c in ctx.session.scalars(select(Claim).where(Claim.chunk_id == chunk.id))
                       if c.value is not None and value_supported_by_text(c.value, exact_text)][:MAX_CLAIMS_PER_EVIDENCE]
            if matched:  # the claims table is linked, not ignored: the evidence row and finding_claims point at the claims the quote states
                row.claim_id = matched[0].id
                for c in matched:
                    linked_claims.setdefault(c.id, _CLAIM_RELATION[ev.role.value])
            ctx.session.add(row)
            ctx.session.flush()
            stored.append(StoredEvidence(evidence_id=row.id, chunk_id=chunk.id, filename=filename, claim_ids=[c.id for c in matched],
                                         locator=locator_text(chunk), quoted_text=exact_text))
            if normalised:
                notes.append(f"evidence {i}: quote matched after whitespace normalisation; the exact source text was stored")
        for claim_id, relationship in linked_claims.items():
            ctx.session.add(FindingClaim(finding_id=finding.id, claim_id=claim_id, relationship_type=relationship))
        ctx.session.add(InvestigationEvent(
            investigation_id=ctx.investigation_id, event_type="finding_saved", node_name="submit_finding",
            message=f"Finding saved: {inp.title}", status="success",
            details={"finding_id": finding.id, "classification": inp.classification.value, "severity": inp.severity.value,
                     "evidence_count": len(stored), "searches_since_last_finding": list(ctx.recent_queries),
                     "calculations": list(ctx.calculations), "claim_ids": list(linked_claims)},
        ))
        ctx.session.commit()
    except SQLAlchemyError as exc:
        ctx.session.rollback()
        logger.exception("Could not persist finding")
        raise ToolError("persist_failed", "The finding could not be saved. Nothing was stored.") from exc

    ctx.finding_ids.append(finding.id)
    ctx.recent_queries.clear()
    ctx.calculations.clear()
    return SubmitFindingOutput(finding_id=finding.id, verification_status="verified", evidence=stored, notes=notes)
