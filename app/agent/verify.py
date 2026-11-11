"""Deterministic evidence rules. No LLM involved: citations either exist and meet the rules, or they don't."""
from __future__ import annotations

import re
from itertools import combinations

from ..mcp_server.client import ToolClient, ToolError
from ..models import Citation, Claim, Classification
from .schemas import CitationOut


def verify_citations(tools: ToolClient, cites: list[CitationOut]) -> list[Citation]:
    """Check each quoted passage really exists (via the MCP verify_quote tool) and attach document/page info."""
    out: list[Citation] = []
    for c in cites:
        cit = Citation(chunk_id=c.chunk_id, quote=c.quote)
        try:
            res = tools.call("verify_quote", {"chunk_id": c.chunk_id, "quote": c.quote})
            if res.get("found"):
                cit.chunk_id = res["chunk_id"]
                cit.verified = True
                cit.note = "quote found in a different passage of the same document; reference corrected" if res.get("relocated") else None
                info = tools.call("get_chunk", {"chunk_id": cit.chunk_id}).get("chunk")
                if info:
                    cit.document_id, cit.filename, cit.page = info["document_id"], info["filename"], info["page"]
            else:
                cit.note = res.get("reason", "not verified")
        except ToolError as e:
            cit.note = f"verification unavailable: {e}"
        out.append(cit)
    return out


def check_citation_rule(classification: Classification, cites: list[Citation]) -> tuple[Classification, list[str]]:
    """A comparison (contradiction / contextual / consistent) needs >= 2 verified passages from distinct chunks.
    Otherwise it is downgraded to insufficient_evidence rather than presented as established."""
    notes: list[str] = []
    bad = [c for c in cites if not c.verified]
    if bad:
        notes.append(f"{len(bad)} citation(s) could not be verified and were excluded from support")
    if classification == Classification.INSUFFICIENT_EVIDENCE:
        return classification, notes
    distinct = {c.chunk_id for c in cites if c.verified}
    if len(distinct) < 2:
        notes.append(f"downgraded from {classification.value}: only {len(distinct)} verified passage(s), need 2")
        return Classification.INSUFFICIENT_EVIDENCE, notes
    return classification, notes


def final_status(classification: Classification, confidence: float, min_confidence: float, downgraded: bool) -> str:
    if downgraded or classification == Classification.INSUFFICIENT_EVIDENCE or confidence < min_confidence:
        return "unresolved"
    return "established"


def _tokens(s: str) -> set[str]:
    return set(re.findall(r"[a-z]{3,}", s.lower()))


def heuristic_candidates(claims: list[Claim], min_jaccard: float = 0.5) -> list[list[int]]:
    """Cheap, deterministic comparison candidates: cross-document claims with overlapping subjects but different
    values. Gives recall that does not depend on the LLM's choice of what to compare."""
    toks = {c.id: _tokens(c.subject) for c in claims}
    out: list[list[int]] = []
    for a, b in combinations(claims, 2):
        if a.document_id == b.document_id:
            continue
        ta, tb = toks[a.id], toks[b.id]
        if not ta or not tb or len(ta & tb) / len(ta | tb) < min_jaccard:
            continue
        if a.value.strip().casefold() == b.value.strip().casefold() and (a.unit or "") == (b.unit or ""):
            continue
        out.append(sorted([a.id, b.id]))
    return out
