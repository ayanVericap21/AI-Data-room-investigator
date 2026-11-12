"""Identifier reconciliation: a record named in one document that another document's list of the same kind of record does not hold.

Claims compare one attribute at a time, so they cannot notice that a lab report names sample SPL-044-B when the field register lists
SPL-001 to SPL-038. This check reads the identifiers straight from the text (letters, a separator, digits, an optional suffix: SPL-044-B,
D-04, PO-1032) and groups them by kind (the letters and separator). For one kind, a document that lists at least ``MIN_LIST`` identifiers is
a list. An identifier of that kind in another document whose number lies outside the list's range (below its lowest or above its
highest) has no record in the list: it is saved as ``missing_information`` with the line that names it and the line that shows the list's
end, through ``submit_finding`` (so both quotes are verified like every other).

An identifier inside the list's range that the list lacks is not raised: documents often hold different subsets of a series.
Nothing is raised unless the whole room was read.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Document, DocumentChunk
from app.models.base import ProcessingStatus
from app.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

MIN_LIST = 5  # distinct identifiers of one kind that make a document's own list of that kind
MAX_FINDINGS = 15
MAX_LISTED = 5  # identifiers named in one finding
_ID = re.compile(r"(?<![A-Za-z0-9])([A-Za-z]{1,6})([-_])(\d{2,6})(?:[-_]([A-Za-z0-9]{1,4}))?(?![A-Za-z0-9])")
_NOT_AN_ID = {"table", "tab", "fig", "figure", "page", "pg", "section", "sec", "annex", "appendix", "version", "ver", "rev", "no", "ref", "item",
              "step", "ch", "chapter", "clause", "art", "para", "line", "row", "col", "iso", "pp", "vol", "eq", "ex"}


@dataclass(frozen=True)
class Mention:
    ident: str  # as written, e.g. SPL-044-B
    number: int
    chunk_id: str
    line: str  # the line of the chunk that holds it, as stored


@dataclass
class RecordStats:
    kinds: int = 0  # kinds of identifier found in at least two documents with a list in one of them
    candidates: int = 0
    saved: int = 0
    rejected: int = 0
    skipped_reason: str = ""
    raised: list[str] = field(default_factory=list)
    rejections: list[str] = field(default_factory=list)


def mentions_in(content: str, chunk_id: str) -> list[tuple[tuple[str, str], Mention]]:
    out = []
    for line in (content or "").splitlines():
        for m in _ID.finditer(line):
            prefix, sep, digits, _suffix = m.groups()
            if prefix.lower() in _NOT_AN_ID or (len(digits) == 4 and 1900 <= int(digits) <= 2100):
                continue  # a section label or a year, not a record
            out.append(((prefix.upper(), sep), Mention(m.group(0), int(digits), chunk_id, line)))
    return out


class IdentifierChecker:
    def __init__(self, session: Session, registry: ToolRegistry):
        self.session = session
        self.registry = registry

    def run(self, not_fully_read: list[str] | None = None) -> RecordStats:
        stats = RecordStats()
        docs = [d for d in self.session.scalars(select(Document)) if d.processing_status != ProcessingStatus.missing.value]
        unread = [d.filename for d in docs if d.processing_status != ProcessingStatus.processed.value] + list(not_fully_read or [])
        if unread:
            stats.skipped_reason = f"{len(unread)} document(s) are not fully processed ({', '.join(unread[:3])}), so a record could be in one of them"
            return stats
        names = {d.id: d.filename for d in docs}
        # kind -> document id -> identifier (upper case) -> first mention
        found: dict[tuple[str, str], dict[str, dict[str, Mention]]] = {}
        for chunk in self.session.scalars(select(DocumentChunk).where(DocumentChunk.document_id.in_(names)).order_by(DocumentChunk.document_id, DocumentChunk.chunk_index)):
            for kind, mention in mentions_in(chunk.content, chunk.id):
                found.setdefault(kind, {}).setdefault(chunk.document_id, {}).setdefault(mention.ident.upper(), mention)
        candidates: list[tuple[str, str, tuple[str, str], list[Mention], Mention]] = []
        for kind, by_doc in found.items():
            if len(by_doc) < 2:
                continue
            lists = {d: ids for d, ids in by_doc.items() if len(ids) >= MIN_LIST}
            if not lists:
                continue
            stats.kinds += 1
            for list_doc, ids in lists.items():
                low = min(ids.values(), key=lambda m: m.number)
                high = max(ids.values(), key=lambda m: m.number)
                for doc, others in by_doc.items():
                    if doc == list_doc:
                        continue
                    outside = sorted((m for m in others.values() if m.number < low.number or m.number > high.number), key=lambda m: m.number)
                    if outside:
                        candidates.append((doc, list_doc, kind, outside, high if outside[0].number > high.number else low))
        stats.candidates = sum(len(c[3]) for c in candidates)
        for doc, list_doc, kind, outside, edge in candidates[:MAX_FINDINGS]:
            self._save(names, doc, list_doc, kind, outside, edge, found[kind][list_doc], stats)
        return stats

    def _save(self, names: dict[str, str], doc: str, list_doc: str, kind: tuple[str, str], outside: list[Mention], edge: Mention,
              ids: dict[str, Mention], stats: RecordStats) -> None:
        listed = outside[:MAX_LISTED]
        low = min(m.number for m in ids.values())
        high = max(m.number for m in ids.values())
        first = min(ids.values(), key=lambda m: m.number).ident
        last = max(ids.values(), key=lambda m: m.number).ident
        title = f"{', '.join(m.ident for m in listed)} named in {names[doc]} has no record in {names[list_doc]} ({first} to {last})"
        evidence, seen = [], set()
        for m in listed:
            quote = m.line.strip()[:300]
            if (m.chunk_id, quote) not in seen:
                seen.add((m.chunk_id, quote))
                evidence.append({"chunk_id": m.chunk_id, "quoted_text": quote, "role": "insufficient",
                                 "relevance_reason": f"Names {m.ident}", "support_assessment": f"{names[doc]} names {m.ident}; it is outside the {first} to {last} range listed in {names[list_doc]}."})
        edge_quote = edge.line.strip()[:300]
        if (edge.chunk_id, edge_quote) not in seen:
            evidence.append({"chunk_id": edge.chunk_id, "quoted_text": edge_quote, "role": "insufficient",
                             "relevance_reason": f"The list in {names[list_doc]} ends at {edge.ident}",
                             "support_assessment": f"{names[list_doc]} lists {len(ids)} identifiers of this kind, from {first} to {last}."})
        stats.raised.extend(m.ident for m in listed)
        result = self.registry.execute("submit_finding", {
            "title": title[:300], "classification": "missing_information", "severity": "medium", "confidence": 0.5,
            "evidence_status": "unavailable",
            "description": (f"{names[doc]} names {', '.join(m.ident for m in listed)}, but {names[list_doc]}, which lists {len(ids)} identifiers of the same kind "
                            f"({first} to {last}), has no record numbered {'that' if len(listed) == 1 else 'those'} ({low} to {high} is its range). "
                            "Found by comparing identifiers, not by reading the claims: confirm that the two documents are meant to cover the same set."),
            "unresolved_questions": [f"Where is the record for {m.ident}?" for m in listed],
            "evidence": evidence,
        })
        if not result.ok:
            stats.rejected += 1
            stats.rejections.append(f"{title[:60]}: {result.error.code}: {result.error.message[:160]}")
            return
        stats.saved += 1
