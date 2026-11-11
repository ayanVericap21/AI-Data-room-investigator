"""Single-claim missing-information check: an item a document says exists elsewhere, but the data room does not hold.

Extraction returns, for each document, the items it refers to (an annex, schedule, attachment, approval, report ...) with the
sentence that refers to them. Here code decides whether each item exists in the room, by looking at file names, sheet names,
headings and the first line of every other chunk. An item that is nowhere is saved as a ``missing_information`` finding with
the quoted sentence as its evidence, through ``submit_finding`` (so the quote is verified like every other).

Nothing is raised unless the whole room was read: a document that failed, is image-only, is still pending or was skipped might
hold the item, so the check stays silent and says why.
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

_FILLER = {"the", "of", "and", "to", "for", "in", "on"}  # not "a": it names things ("Schedule A")


@dataclass(frozen=True)
class DocReference:
    document_id: str
    chunk_id: str
    item: str  # the referenced item as the document names it
    quote: str  # the exact sentence that refers to it, as stored in its chunk


@dataclass
class MissingStats:
    references: int = 0
    present: int = 0
    saved: int = 0
    rejected: int = 0
    duplicates: int = 0
    skipped_reason: str = ""  # set when the check did not run
    missing_items: list[str] = field(default_factory=list)
    rejections: list[str] = field(default_factory=list)


def _words(text: str | None) -> frozenset[str]:
    return frozenset(w for w in re.findall(r"[a-z0-9]+", (text or "").lower()) if w not in _FILLER)


def _item_keys(item: str) -> list[frozenset[str]]:
    """The word sets an existing document, sheet or heading must contain: the item as written, and its name without a trailing
    description in brackets ("Schedule A (technical specification)" is also found as "Schedule A")."""
    keys = [_words(item)]
    head = re.split(r"[(\[]", item, maxsplit=1)[0].strip()
    if head and head != item:
        keys.append(_words(head))
    return [k for k in keys if k]


class MissingItemChecker:
    def __init__(self, session: Session, registry: ToolRegistry):
        self.session = session
        self.registry = registry

    def run(self, references: list[DocReference]) -> MissingStats:
        stats = MissingStats(references=len(references))
        docs = [d for d in self.session.scalars(select(Document)) if d.processing_status != ProcessingStatus.missing.value]
        unread = [d.filename for d in docs if d.processing_status != ProcessingStatus.processed.value]
        if unread:
            stats.skipped_reason = (f"{len(unread)} document(s) are not fully processed ({', '.join(unread[:3])}"
                                    f"{'...' if len(unread) > 3 else ''}), so an item could be in one of them")
            return stats
        if not references:
            return stats
        names = {d.id: d.filename for d in docs}
        pool = self._pool(names)
        seen: set[tuple[str, str]] = set()
        by_sentence: dict[tuple[str, str], list[DocReference]] = {}
        for ref in references:
            keys = _item_keys(ref.item)
            if not keys:
                continue
            if any(chunk_id != ref.chunk_id and any(k <= words for k in keys) for chunk_id, words in pool):
                stats.present += 1
                continue
            key = (ref.document_id, " ".join(sorted(keys[0])))
            if key in seen:
                stats.duplicates += 1
                continue
            seen.add(key)
            by_sentence.setdefault((ref.chunk_id, " ".join(ref.quote.split())), []).append(ref)
        for refs in by_sentence.values():  # items named by one sentence are one gap
            self._save(refs, names, stats)
        return stats

    def _pool(self, names: dict[str, str]) -> list[tuple[str, frozenset[str]]]:
        """Every place an item could be named: each file name, and each chunk's sheet name, heading and first line."""
        pool: list[tuple[str, frozenset[str]]] = [(f"file:{doc_id}", _words(re.sub(r"\.[A-Za-z0-9]+$", "", name))) for doc_id, name in names.items()]
        for chunk in self.session.scalars(select(DocumentChunk).where(DocumentChunk.document_id.in_(names))):
            first_line = (chunk.content or "").strip().splitlines()[0] if (chunk.content or "").strip() else ""
            pool.append((chunk.id, _words(f"{chunk.sheet_name or ''} {chunk.heading or ''} {first_line[:120]}")))
        return pool

    def _save(self, refs: list[DocReference], names: dict[str, str], stats: MissingStats) -> None:
        ref = refs[0]
        items = [r.item for r in refs]
        chunk = self.session.get(DocumentChunk, ref.chunk_id)
        where = chunk.sheet_name or (f"page {chunk.page_number}" if chunk and chunk.page_number else (chunk.heading if chunk else "") or "the document")
        stats.missing_items.extend(items)
        listed = "; ".join(items)
        title = f"Referenced {'items' if len(items) > 1 else 'item'} not in the data room: {listed}"
        result = self.registry.execute("submit_finding", {
            "title": title[:300], "classification": "missing_information", "severity": "medium", "confidence": 0.7,
            "evidence_status": "unavailable",
            "description": f"{names.get(ref.document_id)} ({where}) refers to {' and '.join(chr(34) + i + chr(34) for i in items)}, but no file, sheet or heading in the data room is named like "
                           f"{'them' if len(items) > 1 else 'it'}, and every document was read.",
            "unresolved_questions": [f"Where is {i}?" for i in items],
            "evidence": [{"chunk_id": ref.chunk_id, "quoted_text": ref.quote, "role": "insufficient",
                          "relevance_reason": f"Refers to {listed}", "support_assessment": "The document points to this item; it is not in the data room."}],
        })
        if not result.ok:
            stats.rejected += 1
            stats.rejections.append(f"{title[:60]}: {result.error.code}: {result.error.message[:160]}")
            return
        stats.saved += 1
