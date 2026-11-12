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
from app.services.candidate_matching import TABLE_TYPES
from app.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

_FILLER = {"the", "of", "and", "to", "for", "in", "on"}  # not "a": it names things ("Schedule A")
LINE_START_EXTRA = 2  # a caption or label names an item when its first words are the item's words plus at most this many
SENTENCE_WORDS = 6
MAX_EVIDENCE = 6  # sentences kept as evidence for one finding


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


def _names(words, key: frozenset[str]) -> bool:
    """A pool entry names the item: its word set contains the item's words, or (a line of text, kept in order) it starts with them."""
    if isinstance(words, tuple):
        return key <= frozenset(words[:len(key) + LINE_START_EXTRA])
    return key <= words


class MissingItemChecker:
    def __init__(self, session: Session, registry: ToolRegistry):
        self.session = session
        self.registry = registry

    def run(self, references: list[DocReference], not_fully_read: list[str] | None = None) -> MissingStats:
        stats = MissingStats(references=len(references))
        docs = [d for d in self.session.scalars(select(Document)) if d.processing_status != ProcessingStatus.missing.value]
        unread = [d.filename for d in docs if d.processing_status != ProcessingStatus.processed.value] + list(not_fully_read or [])
        if unread:
            stats.skipped_reason = (f"{len(unread)} document(s) are not fully processed ({', '.join(unread[:3])}"
                                    f"{'...' if len(unread) > 3 else ''}), so an item could be in one of them")
            return stats
        if not references:
            return stats
        names = {d.id: d.filename for d in docs}
        pool = self._pool(names)
        # one entry per missing item, however many documents or sentences name it ("Dive Safety Plan" and "MIMS Dive Safety Plan" are one item)
        items: list[tuple[frozenset[str], list[DocReference]]] = []
        for ref in references:
            keys = _item_keys(ref.item)
            if not keys:
                continue
            own = " ".join(ref.quote.split()).lower()
            if any(not (line and line in own) and not (not line and chunk_id == ref.chunk_id) and any(_names(words, k) for k in keys)
                   for chunk_id, words, line in pool):
                stats.present += 1  # a file, sheet, heading, caption or row label of the data room names it (the reference's own sentence does not count)
                continue
            for key, group in items:
                if key <= keys[0] or keys[0] <= key:
                    group.append(ref)
                    stats.duplicates += 1
                    break
            else:
                items.append((keys[0], [ref]))
        by_sentence: dict[tuple[str, str], list[list[DocReference]]] = {}
        for _key, group in items:  # items named by one sentence are one gap
            by_sentence.setdefault((group[0].chunk_id, " ".join(group[0].quote.split())), []).append(group)
        for groups in by_sentence.values():
            self._save(groups, names, stats)
        return stats

    def _pool(self, names: dict[str, str]) -> list[tuple[str, frozenset[str] | tuple[str, ...], str]]:
        """Every place an item could be named, as (chunk id, words, the line's lower-cased text): each file name; each chunk's sheet name and
        heading; a table row (its ID cell can name the item); and the start of every line of text that is a caption or label ("Table 3. ...",
        "Annex A - ..."): for those the words are kept in order, so that only an item the line STARTS with matches. A line that is a
        full sentence (it ends with a full stop and has six words or more) only mentions things, so it never counts."""
        pool: list = [(f"file:{doc_id}", _words(re.sub(r"\.[A-Za-z0-9]+$", "", name)), "") for doc_id, name in names.items()]
        for chunk in self.session.scalars(select(DocumentChunk).where(DocumentChunk.document_id.in_(names))):
            pool.append((chunk.id, _words(f"{chunk.sheet_name or ''} {chunk.heading or ''}"), ""))
            is_row = chunk.content_type in TABLE_TYPES
            for line in (chunk.content or "").splitlines():
                text = " ".join(line.split())
                if not text:
                    continue
                if is_row:
                    pool.append((chunk.id, _words(text[:240]), text.lower()))
                elif not (text[-1] in ".!?" and len(text.split()) >= SENTENCE_WORDS):
                    pool.append((chunk.id, tuple(w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in _FILLER), text.lower()))
        return pool

    def _save(self, groups: list[list[DocReference]], names: dict[str, str], stats: MissingStats) -> None:
        """One finding for the items one sentence names. Every other place that names an item adds its sentence as evidence."""
        ref = groups[0][0]
        items = [g[0].item for g in groups]
        chunk = self.session.get(DocumentChunk, ref.chunk_id)
        where = chunk.sheet_name or (f"page {chunk.page_number}" if chunk and chunk.page_number else (chunk.heading if chunk else "") or "the document")
        stats.missing_items.extend(items)
        listed = "; ".join(items)
        title = f"Referenced {'items' if len(items) > 1 else 'item'} not in the data room: {listed}"
        evidence, seen = [], set()
        for group in groups:
            for r in group:
                key = (r.chunk_id, " ".join(r.quote.split()))
                if key not in seen and len(evidence) < MAX_EVIDENCE:
                    seen.add(key)
                    evidence.append({"chunk_id": r.chunk_id, "quoted_text": r.quote, "role": "insufficient",
                                     "relevance_reason": f"Refers to {r.item}", "support_assessment": "The document points to this item; it is not in the data room."})
        also = sorted({names.get(r.document_id, "?") for g in groups for r in g} - {names.get(ref.document_id)})
        result = self.registry.execute("submit_finding", {
            "title": title[:300], "classification": "missing_information", "severity": "medium", "confidence": 0.7,
            "evidence_status": "unavailable",
            "description": f"{names.get(ref.document_id)} ({where}) refers to {' and '.join(chr(34) + i + chr(34) for i in items)}, but no file, sheet, heading, caption or row label in the data room is named like "
                           f"{'them' if len(items) > 1 else 'it'}, and every document was read."
                           + (f" Also referred to in: {', '.join(also)}." if also else ""),
            "unresolved_questions": [f"Where is {i}?" for i in items],
            "evidence": evidence,
        })
        if not result.ok:
            stats.rejected += 1
            stats.rejections.append(f"{title[:60]}: {result.error.code}: {result.error.message[:160]}")
            return
        stats.saved += 1
