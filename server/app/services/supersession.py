"""Which document says it replaces which: "This Revision B supersedes Progress Report No. 3 Revision A dated 30 April 2025".

A document that says it supersedes another one, and names it, makes every difference between the two a change of edition, not a conflict: the later edition
states the position, the earlier one stated the position it replaced. The superseded document is found by its name: the words of the name appear in the
heading of exactly one other document. Read from the text, so it works for any kind of document.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Document, DocumentChunk
from app.models.base import ProcessingStatus

SUPERSEDES = re.compile(r"\b(?:supersede[sd]?|replace[sd]?)\s+(?P<old>(?:[^.;\n]|\.(?=\s*\d)){6,160})", re.IGNORECASE)
NEGATED = re.compile(r"(?:\bnot|n't|\bnever)\s+(?:\w+\s+)?$", re.IGNORECASE)
HEADING_WORDS = 60  # the name of a document is in its first words
NAME_END = re.compile(r"\s+(?:dated|of|from|issued|published|sent|as of|as at|on)\s+.*$|,.*$|\s*\(.*$", re.IGNORECASE)


@dataclass(frozen=True)
class Supersession:
    new_document: str  # id of the document that replaces
    old_document: str  # id of the document that is replaced
    chunk_id: str  # where the sentence is
    sentence: str  # the words of the document, exactly as written


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _contains(heading: list[str], name: list[str]) -> bool:
    return bool(name) and any(heading[i:i + len(name)] == name for i in range(len(heading) - len(name) + 1))


def superseded_documents(session: Session) -> list[Supersession]:
    docs = list(session.execute(select(Document.id, Document.filename).where(Document.processing_status != ProcessingStatus.missing.value)))
    chunks = list(session.scalars(select(DocumentChunk).order_by(DocumentChunk.document_id, DocumentChunk.chunk_index)))
    heading: dict[str, list[str]] = {}
    for c in chunks:
        heading.setdefault(c.document_id, _words(c.content)[:HEADING_WORDS])
    found: list[Supersession] = []
    for chunk in chunks:
        for m in SUPERSEDES.finditer(chunk.content):
            if NEGATED.search(chunk.content[max(0, m.start() - 14):m.start()]):
                continue  # "does not replace the Master Services Agreement"
            name = _words(NAME_END.sub("", m.group("old").strip()))
            old = [d for d, _f in docs if d != chunk.document_id and _contains(heading.get(d, []), name)]
            if len(old) != 1 or _contains(_words(chunk.content[:m.start()])[:HEADING_WORDS], name):
                continue  # the name must be exactly one other document's, and not this document's own
            start = max(chunk.content.rfind(". ", 0, m.start()) + 2, chunk.content.rfind("\n", 0, m.start()) + 1, 0)
            sentence = chunk.content[start:m.end()].strip()
            if len(sentence) >= 8:
                found.append(Supersession(chunk.document_id, old[0], chunk.id, sentence))
    return found
