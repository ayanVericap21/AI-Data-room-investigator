"""Document profiles: what each document is, so that a claim read out of it is not judged without that context.

A claim carries a subject, a metric and a value, but not what the document is about, who it concerns or what kind of document it is. A profile is a
short description of the whole document: its main entity, type, purpose, date, status and scope. Profiles are shown to the grouping and judging calls
as a block of document cards, and the entity gate (see ``group_gates``) uses them to keep claims of unrelated subjects apart.

Profiles are made by calls of their own, as few as possible:
- the readable text of every document is laid out with its places (page, sheet) and packed, several documents to a call, into calls of up to
  ``BATCH_CHARS`` characters (largest first, so the calls are as full as they can be);
- a document longer than one call is cut into parts at chunk boundaries; its parts are packed like any other text, and one more call, for all such
  documents together, merges the descriptions of the parts into one profile per document.
So the number of calls is the total text divided by the call size, rounded up, plus one when some document had to be split.
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable

from pydantic import BaseModel

from app.llm.errors import LLMError
from app.models import DocumentChunk

logger = logging.getLogger(__name__)

BATCH_CHARS = 60_000  # characters of document text in one call (the size the claim-grouping calls already use)
UNIT_OVERHEAD = 120  # characters a document's header and spacing add to a call
CARD_FIELD_CHARS = 200

PROFILE_FIELDS = ("main_entity", "document_type", "purpose", "date", "status", "scope")


class DocProfile(BaseModel):
    main_entity: str = ""  # the organisation, project, product, person or event the document is mainly about, by name
    document_type: str = ""  # permit, contract, financial model, lab report, presentation, register, minutes ...
    purpose: str = ""  # what the document is for, one sentence
    date: str = ""  # its date or version, as stated
    status: str = ""  # draft, final, signed, superseded ..., as stated
    scope: str = ""  # what it covers and what it does not, one sentence

    @property
    def empty(self) -> bool:
        return not any((getattr(self, f) or "").strip() for f in PROFILE_FIELDS)


class ProfileItem(DocProfile):
    ref: str = ""  # the document reference shown in the call (d1, d2, ...)


class ProfileBatch(BaseModel):
    documents: list[ProfileItem] = []


DESCRIBE_SYSTEM = (
    "You describe documents. The input holds one or more documents, each starting with a line '### dN | file name' (a long document is cut into parts and the line "
    "says which part). For EACH document shown return one entry with ref (the dN shown), main_entity (the organisation, project, product, person or event the "
    "document is mainly about, by name), document_type (for example permit, contract, financial model, lab report, presentation, register, minutes), purpose "
    "(one sentence), date (the document's date or version, only if it states one), status (draft, final, signed or superseded, only if it states one) and scope "
    "(what it covers and what it leaves out, one sentence; for a part, what that part covers). Describe each document on its own: never mix documents. Use only "
    "what the text states; leave a field empty rather than guess. OUTPUT FORMAT (strict): one JSON object {\"documents\": [{\"ref\": \"d1\", \"main_entity\": \"\", "
    "\"document_type\": \"\", \"purpose\": \"\", \"date\": \"\", \"status\": \"\", \"scope\": \"\"}]} with one entry per document shown and no other fields.")

MERGE_SYSTEM = (
    "You receive, for each of several documents, descriptions written from its consecutive parts: '### dN | file name' followed by 'Part k:' and its fields. For "
    "EACH document return one entry with ref and the description of the whole document: main_entity (when the parts name different things, the one the whole "
    "document is about), document_type, purpose (one sentence), date and status (only if some part states them) and scope (what the whole document covers and "
    "leaves out, one sentence). Use only what the descriptions state; leave a field empty rather than guess. OUTPUT FORMAT (strict): one JSON object "
    "{\"documents\": [{\"ref\": \"d1\", \"main_entity\": \"\", \"document_type\": \"\", \"purpose\": \"\", \"date\": \"\", \"status\": \"\", \"scope\": \"\"}]}.")


def _where(chunk: DocumentChunk) -> str:
    if chunk.sheet_name:
        return f"sheet {chunk.sheet_name}"
    if chunk.page_number:
        return f"page {chunk.page_number}"
    return chunk.heading or chunk.content_type or "text"


def readable_chars(chunks: list[DocumentChunk]) -> int:
    return sum(len(c.content or "") for c in chunks)


def split_parts(chunks: list[DocumentChunk], part_chars: int) -> list[str]:
    """The readable text cut at chunk boundaries into parts of about ``part_chars`` characters, each chunk headed by its place in the document."""
    parts: list[str] = []
    current: list[str] = []
    size = 0
    for chunk in chunks:
        text = (chunk.content or "").strip()
        if not text:
            continue
        line = f"[{_where(chunk)}] {text[:part_chars]}"
        if current and size + len(line) > part_chars:
            parts.append("\n\n".join(current))
            current, size = [], 0
        current.append(line)
        size += len(line) + 2
    if current:
        parts.append("\n\n".join(current))
    return parts


@dataclass
class Unit:
    """The text sent for one document, or for one part of a long one."""
    doc_id: str
    filename: str
    part: int
    parts: int
    text: str

    @property
    def size(self) -> int:
        return len(self.text) + UNIT_OVERHEAD


def make_units(documents: list[tuple[str, str, list[DocumentChunk]]], budget: int) -> list[Unit]:
    """One unit per document that fits in a call, several per document that does not. ``documents`` is (id, file name, readable chunks)."""
    units: list[Unit] = []
    for doc_id, filename, chunks in documents:
        parts = split_parts(chunks, budget - UNIT_OVERHEAD)
        units += [Unit(doc_id, filename, i, len(parts), text) for i, text in enumerate(parts, start=1)]
    return units


def pack(units: list[Unit], budget: int) -> list[list[Unit]]:
    """Units packed into as few calls as possible: largest first, each into the first call it fits in (first-fit decreasing)."""
    calls: list[tuple[int, list[Unit]]] = []
    for unit in sorted(units, key=lambda u: -u.size):
        for i, (used, members) in enumerate(calls):
            if used + unit.size <= budget:
                members.append(unit)
                calls[i] = (used + unit.size, members)
                break
        else:
            calls.append((unit.size, [unit]))
    for _used, members in calls:  # inside a call, documents in file order and parts in order
        members.sort(key=lambda u: (u.filename, u.part))
    return [members for _used, members in calls]


def render_call(members: list[Unit]) -> str:
    blocks = []
    for n, u in enumerate(members, start=1):
        label = f"{u.filename}" + (f" (part {u.part} of {u.parts})" if u.parts > 1 else "")
        blocks.append(f"### d{n} | {label}\n{u.text}")
    return "Describe each document below.\n\n" + "\n\n".join(blocks)


@dataclass
class ProfileRecord:
    filename: str
    chars: int
    parts: int = 1
    ok: bool = True
    note: str = ""


@dataclass
class CallRecord:
    kind: str  # describe | merge
    documents: list[str]
    chars: int
    ok: bool = True
    note: str = ""


@dataclass
class ProfileStats:
    records: list[ProfileRecord] = field(default_factory=list)
    calls: list[CallRecord] = field(default_factory=list)
    budget: int = BATCH_CHARS

    @property
    def total_chars(self) -> int:
        return sum(r.chars for r in self.records)


class ProfileBuilder:
    def __init__(self, llm: Any, budget: int = BATCH_CHARS, workers: int = 1, on_progress: Callable[[str], None] | None = None,
                 should_stop: Callable[[], bool] | None = None):
        self.llm = llm
        self.budget = max(2000, budget)
        self.workers = max(1, workers)
        self.say = on_progress or (lambda _m: None)
        self.should_stop = should_stop or (lambda: False)
        self.stats = ProfileStats(budget=self.budget)

    def _ask(self, system: str, prompt: str) -> tuple[ProfileBatch | None, str]:
        if self.should_stop():
            return None, "stop requested"
        try:
            return self.llm.generate_structured_output(prompt, ProfileBatch, system=system), ""
        except LLMError as exc:
            return None, str(exc)

    def _map(self, system: str, prompts: list[str]) -> list[tuple[ProfileBatch | None, str]]:
        if self.workers <= 1 or len(prompts) <= 1:
            return [self._ask(system, p) for p in prompts]
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            return list(pool.map(lambda p: self._ask(system, p), prompts))

    def run(self, documents: list[tuple[str, str, list[DocumentChunk]]]) -> dict[str, tuple[str, DocProfile]]:
        """The profile of each document: {document id: (file name, profile)}. A document that could not be described is left out."""
        names = {doc_id: name for doc_id, name, _ in documents}
        for doc_id, name, chunks in documents:
            self.stats.records.append(ProfileRecord(name, readable_chars(chunks)))
        units = make_units(documents, self.budget)
        for r in self.stats.records:
            r.parts = sum(1 for u in units if u.filename == r.filename)
            if r.parts == 0:
                r.ok, r.note = False, "no readable text"
        calls = pack(units, self.budget)
        self.say(f"{len(units)} texts from {len(documents)} documents in {len(calls)} calls of up to {self.budget:,} characters")
        partial: dict[str, list[tuple[int, DocProfile]]] = {}  # document id -> (part, description)
        for number, (members, (batch, error)) in enumerate(zip(calls, self._map(DESCRIBE_SYSTEM, [render_call(m) for m in calls])), start=1):
            chars = sum(len(u.text) for u in members)
            record = CallRecord("describe", sorted({u.filename for u in members}), chars)
            self.stats.calls.append(record)
            if batch is None:
                record.ok, record.note = False, error
                self.say(f"describe call {number} of {len(calls)} failed: {error[:120]}")
                continue
            by_ref = {f"d{n}": u for n, u in enumerate(members, start=1)}
            for item in batch.documents:
                unit = by_ref.get(item.ref.strip())
                if unit is not None and not item.empty:
                    partial.setdefault(unit.doc_id, []).append((unit.part, DocProfile(**{f: getattr(item, f) for f in PROFILE_FIELDS})))
            self.say(f"describe call {number} of {len(calls)} done ({len(members)} texts, {chars:,} characters)")
        profiles: dict[str, tuple[str, DocProfile]] = {}
        to_merge: list[str] = []
        for doc_id, parts in partial.items():
            parts.sort(key=lambda x: x[0])
            if len(parts) == 1:  # a document that fit in one call, or a long one of which only one part was described
                profiles[doc_id] = (names[doc_id], parts[0][1])
            else:
                to_merge.append(doc_id)
        if to_merge:
            self._merge(to_merge, partial, names, profiles)
        for r in self.stats.records:
            if r.parts and not any(n == r.filename for n, _p in profiles.values()):
                r.ok, r.note = False, r.note or "no description was returned for it"
        return profiles

    def _merge(self, doc_ids: list[str], partial: dict[str, list[tuple[int, DocProfile]]], names: dict[str, str],
               profiles: dict[str, tuple[str, DocProfile]]) -> None:
        """One call that merges the descriptions of the parts of every long document (a call for each ``budget`` of text, in practice one)."""
        refs = {f"d{n}": doc_id for n, doc_id in enumerate(doc_ids, start=1)}
        blocks = []
        for ref, doc_id in refs.items():
            lines = [f"### {ref} | {names[doc_id]}"]
            for part, d in partial[doc_id]:
                lines.append(f"Part {part}:\n" + "\n".join(f"{f}: {getattr(d, f)}" for f in PROFILE_FIELDS if getattr(d, f)))
            blocks.append("\n".join(lines))
        batch, error = self._ask(MERGE_SYSTEM, "Merge the descriptions of each document.\n\n" + "\n\n".join(blocks))
        record = CallRecord("merge", [names[d] for d in doc_ids], sum(len(b) for b in blocks))
        self.stats.calls.append(record)
        merged: dict[str, DocProfile] = {}
        if batch is None:
            record.ok, record.note = False, error + "; the first part's description of each document is used"
            self.say("merge call failed: " + error[:120])
        else:
            for item in batch.documents:
                doc_id = refs.get(item.ref.strip())
                if doc_id is not None and not item.empty:
                    merged[doc_id] = DocProfile(**{f: getattr(item, f) for f in PROFILE_FIELDS})
        for doc_id in doc_ids:
            profiles[doc_id] = (names[doc_id], merged.get(doc_id) or partial[doc_id][0][1])


def _clip(text: str, n: int = CARD_FIELD_CHARS) -> str:
    t = " ".join((text or "").split())
    return t if len(t) <= n else t[: n - 1] + "…"


def card(filename: str, p: DocProfile) -> str:
    """The full card of one document."""
    about = ", ".join(x for x in (_clip(p.main_entity), _clip(p.document_type)) if x) or "not described"
    line = f"- {filename}: {about}." + (f" {_clip(p.purpose)}" if p.purpose.strip() else "")
    extra = [f"{label}: {_clip(getattr(p, name))}" for name, label in (("date", "date or version"), ("status", "status"), ("scope", "scope")) if getattr(p, name).strip()]
    return line + ("\n    " + "; ".join(extra) if extra else "")


def short_card(filename: str, p: DocProfile) -> str:
    about = ", ".join(x for x in (_clip(p.main_entity, 80), _clip(p.document_type, 60)) if x) or "not described"
    return f"- {filename}: {about}."


def render_cards(entries: dict[str, tuple[str, DocProfile]], focus: set[str] | None = None) -> str:
    """The cards of the documents, in file order. Documents in ``focus`` (document ids) get the full card and the others one line; with no
    ``focus`` every document gets the full card."""
    lines = []
    for doc_id, (name, p) in sorted(entries.items(), key=lambda kv: kv[1][0]):
        lines.append(card(name, p) if focus is None or doc_id in focus else short_card(name, p))
    return "\n".join(lines)
