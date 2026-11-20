"""Claim labels: each claim is labelled on its own with a subject, an attribute, a value type, a basis and a kind; the attribute names are then merged into one vocabulary.

Why: grouping claims by asking the model which ones belong together makes it decide sameness across thousands of claims in changing batches, so the
result depends on the batches. Here the model only does two small, local tasks: label one claim (it sees nothing else of the collection), and merge a
list of attribute names into canonical names (one call over distinct names, not claims). Grouping is then a lookup on the keys (``key_grouping``).

Each label is cached by a hash of the claim's own text and fields (plus the hash of the instructions), so a rerun on the same claims asks nothing and a
claim's label never depends on which other claims were in its call. The vocabulary merge is cached by the hash of the sorted names.
A claim the model did not label gets a fallback label from its extracted metric (never cached).
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Literal, get_args

from pydantic import BaseModel, Field
from sqlalchemy import delete, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.llm.base import LLMClient
from app.llm.errors import LLMError
from app.models import ClaimLabelCacheEntry
from app.schemas.claim import ClaimView

logger = logging.getLogger(__name__)

ValueType = Literal["money", "count", "date", "percent", "duration", "measure", "text"]
KIND_LABEL, KIND_VOCABULARY = "label", "vocabulary"
IN_CHUNK = 500  # keys per database lookup
LABEL_TOKENS_PER_CLAIM = 150  # a label is about 60 to 80 tokens; the cap leaves room for pretty-printing
VOCABULARY_MAX_TOKENS = 16_384
MAX_LABEL_ATTEMPTS = 2  # a call that labels fewer than half of its claims is asked once more

LABEL_SYSTEM = """You label claims taken from the documents of one collection. Label every claim on its own: use only what its own line says, never what other claims say.

For every claim return:
- subject: what the claim is about, as a short noun phrase naming the thing itself (the item, party, section, period or scope), without the attribute. Use the plain common name, not the document's wording.
- attribute: what is stated about the subject, as a short generic noun phrase in singular lower case (for example "unit count", "maximum amount", "price per unit", "completion date"). The same thing stated in different words gets the same attribute. A word that changes what is measured stays in the attribute (a total versus a part of it, gross versus net, a limit versus an amount). The subject and the basis never go in the attribute.
- other_attributes: at most 2 further attributes the claim also states (usually none).
- value_type: money, count, date, percent, duration, measure (a physical or other quantity with a unit) or text.
- basis: the status of the figure, in one lower-case word: planned, budgeted, estimated, forecast, actual, final, revised, limit, contract, or empty when the claim says nothing about it.
- kind: "rule" when the claim states what is permitted, prohibited, required, limited or scheduled for a scope; "record" for anything else.
"""
LABEL_FORMAT = """
OUTPUT FORMAT (strict). Return exactly one JSON object and nothing else: {"labels": [ {"ref": "c1", "subject": "...", "attribute": "...", "other_attributes": [], "value_type": "count", "basis": "", "kind": "record"} ]}.
- One entry for every claim shown, with its ref exactly as shown, each ref once.
- Do not add other fields, comments, explanations or markdown."""
LABEL_INSTRUCTIONS = LABEL_SYSTEM + LABEL_FORMAT

VOCABULARY_INSTRUCTIONS = """Below is a list of attribute names (with the number of claims that carry each). Some names state the same measurable thing in different words ("apartments", "residential units", "unit count").

Return the groups of names that mean exactly the same thing, each with one canonical name taken from the group's members or written as a short generic noun phrase in singular lower case.
Rules:
- Merge only names that state the same thing. Names that differ in what is measured stay apart: a total and a part of it, a limit and an amount, gross and net, a price and a quantity, a start and an end.
- A name that has no synonym in the list is left out of the answer.
- Every name appears in at most one group; use the names exactly as listed.

OUTPUT FORMAT (strict). Return exactly one JSON object and nothing else: {"groups": [ {"canonical": "unit count", "members": ["apartments", "residential units"]} ]}. If nothing merges, return {"groups": []}."""


class ClaimLabel(BaseModel):
    subject: str = ""
    attribute: str = ""
    other_attributes: list[str] = Field(default_factory=list)
    value_type: ValueType = "text"
    basis: str = ""
    kind: Literal["rule", "record"] = "record"


class LabelItem(BaseModel):
    """One label as the model returns it. Every field is required (no defaults): a field with a default is optional in the response schema, and the
    model then returns the ref alone for every claim (run of 21 Nov: all 1,184 claims came back unlabelled). value_type and kind are plain strings: one
    off-list word must not invalidate the labels of all the claims of the call, so ``_clean`` coerces an unknown value_type to "text" and an unknown
    kind to "record"."""
    ref: str
    subject: str
    attribute: str
    other_attributes: list[str]
    value_type: str
    basis: str
    kind: str


class LabelAnswer(BaseModel):
    labels: list[LabelItem] = Field(default_factory=list)


class VocabularyGroup(BaseModel):
    canonical: str
    members: list[str] = Field(default_factory=list)


class VocabularyAnswer(BaseModel):
    groups: list[VocabularyGroup] = Field(default_factory=list)


class LabelStats(BaseModel):
    claims: int = 0
    distinct: int = 0  # distinct claims by text and fields: claims that repeat one sentence share one label
    cached: int = 0  # distinct claims whose label came from the cache
    asked: int = 0  # distinct claims sent to the model
    calls: int = 0
    fallback: int = 0  # claims the model did not label (failed call or missing ref): labelled from the extracted metric
    attributes: int = 0  # distinct attribute names before the merge
    attributes_merged: int = 0  # distinct attribute names after it
    vocabulary_cached: bool = False
    errors: list[str] = Field(default_factory=list)


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize_name(text: str | None) -> str:
    """A name for matching: lower case, words only, single spaces."""
    return " ".join(re.findall(r"[^\W_]+", (text or "").lower()))


def claim_key(view: ClaimView) -> str:
    """The cache key of a claim: its own text and fields, never its id or its neighbours."""
    return digest(json.dumps([view.claim_text, view.subject, view.metric, view.value, view.unit, view.period], ensure_ascii=False, default=str))


VALUE_TYPES = frozenset(get_args(ValueType))


def _clean(label: LabelItem) -> ClaimLabel:
    attribute = normalize_name(label.attribute)
    others = [o for o in dict.fromkeys(normalize_name(x) for x in label.other_attributes) if o and o != attribute][:2]
    value_type = (label.value_type or "").strip().lower()
    kind = (label.kind or "").strip().lower()
    return ClaimLabel(subject=normalize_name(label.subject), attribute=attribute, other_attributes=others,
                      value_type=value_type if value_type in VALUE_TYPES else "text", basis=normalize_name(label.basis), kind="rule" if kind == "rule" else "record")


def fallback_label(view: ClaimView) -> ClaimLabel:
    """A claim the model did not label: its extracted subject and metric together are the attribute (a metric such as "date" or "cost" alone names no
    attribute, so two claims group only when both their subject and their metric agree; the code's own matching of the other routes covers the rest);
    its extracted type, as far as it goes, the value type."""
    value_type = {"date": "date", "quantity": "measure"}.get(view.value_type or "", "text")
    metric = normalize_name(view.metric)
    subject = normalize_name(view.subject)
    return ClaimLabel(subject=subject, attribute=f"{subject} {metric}".strip() if metric else "", value_type=value_type)


class LabelCache:
    """Answers kept in ``claim_label_cache``. Every call opens its own short session (labelling calls run in threads). A database error never stops a
    run: the cache is skipped and the first error is kept in ``error``."""

    def __init__(self, session_factory: Callable[[], Session], prompt_hash: str, refresh: bool = False):
        self._factory = session_factory
        self.prompt_hash = prompt_hash
        self.refresh = refresh  # never read, always write
        self.error: str | None = None
        self._lock = threading.Lock()

    def _failed(self, what: str, exc: Exception) -> None:
        self.error = self.error or f"{what}: {exc.__class__.__name__}: {str(exc)[:200]}"
        logger.warning("Label cache unavailable", extra={"error": self.error})

    def get_many(self, kind: str, keys: list[str]) -> dict[str, dict]:
        if self.refresh or not keys:
            return {}
        found: dict[str, dict] = {}
        try:
            with self._factory() as session:
                for start in range(0, len(keys), IN_CHUNK):
                    rows = session.execute(select(ClaimLabelCacheEntry.key_hash, ClaimLabelCacheEntry.payload).where(
                        ClaimLabelCacheEntry.kind == kind, ClaimLabelCacheEntry.prompt_hash == self.prompt_hash,
                        ClaimLabelCacheEntry.key_hash.in_(keys[start:start + IN_CHUNK])))
                    found.update({key: payload for key, payload in rows if isinstance(payload, dict)})
        except SQLAlchemyError as exc:
            self._failed("reading the label cache", exc)
            return {}
        return found

    def put_many(self, kind: str, items: dict[str, dict]) -> None:
        if not items:
            return
        try:
            with self._lock, self._factory() as session:
                keys = list(items)
                for start in range(0, len(keys), IN_CHUNK):
                    session.execute(delete(ClaimLabelCacheEntry).where(
                        ClaimLabelCacheEntry.kind == kind, ClaimLabelCacheEntry.prompt_hash == self.prompt_hash,
                        ClaimLabelCacheEntry.key_hash.in_(keys[start:start + IN_CHUNK])))
                session.add_all([ClaimLabelCacheEntry(kind=kind, key_hash=key, prompt_hash=self.prompt_hash, payload=payload) for key, payload in items.items()])
                session.commit()
        except SQLAlchemyError as exc:
            self._failed("writing the label cache", exc)


def _line(ref: str, v: ClaimView) -> str:
    parts = [f"subject: {v.subject or '-'}", f"metric: {v.metric or '-'}", f"value: {v.value}" + (f" {v.unit}" if v.unit else "")]
    if v.period:
        parts.append(f"period: {v.period}")
    return f"{ref} | " + " | ".join(parts) + f" | {v.claim_text or ''}"


class ClaimLabeler:
    def __init__(self, llm: LLMClient, cache: LabelCache, batch_size: int = 40, workers: int = 4,
                 should_stop: Callable[[], bool] | None = None, on_progress: Callable[[str], None] | None = None):
        self.llm = llm
        self.cache = cache
        self.batch_size = max(1, batch_size)
        self.workers = max(1, workers)
        self.should_stop = should_stop or (lambda: False)
        self.on_progress = on_progress or (lambda _m: None)

    def label(self, views: list[ClaimView]) -> tuple[dict[str, ClaimLabel], LabelStats]:
        """A label for every claim (by claim id). Claims with the same text and fields are labelled once."""
        stats = LabelStats(claims=len(views))
        by_key: dict[str, list[ClaimView]] = {}
        for v in views:
            by_key.setdefault(claim_key(v), []).append(v)
        stats.distinct = len(by_key)
        labels: dict[str, ClaimLabel] = {}  # by key
        for key, payload in self.cache.get_many(KIND_LABEL, list(by_key)).items():
            try:
                labels[key] = ClaimLabel.model_validate(payload)
            except ValueError:
                continue  # an unreadable entry is asked again
        stats.cached = len(labels)
        todo = sorted(k for k in by_key if k not in labels)  # a fixed order: the same claims make the same calls
        batches = [todo[i:i + self.batch_size] for i in range(0, len(todo), self.batch_size)]
        stats.asked = len(todo)

        def ask(batch: list[str]) -> tuple[dict[str, ClaimLabel], str | None]:
            if self.should_stop():
                return {}, "stop requested"
            refs = {f"c{i}": key for i, key in enumerate(batch, start=1)}
            prompt = "Claims:\n\n" + "\n".join(_line(ref, by_key[key][0]) for ref, key in refs.items()) + f"\n\n=== INSTRUCTIONS ===\n\n{LABEL_INSTRUCTIONS}"
            got: dict[str, ClaimLabel] = {}
            for attempt in range(1, MAX_LABEL_ATTEMPTS + 1):
                try:
                    answer = self.llm.generate_structured_output(prompt, LabelAnswer, max_output_tokens=LABEL_TOKENS_PER_CLAIM * len(batch) + 500)
                except LLMError as exc:
                    return {}, str(exc)
                got = {}
                for item in answer.labels:
                    key = refs.get(item.ref.strip())
                    if key is not None and key not in got and normalize_name(item.attribute):
                        got[key] = _clean(item)
                if len(got) * 2 >= len(batch):
                    break  # most claims came back labelled
                logger.warning(f"Labelling call returned labels for {len(got)} of {len(batch)} claims (attempt {attempt} of {MAX_LABEL_ATTEMPTS})")
            else:
                self.cache.put_many(KIND_LABEL, {k: lab.model_dump() for k, lab in got.items()})
                return got, f"a labelling call returned labels for only {len(got)} of {len(batch)} claims after {MAX_LABEL_ATTEMPTS} attempts"
            self.cache.put_many(KIND_LABEL, {k: lab.model_dump() for k, lab in got.items()})
            return got, None

        if batches:
            with ThreadPoolExecutor(max_workers=self.workers) as pool:
                for number, (got, error) in enumerate(pool.map(ask, batches), start=1):
                    stats.calls += 1
                    labels.update(got)
                    if error:
                        stats.errors.append(error)
                    self.on_progress(f"labelling call {number} of {len(batches)} done" if not error else f"labelling call {number} of {len(batches)} failed")
        out: dict[str, ClaimLabel] = {}
        for key, members in by_key.items():
            for v in members:
                label = labels.get(key)
                if label is None:
                    stats.fallback += 1
                out[v.id] = label or fallback_label(v)
        return out, stats


def merge_vocabulary(llm: LLMClient, cache: LabelCache, counts: dict[str, int], stats: LabelStats | None = None) -> dict[str, str]:
    """name -> canonical name, for the names the model merged (names it left alone are not in the map). One call over the distinct names; the answer is
    cached by the sorted names, so the same names always merge the same way. A failed call merges nothing and is reported in ``stats.errors``."""
    names = sorted(n for n in counts if n)
    if len(names) < 2:
        return {}
    key = digest(json.dumps(names, ensure_ascii=False))
    cached = cache.get_many(KIND_VOCABULARY, [key]).get(key)
    if cached is not None:
        if stats is not None:
            stats.vocabulary_cached = True
        return {str(k): str(v) for k, v in cached.items()}
    listing = "\n".join(f"- {name} ({counts[name]})" for name in names)
    try:
        answer = llm.generate_structured_output(f"Attribute names:\n\n{listing}\n\n=== INSTRUCTIONS ===\n\n{VOCABULARY_INSTRUCTIONS}", VocabularyAnswer, max_output_tokens=VOCABULARY_MAX_TOKENS)
    except LLMError as exc:
        if stats is not None:
            stats.errors.append(f"vocabulary merge failed: {exc}")
        return {}
    known = set(names)
    mapping: dict[str, str] = {}
    for group in answer.groups:
        members = [m for m in dict.fromkeys(normalize_name(x) for x in group.members) if m in known and m not in mapping]
        canonical = normalize_name(group.canonical) or (members[0] if members else "")
        if len(members) < 2 or not canonical:
            continue
        for member in members:
            mapping[member] = canonical
    cache.put_many(KIND_VOCABULARY, {key: mapping})
    return mapping
