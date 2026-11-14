"""Grouping by the LLM (an alternative to the code-based candidate matching, chosen with a flag).

Claims are sorted so that similar labels sit next to each other, cut into batches by size, and shown to
the LLM in compact one-line form. It returns groups of claims that state the same attribute of the same
subject; every claim of a group is a candidate of every other, and the groups go to the same ``ClaimGroupJudge`` as
code-made candidates. No code gate (subject overlap, unit, partner cap) applies here: the LLM decides what belongs
together, and the judge and the quote checks after it decide what becomes a finding.

Claims in different batches are never grouped together. The stats report the number of batches so the gap
is visible.
"""
from __future__ import annotations

import logging
import math
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations
from typing import Callable

from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.llm.base import LLMClient
from app.llm.embeddings import EmbeddingProvider
from app.llm.errors import LLMError
from app.models import Claim, Document, DocumentChunk
from app.models.base import ProcessingStatus
from app.schemas.claim import Candidate, ClaimView
from app.services.aliases import defined_names, resolve
from app.services.blocking import negation_candidates, shared_label_candidates, types_compatible
from app.services.identifiers import entity_ids
from app.services.candidate_matching import (compare_attributes, generate_candidates, normalize_period, normalize_unit, relation_hint, same_statement,
                                             comparable_subjects, same_table, table_key, tokens)
from app.services.numbers import parse_number, parse_range
from app.services.value_rules import unit_info, units_match

logger = logging.getLogger(__name__)
MAX_CLAIMS_PER_IDENTIFIER = 12  # a record id in more claims than this is too common to link them
LEXICAL_FLOOR = 0.7  # topic score from which the code's word matching adds a candidate to the model's groups
LEXICAL_PARTNERS = 12  # at most this many such candidates for one claim

SYSTEM = """You group claims taken from the documents of one collection so that their values can be compared.

Rules:
1. Use ONLY the claims shown. Refer to each claim by its ref (c1, c2, ...) exactly as shown.
2. Put claims in the same group only when they state the same attribute of the same subject. The wording may differ ("power supply" and "connected power load" can be the same attribute).
3. Claims about different attributes are never grouped, even when their units or numbers look alike (a height and a floor level, a distance and a duration).
4. Claims about different subjects are never grouped (different items, parties or sections), even when the attribute is the same.
5. Group claims whether or not their values agree: the comparison happens later.
6. A claim belongs to at most one group. A group has at least two claims. Leave a claim out if nothing else states the same thing.
7. topic is a few words naming what the group is about.
8. Every claim shows its type: quantity, date, identifier, name, state or statement (or - when unknown). The type only describes how the value is written. Do not keep claims apart because their types differ: the same fact can be written in different forms (a status and a count, a date and a duration, a number and a sentence), so group such claims when they state the same attribute of the same subject.

OUTPUT FORMAT (strict). Return exactly one JSON object and nothing else: {"groups": [ {"refs": ["c1", "c2"], "topic": "..."} ]}.
- Every group has "refs", a list of AT LEAST TWO refs taken from the claims shown. Never output a group with one ref, and never put a ref in two groups.
- Every group has "topic", a non-empty string.
- If no claims belong together, return {"groups": []}.
- Do not add other fields, comments, explanations or markdown."""


class GroupOfRefs(BaseModel):
    refs: list[str]  # a group with fewer than two claims is ignored
    topic: str = ""


class GroupingAnswer(BaseModel):
    groups: list[GroupOfRefs] = Field(default_factory=list)


class GroupingStats(BaseModel):
    claims: int = 0
    batches: int = 0
    llm_calls: int = 0
    groups: int = 0
    claims_grouped: int = 0
    candidates: int = 0  # links between two claims that may state one fact (every claim of a group with every other, and nearest neighbours)
    claims_filtered: int = 0  # claims not sent to the grouping calls: nothing else shares their subject or metric words
    identifier_candidates: int = 0  # candidates added because two claims mention the same record id and state the same kind of value
    lexical_candidates: int = 0  # candidates added by the code's own word matching (subject and metric words that largely coincide)
    label_candidates: int = 0  # candidates added because two claims share a rare word of their labels ("purchase order", "headcount")
    polarity_candidates: int = 0  # claims that say there is none of something, linked to the claims that mention it
    neighbor_candidates: int = 0  # candidates added because a claim's nearest neighbour elsewhere was not in its group
    largest_group: int = 0
    distinct_values: int = 0  # clusters of claims that state the same value, over all groups
    unknown_refs: int = 0
    batches_failed: int = 0
    errors: list[str] = Field(default_factory=list)


def _location(chunk: DocumentChunk) -> str:
    if chunk.sheet_name:
        return f"sheet {chunk.sheet_name}, {chunk.cell_reference}"
    return f"page {chunk.page_number}" if chunk.page_number else (chunk.heading or chunk.content_type)


def _sort_key(c: ClaimView):
    return (" ".join(sorted(tokens(c.metric))), " ".join(sorted(tokens(c.subject))), c.document_id, c.id)


def _value_key(c: ClaimView):
    """Claims with the same key state the same value (number or text), in the same unit and period."""
    number, span = parse_number(c.value), parse_range(c.value)
    if span is not None:
        value = ("range", span[0].normalize(), span[1].normalize())
    else:
        value = number.normalize() if number is not None else (str(c.value).strip().lower() if c.value is not None else None)
    return (value, normalize_unit(c.unit), normalize_period(c.period))


def _clusters(members: list[ClaimView]) -> list[list[ClaimView]]:
    """Group claims by the value they state; each cluster keeps the order of ``members``."""
    clusters: dict[tuple, list[ClaimView]] = {}
    for member in members:
        clusters.setdefault(_value_key(member), []).append(member)
    return list(clusters.values())


def load_claim_rows(session: Session) -> list[tuple[ClaimView, str, str, int]]:
    """Every claim of a document that is still in the data room: (claim, file name, where it is, its chunk's position in the document)."""
    stmt = (select(Claim, Document.filename, DocumentChunk).join(Document, Document.id == Claim.document_id)
            .join(DocumentChunk, DocumentChunk.id == Claim.chunk_id).where(Document.processing_status != ProcessingStatus.missing.value))
    return [(ClaimView(id=c.id, document_id=c.document_id, chunk_id=c.chunk_id, subject=c.subject, metric=c.metric, value=c.value, unit=c.unit, period=c.period,
                       conditions=c.conditions, assumptions=c.assumptions or [], claim_text=c.claim_text, value_type=c.claim_category,
                       table_key=table_key(c.document_id, chunk.content_type, chunk.sheet_name, chunk.page_number)),
             filename, _location(chunk), chunk.chunk_index)
            for c, filename, chunk in session.execute(stmt)]


def _line(ref: str, c: ClaimView, filename: str, where: str) -> str:
    parts = [f"type: {c.value_type or '-'}", f"subject: {c.subject or '-'}", f"metric: {c.metric or '-'}", f"value: {c.value}" + (f" {c.unit}" if c.unit else "")]
    if c.period:
        parts.append(f"period: {c.period}")
    return f"{ref} | {filename}, {where} | " + " | ".join(parts) + f" | {c.claim_text or ''}"


class ClaimGrouper:
    def __init__(self, session: Session, llm: LLMClient, max_chars_per_call: int = 12_000,
                 should_stop: Callable[[], bool] | None = None, embedder: EmbeddingProvider | None = None,
                 neighbors: int = 2, min_similarity: float = 0.72, mutual_floor: float = 0.5, near_similarity: float | None = None, near_rank: int = 5,
                 orphan_count: int = 3, orphan_floor: float = 0.5, workers: int = 1, on_progress: Callable[[str], None] | None = None,
                 cards: str = "", prefilter: bool = False):
        self.prefilter = prefilter  # send only the claims that share subject or metric words with another claim to the grouping calls
        self.cards = cards  # the document cards (what each document is about), shown at the top of every grouping call
        self.workers = max(1, workers)  # grouping calls made at the same time; the answers are still handled in batch order
        self.on_progress = on_progress or (lambda _m: None)
        self.embedder = embedder  # when given, nearest neighbours in other documents are candidates too
        self.neighbors = neighbors
        self.min_similarity = min_similarity
        self.mutual_floor = mutual_floor  # two claims that are each other's best match elsewhere become candidates from this similarity
        self.near_similarity = near_similarity  # when set, any of a claim's top ``near_rank`` matches become candidates from this similarity (off by default: it triples the candidates)
        self.near_rank = near_rank
        self.orphan_count = orphan_count  # a claim with a unit that has no candidate gets its best matches elsewhere in the same unit, from ``orphan_floor``
        self.orphan_floor = orphan_floor
        self.should_stop = should_stop or (lambda: False)
        self.session = session
        self.llm = llm
        self.max_chars = max_chars_per_call

    def _load(self) -> list[tuple[ClaimView, str, str]]:
        return [(v, f, w) for v, f, w, _order in load_claim_rows(self.session)]

    def run(self) -> tuple[list[Candidate], GroupingStats]:
        rows = sorted(self._load(), key=lambda r: _sort_key(r[0]))
        stats = GroupingStats(claims=len(rows))
        found: dict[tuple[str, str], Candidate] = {}
        sent = self._with_partners(rows) if self.prefilter else rows
        stats.claims_filtered = len(rows) - len(sent)
        batches = list(self._batches(sent))
        for number, (batch, result) in enumerate(self._ask(batches), start=1):
            if result is None:
                stats.errors.append("stop requested")
                break
            stats.batches += 1
            stats.llm_calls += 1
            refs = {f"c{i}": view for i, (view, _f, _w) in enumerate(batch, start=1)}
            if isinstance(result, LLMError):
                stats.batches_failed += 1
                stats.errors.append(str(result))
                logger.warning("Claim grouping batch failed", extra={"error": str(result)})
                self.on_progress(f"grouping call {number} of {len(batches)} failed")
                continue
            self.on_progress(f"grouping call {number} of {len(batches)} done")
            used: set[str] = set()
            for group in result.groups:
                members = []
                for ref in dict.fromkeys(r.strip() for r in group.refs):
                    if ref not in refs:
                        stats.unknown_refs += 1
                    elif ref not in used:
                        members.append(ref)
                if len(members) < 2:
                    continue  # a group that cannot be formed does not use up its claims
                used.update(members)
                members = [refs[r] for r in members]
                stats.groups += 1
                stats.claims_grouped += len(members)
                stats.largest_group = max(stats.largest_group, len(members))
                stats.distinct_values += len(_clusters(members))
                # every claim of the group is a candidate of every other: the judge sees them all together, whatever values they state
                for a, b in combinations(members, 2):
                    self._add_candidate(found, a, b)
        before = len(found)
        self._add_shared_identifiers([r[0] for r in rows], found)
        stats.identifier_candidates = len(found) - before
        before = len(found)
        names = defined_names(c.content for c in self.session.scalars(select(DocumentChunk))) if self.session is not None else {}
        resolved = [r[0].model_copy(update={"subject": resolve(r[0].subject, names), "metric": resolve(r[0].metric, names)}) for r in rows]  # for matching only
        for c in generate_candidates(resolved, min_topic_score=LEXICAL_FLOOR, max_partners_per_claim=LEXICAL_PARTNERS):
            found.setdefault((c.claim_id, c.candidate_id), c)  # the code's own matching adds what the model did not group
        stats.lexical_candidates = len(found) - before
        before = len(found)
        for c in shared_label_candidates(resolved):
            found.setdefault((c.claim_id, c.candidate_id), c)  # claims in two documents that share a rare word of their labels
        stats.label_candidates = len(found) - before
        before = len(found)
        for c in negation_candidates([r[0] for r in rows]):
            found.setdefault((c.claim_id, c.candidate_id), c)  # "No X" and the claims that assert X
        stats.polarity_candidates = len(found) - before
        if self.embedder is not None and not self.should_stop():
            before = len(found)
            self._add_neighbors([r[0] for r in rows], found)
            stats.neighbor_candidates = len(found) - before
        stats.candidates = len(found)
        return list(found.values()), stats

    def _add_shared_identifiers(self, views: list[ClaimView], found: dict[tuple[str, str], Candidate]) -> None:
        """Claims in different places that mention the same record (RC-04, ART-118) and state the same kind of value (two dates, two amounts in one unit)
        are candidates, however differently their subjects and metrics are worded: a summary's "dating" and a lab's "Calibrated" of one sample. Only
        dates and quantities qualify: words about one record say many different things. A record id that appears in more than a few claims is too
        common to say anything and is left out."""
        by_id: dict[tuple[str, int], list[ClaimView]] = {}
        for v in views:
            if v.value_type not in ("date", "quantity"):
                continue
            for entity in entity_ids(f"{v.subject or ''} {v.claim_text or ''}"):
                by_id.setdefault(entity, []).append(v)
        for members in by_id.values():
            if len(members) < 2 or len(members) > MAX_CLAIMS_PER_IDENTIFIER:
                continue
            for a, b in combinations(members, 2):
                if a.value_type != b.value_type or a.chunk_id == b.chunk_id:
                    continue
                if a.value_type == "quantity" and units_match(unit_info(a.unit, normalize_unit), unit_info(b.unit, normalize_unit)) is not True:
                    continue
                self._add_candidate(found, a, b, how="identifier")

    @staticmethod
    def _with_partners(rows: list) -> list:
        """The claims that have a plausible partner by the code's own matching (shared subject and metric words, compatible subjects); a claim
        nothing else resembles can state no fact that another claim states, so the LLM need not read it. Nearest-neighbour matching, which
        catches synonyms, still looks at every claim."""
        found = generate_candidates([r[0] for r in rows], max_partners_per_claim=max(len(rows), 1))
        with_partner = {i for c in found for i in (c.claim_id, c.candidate_id)}
        return [r for r in rows if r[0].id in with_partner]

    def _ask(self, batches: list):
        """Yield (batch, answer) in batch order. With several workers the calls run at the same time; an answer is an LLMError when its call
        failed, and None once a stop was requested (calls still waiting for a worker are skipped)."""
        def call(batch):
            if self.should_stop():
                return None
            header = (f"Documents in this collection (what each is about). Never put claims of documents about different, unrelated subjects in one group.\n{self.cards}\n\n"
                      if self.cards else "")
            prompt = header + "Claims:\n\n" + "\n".join(_line(f"c{i}", v, f, w) for i, (v, f, w) in enumerate(batch, start=1))
            try:
                return self.llm.generate_structured_output(prompt, GroupingAnswer, system=SYSTEM)
            except LLMError as exc:
                return exc

        if self.workers <= 1 or len(batches) <= 1:
            for batch in batches:
                answer = call(batch)
                yield batch, answer
                if answer is None:
                    return
            return
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            futures = [(batch, pool.submit(call, batch)) for batch in batches]
            for batch, future in futures:
                yield batch, future.result()

    def _add_neighbors(self, views: list[ClaimView], found: dict[tuple[str, str], Candidate]) -> None:
        """Make each claim's most similar claims in OTHER documents its candidates, whatever their labels say. The claims are embedded
        locally (their subject, metric and sentence); rows of one table are never candidates of each other; the judge decides what they mean."""
        texts = [f"{v.subject or ''}: {v.metric or ''}: {v.claim_text or v.value or ''}" for v in views]
        vectors = []
        for vec in self.embedder.embed(texts, task="document"):
            norm = math.sqrt(sum(x * x for x in vec)) or 1.0
            vectors.append([x / norm for x in vec])
        ranked: list[list[tuple[float, int]]] = []  # for each claim: its matches in other documents, best first
        for i, a in enumerate(views):
            scored = []
            for j, b in enumerate(views):
                if i == j or a.document_id == b.document_id or same_table(a, b) or not comparable_subjects(a, b) or not types_compatible(a, b):
                    continue
                scored.append((sum(x * y for x, y in zip(vectors[i], vectors[j])), j))
            ranked.append(sorted(scored, reverse=True))
        best = [row[0][1] if row else None for row in ranked]
        for i, a in enumerate(views):
            for rank, (similarity, j) in enumerate(ranked[i]):
                strong = rank < self.neighbors and similarity >= self.min_similarity
                mutual = rank == 0 and similarity >= self.mutual_floor and best[j] == i  # each other's best match elsewhere
                near = self.near_similarity is not None and rank < self.near_rank and similarity >= self.near_similarity
                if strong or mutual or near:
                    self._add_candidate(found, a, views[j], how="embedding")
                elif similarity < min(self.mutual_floor, self.near_similarity if self.near_similarity is not None else self.mutual_floor):
                    break
        grouped = {claim_id for key in found for claim_id in key}
        for i, a in enumerate(views):
            if a.id in grouped or not (a.unit or "").strip():
                continue  # only a quantity with a unit is matched this way: its unit is what keeps the candidates few
            unit_a = unit_info(a.unit, normalize_unit)
            same_unit = [(sim, j) for sim, j in ranked[i] if units_match(unit_a, unit_info(views[j].unit, normalize_unit)) is True]
            for similarity, j in same_unit[: self.orphan_count]:
                if similarity >= self.orphan_floor:
                    self._add_candidate(found, a, views[j], how="embedding")

    @staticmethod
    def _add_candidate(found: dict[tuple[str, str], Candidate], a: ClaimView, b: ClaimView, how: str = "llm") -> None:
        if same_table(a, b):  # rows of one table are different items: never compared with each other
            return
        if not comparable_subjects(a, b):  # identifiers of different things, rows of different items, one sentence twice
            return
        first, second = (a, b) if a.id <= b.id else (b, a)
        comparison = compare_attributes(first, second)
        hint = "exact" if same_statement(first, second) and comparison.value_relation != "different" else relation_hint(comparison)
        if (first.id, second.id) in found:  # already a candidate through the grouping
            return
        found[(first.id, second.id)] = Candidate(claim_id=first.id, candidate_id=second.id, topic_score=1.0, relation_hint=hint,
                                                     comparison=comparison, matched_by=how)

    def _batches(self, rows):
        current, size = [], 0
        for row in rows:
            n = len(_line("c000", *row)) + 1
            if current and size + n > self.max_chars:
                yield current
                current, size = [], 0
            current.append(row)
            size += n
        if current:
            yield current
