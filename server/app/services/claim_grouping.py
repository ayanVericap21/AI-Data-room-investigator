"""Grouping by the LLM (an alternative to the code-based candidate matching, chosen with a flag).

Claims are put in document order (the order the rule linking uses too, so both calls start with the same text and the provider's
prompt cache can serve the second), cut into batches by size, and shown to the LLM in compact one-line form. It returns groups of claims that state the same attribute of the same
subject; every claim of a group is a candidate of every other, and the groups go to the same ``ClaimGroupJudge`` as
code-made candidates. No code gate (subject overlap, unit, partner cap) applies here: the LLM decides what belongs
together, and the judge and the quote checks after it decide what becomes a finding.

Claims in different batches are never grouped together, unless ``carry_groups`` is on: the batches then go one after the other and each
call after the first is shown the groups (and the ungrouped claims) of the earlier ones, so a claim can join an existing group. The stats
report the number of batches so the gap is visible.
"""
from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations
from typing import Callable

import numpy as np
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.llm.base import LLMClient
from app.llm.embeddings import EmbeddingProvider
from app.llm.errors import LLMError, LLMInvalidOutputError, LLMTransientError
from app.models import Claim, Document, DocumentChunk
from app.models.base import ProcessingStatus
from app.schemas.claim import Candidate, ClaimView
from app.services.aliases import defined_names, resolve
from app.services.blocking import inside_document_ok, negation_candidates, shared_label_candidates, types_compatible
from app.services.identifiers import entity_ids
from app.services.candidate_matching import (compare_attributes, generate_candidates, normalize_period, normalize_unit, relation_hint, same_statement,
                                             comparable_subjects, same_table, table_key)
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
8. Never put claims of documents about different, unrelated subjects in one group.
9. Every claim shows its type: quantity, date, identifier, name, state or statement (or - when unknown). The type only describes how the value is written. Do not keep claims apart because their types differ: the same fact can be written in different forms (a status and a count, a date and a duration, a number and a sentence), so group such claims when they state the same attribute of the same subject.

OUTPUT FORMAT (strict). Return exactly one JSON object and nothing else: {"groups": [ {"refs": ["c1", "c2"], "topic": "..."} ]}.
- Every group has "refs", a list of AT LEAST TWO refs taken from the claims shown. Never output a group with one ref, and never put a ref in two groups.
- Every group has "topic", a non-empty string.
- If no claims belong together, return {"groups": []}.
- Do not add other fields, comments, explanations or markdown."""


class GroupOfRefs(BaseModel):
    refs: list[str]  # a group with fewer than two claims is ignored
    topic: str = ""


class JoinOfRefs(BaseModel):
    group: str  # a group formed by an earlier call, as it was shown (g1, g2, ...)
    refs: list[str]  # claims of this call that belong to it


class GroupingAnswer(BaseModel):
    groups: list[GroupOfRefs] = Field(default_factory=list)
    joins: list[JoinOfRefs] = Field(default_factory=list)  # only in a call that was shown the groups of earlier calls


# Added to the instructions of every call after the first when the groups of earlier calls are carried forward.
EARLIER_GROUPS = """

EARLIER GROUPS. This call also shows the groups that earlier calls formed (g1, g2, ...: one example claim, the topic and the size of each) and the claims of earlier calls that are in no group (u1, u2, ...).
- A claim of this call (c1, c2, ...) that states the same attribute of the same subject as an earlier group belongs in that group: list it under "joins" with the group's ref. Judge by the attribute and the subject, not by the value.
- Claims of this call that belong together, with each other or with an ungrouped earlier claim (u1, u2, ...), form a new group under "groups": its refs may be c and u refs.
- Never put a g ref in "groups" and never put a u ref in "joins". A claim is in one group or one join at most.
OUTPUT FORMAT for this call (it replaces the one above). Return exactly one JSON object and nothing else: {"groups": [ {"refs": ["c1", "u4"], "topic": "..."} ], "joins": [ {"group": "g3", "refs": ["c5", "c9"]} ]}. Both lists may be empty. Do not add other fields, comments, explanations or markdown."""


MIN_SPLIT_CLAIMS = 100  # a failed batch is split only while each half keeps at least this many claims


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
    carried: bool = False  # later calls were shown the groups of the earlier ones
    joined: int = 0  # claims that a later call put into a group an earlier call had formed
    batches_split: int = 0  # batches whose call failed and were asked again as two halves (the claims are not lost with the call)
    errors: list[str] = Field(default_factory=list)


def _location(chunk: DocumentChunk) -> str:
    if chunk.sheet_name:
        return f"sheet {chunk.sheet_name}, {chunk.cell_reference}"
    return f"page {chunk.page_number}" if chunk.page_number else (chunk.heading or chunk.content_type)


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


def ordered_rows(session: Session) -> list[tuple[ClaimView, str, str, int]]:
    """Every claim in document order: by file, then by its chunk's position, then by id (a total order, so two calls see the same text)."""
    rows = load_claim_rows(session)
    rows.sort(key=lambda r: (r[1], r[3], r[0].id))
    return rows


def claims_block(cards: str, lines: list[str]) -> str:
    """The start of the grouping call and of the rule-linking call. It is the same text in both (given the same claims in the same order), and each
    call's instructions come after it, so the provider's prompt cache can serve the second call's claims from the first."""
    header = f"Documents in this collection (what each is about):\n{cards}\n\n" if cards else ""
    return header + "Claims:\n\n" + "\n".join(lines)


def with_instructions(block: str, instructions: str) -> str:
    return f"{block}\n\n=== INSTRUCTIONS ===\n\n{instructions}"


class ClaimGrouper:
    def __init__(self, session: Session, llm: LLMClient, max_chars_per_call: int = 12_000,
                 should_stop: Callable[[], bool] | None = None, embedder: EmbeddingProvider | None = None,
                 neighbors: int = 2, min_similarity: float = 0.72, mutual_floor: float = 0.5, near_similarity: float | None = None, near_rank: int = 5,
                 orphan_count: int = 3, orphan_floor: float = 0.5, workers: int = 1, on_progress: Callable[[str], None] | None = None,
                 cards: str = "", prefilter: bool = False, retry_pause_seconds: float = 30.0, sleep: Callable[[float], None] = time.sleep,
                 carry_groups: bool = False):
        self.carry_groups = carry_groups  # batches go one after the other, and each call after the first is shown the groups (and the ungrouped claims) of the earlier ones
        self._known_groups: list[dict] = []  # {"topic", "rows": [(claim, file, place)]} of the groups formed so far
        self._loose: list[tuple[ClaimView, str, str]] = []  # claims of earlier calls that are in no group
        self.retry_pause = retry_pause_seconds  # the wait before a failed batch is asked again in halves: a busy service needs more than the call's own short retries
        self._sleep = sleep
        self._splits = 0
        self._split_lock = threading.Lock()
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
        return [(v, f, w) for v, f, w, _order in ordered_rows(self.session)]

    def run(self) -> tuple[list[Candidate], GroupingStats]:
        rows = self._load()
        stats = GroupingStats(claims=len(rows))
        self._splits = 0
        found: dict[tuple[str, str], Candidate] = {}
        sent = self._with_partners(rows) if self.prefilter else rows
        stats.claims_filtered = len(rows) - len(sent)
        batches = list(self._batches(sent))
        carry = self.carry_groups and len(batches) > 1
        stats.carried = carry
        self._known_groups, self._loose = [], []
        stream = self._sequential(batches) if carry else ((b, a, [], []) for b, a in self._ask(batches))
        for number, (batch, result, earlier, loose) in enumerate(stream, start=1):
            if result is None:
                stats.errors.append("stop requested")
                break
            stats.batches += 1
            stats.llm_calls += 1
            here = {f"c{i}": row for i, row in enumerate(batch, start=1)}  # the claims of this call, by the ref the call was shown
            known_rows = {**here, **{f"u{i}": row for i, row in enumerate(loose, start=1)}}  # the ungrouped claims of earlier calls can join a new group
            if isinstance(result, LLMError):
                if carry:
                    self._loose = self._loose + list(batch)
                stats.batches_failed += 1
                stats.errors.append(str(result))
                logger.warning("Claim grouping batch failed", extra={"error": str(result)})
                self.on_progress(f"grouping call {number} of {len(batches)} failed" if number <= len(batches) else f"grouping call {number} failed ({len(batch)} claims, after a split)")
                continue
            self.on_progress(f"grouping call {number} of {len(batches)} done" if number <= len(batches) else f"grouping call {number} done ({len(batch)} claims, after a failed batch was split)")
            used: set[str] = set()
            for group in result.groups:
                members = []
                for ref in dict.fromkeys(r.strip() for r in group.refs):
                    if ref not in known_rows:
                        stats.unknown_refs += 1
                    elif ref not in used:
                        members.append(ref)
                if len(members) < 2:
                    continue  # a group that cannot be formed does not use up its claims
                used.update(members)
                if carry:
                    self._known_groups.append({"topic": group.topic.strip(), "rows": [known_rows[r] for r in members]})
                members = [known_rows[r][0] for r in members]
                stats.groups += 1
                stats.claims_grouped += len(members)
                stats.largest_group = max(stats.largest_group, len(members))
                stats.distinct_values += len(_clusters(members))
                # every claim of the group is a candidate of every other: the judge sees them all together, whatever values they state
                for a, b in combinations(members, 2):
                    self._add_candidate(found, a, b)
            for join in result.joins if carry else []:
                ref = join.group.strip()
                if not (ref[:1] == "g" and ref[1:].isdigit() and 1 <= int(ref[1:]) <= len(earlier)):
                    stats.unknown_refs += 1
                    continue
                target = earlier[int(ref[1:]) - 1]
                new = []
                for r in dict.fromkeys(x.strip() for x in join.refs):
                    if r not in here:
                        stats.unknown_refs += 1
                    elif r not in used:
                        new.append(r)
                if not new:
                    continue
                used.update(new)
                before_values = len(_clusters([row[0] for row in target["rows"]]))
                added = [here[r] for r in new]
                for old in target["rows"]:
                    for row in added:
                        self._add_candidate(found, old[0], row[0])
                for a, b in combinations(added, 2):
                    self._add_candidate(found, a[0], b[0])
                target["rows"].extend(added)
                stats.joined += len(added)
                stats.claims_grouped += len(added)
                stats.largest_group = max(stats.largest_group, len(target["rows"]))
                stats.distinct_values += len(_clusters([row[0] for row in target["rows"]])) - before_values
            if carry:  # what no group took stays in view for the next call
                taken = {known_rows[r][0].id for r in used}
                self._loose = [row for row in self._loose if row[0].id not in taken] + [row for r, row in here.items() if r not in used]
        stats.batches_split = self._splits
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

    @staticmethod
    def _render_known(groups: list[dict], loose: list) -> str:
        """The groups formed by earlier calls (one example claim each, with the topic and the size) and the claims they left ungrouped, as the next call sees them."""
        if not groups and not loose:
            return ""
        lines = ["Groups formed by earlier calls (ref | file, place | example claim | topic and size):"]
        for i, g in enumerate(groups, start=1):
            v, f, w = g["rows"][0]
            lines.append(f"{_line(f'g{i}', v, f, w)} | group topic: {g['topic'] or '-'} | {len(g['rows'])} claims")
        lines += ["", "Claims of earlier calls that are in no group:"] + [_line(f"u{i}", v, f, w) for i, (v, f, w) in enumerate(loose, start=1)]
        return "\n".join(lines)

    def _once(self, batch, known: str = ""):
        """One grouping call for this batch (the client's own retries included): the answer, an LLMError when the call failed, None once a stop was requested."""
        if self.should_stop():
            return None
        block = claims_block(self.cards, [_line(f"c{i}", v, f, w) for i, (v, f, w) in enumerate(batch, start=1)])
        prompt = with_instructions(block + (f"\n\n=== GROUPS OF EARLIER CALLS ===\n\n{known}" if known else ""), SYSTEM + (EARLIER_GROUPS if known else ""))
        try:
            return self.llm.generate_structured_output(prompt, GroupingAnswer)
        except LLMError as exc:
            return exc

    def _pause(self) -> None:
        waited = 0.0
        while waited < self.retry_pause and not self.should_stop():
            self._sleep(min(1.0, self.retry_pause - waited))
            waited += 1.0

    def _call(self, batch, known: str = "") -> list:
        """[(batch, answer)]. A call that failed because the service was busy or its answer was unusable is not the end of its claims: after a pause the
        batch is asked again as two halves (smaller prompts), and a half that fails is split again, while each half keeps MIN_SPLIT_CLAIMS claims.
        Halves are grouped separately, so a claim is never grouped with one in the other half; that is better than no grouping at all."""
        answer = self._once(batch, known)
        if isinstance(answer, (LLMTransientError, LLMInvalidOutputError)) and len(batch) >= 2 * MIN_SPLIT_CLAIMS:
            self._pause()
            if self.should_stop():
                return [(batch, None)]
            with self._split_lock:
                self._splits += 1
            self.on_progress(f"grouping call for {len(batch)} claims failed: asking again as two halves")
            half = len(batch) // 2
            return self._call(batch[:half], known) + self._call(batch[half:], known)
        return [(batch, answer)]

    def _sequential(self, batches: list):
        """(batch, answer, groups, ungrouped) one batch after the other: each call is shown the groups and the ungrouped claims the earlier ones left, and the
        caller must have recorded the previous answer before asking for the next (it is a generator)."""
        for batch in batches:
            groups, loose = list(self._known_groups), list(self._loose)
            for piece, answer in self._call(batch, self._render_known(groups, loose)):
                yield piece, answer, groups, loose
                if answer is None:
                    return

    def _ask(self, batches: list):
        """Yield (batch, answer) in batch order; a failed batch yields its halves instead (see ``_call``). With several workers the calls run at the
        same time; an answer is an LLMError when its call failed, and None once a stop was requested (calls still waiting for a worker are skipped)."""
        if self.workers <= 1 or len(batches) <= 1:
            for batch in batches:
                for piece, answer in self._call(batch):
                    yield piece, answer
                    if answer is None:
                        return
            return
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            futures = [pool.submit(self._call, batch) for batch in batches]
            for future in futures:
                yield from future.result()

    def _add_neighbors(self, views: list[ClaimView], found: dict[tuple[str, str], Candidate]) -> None:
        """Make each claim's most similar claims in OTHER documents its candidates, and its most similar claims in other passages of its OWN document (a
        document can contradict itself), whatever their labels say. The claims are embedded locally (their subject, metric and sentence); rows of one
        table are never candidates of each other; the judge decides what they mean."""
        if not views:
            return
        texts = [f"{v.subject or ''}: {v.metric or ''}: {v.claim_text or v.value or ''}" for v in views]
        vectors = np.asarray(self.embedder.embed(texts, task="document"), dtype=np.float64)  # float64: float32 changes the order of near ties
        norms = np.linalg.norm(vectors, axis=1)
        vectors = vectors / np.where(norms == 0.0, 1.0, norms)[:, None]  # every claim as a unit vector
        similarity = vectors @ vectors.T  # every claim against every other one, in one product
        ranked: list[list[tuple[float, int]]] = []  # for each claim: its matches in other documents, best first
        ranked_here: list[list[tuple[float, int]]] = []  # and its matches in other passages of its own document, ranked apart so they never take a cross-document claim's places
        for i, a in enumerate(views):
            scored, scored_here = [], []
            for j, b in enumerate(views):
                if i == j or same_table(a, b) or not comparable_subjects(a, b) or not types_compatible(a, b):
                    continue
                if not inside_document_ok(a, b):
                    continue  # inside one document only numbers and dates of different passages are compared (one passage is one statement)
                # rounded to 12 decimals: the product and a plain sum differ by about 1e-16, which must not reorder two matches that are equal
                (scored_here if a.document_id == b.document_id else scored).append((round(float(similarity[i, j]), 12), j))
            ranked.append(sorted(scored, reverse=True))
            ranked_here.append(sorted(scored_here, reverse=True))
        for lists in (ranked, ranked_here):
            best = [row[0][1] if row else None for row in lists]
            for i, a in enumerate(views):
                for rank, (similarity, j) in enumerate(lists[i]):
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
