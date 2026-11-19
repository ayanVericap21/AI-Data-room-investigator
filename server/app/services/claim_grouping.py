"""Grouping: one stage that puts the claims into disjoint clusters; a claim is in at most one cluster.

Claims are put in document order, cut into batches by size and shown to the LLM in compact one-line form. The claims the extraction flagged as rules are
taken out of the batches and shown to every call in a list of their own (r1, r2, ...). The model returns groups of refs: the claims that state the same
attribute of the same subject, or a rule and the records that fall under it (by place, kind of item, people or period, by meaning and not by shared words).
A rule that comes back from several calls is one cluster, with the records of all of them. No code gate applies to the model's groups, except that claims
that cannot go together (rows of one table) are left out of a cluster.

The code's own matching (shared words, a rare label word, nearest neighbours by meaning, a record id, a restriction between two dates, a negation) is the
fallback of the same stage: it runs only on the claims no model cluster took and returns clusters of those claims only, so it never changes a model
cluster. The clusters it finds that share a claim are one cluster.

The batches are independent and the calls run at the same time (``workers``). Claims in different batches are never grouped together by the model; the
stats report the number of batches so the gap is visible, and the code's own matching still puts such claims together when the model left them alone.
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
from app.schemas.claim import ClaimView, Cluster
from app.services.aliases import defined_names, resolve
from app.services.blocking import inside_document_ok, negation_clusters, shared_label_clusters, types_compatible
from app.services.claim_tags import is_rule
from app.services.clusters import clusters_from_matches, connected_clusters, merge_overlapping
from app.services.identifiers import entity_ids
from app.services.claim_matching import comparable_subjects, match_clusters, normalize_period, normalize_unit, same_table, table_key
from app.services.numbers import parse_number, parse_range
from app.services.temporal_links import windows_and_records
from app.services.value_rules import unit_info, units_match

logger = logging.getLogger(__name__)
MAX_CLAIMS_PER_IDENTIFIER = 12  # a record id in more claims than this is too common to put them together
LEXICAL_FLOOR = 0.7  # topic score from which the code's word matching puts claims together
LEXICAL_PARTNERS = 12  # a claim is matched with at most this many others by its words
MAX_RECORDS_PER_RULE = 14  # the judge reads every record of a rule in one call

HEAD = """You group claims taken from the documents of one collection so that their values can be compared.

Rules:
1. Use ONLY the claims shown. Refer to each claim by its ref exactly as shown (c1, c2, ... in Claims; r1, r2, ... in the RULES list when there is one).
2. Put claims in the same group only when they state the same attribute of the same subject. The wording may differ ("power supply" and "connected power load" can be the same attribute).
3. Claims about different attributes are never grouped, even when their units or numbers look alike (a height and a floor level, a distance and a duration).
4. Claims about different subjects are never grouped (different items, parties or sections), even when the attribute is the same.
5. Group claims whether or not their values agree: the comparison happens later.
6. A claim of Claims belongs to at most one group. A group has at least two claims. Leave a claim out if nothing else states the same thing.
7. topic is a few words naming what the group is about.
8. Never put claims of documents about different, unrelated subjects in one group.
9. Every claim shows its type: quantity, date, identifier, name, state or statement (or - when unknown). The type only describes how the value is written. Do not keep claims apart because their types differ: the same fact can be written in different forms (a status and a count, a date and a duration, a number and a sentence), so group such claims when they state the same attribute of the same subject.
"""

RULES_PART = """10. The RULES list holds claims already known to be rules. A RULE states what is permitted, prohibited, required, limited, scheduled or suspended for a scope (a place, a kind of item, some people, a period), or the status of a scope during a period (closed, suspended, unavailable, expired). A limit or a quota is a rule. So is a universal or negative statement about a scope (none, no, never, all, always).
11. A RECORD is a claim about something that was done, found, used or measured: an event, an item, a sample, a visit, a row of a log. A record falls under a rule when it is inside the rule's scope (the place, the kind of item, the people, the period), whether or not it obeys the rule.
12. For each rule that has records under it, make one group: the rule's ref (r#) and every claim of Claims that is a record under it or asserts compliance with it. Match the scope by meaning, not by shared words: a rule about all trenches covers each trench, a rule about a kind of item covers an item described as that kind, a rule about a period covers every claim dated inside that period, whatever the claim is about. Check every claim of Claims against every rule before you answer. A rule group holds exactly one r# ref.
13. A rule may also be in a group with claims of the same attribute (two limits on one guaranty): a group with two or more r# refs is a group of one attribute. A claim that only states the same attribute as another claim is grouped with that claim, not under a rule.
"""

FORMAT = """
OUTPUT FORMAT (strict). Return exactly one JSON object and nothing else: {"groups": [ {"refs": ["c1", "c2"], "topic": "..."} ]}.
- Every group has "refs", a list of AT LEAST TWO refs taken from the claims shown (and the RULES list, when there is one). Never output a group with one ref, and never put a c ref in two groups.
- Every group has "topic", a non-empty string.
- If no claims belong together, return {"groups": []}.
- Do not add other fields, comments, explanations or markdown."""

SYSTEM = HEAD + FORMAT
SYSTEM_WITH_RULES = HEAD + RULES_PART + FORMAT


class GroupOfRefs(BaseModel):
    refs: list[str]  # a group with fewer than two claims is ignored
    topic: str = ""


class GroupingAnswer(BaseModel):
    groups: list[GroupOfRefs] = Field(default_factory=list)


MIN_SPLIT_CLAIMS = 100  # a failed batch is split only while each half keeps at least this many claims


class GroupingStats(BaseModel):
    claims: int = 0
    rules_shown: int = 0  # claims the extraction flagged as rules, shown to every call (0: the calls only group claims)
    batches: int = 0
    llm_calls: int = 0
    groups: int = 0  # groups of the model that held two or more claims, rule groups included
    rule_groups: int = 0  # of them, a rule and the records under it (before the answers of the calls for one rule are joined)
    claims_grouped: int = 0
    clusters: int = 0  # the clusters of the stage: model clusters, rule clusters and the clusters of the code's matching
    model_clusters: int = 0
    rule_clusters: int = 0  # a rule and its records: found by the model, or by code (a restriction between two dates, a negation)
    code_clusters: int = 0  # clusters of the code's matching of the claims the model left alone
    claims_by_model: int = 0  # claims in a cluster the model made
    claims_by_rule: int = 0  # claims in a rule cluster
    claims_by_code: int = 0  # claims in a cluster the code's matching made
    claims_left: int = 0  # claims no model cluster took: the code's matching looked at these only
    claims_filtered: int = 0  # claims not sent to the grouping calls: nothing else shares their subject or metric words
    identifier_clusters: int = 0  # found by the code on the claims left: claims that mention the same record id and state the same kind of value
    lexical_clusters: int = 0  # subject and metric words that largely coincide
    label_clusters: int = 0  # claims that share a rare word of their labels ("purchase order", "headcount")
    negation_clusters: int = 0  # claims that say there is none of something, each with the claims that mention it
    window_clusters: int = 0  # a claim that restricts something between two dates, with the claims dated inside
    neighbor_clusters: int = 0  # claims whose nearest neighbours elsewhere are alike in meaning
    largest_group: int = 0
    distinct_values: int = 0  # clusters of claims that state the same value, over all groups
    unknown_refs: int = 0
    batches_failed: int = 0
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
    """The start of a grouping call: the document cards, then the claims. The instructions come after it."""
    header = f"Documents in this collection (what each is about):\n{cards}\n\n" if cards else ""
    return header + "Claims:\n\n" + "\n".join(lines)


def with_instructions(block: str, instructions: str) -> str:
    return f"{block}\n\n=== INSTRUCTIONS ===\n\n{instructions}"


class ClaimGrouper:
    def __init__(self, session: Session, llm: LLMClient, max_chars_per_call: int = 12_000,
                 should_stop: Callable[[], bool] | None = None, embedder: EmbeddingProvider | None = None,
                 neighbors: int = 2, min_similarity: float = 0.72, mutual_floor: float = 0.5, near_similarity: float | None = None, near_rank: int = 5,
                 orphan_count: int = 3, orphan_floor: float = 0.5, workers: int = 1, on_progress: Callable[[str], None] | None = None,
                 cards: str = "", prefilter: bool = False, retry_pause_seconds: float = 30.0, sleep: Callable[[float], None] = time.sleep):
        self.retry_pause = retry_pause_seconds  # the wait before a failed batch is asked again in halves: a busy service needs more than the call's own short retries
        self._sleep = sleep
        self._splits = 0
        self._split_lock = threading.Lock()
        self._rules: list = []  # the claims the extraction flagged as rules (rows of ``ordered_rows``), shown to every call
        self._rules_text = ""
        self.prefilter = prefilter  # send only the claims that share subject or metric words with another claim to the grouping calls
        self.cards = cards  # the document cards (what each document is about), shown at the top of every grouping call
        self.workers = max(1, workers)  # grouping calls made at the same time; the answers are handled in batch order
        self.on_progress = on_progress or (lambda _m: None)
        self.embedder = embedder  # when given, nearest neighbours in other documents go into the code's clusters too
        self.neighbors = neighbors
        self.min_similarity = min_similarity
        self.mutual_floor = mutual_floor  # claims that are each other's best match elsewhere go together from this similarity
        self.near_similarity = near_similarity  # when set, any of a claim's top ``near_rank`` matches go together from this similarity (off by default: it triples the matches)
        self.near_rank = near_rank
        self.orphan_count = orphan_count  # a claim with a unit that is in no cluster is put with its best matches elsewhere in the same unit, from ``orphan_floor``
        self.orphan_floor = orphan_floor
        self.should_stop = should_stop or (lambda: False)
        self.session = session
        self.llm = llm
        self.max_chars = max_chars_per_call

    def _load(self) -> list[tuple[ClaimView, str, str]]:
        return [(v, f, w) for v, f, w, _order in ordered_rows(self.session)]

    def run(self) -> tuple[list[Cluster], GroupingStats]:
        everything = self._load()
        self._rules = [row for row in everything if is_rule(row[0].assumptions)]
        rows = [row for row in everything if not is_rule(row[0].assumptions)] if self._rules else everything  # a flagged rule is shown in the RULES list, not in Claims
        rule_refs = {f"r{i}": row[0] for i, row in enumerate(self._rules, start=1)}
        self._rules_text = "\n".join(_line(ref, view, f, w) for ref, (view, f, w) in zip(rule_refs, self._rules))
        stats = GroupingStats(claims=len(everything), rules_shown=len(self._rules))
        self._splits = 0
        sent = self._with_partners(rows) if self.prefilter else rows
        stats.claims_filtered = len(rows) - len(sent)
        batches = list(self._batches(sent))
        by_rule: dict[str, list[str]] = {}  # rule id -> the claims under it, from every call
        rule_topic: dict[str, str] = {}
        facts: list[tuple[list[ClaimView], str]] = []  # the groups of one attribute the model formed: (claims, topic)
        for number, (batch, result) in enumerate(self._ask(batches), start=1):
            if result is None:
                stats.errors.append("stop requested")
                break
            stats.batches += 1
            stats.llm_calls += 1
            here = {f"c{i}": row for i, row in enumerate(batch, start=1)}  # the claims of this call, by the ref the call was shown
            if isinstance(result, LLMError):
                stats.batches_failed += 1
                stats.errors.append(str(result))
                logger.warning("Claim grouping batch failed", extra={"error": str(result)})
                self.on_progress(f"grouping call {number} of {len(batches)} failed" if number <= len(batches) else f"grouping call {number} failed ({len(batch)} claims, after a split)")
                continue
            self.on_progress(f"grouping call {number} of {len(batches)} done" if number <= len(batches) else f"grouping call {number} done ({len(batch)} claims, after a failed batch was split)")
            used: set[str] = set()
            for group in result.groups:
                c_refs: list[str] = []
                r_refs: list[str] = []
                for ref in dict.fromkeys(r.strip() for r in group.refs):
                    if ref in here:
                        if ref not in used:
                            c_refs.append(ref)
                    elif ref in rule_refs:
                        r_refs.append(ref)
                    else:
                        stats.unknown_refs += 1
                if len(r_refs) == 1 and c_refs:  # a rule and the records under it
                    rule = rule_refs[r_refs[0]]
                    used.update(c_refs)
                    under = by_rule.setdefault(rule.id, [])
                    rule_topic.setdefault(rule.id, group.topic.strip())
                    for ref in c_refs:
                        record = here[ref][0]
                        if record.id != rule.id and record.id not in under and len(under) < MAX_RECORDS_PER_RULE:
                            under.append(record.id)
                    stats.groups += 1
                    stats.rule_groups += 1
                    stats.claims_grouped += len(c_refs) + 1
                    stats.largest_group = max(stats.largest_group, len(c_refs) + 1)
                    continue
                if len(c_refs) + len(r_refs) < 2:
                    continue  # a group that cannot be formed does not use up its claims
                used.update(c_refs)
                views = [here[r][0] for r in c_refs] + [rule_refs[r] for r in r_refs]
                facts.append((views, group.topic.strip()))
                stats.groups += 1
                stats.claims_grouped += len(views)
                stats.largest_group = max(stats.largest_group, len(views))
                stats.distinct_values += len(_clusters(views))
                # the judge sees the whole group together, whatever values its claims state
        stats.batches_split = self._splits

        found: list[Cluster] = []
        taken: set[str] = set()  # the claims in a cluster: a claim is in one cluster only

        def add(cluster: Cluster, via: str | None = None) -> bool:
            ids = [i for i in cluster.claim_ids if i not in taken]
            if len(ids) < 2 or (cluster.rule_id is not None and cluster.rule_id not in ids):
                return False
            found.append(cluster.model_copy(update={"claim_ids": ids, "via": {i: via or cluster.via.get(i, cluster.matched_by) for i in ids}}))
            taken.update(ids)
            return True

        for rule_id, under in by_rule.items():  # one cluster for a rule, with the records of every call
            if add(Cluster(claim_ids=[rule_id, *under], topic=rule_topic.get(rule_id, ""), matched_by="rule", rule_id=rule_id), "rule"):
                stats.rule_clusters += 1
        for views, topic in facts:  # the claims that cannot go together (rows of one table) are left out of the model's cluster
            for cluster in connected_clusters(self._matches(views), "llm", topic):
                stats.model_clusters += add(cluster, "llm")
        stats.claims_by_model = sum(len(c.claim_ids) for c in found if c.matched_by == "llm")
        stats.claims_by_rule = sum(len(c.claim_ids) for c in found if c.rule_id is not None)

        # the code's own matching: only the claims no model cluster took
        views_all = [row[0] for row in everything]
        left = [v for v in views_all if v.id not in taken]
        stats.claims_left = len(left)
        for cluster in windows_and_records(left):  # a restriction between two dates and the claims dated inside it
            stats.window_clusters += add(cluster, "window")
        left = [v for v in left if v.id not in taken]
        for cluster in negation_clusters(left):  # "No X" and the claims that mention X
            stats.negation_clusters += add(cluster, "negation")
        stats.rule_clusters = sum(c.rule_id is not None for c in found)
        left = [v for v in left if v.id not in taken]
        names = defined_names(c.content for c in self.session.scalars(select(DocumentChunk))) if self.session is not None else {}
        resolved = [v.model_copy(update={"subject": resolve(v.subject, names), "metric": resolve(v.metric, names)}) for v in left]  # for matching only
        routes: list[tuple[str, Cluster]] = []
        for cluster in self._identifier_clusters(left):
            routes.append(("identifier", cluster))
            stats.identifier_clusters += 1
        for cluster in match_clusters(resolved, min_topic_score=LEXICAL_FLOOR, max_partners_per_claim=LEXICAL_PARTNERS):
            routes.append(("lexical", cluster))
            stats.lexical_clusters += 1
        for cluster in shared_label_clusters(resolved):  # claims in two documents that share a rare word of their labels
            routes.append(("label", cluster))
            stats.label_clusters += 1
        if self.embedder is not None and not self.should_stop():
            for cluster in self._neighbor_clusters(left, {i for _how, c in routes for i in c.claim_ids}):
                routes.append(("embedding", cluster))
                stats.neighbor_clusters += 1
        via: dict[str, str] = {}
        for how, cluster in routes:  # the route that first found a claim is the one reported for it
            for claim_id in cluster.claim_ids:
                via.setdefault(claim_id, how)
        topics: dict[str, str] = {}
        scores: dict[str, float] = {}
        for _how, cluster in routes:
            for claim_id in cluster.claim_ids:
                topics.setdefault(claim_id, cluster.topic)
                scores[claim_id] = max(scores.get(claim_id, 0.0), cluster.topic_score)
        for ids in merge_overlapping([cluster.claim_ids for _how, cluster in routes]):  # code clusters that share a claim are one cluster
            code = Cluster(claim_ids=ids, topic=topics.get(ids[0], ""), matched_by=via[ids[0]], topic_score=max(scores[i] for i in ids), via={i: via[i] for i in ids})
            stats.code_clusters += add(code)
        stats.claims_by_code = sum(len(c.claim_ids) for c in found if c.rule_id is None and set(c.via.values()) != {"llm"})
        stats.clusters = len(found)
        return found, stats

    @staticmethod
    def _together(a: ClaimView, b: ClaimView) -> bool:
        """Two claims can go into one cluster: rows of one table are different items, and identifiers of different things, rows of different items or one
        sentence twice are not about one fact."""
        return not same_table(a, b) and comparable_subjects(a, b)

    @classmethod
    def _matches(cls, views: list[ClaimView]) -> list[tuple[str, str, float]]:
        """The claims of ``views`` that can go together, as the matches the cluster is made of."""
        return [(a.id, b.id, 1.0) for a, b in combinations(views, 2) if cls._together(a, b)]

    def _identifier_clusters(self, views: list[ClaimView]) -> list[Cluster]:
        """Claims in different places that mention the same record (RC-04, ART-118) and state the same kind of value (two dates, two amounts in one unit)
        are one cluster, however differently their subjects and metrics are worded: a summary's "dating" and a lab's "Calibrated" of one sample. Only
        dates and quantities qualify: words about one record say many different things. A record id that appears in more than a few claims is too
        common to say anything and is left out."""
        out: list[Cluster] = []
        by_id: dict[tuple[str, int], list[ClaimView]] = {}
        for v in views:
            if v.value_type not in ("date", "quantity"):
                continue
            for entity in entity_ids(f"{v.subject or ''} {v.claim_text or ''}"):
                by_id.setdefault(entity, []).append(v)
        for members in by_id.values():
            if len(members) < 2 or len(members) > MAX_CLAIMS_PER_IDENTIFIER:
                continue
            matches = []
            for a, b in combinations(members, 2):
                if a.value_type != b.value_type or a.chunk_id == b.chunk_id or not self._together(a, b):
                    continue
                if a.value_type == "quantity" and units_match(unit_info(a.unit, normalize_unit), unit_info(b.unit, normalize_unit)) is not True:
                    continue
                matches.append((a.id, b.id, 1.0))
            out += connected_clusters(matches, "identifier")
        return out

    @staticmethod
    def _with_partners(rows: list) -> list:
        """The claims that have a plausible partner by the code's own matching (shared subject and metric words, compatible subjects); a claim
        nothing else resembles can state no fact that another claim states, so the LLM need not read it. Nearest-neighbour matching, which
        catches synonyms, still looks at every claim."""
        found = match_clusters([r[0] for r in rows], max_partners_per_claim=max(len(rows), 1))
        with_partner = {i for c in found for i in c.claim_ids}
        return [r for r in rows if r[0].id in with_partner]

    def _once(self, batch):
        """One grouping call for this batch (the client's own retries included): the answer, an LLMError when the call failed, None once a stop was requested."""
        if self.should_stop():
            return None
        block = claims_block(self.cards, [_line(f"c{i}", v, f, w) for i, (v, f, w) in enumerate(batch, start=1)])
        if self._rules:
            prompt = with_instructions(f"{block}\n\n=== RULES (claims the extraction flagged as rules) ===\n\n{self._rules_text}", SYSTEM_WITH_RULES)
        else:
            prompt = with_instructions(block, SYSTEM)
        try:
            return self.llm.generate_structured_output(prompt, GroupingAnswer)
        except LLMError as exc:
            return exc

    def _pause(self) -> None:
        waited = 0.0
        while waited < self.retry_pause and not self.should_stop():
            self._sleep(min(1.0, self.retry_pause - waited))
            waited += 1.0

    def _call(self, batch) -> list:
        """[(batch, answer)]. A call that failed because the service was busy or its answer was unusable is not the end of its claims: after a pause the
        batch is asked again as two halves (smaller prompts), and a half that fails is split again, while each half keeps MIN_SPLIT_CLAIMS claims.
        Halves are grouped separately, so a claim is never grouped with one in the other half; that is better than no grouping at all."""
        answer = self._once(batch)
        if isinstance(answer, (LLMTransientError, LLMInvalidOutputError)) and len(batch) >= 2 * MIN_SPLIT_CLAIMS:
            self._pause()
            if self.should_stop():
                return [(batch, None)]
            with self._split_lock:
                self._splits += 1
            self.on_progress(f"grouping call for {len(batch)} claims failed: asking again as two halves")
            half = len(batch) // 2
            return self._call(batch[:half]) + self._call(batch[half:])
        return [(batch, answer)]

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

    def _neighbor_clusters(self, views: list[ClaimView], grouped: set[str]) -> list[Cluster]:
        """Put each claim together with its most similar claims in OTHER documents, and its most similar claims in other passages of its OWN document (a
        document can contradict itself), whatever their labels say: a cluster of each claim and its matches. ``grouped`` holds the claims that are already in
        a cluster of another route. The claims are embedded locally (their subject, metric and sentence); rows of one table never go together; the judge
        decides what they mean."""
        if not views:
            return []
        matches: list[tuple[str, str, float]] = []
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
                        if self._together(a, views[j]):
                            matches.append((a.id, views[j].id, similarity))
                    elif similarity < min(self.mutual_floor, self.near_similarity if self.near_similarity is not None else self.mutual_floor):
                        break
        grouped = grouped | {claim_id for a_id, b_id, _s in matches for claim_id in (a_id, b_id)}
        for i, a in enumerate(views):
            if a.id in grouped or not (a.unit or "").strip():
                continue  # only a quantity with a unit is matched this way: its unit is what keeps the clusters few
            unit_a = unit_info(a.unit, normalize_unit)
            same_unit = [(sim, j) for sim, j in ranked[i] if units_match(unit_a, unit_info(views[j].unit, normalize_unit)) is True]
            for similarity, j in same_unit[: self.orphan_count]:
                if similarity >= self.orphan_floor and self._together(a, views[j]):
                    matches.append((a.id, views[j].id, similarity))
        return clusters_from_matches(matches, "embedding")

    def _batches(self, rows):
        """The claims cut into batches of at most ``max_chars`` characters, less the RULES list every call also holds (at least half of ``max_chars`` stay for claims)."""
        limit = max(self.max_chars - len(self._rules_text), self.max_chars // 2)
        current, size = [], 0
        for row in rows:
            n = len(_line("c000", *row)) + 1
            if current and size + n > limit:
                yield current
                current, size = [], 0
            current.append(row)
            size += n
        if current:
            yield current
