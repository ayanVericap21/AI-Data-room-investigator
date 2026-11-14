"""Rule linking: the claims that state a rule, or a status for a scope and a period, and the records that fall under them.

Grouping puts together claims that state the same attribute of the same subject. A rule is not that: "no digging in area D", "all operations suspended
from the 13th to the 15th" or "intact painted vessels may not be sampled destructively" never share an attribute with the log row, the date or the lab
step that breaks them. One call reads the claims and links each rule to the records that fall under it (by place, kind of item, people or period, by
meaning and not by shared words). Each link becomes a candidate of the kind "rule"; the judge then checks every record against its rule.

As in the grouping, claims in different batches are never linked together, and the stats report the number of batches.
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable

from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.llm.base import LLMClient
from app.llm.errors import LLMError
from app.schemas.claim import AttributeComparison, Candidate
from app.services.claim_grouping import _line, load_claim_rows
from app.services.temporal_links import windows_and_records

logger = logging.getLogger(__name__)
MAX_RECORDS_PER_RULE = 14  # the judge reads every record of a rule in one call

SYSTEM = """You find rules, and the records that fall under them, in the claims taken from the documents of one collection.

A RULE is a claim that states what is permitted, prohibited, required, limited, scheduled or suspended for a scope (a place, a kind of item, some people, a period), or the status of a scope during a period (closed, suspended, unavailable, expired). A limit or a quota is a rule. So is a universal or negative statement about a scope (none, no, never, all, always: "no subcontractors are engaged", "no incidents occurred", "all invoices were paid"): any record that shows the opposite breaks it.
A RECORD is a claim about something that was done, found, used or measured: an event, an item, a sample, a visit, a row of a log. A record falls under a rule when it is inside the rule's scope (the place, the kind of item, the people, the period), whether or not it obeys the rule.

Rules for your answer:
1. Use ONLY the claims shown. Refer to each claim by its ref (c1, c2, ...) exactly as shown.
2. For each rule, list every claim that belongs to a record under it: the claims that place the record inside the rule's scope (what the item is, where it was, when it happened, who did it) AND the claims that say what was done. Match the scope by meaning, not by shared words: a rule about all trenches covers each trench; a rule about a kind of item covers an item whose own claims describe it as that kind; a rule about a place covers a record whose place is that place; a rule about a period covers every record dated inside that period, in ANY document: when a rule suspends, closes or restricts everything from one date to another, each claim whose date or period falls inside that interval (a row of a log, a visit, a delivery, a measurement) is a record under it, whatever the claim is about. Look through ALL the claims for such records before you answer.
3. Also list claims anywhere in the collection that assert compliance with the rule (all activities complied, nothing was breached).
4. Link a rule only to claims that are records under it or assert compliance with it. Never link claims that merely state the same attribute with different values (those are compared elsewhere). Never link a claim to itself. A claim may be a record under several rules.
5. topic is a few words naming the rule.

OUTPUT FORMAT (strict). Return exactly one JSON object and nothing else: {"links": [ {"rule": "c12", "records": ["c40", "c41"], "topic": "..."} ]}.
- "rule" is one ref. "records" is a list of at least one ref taken from the claims shown. "topic" is a non-empty string.
- If no claim is a rule, or no rule has a record under it, return {"links": []}.
- Do not add other fields, comments, explanations or markdown."""


class RuleLink(BaseModel):
    rule: str
    records: list[str] = Field(default_factory=list)
    topic: str = ""


class RuleLinks(BaseModel):
    links: list[RuleLink] = Field(default_factory=list)


@dataclass
class RuleStats:
    claims: int = 0
    batches: int = 0
    llm_calls: int = 0
    batches_failed: int = 0
    links: int = 0  # rules with at least one record
    candidates: int = 0  # rule-to-record links
    temporal_candidates: int = 0  # of them, added by code: a claim that restricts something between two dates, and a claim dated inside
    unknown_refs: int = 0
    errors: list[str] = field(default_factory=list)
    topics: list[str] = field(default_factory=list)


class RuleLinker:
    def __init__(self, session: Session, llm: LLMClient, max_chars_per_call: int = 300_000, should_stop: Callable[[], bool] | None = None,
                 workers: int = 1, on_progress: Callable[[str], None] | None = None, cards: str = ""):
        self.session, self.llm, self.max_chars = session, llm, max_chars_per_call
        self.should_stop = should_stop or (lambda: False)
        self.workers = max(1, workers)
        self.on_progress = on_progress or (lambda _m: None)
        self.cards = cards

    def run(self) -> tuple[list[Candidate], RuleStats]:
        rows = load_claim_rows(self.session)
        rows.sort(key=lambda r: (r[1], r[3]))  # a document's claims in their order in it: a rule and its records are often near each other
        stats = RuleStats(claims=len(rows))
        found: dict[tuple[str, str], Candidate] = {}
        batches = list(self._batches(rows))
        for number, (batch, result) in enumerate(self._ask(batches), start=1):
            if result is None:
                stats.errors.append("stop requested")
                break
            stats.batches += 1
            stats.llm_calls += 1
            if isinstance(result, LLMError):
                stats.batches_failed += 1
                stats.errors.append(str(result))
                logger.warning("Rule linking batch failed", extra={"error": str(result)})
                self.on_progress(f"rule linking call {number} of {len(batches)} failed")
                continue
            self.on_progress(f"rule linking call {number} of {len(batches)} done")
            refs = {f"c{i}": view for i, (view, _f, _w, _o) in enumerate(batch, start=1)}
            for link in result.links:
                rule = refs.get(link.rule.strip())
                if rule is None:
                    stats.unknown_refs += 1
                    continue
                records = []
                for ref in dict.fromkeys(r.strip() for r in link.records):
                    if ref not in refs:
                        stats.unknown_refs += 1
                    elif refs[ref].id != rule.id:
                        records.append(refs[ref])
                records = records[:MAX_RECORDS_PER_RULE]
                if not records:
                    continue
                stats.links += 1
                stats.topics.append(link.topic.strip() or "(no topic)")
                for record in records:
                    found.setdefault((rule.id, record.id), Candidate(
                        claim_id=rule.id, candidate_id=record.id, topic_score=1.0, relation_hint="partial", matched_by="rule",
                        comparison=AttributeComparison(subject_match=None, metric_match=None, unit_match=None, period_match=None, value_relation="unknown",
                                                       conditions_differ=False, assumptions_differ=False)))
        before = len(found)
        for c in windows_and_records([r[0] for r in rows]):
            found.setdefault((c.claim_id, c.candidate_id), c)
        stats.temporal_candidates = len(found) - before
        stats.candidates = len(found)
        return list(found.values()), stats

    def _batches(self, rows):
        current, size = [], 0
        for row in rows:
            n = len(_line("c000", *row[:3])) + 1
            if current and size + n > self.max_chars:
                yield current
                current, size = [], 0
            current.append(row)
            size += n
        if current:
            yield current

    def _ask(self, batches: list):
        """Yield (batch, answer) in batch order; an answer is an LLMError when its call failed, and None once a stop was requested."""
        def call(batch):
            if self.should_stop():
                return None
            header = f"Documents in this collection (what each is about):\n{self.cards}\n\n" if self.cards else ""
            prompt = header + "Claims:\n\n" + "\n".join(_line(f"c{i}", v, f, w) for i, (v, f, w, _o) in enumerate(batch, start=1))
            try:
                return self.llm.generate_structured_output(prompt, RuleLinks, system=SYSTEM)
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
