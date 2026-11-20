"""Relations between figures in DIFFERENT places of the data room that the figures must satisfy: arithmetic across documents and the order of dates.

Grouping compares claims that state the same attribute, so a figure that is a function of other figures (a credit that is a rate times a base, a
total that is the sum of parts stated elsewhere) never meets its inputs, and neither do two dates that must come in a given order (a loan that
matures before the payment meant to repay it). Here the model reads the numeric and date claims of the whole room and PROPOSES relations between
them; code does all the arithmetic and the date comparison, applying each claim's unit and scale, and only a relation that fails becomes a finding.
The model is never asked to calculate, and a relation it proposes is saved only when the figures break it. Quotes are taken from the documents and
verified like every other. Findings the earlier stages already hold (the same passages) are not saved twice.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Callable, Literal

from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.llm.base import LLMClient
from app.llm.errors import LLMError
from app.models import DocumentChunk, Evidence, Finding
from app.services.claim_grouping import _line, load_claim_rows
from app.services.internal_checks import TOLERANCE, _number, _scale, clause_quote, explained_without_a_term
from app.services.value_rules import Num, parse_date
from app.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)
KINDS = ("quantity", "date")  # the claims the model reads: a figure or a date
# a row the document itself sets apart from the figures around it: it is not a part of the total beside it
MEMO = re.compile(r"\(memo\)|\bmemo\b|\binterim only\b|\bfor information only\b|\bpro forma\b", re.IGNORECASE)

SYSTEM = """You find relations that figures from different places of a data room must satisfy. Code checks each relation you give and only a relation that fails becomes a finding, so give only relations that the documents themselves require.

Two kinds of relation:
1. amount: claims, each multiplied by a count and/or by a rate that is itself a claim, add up to another claim. Examples: a credit equals the credit rate times the qualified costs (terms: the costs claim with rate = the rate claim; against: the credit claim); a total equals the sum of parts stated in other places; a figure in a summary equals the figure it summarises; installments add up to the amount they pay. relation is equals, at_most or at_least.
2. order: one date must not be later than another. Examples: a payment that repays a loan arrives on or before the loan matures; an approval comes before the work it permits; a deadline is met by the event it names; an event cannot come before the one it depends on. earlier is the claim whose date must come first; later is the claim whose date must not be before it.

Rules:
1. Use ONLY the claims shown, by ref (c1, c2, ...) exactly as shown. Do not calculate, convert or infer values: code does that, applying each claim's unit and scale.
2. A relation must be required by what the documents say: a rate or formula one of them states, an accounting identity, or a dependency the text names (repaid from, subject to, due by). Do not give relations between figures that merely look related.
3. Use claims of the same currency and the same basis; do not mix a planned and an actual figure unless the relation is between exactly those.
4. reason: one sentence naming the sentence or rule that makes the relation required.
5. Do not repeat a relation.

OUTPUT FORMAT (strict). Return exactly one JSON object and nothing else: {"checks": [ {"label": "...", "kind": "amount", "terms": [{"ref": "c3", "times": 1, "rate": "c9"}], "relation": "equals", "against": "c4", "reason": "..."}, {"label": "...", "kind": "order", "earlier": "c7", "later": "c8", "reason": "..."} ]}. Leave out the fields a kind does not use; rate is a ref or left out. If there is none, return {"checks": []}. Add no other fields, comments or markdown."""


class Term(BaseModel):
    ref: str
    times: float = 1.0  # a count of periods or items, from the words
    rate: str = ""  # a claim holding a percentage or a ratio that multiplies this term


class CrossRelation(BaseModel):
    label: str = ""
    kind: Literal["amount", "order"] = "amount"
    terms: list[Term] = Field(default_factory=list)
    relation: Literal["equals", "at_most", "at_least"] = "equals"
    against: str = ""
    earlier: str = ""
    later: str = ""
    reason: str = ""


class CrossRelations(BaseModel):
    checks: list[CrossRelation] = Field(default_factory=list)


@dataclass
class CrossStats:
    claims: int = 0
    calls: int = 0
    failed: int = 0
    proposed: int = 0  # relations the model proposed
    checked: int = 0  # relations code could check (numbers or dates it could read)
    held: int = 0
    explained: int = 0  # of those held, sums met exactly once one listed figure is set aside (a memo or interim row)
    broken: int = 0
    unusable: int = 0  # a ref that is not a claim, a claim twice, a value that is not a plain number or a date
    already_reported: int = 0  # broken, but the same passages are already in a finding of this run
    saved: int = 0
    rejected: int = 0
    log: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def _amount(view) -> Num | None:
    """The claim's number in whole units: '24,600' with the unit '$ in thousands' is 24,600,000."""
    n = _number(view.value)
    if n is None or n.low is None or n.high is None:
        return None
    return n.scaled(_scale(view))


def _rate(view) -> Num | None:
    """A percentage as a fraction (20% is 0.2); a plain ratio (0.88) as it is."""
    n = _number(view.value)
    if n is None or n.low is None or n.high is None:
        return None
    return n.scaled(Decimal("0.01")) if "%" in str(view.value) or (view.unit or "").strip() == "%" else n


def _product(a: tuple[Decimal, Decimal], b: tuple[Decimal, Decimal]) -> tuple[Decimal, Decimal]:
    corners = [a[0] * b[0], a[0] * b[1], a[1] * b[0], a[1] * b[1]]
    return min(corners), max(corners)


def _span(value):
    found = parse_date(value)
    return found if isinstance(found, tuple) else None


class CrossChecker:
    def __init__(self, session: Session, llm: LLMClient, registry: ToolRegistry, max_chars_per_call: int = 300_000,
                 should_stop: Callable[[], bool] | None = None, on_progress: Callable[[str], None] | None = None):
        self.session, self.llm, self.registry, self.max_chars = session, llm, registry, max_chars_per_call
        self.should_stop = should_stop or (lambda: False)
        self.on_progress = on_progress or (lambda _m: None)

    def run(self) -> CrossStats:
        rows = [r for r in load_claim_rows(self.session) if r[0].value_type in KINDS]
        rows.sort(key=lambda r: (r[1], r[3]))
        stats = CrossStats(claims=len(rows))
        if len(rows) < 2:
            return stats
        chunks = {c.id: c for c in self.session.scalars(select(DocumentChunk).where(DocumentChunk.id.in_({r[0].chunk_id for r in rows})))}
        self._seen: set[frozenset[str]] = set()
        self._reported = self._reported_passages()
        for batch in self._batches(rows):
            if self.should_stop():
                break
            try:
                answer = self.llm.generate_structured_output("Claims, document by document:\n" + self._text(batch), CrossRelations, system=SYSTEM)
            except LLMError as exc:
                stats.calls += 1
                stats.failed += 1
                stats.errors.append(str(exc))
                continue
            stats.calls += 1
            refs = {f"c{i}": r for i, r in enumerate(batch, start=1)}
            for check in answer.checks:
                stats.proposed += 1
                self._verify(check, refs, chunks, stats)
        return stats

    def _batches(self, rows):
        """As many claims to a call as the character budget allows, in document order. A relation uses claims of one call, so a room larger than
        one call loses the relations that cross two calls."""
        current: list = []
        size = 0
        for r in rows:
            n = len(_line("c000", *r[:3])) + 1
            if current and size + n > self.max_chars:
                yield current
                current, size = [], 0
            current.append(r)
            size += n
        if len(current) >= 2:
            yield current

    @staticmethod
    def _text(batch) -> str:
        lines, last = [], None
        for i, (v, f, w, _o) in enumerate(batch, start=1):
            if f != last:
                lines.append(f"\n### Document: {f}")
                last = f
            lines.append(_line(f"c{i}", v, f, w))
        return "\n".join(lines)

    def _reported_passages(self) -> list[frozenset[tuple[str, str]]]:
        """The (chunk, quoted text) set of every finding of this run so far: a relation that quotes the same passages is already a finding."""
        investigation_id = self.registry.context.investigation_id
        if investigation_id is None:
            return []
        rows = self.session.execute(select(Evidence.finding_id, Evidence.chunk_id, Evidence.quoted_text).join(Finding, Finding.id == Evidence.finding_id)
                                    .where(Finding.investigation_id == investigation_id))
        by_finding: dict[str, set[tuple[str, str]]] = {}
        for finding_id, chunk_id, quote in rows:
            by_finding.setdefault(finding_id, set()).add((chunk_id, " ".join(quote.split())))
        return [frozenset(v) for v in by_finding.values()]

    # -- the checks, all in code -----------------------------------------------------------------------------------------------------
    def _verify(self, check: CrossRelation, refs: dict, chunks: dict, stats: CrossStats) -> None:
        if check.kind == "order":
            self._verify_order(check, refs, chunks, stats)
        else:
            self._verify_amount(check, refs, chunks, stats)

    def _verify_amount(self, check: CrossRelation, refs: dict, chunks: dict, stats: CrossStats) -> None:
        terms = [(refs.get(t.ref.strip()), t, refs.get(t.rate.strip()) if t.rate.strip() else None) for t in check.terms]
        against = refs.get(check.against.strip())
        rates_ok = all(rate is not None for _r, t, rate in terms if t.rate.strip())
        if not terms or against is None or any(r is None for r, _t, _x in terms) or not rates_ok:
            stats.unusable += 1
            return
        if any(MEMO.search(f"{r[0].claim_text or ''} {r[0].subject or ''} {r[0].metric or ''}") for r, _t, _x in terms):
            stats.unusable += 1  # a memo or interim row is set apart from the total it sits beside: it is not one of its parts
            return
        members = [r for r, _t, _x in terms] + [x for _r, _t, x in terms if x is not None] + [against]
        if len({m[0].id for m in members}) != len(members) or any(t.times <= 0 for _r, t, _x in terms):
            stats.unusable += 1  # a claim twice, or a count that is not positive
            return
        low = high = point = Decimal(0)  # the interval decides; the point (each figure as written) is what is shown
        parts: list[tuple[Decimal, Decimal]] = []  # each term's own interval, to find a row that is not a part of the total
        for r, t, rate in terms:
            amount = _amount(r[0])
            factor = _rate(rate[0]) if rate is not None else Num("exact", Decimal(1), Decimal(1))
            if amount is None or factor is None or amount.kind == "range":  # a span ("1 to 3") is not an amount
                stats.unusable += 1
                return
            lo, hi = _product((amount.low, amount.high), (factor.low, factor.high))
            count = Decimal(str(t.times))
            low, high = low + min(lo * count, hi * count), high + max(lo * count, hi * count)
            parts.append((min(lo * count, hi * count), max(lo * count, hi * count)))
            point += (amount.low + amount.high) / 2 * ((factor.low + factor.high) / 2) * count
        target = _amount(against[0])
        if target is None or target.kind == "range":
            stats.unusable += 1
            return
        stats.checked += 1
        slack = abs(target.high) * Decimal(str(TOLERANCE))
        broken = {"equals": high < target.low - slack or low > target.high + slack,
                  "at_most": low > target.high + slack, "at_least": high < target.low - slack}[check.relation]
        if not broken:
            stats.held += 1
            return
        if check.relation == "equals":
            odd = explained_without_a_term(parts, target.low, target.high)
            if odd is not None:
                stats.held += 1
                stats.explained += 1
                stats.log.append(f"{check.label}: holds exactly once '{terms[odd][0][0].claim_text or terms[odd][0][0].value}' is set aside, not reported")
                return
        stats.broken += 1
        computed = point
        stated = (target.low + target.high) / 2
        gap = abs(computed - stated) / abs(stated) if stated else Decimal(1)
        sides = " + ".join(f"{r[0].value}" + (f" x {x[0].value}" if x is not None else "") + (f" x {t.times:g}" if t.times != 1 else "") for r, t, x in terms)
        word = {"equals": "should equal", "at_most": "cannot exceed", "at_least": "cannot be less than"}[check.relation]
        where = sorted({m[1] for m in members})
        description = (f"{check.label}. Across {', '.join(where)}: {sides} = {computed:,.0f} once each claim's unit is applied, which {word} the {against[0].value} that "
                       f"{against[1]} states for it (a difference of {gap:.1%} of the stated figure). The relation was proposed from the documents' own wording ({check.reason or 'no reason given'}) "
                       f"and the arithmetic was done in code.")
        self._save(check, members, against, description, "amount", gap, stats)

    def _verify_order(self, check: CrossRelation, refs: dict, chunks: dict, stats: CrossStats) -> None:
        first, second = refs.get(check.earlier.strip()), refs.get(check.later.strip())
        if first is None or second is None or first[0].id == second[0].id:
            stats.unusable += 1
            return
        a, b = _span(first[0].value), _span(second[0].value)
        if a is None or b is None:
            stats.unusable += 1  # a relative or ambiguous date is not ordered here
            return
        stats.checked += 1
        if not a[0] > b[1]:  # the earlier date is on or before the later one (or the periods overlap)
            stats.held += 1
            return
        stats.broken += 1
        description = (f"{check.label}. {first[1]} gives {first[0].value} for {first[0].metric or first[0].subject}, which comes after {second[0].value}, the date "
                       f"{second[1]} gives for {second[0].metric or second[0].subject}, although the first must not come after the second "
                       f"({check.reason or 'no reason given'}). The dates were compared in code.")
        self._save(check, [first, second], first, description, "order", Decimal(0), stats)

    def _save(self, check: CrossRelation, members: list, anchor, description: str, kind: str, gap: Decimal, stats: CrossStats) -> None:
        key = frozenset(m[0].id for m in members)
        if key in self._seen:
            return
        self._seen.add(key)
        quotes, seen = [], set()
        for m in members:
            chunk = self.session.get(DocumentChunk, m[0].chunk_id)
            q = clause_quote(chunk.content, m[0].value) if chunk is not None else None
            if q and " ".join(q.split()) not in seen:
                seen.add(" ".join(q.split()))
                quotes.append((m, q))
        if len(quotes) < 2:
            stats.rejected += 1
            stats.log.append(f"{check.label}: broken, but the figures are not in two different passages")
            return
        passages = frozenset((m[0].chunk_id, " ".join(q.split())) for m, q in quotes)
        if any(passages <= held or held <= passages for held in self._reported):
            stats.already_reported += 1
            stats.log.append(f"{check.label}: broken, and the same passages are already in a finding")
            return
        # never critical: the relation is the model's proposal, only the arithmetic is certain
        severity = "high" if kind == "order" or gap >= Decimal("0.01") else "medium"
        result = self.registry.execute("submit_finding", {
            "title": f"Cross-document check: {check.label}"[:300], "description": description, "classification": "contradiction", "severity": severity,
            "confidence": 0.7, "evidence_status": "adequate", "materiality_reasoning": check.reason or None,
            "evidence": [{"chunk_id": m[0].chunk_id, "quoted_text": q, "role": "contradicting", "relevance_reason": m[0].claim_text or check.label,
                          "support_assessment": f"{m[1]} states {m[0].value} ({m[2]})"} for m, q in quotes[:12]]})
        if not result.ok:
            stats.rejected += 1
            stats.log.append(f"{check.label}: {result.error.code}: {result.error.message[:140]}")
            return
        self._reported.append(passages)
        stats.saved += 1
        stats.log.append(f"{check.label} ({severity}): {description[:200]}")
