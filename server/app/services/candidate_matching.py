"""Candidate Pair Generation (deterministic).

Goal: avoid comparing every claim with every other claim by proposing pairs that look
like they are about the same thing (subject + metric overlap), then compare their
attributes. The output is a *candidate* with a relation hint; whether it is a real
contradiction, a contextual difference, etc. is decided later by evidence + LLM reasoning.

Design notes:
- Matching is lexical on normalised subject/metric tokens plus unit/period metadata, with an
  optional embedding route (gated by unit/period) to catch synonyms. Each pair records which
  route proposed it. Similarity is never used to *decide* a contradiction.
- A differing unit does not remove a pair (it may be a unit-conversion case); it is
  recorded in ``comparison.unit_match``.
"""
from __future__ import annotations

import math
import re
from collections import defaultdict

from app.schemas.claim import AttributeComparison, CandidatePair, ClaimView
from app.services.numbers import parse_number

STOPWORDS = {
    "the", "a", "an", "of", "for", "and", "or", "to", "in", "on", "at", "by", "per", "is", "are",
    "total", "overall", "expected", "estimated", "projected", "annual", "yearly", "annually",
}
UNIT_ALIASES = {
    "t": "tonne", "tonne": "tonne", "tonnes": "tonne", "ton": "tonne", "tons": "tonne", "mt": "tonne",
    "metric ton": "tonne", "metric tons": "tonne", "metric tonnes": "tonne",
    "usd": "usd", "$": "usd", "us$": "usd", "dollar": "usd", "dollars": "usd", "us dollars": "usd",
    "eur": "eur", "€": "eur", "euro": "eur", "euros": "eur",
    "%": "percent", "percent": "percent", "pct": "percent", "percentage": "percent",
    "kg": "kg", "kilogram": "kg", "kilograms": "kg",
}
PERIOD_ALIASES = {
    "annual": "annual", "annually": "annual", "yearly": "annual", "per year": "annual", "per annum": "annual",
    "pa": "annual", "p.a.": "annual", "/year": "annual", "year": "annual",
    "monthly": "monthly", "per month": "monthly", "month": "monthly",
    "quarterly": "quarterly", "per quarter": "quarterly",
}
DEFAULT_MIN_TOPIC_SCORE = 0.5


def tokens(text: str | None) -> frozenset[str]:
    if not text:
        return frozenset()
    out = set()
    for tok in re.findall(r"[a-z0-9]+", text.lower()):
        if tok in STOPWORDS:
            continue
        if len(tok) > 3 and tok.endswith("s"):
            tok = tok[:-1]
        out.add(tok)
    return frozenset(out)


def normalize_unit(unit: str | None) -> str | None:
    if not unit or not unit.strip():
        return None
    key = unit.strip().lower()
    return UNIT_ALIASES.get(key, key)


def normalize_period(period: str | None) -> str | None:
    if not period or not period.strip():
        return None
    key = re.sub(r"\s+", " ", period.strip().lower())
    return PERIOD_ALIASES.get(key, key)


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float | None:
    if not a or not b:
        return None
    return len(a & b) / len(a | b)


def topic_score(a: ClaimView, b: ClaimView) -> float:
    """0..1 relatedness from metric and subject overlap. Metric carries more weight; a
    missing metric/subject on either side falls back to the other, with a penalty."""
    metric = _jaccard(tokens(a.metric), tokens(b.metric))
    subject = _jaccard(tokens(a.subject), tokens(b.subject))
    if metric is not None and subject is not None:
        return 0.6 * metric + 0.4 * subject
    only = metric if metric is not None else subject
    return 0.0 if only is None else 0.7 * only


def _norm_text(s: str | None) -> str:
    return " ".join(sorted(tokens(s)))


def compare_attributes(a: ClaimView, b: ClaimView) -> AttributeComparison:
    ua, ub = normalize_unit(a.unit), normalize_unit(b.unit)
    pa, pb = normalize_period(a.period), normalize_period(b.period)
    na, nb = parse_number(a.value), parse_number(b.value)
    if a.value is None or b.value is None:
        relation = "unknown"
    elif na is not None and nb is not None:
        relation = "equal" if na == nb else "different"
    else:
        relation = "equal" if str(a.value).strip().lower() == str(b.value).strip().lower() else "different"

    def match(x, y):
        return None if x is None or y is None else x == y

    return AttributeComparison(
        subject_match=match(_norm_text(a.subject) or None, _norm_text(b.subject) or None),
        metric_match=match(_norm_text(a.metric) or None, _norm_text(b.metric) or None),
        unit_match=match(ua, ub),
        period_match=match(pa, pb),
        value_relation=relation,
        conditions_differ=_norm_text(a.conditions) != _norm_text(b.conditions),
        assumptions_differ=frozenset(map(_norm_text, a.assumptions)) != frozenset(map(_norm_text, b.assumptions)),
    )


def relation_hint(c: AttributeComparison) -> str:
    """exact | conflicting | contextual | partial. A hint for the investigator, never a verdict."""
    comparable = c.unit_match is not False and c.period_match is not False
    context_differs = c.conditions_differ or c.assumptions_differ or c.period_match is False
    if c.value_relation == "equal" and comparable and not context_differs:
        return "exact"
    if c.value_relation == "different":
        if context_differs:
            return "contextual"
        if comparable:
            return "conflicting"
    return "partial"


DEFAULT_MIN_EMBEDDING_SIMILARITY = 0.80


def topic_text(c: ClaimView) -> str:
    """Short text describing what a claim is about; embedded to catch synonyms."""
    parts = [("subject", c.subject), ("metric", c.metric), ("unit", c.unit), ("period", c.period)]
    return "; ".join(f"{k}: {v}" for k, v in parts if v)


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def generate_candidate_pairs(claims: list[ClaimView], min_topic_score: float = DEFAULT_MIN_TOPIC_SCORE,
                             max_partners_per_claim: int = 5, max_pairs: int | None = None,
                             vectors: dict[str, list[float]] | None = None,
                             min_embedding_similarity: float = DEFAULT_MIN_EMBEDDING_SIMILARITY) -> list[CandidatePair]:
    """Propose comparable pairs. Never pairs a claim with itself or with a claim from the
    same chunk.

    Two routes, both recorded in ``matched_by``:
    1. lexical - normalised subject/metric token overlap (an inverted index, so unrelated
       claims are never scored against each other);
    2. embedding (only when ``vectors`` is given) - catches synonyms such as "production
       capacity" vs "delivery volume". Embedding similarity alone is NOT trusted: it only
       proposes a pair when both units are known and equal and the periods do not conflict,
       because unrelated metrics also score fairly high (e.g. "lease end year" vs "monthly rent").
    """
    index: dict[str, set[int]] = defaultdict(set)
    toks = []
    for i, c in enumerate(claims):
        t = tokens(c.metric) | tokens(c.subject)
        toks.append(t)
        for token in t:
            index[token].add(i)

    def make_pair(a: ClaimView, b: ClaimView, score: float, comparison: AttributeComparison, how: str) -> CandidatePair:
        first, second = (a, b) if a.id <= b.id else (b, a)
        return CandidatePair(claim_a_id=first.id, claim_b_id=second.id, topic_score=round(score, 4),
                             relation_hint=relation_hint(comparison), comparison=comparison, matched_by=how)

    scored: dict[tuple[str, str], CandidatePair] = {}
    for i, a in enumerate(claims):
        neighbours: set[int] = set()
        for token in toks[i]:
            neighbours |= index[token]
        for j in neighbours:
            if j <= i:
                continue
            b = claims[j]
            if a.chunk_id == b.chunk_id:
                continue
            score = topic_score(a, b)
            if score < min_topic_score:
                continue
            scored[tuple(sorted((a.id, b.id)))] = make_pair(a, b, score, compare_attributes(a, b), "lexical")

    if vectors:
        for i, a in enumerate(claims):
            for b in claims[i + 1:]:
                key = tuple(sorted((a.id, b.id)))
                if key in scored or a.chunk_id == b.chunk_id or a.id not in vectors or b.id not in vectors:
                    continue
                comparison = compare_attributes(a, b)
                if comparison.unit_match is not True or comparison.period_match is False:
                    continue
                similarity = cosine(vectors[a.id], vectors[b.id])
                if similarity >= min_embedding_similarity:
                    scored[key] = make_pair(a, b, similarity, comparison, "embedding")

    # Greedy selection, best pairs first: a pair is kept only while BOTH claims still have
    # room, so no claim ever exceeds max_partners_per_claim (hubs cannot crowd others out).
    ordered = sorted(scored.values(), key=lambda p: (-p.topic_score, p.claim_a_id, p.claim_b_id))
    partners: dict[str, int] = defaultdict(int)
    pairs: list[CandidatePair] = []
    for pair in ordered:
        if partners[pair.claim_a_id] >= max_partners_per_claim or partners[pair.claim_b_id] >= max_partners_per_claim:
            continue
        partners[pair.claim_a_id] += 1
        partners[pair.claim_b_id] += 1
        pairs.append(pair)
    return pairs[:max_pairs] if max_pairs else pairs
