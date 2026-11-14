"""Candidate generation (deterministic).

Goal: avoid comparing every claim with every other claim by proposing candidates that look
like they are about the same thing (subject + metric overlap), then compare their
attributes. The output is a *candidate* with a relation hint; whether it is a real
contradiction, a contextual difference, etc. is decided later by evidence + LLM reasoning.

Design notes:
- Matching is lexical on normalised subject/metric tokens plus unit/period metadata, with an
  optional embedding route (gated by unit/period) to catch synonyms. Each candidate records which
  route proposed it. Similarity is never used to *decide* a contradiction.
- A differing unit does not remove a candidate (it may be a unit-conversion case); it is
  recorded in ``comparison.unit_match``.
"""
from __future__ import annotations

import math
import re
from collections import defaultdict

from app.schemas.claim import AttributeComparison, Candidate, ClaimView
from app.services import value_rules
from app.services.claim_tags import plain
from app.services.identifiers import entity_ids

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


TABLE_TYPES = {"table", "table_row"}


def table_key(document_id: str, content_type: str | None, sheet_name: str | None, page_number: int | None) -> str | None:
    """Identifies the table a chunk belongs to (a sheet, or a table on a page); None for ordinary text."""
    if content_type not in TABLE_TYPES:
        return None
    return f"{document_id}|{sheet_name or ''}|{page_number or ''}"


def same_table(a: ClaimView, b: ClaimView) -> bool:
    """Rows of one table are different items that share column headings: comparing them proves nothing."""
    return a.table_key is not None and a.table_key == b.table_key


def same_statement(a: ClaimView, b: ClaimView) -> bool:
    """True when both claims restate exactly the same text (ignoring case, spacing and punctuation)."""
    def norm(c: ClaimView) -> str:
        return " ".join(re.findall(r"\w+", (c.claim_text or "").lower()))
    return bool(norm(a)) and norm(a) == norm(b)


def _norm_text(s: str | None) -> str:
    return " ".join(sorted(tokens(s)))


def compare_attributes(a: ClaimView, b: ClaimView) -> AttributeComparison:
    pa, pb = normalize_period(a.period), normalize_period(b.period)
    # the value rules (intervals, scale words, units of one dimension, dates, identifiers and names) decide equal / different / unknown
    relation = value_rules.compare_values(a.value, a.unit, b.value, b.unit, a.value_type, b.value_type, normalize_unit)

    def match(x, y):
        return None if x is None or y is None else x == y

    return AttributeComparison(
        subject_match=match(_norm_text(a.subject) or None, _norm_text(b.subject) or None),
        metric_match=match(_norm_text(a.metric) or None, _norm_text(b.metric) or None),
        unit_match=value_rules.units_match(value_rules.unit_info(a.unit, normalize_unit), value_rules.unit_info(b.unit, normalize_unit)),
        period_match=match(pa, pb),
        value_relation=relation,
        conditions_differ=_norm_text(a.conditions) != _norm_text(b.conditions),
        assumptions_differ=frozenset(map(_norm_text, plain(a.assumptions))) != frozenset(map(_norm_text, plain(b.assumptions))),
    )


def relation_hint(c: AttributeComparison) -> str:
    """exact | conflicting | contextual | partial. A hint for the investigator, never a verdict."""
    comparable = c.unit_match is not False and c.period_match is not False
    context_differs = c.conditions_differ or c.assumptions_differ or c.period_match is False
    if c.value_relation == "equal" and comparable:
        return "exact"  # the same value cannot contradict itself; free-text conditions that differ in wording do not matter
    if c.value_relation == "different":
        if context_differs:
            return "contextual"
        if comparable:
            return "conflicting"
    return "partial"


DEFAULT_MIN_EMBEDDING_SIMILARITY = 0.80
DEFAULT_MIN_SUBJECT_OVERLAP = 0.5


def subjects_overlap(subject_a: str | None, subject_b: str | None, min_overlap: float | None = DEFAULT_MIN_SUBJECT_OVERLAP) -> bool:
    """Two subject texts share enough words to be about the same thing; true when either names nothing."""
    if min_overlap is None:
        return True
    overlap = _jaccard(tokens(subject_a), tokens(subject_b))
    return overlap is None or overlap >= min_overlap


def subjects_compatible(a: ClaimView, b: ClaimView, min_overlap: float | None) -> bool:
    """A value is only comparable with another value about the same subject. When both claims name a
    subject, their words must overlap enough; if either has no subject, nothing is known, so the candidate stays."""
    if min_overlap is None:
        return True
    overlap = _jaccard(tokens(a.subject), tokens(b.subject))
    return overlap is None or overlap >= min_overlap


def identifiers_about_one_subject(a: ClaimView, b: ClaimView) -> bool:
    """Two different identifiers say nothing against each other unless they label the same thing (two tools, two rows, two orders
    each have their own). So a candidate with an identifier claim is kept only when both name the same subject (or one names none)."""
    if "identifier" not in (a.value_type, b.value_type):
        return True
    return subjects_compatible(a, b, DEFAULT_MIN_SUBJECT_OVERLAP)


def _looks_like_key(subject: str | None) -> bool:
    """A row identifier such as MET-004 or PO-2026-101: one token, with a digit in it."""
    text = (subject or "").strip()
    return bool(text) and len(text) <= 40 and not any(ch.isspace() for ch in text) and any(ch.isdigit() for ch in text)


def table_rows_about_one_item(a: ClaimView, b: ClaimView) -> bool:
    """Two table rows keyed by different identifiers (MET-004 and MET-003) are different items, whatever their columns hold."""
    if a.table_key is None or b.table_key is None or not (_looks_like_key(a.subject) and _looks_like_key(b.subject)):
        return True
    return " ".join((a.subject or "").lower().split()) == " ".join((b.subject or "").lower().split())


def different_records(a: ClaimView, b: ClaimView) -> bool:
    """Both subjects name records (RC-02, ART-118) and no record is named by both: two different samples, artifacts or rows, however alike their
    sentences read, so nothing in one says anything against the other."""
    ia, ib = entity_ids(a.subject), entity_ids(b.subject)
    return bool(ia) and bool(ib) and not (ia & ib)


def one_sentence_in_one_document(a: ClaimView, b: ClaimView) -> bool:
    """Two claims read from the same sentence (same chunk, same text) are parts of one statement, not two statements to compare."""
    return a.chunk_id == b.chunk_id and same_statement(a, b)


def comparable_subjects(a: ClaimView, b: ClaimView) -> bool:
    """The subject gates every candidate route applies: identifiers of different things, rows of different items, subjects that name different
    records, one sentence twice."""
    return (identifiers_about_one_subject(a, b) and table_rows_about_one_item(a, b) and not different_records(a, b)
            and not one_sentence_in_one_document(a, b))


def topic_text(c: ClaimView) -> str:
    """Short text describing what a claim is about; embedded to catch synonyms."""
    parts = [("subject", c.subject), ("metric", c.metric), ("unit", c.unit), ("period", c.period)]
    return "; ".join(f"{k}: {v}" for k, v in parts if v)


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def generate_candidates(claims: list[ClaimView], min_topic_score: float = DEFAULT_MIN_TOPIC_SCORE,
                             max_partners_per_claim: int = 5, max_candidates: int | None = None,
                             vectors: dict[str, list[float]] | None = None,
                             min_embedding_similarity: float = DEFAULT_MIN_EMBEDDING_SIMILARITY,
                             min_subject_overlap: float | None = DEFAULT_MIN_SUBJECT_OVERLAP) -> list[Candidate]:
    """Propose comparable claims: for each claim, the other claims it may be compared with. Never a claim with itself. Claims from the same chunk
    may be candidates of each other: a passage can contradict itself.

    Claims that share metric words must also have overlapping subjects (``min_subject_overlap``, None turns
    the check off), so a value is never compared with a value about a different subject just because the metric
    label is the same. The embedding route may still match claims whose subjects and metrics are worded differently.

    Two routes, both recorded in ``matched_by``:
    1. lexical - normalised subject/metric token overlap (an inverted index, so unrelated
       claims are never scored against each other);
    2. embedding (only when ``vectors`` is given) - catches synonyms such as "production
       capacity" vs "delivery volume". Embedding similarity alone is NOT trusted: it only
       proposes a candidate when both units are known and equal and the periods do not conflict,
       because unrelated metrics also score fairly high (e.g. "lease end year" vs "monthly rent").
    """
    index: dict[str, set[int]] = defaultdict(set)
    toks = []
    for i, c in enumerate(claims):
        t = tokens(c.metric) | tokens(c.subject)
        toks.append(t)
        for token in t:
            index[token].add(i)

    def make_candidate(a: ClaimView, b: ClaimView, score: float, comparison: AttributeComparison, how: str) -> Candidate:
        first, second = (a, b) if a.id <= b.id else (b, a)
        # the same statement twice cannot disagree: it is corroboration and is never sent to the judge
        hint = "exact" if same_statement(a, b) and comparison.value_relation != "different" else relation_hint(comparison)
        return Candidate(claim_id=first.id, candidate_id=second.id, topic_score=round(score, 4),
                             relation_hint=hint, comparison=comparison, matched_by=how)

    scored: dict[tuple[str, str], Candidate] = {}
    for i, a in enumerate(claims):
        neighbours: set[int] = set()
        for token in toks[i]:
            neighbours |= index[token]
        for j in neighbours:
            if j <= i:
                continue
            b = claims[j]
            if same_table(a, b) or not subjects_compatible(a, b, min_subject_overlap) or not comparable_subjects(a, b):
                continue
            score = topic_score(a, b)
            if score < min_topic_score:
                continue
            scored[tuple(sorted((a.id, b.id)))] = make_candidate(a, b, score, compare_attributes(a, b), "lexical")

    if vectors:
        for i, a in enumerate(claims):
            for b in claims[i + 1:]:
                key = tuple(sorted((a.id, b.id)))
                if key in scored or a.id not in vectors or b.id not in vectors or same_table(a, b) or not comparable_subjects(a, b):
                    continue
                if not subjects_compatible(a, b, min_subject_overlap) and _jaccard(tokens(a.metric), tokens(b.metric)):
                    continue  # same metric words, different subject: the lexical route already refused it
                comparison = compare_attributes(a, b)
                if comparison.unit_match is not True or comparison.period_match is False:
                    continue
                similarity = cosine(vectors[a.id], vectors[b.id])
                if similarity >= min_embedding_similarity:
                    scored[key] = make_candidate(a, b, similarity, comparison, "embedding")

    # Greedy selection, best candidates first: a candidate is kept only while BOTH claims still have
    # room, so no claim ever exceeds max_partners_per_claim (hubs cannot crowd others out).
    ordered = sorted(scored.values(), key=lambda p: (-p.topic_score, p.claim_id, p.candidate_id))
    partners: dict[str, int] = defaultdict(int)
    kept: list[Candidate] = []
    for candidate in ordered:
        if partners[candidate.claim_id] >= max_partners_per_claim or partners[candidate.candidate_id] >= max_partners_per_claim:
            continue
        partners[candidate.claim_id] += 1
        partners[candidate.candidate_id] += 1
        kept.append(candidate)
    return kept[:max_candidates] if max_candidates else kept
