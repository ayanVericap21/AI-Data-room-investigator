"""Severity by rule, so the model's mood does not set it.

A contradiction is critical when the gap is large (5% or more, or any identifier, name, yes/no, status or date that cannot both be true) AND the claim is
prominent (a total, a summary, a definition, an obligation or a signature); high when it is large OR prominent; low when the gap is under 1%; medium
otherwise. Missing information is medium, high when an obligation or another figure depends on the missing item.
"""
from __future__ import annotations

import re
from typing import Iterable

from app.services.claim_matching import normalize_unit
from app.services.value_rules import relative_gap

PROMINENT = re.compile(r"\b(total|summary|overall|contract value|liabilit\w*|cap(?:ped)?|shall|must|obligation\w*|payable|definition|signed|signature|warrant\w*|penalt\w*)\b", re.IGNORECASE)
OBLIGATION = re.compile(r"\b(shall|must|required?|requires|consent|approval|approved|permit|licen[cs]e|certificate|condition|obligat\w*|prior written)\b", re.IGNORECASE)
NON_NUMERIC = {"identifier", "name", "state", "date", "statement"}


def _text(c) -> str:
    return f"{c.subject or ''} {c.metric or ''} {c.claim_text or ''}"


def contradiction_severity(claims: Iterable, context: Iterable = ()) -> str:
    """The severity of a contradiction between these claims (objects with subject, metric, value, unit, claim_text, claim_category). ``context`` are the
    other claims of the group the finding came from (a prohibition the contradicted statement is about): they make it prominent but not larger."""
    claims = list(claims)
    context = list(context)
    worst: float | None = None
    for i, a in enumerate(claims):
        for b in claims[i + 1:]:
            gap = relative_gap(a.value, a.unit, b.value, b.unit, normalize_unit)
            if gap is not None:
                worst = gap if worst is None else max(worst, gap)
    non_numeric = worst is None and any(((getattr(c, "claim_category", None) or getattr(c, "value_type", None)) or "") in NON_NUMERIC for c in claims)  # a claim row or a matching view
    large = non_numeric or (worst is not None and worst >= 0.05)
    small = worst is not None and worst < 0.01
    prominent = any(PROMINENT.search(_text(c)) for c in claims + context)
    if large and prominent:
        return "critical"
    if large or prominent:
        return "high"
    return "low" if small else "medium"


def missing_severity(*texts: str) -> str:
    """Medium; high when an obligation, a consent, an approval or a condition depends on the missing item."""
    return "high" if any(OBLIGATION.search(t or "") for t in texts) else "medium"
