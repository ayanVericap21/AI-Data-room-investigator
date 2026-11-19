"""Windows of restriction and the records dated inside them, found by code.

A claim that suspends, closes, prohibits or restricts something between two dates ("all operations are suspended from May 13 at 15:30 until May 15 at
07:30") is a rule with a time window. Any claim from elsewhere dated inside that window (a log row of the 14th) is a record that falls under it. The
grouping model is asked for the records under each rule, but a date comparison is exact and the model sometimes misses it, so the code's matching adds
them for the claims the model left alone: a window and its records are one rule cluster, judged by the same rule check.

Dates are read in ISO form (2026-05-14), as "May 13, 2026" and as "13 May 2026", with an optional time. A date without a year takes the year most
often written in its own document, then in the whole collection; without any year it is not used.
"""
from __future__ import annotations

import re
from collections import Counter
from datetime import datetime

from app.schemas.claim import ClaimView, Cluster

MONTHS = {m: i for i, names in enumerate(
    ("jan january", "feb february", "mar march", "apr april", "may", "jun june", "jul july", "aug august", "sep sept september", "oct october", "nov november",
     "dec december"), start=1) for m in names.split()}
_MON = r"(?P<mon>jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
_TIME = r"(?:\s*(?:at|,)?\s*(?P<h>\d{1,2}):(?P<mi>\d{2}))?"
ISO = re.compile(r"\b(?P<y>\d{4})-(?P<mo>\d{2})-(?P<d>\d{2})(?:[T ](?P<h>\d{1,2}):(?P<mi>\d{2}))?\b")
MONTH_DAY = re.compile(rf"\b{_MON}\.?\s+(?P<d>\d{{1,2}})(?:st|nd|rd|th)?(?:,?\s+(?P<y>\d{{4}}))?{_TIME}", re.IGNORECASE)
DAY_MONTH = re.compile(rf"\b(?P<d>\d{{1,2}})(?:st|nd|rd|th)?\s+{_MON}\.?(?:,?\s+(?P<y>\d{{4}}))?{_TIME}", re.IGNORECASE)
# generic words for what is stopped or forbidden for a while; a validity period ("valid from ... to ...") is not a restriction
RESTRICTION = re.compile(r"\b(suspend\w*|clos(?:ed|ure|ing)|prohibit\w*|forbid\w*|ban(?:ned)?|halt\w*|cancel\w*|shut\w*|restrict\w*|not\s+(?:permitted|allowed)|"
                         r"unavailable|embargo\w*|frozen|freeze|lockdown|moratorium|no\s+(?:work|access|entry))\b", re.IGNORECASE)
MAX_RECORDS = 14


def _blob(c: ClaimView) -> str:
    return " ".join(str(x) for x in (c.subject, c.metric, c.value, c.period, c.claim_text) if x)


def parse_dates(text: str, default_year: int | None) -> list[datetime]:
    """Every date the text writes, in order of appearance; a date without a year takes ``default_year`` (none: it is left out)."""
    found: list[tuple[int, datetime]] = []
    taken: list[tuple[int, int]] = []
    for pattern, iso in ((ISO, True), (MONTH_DAY, False), (DAY_MONTH, False)):
        for m in pattern.finditer(text):
            if any(a < m.end() and m.start() < b for a, b in taken):
                continue  # already read by another pattern
            try:
                year = int(m.group("y")) if m.group("y") else default_year
                month = int(m.group("mo")) if iso else MONTHS[m.group("mon").lower().rstrip(".")]
                if year is None:
                    continue
                hour, minute = (int(m.group("h")), int(m.group("mi"))) if m.group("h") else (0, 0)
                found.append((m.start(), datetime(year, month, int(m.group("d")), hour, minute)))
                taken.append((m.start(), m.end()))
            except (ValueError, KeyError):
                continue
    return [d for _pos, d in sorted(found, key=lambda x: x[0])]


def _years(views: list[ClaimView]) -> tuple[dict[str, int], int | None]:
    """The year most often written in each document, and in the whole collection."""
    per_doc: dict[str, Counter] = {}
    total: Counter = Counter()
    for v in views:
        for pattern in (ISO, MONTH_DAY, DAY_MONTH):
            for m in pattern.finditer(_blob(v)):
                if m.group("y"):
                    per_doc.setdefault(v.document_id, Counter())[int(m.group("y"))] += 1
                    total[int(m.group("y"))] += 1
    return {doc: c.most_common(1)[0][0] for doc, c in per_doc.items()}, (total.most_common(1)[0][0] if total else None)


def windows_and_records(views: list[ClaimView]) -> list[Cluster]:
    """For each claim that restricts something between two dates: a rule cluster of that claim and each claim from another place dated inside that window."""
    doc_year, any_year = _years(views)
    dated = {v.id: parse_dates(_blob(v), doc_year.get(v.document_id, any_year)) for v in views}
    windows: dict[str, tuple[datetime, datetime]] = {}
    for v in views:
        dates = dated[v.id]
        if len(set(dates)) >= 2 and RESTRICTION.search(f"{v.metric or ''} {v.value or ''} {v.claim_text or ''}"):
            windows[v.id] = (min(dates), max(dates))
    found: list[Cluster] = []
    for window in views:
        if window.id not in windows:
            continue
        start, end = windows[window.id]
        records = [v for v in views if v.id not in windows and v.chunk_id != window.chunk_id and any(start <= d <= end for d in dated[v.id])]
        if records:
            found.append(Cluster(claim_ids=[window.id, *[r.id for r in records[:MAX_RECORDS]]], matched_by="rule", rule_id=window.id))
    return found
