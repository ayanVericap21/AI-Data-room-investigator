"""Record identifiers in text (SPL-044-B, RC-02, D-04, PO-1032): letters, a separator, digits, an optional suffix.

Two checks use them. Identifier reconciliation reads the identifiers a document names and compares them with another document's list of the same kind,
and must not count an identifier a line only gives as a limit ("up to 10 samples (RC-01 through RC-10)" authorises ten, it does not say RC-10 exists).
Grouping must not make candidates of two claims whose subjects name different records (RC-02 and RC-04 are different samples, however alike their sentences read).
"""
from __future__ import annotations

import re

TOKEN = r"[A-Za-z]{1,6}[-_]\d{2,6}(?:[-_][A-Za-z0-9]{1,4})?"
ID = re.compile(r"(?<![A-Za-z0-9])([A-Za-z]{1,6})([-_])(\d{2,6})(?:[-_]([A-Za-z0-9]{1,4}))?(?![A-Za-z0-9])")
NOT_AN_ID = {"table", "tab", "fig", "figure", "page", "pg", "section", "sec", "annex", "appendix", "version", "ver", "rev", "no", "ref", "item",
             "step", "ch", "chapter", "clause", "art", "para", "line", "row", "col", "iso", "pp", "vol", "eq", "ex"}

# A line that states a limit: the identifiers at the ends of a range it gives, or right after "up to" / "a maximum of", are quotas, not records.
LIMIT = re.compile(r"\b(?:up to|at most|no more than|a maximum of|maximum of|maximum|max\.?|ceiling|quota|not (?:to )?exceed|limited to|authori[sz]ed|allowance|capped)\b", re.IGNORECASE)
RANGE = re.compile(rf"({TOKEN})\s*(?:through|thru|to|–|—)\s*(?:a\s+maximum\s+of\s+|up\s+to\s+)?({TOKEN})", re.IGNORECASE)
AFTER_LIMIT = re.compile(rf"(?:up to|a maximum of|maximum of|at most|no more than|not to exceed)\s+({TOKEN})", re.IGNORECASE)


def is_record_id(prefix: str, digits: str) -> bool:
    """Not a section label (Table-12, Fig-10) and not a year (X-2025)."""
    return prefix.lower() not in NOT_AN_ID and not (len(digits) == 4 and 1900 <= int(digits) <= 2100)


def ceiling_idents(line: str) -> set[str]:
    """Identifiers (upper case, as written) that this line gives as a limit or the ends of an authorised range; empty when the line states no limit."""
    if not LIMIT.search(line or ""):
        return set()
    found: set[str] = set()
    for m in RANGE.finditer(line):
        found |= {m.group(1).upper(), m.group(2).upper()}
    for m in AFTER_LIMIT.finditer(line):
        found.add(m.group(1).upper())
    return found


def entity_labels(text: str | None) -> dict[tuple[str, int], str]:
    """The records a text names, each with the way the text writes it (RC-04, ART-118): for showing a record's name."""
    return {(m.group(1).upper(), int(m.group(3))): m.group(0) for m in ID.finditer(text or "") if is_record_id(m.group(1), m.group(3))}


def entity_ids(text: str | None) -> set[tuple[str, int]]:
    """The records a subject names, as (letters, number): the suffix is ignored, so RC-12 and RC-12-B count as the same record."""
    return {(m.group(1).upper(), int(m.group(3))) for m in ID.finditer(text or "") if is_record_id(m.group(1), m.group(3))}
