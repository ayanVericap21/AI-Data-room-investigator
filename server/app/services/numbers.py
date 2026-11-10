"""Deterministic number helpers (Decimal; no floats for comparisons)."""
from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation

_NUMBER_IN_TEXT = re.compile(r"(?<![\w.])-?\d[\d,]*(?:\.\d+)?")


def parse_number(value: object) -> Decimal | None:
    """Parse 30000, 30000.0, "30,000", "$1,250.50", "12%" into a Decimal; else None."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return Decimal(str(value))
    if isinstance(value, str):
        cleaned = re.sub(r"[\s$€£%]|(?<=\d),(?=\d{3}\b)", "", value.strip())
        try:
            return Decimal(cleaned)
        except InvalidOperation:
            return None
    return None


def numbers_in_text(text: str) -> list[Decimal]:
    found = []
    for match in _NUMBER_IN_TEXT.findall(text):
        try:
            found.append(Decimal(match.replace(",", "")))
        except InvalidOperation:
            continue
    return found


def value_supported_by_text(value: object, text: str) -> bool:
    """True if a claimed value is traceable to the source text: a numeric value must equal a
    number appearing in the text; a string value must appear in it (case-insensitive).
    Non-values (None) are trivially supported."""
    if value is None:
        return True
    number = parse_number(value)
    if number is not None:
        return any(n == number for n in numbers_in_text(text)) or str(value).strip().lower() in text.lower()
    return str(value).strip().lower() in text.lower()
