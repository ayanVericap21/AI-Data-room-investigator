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
    if number is None and isinstance(value, str):
        wanted = numbers_in_text(value)
        if len(wanted) >= 2:  # a range or list such as "395 to 7,916": every number in it must appear in the text
            have = set(numbers_in_text(text))
            return all(n in have for n in wanted)
    if number is not None:  # the whole value as a standalone number: "26" is not found inside "2026-03-14" or "1264"
        raw = str(value).strip()
        return any(n == number for n in numbers_in_text(text)) or bool(re.search(rf"(?<![\w.,]){re.escape(raw)}(?![\w])", text, re.IGNORECASE))
    wanted = str(value).strip().lower()
    if wanted in text.lower():
        return True
    if isinstance(value, str):  # the same numbers and the same words, whatever the separators and spacing
        numbers = numbers_in_text(value)
        words = re.findall(r"[a-z]+", wanted)
        if numbers and set(numbers) <= set(numbers_in_text(text)) and all(w in text.lower() for w in words):
            return True
    return False


def numbers_with_context(text: str, width: int = 90) -> list[tuple[Decimal, str]]:
    """Each distinct number in ``text`` with the stretch of text around its first occurrence."""
    seen: dict[Decimal, str] = {}
    for match in _NUMBER_IN_TEXT.finditer(text):
        try:
            value = Decimal(match.group(0).replace(",", ""))
        except InvalidOperation:
            continue
        if value not in seen:
            start, end = max(0, match.start() - width), min(len(text), match.end() + width)
            seen[value] = " ".join(text[start:end].split())
    return list(seen.items())


_RANGE_SEPARATOR = re.compile(r"-|–|—|~|\bto\b|\band\b", re.IGNORECASE)


def parse_range(value: object) -> tuple[Decimal, Decimal] | None:
    """A value such as "395-7,916", "395 to 7,916" or "395m2 – 7,916m2" as (low, high); None if it is not a range.
    Exactly two numbers with a range word or dash between them; numbers inside units (the 2 of m2) do not count."""
    if not isinstance(value, str):
        return None
    matches = list(_NUMBER_IN_TEXT.finditer(value))
    if len(matches) != 2:
        return None
    between = value[matches[0].end():matches[1].start()]
    if not _RANGE_SEPARATOR.search(between):
        return None
    try:
        first, second = (Decimal(m.group(0).replace(",", "")) for m in matches)
    except InvalidOperation:
        return None
    return (min(first, second), max(first, second))
