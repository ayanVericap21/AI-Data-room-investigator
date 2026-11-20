"""A claim's value as a typed, comparable value (``NormalizedValue``), made once right after the claims are loaded and kept in memory.

It gathers what the pipeline already knew in separate places (the value parser, the unit table, the "$ in thousands" scale of the unit or the sentence, the date
parser) into one reading, so that grouping, the judge's prompt and the sum checks all see the same figure. The raw value stays on the claim: quote checks compare
against the source wording. When a value cannot be read with confidence the answer is ``unknown``; nothing here guesses.
"""
from __future__ import annotations

import re
from decimal import Decimal

from app.schemas.claim import ClaimView, NormalizedValue
from app.services.claim_matching import normalize_unit
from app.services.internal_checks import CURRENCIES, _scale
from app.services.value_rules import AMBIGUOUS, _unit_words, normalise_identifier, normalise_text, parse_date, parse_numeric, unit_info

_SYMBOLS = {"$": "usd", "€": "eur", "£": "gbp", "₹": "inr", "¥": "jpy"}
_ONE = Decimal(1)
NUMERIC_KINDS = ("amount", "count", "percent", "measure")


def _currency(value: object, unit: str | None) -> str | None:
    text = f"{value} {unit or ''}".lower()
    for symbol, code in _SYMBOLS.items():
        if symbol in text:
            return code
    for word in re.findall(r"[a-z]+", text):
        if word in CURRENCIES:
            return word
    return None


def normalize_claim(view) -> NormalizedValue:
    """The typed value of a claim (a ``ClaimView`` or a ``Claim`` row: it reads ``value``, ``unit``, ``claim_text`` and the extracted type)."""
    value = getattr(view, "value", None)
    if value is None or isinstance(value, bool) or str(value).strip() == "":
        return NormalizedValue()
    kind = getattr(view, "value_type", None) or getattr(view, "claim_category", None)
    unit = getattr(view, "unit", None)
    if kind == "identifier":
        return NormalizedValue(kind="text", text_key=normalise_identifier(value))
    if kind != "name":
        span = parse_date(value, allow_bare_year=kind == "date")
        if span == AMBIGUOUS:
            return NormalizedValue()  # day and month cannot be told apart: unknown, not a guess
        if isinstance(span, tuple):
            return NormalizedValue(kind="date", start=span[0], end=span[1])
        ignore = _unit_words(unit) | CURRENCIES
        number = parse_numeric(value, ignore=ignore)
        inner = None
        if number is None:  # "84 apartments", "112,400 SF": a unit or a noun written after the number
            left: list[str] = []
            parse_numeric(value, ignore, left)
            if 0 < len(left) <= 2:
                number, inner = parse_numeric(value, ignore | set(left)), " ".join(left)
        if number is not None:
            factor = _scale(view)
            number = number.scaled(factor)
            low, high = number.low, number.high
            info = unit_info(unit or inner, normalize_unit)
            text = f"{value} {unit or ''}".lower()
            currency = _currency(value, unit)
            if "%" in text or "percent" in text:
                return NormalizedValue(kind="percent", low=low, high=high, unit="%", scale_applied=factor)
            if currency is not None or "cent" in text:
                return NormalizedValue(kind="amount", low=low, high=high, unit=currency, scale_applied=factor)
            if info.dimension is not None:
                return NormalizedValue(kind="measure", low=low, high=high, unit=info.dimension, scale_applied=factor)
            whole = low is not None and high is not None and (low + high) / 2 == (low + high) / 2 // 1
            if whole and (info.key is None or inner is not None):  # a whole number of things ("84 apartments", "86 units") is a count whatever the noun
                return NormalizedValue(kind="count", low=low, high=high, scale_applied=factor)
            return NormalizedValue(kind="measure", low=low, high=high, unit=info.key, scale_applied=factor)
    key = normalise_text(value)
    return NormalizedValue(kind="text", text_key=key) if key else NormalizedValue()


def compatible(a: NormalizedValue, b: NormalizedValue) -> bool:
    """Two values can be compared as numbers: the same kind of quantity, and the same unit where both name one (an unknown currency fits any currency)."""
    if a.kind != b.kind or a.kind not in NUMERIC_KINDS or a.low is None or b.low is None:
        return False
    return a.unit is None or b.unit is None or a.unit == b.unit


def relative_gap(a: NormalizedValue, b: NormalizedValue) -> Decimal | None:
    """How far apart two compatible numbers are, as a share of the larger one (0.02 = 2 %); 0 when their intervals overlap; None when they cannot be compared."""
    if not compatible(a, b) or a.high is None or b.high is None:
        return None
    if a.low <= b.high and b.low <= a.high:
        return Decimal(0)
    x, y = (a.low + a.high) / 2, (b.low + b.high) / 2
    top = max(abs(x), abs(y))
    return None if top == 0 else abs(x - y) / top


def describe(view, norm: NormalizedValue) -> str:
    """The normalized form for the judge's prompt, only when it adds something to the raw value ("24,600 ($ in thousands)" is 24,600,000 USD)."""
    if norm.kind in NUMERIC_KINDS and norm.low is not None and norm.high is not None:
        point = (norm.low + norm.high) / 2
        shown = f"{point:,.2f}".rstrip("0").rstrip(".")
        if re.sub(r"[^0-9.]", "", str(view.value)) == re.sub(r"[^0-9.]", "", shown) and norm.scale_applied == _ONE:
            return ""
        return f"{shown}{' ' + norm.unit if norm.unit and norm.unit != '%' else '%' if norm.unit == '%' else ''}"
    if norm.kind == "date" and norm.start is not None:
        return norm.start.isoformat() if norm.start == norm.end else f"{norm.start.isoformat()} to {norm.end.isoformat()}"
    return ""
