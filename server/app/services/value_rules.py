"""Deterministic rules for comparing two claim values, by what kind of value they are (Decimal, no floats).

By claim type (quantity, date, identifier, name, state, statement; a claim with no type is read by its value alone):
- quantity: a value is an interval from its written precision ("10.2" is 10.15 to 10.25); "about" widens it by 10 %,
  "more than" or "up to" makes it a bound, a range is compared end to end, scale words (thousand, million, crore ...) and
  units of one dimension (kg and tonne) are converted first, a bracketed value is negative.
- date: a date is an interval at its stated precision ("2024" is the whole year, "Q3 2024" the quarter); numeric day/month
  order that cannot be told apart is unknown, never guessed.
- identifier (codes, phone numbers): equal only when letters and digits are identical, whatever the separators and case.
- name: equal after case, accents, spacing, punctuation and legal-form words are removed.
- state, statement: equal when their normalised text is identical; anything that needs meaning is left to the judge.

Words such as "about" or "more than" and month names are English. Meaning (yes/no, statements, negation, anything that needs
context) is left to the judge; this module only answers equal, different or unknown.
"""
from __future__ import annotations

import calendar
import re
import unicodedata
from dataclasses import dataclass
from datetime import date
from decimal import Decimal, InvalidOperation
from typing import Callable

from app.services.numbers import parse_range

APPROX_BAND = Decimal("0.10")  # "about", "around", "approximately": plus or minus this share of the value

SCALE_WORDS = {"thousand": Decimal(10) ** 3, "million": Decimal(10) ** 6, "mn": Decimal(10) ** 6, "billion": Decimal(10) ** 9,
               "bn": Decimal(10) ** 9, "trillion": Decimal(10) ** 12, "lakh": Decimal(10) ** 5, "lac": Decimal(10) ** 5,
               "crore": Decimal(10) ** 7, "cr": Decimal(10) ** 7}

# unit -> (dimension, factor to the dimension's base unit). Only units of one dimension are ever converted into each other.
_D = Decimal
DIMENSIONS: dict[str, tuple[str, Decimal]] = {}
for _dim, _units in {
    "mass": {"mg": _D("0.001"), "g": _D(1), "gram": _D(1), "grams": _D(1), "kg": _D(1000), "kilogram": _D(1000), "kilograms": _D(1000),
             "tonne": _D(10) ** 6, "tonnes": _D(10) ** 6, "t": _D(10) ** 6, "ton": _D(10) ** 6, "tons": _D(10) ** 6, "mt": _D(10) ** 6,
             "lb": _D("453.59237")},
    "length": {"nm": _D("1e-9"), "mm": _D("0.001"), "cm": _D("0.01"), "m": _D(1), "metre": _D(1), "metres": _D(1), "meter": _D(1),
               "meters": _D(1), "km": _D(1000), "kilometre": _D(1000), "kilometer": _D(1000)},
    "area": {"m2": _D(1), "sqm": _D(1), "km2": _D(10) ** 6, "ha": _D(10) ** 4, "hectare": _D(10) ** 4, "hectares": _D(10) ** 4},
    "volume": {"ml": _D("0.001"), "l": _D(1), "litre": _D(1), "litres": _D(1), "liter": _D(1), "liters": _D(1), "kl": _D(1000), "m3": _D(1000)},
    "power": {"w": _D(1), "kw": _D(1000), "mw": _D(10) ** 6, "gw": _D(10) ** 9},
    "energy": {"wh": _D(1), "kwh": _D(1000), "mwh": _D(10) ** 6, "gwh": _D(10) ** 9},
    "months": {"month": _D(1), "months": _D(1), "quarter": _D(3), "quarters": _D(3), "year": _D(12), "years": _D(12), "yr": _D(12), "yrs": _D(12)},
    "days": {"day": _D(1), "days": _D(1), "week": _D(7), "weeks": _D(7)},
    "seconds": {"s": _D(1), "sec": _D(1), "second": _D(1), "seconds": _D(1), "min": _D(60), "minute": _D(60), "minutes": _D(60),
                "h": _D(3600), "hr": _D(3600), "hour": _D(3600), "hours": _D(3600)},
}.items():
    for _name, _factor in _units.items():
        DIMENSIONS[_name] = (_dim, _factor)

_NUMBER = re.compile(r"(?<![\w.])-?\d[\d,]*(?:\.\d+)?")
_CURRENCY_SYMBOLS = re.compile(r"[€$£₹¥%]")
_APPROX = re.compile(r"~|≈|\b(?:about|approx\.?|approximately|around|circa|roughly)\b|\bc\.(?=\s*\d)")
_LOWER_BOUND = re.compile(r">=?|≥|\b(?:more than|greater than|over|above|at least|minimum|min|exceeding|exceeds)\b")
_UPPER_BOUND = re.compile(r"<=?|≤|\b(?:less than|under|below|up to|at most|maximum|max|no more than|not more than|fewer than)\b")
_RANGE_WORDS = re.compile(r"\b(?:to|and|between|from)\b|[-–—~]")


# ---- units --------------------------------------------------------------------
@dataclass(frozen=True)
class UnitInfo:
    key: str | None  # the unit once scale words are taken out; None when no unit is given
    dimension: str | None
    factor: Decimal  # value * factor is in the dimension's base unit (or just scaled when the unit has no dimension)


def unit_info(unit: str | None, normalize: Callable[[str | None], str | None] | None = None) -> UnitInfo:
    if not unit or not unit.strip():
        return UnitInfo(None, None, Decimal(1))
    text = re.sub(r"_+", " ", unicodedata.normalize("NFKC", unit).lower())  # "INR_Crore" is "INR Crore"
    scale, rest = Decimal(1), []
    for token in text.split():
        word = token.strip(".,;()")
        if word in SCALE_WORDS and scale == 1:
            scale = SCALE_WORDS[word]
        else:
            rest.append(token)
    remaining = " ".join(rest).strip()
    if not remaining:
        return UnitInfo(None, None, scale)
    key = (normalize(remaining) if normalize else remaining) or remaining
    dimension = DIMENSIONS.get(key)
    if dimension is None:
        return UnitInfo(key, None, scale)
    return UnitInfo(key, dimension[0], scale * dimension[1])


def units_match(a: UnitInfo, b: UnitInfo) -> bool | None:
    """None when a unit is missing; True for the same unit or two units of one dimension (kg and tonne); else False."""
    if a.key is None or b.key is None:
        return None
    return a.key == b.key or (a.dimension is not None and a.dimension == b.dimension)


# ---- numbers ------------------------------------------------------------------
@dataclass(frozen=True)
class Num:
    kind: str  # exact | approx | bound | range
    low: Decimal | None  # None = unbounded
    high: Decimal | None
    unit: str | None = None  # a unit written inside the value ("1.5 t")
    ends: tuple[tuple[Decimal, Decimal], tuple[Decimal, Decimal]] | None = None  # a range: (low end interval, high end interval)

    def scaled(self, factor: Decimal) -> "Num":
        def f(x):
            return None if x is None else x * factor
        ends = None if self.ends is None else tuple((lo * factor, hi * factor) for lo, hi in self.ends)
        return Num(self.kind, f(self.low), f(self.high), self.unit, ends)


def _half_unit(text: str) -> Decimal:
    exponent = Decimal(text).as_tuple().exponent
    return Decimal(5) * Decimal(10) ** (exponent - 1)


def _clean(value: object) -> str:
    return unicodedata.normalize("NFKC", str(value)).strip().replace("−", "-")


def parse_numeric(value: object, ignore: frozenset[str] = frozenset(), leftovers: list[str] | None = None) -> Num | None:
    """The value as an interval, or None when it is not a plain number (a code, a name, a sentence). Words in ``ignore`` (the claim's own
    unit written inside the value, "12,000 WSPM") are allowed; with ``leftovers`` given, the other words found are collected there."""
    if value is None or isinstance(value, bool):
        return None
    text = _clean(value)
    negative = False
    wrapped = re.fullmatch(r"\(\s*([^()]*\d[^()]*?)\s*\)", text)  # a bracketed table amount is negative
    if wrapped:
        negative, text = True, wrapped.group(1)
    lower = text.lower()
    is_lower, is_upper, is_approx = bool(_LOWER_BOUND.search(lower)), bool(_UPPER_BOUND.search(lower)), bool(_APPROX.search(lower))
    kind = "bound" if (is_lower or is_upper) else "approx" if is_approx else "exact"
    upper = is_upper and not is_lower
    cleaned = _CURRENCY_SYMBOLS.sub(" ", _APPROX.sub(" ", _UPPER_BOUND.sub(" ", _LOWER_BOUND.sub(" ", lower))))
    numbers = list(_NUMBER.finditer(cleaned))
    if not numbers or len(numbers) > 2:
        return None
    residual = _NUMBER.sub(" ", cleaned)
    scale, unit = Decimal(1), None
    for token in re.findall(r"[a-z0-9²³./]+", _RANGE_WORDS.sub(" ", residual)):
        word = token.strip("./")
        if not word:
            continue
        if word in SCALE_WORDS:
            scale = SCALE_WORDS[word]
        elif unicodedata.normalize("NFKC", word) in DIMENSIONS and unit is None:
            unit = unicodedata.normalize("NFKC", word)
        elif word in ignore:
            continue
        elif leftovers is None:
            return None  # other words: not a plain number
        else:
            leftovers.append(word)
    if leftovers:
        return None
    sign = Decimal(-1) if negative else Decimal(1)
    try:
        if len(numbers) == 2:
            if kind != "exact" or parse_range(_CURRENCY_SYMBOLS.sub(" ", lower)) is None:  # a range carries no other qualifier
                return None
            ends = []
            for m in numbers:
                raw = m.group(0).replace(",", "")
                v, h = Decimal(raw) * scale, _half_unit(raw) * scale
                ends.append((v - h, v + h))
            return Num("range", min(e[0] for e in ends), max(e[1] for e in ends), unit, (ends[0], ends[1]) if ends[0][0] <= ends[1][0] else (ends[1], ends[0]))
        raw = numbers[0].group(0).replace(",", "")
        v, h = Decimal(raw) * scale * sign, _half_unit(raw) * scale
    except (InvalidOperation, ArithmeticError):
        return None
    if kind == "approx":
        band = max(abs(v) * APPROX_BAND, h)
        return Num("approx", v - band, v + band, unit)
    if kind == "bound":
        return Num("bound", None, v, unit) if upper else Num("bound", v, None, unit)
    return Num("exact", v - h, v + h, unit)


def _unit_words(unit: str | None) -> frozenset[str]:
    return frozenset(re.findall(r"[a-z0-9²³]+", unicodedata.normalize("NFKC", unit or "").lower()))


def parse_numeric_pair(a: object, unit_a: str | None, b: object, unit_b: str | None) -> tuple[Num | None, Num | None]:
    """Both values as intervals. A unit written inside a value is ignored when it is the claim's unit ("12,000" with unit WSPM and
    "12,000 WSPM"), and so are extra words that both values share ("12,000 wafers" and "12000 wafers")."""
    ignore = _unit_words(unit_a) | _unit_words(unit_b)
    na, nb = parse_numeric(a, ignore), parse_numeric(b, ignore)
    if na is None or nb is None:
        left_a, left_b = [], []
        parse_numeric(a, ignore, left_a)
        parse_numeric(b, ignore, left_b)
        if left_a and set(left_a) == set(left_b):
            na, nb = parse_numeric(a, ignore | set(left_a)), parse_numeric(b, ignore | set(left_b))
    return na, nb


def _overlap(a: tuple[Decimal | None, Decimal | None], b: tuple[Decimal | None, Decimal | None]) -> bool:
    (la, ha), (lb, hb) = a, b
    return (la is None or hb is None or la < hb) and (lb is None or ha is None or lb < ha)


def _compare_numbers(a: Num, b: Num) -> str:
    if a.kind == "range" or b.kind == "range":  # a range equals only a range with the same ends; never a single number
        if a.kind != b.kind:
            return "different"
        return "equal" if all(_overlap(x, y) for x, y in zip(a.ends, b.ends)) else "different"
    if a.kind == "bound" and b.kind == "bound":  # two bounds are the same statement only when they are the same bound
        return "equal" if (a.low, a.high) == (b.low, b.high) else "unknown"
    return "equal" if _overlap((a.low, a.high), (b.low, b.high)) else "different"


def _factor(a: UnitInfo, b: UnitInfo, a_inner: str | None, b_inner: str | None) -> tuple[Decimal, Decimal]:
    """Factors that put both values in one unit. Units that cannot be converted into each other are left as written."""
    if a.key is None and a_inner:
        a = unit_info(a_inner)
    if b.key is None and b_inner:
        b = unit_info(b_inner)
    if a.key is None and b.key is None:
        return a.factor, b.factor
    if a.key == b.key or (a.dimension is not None and a.dimension == b.dimension):
        return a.factor, b.factor
    return Decimal(1), Decimal(1)


# ---- dates --------------------------------------------------------------------
MONTHS = {name.lower(): i for i, name in enumerate(calendar.month_name) if name}
MONTHS.update({name.lower(): i for i, name in enumerate(calendar.month_abbr) if name})
MONTHS["sept"] = 9
AMBIGUOUS = "ambiguous"
_MONTH_WORDS = "|".join(sorted(MONTHS, key=len, reverse=True))


def _span(year: int, month: int | None = None, day: int | None = None) -> tuple[date, date] | None:
    try:
        if day is not None:
            d = date(year, month, day)
            return d, d
        if month is not None:
            return date(year, month, 1), date(year, month, calendar.monthrange(year, month)[1])
        return date(year, 1, 1), date(year, 12, 31)
    except ValueError:
        return None


def parse_date(value: object, allow_bare_year: bool = False) -> tuple[date, date] | str | None:
    """(first day, last day) the value covers; AMBIGUOUS when day and month cannot be told apart; None when it is not a date.
    Fiscal periods are not guessed (their year end is not known here)."""
    if value is None or isinstance(value, bool):
        return None
    text = _clean(value).lower()
    text = re.sub(r"(?<=\d)(?:st|nd|rd|th)\b", "", text)
    text = re.sub(r"[,]", " ", text).strip()
    text = re.sub(r"\s+", " ", text)
    if "fy" in text.split() or "fiscal" in text or re.search(r"\bfy\s*\d", text):
        return None
    if (m := re.fullmatch(r"(\d{4})-(\d{1,2})-(\d{1,2})(?:[t ].*)?", text)) or (m := re.fullmatch(r"(\d{4})/(\d{1,2})/(\d{1,2})", text)):
        return _span(int(m[1]), int(m[2]), int(m[3]))
    if m := re.fullmatch(r"(\d{4})-(\d{1,2})", text):
        return _span(int(m[1]), int(m[2]))
    if m := re.fullmatch(r"(\d{1,2})[/.\-](\d{1,2})[/.\-](\d{4}|\d{2})", text):
        a, b, y = int(m[1]), int(m[2]), int(m[3])
        y = y if len(m[3]) == 4 else (2000 + y if y < 70 else 1900 + y)
        if a > 12 and b <= 12:
            return _span(y, b, a)
        if b > 12 and a <= 12:
            return _span(y, a, b)
        if a == b:
            return _span(y, a, b)
        return AMBIGUOUS if a <= 12 and b <= 12 else None
    if m := re.fullmatch(rf"(\d{{1,2}})[ \-]({_MONTH_WORDS})\.?[ \-](\d{{4}})", text):
        return _span(int(m[3]), MONTHS[m[2]], int(m[1]))
    if m := re.fullmatch(rf"({_MONTH_WORDS})\.? (\d{{1,2}}) (\d{{4}})", text):
        return _span(int(m[3]), MONTHS[m[1]], int(m[2]))
    if m := re.fullmatch(rf"({_MONTH_WORDS})\.? (\d{{4}})", text):
        return _span(int(m[2]), MONTHS[m[1]])
    if m := re.fullmatch(r"q([1-4]) ?(\d{4})", text) or re.fullmatch(r"(\d{4}) ?q([1-4])", text):
        year, quarter = (int(m[2]), int(m[1])) if text.startswith("q") else (int(m[1]), int(m[2]))
        first, last = _span(year, 3 * quarter - 2)[0], _span(year, 3 * quarter)[1]
        return first, last
    if m := re.fullmatch(r"h([12]) ?(\d{4})", text) or re.fullmatch(r"(\d{4}) ?h([12])", text):
        year, half = (int(m[2]), int(m[1])) if text.startswith("h") else (int(m[1]), int(m[2]))
        return (_span(year, 1)[0], _span(year, 6)[1]) if half == 1 else (_span(year, 7)[0], _span(year, 12)[1])
    if allow_bare_year and (m := re.fullmatch(r"(?:19|20)\d{2}", text)):
        return _span(int(m[0]))
    return None


# ---- text ---------------------------------------------------------------------
LEGAL_FORMS = {"ltd", "limited", "pvt", "private", "inc", "incorporated", "llc", "llp", "plc", "corp", "corporation", "gmbh", "co", "company",
               "sa", "ag", "bv", "nv", "pte"}


def normalise_text(value: object) -> str:
    """Case, accents, spacing, punctuation and legal-form words removed; what is left is what identifies the thing."""
    text = unicodedata.normalize("NFKC", str(value))
    text = "".join(ch for ch in unicodedata.normalize("NFKD", text) if not unicodedata.combining(ch)).casefold()
    words = [w for w in re.findall(r"[^\W_]+", text) if w not in LEGAL_FORMS]
    return "".join(words) if words else re.sub(r"[\W_]+", "", text)


def name_words(value: object) -> frozenset[str]:
    """The words that identify a name or short phrase: case, accents, punctuation and legal-form words removed."""
    text = unicodedata.normalize("NFKC", str(value))
    text = "".join(ch for ch in unicodedata.normalize("NFKD", text) if not unicodedata.combining(ch)).casefold()
    return frozenset(w for w in re.findall(r"[^\W_]+", text) if w not in LEGAL_FORMS)


MAX_CONTAINED_WORDS = 8  # longer texts are statements: an extra clause there can change the meaning


def one_contains_other(a: object, b: object) -> bool:
    """Two different short texts where every word of one is in the other ("Technical Officer" and "Technical Officer (instrumentation)"):
    the same thing described in more or less detail. False for numbers and dates, for equal texts and for long texts."""
    if parse_numeric(a) is not None or parse_numeric(b) is not None:
        return False
    wa, wb = name_words(a), name_words(b)
    if not wa or not wb or wa == wb or max(len(wa), len(wb)) > MAX_CONTAINED_WORDS:
        return False
    return wa <= wb or wb <= wa


_UPPER = re.compile(r"\b(maximum|max|limit|limits|cap|capped|ceiling|at most|no more than|not exceed\w*|up to|permitted|allowed|allowable|quota|upper)\b", re.I)
_LOWER = re.compile(r"\b(minimum|min|at least|no less than|not less than|floor|lower)\b", re.I)


def bound_direction(text: str | None) -> str | None:
    """"upper" when the wording says the value is a ceiling (maximum, limit, at most, permitted), "lower" for a floor (minimum, at least), else None."""
    upper, lower = bool(_UPPER.search(text or "")), bool(_LOWER.search(text or ""))
    return "upper" if upper and not lower else "lower" if lower and not upper else None


def within_limit(limit: object, unit_limit: str | None, direction: str, other: object, unit_other: str | None,
                 normalize: Callable[[str | None], str | None] | None = None) -> bool | None:
    """Whether a plain number satisfies a limit (True/False); None when either value is not a plain number or the units cannot be reconciled."""
    nl, no = parse_numeric_pair(limit, unit_limit, other, unit_other)
    if nl is None or no is None or nl.kind != "exact" or no.kind != "exact":
        return None
    fl, fo = _factor(unit_info(unit_limit, normalize), unit_info(unit_other, normalize), nl.unit, no.unit)
    lim, val = nl.low * fl, no.low * fo
    return val <= lim if direction == "upper" else val >= lim


def normalise_identifier(value: object) -> str:
    """Letters and digits only, case folded: separators never make two identifiers different, and nothing else is ignored."""
    return re.sub(r"[\W_]+", "", unicodedata.normalize("NFKC", str(value)).casefold())


# ---- the comparison -----------------------------------------------------------
def compare_values(a: object, unit_a: str | None, b: object, unit_b: str | None, type_a: str | None = None, type_b: str | None = None,
                   normalize: Callable[[str | None], str | None] | None = None) -> str:
    """equal | different | unknown, by the rules for the kind of value."""
    if a is None or b is None or str(a).strip() == "" or str(b).strip() == "":
        return "unknown"
    if "identifier" in (type_a, type_b):  # character by character, never as a number or by similarity
        if type_a and type_b and type_a != type_b:  # an identifier set against a date, quantity or name has no common method
            return "different"
        return "equal" if normalise_identifier(a) == normalise_identifier(b) else "different"
    if "name" in (type_a, type_b):
        return "equal" if normalise_text(a) == normalise_text(b) else "different"
    both_dates = type_a == "date" and type_b == "date"  # only then is a bare year ("2024") a date rather than a number
    da, db = parse_date(a, allow_bare_year=both_dates), parse_date(b, allow_bare_year=both_dates)
    if da is not None and db is not None:
        if AMBIGUOUS in (da, db):
            return "unknown"
        return "equal" if da[0] <= db[1] and db[0] <= da[1] else "different"
    na, nb = parse_numeric_pair(a, unit_a, b, unit_b)
    if na is not None and nb is not None:
        fa, fb = _factor(unit_info(unit_a, normalize), unit_info(unit_b, normalize), na.unit, nb.unit)
        return _compare_numbers(na.scaled(fa), nb.scaled(fb))
    return "equal" if normalise_text(a) == normalise_text(b) else "different"


def relative_gap(a: object, unit_a: str | None, b: object, unit_b: str | None,
                 normalize: Callable[[str | None], str | None] | None = None) -> Decimal | None:
    """How far apart two plain numbers are, as a share of the larger one (0.05 = 5 %); None when code cannot say."""
    na, nb = parse_numeric_pair(a, unit_a, b, unit_b)
    if na is None or nb is None or "range" in (na.kind, nb.kind) or "bound" in (na.kind, nb.kind):
        return None
    fa, fb = _factor(unit_info(unit_a, normalize), unit_info(unit_b, normalize), na.unit, nb.unit)
    x, y = (na.low + na.high) / 2 * fa, (nb.low + nb.high) / 2 * fb
    top = max(abs(x), abs(y))
    return None if top == 0 else abs(x - y) / top


def comparison_note(a: object, unit_a: str | None, b: object, unit_b: str | None, type_a: str | None = None, type_b: str | None = None,
                    normalize: Callable[[str | None], str | None] | None = None) -> str:
    """One line for the judge: what code found when it compared the two values."""
    relation = compare_values(a, unit_a, b, unit_b, type_a, type_b, normalize)
    if relation == "unknown":
        return "Code check: the values could not be compared by code."
    if relation == "equal":
        return "Code check: the values agree within their stated precision."
    if one_contains_other(a, b):
        return "Code check: the values differ in wording only: every word of the shorter one is in the longer one, so one is the other with more or less detail."
    gap = relative_gap(a, unit_a, b, unit_b, normalize)
    return "Code check: the values differ" + (f" (gap {gap * 100:.1f}% of the larger value)." if gap is not None else ".")
