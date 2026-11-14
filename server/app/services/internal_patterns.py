"""Relations that one sentence states and its own figures should satisfy, found by patterns in code (no model, no claims).

* a part out of a whole: "1,840 bays out of the 1,600 bay positions": a part cannot be larger than its whole;
* an average over a number of periods and a total: "average spend is 410,000 across March and April, giving a total of 900,000": 2 x 410,000 is not 900,000;
* a change stated against its before and after: "rose 2%, from 18.0% to 20.0%": 2 percentage points, 11.1 percent, and the sentence does not say which.

Each finding carries two quotes cut from the sentence exactly as written, so they verify like every other quote.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

from app.services.severity import PROMINENT

NUM = r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?"
MONTH = r"(?:January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sept|Sep|Oct|Nov|Dec)\.?"
WORD_NUMBERS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12}
MONTH_INDEX = {m: i for i, names in enumerate([("jan", "january"), ("feb", "february"), ("mar", "march"), ("apr", "april"), ("may",), ("jun", "june"), ("jul", "july"),
                                                 ("aug", "august"), ("sep", "sept", "september"), ("oct", "october"), ("nov", "november"), ("dec", "december")]) for m in names}
CURRENCY_MARK = re.compile(r"(GBP|EUR|USD|INR|AUD|CAD|CHF|JPY|£|€|\$)\s*$")
FIGURE = re.compile(rf"(?<![\w.,])(?P<n>{NUM})(?![\w,]|\.\d)")

OUT_OF = re.compile(rf"(?<![\w.,])(?P<a>{NUM})(?P<ua>\s*%)?[^.;:\n]{{0,40}}?\s(?P<out>out of)\s+(?:the\s+|a\s+total\s+of\s+)?(?P<b>{NUM})(?P<ub>\s*%)?(?![\w,])", re.IGNORECASE)
AVERAGE = re.compile(r"\b(?:average|mean)\b", re.IGNORECASE)
TOTAL = re.compile(r"\b(?:total|giving|sum|cumulative|altogether|overall)\b", re.IGNORECASE)
CHANGE = re.compile(rf"\b(?:rose|increased|grew|climbed|fell|decreased|declined|dropped|reduced)\b[^.;]{{0,40}}?(?:\bby\s+)?(?P<s>{NUM})\s*(?P<us>%|percent|per cent|pp\b|percentage points?)", re.IGNORECASE)
FROM_TO = re.compile(rf"\bfrom\s+(?P<a>{NUM})\s*(?P<ua>%|percent)?\s*to\s+(?P<b>{NUM})\s*(?P<ub>%|percent)?", re.IGNORECASE)


@dataclass(frozen=True)
class PatternFinding:
    kind: str  # "part_of_whole", "average_total", "change"
    classification: str  # contradiction or insufficient_evidence
    severity: str
    title: str
    description: str
    quotes: tuple[str, ...]  # exact passages of the chunk
    numbers: frozenset[str]  # the figures involved, digits only (to recognise the same relation found another way)


def _dec(text: str) -> Decimal | None:
    try:
        return Decimal(text.replace(",", ""))
    except InvalidOperation:
        return None


def sentences(text: str) -> list[str]:
    """The sentences and lines of a chunk, each exactly as written (no split inside "No. 3": a full stop followed by a capital starts a sentence)."""
    return [p.strip() for p in re.split(r"(?<=[.!?])\s+(?=[A-Z])|\n", text) if len(p.strip()) >= 8]


def period_count(text: str) -> int | None:
    """How many periods the words name: "across March and April" 2, "March to May" 3, "for 6 months" 6, "Q1 and Q2" 2, "2023 and 2024" 2."""
    m = re.search(rf"\b(\d+|{'|'.join(WORD_NUMBERS)})\s+(?:consecutive\s+)?(?:months?|weeks?|quarters?|years?|days?)\b", text, re.IGNORECASE)
    if m:
        return int(m.group(1)) if m.group(1).isdigit() else WORD_NUMBERS[m.group(1).lower()]
    months = [MONTH_INDEX[x.lower().rstrip(".")] for x in re.findall(rf"\b{MONTH}(?=\W|$)", text)]
    if months:
        return (months[1] - months[0]) % 12 + 1 if len(months) == 2 and re.search(r"\b(?:to|through|until|till)\b|[-–]", text) else len(set(months))
    quarters = set(re.findall(r"\bQ([1-4])\b", text))
    if len(quarters) >= 2:
        return len(quarters)
    years = set(re.findall(r"\b(19\d\d|20\d\d)\b", text))
    return len(years) if len(years) >= 2 else None


def _figure_after(text: str, start: int, end: int | None = None):
    for m in FIGURE.finditer(text, start, end if end is not None else len(text)):
        if re.fullmatch(r"(?:19|20)\d\d", m.group("n")):
            continue  # a year
        if re.match(rf"\s*{MONTH}", text[m.end():m.end() + 12]) or re.search(rf"{MONTH}\s*$", text[max(0, m.start() - 12):m.start()]):
            continue  # a day of a month
        return m
    return None


def _close(x: Decimal, y: Decimal) -> bool:
    return abs(x - y) <= Decimal("0.5") + abs(y) * Decimal("0.01")


def _part_of_whole(sentence: str) -> PatternFinding | None:
    m = OUT_OF.search(sentence)
    if not m or m.group("ua") or m.group("ub"):
        return None
    a, b = _dec(m.group("a")), _dec(m.group("b"))
    if a is None or b is None or b <= 0 or a <= b:
        return None
    left, right = sentence[:m.start("out")].strip(" ,;:"), sentence[m.end("out"):].strip(" ,;:.")
    if len(left) < 8 or len(right) < 8:
        return None
    large = (a - b) / b >= Decimal("0.05")
    severity = ("critical" if PROMINENT.search(sentence) else "high") if large else "medium"
    return PatternFinding("part_of_whole", "contradiction", severity, f"A part larger than its whole: {m.group('a')} out of {m.group('b')}",
                          f"The document states {m.group('a')} out of {m.group('b')}: a part ({m.group('a')}) cannot be larger than the whole it is part of ({m.group('b')}). "
                          "Found by checking the sentence in code.", (left, right), frozenset({re.sub(r"[^\d.]", "", m.group("a")), re.sub(r"[^\d.]", "", m.group("b"))}))


def _average_total(sentence: str) -> PatternFinding | None:
    avg = AVERAGE.search(sentence)
    total = TOTAL.search(sentence, avg.end()) if avg else None
    if not avg or not total:
        return None
    a, t = _figure_after(sentence, avg.end(), total.start()), _figure_after(sentence, total.end())
    if not a or not t:
        return None
    count = period_count(sentence[a.end():total.start()])
    marks = [CURRENCY_MARK.search(sentence[max(0, x.start() - 6):x.start()]) for x in (a, t)]
    if count is None or count < 2 or (all(marks) and marks[0].group(1) != marks[1].group(1)):
        return None
    value_a, value_t = _dec(a.group("n")), _dec(t.group("n"))
    if value_a is None or value_t is None or value_t <= 0:
        return None
    expected = value_a * count
    if abs(expected - value_t) <= max(value_t * Decimal("0.005"), Decimal("0.5")):
        return None
    left = sentence[:a.end()].strip(" ,;:")
    right = sentence[total.start():].strip(" ,;:.")
    if len(left) < 8 or len(right) < 8 or left == right:
        return None
    large = abs(expected - value_t) / value_t >= Decimal("0.05")
    return PatternFinding("average_total", "contradiction", ("critical" if PROMINENT.search(sentence) else "high") if large else "medium", f"The figures do not add up: {a.group('n')} x {count} periods against a total of {t.group('n')}",
                          f"The document states an average of {a.group('n')} over {count} periods and a total of {t.group('n')}, but {a.group('n')} x {count} = {expected:,.0f}, "
                          f"not {t.group('n')}. Found by checking the sentence in code.", (left, right),
                          frozenset({re.sub(r"[^\d.]", "", a.group("n")), re.sub(r"[^\d.]", "", t.group("n"))}))


def _change(sentence: str) -> PatternFinding | None:
    c, f = CHANGE.search(sentence), FROM_TO.search(sentence)
    if not c or not f:
        return None
    stated, a, b = _dec(c.group("s")), _dec(f.group("a")), _dec(f.group("b"))
    if stated is None or a is None or b is None or a <= 0:
        return None
    points, relative = abs(b - a), abs(b - a) / a * 100
    in_points = c.group("us").lower().startswith(("pp", "percentage"))
    both_percent = bool(f.group("ua")) and bool(f.group("ub"))
    left, right = sentence[:f.start()].strip(" ,;:"), sentence[f.start():].strip(" ,;:.")
    if len(left) < 8 or len(right) < 8:
        return None
    numbers = frozenset({re.sub(r"[^\d.]", "", x) for x in (c.group("s"), f.group("a"), f.group("b"))})
    if in_points:
        if _close(stated, points):
            return None
        reading = f"{points:g} percentage points"
    elif both_percent:
        if _close(stated, relative):
            return None
        if _close(stated, points):
            return PatternFinding("change", "insufficient_evidence", "low", f"Ambiguous change: '{c.group(0).strip()}' from {f.group('a')}% to {f.group('b')}%",
                                  f"The document states a change of {c.group('s')}% from {f.group('a')}% to {f.group('b')}%. That is {points:g} percentage points but {relative:.1f} percent, "
                                  "and the sentence does not say which it means. Found by checking the sentence in code.", (left, right), numbers)
        reading = f"{points:g} percentage points or {relative:.1f} percent"
    else:
        if _close(stated, relative):
            return None
        reading = f"{relative:.1f} percent"
    return PatternFinding("change", "contradiction", "medium", f"The change does not match its figures: {c.group('s')} against {f.group('a')} to {f.group('b')}",
                          f"The document states a change of {c.group('s')}{c.group('us')} from {f.group('a')} to {f.group('b')}, but those figures give {reading}. "
                          "Found by checking the sentence in code.", (left, right), numbers)


def find_relations(text: str) -> list[PatternFinding]:
    out = []
    for sentence in sentences(text):
        for check in (_part_of_whole, _average_total, _change):
            found = check(sentence)
            if found:
                out.append(found)
    return out
