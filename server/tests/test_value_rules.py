"""Deterministic comparison rules by value type: numbers (precision, scale, units, bounds, ranges, sign), dates, identifiers and names."""
from decimal import Decimal

import pytest

from app.services.candidate_matching import compare_attributes, normalize_unit, relation_hint
from app.services.value_rules import compare_values, comparison_note, normalise_text, parse_date, relative_gap, unit_info, units_match
from tests.test_claims import view


def rel(a, b, ua=None, ub=None, ta=None, tb=None):
    return compare_values(a, ua, b, ub, ta, tb, normalize_unit)


# ---- numerical ----------------------------------------------------------------
@pytest.mark.parametrize("a, b, ua, ub, expected", [
    (412, 388, None, None, "different"),
    ("10.2", "10,213,000", "million", None, "equal"),          # scale word in the unit; precision of the shorter number
    ("about 10,000", "9,640", None, None, "equal"),             # approximately: plus or minus 10 %
    ("more than 500", 612, None, None, "equal"),                 # a bound that the other value satisfies
    ("up to 20%", 12, None, None, "equal"),
    ("more than 500", 400, None, None, "different"),
    ("1.5 t", "1,500 kg", None, None, "equal"),                  # units of one dimension convert
    ("185", "1,850,000,000", "INR crore", "INR", "equal"),
    ("24,000", "24000", "WSPM", "WSPM", "equal"),
    ("500", "512", None, None, "different"),                     # trailing zeros are precision: no tolerance
    ("28", "29", None, None, "different"),
    ("10.2", "10.21", None, None, "equal"),                      # a rounded value agrees with a more precise one
    ("(1,200)", "-1,200", None, None, "equal"),                  # a bracketed amount is negative
    ("(1,200)", "1,200", None, None, "different"),
    ("395-7,916", "395-7,916", None, None, "equal"),
    ("395-7,916", "395-8,000", None, None, "different"),
    ("395-7,916", 395, None, None, "different"),                 # a range is never a single number
    ("more than 500", "more than 400", None, None, "unknown"),   # two different bounds are different statements
    (None, 5, None, None, "unknown"),
])
def test_numbers(a, b, ua, ub, expected):
    assert rel(a, b, ua, ub) == expected


def test_units_convert_only_within_one_dimension():
    assert units_match(unit_info("kg", normalize_unit), unit_info("tonnes", normalize_unit)) is True
    assert units_match(unit_info("usd", normalize_unit), unit_info("eur", normalize_unit)) is False
    assert units_match(unit_info(None), unit_info("kg")) is None
    assert rel(30, 30000, "t", "kg") == "equal" and rel(30, 30, "t", "kg") == "different"
    assert rel(5, 5, "usd", "eur") == "equal"  # numbers alone; the unit mismatch is reported by unit_match, as before


def test_gap_and_the_note_the_judge_sees():
    assert relative_gap(412, None, 388, None, normalize_unit) == Decimal(24) / Decimal(412)
    assert "differ (gap 5.8% of the larger value)" in comparison_note(412, None, 388, None)
    assert "agree within their stated precision" in comparison_note("10.2", "million", "10,213,000", None)
    assert "could not be compared" in comparison_note("a", None, None, None)


# ---- dates --------------------------------------------------------------------
@pytest.mark.parametrize("a, b, expected", [
    ("1 July 2024", "Q3 2024", "equal"),
    ("14 March 2023", "2023-04-14", "different"),
    ("14 March 2023", "March 14, 2023", "equal"),
    ("04/05/2024", "4 May 2024", "unknown"),                      # day and month order cannot be told apart
    ("25/05/2024", "2024-05-25", "equal"),                        # 25 can only be a day
    ("2024-03", "15 March 2024", "equal"),
    ("H1 2024", "2024-08-01", "different"),
    ("FY2024", "2024", "different"),                              # fiscal periods are not guessed: treated as text
])
def test_dates(a, b, expected):
    assert rel(a, b) == expected


def test_a_bare_year_is_a_date_only_when_both_claims_are_typed_date():
    assert rel("2026", "2027", ta="date", tb="date") == "different"
    assert rel("2015", "5 November 2026", ta="date", tb="date") == "different"
    assert parse_date("2024") is None and parse_date("2024", allow_bare_year=True)[0].year == 2024


def test_durations_are_numbers_with_units_that_convert():
    assert rel("36 months", "3 years") == "equal" and rel("36 months", "4 years") == "different"


# ---- text ---------------------------------------------------------------------
@pytest.mark.parametrize("a, b, expected", [
    ("SF-65-A", "sf 65 a", "equal"),                              # identifier: separators and case do not matter
    ("Certificate no. 4471-A", "Certificate no. 4417-A", "different"),
    ("0481-2217", "04812271", "different"),                       # one transposed digit is a different identifier
    ("+91 80 1234 5678", "918012345678", "equal"),
    ("ACME Ltd", "Acme Limited", "equal"),                        # legal-form words do not matter
    ("ACME Holdings Ltd", "ACME Ltd", "different"),
    ("ISO 9001:2015", "iso 9001:2015", "equal"),
])
def test_text(a, b, expected):
    assert rel(a, b) == expected


def test_normalise_text_removes_case_accents_spacing_punctuation_and_legal_forms():
    assert normalise_text("Société  Générale") == "societegenerale"
    assert normalise_text("ACME Ltd.") == "acme" == normalise_text("Acme, Limited")


# ---- wired into the candidate hints ---------------------------------------------------
def test_candidate_hints_use_the_rules():
    assert relation_hint(compare_attributes(view("a", value="10.2", unit="million"), view("b", doc="x", value="10,213,000", unit=None))) == "exact"
    assert relation_hint(compare_attributes(view("a", value=30, unit="t"), view("b", doc="x", value=30000, unit="kg"))) == "exact"
    assert relation_hint(compare_attributes(view("a", value=412, unit=None), view("b", doc="x", value=388, unit=None))) == "conflicting"


def test_a_table_heading_in_thousands_scales_the_numbers_beneath_it():
    for unit in ("$ in thousands", "thousands of dollars", "USD thousands", "$ thousands"):
        info = unit_info(unit, normalize_unit)
        assert (info.key, info.factor) == ("usd", Decimal(1000)), unit
    assert unit_info("in thousands", normalize_unit).key is None and unit_info("in thousands").factor == Decimal(1000)  # no unit left once the scale is out
    assert unit_info("in").key == "in" and unit_info("in").factor == Decimal(1)  # a lone "in" with no scale word stays a unit
    assert unit_info("$ in millions", normalize_unit).factor == Decimal(1_000_000) and unit_info("INR crores", normalize_unit).factor == Decimal(10) ** 7


def test_a_figure_in_thousands_agrees_with_the_same_figure_in_whole_dollars_and_only_with_it():
    assert rel("24,600", "$24,600,000", "$ in thousands", "USD") == "equal"
    assert comparison_note("24,600", "$ in thousands", "$24,600,000", "USD", None, None, normalize_unit) == "Code check: the values agree within their stated precision."
    assert rel("24,600", "$25,310,000", "$ in thousands", "USD") == "different"
    assert rel("24,600", "$24,600,000", "$ in millions", "USD") == "different"  # millions is not thousands
    assert rel("1,500", "1,500", "$ in thousands", "$ thousands") == "equal"  # two headings that both say thousands scale both numbers alike
    assert rel("1.5", "1,500", "$ in thousands", "USD") == "equal"
