"""A quote that differs from the source only by how text was parsed or copied still matches, and the stored span is the source's own."""
from app.tools.submit import find_source_span


def test_curly_quotes_dashes_nonbreaking_spaces_and_ligatures_do_not_stop_a_match():
    source = "The “qualified” capacity is 12,000 WSPM – as of 30 September; the oﬃce is open."
    assert find_source_span(source, "the office is open") is not None
    span, normalised = find_source_span(source, 'The "qualified" capacity is 12,000 WSPM - as of 30 September')
    assert normalised and span == "The “qualified” capacity is 12,000 WSPM – as of 30 September"


def test_a_word_hyphenated_across_a_line_break_and_zero_width_characters_match():
    source = "Commercial deliveries may begin only after written cus-\ntomer product qualification.​"
    span, normalised = find_source_span(source, "written customer product qualification.")
    assert normalised and span.startswith("written cus-\ntomer")


def test_table_pipes_and_row_numbers_do_not_stop_a_match_and_the_stored_span_is_the_source_row():
    source = "21 | Insurance, taxes, interest and permits | $1,000,000\n22 | Developer fee | $2,750,000\n23 | Financing costs and reserves | $700,000"
    span, normalised = find_source_span(source, "Developer fee $2,750,000")
    assert normalised and span == "Developer fee | $2,750,000"
    assert find_source_span(source, "22 Developer fee $2,750,000")[0] == "22 | Developer fee | $2,750,000"
    assert find_source_span(source, "Developer fee $2,750,001") is None  # the words and numbers still have to be the source's own
    assert find_source_span(source, "Developer fee $700,000") is None  # a value from another row is not stitched on


def test_other_differences_still_fail_and_exact_matches_are_unchanged():
    assert find_source_span("capacity is 12,000 WSPM", "capacity is 12,001 WSPM") is None
    assert find_source_span("a “b” c", "a “b” c") == ("a “b” c", False)
    assert find_source_span("a b", "   ") is None


def test_html_table_tags_do_not_stop_a_match_and_the_stored_span_is_the_source_text():
    source = "<table><tr><td>CO-1 Structural repairs</td><td>02/12/2024</td><td>$315,000</td></tr></table>"
    span, normalised = find_source_span(source, "CO-1 Structural repairs | 02/12/2024 | $315,000")
    assert normalised and span == "CO-1 Structural repairs</td><td>02/12/2024</td><td>$315,000"
    assert find_source_span(source, "Structural repairs 02/12/2024")[0] == "Structural repairs</td><td>02/12/2024"
    assert find_source_span(source, "<td>$315,000</td>")[0] == "<td>$315,000</td>"  # a quote that carries the tags still matches as written


def test_a_less_than_sign_that_is_not_a_tag_is_text():
    source = "Occupancy must stay at 5 <  10 units and the rest is >= 3"
    assert find_source_span(source, "must stay at 5 < 10 units") is not None
    assert find_source_span(source, "5 10 units") is None
