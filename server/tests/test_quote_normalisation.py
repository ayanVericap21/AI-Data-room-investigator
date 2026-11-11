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


def test_other_differences_still_fail_and_exact_matches_are_unchanged():
    assert find_source_span("capacity is 12,000 WSPM", "capacity is 12,001 WSPM") is None
    assert find_source_span("a “b” c", "a “b” c") == ("a “b” c", False)
    assert find_source_span("a b", "   ") is None
