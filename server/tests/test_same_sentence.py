"""One sentence repeated on several pages of a document is one statement: two claims read from it are never compared with each other."""
from app.schemas.claim import ClaimView
from app.services.candidate_matching import comparable_subjects, one_sentence_in_one_document
from app.services.claim_grouping import ClaimGrouper

SENTENCE = "The Managing Member shall deliver quarterly financial statements and annual audited statements."


def claim(id, doc, chunk, value, text=SENTENCE, metric="delivery frequency"):
    return ClaimView(id=id, document_id=doc, chunk_id=chunk, subject="Managing Member", metric=metric, value=value, claim_text=text, value_type="statement")


def test_the_same_sentence_on_two_pages_of_one_document_is_one_statement_whatever_the_page():
    quarterly, annual = claim("a", "d1", "page4", "quarterly"), claim("b", "d1", "page15", "annual")
    assert one_sentence_in_one_document(quarterly, annual) and not comparable_subjects(quarterly, annual)
    found = {}
    ClaimGrouper._add_candidate(found, quarterly, annual, how="llm")
    assert found == {}  # no candidate is made, by any route


def test_the_same_sentence_in_one_chunk_is_still_one_statement():
    assert one_sentence_in_one_document(claim("a", "d1", "c1", "quarterly"), claim("b", "d1", "c1", "annual"))


def test_the_same_sentence_in_two_documents_stays_comparable():
    signed, draft = claim("a", "d1", "c1", "quarterly"), claim("b", "d2", "c2", "annual")
    assert not one_sentence_in_one_document(signed, draft) and comparable_subjects(signed, draft)


def test_a_text_that_does_not_state_both_values_is_not_one_sentence_even_in_one_document():
    # a bare label repeated in two tables of one document: its two values are not both in it, so they are two statements
    one, two = claim("a", "d1", "c1", "1,180,000", text="Total"), claim("b", "d1", "c2", "640,000", text="Total")
    assert not one_sentence_in_one_document(one, two) and comparable_subjects(one, two)


def test_different_sentences_in_one_document_stay_comparable():
    a = claim("a", "d1", "c1", "quarterly")
    b = claim("b", "d1", "c2", "annual", text="Statements are delivered annually under Section 9.")
    assert not one_sentence_in_one_document(a, b) and comparable_subjects(a, b)


def test_two_claims_with_one_value_from_the_same_repeated_sentence_are_one_statement_too():
    assert one_sentence_in_one_document(claim("a", "d1", "page4", "quarterly"), claim("b", "d1", "page19", "quarterly"))
