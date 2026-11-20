"""Code guards: a fragment of another claim's value, a sum met exactly once a memo row is set aside, a value in cents."""
from __future__ import annotations

from decimal import Decimal

from app.models import Claim
from app.schemas.claim import Cluster
from app.services.claim_matching import normalize_unit
from app.services.group_gates import apply_gates, is_fragment_of
from app.services.internal_checks import explained_without_a_term
from app.services.value_rules import compare_values
from tests.test_claim_group_judge import room  # noqa: F401  (fixture)

SENTENCE = "have not been sold, assigned or pledged to any other person"


def claim(value, **kw) -> Claim:
    return Claim(subject="Credits", metric="sold", value=value, claim_text=str(value), **kw)


def test_a_few_words_inside_a_longer_text_value_are_a_fragment():
    assert is_fragment_of(claim("not"), claim(SENTENCE))
    assert is_fragment_of(claim("No person"), claim("No person shall be admitted as a member except as provided in this Agreement."))
    assert is_fragment_of(claim("have not been"), claim(SENTENCE))


def test_numbers_dates_equal_or_unrelated_values_are_never_fragments():
    assert not is_fragment_of(claim("85"), claim("the price is 85 cents each, as agreed"))
    assert not is_fragment_of(claim(85), claim(SENTENCE))
    assert not is_fragment_of(claim("March 28"), claim("placed in service on March 28 with the Phase 1 TCO"))
    assert not is_fragment_of(claim("annual"), claim("annual"))  # the same value is agreement, not a cut
    assert not is_fragment_of(claim("not"), claim("not sold"))  # the longer value must hold at least two more words
    assert not is_fragment_of(claim("sold"), claim("assigned or pledged to any other person"))  # not inside it
    assert not is_fragment_of(claim("one two three four"), claim("one two three four five six"))  # longer than a fragment can be


def test_the_gate_leaves_a_fragment_out_of_its_cluster_and_keeps_the_others(session, room):  # noqa: F811
    a = claim(SENTENCE, document_id=room.deck.id, chunk_id=room.deck_p1)
    b = claim("not", document_id=room.capex.id, chunk_id=room.capex_current)
    c = claim(SENTENCE, document_id=room.lease.id, chunk_id=room.rent_1)
    session.add_all([a, b, c])
    session.commit()
    result = apply_gates(session, [Cluster(claim_ids=[a.id, b.id, c.id])], None, list_guard=False)
    assert result.fragment_dropped == 1 and len(result.clusters) == 1 and set(result.clusters[0].claim_ids) == {a.id, c.id}
    assert any("fragment" in line for line in result.log)


def test_a_cluster_that_falls_below_two_claims_after_the_fragment_guard_is_dropped(session, room):  # noqa: F811
    a = claim(SENTENCE, document_id=room.deck.id, chunk_id=room.deck_p1)
    b = claim("not", document_id=room.capex.id, chunk_id=room.capex_current)
    session.add_all([a, b])
    session.commit()
    result = apply_gates(session, [Cluster(claim_ids=[a.id, b.id])], None, list_guard=False)
    assert result.clusters == [] and result.clusters_dropped == 1 and result.fragment_dropped == 1


def parts(*values):
    return [(Decimal(v), Decimal(v)) for v in values]


def test_a_sum_is_explained_when_one_listed_figure_set_aside_makes_it_equal():
    # sources 18.5 + 4.3 + 5.0 + 2.7 + 0.6 + 1.7 against a stated total of 31.1 (the 1.7 row is a bridge, not a source)
    assert explained_without_a_term(parts("18.5", "4.3", "5.0", "2.7", "0.6", "1.7"), Decimal("31.1"), Decimal("31.1")) == 5


def test_a_sum_that_no_single_removal_explains_stays_broken():
    assert explained_without_a_term(parts("865920", "1298880", "1831840", "432960"), Decimal("4329600"), Decimal("4329600")) is None  # the installments
    assert explained_without_a_term(parts("10", "20", "5"), Decimal("29"), Decimal("29")) is None
    assert explained_without_a_term(parts("10", "20"), Decimal("10"), Decimal("10")) is None  # two terms: none would be left to add up


def test_a_total_larger_than_its_parts_is_not_explained_by_removing_one():
    assert explained_without_a_term(parts("1320000", "860000", "1950000"), Decimal("22540000"), Decimal("22540000")) is None  # the hard cost subtotal


def test_cents_are_a_hundredth_of_the_currency():
    for a, b, expected in [("85 cents", "$0.85", "equal"), ("85¢", "0.85", "equal"), ("$0.85", "85 cents", "equal"), ("85 cents", "$0.82", "different")]:
        assert compare_values(a, None, b, None, None, None, normalize_unit) == expected
