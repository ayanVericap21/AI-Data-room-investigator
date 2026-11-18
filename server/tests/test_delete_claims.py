"""Claims are rebuilt at the start of a run; the findings of an earlier run keep their evidence, which only loses its link to the deleted claim."""
from __future__ import annotations

from sqlalchemy import select

from app.models import Claim, Evidence, Finding, FindingClaim
from app.services.claim_service import delete_claims
from tests.test_tools import finding_args, registry, room  # noqa: F401  (fixtures and helper)


def claims_and_a_finding(session, registry, room):  # noqa: F811
    deck = Claim(document_id=room.deck.id, chunk_id=room.deck_p1, subject="plant", metric="capacity", value="120,000", claim_text="capacity is 120,000 tonnes per year")
    capex = Claim(document_id=room.capex.id, chunk_id=room.capex_current, subject="plant", metric="capacity", value="100,000", claim_text="Capacity | 100,000 tonnes")
    session.add_all([deck, capex])
    session.commit()
    assert registry.execute("submit_finding", finding_args(room)).ok
    assert {e.claim_id for e in session.scalars(select(Evidence))} == {deck.id, capex.id}  # the evidence cites the claims it quotes
    return deck, capex


def test_deleting_every_claim_keeps_the_finding_and_its_quotes_and_unlinks_the_evidence(session, registry, room):  # noqa: F811
    claims_and_a_finding(session, registry, room)
    assert delete_claims(session) == 2
    session.commit()
    assert session.scalars(select(Claim)).all() == []
    finding = session.scalars(select(Finding)).one()
    evidence = session.scalars(select(Evidence)).all()
    assert finding.title == "Plant capacity differs" and len(evidence) == 2
    assert all(e.claim_id is None and e.quoted_text for e in evidence)  # the quote and its chunk stay; only the claim link is gone


def test_deleting_one_documents_claims_unlinks_only_its_evidence(session, registry, room):  # noqa: F811
    deck, capex = claims_and_a_finding(session, registry, room)
    assert delete_claims(session, room.deck.id) == 1
    session.commit()
    assert [c.id for c in session.scalars(select(Claim))] == [capex.id]
    by_chunk = {e.chunk_id: e.claim_id for e in session.scalars(select(Evidence))}
    assert by_chunk[room.deck_p1] is None and by_chunk[room.capex_current] == capex.id


def test_deleting_claims_when_there_are_none_is_fine(session):
    assert delete_claims(session) == 0 and delete_claims(session, "doc_missing") == 0
