"""Document profiles (made by document length), the entity gate, the list-member guard, and the judge changes that go with them."""
from __future__ import annotations

import re

from sqlalchemy import select

from app.llm.errors import LLMPermanentError
from app.llm.mock import MockLLMClient
from app.models import Claim, DocumentChunk, Finding
from app.schemas.claim import AttributeComparison, Candidate
from app.services.claim_service import ExtractionSummary
from app.services.doc_profiles import DocProfile, ProfileBuilder, Unit, card, pack, render_cards, split_parts
from app.services.group_gates import EntityGate, apply_gates, decide_entity_groups
from app.services.claim_group_judge import ClaimGroupJudge, denies_contradiction
from app.tools import build_registry
from tests.test_tools import add_doc, room  # noqa: F401  (fixture and helper)

PROFILE = {"main_entity": "Acme Corp", "document_type": "investor deck", "purpose": "Presents the company to investors.", "date": "March 2025", "status": "final",
           "scope": "Company overview; no financial statements."}


def chunks_of(session, doc):
    return list(session.scalars(select(DocumentChunk).where(DocumentChunk.document_id == doc.id).order_by(DocumentChunk.chunk_index)))


# ---- the calls that describe the documents ----------------------------------------------------------------------------------------------------
def doc_chunks(session, name, sizes):
    """A document whose chunks have the given numbers of characters (each chunk starts with the document's name, so the prompt shows who it belongs to)."""
    doc, chunks = add_doc(session, name, "pdf", [dict(content=(f"{name} " + "x" * n)[:n], page_number=i) for i, n in enumerate(sizes, start=1)])
    session.commit()
    return doc.id, name, chunks


def described(*refs_and_entities):
    return {"documents": [{"ref": ref, "main_entity": entity, "document_type": "report", "purpose": f"About {entity}."} for ref, entity in refs_and_entities]}


def test_several_documents_share_one_call_when_they_fit(session):
    docs = [doc_chunks(session, "a.pdf", [3000]), doc_chunks(session, "b.pdf", [4000]), doc_chunks(session, "c.pdf", [2500])]
    llm = MockLLMClient([described(("d1", "Alpha"), ("d2", "Beta"), ("d3", "Gamma"))])
    builder = ProfileBuilder(llm, budget=20_000)
    profiles = builder.run(docs)
    assert len(llm.prompts) == 1 and len(builder.stats.calls) == 1 and builder.stats.calls[0].kind == "describe"
    assert "### d1 | a.pdf" in llm.prompts[0] and "### d2 | b.pdf" in llm.prompts[0] and "### d3 | c.pdf" in llm.prompts[0] and "[page 1] a.pdf" in llm.prompts[0]
    assert {name: p.main_entity for name, p in profiles.values()} == {"a.pdf": "Alpha", "b.pdf": "Beta", "c.pdf": "Gamma"} and all(r.ok for r in builder.stats.records)


def test_the_calls_are_filled_largest_first_and_never_exceed_the_budget():
    units = [Unit("1", "a", 1, 1, "x" * n) for n in (1000, 3000, 1500, 2500)]  # each also costs its header
    calls = pack(units, 6000)
    assert [sorted(len(u.text) for u in call) for call in calls] == [[2500, 3000], [1000, 1500]]
    assert all(sum(u.size for u in call) <= 6000 for call in calls) and len(pack(units, 100_000)) == 1 and len(pack(units, 3500)) == 3


def test_a_document_longer_than_a_call_is_cut_in_parts_and_one_extra_call_merges_the_parts(session):
    docs = [doc_chunks(session, "long.pdf", [3000, 3000, 3000]), doc_chunks(session, "short.pdf", [1000])]
    # budget 5000: long.pdf makes 3 parts of one chunk (3000 + overhead each), short.pdf fits beside a part (3120 + 1120 <= 5000)
    describe = lambda prompt: described(*[(ref, "Long Corp" if "long.pdf" in label else "Short Ltd") for ref, label in re.findall(r"### (d\d+) \| (\S+)", prompt)])
    merged = {"documents": [{"ref": "d1", "main_entity": "Long Corp", "document_type": "annual report", "purpose": "Merged purpose."}]}
    llm = MockLLMClient([describe, describe, describe, merged])
    builder = ProfileBuilder(llm, budget=5000)
    profiles = builder.run(docs)
    kinds = [c.kind for c in builder.stats.calls]
    assert kinds == ["describe", "describe", "describe", "merge"] and len(llm.prompts) == 4
    assert "(part 1 of 3)" in "".join(llm.prompts[:3]) and "Part 1:" in llm.prompts[3] and "Part 3:" in llm.prompts[3] and "short.pdf" not in llm.prompts[3]
    by_name = {name: p for name, p in profiles.values()}
    assert by_name["long.pdf"].purpose == "Merged purpose." and by_name["short.pdf"].main_entity == "Short Ltd"
    assert [r.parts for r in builder.stats.records] == [3, 1]


def test_a_failed_call_loses_only_its_documents_and_a_failed_merge_falls_back_to_the_first_part(session):
    docs = [doc_chunks(session, "a.pdf", [4000]), doc_chunks(session, "b.pdf", [4000])]
    builder = ProfileBuilder(MockLLMClient([LLMPermanentError("boom"), described(("d1", "Beta"))]), budget=5000)
    profiles = builder.run(docs)
    assert [name for name, _p in profiles.values()] == ["b.pdf"] or [name for name, _p in profiles.values()] == ["a.pdf"]
    assert sum(not c.ok for c in builder.stats.calls) == 1 and sum(not r.ok for r in builder.stats.records) == 1
    long = [doc_chunks(session, "l.pdf", [3000, 3000])]
    part = lambda prompt: described(("d1", "Long Corp"))
    fallback = ProfileBuilder(MockLLMClient([part, part, LLMPermanentError("boom")]), budget=4000).run(long)
    assert [p.main_entity for _n, p in fallback.values()] == ["Long Corp"]


def test_entries_for_refs_that_were_not_shown_are_ignored_and_a_document_without_text_is_noted(session):
    docs = [doc_chunks(session, "a.pdf", [500]), ("empty", "empty.pdf", [])]
    builder = ProfileBuilder(MockLLMClient([described(("d1", "Alpha"), ("d9", "Ghost"))]))
    profiles = builder.run(docs)
    assert [name for name, _p in profiles.values()] == ["a.pdf"] and builder.stats.records[1].note == "no readable text" and not builder.stats.records[1].ok


def test_parts_are_cut_at_chunk_boundaries_and_carry_their_place(session):
    _id, _n, chunks = doc_chunks(session, "d.pdf", [60, 60])
    parts = split_parts(chunks, part_chars=80)
    assert len(parts) == 2 and parts[0].startswith("[page 1]") and parts[1].startswith("[page 2]") and len(split_parts(chunks, part_chars=1000)) == 1


def test_the_calls_can_run_at_the_same_time(session):
    docs = [doc_chunks(session, f"{n}.pdf", [4000]) for n in "abc"]
    echo = lambda prompt: described(*[(ref, label.split(".")[0]) for ref, label in re.findall(r"### (d\d+) \| (\S+)", prompt)])
    builder = ProfileBuilder(MockLLMClient([echo, echo, echo]), budget=5000, workers=3)
    assert sorted(p.main_entity for _n, p in builder.run(docs).values()) == ["a", "b", "c"] and len(builder.stats.calls) == 3


def test_cards_give_the_documents_in_focus_in_full_and_the_others_in_one_line():
    entries = {"d1": ("deck.pdf", DocProfile(**PROFILE)), "d2": ("lease.docx", DocProfile(main_entity="Acme Corp", document_type="lease", scope="Premises at Pune."))}
    full = render_cards(entries)
    assert "deck.pdf: Acme Corp, investor deck. Presents the company to investors." in full and "status: final" in full and "scope: Premises at Pune." in full
    focused = render_cards(entries, focus={"d2"})
    assert "scope: Premises at Pune." in focused and "status: final" not in focused and "deck.pdf: Acme Corp, investor deck." in focused
    assert card("x.pdf", DocProfile()) == "- x.pdf: not described."


# ---- the entity gate ---------------------------------------------------------------------------------------------------------------------------
def entries():
    return {"da": ("a_company.pdf", DocProfile(main_entity="Alpha Ltd", document_type="deck")), "db": ("b_company.pdf", DocProfile(main_entity="Beta Inc", document_type="deck")),
            "dc": ("alpha_lease.docx", DocProfile(main_entity="Alpha Ltd", document_type="lease"))}


def test_documents_about_different_subjects_are_separated_and_related_ones_are_not():
    reply = {"groups": [{"docs": ["d1", "d2"], "entity": "Alpha Ltd"}, {"docs": ["d3"], "entity": "Beta Inc"}], "related": []}
    llm = MockLLMClient([reply])
    gate = decide_entity_groups(llm, entries())
    assert gate.applied and gate.calls == 1 and "Alpha Ltd" in llm.prompts[0] and "d1: a_company.pdf" in llm.prompts[0]
    assert gate.allows("da", "dc") and not gate.allows("da", "db") and gate.allows("da", "da") and gate.allows("da", "unknown")
    related = decide_entity_groups(MockLLMClient([{**reply, "related": [{"a": 1, "b": 2}, {"a": 1, "b": 9}]}]), entries())
    assert related.allows("da", "db") and related.related == [(1, 2)]


def test_a_failed_or_partial_gate_call_never_drops_a_candidate():
    failed = decide_entity_groups(MockLLMClient([LLMPermanentError("boom"), LLMPermanentError("boom")]), entries())
    assert not failed.applied and "failed" in failed.note
    partial = decide_entity_groups(MockLLMClient([{"groups": [{"docs": ["d1"], "entity": "Alpha"}]}]), entries())
    assert partial.applied and partial.allows("da", "db") and "compared with everything" in partial.note
    assert not decide_entity_groups(MockLLMClient([]), {"da": entries()["da"]}).applied  # one document: no call, nothing to separate


def candidate_of(a, b, hint="conflicting"):
    first, second = (a, b) if a.id <= b.id else (b, a)
    return Candidate(claim_id=first.id, candidate_id=second.id, topic_score=0.9, relation_hint=hint, matched_by="llm",
                         comparison=AttributeComparison(subject_match=True, metric_match=True, unit_match=True, period_match=True, value_relation="different",
                                                        conditions_differ=False, assumptions_differ=False))


def test_candidates_across_unrelated_documents_are_dropped_before_judging(session, room):  # noqa: F811
    a = Claim(document_id=room.deck.id, chunk_id=room.deck_p1, subject="plant", metric="capacity", value=120000, unit="tonnes", claim_text="Capacity is 120,000 tonnes")
    b = Claim(document_id=room.capex.id, chunk_id=room.capex_current, subject="plant", metric="capacity", value=100000, unit="tonnes", claim_text="Capacity 100,000 tonnes")
    c = Claim(document_id=room.lease.id, chunk_id=room.term, subject="lease", metric="term", value=9, unit="years", claim_text="The lease term is 9 years.")
    session.add_all([a, b, c])
    session.commit()
    gate = EntityGate(applied=True, doc_group={room.deck.id: 1, room.capex.id: 2, room.lease.id: 1})
    result = apply_gates(session, [candidate_of(a, b), candidate_of(a, c)], gate)
    assert result.entity_dropped == 1 and [(p.claim_id, p.candidate_id) for p in result.candidates] == [(candidate_of(a, c).claim_id, candidate_of(a, c).candidate_id)]
    assert "entity gate" in result.log[0]
    assert len(apply_gates(session, [candidate_of(a, b)], None).candidates) == 1  # no gate: nothing dropped


# ---- the list-member guard ---------------------------------------------------------------------------------------------------------------------
def test_two_name_claims_on_one_page_of_one_document_are_list_items(session):
    doc, chunks = add_doc(session, "deck.pdf", "pdf", [dict(content="Focus: Sustainable Agriculture", page_number=14), dict(content="Focus: Marine and Terrestrial", page_number=14),
                                                      dict(content="Focus: Forestry", page_number=20), dict(content="Revenue 5", page_number=14)])
    session.commit()
    mk = lambda chunk, value, kind: Claim(document_id=doc.id, chunk_id=chunk.id, subject="Projects", metric="focus", value=value, claim_text=chunk.content, claim_category=kind)
    a, b, far, num = mk(chunks[0], "Sustainable Agriculture", "name"), mk(chunks[1], "Marine and Terrestrial", "name"), mk(chunks[2], "Forestry", "name"), mk(chunks[3], "5", "quantity")
    session.add_all([a, b, far, num])
    session.commit()
    result = apply_gates(session, [candidate_of(a, b), candidate_of(a, far), candidate_of(a, num)], None)
    assert result.list_dropped == 1 and len(result.candidates) == 2 and "list members" in result.log[0]  # other page and a quantity still compare
    assert apply_gates(session, [candidate_of(a, b)], None, list_guard=False).candidates != []


# ---- the judge: cards, withdrawn contradictions, consistent results --------------------------------------------------------------------------








def test_the_self_denial_check_reads_only_denials():
    assert denies_contradiction("They do not establish a contradiction.") and denies_contradiction("This is not a contradiction.") and denies_contradiction("not competing values")
    assert not denies_contradiction("The two values conflict and no distinguishing context is stated.") and not denies_contradiction("", None) and not denies_contradiction("Both cannot be true.")


