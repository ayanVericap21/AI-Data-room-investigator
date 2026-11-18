"""ClaimGroupJudge: a fact and every claim that may state it get ONE verdict; findings need exact quotes for the claims they name."""
from __future__ import annotations

import sys
import threading
import time
from itertools import combinations
from pathlib import Path
from types import SimpleNamespace

from sqlalchemy import func, select

from app.llm.base import LLMClient, RawResponse
from app.llm.errors import LLMPermanentError
from app.llm.mock import MockLLMClient
from app.models import Claim, Evidence, Finding
from app.schemas.claim import AttributeComparison, Candidate
from app.services.claim_group_judge import SYSTEM, ClaimGroupJudge, GroupVerdicts, denies_contradiction
from app.services.doc_profiles import DocProfile
from app.tools import build_registry
from tests.test_tools import add_doc, room  # noqa: F401  (fixtures and helper)


def comparison(**kw):
    base = dict(subject_match=True, metric_match=True, unit_match=True, period_match=True, value_relation="different",
                conditions_differ=False, assumptions_differ=False)
    base.update(kw)
    return AttributeComparison(**base)


def candidate(a, b, hint="conflicting", how="llm"):
    return Candidate(claim_id=a.id, candidate_id=b.id, topic_score=1.0, relation_hint=hint, comparison=comparison(), matched_by=how)


def candidates_of(claims, how="llm"):
    """Every claim a candidate of every other, in the order the claims are given."""
    return [candidate(a, b, how=how) for a, b in combinations(claims, 2)]


def make_claims(session, room):  # noqa: F811
    a = Claim(document_id=room.deck.id, chunk_id=room.deck_p1, subject="plant", metric="capacity", value=120000, unit="tonnes",
              claim_text="Capacity is 120,000 tonnes per year")
    b = Claim(document_id=room.capex.id, chunk_id=room.capex_current, subject="plant", metric="capacity", value=100000, unit="tonnes",
              claim_text="Capacity 100,000 tonnes")
    session.add_all([a, b])
    session.commit()
    return a, b


def sentence(value) -> str:
    return f"The warehouse stores {value} tonnes."


def warehouse_claims(session, values, prefix="doc", **kw):
    """One claim per value, each in a document of its own: 'The warehouse stores <value> tonnes.'"""
    claims = []
    for n, value in enumerate(values):
        doc, chunks = add_doc(session, f"{prefix}{n}.pdf", "pdf", [dict(content=sentence(value), page_number=1)])
        claims.append(Claim(document_id=doc.id, chunk_id=chunks[0].id, subject=kw.get("subject", "warehouse"), metric=kw.get("metric", "storage capacity"),
                            value=str(value), claim_text=sentence(value)))
    session.add_all(claims)
    session.commit()
    return claims


def judge(session, room, llm, budget=60_000, **kw):  # noqa: F811
    return ClaimGroupJudge(session, llm, build_registry(session, None, room.investigation.id), max_chars_per_call=budget, **kw)


def verdict(claims, start=1, group=1, cls="contradiction", aside=(), quote=sentence, **kw):
    """One group's verdict: an evidence entry for every claim not set aside. Claim numbers run from ``start`` in the order given."""
    evidence = [dict(claim=start + i, value=str(c.value), quote=quote(c.value)) for i, c in enumerate(claims) if (start + i) not in aside]
    base = dict(group=group, same_attribute=True, attribute_reasoning="all state the capacity", set_aside=list(aside), classification=cls, severity="high",
                confidence=0.9, title="Storage capacity differs", description="The documents state different capacities.", evidence=evidence)
    base.update(kw)
    return base


def answer(*verdicts):
    return {"verdicts": list(verdicts)}


def evidence_rows(session):
    return session.scalars(select(Evidence)).all()


BUDGET_TABLE = ("1 | Source | Amount\n7 | Deferred developer fee | $600,000\n21 | Insurance, taxes, interest and permits | $1,000,000\n"
                "22 | Developer fee | $2,750,000\n23 | Financing costs and reserves | $700,000")


def table_and_cap(session):
    """A budget table row ('22 | Developer fee | $2,750,000') and a sentence stating a cap on the same fee."""
    doc_a, rows_a = add_doc(session, "budget.pdf", "pdf", [dict(content=BUDGET_TABLE, page_number=1, content_type="table")])
    doc_b, rows_b = add_doc(session, "application.pdf", "pdf", [dict(content="The developer fee is capped at $2,000,000.", page_number=1)])
    a = Claim(document_id=doc_a.id, chunk_id=rows_a[0].id, subject="Developer fee", metric="developer fee", value="$2,750,000", claim_text="Developer fee: $2,750,000")
    b = Claim(document_id=doc_b.id, chunk_id=rows_b[0].id, subject="Developer fee", metric="developer fee", value="$2,000,000",
              claim_text="The developer fee is capped at $2,000,000.")
    session.add_all([a, b])
    session.commit()
    return a, b


def test_a_quote_retyped_without_the_table_pipes_is_kept_as_the_rows_own_text(session, room):
    a, b = table_and_cap(session)
    v = verdict([a, b], quote=lambda value: f"Developer fee {value}")  # the model dropped the row number and the pipes
    stats = judge(session, room, MockLLMClient([answer(v)])).run(candidates_of([a, b]))
    assert stats.findings_saved == 1 and stats.findings_rejected == 0
    stored = {row.quoted_text for row in evidence_rows(session)}
    assert "The developer fee is capped at $2,000,000." in stored and any("$2,750,000" in text and "|" in text for text in stored)  # source text, not the model's


def test_a_quote_that_is_not_in_the_chunk_is_replaced_by_the_claims_own_row_and_counted(session, room):
    a, b = table_and_cap(session)
    v = verdict([a, b], quote=lambda value: f"Developer fee budgeted at {value} in total")  # words the table does not contain
    stats = judge(session, room, MockLLMClient([answer(v)])).run(candidates_of([a, b]))
    assert stats.findings_saved == 1 and stats.findings_rejected == 0 and stats.quotes_replaced >= 1
    assert "22 | Developer fee | $2,750,000" in {row.quoted_text for row in evidence_rows(session)}


# ---- one verdict for the whole group -------------------------------------------------------------------------------------------------------
def test_one_differing_claim_among_four_makes_the_group_a_contradiction_with_a_quote_for_each_claim(session, room):
    claims = warehouse_claims(session, [500, 500, 500, 450])
    stats = judge(session, room, MockLLMClient([answer(verdict(claims))])).run(candidates_of(claims))
    finding = session.scalars(select(Finding)).one()
    assert (stats.groups_total, stats.groups_judged, stats.llm_calls, stats.findings_saved) == (1, 1, 1, 1)
    assert finding.classification == "contradiction" and len(evidence_rows(session)) == 4
    assert all(stats.findings_by_document[c.document_id] == 1 for c in claims)


def test_the_judge_sees_every_claim_and_excerpt_of_the_group_once_in_one_call(session, room):
    claims = warehouse_claims(session, [500, 450, 400, 350, 300])
    llm = MockLLMClient([answer(verdict(claims))])
    judge(session, room, llm).run(candidates_of(claims))
    prompt = llm.prompts[0]
    assert prompt.count("### Group 1") == 1 and prompt.count("Excerpt [") == 5 and "Code checks:" in prompt  # 10 comparisons, 5 excerpts
    assert "Claim A" not in prompt


def test_several_groups_share_a_call_and_their_claims_are_numbered_through_the_call(session, room):
    one, two = warehouse_claims(session, [500, 450, 400]), warehouse_claims(session, [10, 20], prefix="other")
    llm = MockLLMClient([answer(verdict(one), verdict(two, start=4, group=2))])
    stats = judge(session, room, llm).run(candidates_of(one) + candidates_of(two))
    assert stats.groups_total == 2 and stats.llm_calls == 1 and stats.findings_saved == 2
    assert "### Group 2" in llm.prompts[0] and "Claim [4]" in llm.prompts[0]


def shared_chunk_groups(session, big=1000):
    """Two groups that share one claim: a with b (a strong pair) and a with c (a weak pair, so a group of its own). a's chunk is ``big`` characters long."""
    texts = [("A", 500, "x" * big), ("B", 450, "y" * 50), ("C", 400, "z" * 50)]
    claims = []
    for n, (letter, value, filler) in enumerate(texts):
        doc, chunks = add_doc(session, f"share{n}.pdf", "pdf", [dict(content=f"UNIQUE-CHUNK-{letter} capacity {value} t {filler}", page_number=1)])
        claims.append(Claim(document_id=doc.id, chunk_id=chunks[0].id, subject="warehouse", metric="storage capacity", value=str(value) + ("%" if letter == "C" else ""), claim_text=f"stores {value} t"))  # C is a percentage: it cannot join the cluster of A and B (an absolute figure), so its candidate is a group of its own
    session.add_all(claims)
    session.commit()
    a, b, c = claims
    return claims, [candidate(a, b, how="llm"), candidate(a, c, how="embedding")]


class Reader(LLMClient):
    """Reads each call's prompt like the model would: a contradiction for every group, with one quote per claim taken from the claim's own line."""

    def __init__(self, answers: bool = True):
        super().__init__(max_attempts=1)
        self.prompts, self.answers = [], answers

    def _raw_generate(self, prompt, system, json_mode):
        import json
        import re

        self.prompts.append(prompt)
        verdicts = []
        if self.answers:
            for group in re.split(r"(?=### Group \d+)", prompt)[1:]:
                number = int(re.match(r"### Group (\d+)", group).group(1))
                found = re.findall(r"Claim \[(\d+)\].*?stores (\d+) t", group)
                letter = {"500": "A", "450": "B", "400": "C"}
                verdicts.append(dict(group=number, same_attribute=True, attribute_reasoning="capacity", set_aside=[], classification="contradiction", severity="high", confidence=0.9,
                                     title=f"group {number} differs", description="the capacities differ",
                                     evidence=[dict(claim=int(n), value=v, quote=f"UNIQUE-CHUNK-{letter[v]} capacity {v} t") for n, v in found]))
        return RawResponse(text=json.dumps({"verdicts": verdicts}))


def test_a_chunk_shared_by_two_groups_of_one_call_is_printed_once(session, room):
    claims, candidates = shared_chunk_groups(session)
    llm = Reader(answers=False)
    judge(session, room, llm).run(candidates)
    prompt = llm.prompts[0]
    assert "### Group 1" in prompt and "### Group 2" in prompt  # both groups are in the first call
    assert prompt.count("UNIQUE-CHUNK-A") == 1 and prompt.count("UNIQUE-CHUNK-B") == 1 and prompt.count("UNIQUE-CHUNK-C") == 1  # each excerpt once in the whole call
    assert "(printed under Group 1)" in prompt  # the second group points to where the shared excerpt is printed
    assert "may be in an earlier group of this call" in prompt  # and the call says what such a pointer means


def test_a_call_without_a_shared_chunk_has_no_pointer_and_no_note(session, room):
    claims = warehouse_claims(session, [500, 450])
    llm = MockLLMClient([answer(verdict(claims))])
    judge(session, room, llm).run(candidates_of(claims))
    assert "the same excerpt" not in llm.prompts[0] and "may be in an earlier group" not in llm.prompts[0]


def test_groups_that_share_a_chunk_need_less_room_so_more_of_them_fit_in_one_call(session, room):
    claims, candidates = shared_chunk_groups(session)
    # each group alone is about 1.7k characters; together about 2.5k when the shared chunk is counted once, 3.5k when it is counted twice
    fits, split = Reader(answers=False), Reader(answers=False)
    judge(session, room, fits, budget=3000).run(candidates)
    judge(session, room, split, budget=2000).run(candidates)
    assert "### Group 2" in fits.prompts[0]  # one call holds both groups
    assert "### Group 2" not in split.prompts[0]  # too little room for both: the second group goes to a call of its own


def test_a_verdict_for_a_group_whose_excerpt_is_printed_under_another_group_is_handled_by_its_own_claim_numbers(session, room):
    claims, candidates = shared_chunk_groups(session)
    llm = Reader()
    stats = judge(session, room, llm).run(candidates)
    assert "(printed under Group 1)" in llm.prompts[0] and len(llm.prompts) == 1  # one call, answered for both groups
    assert stats.findings_saved == 2 and stats.findings_rejected == 0 and stats.misaligned == 0 and stats.claims_dropped == 0


def test_a_group_too_large_for_one_call_is_folded_to_its_distinct_values(session, room):
    claims = warehouse_claims(session, [10, 10, 20, 20, 30, 30])
    llm = MockLLMClient([answer(verdict([claims[0], claims[2], claims[4]]))])
    stats = judge(session, room, llm, budget=700).run(candidates_of(claims))
    assert llm.prompts[0].count("Claim [") == 3 and stats.findings_saved == 1


def test_a_group_whose_claims_all_state_the_same_value_is_not_sent_to_the_llm(session, room):
    claims = warehouse_claims(session, [500, 500, 500])
    llm = MockLLMClient([])
    stats = judge(session, room, llm).run(candidates_of(claims))
    assert llm.prompts == [] and (stats.groups_agreed, stats.groups_judged) == (1, 0)
    assert stats.agreement_findings == 1 and session.scalars(select(Finding)).one().classification == "consistent"


def test_agreements_are_counted_but_not_saved_when_consistent_findings_are_off(session, room):
    claims = warehouse_claims(session, [500, 500])
    stats = judge(session, room, MockLLMClient([]), save_consistent=False).run(candidates_of(claims))
    assert stats.groups_agreed == 1 and stats.consistent_not_saved == 1 and stats.findings_saved == 0


def test_agreements_of_one_attribute_between_the_same_documents_become_one_finding(session, room):
    docs = [add_doc(session, f"{name}.pdf", "pdf", [dict(content=f"Warehouse one stores 500 tonnes.", page_number=1), dict(content="Warehouse two stores 700 tonnes.", page_number=2)])
            for name in ("a", "b")]
    claims = [Claim(document_id=d.id, chunk_id=chunks[i].id, subject=f"warehouse {n}", metric="storage capacity", value=v, claim_text=chunks[i].content)
              for d, chunks in docs for i, (n, v) in enumerate((("one", "500"), ("two", "700")))]
    session.add_all(claims)
    session.commit()
    a1, a2, b1, b2 = claims
    stats = judge(session, room, MockLLMClient([])).run([candidate(a1, b1), candidate(a2, b2)])
    assert stats.groups_agreed == 2 and stats.grouped == 1
    assert session.scalars(select(Finding)).one().title == "Consistent: storage capacity agrees in 2 places"


# ---- what is checked before a finding is saved ---------------------------------------------------------------------------------------------
def test_a_claim_set_aside_is_left_out_of_the_finding(session, room):
    claims = warehouse_claims(session, [500, 450, 7])
    stats = judge(session, room, MockLLMClient([answer(verdict(claims, aside={3}))])).run(candidates_of(claims))
    assert stats.findings_saved == 1 and len(evidence_rows(session)) == 2
    assert "set aside" in stats.records[(claims[0].id, claims[2].id)].outcome


def test_a_group_about_different_attributes_is_not_comparable_and_never_becomes_a_finding(session, room):
    claims = warehouse_claims(session, [500, 450])
    stats = judge(session, room, MockLLMClient([answer(verdict(claims, same_attribute=False, attribute_reasoning="a height and a floor level"))])).run(candidates_of(claims))
    assert stats.verdict_counts["not_comparable"] == 1 and stats.findings_saved == 0
    assert "not comparable. a height and a floor level" in stats.records[(claims[0].id, claims[1].id)].outcome


def test_an_invented_quote_is_never_stored_the_claims_own_passage_is_stored_instead(session, room):
    claims = warehouse_claims(session, [500, 450])
    invented = verdict(claims, quote=lambda v: f"Capacity of {v} tonnes was quoted from nowhere.")
    stats = judge(session, room, MockLLMClient([answer(invented)])).run(candidates_of(claims))
    stored = {row.quoted_text for row in evidence_rows(session)}
    assert stats.findings_saved == 1 and stats.quotes_replaced == 2
    assert stored == {sentence(500), sentence(450)} and not any("nowhere" in text for text in stored)


def test_a_quote_that_cannot_be_replaced_still_rejects_the_finding(session, room):
    claims = warehouse_claims(session, [500, 450])
    for c, value in zip(claims, ("999", "888")):
        c.claim_text, c.value = "stated elsewhere", value  # the claim's own value is not in its chunk, so there is nothing real to quote
    session.commit()
    invented = verdict(claims, quote=lambda v: f"Capacity of {v} tonnes was quoted from nowhere.")
    stats = judge(session, room, MockLLMClient([answer(invented)])).run(candidates_of(claims))
    assert stats.findings_saved == 0 and stats.findings_rejected == 1 and session.scalar(select(func.count()).select_from(Finding)) == 0


def test_a_claim_whose_quote_does_not_show_its_value_is_left_out_but_the_rest_still_make_the_finding(session, room):
    claims = warehouse_claims(session, [500, 450, 400, 300])
    v = verdict(claims)
    v["evidence"][3]["quote"] = sentence(500)  # shows 500, not 300
    stats = judge(session, room, MockLLMClient([answer(v)])).run(candidates_of(claims))
    finding = session.scalars(select(Finding)).one()
    assert stats.findings_saved == 1 and stats.claims_dropped == 1 and len(evidence_rows(session)) == 3 and "Not quoted" in finding.description


def test_a_contradiction_that_loses_the_claim_that_differed_is_rejected(session, room):
    claims = warehouse_claims(session, [500, 500, 450])
    v = verdict(claims)
    v["evidence"][2]["quote"] = sentence(500)  # the claim that differs cannot be quoted: the rest agree, so there is nothing to report
    stats = judge(session, room, MockLLMClient([answer(v)])).run(candidates_of(claims))
    assert stats.findings_saved == 0 and stats.findings_rejected == 1


def test_a_contradiction_that_denies_itself_is_withdrawn(session, room):
    claims = warehouse_claims(session, [500, 450])
    v = verdict(claims, description="The excerpts do not establish a contradiction: these are not competing values.")
    stats = judge(session, room, MockLLMClient([answer(v)])).run(candidates_of(claims))
    assert stats.withdrawn == 1 and stats.findings_saved == 0
    assert denies_contradiction("There is no real conflict") and not denies_contradiction("The values conflict.")


def short(claims, group=1, same_attribute=True, reasoning="all state the same capacity"):
    """A short verdict: no severity, confidence, title, description or evidence (what the prompt asks for consistent and not comparable groups)."""
    return dict(group=group, same_attribute=same_attribute, attribute_reasoning=reasoning, set_aside=[], classification="consistent")


def test_the_prompts_ask_for_a_short_verdict_when_a_group_is_consistent_or_not_comparable():
    from app.services.claim_group_judge import RULE_SYSTEM

    for system in (SYSTEM, RULE_SYSTEM):
        assert "SHORT verdict" in system and "at most 15 words" in system and "classification: consistent" in system
        assert "FULL verdict" in system and "Every verdict has ALL of these fields" not in system


def test_a_short_consistent_verdict_is_counted_and_needs_no_evidence(session, room):
    claims = warehouse_claims(session, [500, 450])  # different values, so code does not settle the group itself
    stats = judge(session, room, MockLLMClient([answer(short(claims))]), save_consistent=False).run(candidates_of(claims, how="lexical"))
    assert stats.verdict_counts["consistent"] == 1 and stats.consistent_not_saved == 1
    assert stats.findings_saved == 0 and stats.misaligned == 0 and stats.echo_missing == 0


def test_a_short_consistent_verdict_is_saved_with_the_claims_own_quotes_when_consistent_findings_are_on(session, room):
    claims = warehouse_claims(session, [500, 450])
    stats = judge(session, room, MockLLMClient([answer(short(claims))]), save_consistent=True).run(candidates_of(claims, how="lexical"))
    assert stats.findings_saved == 1 and stats.quotes_from_code == 2
    assert {row.quoted_text for row in evidence_rows(session)} == {sentence(500), sentence(450)}


def test_a_short_not_comparable_verdict_is_counted_with_its_reason_and_nothing_is_saved(session, room):
    claims = warehouse_claims(session, [500, 450])
    stats = judge(session, room, MockLLMClient([answer(short(claims, same_attribute=False, reasoning="a height and a floor level"))])).run(candidates_of(claims, how="lexical"))
    assert stats.verdict_counts["not_comparable"] == 1 and stats.findings_saved == 0 and stats.echo_missing == 0
    assert any("a height and a floor level" in (record.outcome or "") for record in stats.records.values())


def test_a_full_verdict_without_echoed_values_is_still_counted_as_missing_them(session, room):
    claims = warehouse_claims(session, [500, 450])
    v = verdict(claims)
    for ev in v["evidence"]:
        ev["value"] = ""
    stats = judge(session, room, MockLLMClient([answer(v)])).run(candidates_of(claims))
    assert stats.echo_missing == 1 and stats.findings_saved == 1


def test_a_consistent_verdict_is_counted_but_not_saved_when_consistent_findings_are_off(session, room):
    claims = warehouse_claims(session, [500, 450])
    stats = judge(session, room, MockLLMClient([answer(verdict(claims, cls="consistent"))]), save_consistent=False).run(candidates_of(claims))
    assert stats.consistent_not_saved == 1 and stats.findings_saved == 0


def test_a_contradiction_between_names_where_one_contains_the_other_is_saved_as_consistent(session, room):
    names = ["Technical Officer (instrumentation)", "Technical Officer"]
    claims = warehouse_claims(session, names, subject="surveyor", metric="role")
    for c in claims:
        c.claim_category = "name"
    session.commit()
    stats = judge(session, room, MockLLMClient([answer(verdict(claims))])).run(candidates_of(claims))
    assert stats.reclassified_names == 1 and session.scalars(select(Finding)).one().classification == "consistent"


def test_insufficient_evidence_without_quotes_is_saved_with_the_claims_own_sentences(session, room):
    claims = warehouse_claims(session, ["a large amount of", "a small amount of"])
    v = verdict(claims, cls="insufficient_evidence", severity="low", quote=lambda _v: "")
    stats = judge(session, room, MockLLMClient([answer(v)])).run(candidates_of(claims))
    assert stats.findings_saved == 1 and stats.quotes_from_code == 2


def test_insufficient_evidence_about_unrelated_attributes_is_not_comparable(session, room):
    a = warehouse_claims(session, [500])[0]
    b = warehouse_claims(session, [450], prefix="rent", subject="lease", metric="annual rent")[0]
    b.document_id = a.document_id
    session.commit()
    stats = judge(session, room, MockLLMClient([answer(verdict([a, b], cls="insufficient_evidence"))])).run([candidate(a, b)])
    assert stats.verdict_counts["not_comparable"] == 1 and stats.findings_saved == 0


# ---- values that do not belong to their claims, failed calls, retries ----------------------------------------------------------------------
def test_a_verdict_for_values_that_are_not_the_claims_is_asked_again_alone(session, room):
    claims = warehouse_claims(session, [500, 450])
    wrong = verdict(claims)
    wrong["evidence"][0]["value"] = "999 tonnes"
    llm = MockLLMClient([answer(wrong), answer(verdict(claims))])
    stats = judge(session, room, llm).run(candidates_of(claims))
    assert (stats.misaligned, stats.retries, stats.llm_calls, stats.findings_saved) == (1, 1, 2, 1) and stats.echo_log


def test_a_value_the_model_repeats_from_the_claims_excerpt_belongs_to_the_claim(session, room):
    claims = warehouse_claims(session, [500, 450])
    v = verdict(claims)
    v["evidence"][0]["value"] = "stores 500 tonnes"  # the words around the value, as written in the excerpt
    llm = MockLLMClient([answer(v)])
    stats = judge(session, room, llm).run(candidates_of(claims))
    assert stats.misaligned == 0 and stats.retries == 0 and stats.findings_saved == 1


def test_a_failed_call_is_recorded_and_its_group_is_asked_again_once(session, room):
    claims = warehouse_claims(session, [500, 450])
    llm = MockLLMClient([LLMPermanentError("401 bad key"), answer(verdict(claims))])
    stats = judge(session, room, llm).run(candidates_of(claims))
    assert (stats.batches_failed, stats.retries, stats.findings_saved) == (1, 1, 1) and "401" in stats.errors[0]


def test_a_group_the_model_skips_gets_an_outcome_saying_so(session, room):
    claims = warehouse_claims(session, [500, 450])
    stats = judge(session, room, MockLLMClient([answer()] * 20)).run(candidates_of(claims))
    assert stats.findings_saved == 0 and "no verdict returned" in stats.records[(claims[0].id, claims[1].id)].outcome


# ---- the prompt ----------------------------------------------------------------------------------------------------------------------------
def test_the_judge_is_told_when_a_value_is_inside_a_limit(session, room):
    da, ca = add_doc(session, "limits.pdf", "pdf", [dict(content="The permit allows a maximum of 8 samples per site.", page_number=1)])
    db, cb = add_doc(session, "log.pdf", "pdf", [dict(content="At most 6 samples were taken per site in practice.", page_number=1)])
    a = Claim(document_id=da.id, chunk_id=ca[0].id, subject="site", metric="maximum samples per site", value="8", claim_text="a maximum of 8 samples per site")
    b = Claim(document_id=db.id, chunk_id=cb[0].id, subject="site", metric="samples taken per site", value="6", claim_text="6 samples were taken per site")
    session.add_all([a, b])
    session.commit()
    llm = MockLLMClient([answer()] * 20)
    judge(session, room, llm).run([candidate(a, b)])
    assert "A value inside a limit is compliant" in llm.prompts[0]


def test_the_judge_sees_the_cards_of_the_documents_in_its_call(session, room):
    claims = warehouse_claims(session, [500, 450])
    cards = {c.document_id: (f"doc{n}.pdf", DocProfile(main_entity="Acme Warehousing", document_type="lease")) for n, c in enumerate(claims)}
    llm = MockLLMClient([answer()] * 20)
    judge(session, room, llm, profiles=cards).run(candidates_of(claims))
    assert "Documents in this collection" in llm.prompts[0] and "Acme Warehousing" in llm.prompts[0]


def test_every_call_starts_with_the_same_cards_of_all_the_documents_so_the_start_can_be_cached(session, room):
    one, two = warehouse_claims(session, [500, 450]), warehouse_claims(session, [10, 20], prefix="other")
    every = one + two
    cards = {c.document_id: (f"doc{n}.pdf", DocProfile(main_entity=f"Entity {n}", document_type="lease")) for n, c in enumerate(every)}
    llm = MockLLMClient([answer()] * 20)
    judge(session, room, llm, profiles=cards, budget=700).run(candidates_of(one) + candidates_of(two))  # a small budget: each group goes in a call of its own
    first, second = llm.prompts[0], llm.prompts[1]
    assert first != second and "### Group 1" in first and "### Group 1" in second
    start_a, start_b = first.split("### Group 1")[0], second.split("### Group 1")[0]
    assert start_a == start_b  # the same text before the groups in both calls
    assert all(f"Entity {n}" in start_a for n in range(4))  # a full card for every document, also the ones this call does not hold


def test_the_rules_prompt_states_the_group_verdict_and_stays_under_the_service_limit():
    assert len(SYSTEM) < 10_000
    for must in ("Even ONE claim that differs", "set_aside", "consistent", "contradiction", "missing_information", "insufficient_evidence", "evidence:", "ONE verdict per group"):
        assert must in SYSTEM, must


# ---- table rows ----------------------------------------------------------------------------------------------------------------------------
ROWS = ("[Installed_Tools!A8:F8] Tool_ID: CLEAN-011; Installed_Cost_INR_Crore: 19\n"
        "[Installed_Tools!A9:F9] Tool_ID: TEST-021; Installed_Cost_INR_Crore: 24")


def table_claim(session, value="19"):
    doc, chunks = add_doc(session, "tool_register.xlsx", "xlsx", [dict(content=ROWS, sheet_name="Installed_Tools", cell_reference="A8:F9", content_type="table_row")])
    claim = Claim(document_id=doc.id, chunk_id=chunks[0].id, subject="CLEAN-011", metric="Installed_Cost_INR_Crore", value=value, claim_text="Tool_ID: CLEAN-011; Installed_Cost_INR_Crore: 19")
    session.add(claim)
    session.commit()
    return claim, chunks[0]


def test_a_quote_from_another_row_of_the_same_chunk_is_refused(session):
    claim, chunk = table_claim(session, value="24")  # the quote is the neighbouring row (TEST-021, 24)
    quote, error = ClaimGroupJudge._row_quote(claim, chunk, "Installed_Cost_INR_Crore: 24")
    assert error and "different row" in error


def test_a_quote_from_the_right_row_without_its_label_is_widened_to_the_whole_row(session):
    claim, chunk = table_claim(session)
    quote, error = ClaimGroupJudge._row_quote(claim, chunk, "Installed_Cost_INR_Crore: 19")
    assert error is None and "CLEAN-011" in quote and "TEST-021" not in quote


# ---- parallel calls, the report ------------------------------------------------------------------------------------------------------------
class SlowFake(LLMClient):
    """Answers every call with no verdicts after a short wait, and records how many calls were in flight together."""

    def __init__(self):
        super().__init__(max_attempts=1)
        self.in_flight = self.peak = 0
        self._guard = threading.Lock()

    def _raw_generate(self, prompt, system, json_mode):
        with self._guard:
            self.in_flight += 1
            self.peak = max(self.peak, self.in_flight)
        time.sleep(0.08)
        with self._guard:
            self.in_flight -= 1
        return RawResponse(text='{"verdicts": []}')


def test_workers_make_the_model_calls_at_the_same_time_and_one_worker_does_not(session, room):
    groups = [candidates_of(warehouse_claims(session, [10 + n, 20 + n], prefix=f"g{n}-")) for n in range(4)]
    flat = [c for g in groups for c in g]
    fast, slow = SlowFake(), SlowFake()
    judge(session, room, fast, budget=300, workers=4).run(flat)  # each group is alone in its call
    judge(session, room, slow, budget=300, workers=1).run(flat)
    assert fast.peak >= 2 and slow.peak == 1


def test_the_client_keeps_each_calls_own_details_when_threads_share_it():
    client = SlowFake()

    def call(n):
        client.generate_structured_output("x" * n, GroupVerdicts, system=None)

    threads = [threading.Thread(target=call, args=(100 * (i + 1),)) for i in range(4)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert len(client.call_metrics) == 4 and len({m.prompt_chars for m in client.call_metrics}) == 4
    assert client.get_token_usage().calls == 4


def test_the_report_shows_each_claim_group_with_a_row_per_claim_and_its_candidates(session, room):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import ledger_run

    claims = warehouse_claims(session, [500, 450, 400])
    stats = judge(session, room, MockLLMClient([answer(verdict(claims))])).run(candidates_of(claims))
    text = "\n".join(ledger_run.sec_group_log(SimpleNamespace(judge=stats, session=session)))
    assert "GROUP 1: storage capacity  (3 claims; judged together)" in text
    assert all(f"doc{n}.pdf" in text for n in range(3)) and "claim 1 with claim 2 [conflicting, found by llm]: judge: contradiction" in text
    summary = "\n".join(ledger_run.sec_groups(SimpleNamespace(judge=stats, grouping_stats=None, candidates=candidates_of(claims), candidates_cross=3)))
    assert "claim groups: 1; agreed" in summary and "judged by the LLM: 1 in 1 calls" in summary


# ---- what the run on two different projects showed -----------------------------------------------------------------------------------------
def one_document_claims(session, texts_values, **kw):
    """Claims in different places (pages) of ONE document."""
    doc, chunks = add_doc(session, "pitch.pdf", "pdf", [dict(content=text, page_number=n) for n, (text, _v) in enumerate(texts_values, start=1)])
    claims = [Claim(document_id=doc.id, chunk_id=chunks[i].id, subject=kw.get("subject", "Tony Oehm"), metric=kw.get("metric", "Role"), value=v,
                    claim_text=text, claim_category=kw.get("category", "statement")) for i, (text, v) in enumerate(texts_values)]
    session.add_all(claims)
    session.commit()
    return claims


def test_the_same_words_in_another_order_are_one_value_whatever_the_claims_are_typed_as(session, room):
    claims = one_document_claims(session, [("Tony Oehm, Co-Founder | CEO", "Co-Founder | CEO"), ("Tony Oehm, CEO I Co-Founder", "CEO I Co-Founder")])
    v = verdict(claims, quote=lambda value: f"Tony Oehm, {value}", description="The ordering and punctuation differ but the same words are present.")
    stats = judge(session, room, MockLLMClient([answer(v)])).run(candidates_of(claims))
    assert stats.reclassified_names == 1 and session.scalars(select(Finding)).one().classification == "consistent"


def test_a_contradiction_that_says_the_same_words_are_present_is_withdrawn():
    assert denies_contradiction("the ordering and punctuation differ but the same words are present.") and denies_contradiction("The values differ only in wording.")
    assert not denies_contradiction("The values conflict: 100 against 120.")


def test_a_yearly_series_is_not_compared_and_never_sent_to_the_llm(session, room):
    claims = warehouse_claims(session, [33, 917, 900])
    for claim, year in zip(claims, ("2022", "2023", "2024")):
        claim.period = year
    session.commit()
    llm = MockLLMClient([])
    stats = judge(session, room, llm).run(candidates_of(claims))
    assert llm.prompts == [] and (stats.groups_series, stats.groups_judged, stats.findings_saved) == (1, 0, 0)
    assert "series" in stats.records[(claims[0].id, claims[1].id)].outcome


def test_two_claims_for_the_same_period_are_still_judged(session, room):
    claims = warehouse_claims(session, [500, 450, 700])
    for claim, year in zip(claims, ("2023", "2023", "2024")):
        claim.period = year
    session.commit()
    stats = judge(session, room, MockLLMClient([answer(verdict(claims))])).run(candidates_of(claims))
    assert stats.groups_series == 0 and stats.findings_saved == 1


def test_a_quote_too_short_to_keep_is_replaced_by_the_claims_own_sentence(session, room):
    claims = warehouse_claims(session, [500, 450])
    stats = judge(session, room, MockLLMClient([answer(verdict(claims, quote=lambda value: str(value)))])).run(candidates_of(claims))
    assert stats.findings_saved == 1 and stats.findings_rejected == 0 and stats.quotes_from_code == 2


def test_the_description_says_where_each_value_is_stated_and_when_it_is_one_document(session, room):
    same = one_document_claims(session, [("Seats: 500 in total.", "500"), ("Total seats are 450.", "450")], subject="hall", metric="seats")
    stats = judge(session, room, MockLLMClient([answer(verdict(same, quote=lambda v: {"500": "Seats: 500 in total.", "450": "Total seats are 450."}[v]))])).run(candidates_of(same))
    text = session.scalars(select(Finding)).one().description
    assert stats.findings_saved == 1 and "All of these are stated in pitch.pdf, in different places: 500 (page 1); 450 (page 2)." in text


def test_the_description_names_the_document_of_each_value_when_they_differ(session, room):
    claims = warehouse_claims(session, [500, 450])
    judge(session, room, MockLLMClient([answer(verdict(claims))])).run(candidates_of(claims))
    assert "Stated in: doc0.pdf, 500 (page 1); doc1.pdf, 450 (page 1)." in session.scalars(select(Finding)).one().description


def test_a_missing_figure_the_comparison_needs_is_saved_as_missing_information_with_the_claims_own_sentences(session, room):
    claims = warehouse_claims(session, ["12%", "about 400 tonnes"])
    v = verdict(claims, cls="missing_information", severity="medium", quote=lambda _v: "", description="The base of the 12% is not given.")
    stats = judge(session, room, MockLLMClient([answer(v)])).run(candidates_of(claims))
    finding = session.scalars(select(Finding)).one()
    assert finding.classification == "missing_information" and stats.quotes_from_code == 2


def test_a_candidate_found_by_nearest_neighbours_joins_the_cluster_it_links_to_when_the_claims_can_state_one_attribute(session, room):
    a, b, c, d = warehouse_claims(session, [500, 450, 400, 350])
    groups = ClaimGroupJudge._form_groups([candidate(a, b, how="llm"), candidate(a, c, how="embedding"), candidate(a, d, how="embedding"), candidate(a, b, how="embedding")])
    assert [g.claim_ids for g in groups] == [[a.id, b.id, c.id, d.id]]  # one group of four: the model's cluster, and the two claims the neighbours link to it
    assert len(groups[0].edges) == 4


def test_a_neighbour_that_cannot_join_the_cluster_is_compared_with_the_claim_it_links_to_as_a_pair(session, room):
    a, b, c, d = warehouse_claims(session, [500, 450, 400, 350])
    never = lambda x, y, how: y != c.id and x != c.id  # c cannot state the attribute of the cluster
    groups = ClaimGroupJudge._form_groups([candidate(a, b, how="llm"), candidate(a, c, how="embedding"), candidate(a, d, how="embedding")], never)
    assert [g.claim_ids for g in groups] == [[a.id, b.id, d.id], [a.id, c.id]]


def test_the_models_clusters_stay_whole_and_a_candidate_between_two_clusters_is_a_pair_not_a_merge(session, room):
    a, b, c, d, e = warehouse_claims(session, [500, 450, 400, 350, 300])
    groups = ClaimGroupJudge._form_groups(candidates_of([a, b, c]) + candidates_of([d, e]) + [candidate(c, d, how="lexical")])
    assert [g.claim_ids for g in groups] == [[a.id, b.id, c.id], [d.id, e.id], [c.id, d.id]]


def test_claims_linked_by_words_alone_form_one_group_up_to_the_cap_and_a_larger_set_is_split_by_attribute_not_into_pairs(session, room):
    claims = warehouse_claims(session, [100 + i for i in range(6)])
    chain = [candidate(x, y, how="lexical") for x, y in zip(claims, claims[1:])]
    assert [len(g.claim_ids) for g in ClaimGroupJudge._form_groups(chain)] == [6]  # a chain of six is one group
    groups = ClaimGroupJudge._form_groups(chain, cap=4)  # too large for one fact: the links that can state one attribute still keep claims together
    assert [len(g.claim_ids) for g in groups] == [2, 2, 2, 2, 2] and all(len(g.claim_ids) == 2 for g in groups)  # without an attribute test the chain falls back to pairs
    kept = ClaimGroupJudge._split_large([c.id for c in claims], chain, lambda x, y, how: x in {claims[0].id, claims[1].id, claims[2].id} and y in {claims[0].id, claims[1].id, claims[2].id}, 4)
    assert sorted(len(g.claim_ids) for g in kept) == [2, 2, 2, 3]  # three claims of one attribute stay one group; the three links that leave them are pairs


def test_a_central_estimate_inside_its_interval_agrees_and_other_values_do_not():
    from app.services.claim_group_judge import point_inside_range
    assert point_inside_range("approximately 1810 cal BCE", "1885-1740 cal BCE cal BCE") and point_inside_range("1740-1885 cal BCE", "1800 cal BCE")
    assert not point_inside_range("approximately 1500 cal BCE", "1885-1740 cal BCE")  # outside the range
    assert not point_inside_range("1810 AD", "1885-1740 cal BCE")  # not the same unit
    assert not point_inside_range("3000", "3480 +/- 25 BP")  # a plus-or-minus is not a range of this kind
    assert not point_inside_range("1810 cal BCE", "850-750 cal BCE")


# ---- every group is judged: groups asked again go in as few calls as fit, a call that keeps failing is halved ---------------------------------
class BusyForLargeCalls(LLMClient):
    """Fails with a transient error while a call holds more than ``limit`` groups (a service overloaded by large requests); otherwise answers every group
    in the call with a contradiction. Records how many groups each call held."""

    def __init__(self, limit, fail_first=0):
        super().__init__(max_attempts=1)
        self.limit, self.sizes, self.fail_first = limit, [], fail_first

    def _raw_generate(self, prompt, system, json_mode):
        import json
        import re

        groups = re.findall(r"### Group (\d+)", prompt)
        self.sizes.append(len(groups))
        if len(groups) > self.limit or len(self.sizes) <= self.fail_first:
            raise LLMTransientError("503 high demand")
        verdicts = []
        for group in re.split(r"(?=### Group \d+)", prompt)[1:]:
            number = int(re.match(r"### Group (\d+)", group).group(1))
            found = re.findall(r"Claim \[(\d+)\].*?stores (\d+) tonnes", group)
            verdicts.append(dict(group=number, same_attribute=True, attribute_reasoning="capacity", set_aside=[], classification="contradiction", severity="high", confidence=0.9,
                                 title=f"group {number}", description="differs", evidence=[dict(claim=int(n), value=v, quote=sentence(v)) for n, v in found]))
        return RawResponse(text=json.dumps({"verdicts": verdicts}))


def pairs_of_claims(session, count):
    candidates = []
    for k in range(count):
        a, b = warehouse_claims(session, [500 + k, 400 + k], prefix=f"p{k}-", subject=f"warehouse {k}")
        candidates.append(candidate(a, b))
    return candidates


from app.llm.errors import LLMTransientError  # noqa: E402


def test_after_a_failed_call_all_its_groups_are_asked_again_together_not_one_by_one(session, room):  # noqa: F811
    llm, pauses = BusyForLargeCalls(limit=100, fail_first=1), []
    stats = judge(session, room, llm, budget=1_000_000, retry_pause_seconds=7, sleep=pauses.append).run(pairs_of_claims(session, 12))
    assert llm.sizes == [12, 12]  # one call for all twelve groups, which failed; the same twelve in one call the second time
    assert (stats.findings_saved, stats.groups_unjudged, stats.batches_failed) == (12, 0, 1) and sum(pauses) == 7  # a pause before asking again


def test_a_call_that_keeps_failing_is_halved_until_it_goes_through_and_every_group_is_judged(session, room):  # noqa: F811
    llm = BusyForLargeCalls(limit=3)
    stats = judge(session, room, llm, budget=1_000_000, retry_pause_seconds=0, sleep=lambda _s: None).run(pairs_of_claims(session, 12))
    assert llm.sizes[0] == 12 and max(llm.sizes[1:]) <= 12 and 12 in llm.sizes[:2]  # the whole set first, again whole, then halves
    assert stats.findings_saved == 12 and stats.groups_unjudged == 0 and max(s for s in llm.sizes[2:]) <= 6


def test_a_stop_request_ends_the_asking_again(session, room):  # noqa: F811
    state = {"stop": False}
    llm = BusyForLargeCalls(limit=0)
    stats = judge(session, room, llm, budget=1_000_000, retry_pause_seconds=30, should_stop=lambda: state["stop"], sleep=lambda _s: state.update(stop=True)).run(pairs_of_claims(session, 4))
    assert llm.sizes == [4] and stats.groups_unjudged == 4  # one call, then the pause was interrupted: nothing more was asked


# ---- a "not comparable" verdict whose own reason names a difference is asked again ---------------------------------------------------------
def test_a_reason_that_names_a_difference_is_not_taken_as_not_comparable(session, room):  # noqa: F811
    from app.services.claim_group_judge import describes_a_difference

    assert describes_a_difference("Minor spelling difference in owner name between documents ('Ashbey' vs 'Ashby').")
    assert describes_a_difference("The values mismatch.") and describes_a_difference("Total differs from the sum")
    assert not describes_a_difference("Different subjects and metrics") and not describes_a_difference("A height and a floor level")
    assert not describes_a_difference("The claims measure different attributes; the periods differ") and not describes_a_difference("")


def test_such_a_verdict_is_asked_again_and_the_second_answer_is_used(session, room):  # noqa: F811
    claims = warehouse_claims(session, [500, 450])
    first = verdict(claims, same_attribute=False, attribute_reasoning="Minor spelling difference in the capacity figure ('500' vs '450').")
    llm = MockLLMClient([answer(first), answer(verdict(claims))])
    stats = judge(session, room, llm).run(candidates_of(claims))
    assert stats.reason_contradicts_verdict == 1 and stats.retries == 1 and stats.findings_saved == 1 and stats.verdict_counts["not_comparable"] == 0


def test_a_reason_about_different_things_is_final_at_once(session, room):  # noqa: F811
    claims = warehouse_claims(session, [500, 450])
    llm = MockLLMClient([answer(verdict(claims, same_attribute=False, attribute_reasoning="Different subjects and metrics"))])
    stats = judge(session, room, llm).run(candidates_of(claims))
    assert stats.reason_contradicts_verdict == 0 and stats.retries == 0 and stats.verdict_counts["not_comparable"] == 1


def test_if_the_second_answer_says_the_same_it_is_final_and_counted_not_comparable(session, room):  # noqa: F811
    claims = warehouse_claims(session, [500, 450])
    same = verdict(claims, same_attribute=False, attribute_reasoning="The figures differ in spelling only.")
    stats = judge(session, room, MockLLMClient([answer(same), answer(same)])).run(candidates_of(claims))
    assert stats.reason_contradicts_verdict == 1 and stats.retries == 1 and stats.verdict_counts["not_comparable"] == 1 and stats.findings_saved == 0
