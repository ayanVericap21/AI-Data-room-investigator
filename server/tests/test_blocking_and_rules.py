"""Blocking by code, the shortcuts that decide a group without the model (each as strict as the judge), and checks inside one document by patterns."""
from __future__ import annotations

from sqlalchemy import select

from app.llm.mock import MockLLMClient
from app.models import Claim, Evidence, Finding
from app.schemas.claim import ClaimView
from app.services.blocking import negated_terms, negation_candidates, shared_label_candidates, types_compatible
from app.services.internal_checks import InternalChecker
from app.services.internal_patterns import find_relations, period_count
from app.services.supersession import superseded_documents
from app.tools import build_registry
from tests.test_claim_group_judge import answer, candidate, candidates_of, judge, verdict
from tests.test_evaluator_fixes import SPEND, claim_in
from tests.test_tools import add_doc, room  # noqa: F401  (fixtures and helper)


def view(n, doc, subject, metric, value, kind="quantity", text=""):
    return ClaimView(id=f"c{n}", document_id=f"d{doc}", chunk_id=f"ch{n}", subject=subject, metric=metric, value=value, claim_text=text or f"{subject} {metric} {value}", value_type=kind)


def pairs(found):
    return {frozenset((c.claim_id, c.candidate_id)) for c in found}


# ---- blocking: shared labels ------------------------------------------------------------------------------------------------------------
def test_two_documents_that_name_the_same_rare_label_are_linked_even_when_their_subjects_differ():
    views = [view(1, 1, "Progress Report No. 3", "purchase order", "PO-55210", "identifier"), view(2, 2, "Invoice INV-7781", "purchase order", "PO-55120", "identifier"),
             view(3, 3, "Board update", "headcount", "60"), view(4, 4, "Safety log", "weather", "dry", "state")]
    assert pairs(shared_label_candidates(views)) == {frozenset({"c1", "c2"})}


def test_a_word_every_label_shares_links_nothing():
    views = [view(n, n, f"Document {n}", "date", f"{n} May 2025", "date") for n in range(1, 9)]
    assert shared_label_candidates(views) == []


def test_a_subject_and_a_metric_that_share_two_rare_words_are_linked_and_a_date_is_never_set_against_a_status():
    views = [view(1, 1, "Practical completion", "date", "July 2025", "date"), view(2, 2, "Harbor Point works", "Practical completion date", "08/07/2025", "date"),
             view(3, 3, "Practical completion", "status", "achieved", "state"), view(4, 4, "Site", "headcount", "48")]
    assert pairs(shared_label_candidates(views)) == {frozenset({"c1", "c2"})}
    assert not types_compatible(views[0], views[2]) and types_compatible(views[3], views[2]) is True


# ---- blocking: negation -----------------------------------------------------------------------------------------------------------------
def test_a_negation_names_the_thing_there_is_none_of():
    assert negated_terms(view(1, 1, "Project", "subcontractors engaged", "No", "state", "No subcontractors are engaged on the project.")) == {"subcontractor"}
    assert negated_terms(view(2, 1, "Site", "safety incidents", "No", "state", "No safety incidents have occurred on site.")) == {"safety", "incident"}
    assert negated_terms(view(3, 1, "Project", "subcontractors engaged", "None", "state", "")) == {"subcontractor"}
    assert negated_terms(view(4, 1, "Works", "start date", "1 March 2025", "date", "Works start on 1 March 2025.")) == frozenset()


def test_a_negation_is_linked_as_the_rule_to_the_claims_that_assert_the_thing():
    views = [view(1, 1, "Project", "subcontractors engaged", "No", "state", "No subcontractors are engaged on the project."),
             view(2, 2, "Brightline Steel Ltd", "role", "our installation subcontractor", "statement", "Installation is carried out by Brightline Steel Ltd, our installation subcontractor."),
             view(3, 3, "Invoice", "payment terms", "30 days", "quantity", "Payment terms: 30 days.")]
    found = negation_candidates(views)
    assert [(c.claim_id, c.candidate_id, c.matched_by) for c in found] == [("c1", "c2", "rule")]


# ---- decided in code only where code is as strict as the judge ----------------------------------------------------------------------
def test_a_planned_a_required_and_an_actual_start_date_are_one_contextual_difference_decided_without_the_model(session, room):  # noqa: F811
    msa = claim_in(session, "msa.pdf", "Works commence on 1 March 2025 for a term of 6 months, ending on 31 August 2025.", "1 March 2025", subject="works", metric="commencement date", category="date")
    po = claim_in(session, "po.pdf", "Required start date: 1 March 2025.", "1 March 2025", subject="Purchase Order PO-55120", metric="required start date", category="date")
    report = claim_in(session, "report.pdf", "Works commenced on 3 March 2025.", "3 March 2025", subject="Works", metric="commencement date", category="date")
    llm = MockLLMClient([])
    stats = judge(session, room, llm).run(candidates_of([msa, po, report]))
    finding = session.scalars(select(Finding)).one()
    assert llm.prompts == [] and stats.groups_basis == 1 and finding.classification == "contextual_difference" and len(session.scalars(select(Evidence)).all()) == 3


def test_a_final_value_against_a_contract_value_is_a_basis_difference_but_a_variation_is_not_a_contract_value(session, room):  # noqa: F811
    contract = claim_in(session, "msa.pdf", "The contract value is GBP 2.5 million, excluding VAT.", "GBP 2.5 million", subject="contract", metric="contract value")
    final = claim_in(session, "close.pdf", "Final contract value is GBP 2.65 million excluding VAT.", "GBP 2.65 million", subject="Final contract", metric="final contract value")
    variation = claim_in(session, "close2.pdf", "Final contract value, including approved variations of GBP 150,000, is GBP 2.65 million.", "GBP 150,000", subject="Final contract", metric="approved variations")
    assert judge(session, room, MockLLMClient([])).run([candidate(contract, final)]).groups_basis == 1
    llm = MockLLMClient([{"verdicts": []}] * 20)
    stats = judge(session, room, llm).run([candidate(contract, variation)])  # different attributes: the model reads them
    assert stats.groups_basis == 0 and llm.prompts


def test_a_yes_no_and_a_date_are_never_a_basis_difference(session, room):  # noqa: F811
    none = claim_in(session, "board.pdf", "No subcontractors are engaged on the project, and the final report is due.", "No", subject="project", metric="completion", category="state")
    done = claim_in(session, "mail.pdf", "Practical completion was achieved on 08/07/2025.", "08/07/2025", subject="project", metric="completion", category="date")
    llm = MockLLMClient([{"verdicts": []}] * 20)
    stats = judge(session, room, llm).run([candidate(none, done)])
    assert stats.groups_basis == 0 and llm.prompts


def test_a_document_that_says_it_supersedes_another_makes_their_difference_a_change_of_edition(session, room):  # noqa: F811
    old = claim_in(session, "p3a.pdf", "DOC-03 | Progress Report No. 3 | Revision A | dated 30 April 2025. Racking installed to date: 1,840 bays.", "1,840 bays",
                   subject="Racking", metric="installed to date")
    new = claim_in(session, "p3b.pdf", "DOC-07 | Progress Report No. 3 | Revision B | dated 20 May 2025. This Revision B supersedes Progress Report No. 3 Revision A dated "
                   "30 April 2025. Corrected position: racking installed to date is 1,480 bays.", "1,480 bays", subject="racking", metric="installed to date")
    llm = MockLLMClient([])
    stats = judge(session, room, llm).run([candidate(old, new)])
    finding = session.scalars(select(Finding)).one()
    assert llm.prompts == [] and stats.groups_superseded == 1 and finding.classification == "contextual_difference" and len(session.scalars(select(Evidence)).all()) == 3


def test_a_sentence_that_says_a_document_does_not_replace_another_names_no_supersession(session, room):  # noqa: F811
    add_doc(session, "msa.pdf", "pdf", [dict(content="DOC-01 | Master Services Agreement | Version 1.0 | dated 3 February 2025.", page_number=1)])
    add_doc(session, "board.pdf", "pdf", [dict(content="DOC-04 | Board Update. This summary does not replace the Master Services Agreement, purchase orders or reports.", page_number=1)])
    assert superseded_documents(session) == []


def test_claims_that_only_name_their_own_document_are_never_compared(session, room):  # noqa: F811
    a = claim_in(session, "a.pdf", "DOC-03 | Revision A | dated 30 April 2025.", "Revision A", subject="Progress Report No. 3", metric="revision", category="identifier")
    b = claim_in(session, "b.pdf", "DOC-07 | Revision B | dated 20 May 2025.", "Revision B", subject="Progress Report No. 3", metric="revision", category="identifier")
    llm = MockLLMClient([])
    stats = judge(session, room, llm).run([candidate(a, b)])
    assert llm.prompts == [] and stats.groups_labels == 1 and session.scalar(select(Finding)) is None


def test_amounts_in_two_currencies_are_insufficient_evidence_decided_without_the_model(session, room):  # noqa: F811
    a = claim_in(session, "inv.pdf", "Amount: EUR 450,000 excluding VAT.", "EUR 450,000", metric="invoice amount")
    b = claim_in(session, "ap.pdf", "Invoice recorded at GBP 384,000.", "GBP 384,000", metric="invoice amount")
    llm = MockLLMClient([])
    stats = judge(session, room, llm).run([candidate(a, b)])
    assert llm.prompts == [] and stats.groups_currency == 1 and session.scalars(select(Finding)).one().classification == "insufficient_evidence"


def test_a_hedged_value_is_read_by_the_judge_not_trusted_as_equal(session, room):  # noqa: F811
    a = claim_in(session, "board.pdf", "Payments to date total about GBP 0.5 million.", "about GBP 0.5 million", metric="payments")
    b = claim_in(session, "ap.pdf", "Payments made to date: GBP 512,400.", "GBP 512,400", metric="payments")
    quotes = {"about GBP 0.5 million": "Payments to date total about GBP 0.5 million.", "GBP 512,400": "Payments made to date: GBP 512,400."}
    llm = MockLLMClient([answer(verdict([a, b], cls="consistent", quote=lambda v: quotes[v]))])
    stats = judge(session, room, llm, save_consistent=True).run([candidate(a, b)])
    assert stats.groups_agreed == 0 and len(llm.prompts) == 1


def test_one_difference_stated_against_two_documents_is_one_finding(session, room):  # noqa: F811
    contract = claim_in(session, "msa.pdf", "The contract value is GBP 2.5 million, excluding VAT.", "GBP 2.5 million", subject="contract", metric="contract value")
    order = claim_in(session, "po.pdf", "The contract value on the order is GBP 2,500 thousand.", "GBP 2,500 thousand", subject="contract", metric="contract value")
    final = claim_in(session, "close.pdf", "Final contract value is GBP 2.65 million excluding VAT.", "GBP 2.65 million", subject="Final contract", metric="final contract value")
    stats = judge(session, room, MockLLMClient([])).run([candidate(contract, final, how="embedding"), candidate(order, final, how="embedding")])
    assert stats.groups_total == 1 and len(session.scalars(select(Finding)).all()) == 1 and len(session.scalars(select(Evidence)).all()) == 3  # the three claims are one cluster


# ---- inside one document, by patterns ------------------------------------------------------------------------------------------------
def test_patterns_find_a_part_larger_than_its_whole_an_average_that_misses_its_total_and_an_ambiguous_change():
    text = ("Racking installed to date: 1,840 bays out of the 1,600 bay positions in the warehouse layout.\n" + SPEND + "\n"
            "Gross margin on the warehouse programme rose 2% in the quarter, from 18.0% to 20.0%.")
    found = {f.kind: f for f in find_relations(text)}
    assert set(found) == {"part_of_whole", "average_total", "change"}
    assert found["part_of_whole"].classification == "contradiction" and found["average_total"].classification == "contradiction"
    assert found["change"].classification == "insufficient_evidence" and all(q in text for f in found.values() for q in f.quotes)


def test_patterns_say_nothing_about_figures_that_agree():
    text = ("Racking installed to date: 1,480 bays out of the 1,600 bay positions.\n"
            "Average monthly spend is GBP 450,000 across March and April, giving total spend of GBP 900,000.\n"
            "Gross margin rose 11% from 18.0% to 20.0%, and revenue rose 20% from 100 to 120.")
    assert find_relations(text) == []


def test_periods_are_counted_from_the_words():
    assert [period_count(t) for t in ("across March and April", "from March to May", "for 6 months", "in Q1 and Q2", "in 2023 and 2024", "in the quarter")] == [2, 3, 6, 2, 2, None]


def test_a_pattern_saves_a_finding_with_a_quote_for_each_figure_and_the_model_does_not_save_it_again(session, room):  # noqa: F811
    doc, chunks = add_doc(session, "board.pdf", "pdf", [dict(content=SPEND, page_number=1)])
    for value, metric in (("GBP 410,000", "average monthly spend"), ("GBP 900,000", "total spend")):
        session.add(Claim(document_id=doc.id, chunk_id=chunks[0].id, subject="Harbor Point", metric=metric, value=value, claim_text=SPEND, claim_category="quantity"))
    session.commit()
    relation = {"label": "monthly spend times periods", "terms": [{"ref": "c1", "times": 2}], "relation": "equals", "against": "c2"}
    stats = InternalChecker(session, MockLLMClient([{"checks": [relation]}]), build_registry(session, None, room.investigation.id)).run()
    assert stats.patterns == 1 and stats.saved == 1 and len(session.scalars(select(Finding)).all()) == 1 and len(session.scalars(select(Evidence)).all()) == 2


def test_the_model_reads_one_document_at_a_time(session, room):  # noqa: F811
    for name in ("one.pdf", "two.pdf"):
        doc, chunks = add_doc(session, name, "pdf", [dict(content=f"{name}: the plant has 5 lines and 7 shifts.", page_number=1)])
        session.add_all([Claim(document_id=doc.id, chunk_id=chunks[0].id, subject="plant", metric=m, value=v, claim_text=f"{name}: the plant has 5 lines and 7 shifts.", claim_category="quantity")
                         for m, v in (("lines", "5"), ("shifts", "7"))])
    session.commit()
    llm = MockLLMClient([{"checks": []}, {"checks": []}])
    stats = InternalChecker(session, llm, build_registry(session, None, room.investigation.id)).run()
    assert stats.calls == 2 and len(llm.prompts) == 2 and all(p.count("### Document:") == 1 for p in llm.prompts)


def test_a_metric_word_stays_rare_even_when_many_subjects_repeat_it():
    views = [view(n, 1, f"Purchase Order PO-5512{n}", "value", "85,000") for n in range(1, 8)]
    views += [view(20, 2, "Progress Report No. 3", "purchase order", "PO-55210", "identifier"), view(21, 3, "Invoice INV-7781", "purchase order", "PO-55120", "identifier")]
    assert frozenset({"c20", "c21"}) in pairs(shared_label_candidates(views))


def test_a_balance_of_nil_is_a_quantity_not_a_negation():
    assert negated_terms(view(1, 1, "Northwind", "outstanding balance owed", "nil", "state", "Outstanding balance owed to Northwind: nil.")) == frozenset()


def test_claims_whose_metrics_are_word_for_word_the_same_are_linked_even_when_the_word_is_used_elsewhere():
    views = [view(n, 1, f"Purchase Order {n}", "purchase order recipient", "Northwind", "name") for n in range(1, 7)]
    views += [view(20, 2, "Progress Report", "purchase order", "PO-55210", "identifier"), view(21, 3, "Invoice", "purchase order", "PO-55120", "identifier")]
    assert frozenset({"c20", "c21"}) in pairs(shared_label_candidates(views))


def test_a_day_month_date_the_documents_do_not_settle_is_insufficient_evidence_not_agreement(session, room):  # noqa: F811
    mail = claim_in(session, "mail.pdf", "Practical completion was achieved on 08/07/2025.", "08/07/2025", subject="practical completion", metric="practical completion date", category="date")
    cert = claim_in(session, "cert.pdf", "Practical completion occurred in July 2025.", "July 2025", subject="practical completion", metric="date", category="date")
    quotes = {"08/07/2025": "Practical completion was achieved on 08/07/2025.", "July 2025": "Practical completion occurred in July 2025."}
    judge(session, room, MockLLMClient([answer(verdict([mail, cert], cls="consistent", quote=lambda v: quotes[v]))]), save_consistent=True).run([candidate(mail, cert)])
    finding = session.scalars(select(Finding)).one()
    assert finding.classification == "insufficient_evidence" and "8 July 2025" in finding.description and "7 August 2025" in finding.description


def test_a_percentage_with_no_base_is_missing_information_even_when_the_model_says_insufficient_evidence(session, room):  # noqa: F811
    pct = claim_in(session, "close.pdf", "Defects at handover: 3% of installed bays.", "3%", metric="defects", category="quantity")
    cnt = claim_in(session, "msa.pdf", "Defects at handover shall not exceed 40 bays.", "40 bays", metric="defects limit", category="quantity")
    quotes = {"3%": "Defects at handover: 3% of installed bays.", "40 bays": "Defects at handover shall not exceed 40 bays."}
    judge(session, room, MockLLMClient([answer(verdict([pct, cnt], cls="insufficient_evidence", quote=lambda v: quotes[v]))])).run([candidate(pct, cnt)])
    assert session.scalars(select(Finding)).one().classification == "missing_information"


def test_two_findings_that_quote_the_same_two_passages_and_state_one_figure_are_one(session, room):  # noqa: F811
    first, second = "The contract value is GBP 2.5 million, excluding VAT.", "Final contract value, including approved variations of GBP 150,000, is GBP 2.65 million."
    doc_a, chunks_a = add_doc(session, "msa.pdf", "pdf", [dict(content=first, page_number=1)])
    doc_b, chunks_b = add_doc(session, "close.pdf", "pdf", [dict(content=second, page_number=1)])
    rows = [Claim(document_id=d.id, chunk_id=c[0].id, subject="contract", metric=m, value=v, claim_text=t, claim_category="quantity")
            for d, c, t, m, v in ((doc_a, chunks_a, first, "contract value", "GBP 2.5 million"), (doc_a, chunks_a, first, "VAT exclusion", "excluding VAT"),
                                  (doc_b, chunks_b, second, "final value", "GBP 2.65 million"), (doc_b, chunks_b, second, "approved variations", "GBP 150,000"))]
    session.add_all(rows)
    session.commit()
    contract, vat, final, other = rows  # the second pair is built from other claims of the same two sentences
    q = {"GBP 2.5 million": first, "GBP 2.65 million": second}
    one = verdict([contract, final], cls="contextual_difference", quote=lambda v: q[v], description="GBP 2.5 million in the contract against GBP 2.65 million at the end, explained by variations.")
    two = verdict([vat, other], start=3, group=2, cls="contextual_difference", quote=lambda v: {"excluding VAT": first, "GBP 150,000": second}[v], description="GBP 2.5 million at the start and GBP 2.65 million at the end.")
    stats = judge(session, room, MockLLMClient([answer(one, two)])).run([candidate(contract, final, how="embedding"), candidate(vat, other, how="embedding")])
    assert stats.merged == 1 and len(session.scalars(select(Finding)).all()) == 1


# ---- groups of one attribute ---------------------------------------------------------------------------------------------------------------
def test_a_group_chained_across_attributes_is_split_so_the_real_comparisons_are_judged(session, room):  # noqa: F811
    texts = {"2.5": "The contract value is GBP 2.5 million.", "2.8": "The contract value is GBP 2.8 million.", "150": "The total liability is capped at 150% of the contract value.",
             "120": "The liability cap remains at 120% of the contract value."}
    value = claim_in(session, "msa.pdf", texts["2.5"], "GBP 2.5 million", subject="contract", metric="contract value", unit="GBP")
    other = claim_in(session, "board.pdf", texts["2.8"], "GBP 2.8 million", subject="contract", metric="contract value", unit="GBP")
    cap_a = claim_in(session, "msa2.pdf", texts["150"], "150%", subject="Supplier", metric="total liability cap", unit="%")
    cap_b = claim_in(session, "rev.pdf", texts["120"], "120%", subject="Supplier", metric="agreed cap", unit="%")
    quote = {"GBP 2.5 million": texts["2.5"], "GBP 2.8 million": texts["2.8"], "150%": texts["150"], "120%": texts["120"]}
    first = verdict([value, other], quote=lambda v: quote[v])
    second = verdict([cap_a, cap_b], start=3, group=2, quote=lambda v: quote[v], title="Liability cap differs")
    stats = judge(session, room, MockLLMClient([answer(first, second)])).run(candidates_of([value, other, cap_a, cap_b]))
    assert stats.groups_split == 1 and stats.groups_total == 2 and len(session.scalars(select(Finding)).all()) == 2


def test_a_documents_own_dates_are_never_compared_or_called_consistent_or_superseded(session, room):  # noqa: F811
    a = claim_in(session, "p3a.pdf", "DOC-03 | Progress Report No. 3 | Revision A | dated 30 April 2025.", "30 April 2025", subject="Progress Report No. 3", metric="date", category="date")
    b = claim_in(session, "p3b.pdf", "DOC-07 | Progress Report No. 3 | Revision B | dated 20 May 2025. This Revision B supersedes Progress Report No. 3 Revision A dated 30 April 2025.",
                 "20 May 2025", subject="DOC-07", metric="date", category="date")
    llm = MockLLMClient([])
    stats = judge(session, room, llm).run([candidate(a, b)])
    assert llm.prompts == [] and stats.groups_labels == 1 and stats.groups_superseded == 0 and session.scalar(select(Finding)) is None


def test_a_record_that_meets_the_rules_own_condition_is_consistent_not_a_contextual_difference(session, room):  # noqa: F811
    rule = claim_in(session, "msa.pdf", "The Supplier shall not use subcontractors without the Customer's prior written consent.", "shall not use subcontractors without the Customer's prior written consent",
                    subject="Supplier", metric="subcontractor condition", category="statement")
    record = claim_in(session, "report.pdf", "Brightline Steel Ltd, our installation subcontractor, works under Calder's written consent dated 20 February 2025.",
                      "our installation subcontractor", subject="Brightline Steel Ltd", metric="role", category="statement")
    quotes = {rule.value: rule.claim_text, record.value: record.claim_text}
    one = verdict([rule, record], cls="contextual_difference", quote=lambda v: quotes[v], description="Brightline works as a subcontractor under Calder's written consent, which the rule requires.")
    judge(session, room, MockLLMClient([answer(one)]), save_consistent=True).run([candidate(rule, record, how="rule")])
    assert session.scalars(select(Finding)).one().classification == "consistent"


def test_a_contradiction_and_a_contextual_difference_about_one_difference_are_one_finding_the_one_that_names_the_context(session, room):  # noqa: F811
    first, second = "The contract value is GBP 2.5 million, excluding VAT.", "Final contract value, including approved variations of GBP 150,000, is GBP 2.65 million."
    doc_a, chunks_a = add_doc(session, "msa.pdf", "pdf", [dict(content=first, page_number=1)])
    doc_b, chunks_b = add_doc(session, "close.pdf", "pdf", [dict(content=second, page_number=1)])
    rows = [Claim(document_id=d.id, chunk_id=c[0].id, subject="contract", metric=m, value=v, claim_text=t, claim_category="quantity")
            for d, c, t, m, v in ((doc_a, chunks_a, first, "contract value", "GBP 2.5 million"), (doc_a, chunks_a, first, "VAT exclusion", "excluding VAT"),
                                  (doc_b, chunks_b, second, "final value", "GBP 2.65 million"), (doc_b, chunks_b, second, "approved variations", "GBP 150,000"))]
    session.add_all(rows)
    session.commit()
    contract, vat, final, variation = rows
    q = {"GBP 2.5 million": first, "GBP 2.65 million": second}
    wrong = verdict([contract, final], cls="contradiction", quote=lambda v: q[v], description="GBP 2.5 million against GBP 2.65 million.")
    right = verdict([vat, variation], start=3, group=2, cls="contextual_difference", quote=lambda v: {"excluding VAT": first, "GBP 150,000": second}[v],
                    description="GBP 2.5 million at the start and GBP 2.65 million at the end, because of approved variations of GBP 150,000.")
    judge(session, room, MockLLMClient([answer(wrong, right)])).run([candidate(contract, final, how="embedding"), candidate(vat, variation, how="embedding")])
    assert [f.classification for f in session.scalars(select(Finding)).all()] == ["contextual_difference"]


def test_a_reference_to_a_document_named_in_a_header_is_found_when_its_name_is_spread_over_the_header(session, room):  # noqa: F811
    from app.services.missing_items import DocReference, MissingItemChecker
    add_doc(session, "p3a.docx", "docx", [dict(content="DOC-03 | Progress Report No. 3 | Revision A | dated 30 April 2025. Works commenced on 3 March 2025.")])
    doc, chunks = add_doc(session, "p3b.docx", "docx", [dict(content="DOC-07 | Progress Report No. 3 | Revision B. This Revision B supersedes Progress Report No. 3 Revision A dated 30 April 2025.")])
    ref = DocReference(doc.id, chunks[0].id, "Progress Report No. 3 Revision A", "This Revision B supersedes Progress Report No. 3 Revision A dated 30 April 2025.")
    stats = MissingItemChecker(session, build_registry(session, None, room.investigation.id)).run([ref])
    assert stats.present == 1 and stats.saved == 0


def test_severity_is_critical_for_a_large_gap_when_the_rest_of_the_rule_group_is_an_obligation_and_for_a_total_that_misses_its_parts():
    from app.services.severity import contradiction_severity
    claims = [view(1, 1, "project", "subcontractors engaged", "No", "state", "No subcontractors are engaged on the project."),
              view(2, 2, "Brightline", "role", "our installation subcontractor", "statement", "Brightline is our installation subcontractor.")]
    claims = [type("C", (), dict(subject=v.subject, metric=v.metric, value=v.value, unit=None, claim_text=v.claim_text, claim_category=v.value_type)) for v in claims]
    obligation = type("C", (), dict(subject="Supplier", metric="condition", value="x", unit=None, claim_text="The Supplier shall not use subcontractors.", claim_category="statement"))
    assert contradiction_severity(claims) == "high" and contradiction_severity(claims, [obligation]) == "critical"
    found = {f.kind: f for f in find_relations(SPEND)}
    assert found["average_total"].severity == "critical"


# ---- the judge reads the extraction call's tags ---------------------------------------------------------------------------------------
def test_the_basis_the_extraction_call_names_decides_a_planned_and_an_actual_figure_without_the_model(session, room):  # noqa: F811
    plan = claim_in(session, "plan.pdf", "Start: 1 March 2025.", "1 March 2025", subject="works", metric="start date", category="date", assumptions=["@basis: planned"])
    actual = claim_in(session, "log.pdf", "Start: 3 March 2025.", "3 March 2025", subject="works", metric="start date", category="date", assumptions=["@basis: actual"])
    llm = MockLLMClient([])
    stats = judge(session, room, llm).run([candidate(plan, actual)])
    assert llm.prompts == [] and stats.groups_basis == 1


def test_a_claim_the_extraction_call_marks_as_about_its_document_is_never_compared_with_content(session, room):  # noqa: F811
    a = claim_in(session, "a.pdf", "Signed by A. Rao.", "A. Rao", subject="agreement", metric="signatory", category="name", assumptions=["@about: document"])
    b = claim_in(session, "b.pdf", "Signed by B. Khan.", "B. Khan", subject="agreement", metric="signatory", category="name", assumptions=["@about: document"])
    llm = MockLLMClient([])
    stats = judge(session, room, llm).run([candidate(a, b)])
    assert llm.prompts == [] and stats.groups_labels == 1


def test_a_vague_or_approximate_value_is_never_agreed_in_code(session, room):  # noqa: F811
    a = claim_in(session, "a.pdf", "Spend is broadly 500 units.", "500 units", metric="spend", assumptions=["@precision: vague"])
    b = claim_in(session, "b.pdf", "Spend is 500 units.", "500 units", metric="spend")
    llm = MockLLMClient([{"verdicts": []}] * 20)
    stats = judge(session, room, llm).run([candidate(a, b)])
    assert stats.groups_agreed == 0 and llm.prompts
