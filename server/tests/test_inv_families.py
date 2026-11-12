"""The check families build their tasks and deterministic candidates from the facts, the tables and the case model."""
from app.investigator import families as fam
from app.investigator.case_model import CaseModelBuilder, CaseModelDraft
from app.investigator.models import ReferenceKinds
from app.investigator.tools import build_timeline
from app.services.missing_items import MissingItemChecker
from tests.inv_support import RoutedLLM, build_store, dive_room, make_fact, xlsx_chunk
from tests.test_inv_case_tools_workers import draft
from tests.test_tools import add_doc, room  # noqa: F401  (fixture and helper)


def prepared(session, tmp_path):
    dive_room(session)
    add_doc(session, "samples.xlsx", "xlsx", [xlsx_chunk("Samples", {1: [(None, "Sample"), (None, "Weight")], **{i + 1: [("Sample", f"SPL-{i:03d}"), ("Weight", 10 + i)] for i in range(1, 7)}})])
    add_doc(session, "received.xlsx", "xlsx", [xlsx_chunk("Received", {1: [(None, "Dive"), (None, "Received")], **{i + 1: [("Dive", f"D-{i:02d}"), ("Received", f"2025-03-{10 + i:02d}")] for i in range(1, 7)}})])
    session.commit()
    store = build_store(session, tmp_path)
    builder = CaseModelBuilder(store, RoutedLLM({CaseModelDraft: draft()}))
    case = builder.build()
    builder.scan(case)
    return store, case


def test_every_constraint_becomes_a_task_with_its_rule_and_source(session, tmp_path):
    store, case = prepared(session, tmp_path)
    rule = make_fact(store, "permit.docx", "No more than 8 fragments may be collected per site.", "constraint", subject="site", attribute="fragments collected", operator="<=", value="8", modality="must")
    plan = fam.Plan()
    fam.family_a([rule], store, plan, 10)
    assert len(plan.tasks) == 1 and plan.tasks[0].family == "A" and "No more than 8 fragments" in plan.tasks[0].context and "permit.docx" in plan.tasks[0].context


def test_the_per_family_limit_is_logged_not_silent(session, tmp_path):
    store, case = prepared(session, tmp_path)
    rules = [make_fact(store, "permit.docx", "No more than 8 fragments may be collected per site.", "constraint", attribute=f"a{i}") for i in range(4)]
    plan = fam.Plan()
    fam.family_a(rules, store, plan, 2)
    assert len(plan.tasks) == 2 and plan.records[0].status == "skipped" and "2 more" in plan.records[0].title


def test_identifier_sets_that_differ_between_documents_become_a_reconciliation_task(session, tmp_path):
    store, case = prepared(session, tmp_path)
    plan = fam.Plan()
    fam.family_b(store, case, plan, 10)
    task = next(t for t in plan.tasks if "sample" in t.title)
    assert "SPL044B" in task.context and "lab.pdf" in task.title and "samples.xlsx" in task.title and "SPL002" in task.context


def test_two_dated_tables_about_the_same_entities_get_an_ordering_task(session, tmp_path):
    store, case = prepared(session, tmp_path)
    plan = fam.Plan()
    fam.family_c([], store, plan, 10)
    task = next(t for t in plan.tasks if t.family == "C")
    assert "register.xlsx" in task.title and "received.xlsx" in task.title and "days_between" in task.instructions and len(task.tables) == 2


def test_stated_aggregates_and_universal_claims_become_recomputation_and_counterexample_tasks(session, tmp_path):
    store, case = prepared(session, tmp_path)
    mean = make_fact(store, "lab.pdf", "Mean sample count per dive was 5.7.", "observation", subject="dives", attribute="mean sample count per dive", value="5.7", aggregate_of="samples per dive")
    definition = make_fact(store, "lab.pdf", "Mean sample count per dive was 5.7.", "definition", subject="mean sample count", attribute="mean sample count per dive", value="samples divided by dives")
    none_claim = make_fact(store, "lab.pdf", "Samples received: SPL-001 to SPL-006 and SPL-044-B. Sample SPL-044-B arrived without field metadata.", "assertion",
                           subject="samples", attribute="metadata", value="all samples have field metadata")
    plan = fam.Plan()
    fam.family_d([mean, definition, none_claim], store, plan, 10)
    fam.family_e([mean, none_claim], store, plan, 10)
    d = [t for t in plan.tasks if t.family == "D"]
    e = [t for t in plan.tasks if t.family == "E"]
    assert len(d) == 1 and "samples divided by dives" in d[0].context and len(e) == 1 and "counterexample" in e[0].instructions


def test_conclusions_are_audited(session, tmp_path):
    store, case = prepared(session, tmp_path)
    c = make_fact(store, "lab.pdf", "Mean sample count per dive was 5.7.", "conclusion", subject="reef", attribute="recovery", value="the reef is recovering")
    plan = fam.Plan()
    fam.family_h([c], store, plan, 10)
    assert plan.tasks[0].family == "H" and "like for like" in plan.tasks[0].instructions


def test_identical_series_and_repeated_keys_are_found_by_code_with_row_evidence(session, tmp_path):
    values = [1.5, 2.5, 4.0, 3.5, 7.25, 6.0, 9.5, 8.0, 5.5, 4.5]
    for name in ("logger_a.xlsx", "logger_b.xlsx"):
        add_doc(session, name, "xlsx", [xlsx_chunk("L", {1: [(None, "Temp"), (None, "Id")], **{i + 2: [("Temp", v), ("Id", f"K{i:02d}" if name == "logger_a.xlsx" else f"K{min(i, 8):02d}")] for i, v in enumerate(values)}})])
    session.commit()
    store = build_store(session, tmp_path)
    plan = fam.Plan()
    fam.family_f(store, plan, 10)
    titles = [c.title for c in plan.candidates]
    assert any("Identical data series" in t for t in titles) and any("used for more than one row" in t for t in titles)
    series = next(c for c in plan.candidates if "Identical" in c.title)
    assert series.family == "F" and len(series.evidence) >= 2 and {e.filename for e in series.evidence} == {"logger_a.xlsx", "logger_b.xlsx"}
    assert any(t.family == "F" for t in plan.tasks)


def test_a_counter_column_is_not_an_identical_series(session, tmp_path):
    for name in ("a.xlsx", "b.xlsx"):
        add_doc(session, name, "xlsx", [xlsx_chunk("S", {1: [(None, "N"), (None, "X")], **{i + 1: [("N", i), ("X", i * i)] for i in range(1, 11)}})])
    session.commit()
    plan = fam.Plan()
    fam.family_f(build_store(session, tmp_path), plan, 10)
    assert not any("Identical" in c.title and "N" in c.computed for c in plan.candidates)


def test_references_are_resolved_in_a_fixed_order_and_only_expected_artifacts_are_reported(session, tmp_path):
    store, case = prepared(session, tmp_path)
    names = {d_id: name for d_id, name in store.conn.execute("SELECT DISTINCT document_id, filename FROM _chunks")}
    pool = MissingItemChecker(session, None)._pool(names)
    log = make_fact(store, "permit.docx", "The decontamination log must be submitted with the final report.", "reference", value="decontamination log")
    reg = make_fact(store, "permit.docx", "No more than 8 fragments may be collected per site.", "reference", value="register")
    std = make_fact(store, "permit.docx", "Sampling is permitted between 3 March 2025 and 21 March 2025.", "reference", value="NOAA Coral Reef Watch product")
    llm = RoutedLLM({ReferenceKinds: {"items": [{"number": 1, "kind": "external_standard", "reason": "a public product"}, {"number": 0, "kind": "expected_artifact"}]}})
    plan = fam.Plan()
    fam.family_g([log, reg, std], store, case, pool, llm, plan, 10)
    assert [c.title for c in plan.candidates] == ["Referenced item not in the data room: decontamination log"]
    assert plan.candidates[0].type == "missing_evidence" and any("NOAA" in q and "external_standard" in q for q in plan.quiet)
    assert llm.count(ReferenceKinds) == 1  # the declared absence ("must be submitted" is not one; only 'not provided' style wording is) goes to the model with the standard
    assert any(r.task_id == "G-code" and "1 found" in r.reason for r in plan.records)


def test_an_identifier_named_elsewhere_resolves_a_reference(session, tmp_path):
    store, case = prepared(session, tmp_path)
    ref = make_fact(store, "permit.docx", "The decontamination log must be submitted with the final report.", "reference", value="sample SPL-044-B")
    plan = fam.Plan()
    fam.family_g([ref], store, case, [], RoutedLLM({}), plan, 10)
    assert plan.candidates == [] and any("1 found" in r.reason for r in plan.records)


def test_an_identifier_outside_the_range_of_another_documents_list_is_a_code_candidate_with_both_quotes(session, tmp_path):
    store, case = prepared(session, tmp_path)
    plan = fam.Plan()
    fam.family_b(store, case, plan, 10)
    gap = next(c for c in plan.candidates if c.family == "B")
    assert gap.type == "reconciliation_gap" and "SPL044B" in gap.title and gap.source == "code"
    assert {e.filename for e in gap.evidence} == {"lab.pdf", "samples.xlsx"} and any("SPL-044-B" in e.quote for e in gap.evidence)
