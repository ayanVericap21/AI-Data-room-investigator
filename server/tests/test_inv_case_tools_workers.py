"""The case model (shapes, normalisation, aliases, mentions), the tools, and the worker loop."""
import json

from app.investigator.case_model import (CaseModelBuilder, CaseModelDraft, candidate_shapes, deterministic_aliases, find_entity, ids_by_document,
                                         normalise_ident, regex_is_safe, shape_of, shape_regex)
from app.investigator.models import (CaseModel, EvidenceRef, Fact, IdPattern, ToolCall, WorkerOutcome, WorkerPlan)
from app.investigator.tools import ToolBox, build_timeline
from app.investigator.workers import WorkerRunner, WorkerTask
from tests.inv_support import RoutedLLM, build_store, dive_room
from tests.test_tools import room  # noqa: F401  (fixture)


def draft():
    return CaseModelDraft(
        case_summary="A sampling survey.",
        roles=[{"document": "permit.docx", "role": "authority", "summary": "Sets limits."}, {"document": "register.xlsx", "role": "primary record"}],
        shape_labels=[{"shape": "SPL-#", "entity_type": "sample"}, {"shape": "D-#", "entity_type": "dive"}, {"shape": "IR-#", "entity_type": "site"}],
        aliases=[{"canonical": "Tomas Varga", "entity_type": "person", "aliases": ["TV"]}])


def test_shapes_group_identifiers_by_family():
    assert shape_of("CF-IR06-03") == "CF-IR#-#" and shape_of("SPL-044-B") == "SPL-#" and shape_of("SPL-001") == "SPL-#"
    assert shape_regex("SPL-#").startswith("(?<![A-Za-z0-9])SPL")


def test_candidate_shapes_are_found_in_the_text(session, tmp_path):
    dive_room(session)
    store = build_store(session, tmp_path)
    shapes = {s["shape"]: s for s in candidate_shapes(store)}
    assert {"SPL-#", "D-#", "IR-#"} <= set(shapes) and shapes["SPL-#"]["documents"] == ["lab.pdf"]


def test_identifiers_are_normalised_so_spellings_match():
    p = IdPattern(entity_type="sample", regex="x", pad_last_number_to=2)
    assert normalise_ident("CF-IR06-3", p) == normalise_ident("cf-ir06-03", p) == "CFIR0603"


def test_unsafe_or_broken_patterns_are_refused():
    assert regex_is_safe("(a+)+$") is None and regex_is_safe("([") is None and regex_is_safe("x" * 300) is None and regex_is_safe(r"SPL-\d+") is not None


def test_aliases_are_grouped_by_surname_and_initial():
    groups = deterministic_aliases(["Dr Helena Marsh", "H. Marsh", "Tomas Varga", "Marsh"])
    assert len(groups) == 1 and groups[0].canonical == "Helena Marsh" and "H Marsh" in groups[0].aliases


def test_the_case_model_scans_every_chunk_for_identifiers_and_aliases(session, tmp_path):
    dive_room(session)
    store = build_store(session, tmp_path)
    builder = CaseModelBuilder(store, RoutedLLM({CaseModelDraft: draft()}))
    result = builder.build()
    assert {p.entity_type for p, _ in result.patterns} >= {"sample", "dive", "site"} and result.role_of("permit.docx") == "authority"
    builder.scan(result)
    samples = ids_by_document(store, "sample")
    assert set(samples["lab.pdf"]) == {"SPL001", "SPL006", "SPL044B"}
    assert [h["filename"] for h in find_entity(store, "spl-044-b")] == ["lab.pdf"]
    out = tmp_path / "case.json"
    builder.save(result, out)
    saved = json.loads(out.read_text(encoding="utf-8"))
    assert saved["case_summary"] == "A sampling survey." and saved["shapes_seen"]


def test_the_case_model_still_works_when_the_model_call_fails(session, tmp_path):
    dive_room(session)
    store = build_store(session, tmp_path)
    result = CaseModelBuilder(store, RoutedLLM({})).build()
    assert result.patterns and any("failed" in n for n in result.notes)


def toolbox(session, tmp_path, facts=None):
    dive_room(session)
    store = build_store(session, tmp_path)
    builder = CaseModelBuilder(store, RoutedLLM({CaseModelDraft: draft()}))
    builder.scan(builder.build())
    return store, ToolBox(store, facts or [])


def table_name(store):
    return next(t.name for t in store.table_list() if t.filename == "register.xlsx")


def test_tools_run_sql_search_read_entity_and_timeline(session, tmp_path):
    store, tb = toolbox(session, tmp_path)
    t = table_name(store)
    assert "D-04" in tb.run(ToolCall(tool="sql", sql=f'SELECT dive FROM "{t}" WHERE samples > 8'))
    assert tb.run(ToolCall(tool="sql", sql="DROP TABLE _facts")).startswith("SQL error")
    assert "decontamination" in tb.run(ToolCall(tool="search", query="decontamination log"))
    assert "Samples received" in tb.run(ToolCall(tool="read", document="lab.pdf", location="page 1"))
    assert "lab.pdf" in tb.run(ToolCall(tool="find_entity", entity="SPL-044-B"))
    assert tb.run(ToolCall(tool="nonsense")).startswith("unknown tool")
    assert build_timeline(store, []) == 6
    assert "2025-03-12" in tb.run(ToolCall(tool="timeline", entity="D-04")) and "(no events)" == tb.run(ToolCall(tool="timeline", entity="D-04", window_start="2026-01-01"))


def test_evidence_references_become_exact_chunk_text(session, tmp_path):
    store, tb = toolbox(session, tmp_path)
    t = table_name(store)
    row = tb.resolve(EvidenceRef(kind="row", table=t, ref="A5:D5", shows="the dive", role="contradicting"))
    assert row.filename == "register.xlsx" and "D-04" in row.quote and row.role == "contradicting"
    loose = tb.resolve(EvidenceRef(kind="quote", document="permit.docx", quote="no more than 8 fragments may be collected from each site", shows="the limit"))
    assert loose.quote == "No more than 8 fragments may be collected per site." and "matched approximately" in loose.shows  # the stored text is the document's, not the model's
    assert tb.resolve(EvidenceRef(kind="quote", document="permit.docx", quote="a sentence about something else entirely, with many other words")) is None
    quote = tb.resolve(EvidenceRef(kind="quote", document="permit.docx", quote="No more than 8 fragments may be collected per site."))
    assert quote.quote == "No more than 8 fragments may be collected per site." and quote.location == "section 'Conditions'"
    assert tb.resolve(EvidenceRef(kind="quote", document="wrong.pdf", quote="Mean sample count per dive was 5.7.")).filename == "lab.pdf"  # a wrong file name is searched past
    assert tb.resolve(EvidenceRef(kind="row", table=t, ref="Z99")) is None and tb.resolve(EvidenceRef(kind="quote", quote="invented text that is nowhere")) is None


def rule_task(store):
    return WorkerTask("A1", "A", "Rule: dates within 3 to 21 March", "Test the dives against the window.", tables={table_name(store)})


def runner(store, tb, llm, **kw):
    return WorkerRunner(llm, store, tb, **kw)


def test_a_worker_plans_runs_the_calls_and_returns_a_candidate_with_verified_evidence(session, tmp_path):
    store, tb = toolbox(session, tmp_path)
    t = table_name(store)
    seen = {}

    def conclude(prompt):
        seen["prompt"] = prompt
        return {"results": [{"task_id": "A1", "status": "finding", "candidates": [{
            "type": "constraint_violation", "title": "Dive D-03 is after the permit window", "description": "D-03 is dated 22 March; the permit ends on 21 March.", "severity": "high", "confidence": 0.8,
            "evidence": [{"kind": "row", "table": t, "ref": "A4:D4", "shows": "dive D-03 dated 2025-03-22", "role": "contradicting"},
                         {"kind": "quote", "document": "permit.docx", "quote": "Sampling is permitted between 3 March 2025 and 21 March 2025.", "shows": "the window", "role": "supporting"}],
            "computed": "date > 2025-03-21 gives D-03", "entities": ["D-03"]}]}]}

    llm = RoutedLLM({WorkerPlan: {"note": "n", "calls": [{"task_id": "A1", "tool": "sql", "sql": f"SELECT dive, date, _ref FROM \"{t}\" WHERE to_date(date) > '2025-03-21'"}]},
                     WorkerOutcome: conclude})
    candidates, records = runner(store, tb, llm).run([rule_task(store)])
    assert "D-03" in seen["prompt"] and len(candidates) == 1 and candidates[0].family == "A" and len(candidates[0].evidence) == 2
    assert candidates[0].evidence[0].filename == "register.xlsx" and records[0].status == "finding" and records[0].calls == 1


def test_a_candidate_whose_evidence_cannot_be_found_is_dropped(session, tmp_path):
    store, tb = toolbox(session, tmp_path)
    bad = {"results": [{"task_id": "A1", "status": "finding", "candidates": [{"type": "constraint_violation", "title": "x", "description": "y",
                                                                               "evidence": [{"kind": "quote", "quote": "made up words that no document holds"}]}]}]}
    r = runner(store, tb, RoutedLLM({WorkerPlan: {"calls": []}, WorkerOutcome: bad}))
    candidates, records = r.run([rule_task(store)])
    assert candidates == [] and r.stats.dropped_candidates == 1 and records[0].status == "untestable" and "could be found" in records[0].reason


def test_passed_untestable_and_unanswered_tasks_are_told_apart(session, tmp_path):
    store, tb = toolbox(session, tmp_path)
    tasks = [WorkerTask(f"A{i}", "A", f"rule {i}", "x") for i in (1, 2, 3)]
    outcome = {"results": [{"task_id": "A1", "status": "passed", "reason": "all comply"}, {"task_id": "A2", "status": "untestable", "reason": "no records"}]}
    candidates, records = runner(store, tb, RoutedLLM({WorkerPlan: {"calls": []}, WorkerOutcome: outcome})).run(tasks)
    status = {r.task_id: (r.status, r.reason) for r in records}
    assert status["A1"] == ("passed", "all comply") and status["A2"][0] == "untestable" and status["A3"][0] == "untestable" and "no result" in status["A3"][1]


def test_needs_more_gets_a_second_round_with_the_earlier_results(session, tmp_path):
    store, tb = toolbox(session, tmp_path)
    t = table_name(store)
    outcomes = [{"results": [{"task_id": "A1", "status": "needs_more"}]}, {"results": [{"task_id": "A1", "status": "passed", "reason": "checked twice"}]}]
    llm = RoutedLLM({WorkerPlan: [{"calls": [{"task_id": "A1", "tool": "sql", "sql": f'SELECT COUNT(*) FROM "{t}"'}]}, {"calls": []}], WorkerOutcome: outcomes})
    candidates, records = runner(store, tb, llm).run([rule_task(store)])
    assert llm.count(WorkerPlan) == 2 and records[0].status == "passed" and "COUNT(*)" in llm.calls[2][1]


def test_a_failing_model_call_marks_the_batch_as_an_error_and_a_stop_skips_it(session, tmp_path):
    store, tb = toolbox(session, tmp_path)
    candidates, records = runner(store, tb, RoutedLLM({})).run([rule_task(store)])
    assert records[0].status == "error"
    candidates, records = runner(store, tb, RoutedLLM({}), should_stop=lambda: True).run([rule_task(store)])
    assert records[0].status == "skipped"


def test_tasks_are_batched_by_family_and_size(session, tmp_path):
    store, tb = toolbox(session, tmp_path)
    r = runner(store, tb, RoutedLLM({}), tasks_per_batch={"A": 2, "D": 2})
    tasks = [WorkerTask(f"A{i}", "A", "t", "x") for i in range(5)] + [WorkerTask("D1", "D", "t", "x")]
    assert [len(b) for b in r.batches(tasks)] == [2, 2, 1, 1]


def test_small_tables_are_shown_in_full_with_their_row_addresses_and_large_ones_as_a_sample(session, tmp_path):
    dive_room(session)
    store = build_store(session, tmp_path)
    text = store.schema_text()
    assert text.count("  row ") == 6 and "[_ref A5:D5]" in text and "D-04 | 2025-03-12 | IR-01 | 9" in text
    assert "e.g." in store.schema_text(full_up_to=3)


def test_a_task_the_worker_did_not_answer_is_asked_again_and_dropped_candidates_are_logged(session, tmp_path):
    store, tb = toolbox(session, tmp_path)
    t = table_name(store)
    unresolvable = {"task_id": "A2", "status": "finding", "candidates": [{"type": "constraint_violation", "title": "Ghost finding", "description": "x",
                                                                           "evidence": [{"kind": "quote", "quote": "nothing like this appears anywhere in the room at all"}]}]}
    outcomes = [{"results": [{"task_id": "A1", "status": "passed", "reason": "ok"}]}, {"results": [unresolvable]}]
    llm = RoutedLLM({WorkerPlan: [{"calls": [{"task_id": "", "tool": "sql", "sql": f'SELECT COUNT(*) FROM "{t}"'}]}, {"calls": []}], WorkerOutcome: outcomes})
    r = runner(store, tb, llm)
    candidates, records = r.run([WorkerTask("A1", "A", "one", "x"), WorkerTask("A2", "A", "two", "x")])
    status = {x.task_id: x for x in records}
    assert llm.count(WorkerPlan) == 2 and status["A1"].status == "passed" and status["A2"].status == "untestable" and status["A1"].calls == 1  # an untagged call is counted
    assert candidates == [] and "Ghost finding" in r.stats.dropped_log[0] and "nothing like this" in r.stats.dropped_log[0]
