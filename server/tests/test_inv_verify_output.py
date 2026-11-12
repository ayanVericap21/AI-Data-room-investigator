"""The verifier decides what is reported; consolidation merges symptoms of one cause; saving maps the taxonomy onto the existing columns."""
from sqlalchemy import select

from app.investigator.models import Candidate, ConsolidationPlan, ResolvedEvidence, VerifierBatch
from app.investigator.output import Consolidator, Saver
from app.investigator.tables import TableStore
from app.investigator.verifier import Verifier
from app.models import Evidence, Finding
from app.tools import build_registry
from tests.inv_support import RoutedLLM, build_store
from tests.test_tools import room  # noqa: F401  (fixture)

DECK = "capacity is 120,000 tonnes per year"
CAPEX = "Capacity | 100,000 tonnes"


def ev(room, which, role="contradicting", shows="states capacity"):  # noqa: F811
    chunk, quote, name = (room.deck_p1, DECK, "deck.pdf") if which == "deck" else (room.capex_current, CAPEX, "capex.xlsx")
    return ResolvedEvidence(chunk, "doc", name, "page 1", quote, role, shows)


def cand(room, type="direct_contradiction", title="Capacity differs", **kw):  # noqa: F811
    base = dict(family="I", type=type, title=title, description="The deck and the plan state different capacities.", severity="high", confidence=0.8, status="definitive",
                evidence_needed=[], evidence=[ev(room, "deck"), ev(room, "capex")], entities=["plant"])
    base.update(kw)
    return Candidate(**base)


def verifier(llm, tmp_path, **kw):
    return Verifier(llm, TableStore(tmp_path / "v.sqlite"), **kw)


def verdicts(*items):
    return {"verdicts": [{"candidate": i, **v} for i, v in enumerate(items, start=1)]}


def test_each_verdict_decides_whether_a_candidate_is_reported(room, tmp_path):  # noqa: F811
    cs = [cand(room, title=f"c{i}") for i in range(5)]
    cs[3].type = "unsupported_conclusion"
    llm = RoutedLLM({VerifierBatch: verdicts({"verdict": "confirmed", "reasons": "stands"}, {"verdict": "contextual_difference", "reasons": "different scope"},
                                             {"verdict": "rejected", "reasons": "wording"}, {"verdict": "unsupported", "reasons": "no backing"})})
    r = verifier(llm, tmp_path, batch_size=4).verify(cs[:4])
    assert [c.title for c in r.kept] == ["c0", "c1", "c3"] and [c.title for c in r.rejected] == ["c2"]
    assert r.kept[1].type == "contextual_difference" and r.kept[1].severity == "low" and r.kept[2].type == "unsupported_conclusion"


def test_an_unsupported_verdict_on_a_contradiction_is_not_reported_and_missing_evidence_is(room, tmp_path):  # noqa: F811
    cs = [cand(room, title="a"), cand(room, title="b")]
    llm = RoutedLLM({VerifierBatch: verdicts({"verdict": "unsupported"}, {"verdict": "missing_evidence", "reasons": "the approval is absent"})})
    r = verifier(llm, tmp_path).verify(cs)
    assert [c.title for c in r.rejected] == ["a"] and r.kept[0].type == "missing_evidence" and r.kept[0].status == "needs_investigation"


def test_a_confirmed_verdict_is_overruled_when_no_quote_supports_the_finding(room, tmp_path):  # noqa: F811
    llm = RoutedLLM({VerifierBatch: verdicts({"verdict": "confirmed", "reasons": "ok", "quotes": [{"index": 1, "supports": False}, {"index": 2, "supports": False}]})})
    r = verifier(llm, tmp_path).verify([cand(room)])
    assert r.kept == [] and "none of the quotes supports" in r.rejected[0].verifier_notes


def test_a_missing_verdict_rejects_and_a_failed_call_leaves_the_candidates_unverified(room, tmp_path):  # noqa: F811
    r = verifier(RoutedLLM({VerifierBatch: verdicts({"verdict": "confirmed"})}), tmp_path, batch_size=2).verify([cand(room, title="a"), cand(room, title="b")])
    assert [c.title for c in r.kept] == ["a"] and r.rejected[0].title == "b"
    r = verifier(RoutedLLM({}), tmp_path).verify([cand(room)])
    assert r.kept[0].verdict == "unverified" and r.kept[0].status == "needs_investigation" and r.kept[0].confidence <= 0.4 and r.stats.unverified == 1


def test_the_verifier_sees_the_evidence_and_not_the_workers_reasoning(room, tmp_path):  # noqa: F811
    llm = RoutedLLM({VerifierBatch: verdicts({"verdict": "confirmed"})})
    c = cand(room, computed="SELECT 1", root_cause="a secret worker rationale")
    verifier(llm, tmp_path).verify([c])
    prompt = llm.calls[0][1]
    assert DECK in prompt and CAPEX in prompt and "SELECT 1" in prompt and "secret worker rationale" not in prompt


def test_exact_duplicates_merge_and_a_group_becomes_one_finding_with_the_symptoms_linked(room):  # noqa: F811
    a = cand(room, title="first", family="A")
    dup = cand(room, title="first again", family="E", confidence=0.5)
    b = cand(room, type="data_integrity_issue", title="second symptom", evidence=[ev(room, "capex")], severity="medium", entities=["logger"], family="F")
    plan = {"groups": [{"members": [1, 2], "title": "One cause", "description": "They share a cause."}]}
    final, stats = Consolidator(RoutedLLM({ConsolidationPlan: plan})).run([a, dup, b])
    assert stats.exact_duplicates == 1 and stats.groups == 1 and len(final) == 1
    assert final[0].title == "One cause" and any("second symptom" in x for x in final[0].linked) and final[0].severity == "high"


def test_a_worker_candidate_is_preferred_as_the_primary_and_a_pairing_finding_is_absorbed(room):  # noqa: F811
    paired = cand(room, title="from pairing", source="pairing", finding_id="fnd_x", confidence=0.99)
    worked = cand(room, type="constraint_violation", title="from a worker", evidence=[ev(room, "deck"), ev(room, "capex")], confidence=0.5)
    final, _ = Consolidator(RoutedLLM({ConsolidationPlan: {"groups": [{"members": [1, 2], "title": "", "description": ""}]}})).run([paired, worked])
    assert len(final) == 1 and final[0].title == "from a worker" and final[0].absorbed == ["fnd_x"]


def test_a_failed_grouping_call_leaves_candidates_separate(room):  # noqa: F811
    final, stats = Consolidator(RoutedLLM({})).run([cand(room, title="a"), cand(room, title="b", evidence=[ev(room, "capex")])])
    assert len(final) == 2 and "failed" in stats.log[0]


def saver(session, room):  # noqa: F811
    return Saver(session, build_registry(session, None, room.investigation.id))


def test_the_taxonomy_maps_onto_the_existing_classification_with_the_label_in_the_title(session, room):  # noqa: F811
    s = saver(session, room)
    items = s.save([cand(room, type="constraint_violation", title="Dive after the window", computed="date > window", evidence_needed=["the amended permit"], status="needs_investigation")])
    f = session.get(Finding, items[0].finding_id)
    assert f.classification == "contradiction" and f.title.startswith("[constraint violation] ") and f.evidence_status == "partial"
    assert "Status: needs investigation" in f.description and "Computed: date > window" in f.description and f.unresolved_questions == ["the amended permit"]
    assert len(session.scalars(select(Evidence).where(Evidence.finding_id == f.id)).all()) == 2


def test_a_finding_with_one_passage_is_not_saved_as_a_contradiction(session, room):  # noqa: F811
    s = saver(session, room)
    items = s.save([cand(room, type="data_integrity_issue", evidence=[ev(room, "deck")]), cand(room, type="missing_evidence", title="Log absent", evidence=[ev(room, "capex", role="insufficient")])])
    assert [i.classification for i in items] == ["insufficient_evidence", "missing_information"]
    assert session.get(Finding, items[1].finding_id).evidence_status == "unavailable"


def test_a_family_i_finding_is_relabelled_and_absorbed_findings_are_removed(session, room):  # noqa: F811
    s = saver(session, room)
    first = s.save([cand(room, title="pair one")])[0]
    second = s.save([cand(room, title="pair two", evidence=[ev(room, "capex")])])[0]
    keep = cand(room, title="pair one", source="pairing", finding_id=first.finding_id, absorbed=[second.finding_id])
    s.save([keep])
    assert session.get(Finding, second.finding_id) is None
    f = session.get(Finding, first.finding_id)
    assert f.title.startswith("[direct contradiction]") and "Check family I" in f.description
