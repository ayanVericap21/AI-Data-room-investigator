"""Unit inside a value, subject gates, wider neighbour matching, echoed values with a retry, and quotes supplied by code."""
from sqlalchemy import select

from app.llm.mock import MockLLMClient
from app.models import Claim, Evidence, Finding
from app.services.candidate_matching import normalize_unit, pair_allowed
from app.services.llm_pairing import LLMPairer
from app.services.pair_judge import SYSTEM, PairJudge
from app.services.value_rules import compare_values
from app.tools import build_registry
from tests.test_claims import view
from tests.test_pair_judge import make_claims, pair, verdict_for
from tests.test_tools import add_doc, room  # noqa: F401  (fixtures and helper)


# ---- 1. a unit written inside the value --------------------------------------
def rel(a, b, ua=None, ub=None):
    return compare_values(a, ua, b, ub, None, None, normalize_unit)


def test_a_value_that_carries_its_own_unit_is_still_a_number():
    assert rel("12,000 WSPM", "12000", "WSPM", "WSPM") == "equal"
    assert rel("24000", "24,000 WSPM", "WSPM", "WSPM") == "equal"
    assert rel("12,000 wafers", "12000 wafers") == "equal"  # the same extra words on both sides
    assert rel("12,000 WSPM", "13,000", "WSPM", "WSPM") == "different"
    assert rel("Phase 2", "2") == "different"  # a word that is not a unit is not ignored on one side only


# ---- 4. subject gates ----------------------------------------------------------
def row(id, doc, subject, table, text=None, chunk=None):
    return view(id, doc=doc, subject=subject, metric="cost", value=1, claim_text=text or f"cost {id}", table_key=table, chunk=chunk)


def test_rows_with_different_identifiers_are_not_paired_but_descriptive_labels_are():
    assert not pair_allowed(row("a", "d1", "MET-004", "d1|S|"), row("b", "d2", "MET-003", "d2|S|"))
    assert pair_allowed(row("a", "d1", "MET-004", "d1|S|"), row("c", "d2", "MET-004", "d2|S|"))
    assert pair_allowed(row("d", "d1", "Aggregate yield", "d1|S|"), row("e", "d2", "Q3 2026 production yield", "d2|S|"))  # labels, not keys
    assert pair_allowed(row("f", "d1", "MET-004", None), row("g", "d2", "MET-003", None))  # not table rows: this rule is not theirs


def test_two_claims_read_from_one_sentence_are_not_a_pair_but_a_repeated_sentence_elsewhere_is():
    same = "SiliconForge is ISO 9001:2015 certified."
    assert not pair_allowed(row("a", "d1", "X", None, same, "chunk1"), row("b", "d1", "Y", None, same, "chunk1"))
    assert pair_allowed(row("a", "d1", "X", None, same, "chunk1"), row("c", "d1", "X", None, same, "chunk2"))


# ---- 5. neighbours: strong top-2, mutual best match, top-5 from a lower floor ----
class FixedEmbedder:
    def __init__(self, vectors):
        self.vectors = vectors

    def embed(self, texts, task="document"):
        return [self.vectors[t.split(": ")[0]] for t in texts]


def claim(id, doc, unit=None):
    return view(id, doc=doc, subject=id, metric="m", value=1, claim_text="x", unit=unit, period=None)


def test_a_best_match_below_the_old_cut_off_is_now_paired_and_a_weak_one_is_not():
    vectors = {"a": [1.0, 0.0, 0.0], "b": [0.6, 0.8, 0.0], "c": [0.3, 0.954, 0.0], "g": [0.45, 0.0, 0.893]}
    views = [claim("a", "d1"), claim("b", "d2"), claim("c", "d2"), claim("g", "d3")]
    pairs: dict = {}
    LLMPairer(None, None, embedder=FixedEmbedder(vectors))._add_neighbors(views, pairs)
    keys = {tuple(sorted(k)) for k in pairs}
    assert ("a", "b") in keys  # 0.60: each other's best match elsewhere, below the old 0.72 cut-off
    assert ("a", "c") not in keys and ("a", "g") not in keys  # 0.30, and 0.45 (below the floor, and not mutual)


def test_the_top_five_route_is_off_unless_asked_for():
    vectors = {"a": [1.0, 0.0, 0.0], "b": [0.6, 0.8, 0.0], "h": [0.58, 0.0, 0.815]}  # a-h 0.58 is a's second match
    views = [claim("a", "d1"), claim("b", "d2"), claim("h", "d3")]
    off: dict = {}
    LLMPairer(None, None, embedder=FixedEmbedder(vectors), orphan_count=0)._add_neighbors(views, off)
    on: dict = {}
    LLMPairer(None, None, embedder=FixedEmbedder(vectors), near_similarity=0.55, orphan_count=0)._add_neighbors(views, on)
    assert ("a", "h") not in {tuple(sorted(k)) for k in off} and ("a", "h") in {tuple(sorted(k)) for k in on}


# ---- 3. echoed values, retry, quotes from code -------------------------------------
def test_a_verdict_whose_echoed_values_belong_to_another_pair_is_asked_again_alone(session, room):  # noqa: F811
    a, b = make_claims(session, room)
    p = pair(a, b)
    wrong = verdict_for(p, a, value_a="29", value_b="26")["verdicts"][0]
    right = verdict_for(p, a, value_a="120000", value_b="100000")["verdicts"][0]
    llm = MockLLMClient([{"verdicts": [wrong]}, {"verdicts": [right]}])
    stats = PairJudge(session, llm, build_registry(session, None, room.investigation.id)).run([p])
    assert (stats.misaligned, stats.retries, stats.findings_saved) == (1, 1, 1)
    assert len(llm.prompts) == 2 and session.scalars(select(Finding)).one().classification == "contradiction"


def test_a_verdict_that_repeats_the_right_values_in_either_order_is_accepted_without_a_retry(session, room):  # noqa: F811
    a, b = make_claims(session, room)
    p = pair(a, b)
    v = verdict_for(p, a, value_a="100000", value_b="120000")["verdicts"][0]
    stats = PairJudge(session, MockLLMClient([{"verdicts": [v]}]), build_registry(session, None, room.investigation.id)).run([p])
    assert (stats.misaligned, stats.retries, stats.findings_saved) == (0, 0, 1)


def test_insufficient_evidence_without_quotes_is_saved_with_the_claims_own_sentences(session, room):  # noqa: F811
    a = Claim(document_id=room.deck.id, chunk_id=room.deck_p1, subject="plant", metric="capacity", value="120,000", unit="tonnes",
              claim_text="Annual production capacity is 120,000 tonnes per year")
    b = Claim(document_id=room.capex.id, chunk_id=room.capex_current, subject="plant", metric="capacity", value="100,000", unit="tonnes",
              claim_text="Capacity | 100,000 tonnes")
    session.add_all([a, b])
    session.commit()
    p = pair(a, b)
    v = verdict_for(p, a, classification="insufficient_evidence", severity="low", quote_a="", quote_b="")["verdicts"][0]
    stats = PairJudge(session, MockLLMClient([{"verdicts": [v]}]), build_registry(session, None, room.investigation.id)).run([p])
    assert stats.findings_saved == 1 and stats.quotes_from_code == 1
    assert session.scalars(select(Finding)).one().classification == "insufficient_evidence"
    assert {e.quoted_text for e in session.scalars(select(Evidence))} == {"Annual production capacity is 120,000 tonnes per year", "Capacity | 100,000 tonnes"}


# ---- 2. judge prompt ------------------------------------------------------------------
def test_the_judge_prompt_has_the_basis_asof_label_and_declared_absence_rules():
    assert "BASIS: two values of the same kind" in SYSTEM and "different bases are a contextual_difference, the same basis is a contradiction" in SYSTEM and "whatever their bases" in SYSTEM
    assert "different document dates alone never explain a gap" in SYSTEM and "A different label is a context difference only when" in SYSTEM
    assert "is checked by code, not by you" in SYSTEM and "two claims read from one sentence are not a pair" in SYSTEM
    assert "a passage can contradict itself" not in SYSTEM and "value_a and value_b" in SYSTEM


# ---- A, B, C, D (second round) ---------------------------------------------------
def test_units_differing_only_by_underscore_or_case_are_the_same_unit():
    from app.services.candidate_matching import compare_attributes, relation_hint

    a = view("a", value=185, unit="INR Crore", metric="Installed_Cost", subject="LTH-001")
    b = view("b", doc="d2", value=185, unit="INR_Crore", metric="Budgeted_Cost", subject="LTH-001")
    comparison = compare_attributes(a, b)
    assert comparison.unit_match is True and relation_hint(comparison) == "exact"  # an agreement: it never reaches the judge
    assert rel(185, "1,850,000,000", "INR Crore", "INR") == "equal"


def test_a_table_claim_is_quoted_from_its_own_row_even_when_the_value_is_in_every_row(session, room):  # noqa: F811
    from app.services.pair_judge import quote_for

    rows = ("[S!A2:F2] Tool_ID: LTH-001; Status: Installed; Installed_Cost: 185\n"
            "[S!A3:F3] Tool_ID: IMPL-002; Status: Installed; Installed_Cost: 54\n"
            "[S!A4:F4] Tool_ID: DEP-008; Status: Installed; Installed_Cost: 36")
    doc, chunks = add_doc(session, "tools.xlsx", "xlsx", [dict(content=rows, sheet_name="S", cell_reference="A2:F4", content_type="table_row")])
    claim = Claim(document_id=doc.id, chunk_id=chunks[0].id, subject="IMPL-002", metric="Status", value="Installed", claim_text="Installed")
    assert quote_for(claim, chunks[0]) == "[S!A3:F3] Tool_ID: IMPL-002; Status: Installed; Installed_Cost: 54"


def test_a_claim_that_is_in_no_pair_gets_its_best_matches_but_only_above_the_floor():
    vectors = {"a": [1.0, 0.0, 0.0, 0.0], "b": [0.6, 0.8, 0.0, 0.0], "h": [0.58, 0.0, 0.815, 0.0], "w": [0.2, 0.0, 0.0, 0.98]}
    # a-b 0.60 is mutual; h (0.58 to a) and w (0.20 to a) are in no pair after that
    views = [claim("a", "d1"), claim("b", "d2"), claim("h", "d3", "%"), claim("w", "d4", "%")]
    views[0].unit = views[1].unit = "%"
    got: dict = {}
    LLMPairer(None, None, embedder=FixedEmbedder(vectors))._add_neighbors(views, got)
    keys = {tuple(sorted(k)) for k in got}
    assert ("a", "h") in keys  # h would otherwise be in no pair
    assert not any("w" in k for k in keys)  # w's best match is below the floor
    nounit: dict = {}
    for v in views:
        v.unit = None
    LLMPairer(None, None, embedder=FixedEmbedder(vectors))._add_neighbors(views, nounit)
    assert ("a", "h") not in {tuple(sorted(k)) for k in nounit}  # a claim without a unit is not matched this way
    off: dict = {}
    LLMPairer(None, None, embedder=FixedEmbedder(vectors), orphan_count=0)._add_neighbors(views, off)
    assert ("a", "h") not in {tuple(sorted(k)) for k in off}


def test_a_mismatched_echo_is_logged_with_both_sides(session, room):  # noqa: F811
    a, b = make_claims(session, room)
    p = pair(a, b)
    wrong = verdict_for(p, a, value_a="29", value_b="26")["verdicts"][0]
    right = verdict_for(p, a, value_a="120000", value_b="100000")["verdicts"][0]
    stats = PairJudge(session, MockLLMClient([{"verdicts": [wrong]}, {"verdicts": [right]}]), build_registry(session, None, room.investigation.id)).run([p])
    assert len(stats.echo_log) == 1 and "model repeated '29' / '26'" in stats.echo_log[0] and "first call" in stats.echo_log[0]
