"""Unit inside a value, subject gates, wider neighbour matching, echoed values with a retry, and quotes supplied by code."""
from sqlalchemy import select

from app.llm.mock import MockLLMClient
from app.models import Claim, Evidence, Finding
from app.services.claim_matching import normalize_unit, comparable_subjects
from app.services.claim_grouping import ClaimGrouper
from app.services.value_rules import compare_values, unit_info, units_match
from app.tools import build_registry
from tests.test_claims import view
from tests.test_claim_group_judge import make_claims
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


def test_rows_with_different_identifiers_are_not_grouped_but_descriptive_labels_are():
    assert not comparable_subjects(row("a", "d1", "MET-004", "d1|S|"), row("b", "d2", "MET-003", "d2|S|"))
    assert comparable_subjects(row("a", "d1", "MET-004", "d1|S|"), row("c", "d2", "MET-004", "d2|S|"))
    assert comparable_subjects(row("d", "d1", "Aggregate yield", "d1|S|"), row("e", "d2", "Q3 2026 production yield", "d2|S|"))  # labels, not keys
    assert not comparable_subjects(row("f", "d1", "MET-004", None), row("g", "d2", "MET-003", None))  # not table rows either: subjects that name different records never go together


def test_two_claims_read_from_one_sentence_do_not_go_together_but_a_repeated_sentence_elsewhere_does():
    same = "SiliconForge is ISO 9001:2015 certified."
    assert not comparable_subjects(row("a", "d1", "X", None, same, "chunk1"), row("b", "d1", "Y", None, same, "chunk1"))
    assert comparable_subjects(row("a", "d1", "X", None, same, "chunk1"), row("c", "d1", "X", None, same, "chunk2"))


# ---- 5. neighbours: strong top-2, mutual best match, top-5 from a lower floor ----
class FixedEmbedder:
    def __init__(self, vectors):
        self.vectors = vectors

    def embed(self, texts, task="document"):
        return [self.vectors[t.split(": ")[0]] for t in texts]


def claim(id, doc, unit=None):
    return view(id, doc=doc, subject=id, metric="m", value=1, claim_text="x", unit=unit, period=None)


def neighbour_clusters(views, vectors, **kw):
    return ClaimGrouper(None, None, embedder=FixedEmbedder(vectors), **kw)._neighbor_clusters(views, set())


def together(clusters, x, y):
    return any({x, y} <= set(c.claim_ids) for c in clusters)


def test_a_best_match_below_the_old_cut_off_is_now_grouped_and_a_weak_one_is_not():
    vectors = {"a": [1.0, 0.0, 0.0], "b": [0.6, 0.8, 0.0], "c": [0.3, 0.954, 0.0], "g": [0.45, 0.0, 0.893]}
    views = [claim("a", "d1"), claim("b", "d2"), claim("c", "d2"), claim("g", "d3")]
    clusters = neighbour_clusters(views, vectors)
    assert together(clusters, "a", "b")  # 0.60: each other's best match elsewhere, below the old 0.72 cut-off
    assert not together(clusters, "a", "c") and not together(clusters, "a", "g")  # 0.30, and 0.45 (below the floor, and not mutual)


def test_the_top_five_route_is_off_unless_asked_for():
    vectors = {"a": [1.0, 0.0, 0.0], "b": [0.6, 0.8, 0.0], "h": [0.58, 0.0, 0.815]}  # a-h 0.58 is a's second match
    views = [claim("a", "d1"), claim("b", "d2"), claim("h", "d3")]
    off = neighbour_clusters(views, vectors, orphan_count=0)
    on = neighbour_clusters(views, vectors, near_similarity=0.55, orphan_count=0)
    assert not together(off, "a", "h") and together(on, "a", "h")


# ---- 3. echoed values, retry, quotes from code -------------------------------------






# ---- 2. judge prompt ------------------------------------------------------------------


# ---- A, B, C, D (second round) ---------------------------------------------------
def test_units_differing_only_by_underscore_or_case_are_the_same_unit():
    a = view("a", value=185, unit="INR Crore", metric="Installed_Cost", subject="LTH-001")
    b = view("b", doc="d2", value=185, unit="INR_Crore", metric="Budgeted_Cost", subject="LTH-001")
    assert units_match(unit_info(a.unit, normalize_unit), unit_info(b.unit, normalize_unit)) is True
    assert compare_values(a.value, a.unit, b.value, b.unit, None, None, normalize_unit) == "equal"  # an agreement: it never reaches the judge
    assert rel(185, "1,850,000,000", "INR Crore", "INR") == "equal"


def test_a_table_claim_is_quoted_from_its_own_row_even_when_the_value_is_in_every_row(session, room):  # noqa: F811
    from app.services.claim_group_judge import quote_for

    rows = ("[S!A2:F2] Tool_ID: LTH-001; Status: Installed; Installed_Cost: 185\n"
            "[S!A3:F3] Tool_ID: IMPL-002; Status: Installed; Installed_Cost: 54\n"
            "[S!A4:F4] Tool_ID: DEP-008; Status: Installed; Installed_Cost: 36")
    doc, chunks = add_doc(session, "tools.xlsx", "xlsx", [dict(content=rows, sheet_name="S", cell_reference="A2:F4", content_type="table_row")])
    claim = Claim(document_id=doc.id, chunk_id=chunks[0].id, subject="IMPL-002", metric="Status", value="Installed", claim_text="Installed")
    assert quote_for(claim, chunks[0]) == "[S!A3:F3] Tool_ID: IMPL-002; Status: Installed; Installed_Cost: 54"


def test_a_claim_that_is_in_no_cluster_gets_its_best_matches_but_only_above_the_floor():
    vectors = {"a": [1.0, 0.0, 0.0, 0.0], "b": [0.6, 0.8, 0.0, 0.0], "h": [0.58, 0.0, 0.815, 0.0], "w": [0.2, 0.0, 0.0, 0.98]}
    # a-b 0.60 is mutual; h (0.58 to a) and w (0.20 to a) are in no cluster after that
    views = [claim("a", "d1"), claim("b", "d2"), claim("h", "d3", "%"), claim("w", "d4", "%")]
    views[0].unit = views[1].unit = "%"
    got = neighbour_clusters(views, vectors)
    assert together(got, "a", "h")  # h would otherwise be in no cluster
    assert not any("w" in c.claim_ids for c in got)  # w's best match is below the floor
    for v in views:
        v.unit = None
    assert not together(neighbour_clusters(views, vectors), "a", "h")  # a claim without a unit is not matched this way
    assert not together(neighbour_clusters(views, vectors, orphan_count=0), "a", "h")


