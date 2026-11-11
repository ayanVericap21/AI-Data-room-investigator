"""The claim type is shown to the grouping model as information only: it never keeps two claims from being paired."""
from app.services.candidate_matching import generate_candidate_pairs
from app.services.llm_pairing import SYSTEM, LLMPairer, _line
from tests.test_claims import view
from tests.test_neighbors_and_pdf_tables import KeywordEmbedder


def typed(id, kind, doc, **kw):
    v = view(id, doc=doc, metric="capacity", value=kw.pop("value", 1), claim_text=f"capacity {id}", **kw)
    v.value_type = kind
    return v


def test_claims_of_different_types_are_still_paired_by_every_route():
    a, b, c = typed("a", "numerical", "d1", value=10), typed("b", "textual", "d2", value="ten"), typed("c", "numerical", "d3", value=12)
    assert {frozenset((p.claim_a_id, p.claim_b_id)) for p in generate_candidate_pairs([a, b, c])} == {
        frozenset(("a", "b")), frozenset(("a", "c")), frozenset(("b", "c"))}
    pairs: dict = {}
    LLMPairer._add_pair(pairs, a, b)
    LLMPairer._add_pair(pairs, a, c)
    assert set(pairs) == {("a", "b"), ("a", "c")}
    near: dict = {}
    LLMPairer(None, None, embedder=KeywordEmbedder())._add_neighbors([a, b, c], near)
    assert ("a", "b") in near and ("a", "c") in near


def test_the_grouping_prompt_shows_each_type_and_does_not_forbid_mixed_groups():
    a = typed("a", "numerical", "d1", value=10)
    assert "type: numerical" in _line("c1", a, "f.pdf", "page 1") and "type: -" in _line("c2", typed("x", None, "d1"), "f.pdf", "page 1")
    assert "Do not keep claims apart because their types differ" in SYSTEM and "never grouped; a group holds claims of one type only" not in SYSTEM
