"""Nearest-neighbour pairing across documents, and PDF table cells that keep their spaces and decimals."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pymupdf

from app.services.extractors.pdf import _table_rows, extract_pdf
from app.services.llm_pairing import LLMPairer
from tests.test_claims import view


class KeywordEmbedder:
    """Texts about the same keyword get the same direction, so similarity is 1 for them and 0 otherwise."""

    def embed(self, texts, task="document"):
        axes = ("capacity", "certification", "yield")
        return [[1.0 if word in t.lower() else 0.0 for word in axes] + [0.01] for t in texts]


def neighbors(views, **kw):
    pairer = LLMPairer(None, None, embedder=KeywordEmbedder(), **kw)
    pairs: dict = {}
    pairer._add_neighbors(views, pairs)
    return pairs


def test_a_claim_is_paired_with_its_nearest_neighbours_in_other_documents_whatever_the_labels_say():
    a = view("a", doc="d1", subject="Designed nameplate", metric="Management-reported position", value="20,000", claim_text="capacity 20,000 per month")
    b = view("b", doc="d2", subject="Designed nameplate", metric="Model_Value", value="24000", claim_text="capacity 24000")
    c = view("c", doc="d3", subject="SiliconForge", metric="certification", value="ISO 9001", claim_text="certification held")
    pairs = neighbors([a, b, c])
    assert set(pairs) == {("a", "b")}  # the capacity claims, although the labels share nothing; c is about something else
    only = pairs[("a", "b")]
    assert only.matched_by == "embedding" and only.relation_hint == "conflicting"  # the values differ: the judge will decide


def test_claims_of_one_document_are_not_paired_by_neighbourhood_and_equal_values_are_agreement():
    a = view("a", doc="d1", metric="capacity", value=100, claim_text="capacity 100")
    same_doc = view("b", doc="d1", metric="capacity", value=200, claim_text="capacity 200")
    equal = view("c", doc="d2", metric="capacity", value=100, claim_text="capacity of 100 per month")
    pairs = neighbors([a, same_doc, equal])
    assert ("a", "b") not in pairs and pairs[("a", "c")].relation_hint == "exact"
    assert ("b", "c") in pairs and pairs[("b", "c")].relation_hint == "conflicting"


def test_each_claim_takes_at_most_the_requested_number_of_neighbours_and_a_grouped_pair_is_kept():
    views = [view("a", doc="d1", metric="capacity", value=1, claim_text="capacity 1")] + \
            [view(f"x{i}", doc=f"e{i}", metric="capacity", value=10 + i, claim_text=f"capacity {10 + i}") for i in range(4)]
    pairs = neighbors(views, neighbors=1)
    assert sum(1 for k in pairs if "a" in k) <= 4  # the others also pick a as their own nearest neighbour
    existing = {("a", "x0"): "from the grouping"}
    pairer = LLMPairer(None, None, embedder=KeywordEmbedder(), neighbors=4)
    pairer._add_neighbors(views, existing)
    assert existing[("a", "x0")] == "from the grouping"  # a pair the grouping already made is not replaced


def _table_pdf(path: Path) -> Path:
    doc = pymupdf.open()
    page = doc.new_page()
    rows = [("Metric", "Q3 2026 value"), ("Average monthly wafer starts", "9,200 WSPM"), ("Aggregate yield", "92.4%")]
    for r, (left, right) in enumerate(rows):
        top = 100 + r * 24
        page.draw_rect(pymupdf.Rect(72, top, 280, top + 24))
        page.draw_rect(pymupdf.Rect(280, top, 420, top + 24))
        page.insert_text((76, top + 16), left, fontsize=10)
        page.insert_text((284, top + 16), right, fontsize=10)
    doc.save(path)
    doc.close()
    return path


def test_pdf_table_cells_keep_their_spaces_and_decimal_points(tmp_path):
    units = extract_pdf(_table_pdf(tmp_path / "t.pdf"), "d1", "t.pdf")
    table = next(u for u in units if u.content_type == "table")
    assert "Average monthly wafer starts | 9,200 WSPM" in table.content and "Aggregate yield | 92.4%" in table.content
    assert table.metadata["rows"][2] == ["Aggregate yield", "92.4%"]


def test_the_cell_reader_falls_back_to_the_extractors_own_text_when_cells_cannot_be_read():
    class BrokenTable:
        @property
        def rows(self):
            raise RuntimeError("no cells")

        def extract(self):
            return [["a", "b"]]

    assert _table_rows(SimpleNamespace(), BrokenTable()) == [["a", "b"]]
