from app.schemas.extraction import ExtractedUnit, Locator
from app.services.chunking import chunk_units, embedding_text, split_text


def unit(content, ctype="paragraph", **loc):
    return ExtractedUnit(document_id="d", filename="f.pdf", source_type="pdf", locator=Locator(**loc),
                         content=content, content_type=ctype)


def test_small_units_map_one_to_one_and_keep_locators():
    drafts = chunk_units([unit("A.", page=1), unit("B.", page=2, heading="H")])
    assert [(d.chunk_index, d.page_number, d.heading) for d in drafts] == [(0, 1, None), (1, 2, "H")]
    assert drafts[1].metadata["locator"] == {"page": 2, "heading": "H"}


def test_long_paragraph_splits_on_sentences_and_keeps_locator():
    text = " ".join(f"Sentence number {i} is here." for i in range(40))
    drafts = chunk_units([unit(text, page=3)], max_chars=200)
    assert len(drafts) > 1
    assert all(len(d.content) <= 200 and d.page_number == 3 for d in drafts)
    assert " ".join(d.content for d in drafts) == text  # nothing lost or altered
    assert drafts[0].metadata["parts"] == len(drafts) and drafts[1].metadata["part"] == 2
    assert [d.chunk_index for d in drafts] == list(range(len(drafts)))


def test_long_table_splits_on_rows():
    rows = "\n".join(f"row{i} | {i}" for i in range(100))
    parts = split_text(rows, "table", 100)
    assert all(len(p) <= 100 for p in parts)
    assert "\n".join(parts) == rows


def test_single_oversized_token_is_hard_split():
    assert all(len(p) <= 10 for p in split_text("x" * 95, "paragraph", 10))


def test_embedding_text_adds_context_but_content_untouched():
    assert embedding_text("a.pdf", "Ops", "body") == "Document: a.pdf | Section: Ops\nbody"
    assert embedding_text("a.pdf", None, "body") == "Document: a.pdf\nbody"
