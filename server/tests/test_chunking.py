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


# ---- merging of small neighbouring units --------------------------------------
def merged(units, **kw):
    kw.setdefault("merge", True)
    return chunk_units(units, **kw)

def xunit(row, text, sheet="Revenue", cols="A{r}:D{r}"):
    return ExtractedUnit(document_id="d", filename="m.xlsx", source_type="xlsx",
                         locator=Locator(sheet_name=sheet, cell_reference=cols.format(r=row)),
                         content=f"[{sheet}!{cols.format(r=row)}] {text}", content_type="table_row",
                         metadata={"cells": [{"ref": f"A{row}"}], "header_row": 1, "is_header_row": row == 1})


def test_small_units_on_the_same_page_become_one_chunk_in_reading_order():
    units = [unit("Title", page=1), unit("Capacity 120,000 tonnes", page=1), unit("Pune plant", page=1)]
    drafts = merged(units)
    assert len(drafts) == 1 and drafts[0].content == "Title\nCapacity 120,000 tonnes\nPune plant"
    assert drafts[0].page_number == 1 and drafts[0].metadata["merged_units"] == 3
    assert all(u.content in drafts[0].content for u in units)  # nothing reworded or lost


def test_units_on_different_pages_are_never_merged():
    drafts = merged([unit("A", page=1), unit("B", page=2), unit("C", page=2)])
    assert [(d.page_number, d.content) for d in drafts] == [(1, "A"), (2, "B\nC")]


def test_merging_stops_at_the_target_size_and_never_exceeds_the_hard_limit():
    units = [unit("x" * 400, page=1) for _ in range(5)]
    drafts = merged(units, max_chars=1500, target_chars=1000)
    assert [len(d.content) for d in drafts] == [801, 801, 400]  # 2 + 2 + 1 units
    assert all(len(d.content) <= 1000 for d in drafts)
    assert all(len(d.content) <= 1500 for d in merged(units, max_chars=1500, target_chars=9999))


def test_tables_and_oversized_units_stay_on_their_own():
    drafts = merged([unit("intro", page=1), unit("a | b\nc | d", ctype="table", page=1), unit("after", page=1)])
    assert [d.content_type for d in drafts] == ["paragraph", "table", "paragraph"]
    big = merged([unit("small", page=1), unit(" ".join(["Sentence here."] * 200), page=1)], max_chars=300)
    assert big[0].content == "small" and len(big) > 2


def test_merge_can_be_turned_off():
    units = [unit("A", page=1), unit("B", page=1)]
    assert len(chunk_units(units, merge=False)) == 2 and len(merged(units)) == 1
    assert len(chunk_units(units)) == 1  # merging is on by default


def test_spreadsheet_rows_merge_within_a_sheet_with_a_bounding_range_and_row_tags():
    units = [xunit(1, "Product; Tonnes"), xunit(2, "Steel; 100"), xunit(3, "Copper; 50", cols="A{r}:F{r}"),
             xunit(1, "Note", sheet="Notes")]
    drafts = merged(units)
    assert len(drafts) == 2  # the second sheet never joins the first
    first = drafts[0]
    assert first.sheet_name == "Revenue" and first.cell_reference == "A1:F3" and first.content_type == "table_row"
    assert "[Revenue!A2:D2] Steel; 100" in first.content and "[Revenue!A3:F3] Copper; 50" in first.content
    assert len(first.metadata["cells"]) == 3 and first.metadata["is_header_row"] is True
    assert drafts[1].sheet_name == "Notes" and drafts[1].cell_reference == "A1:D1"


def test_docx_sections_merge_under_their_heading_and_a_new_heading_starts_a_new_chunk():
    def d(text, ctype, heading, n):
        return ExtractedUnit(document_id="d", filename="x.docx", source_type="docx",
                             locator=Locator(paragraph_index=n, heading=heading), content=text, content_type=ctype)

    drafts = merged([d("Rent", "heading", "Rent", 1), d("Annual rent is 1.2 crore.", "paragraph", "Rent", 2),
                          d("Escalates 5% a year.", "paragraph", "Rent", 3), d("Term", "heading", "Term", 4),
                          d("Nine years.", "paragraph", "Term", 5)])
    assert [(x.heading, x.content) for x in drafts] == [
        ("Rent", "Rent\nAnnual rent is 1.2 crore.\nEscalates 5% a year."), ("Term", "Term\nNine years.")]
    assert drafts[0].content_type == "paragraph"
