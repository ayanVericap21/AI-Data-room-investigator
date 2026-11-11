import pytest

from app.core.exceptions import ExtractionError, UnsupportedDocumentError
from app.services.extraction_service import extract_document, unit_count
from tests.conftest import make_docx, make_pdf, make_xlsx


def test_pdf_pages_are_preserved(tmp_path):
    p = make_pdf(tmp_path / "a.pdf", ["Capacity is 30000 tonnes.", "Second page text."])
    units = extract_document(p, "doc_1", "a.pdf")
    assert [u.locator.page for u in units] == [1, 2]
    assert units[0].content_type == "paragraph"
    assert "30000 tonnes" in units[0].content
    assert unit_count(p, "pdf") == 2


def test_docx_headings_paragraphs_tables(tmp_path):
    units = extract_document(make_docx(tmp_path / "a.docx"), "doc_1", "a.docx")
    kinds = [(u.content_type, u.locator.heading) for u in units]
    assert ("heading", "Operations") in kinds
    para = next(u for u in units if u.content.startswith("Annual production"))
    assert para.locator.heading == "Operations" and para.locator.paragraph_index == 2
    table = next(u for u in units if u.content_type == "table")
    assert table.locator.table_index == 1 and table.locator.heading == "Operations"
    assert "Capacity | 30000" in table.content
    finance = next(u for u in units if u.content.startswith("Revenue projections"))
    assert finance.locator.heading == "Finance"


def test_xlsx_formulas_are_not_plain_text(tmp_path):
    p = make_xlsx(tmp_path / "a.xlsx")
    units = extract_document(p, "doc_1", "a.xlsx")
    row = next(u for u in units if u.locator.cell_reference == "A2:D2")
    assert row.locator.sheet_name == "Revenue"
    cells = {c["ref"]: c for c in row.metadata["cells"]}
    assert cells["D2"]["formula"] == "=B2*C2"
    assert cells["D2"]["cached_value_available"] is False  # openpyxl does not calculate
    assert "cached value not available" in row.content
    assert cells["D2"]["column_header"] == "Revenue"
    assert cells["B2"]["value"] == 100
    assert {u.locator.sheet_name for u in units} == {"Revenue", "Notes"}
    assert unit_count(p, "xlsx") == 2


def test_corrupt_file_raises(tmp_path):
    bad = tmp_path / "bad.pdf"
    bad.write_bytes(b"not a pdf")
    with pytest.raises(ExtractionError):
        extract_document(bad, "doc_1", "bad.pdf")
    assert unit_count(bad, "pdf") is None


def test_empty_pdf_raises(tmp_path):
    p = make_pdf(tmp_path / "empty.pdf", [""])
    with pytest.raises(ExtractionError):
        extract_document(p, "doc_1", "empty.pdf")


def test_unsupported_type(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("x")
    with pytest.raises(UnsupportedDocumentError):
        extract_document(f, "doc_1", "a.txt")


def test_xlsm_is_read_like_xlsx(tmp_path):
    from app.services.extraction_service import file_type_for

    p = make_xlsx(tmp_path / "model.xlsx").rename(tmp_path / "model.xlsm")
    assert file_type_for(p) == "xlsx"  # same stored type, so nothing else needs to change
    units = extract_document(p, "doc_1", "model.xlsm")
    assert {u.locator.sheet_name for u in units} == {"Revenue", "Notes"}
    assert unit_count(p, "xlsx") == 2
