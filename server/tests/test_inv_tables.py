"""The table store: spreadsheet blocks and document tables as SQLite tables with provenance, read-only SQL, search, coverage."""
import pytest

from app.investigator.tables import TableStore, ToolSqlError, days_between, parse_when, to_num
from tests.inv_support import build_store, dive_room, xlsx_chunk
from tests.test_tools import add_doc, room  # noqa: F401  (fixtures and helper)


def test_helpers_read_numbers_and_dates():
    assert to_num("(1,200)") == -1200 and to_num("92.4%") == 92.4 and to_num("12 kg") == 12 and to_num("abc") is None and to_num("2026-10-08") is None
    assert parse_when("9 October 2026 14:30") == "2026-10-09T14:30" and parse_when("March 3, 2025") == "2025-03-03" and parse_when("05/06/2026") is None
    assert parse_when("13/06/2026") == "2026-06-13" and days_between("2025-03-01", "2025-03-11") == 10


def test_a_spreadsheet_becomes_a_table_with_headers_and_row_provenance(session, tmp_path):
    permit, register, lab = dive_room(session)
    store = build_store(session, tmp_path)
    table = next(t for t in store.table_list() if t.filename == "register.xlsx")
    assert [c.header for c in table.columns] == ["Dive", "Date", "Site", "Samples"] and table.rows == 6
    res = store.query(f'SELECT dive, samples, _ref FROM "{table.name}" WHERE samples > 8')
    assert res.rows == [("D-04", 9, "A5:D5")]
    chunk_id, line = store.row_source(table.name, "A5:D5")
    assert "D-04" in line and line.startswith("[Dives!A5:D5]") and store.chunk(chunk_id)[3].count("\n") == 6


def test_a_row_without_cell_detail_is_read_back_from_its_line(session, tmp_path):
    rows = {1: [(None, "Site"), (None, "Value")], 2: [("Site", "IR-01"), ("Value", 5)], 3: [("Site", "IR-02"), ("Value", 7)]}
    add_doc(session, "bare.xlsx", "xlsx", [xlsx_chunk("S", rows, with_cells=False)])
    store = build_store(session, tmp_path)
    t = store.table_list()[0]
    assert store.query(f'SELECT site, value FROM "{t.name}" ORDER BY _rid').rows == [("IR-01", 5), ("IR-02", 7)]


def test_a_blank_row_splits_a_sheet_into_two_tables(session, tmp_path):
    rows = {1: [(None, "Site"), (None, "Count")], 2: [("Site", "A"), ("Count", 1)], 3: [("Site", "B"), ("Count", 2)],
            6: [(None, "Total"), (None, "Mean")], 7: [("Total", 3), ("Mean", 1.5)]}
    add_doc(session, "two.xlsx", "xlsx", [xlsx_chunk("S", rows)])
    store = build_store(session, tmp_path)
    assert len(store.table_list()) == 2
    second = store.table_list()[1]
    assert [c.header for c in second.columns] == ["Total", "Mean"] and second.rows == 1


def test_a_table_inside_a_document_is_loaded_with_its_header_row(session, tmp_path):
    add_doc(session, "report.docx", "docx", [dict(content="Site | Cover\nIR-01 | 33.3\nIR-02 | 41.0", content_type="table", metadata_={"locator": {"table_index": 1}})])
    store = build_store(session, tmp_path)
    t = store.table_list()[0]
    assert t.kind == "doc_table" and store.query(f'SELECT site, cover FROM "{t.name}" ORDER BY _rid').rows == [("IR-01", 33.3), ("IR-02", 41.0)]
    assert store.row_source(t.name, "row 2")[1] == "IR-02 | 41.0"


def test_identifiers_with_leading_zeros_stay_text(session, tmp_path):
    add_doc(session, "ids.xlsx", "xlsx", [xlsx_chunk("S", {1: [(None, "Code"), (None, "N")], 2: [("Code", "007"), ("N", 7)], 3: [("Code", "008"), ("N", 8)]})])
    store = build_store(session, tmp_path)
    t = store.table_list()[0]
    assert store.query(f'SELECT code FROM "{t.name}" ORDER BY _rid').rows == [("007",), ("008",)]


def test_queries_are_read_only_single_statements_with_helper_functions(session, tmp_path):
    dive_room(session)
    store = build_store(session, tmp_path)
    t = next(t for t in store.table_list() if t.filename == "register.xlsx")
    for bad in ("DROP TABLE _tables", "DELETE FROM _facts", "SELECT 1; SELECT 2", "UPDATE _tables SET name = 'x'", "PRAGMA table_info(_tables)", ""):
        with pytest.raises(ToolSqlError):
            store.query(bad)
    assert store.query(f'SELECT days_between(MIN(date), MAX(date)) FROM "{t.name}"').rows == [(18.0,)]
    assert store.query(f"SELECT COUNT(*) FROM \"{t.name}\" WHERE to_date(date) > '2025-03-21'").rows == [(1,)]
    assert store.query(f"SELECT dive FROM \"{t.name}\" WHERE site REGEXP '^ir-0[12]$' ORDER BY _rid").rows == [("D-01",), ("D-02",), ("D-04",), ("D-05",)]
    capped = store.query(f'SELECT * FROM "{t.name}"', limit=2)
    assert len(capped.rows) == 2 and capped.truncated


def test_search_finds_passages_and_the_schema_text_lists_tables(session, tmp_path):
    dive_room(session)
    store = build_store(session, tmp_path)
    hits = store.search("decontamination log submitted")
    assert hits and hits[0]["filename"] == "permit.docx" and "decontamination" in hits[0]["snippet"]
    assert store.search("SPL-044-B", document="lab.pdf")[0]["filename"] == "lab.pdf"
    text = store.schema_text()
    assert "register.xlsx" in text and "columns:" in text


def test_coverage_counts_chunks_with_numbers_and_tables(session, tmp_path):
    permit, register, lab = dive_room(session)
    add_doc(session, "blank.docx", "docx", [dict(content="Nothing numeric here at all.", content_type="paragraph")])
    cov = TableStore(tmp_path / "c.sqlite").build(session)
    assert cov[register.id].tables == 1 and cov[register.id].ok and cov[lab.id].tables == 0
    # a document with numbers, no tables and no facts is not read; one without numbers has nothing to lose
    assert not cov[lab.id].ok and "contain numbers" in cov[lab.id].reason()
    blank = next(c for c in cov.values() if c.filename == "blank.docx")
    assert blank.ok
    cov[lab.id].facts = 3
    assert cov[lab.id].ok
