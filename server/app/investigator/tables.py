"""The table store: every spreadsheet block and every table inside a document, loaded into SQLite with its provenance kept.

Each table row keeps the chunk it came from and the exact line of that chunk, so a query result can always be turned into a verified
quote. Sheets are split into blocks wherever a blank row separates them (a data block followed by a totals block is two tables).
Formulas are recorded next to the cached values. The store also holds the chunks (with full-text search), the typed facts, the
entity mentions and the timeline, so workers can query all of it with read-only SQL.

Queries run on a separate read-only connection with an authorizer that only allows reads, one statement, a row cap and a time cap.
"""
from __future__ import annotations

import calendar
import json
import re
import sqlite3
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Document, DocumentChunk
from app.models.base import ProcessingStatus

_COL_REF = re.compile(r"^([A-Z]+)(\d+)$")
_ROW_TAG = re.compile(r"^\[(.+)!([A-Z]+)(\d+):([A-Z]+)(\d+)\]\s*")
_PLAIN_NUMBER = re.compile(r"^[-+]?(?:0|[1-9]\d*)(?:\.\d+)?$")  # no leading zeros: 007 is an identifier, not 7
_NUMBER_WITH_UNIT = re.compile(r"^\s*(\(?[-+]?\$?\d[\d,]*\.?\d*\)?)\s*[A-Za-z%°/²³ ]{0,14}$")
MAX_QUERY_ROWS = 200
QUERY_SECONDS = 8.0


class ToolSqlError(Exception):
    """A query that is not allowed or failed; the message is shown to the worker so it can correct the query."""


# ---- value helpers ---------------------------------------------------------------------------------------------------------------------
def col_letters_to_number(letters: str) -> int:
    n = 0
    for ch in letters:
        n = n * 26 + ord(ch) - 64
    return n


def number_to_col_letters(n: int) -> str:
    out = ""
    while n:
        n, rem = divmod(n - 1, 26)
        out = chr(65 + rem) + out
    return out


def to_num(value: Any) -> float | None:
    """A number read from a cell: plain numbers, thousands separators, a percent sign, brackets for negatives, a trailing unit."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    m = _NUMBER_WITH_UNIT.match(str(value))
    if not m:
        return None
    text = m.group(1)
    negative = text.startswith("(") and text.endswith(")")
    text = re.sub(r"[()$,+]", "", text)
    try:
        number = float(text)
    except ValueError:
        return None
    return -abs(number) if negative else number


_MONTHS = {name.lower(): i for i, name in enumerate(calendar.month_name) if name}
_MONTHS.update({name.lower(): i for i, name in enumerate(calendar.month_abbr) if name})
_MONTHS["sept"] = 9
_TIME = re.compile(r"(?<!\d)(\d{1,2}):(\d{2})(?::\d{2})?(?!\d)")


def _build(year: int, month: int, day: int, hour: int | None, minute: int | None) -> str | None:
    try:
        d = datetime(year, month, day, hour or 0, minute or 0)
    except ValueError:
        return None
    return d.strftime("%Y-%m-%d") if hour is None else d.strftime("%Y-%m-%dT%H:%M")


def parse_when(value: Any) -> str | None:
    """A date (and time, when written) as ISO text, or None when it is not a date or is ambiguous (05/06/2026)."""
    if value is None or isinstance(value, (int, float)):
        return None
    text = str(value).strip()
    if not text:
        return None
    tm = _TIME.search(text)
    hour, minute = (int(tm.group(1)), int(tm.group(2))) if tm else (None, None)
    m = re.search(r"(\d{4})-(\d{2})-(\d{2})", text)
    if m:
        return _build(int(m.group(1)), int(m.group(2)), int(m.group(3)), hour, minute)
    m = re.search(r"(\d{1,2})(?:st|nd|rd|th)?\s+([A-Za-z]{3,9})\.?,?\s+(\d{4})", text)
    if m and m.group(2).lower() in _MONTHS:
        return _build(int(m.group(3)), _MONTHS[m.group(2).lower()], int(m.group(1)), hour, minute)
    m = re.search(r"([A-Za-z]{3,9})\.?\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})", text)
    if m and m.group(1).lower() in _MONTHS:
        return _build(int(m.group(3)), _MONTHS[m.group(1).lower()], int(m.group(2)), hour, minute)
    m = re.search(r"(\d{1,2})/(\d{1,2})/(\d{4})", text)
    if m:
        a, b, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if a > 12 >= b:
            return _build(y, b, a, hour, minute)
        if b > 12 >= a:
            return _build(y, a, b, hour, minute)
        return None
    return None


def _as_datetime(value: Any) -> datetime | None:
    iso = parse_when(value)
    if iso is None:
        return None
    return datetime.strptime(iso, "%Y-%m-%dT%H:%M" if "T" in iso else "%Y-%m-%d")


def hours_between(a: Any, b: Any) -> float | None:
    da, db = _as_datetime(a), _as_datetime(b)
    return None if da is None or db is None else (db - da).total_seconds() / 3600


def days_between(a: Any, b: Any) -> float | None:
    hours = hours_between(a, b)
    return None if hours is None else hours / 24


def norm_id(value: Any) -> str | None:
    return None if value is None else re.sub(r"[\W_]+", "", str(value)).upper()


def _regexp(pattern: Any, value: Any) -> int:
    if pattern is None or value is None:
        return 0
    try:
        return 1 if re.search(str(pattern), str(value), re.I) else 0
    except re.error:
        return 0


def _register(conn: sqlite3.Connection) -> None:
    conn.create_function("to_num", 1, to_num, deterministic=True)
    conn.create_function("to_date", 1, parse_when, deterministic=True)
    conn.create_function("hours_between", 2, hours_between, deterministic=True)
    conn.create_function("days_between", 2, days_between, deterministic=True)
    conn.create_function("norm_id", 1, norm_id, deterministic=True)
    conn.create_function("regexp", 2, _regexp, deterministic=True)


FUNCTIONS_HELP = (
    "SQL helpers: to_num(x) reads a number from text (commas, %, brackets, trailing unit); to_date(x) gives ISO text or NULL; "
    "days_between(a, b) and hours_between(a, b) work on any date text; norm_id(x) removes separators and case; `x REGEXP 'pattern'` is case-insensitive.")


# ---- results -----------------------------------------------------------------------------------------------------------------------------
@dataclass
class QueryResult:
    columns: list[str]
    rows: list[tuple]
    truncated: bool = False

    def text(self, max_chars: int = 3000) -> str:
        if not self.rows:
            return "(no rows)"
        lines = [" | ".join(self.columns)]
        for row in self.rows:
            lines.append(" | ".join("" if v is None else (f"{v:g}" if isinstance(v, float) else str(v)) for v in row))
        out = "\n".join(lines)
        if len(out) > max_chars:
            out = out[:max_chars] + "\n... (cut)"
        return out + ("\n(more rows exist; add LIMIT or narrow the query)" if self.truncated else "")


@dataclass
class ColumnInfo:
    name: str  # the SQL column
    header: str  # the header as written
    formulas: int = 0


@dataclass
class TableInfo:
    name: str
    document_id: str
    filename: str
    title: str
    kind: str  # sheet | doc_table
    location: str
    rows: int
    columns: list[ColumnInfo] = field(default_factory=list)


@dataclass
class DocCoverage:
    document_id: str
    filename: str
    chunks: int = 0
    chunks_with_numbers: int = 0
    tables: int = 0
    table_rows: int = 0
    facts: int = 0
    claims: int = 0

    @property
    def ok(self) -> bool:
        """Read: it produced facts or tables, or it has no numbers to lose."""
        return bool(self.facts or self.tables or not self.chunks_with_numbers)

    def reason(self) -> str:
        return "" if self.ok else f"{self.chunks_with_numbers} of {self.chunks} chunks contain numbers, but no fact and no table was produced"


@dataclass
class _RowData:
    sheet: str
    row: int
    cells: dict[str, dict]  # column letters -> {value, formula}
    line: str
    chunk_id: str


def _sql_name(header: str, taken: set[str]) -> str:
    base = re.sub(r"[^a-z0-9]+", "_", header.lower()).strip("_") or "col"
    if base[0].isdigit():
        base = "c_" + base
    name, n = base, 2
    while name in taken or name.startswith("_"):
        name = f"{base}_{n}"
        n += 1
    taken.add(name)
    return name


def _cell_value(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, (int, float)):
        return v
    text = str(v).strip()
    if text == "":
        return None
    if _PLAIN_NUMBER.match(text):
        return float(text) if "." in text else int(text)
    return text


def _parse_row_line(line: str) -> tuple[int, dict[str, dict]] | None:
    """A spreadsheet row read back from its chunk line ("[Sheet!A2:F2] Header: value; ...") when the chunk has no cell detail."""
    m = _ROW_TAG.match(line)
    if not m:
        return None
    first_col, row = col_letters_to_number(m.group(2)), int(m.group(3))
    cells: dict[str, dict] = {}
    for i, part in enumerate(p for p in line[m.end():].split("; ") if p.strip()):
        label, sep, value = part.partition(": ")
        letter = number_to_col_letters(first_col + i)
        cells[letter] = {"value": value if sep else part, "formula": None, "header": None if _COL_REF.match(label) else label}
    return row, cells


class TableStore:
    def __init__(self, path: Path):
        self.path = Path(path)
        self._lock = threading.RLock()
        self.conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")
        _register(self.conn)
        self._init()
        self.tables: dict[str, TableInfo] = {}
        self._n = 0
        self.fts = self._fts_available()

    # -- schema ---------------------------------------------------------------------------------------------------------------------
    def _init(self) -> None:
        with self._lock:
            self.conn.executescript("""
                CREATE TABLE IF NOT EXISTS _tables (name TEXT PRIMARY KEY, document_id TEXT, filename TEXT, title TEXT, kind TEXT, location TEXT, rows INTEGER, columns TEXT);
                CREATE TABLE IF NOT EXISTS _formulas (tbl TEXT, ref TEXT, col TEXT, formula TEXT);
                CREATE TABLE IF NOT EXISTS _chunks (id INTEGER PRIMARY KEY, chunk_id TEXT UNIQUE, document_id TEXT, filename TEXT, location TEXT, content TEXT);
                CREATE TABLE IF NOT EXISTS _facts (id TEXT PRIMARY KEY, document_id TEXT, filename TEXT, chunk_id TEXT, type TEXT, subject TEXT, attribute TEXT, operator TEXT,
                    value TEXT, unit TEXT, scope TEXT, when_text TEXT, when_iso TEXT, modality TEXT, aggregate_of TEXT, quote TEXT);
                CREATE TABLE IF NOT EXISTS _mentions (entity_type TEXT, ident TEXT, written TEXT, document_id TEXT, filename TEXT, chunk_id TEXT, line TEXT);
                CREATE TABLE IF NOT EXISTS _aliases (canonical TEXT, entity_type TEXT, alias TEXT);
                CREATE TABLE IF NOT EXISTS _events (when_iso TEXT, label TEXT, entity TEXT, source TEXT, document_id TEXT, filename TEXT, chunk_id TEXT, line TEXT, detail TEXT);
            """)
            self.conn.commit()

    def _fts_available(self) -> bool:
        try:
            with self._lock:
                self.conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS _chunks_fts USING fts5(content, filename)")
                self.conn.commit()
            return True
        except sqlite3.Error:
            return False

    # -- loading --------------------------------------------------------------------------------------------------------------------
    def build(self, session: Session, only: set[str] | None = None) -> dict[str, DocCoverage]:
        """Load every processed document: its chunks (for search) and all of its tables. Returns the coverage of each document."""
        coverage: dict[str, DocCoverage] = {}
        docs = [d for d in session.scalars(select(Document).order_by(Document.filename)) if d.processing_status == ProcessingStatus.processed.value
                and (only is None or d.filename in only)]
        for doc in docs:
            chunks = list(session.scalars(select(DocumentChunk).where(DocumentChunk.document_id == doc.id).order_by(DocumentChunk.chunk_index)))
            cov = DocCoverage(doc.id, doc.filename, chunks=len(chunks), chunks_with_numbers=sum(bool(re.search(r"\d", c.content or "")) for c in chunks))
            self._add_chunks(doc, chunks)
            if doc.file_type == "xlsx":
                tables = self._xlsx_tables(doc, chunks)
            else:
                tables = self._doc_tables(doc, chunks)
            cov.tables, cov.table_rows = len(tables), sum(t.rows for t in tables)
            coverage[doc.id] = cov
        with self._lock:
            self.conn.commit()
        return coverage

    def _add_chunks(self, doc: Document, chunks: list[DocumentChunk]) -> None:
        with self._lock:
            for c in chunks:
                where = location_of(c)
                cur = self.conn.execute("INSERT OR IGNORE INTO _chunks (chunk_id, document_id, filename, location, content) VALUES (?,?,?,?,?)",
                                        (c.id, doc.id, doc.filename, where, c.content or ""))
                if self.fts and cur.rowcount:
                    self.conn.execute("INSERT INTO _chunks_fts (rowid, content, filename) VALUES (?,?,?)", (cur.lastrowid, c.content or "", doc.filename))

    def _xlsx_tables(self, doc: Document, chunks: list[DocumentChunk]) -> list[TableInfo]:
        sheets: dict[str, list[_RowData]] = defaultdict(list)
        for chunk in chunks:
            if chunk.content_type not in ("table_row", "mixed") or not chunk.sheet_name:
                continue
            lines = {int(m.group(3)): line for line in (chunk.content or "").split("\n") if (m := _ROW_TAG.match(line))}
            cells = (chunk.metadata_ or {}).get("cells") or []
            if cells:
                by_row: dict[int, dict[str, dict]] = defaultdict(dict)
                for c in cells:
                    m = _COL_REF.match(str(c.get("ref", "")))
                    if m:
                        by_row[int(m.group(2))][m.group(1)] = {"value": c.get("value"), "formula": c.get("formula"), "header": None}
                for row, cs in by_row.items():
                    sheets[chunk.sheet_name].append(_RowData(chunk.sheet_name, row, cs, lines.get(row, ""), chunk.id))
            else:
                for row, line in sorted(lines.items()):
                    parsed = _parse_row_line(line)
                    if parsed:
                        sheets[chunk.sheet_name].append(_RowData(chunk.sheet_name, parsed[0], parsed[1], line, chunk.id))
        out: list[TableInfo] = []
        for sheet, rows in sheets.items():
            rows = sorted({r.row: r for r in rows}.values(), key=lambda r: r.row)
            blocks: list[list[_RowData]] = []
            for r in rows:
                if blocks and r.row - blocks[-1][-1].row <= 1:
                    blocks[-1].append(r)
                else:
                    blocks.append([r])
            for k, block in enumerate(blocks, start=1):
                info = self._add_block(doc, sheet, block, k, len(blocks))
                if info:
                    out.append(info)
        return out

    def _add_block(self, doc: Document, sheet: str, block: list[_RowData], k: int, total: int) -> TableInfo | None:
        letters = sorted({c for r in block for c in r.cells}, key=col_letters_to_number)
        first = block[0]
        header_text: dict[str, str] = {}
        data = block
        explicit = {c: v["header"] for r in block for c, v in r.cells.items() if v.get("header")}
        first_vals = [v.get("value") for v in first.cells.values()]
        header_like = len(block) >= 2 and first_vals and all(isinstance(v, str) and to_num(v) is None and parse_when(v) is None for v in first_vals if v not in (None, ""))
        if header_like:
            header_text = {c: str(v.get("value")).strip() for c, v in first.cells.items() if v.get("value") not in (None, "")}
            data = block[1:]
        elif explicit:
            header_text = {c: str(h) for c, h in explicit.items()}
        if not data:
            return None
        if len(block) == 1 and all(isinstance(v.get("value"), str) and to_num(v.get("value")) is None for v in first.cells.values()):
            return None  # a lone line of text (a title or a note) is not a table
        taken: set[str] = set()
        columns = [ColumnInfo(_sql_name(header_text.get(c) or f"col_{c}", taken), header_text.get(c) or f"col_{c}") for c in letters]
        title = sheet if total == 1 else f"{sheet} (block {k})"
        location = f"{sheet}!{letters[0]}{block[0].row}:{letters[-1]}{block[-1].row}"
        rows = []
        formulas = []
        name = self._new_table_name()
        for r in data:
            ref = f"{letters[0]}{r.row}:{letters[-1]}{r.row}" if not r.line else (_ROW_TAG.match(r.line).group(0).strip()[1:-1].split("!", 1)[1])
            values = []
            for c, col in zip(letters, columns):
                cell = r.cells.get(c)
                values.append(_cell_value(cell.get("value")) if cell else None)
                if cell and cell.get("formula"):
                    col.formulas += 1
                    formulas.append((name, f"{c}{r.row}", col.name, cell["formula"]))
            rows.append((ref, r.chunk_id, r.line, *values))
        return self._create(name, doc, title, "sheet", location, columns, rows, formulas)

    def _doc_tables(self, doc: Document, chunks: list[DocumentChunk]) -> list[TableInfo]:
        groups: list[list[DocumentChunk]] = []
        key_of = lambda c: ((c.metadata_ or {}).get("locator", {}).get("page"), (c.metadata_ or {}).get("locator", {}).get("table_index"), c.heading)
        for c in chunks:
            if c.content_type != "table":
                continue
            has_loc = (c.metadata_ or {}).get("locator", {}).get("table_index") is not None
            if groups and has_loc and key_of(groups[-1][-1]) == key_of(c) and groups[-1][-1].chunk_index + 1 == c.chunk_index:
                groups[-1].append(c)
            else:
                groups.append([c])
        out: list[TableInfo] = []
        for n, group in enumerate(groups, start=1):
            lines: list[tuple[str, str]] = [(line, c.id) for c in group for line in (c.content or "").split("\n") if line.strip()]
            if not lines:
                continue
            split = [([p.strip() for p in line.split(" | ")], cid, line) for line, cid in lines]
            width = max(len(s[0]) for s in split)
            header_like = len(split) >= 2 and sum(1 for p in split[0][0] if p) >= max(1, width // 2) and all(p == "" or (to_num(p) is None and parse_when(p) is None) for p in split[0][0])
            header = split[0][0] if header_like else []
            data = split[1:] if header_like else split
            if not data:
                continue
            taken: set[str] = set()
            columns = [ColumnInfo(_sql_name((header[i] if i < len(header) and header[i] else f"col_{i + 1}"), taken),
                                  header[i] if i < len(header) and header[i] else f"col_{i + 1}") for i in range(width)]
            first = group[0]
            loc = (first.metadata_ or {}).get("locator", {})
            title = f"table {loc.get('table_index') or n}" + (f", page {loc.get('page')}" if loc.get("page") else "") + (f", {first.heading}" if first.heading else "")
            rows = [(f"row {i}", cid, line, *[_cell_value(p) for p in (parts + [None] * width)[:width]]) for i, (parts, cid, line) in enumerate(data, start=1)]
            out.append(self._create(self._new_table_name(), doc, title, "doc_table", title, columns, rows, []))
        return out

    def _new_table_name(self) -> str:
        with self._lock:
            self._n += 1
            return f"t{self._n}"

    def _create(self, name: str, doc: Document, title: str, kind: str, location: str, columns: list[ColumnInfo], rows: list[tuple], formulas: list[tuple]) -> TableInfo:
        cols_sql = ", ".join(f'"{c.name}"' for c in columns)
        marks = ",".join("?" * (3 + len(columns)))
        with self._lock:
            self.conn.execute(f'CREATE TABLE "{name}" (_rid INTEGER PRIMARY KEY, _ref TEXT, _chunk TEXT, _line TEXT, {cols_sql})')
            self.conn.executemany(f'INSERT INTO "{name}" (_ref, _chunk, _line, {cols_sql}) VALUES ({marks})', rows)
            if formulas:
                self.conn.executemany("INSERT INTO _formulas VALUES (?,?,?,?)", formulas)
            info = TableInfo(name, doc.id, doc.filename, title, kind, location, len(rows), columns)
            self.conn.execute("INSERT INTO _tables VALUES (?,?,?,?,?,?,?,?)", (name, doc.id, doc.filename, title, kind, location, len(rows),
                                                                               json.dumps([[c.name, c.header, c.formulas] for c in columns])))
            self.tables[name] = info
        return info

    # -- reading -----------------------------------------------------------------------------------------------------------------------
    def table_list(self) -> list[TableInfo]:
        return list(self.tables.values())

    def schema_text(self, max_chars: int = 9000, samples: int = 2, only: set[str] | None = None, full_up_to: int = 12) -> str:
        """The tables as a worker needs to see them: name, where it came from, columns (with the header as written) and a few rows."""
        out: list[str] = []
        used = 0
        for t in self.tables.values():
            if only is not None and t.name not in only:
                continue
            cols = ", ".join(c.name if c.name == c.header.lower() else f"{c.name} ({c.header})" for c in t.columns)
            block = f"{t.name}: {t.filename}, {t.title}, {t.location}, {t.rows} rows\n  columns: {cols}"
            try:
                small = t.rows <= full_up_to  # a small table is shown in full, so a worker is never judging a table from a sample
                res = self.query(f'SELECT {", ".join(chr(34) + c.name + chr(34) for c in t.columns[:12])}, _ref FROM "{t.name}" ORDER BY _rid LIMIT {t.rows if small else samples}')
                for row in res.rows:
                    block += ("\n  row " if small else "\n  e.g. ") + " | ".join("" if v is None else str(v)[:40] for v in row[:-1]) + (f"   [_ref {row[-1]}]" if small else "")
            except ToolSqlError:
                pass
            if used + len(block) > max_chars:
                out.append(f"... {len(self.tables) - len(out)} more tables (ask for them by name)")
                break
            out.append(block)
            used += len(block)
        return "\n".join(out) if out else "(no tables)"

    def query(self, sql: str, limit: int = MAX_QUERY_ROWS, params: tuple = ()) -> QueryResult:
        stmt = (sql or "").strip().rstrip(";").strip()
        if not stmt:
            raise ToolSqlError("empty query")
        if ";" in stmt:
            raise ToolSqlError("one statement only")
        if not re.match(r"(?is)^(select|with)\b", stmt):
            raise ToolSqlError("only SELECT queries are allowed")
        conn = sqlite3.connect(f"file:{self.path.as_posix()}?mode=ro", uri=True, check_same_thread=False, timeout=30)
        try:
            _register(conn)
            allowed = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION, getattr(sqlite3, "SQLITE_RECURSIVE", 33)}
            conn.set_authorizer(lambda action, *_: sqlite3.SQLITE_OK if action in allowed else sqlite3.SQLITE_DENY)
            deadline = datetime.now().timestamp() + QUERY_SECONDS
            conn.set_progress_handler(lambda: 1 if datetime.now().timestamp() > deadline else 0, 20000)
            try:
                cur = conn.execute(stmt, params)
                rows = cur.fetchmany(limit + 1)
                columns = [d[0] for d in (cur.description or [])]
            except sqlite3.Error as exc:
                raise ToolSqlError(str(exc)) from exc
            return QueryResult(columns, [tuple(r) for r in rows[:limit]], truncated=len(rows) > limit)
        finally:
            conn.close()

    def row_source(self, table: str, ref: str) -> tuple[str, str] | None:
        """(chunk id, exact line) of a table row, or None."""
        if table not in self.tables:
            return None
        res = self.query(f'SELECT _chunk, _line FROM "{table}" WHERE _ref = ? LIMIT 1', params=(str(ref),))
        return (res.rows[0][0], res.rows[0][1]) if res.rows else None

    def column_values(self, table: str, column: str) -> list[Any]:
        return [r[0] for r in self.query(f'SELECT "{column}" FROM "{table}" ORDER BY _rid', limit=100000).rows]

    def date_columns(self, table: str, min_share: float = 0.6, min_rows: int = 3) -> list[str]:
        info = self.tables.get(table)
        if not info:
            return []
        out = []
        for col in info.columns:
            vals = [v for v in self.column_values(table, col.name)[:60] if v not in (None, "")]
            if len(vals) >= min_rows and sum(parse_when(v) is not None for v in vals) / len(vals) >= min_share:
                out.append(col.name)
        return out

    def search(self, query: str, limit: int = 5, document: str | None = None) -> list[dict]:
        """Passages that match the words of ``query`` (BM25 when full-text search exists), best first."""
        words = [w for w in re.findall(r"[A-Za-z0-9][A-Za-z0-9\-_.]*", query or "") if len(w) > 1][:12]
        if not words:
            return []
        rows: list[tuple] = []
        with self._lock:
            if self.fts:
                match = " OR ".join('"' + w.replace('"', "") + '"' for w in words)
                try:
                    rows = self.conn.execute(
                        "SELECT c.chunk_id, c.document_id, c.filename, c.location, c.content FROM _chunks_fts f JOIN _chunks c ON c.id = f.rowid "
                        "WHERE _chunks_fts MATCH ? " + ("AND c.filename = ? " if document else "") + "ORDER BY bm25(_chunks_fts) LIMIT ?",
                        (match, *( [document] if document else []), limit)).fetchall()
                except sqlite3.Error:
                    rows = []
            if not rows:
                like = " OR ".join("lower(c.content) LIKE ?" for _ in words)
                rows = self.conn.execute(f"SELECT c.chunk_id, c.document_id, c.filename, c.location, c.content FROM _chunks c WHERE ({like})"
                                         + (" AND c.filename = ?" if document else "") + " LIMIT ?",
                                         ([f"%{w.lower()}%" for w in words] + ([document] if document else []) + [limit])).fetchall()
        out = []
        for chunk_id, doc_id, filename, location, content in rows:
            hit = next((ln for ln in content.split("\n") if any(w.lower() in ln.lower() for w in words)), content[:200])
            out.append({"chunk_id": chunk_id, "document_id": doc_id, "filename": filename, "location": location, "snippet": hit.strip()[:300]})
        return out

    def chunk(self, chunk_id: str) -> tuple[str, str, str, str] | None:
        """(document id, filename, location, content) of a chunk."""
        with self._lock:
            row = self.conn.execute("SELECT document_id, filename, location, content FROM _chunks WHERE chunk_id = ?", (chunk_id,)).fetchone()
        return tuple(row) if row else None

    def chunks_of(self, filename: str) -> list[tuple[str, str, str]]:
        """(chunk id, location, content) of every chunk of a document, in order."""
        with self._lock:
            return [tuple(r) for r in self.conn.execute("SELECT chunk_id, location, content FROM _chunks WHERE filename = ? ORDER BY id", (filename,))]

    def commit(self) -> None:
        with self._lock:
            self.conn.commit()

    def execute(self, sql: str, params: tuple = ()) -> None:
        with self._lock:
            self.conn.execute(sql, params)

    def executemany(self, sql: str, rows: list[tuple]) -> None:
        with self._lock:
            self.conn.executemany(sql, rows)

    def close(self) -> None:
        with self._lock:
            self.conn.commit()
            self.conn.close()


def location_of(c: DocumentChunk) -> str:
    if c.sheet_name:
        return f"sheet {c.sheet_name}, {c.cell_reference or ''}".strip().rstrip(",")
    parts = []
    if c.page_number:
        parts.append(f"page {c.page_number}")
    if c.heading:
        parts.append(f"section '{c.heading}'")
    return ", ".join(parts) or f"chunk {c.chunk_index}"
