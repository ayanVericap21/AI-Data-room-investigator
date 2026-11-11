"""Database access. Backend is chosen by URL scheme:

  postgresql://...   the real database (docker compose `db` service). Full-text search = tsvector + GIN.
  sqlite:///path     zero-setup backend used by the default test run. Full-text search = FTS5.

All application SQL is written with `?` placeholders; they are translated to `%s` for Postgres. One connection per
operation keeps this thread-safe for FastAPI background threads. `schema` (Postgres only) isolates a Database inside its
own schema - used by tests and the eval harness so they never touch real data.
"""
from __future__ import annotations

import json
import re
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

_SCHEMA_RE = re.compile(r"^[a-z_][a-z0-9_]{0,62}$")

_COMMON = """
CREATE TABLE IF NOT EXISTS documents(
  id {pk}, filename TEXT NOT NULL, stored_path TEXT NOT NULL,
  sha256 TEXT NOT NULL UNIQUE, size_bytes BIGINT NOT NULL, status TEXT NOT NULL,
  failure_code TEXT, error TEXT, page_count INTEGER DEFAULT 0, chunk_count INTEGER DEFAULT 0,
  summary TEXT, summary_status TEXT NOT NULL DEFAULT 'pending', summary_error TEXT,
  created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS chunks(
  id {pk}, document_id BIGINT NOT NULL REFERENCES documents(id),
  ordinal INTEGER NOT NULL, page INTEGER NOT NULL, char_start INTEGER NOT NULL,
  char_end INTEGER NOT NULL, text TEXT NOT NULL{chunk_extra});
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(document_id, ordinal);
CREATE TABLE IF NOT EXISTS ingest_jobs(
  id {pk}, status TEXT NOT NULL, document_ids TEXT NOT NULL,
  idempotency_key TEXT NOT NULL UNIQUE, results TEXT NOT NULL DEFAULT '[]', usage TEXT NOT NULL DEFAULT '{{}}',
  error TEXT, created_at TEXT NOT NULL, started_at TEXT, finished_at TEXT);
CREATE TABLE IF NOT EXISTS investigations(
  id {pk}, status TEXT NOT NULL, phase TEXT NOT NULL, document_ids TEXT NOT NULL,
  error TEXT, usage TEXT NOT NULL DEFAULT '{{}}', created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS claims(
  id {pk}, investigation_id BIGINT NOT NULL, document_id BIGINT NOT NULL,
  chunk_id BIGINT NOT NULL, subject TEXT NOT NULL, value TEXT NOT NULL, unit TEXT,
  conditions TEXT, quote TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS hypotheses(
  id {pk}, investigation_id BIGINT NOT NULL, claim_ids TEXT NOT NULL,
  question TEXT NOT NULL, rationale TEXT NOT NULL DEFAULT '', status TEXT NOT NULL,
  attempts INTEGER NOT NULL DEFAULT 0, clarifications TEXT NOT NULL DEFAULT '[]');
CREATE TABLE IF NOT EXISTS findings(
  id {pk}, investigation_id BIGINT NOT NULL, hypothesis_id BIGINT,
  data TEXT NOT NULL, superseded INTEGER NOT NULL DEFAULT 0);
CREATE TABLE IF NOT EXISTS events(
  id {pk}, investigation_id BIGINT NOT NULL, ts TEXT NOT NULL,
  kind TEXT NOT NULL, message TEXT NOT NULL, data TEXT NOT NULL DEFAULT '{{}}');
CREATE INDEX IF NOT EXISTS idx_events_inv ON events(investigation_id, id);
"""

SQLITE_SCHEMA = _COMMON.format(pk="INTEGER PRIMARY KEY", chunk_extra="") + \
    "CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(text);"

PG_SCHEMA = _COMMON.format(
    pk="BIGSERIAL PRIMARY KEY",
    chunk_extra=", tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED",
) + "CREATE INDEX IF NOT EXISTS idx_chunks_tsv ON chunks USING GIN(tsv);"


class _Conn:
    """Uniform connection facade: `?` placeholders on both backends, rows addressable by column name."""

    def __init__(self, raw: Any, pg: bool):
        self.raw, self.pg = raw, pg

    def execute(self, sql: str, params: tuple | list = ()):
        return self.raw.execute(sql.replace("?", "%s") if self.pg else sql, tuple(params))


class Database:
    def __init__(self, url: str | Path, schema: str | None = None, init: bool = True):
        url = str(url)
        if not url:
            raise ValueError("database url is empty")
        self.url, self.schema = url, schema
        self.pg = url.startswith(("postgresql://", "postgres://"))
        if self.pg:
            if schema and not _SCHEMA_RE.match(schema):
                raise ValueError(f"invalid schema name {schema!r}")
        elif url.startswith("sqlite:///"):
            self._sqlite_path = Path(url[len("sqlite:///"):])
            self._sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            raise ValueError("database url must start with postgresql:// or sqlite:///")
        if init:
            self._init_schema()

    # ---------- connections ----------
    def _raw(self):
        if self.pg:
            import psycopg
            from psycopg.rows import dict_row

            opts = f"-c search_path={self.schema},public" if self.schema else None
            return psycopg.connect(self.url, row_factory=dict_row, options=opts)
        c = sqlite3.connect(self._sqlite_path, timeout=30)
        c.row_factory = sqlite3.Row
        c.execute("PRAGMA foreign_keys=ON")
        return c

    @contextmanager
    def conn(self) -> Iterator[_Conn]:
        raw = self._raw()
        try:
            yield _Conn(raw, self.pg)
            raw.commit()
        except BaseException:
            raw.rollback()
            raise
        finally:
            raw.close()

    def _init_schema(self) -> None:
        with self.conn() as c:
            if self.pg:
                # Concurrent start-up of API + MCP server must not race on CREATE ... IF NOT EXISTS.
                c.execute("SELECT pg_advisory_xact_lock(727401)")
                if self.schema:
                    c.execute(f"CREATE SCHEMA IF NOT EXISTS {self.schema}")
                c.raw.execute(PG_SCHEMA)
            else:
                c.raw.executescript(SQLITE_SCHEMA)

    def drop_schema(self) -> None:
        """Postgres only: remove this Database's schema (tests / eval cleanup). Refuses to touch `public`."""
        if not (self.pg and self.schema):
            return
        with self.conn() as c:
            c.execute(f"DROP SCHEMA IF EXISTS {self.schema} CASCADE")

    # ---------- generic helpers ----------
    def execute(self, sql: str, params: tuple | list = ()) -> int:
        """Run a statement. For INSERT returns the new row id; otherwise 0."""
        is_insert = sql.lstrip().upper().startswith("INSERT")
        with self.conn() as c:
            if self.pg and is_insert and "RETURNING" not in sql.upper():
                row = c.execute(sql + " RETURNING id", params).fetchone()
                return int(row["id"])
            cur = c.execute(sql, params)
            return int(cur.lastrowid or 0) if (is_insert and not self.pg) else 0

    def query(self, sql: str, params: tuple | list = ()) -> list[Any]:
        with self.conn() as c:
            return c.execute(sql, params).fetchall()

    def one(self, sql: str, params: tuple | list = ()) -> Any | None:
        rows = self.query(sql, params)
        return rows[0] if rows else None

    # ---------- backend-specific chunk storage / search ----------
    def replace_chunks(self, document_id: int, chunks: list[Any]) -> None:
        """Atomically replace a document's chunks (re-ingest safe). `chunks` have page/char_start/char_end/text."""
        with self.conn() as c:
            if not self.pg:
                for r in c.execute("SELECT id FROM chunks WHERE document_id=?", (document_id,)).fetchall():
                    c.execute("DELETE FROM chunks_fts WHERE rowid=?", (r["id"],))
            c.execute("DELETE FROM chunks WHERE document_id=?", (document_id,))
            for i, ch in enumerate(chunks):
                sql = ("INSERT INTO chunks(document_id, ordinal, page, char_start, char_end, text) VALUES(?,?,?,?,?,?)")
                args = (document_id, i, ch.page, ch.char_start, ch.char_end, ch.text)
                if self.pg:
                    c.execute(sql, args)  # tsv is a generated column: indexed automatically
                else:
                    cid = c.execute(sql, args).lastrowid
                    c.execute("INSERT INTO chunks_fts(rowid, text) VALUES(?,?)", (cid, ch.text))

    def search_chunks(self, tokens: list[str], limit: int, document_id: int | None = None) -> list[Any]:
        """Ranked keyword search (OR of tokens). Tokens must already be sanitised to word characters."""
        extra, params = "", []
        if document_id is not None:
            extra, params = " AND c.document_id=?", [document_id]
        if self.pg:
            q = " | ".join(tokens)
            sql = ("SELECT c.id, c.document_id, c.ordinal, c.page, c.text, d.filename FROM chunks c "
                   "JOIN documents d ON d.id=c.document_id WHERE c.tsv @@ to_tsquery('english', ?)" + extra +
                   " ORDER BY ts_rank(c.tsv, to_tsquery('english', ?)) DESC, c.id LIMIT ?")
            return self.query(sql, [q, *params, q, limit])
        q = " OR ".join(f'"{t}"' for t in tokens)
        sql = ("SELECT c.*, d.filename FROM chunks_fts f JOIN chunks c ON c.id=f.rowid "
               "JOIN documents d ON d.id=c.document_id WHERE chunks_fts MATCH ?" + extra + " ORDER BY rank LIMIT ?")
        return self.query(sql, [q, *params, limit])


def dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False)


def loads(s: str | None, default: Any = None) -> Any:
    return json.loads(s) if s else default
