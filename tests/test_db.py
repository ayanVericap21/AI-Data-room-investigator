"""Database layer: URL handling, Postgres SQL generation (checked without a server), isolation helpers."""
import pytest

from app.config import Settings
from app.db import Database, _Conn


def test_invalid_urls():
    with pytest.raises(ValueError):
        Database("mysql://x", init=False)
    with pytest.raises(ValueError):
        Database("", init=False)
    with pytest.raises(ValueError):
        Database("postgresql://u@h/db", schema="bad; DROP SCHEMA public", init=False)


def test_missing_database_url_is_a_clear_error():
    with pytest.raises(RuntimeError, match="docker compose up -d db"):
        Settings().require_database_url()


def test_drop_schema_is_noop_without_a_schema(tmp_path):
    db = Database(f"sqlite:///{tmp_path / 'x.sqlite3'}")
    db.drop_schema()  # sqlite: nothing to do, and must never raise
    pg_public = Database("postgresql://u@localhost/db", init=False)
    pg_public.drop_schema()  # no schema -> refuses to touch `public`


class _Raw:
    def __init__(self):
        self.calls = []

    def execute(self, sql, params=()):
        self.calls.append((sql, params))
        return self

    def fetchone(self):
        return {"id": 7}

    def fetchall(self):
        return []

    def commit(self): ...
    def rollback(self): ...
    def close(self): ...


def _pg_with_stub():
    db = Database("postgresql://u@localhost/db", schema="t_abc", init=False)
    raw = _Raw()
    db._raw = lambda: raw
    return db, raw


def test_postgres_placeholders_and_insert_returning():
    db, raw = _pg_with_stub()
    assert db.execute("INSERT INTO documents(filename, sha256) VALUES(?,?)", ("a", "b")) == 7
    sql, params = raw.calls[-1]
    assert sql.endswith("RETURNING id") and "?" not in sql and sql.count("%s") == 2 and params == ("a", "b")
    db.execute("UPDATE documents SET status=? WHERE id=?", ("x", 1))
    assert raw.calls[-1][0] == "UPDATE documents SET status=%s WHERE id=%s"


def test_postgres_search_sql_uses_tsvector_and_orders_params():
    db, raw = _pg_with_stub()
    db.search_chunks(["capacity", "tonnes"], 5, document_id=3)
    sql, params = raw.calls[-1]
    assert "tsv @@ to_tsquery('english', %s)" in sql and "ts_rank" in sql and "document_id=%s" in sql
    assert list(params) == ["capacity | tonnes", 3, "capacity | tonnes", 5]
    db.search_chunks(["x"], 2)
    assert list(raw.calls[-1][1]) == ["x", "x", 2]


def test_conn_translates_only_for_postgres():
    raw = _Raw()
    _Conn(raw, pg=True).execute("SELECT 1 WHERE a=? AND b=?", [1, 2])
    _Conn(raw, pg=False).execute("SELECT 1 WHERE a=?", [1])
    assert raw.calls[0] == ("SELECT 1 WHERE a=%s AND b=%s", (1, 2)) and raw.calls[1][0] == "SELECT 1 WHERE a=?"


def test_postgres_schema_script_is_valid_looking():
    from app.db import PG_SCHEMA
    assert "BIGSERIAL PRIMARY KEY" in PG_SCHEMA and "GENERATED ALWAYS AS (to_tsvector('english', text)) STORED" in PG_SCHEMA
    assert "USING GIN(tsv)" in PG_SCHEMA and "AUTOINCREMENT" not in PG_SCHEMA and "fts5" not in PG_SCHEMA
    assert "{" not in PG_SCHEMA.replace("'{}'", "")  # all format placeholders substituted
