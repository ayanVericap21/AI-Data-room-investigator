"""Minimal pgvector column type (no extra dependency).

On PostgreSQL the column is ``vector(N)`` and values are sent as text literals cast by
the server. On other dialects (SQLite in unit tests) it degrades to JSON so the rest of
the schema can still be exercised; similarity search itself is PostgreSQL-only.
"""
from __future__ import annotations

from sqlalchemy import JSON
from sqlalchemy.types import TypeDecorator, UserDefinedType


class PgVector(UserDefinedType):
    cache_ok = True

    def __init__(self, dimensions: int):
        self.dimensions = dimensions

    def get_col_spec(self, **kw) -> str:
        return f"vector({self.dimensions})"

    def bind_processor(self, dialect):
        def process(value):
            return None if value is None else vector_literal(value)
        return process

    def result_processor(self, dialect, coltype):
        def process(value):
            if value is None or isinstance(value, list):
                return value
            return [float(x) for x in value.strip("[]").split(",")]
        return process


def vector_literal(values: list[float]) -> str:
    return "[" + ",".join(repr(float(v)) for v in values) + "]"


class Embedding(TypeDecorator):
    """vector(N) on PostgreSQL, JSON elsewhere."""

    impl = JSON
    cache_ok = True

    def __init__(self, dimensions: int):
        super().__init__()
        self.dimensions = dimensions

    def load_dialect_impl(self, dialect):
        if dialect.name == "postgresql":
            return dialect.type_descriptor(PgVector(self.dimensions))
        return dialect.type_descriptor(JSON())
