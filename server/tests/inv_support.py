"""Helpers for the investigator tests: a model that answers by schema, and a small data room built in the test database."""
from __future__ import annotations

from typing import Any

from app.investigator.tables import TableStore
from app.llm.base import LLMClient
from app.llm.errors import LLMError
from tests.test_tools import add_doc


class RoutedLLM(LLMClient):
    """Answers each structured request with the handler registered for its schema: a value, a list consumed in order, or a function of the prompt."""

    def __init__(self, handlers: dict[type, Any]):
        super().__init__(max_attempts=1)
        self.handlers = handlers
        self.calls: list[tuple[type, str]] = []

    def _raw_generate(self, prompt, system, json_mode):  # never used: generate_structured_output is replaced
        raise AssertionError("RoutedLLM answers structured requests only")

    def generate_structured_output(self, prompt, schema, system=None):
        self.calls.append((schema, prompt))
        handler = self.handlers.get(schema)
        if handler is None:
            raise LLMError(f"no handler for {schema.__name__}")
        if isinstance(handler, list):
            handler = handler.pop(0) if handler else None
            if handler is None:
                raise LLMError(f"handler list for {schema.__name__} is exhausted")
        result = handler(prompt) if callable(handler) else handler
        return result if isinstance(result, schema) else schema.model_validate(result)

    def count(self, schema: type) -> int:
        return sum(1 for s, _ in self.calls if s is schema)


def xlsx_chunk(sheet: str, rows: dict[int, list[tuple[str | None, Any]]], with_cells: bool = True) -> dict:
    """A merged spreadsheet chunk: rows maps a row number to [(header-or-None, value), ...] from column A. A header of None labels the cell with its
    coordinate, which is how the real extractor writes the header row."""
    lines, cells = [], []
    for row, cols in rows.items():
        last = chr(64 + len(cols))
        parts = []
        for i, (header, value) in enumerate(cols):
            ref = f"{chr(65 + i)}{row}"
            parts.append(f"{header or ref}: {value}")
            cells.append({"ref": ref, "column_header": header, "value": value, "formula": None, "cached_value_available": True})
        lines.append(f"[{sheet}!A{row}:{last}{row}] " + "; ".join(parts))
    first, last_row = min(rows), max(rows)
    return dict(content="\n".join(lines), content_type="table_row", sheet_name=sheet, cell_reference=f"A{first}:A{last_row}",
                metadata_={"cells": cells} if with_cells else {})


def dive_room(session):
    """A permit (docx), a dive register (xlsx) and a lab report (pdf) with a few planted differences."""
    permit, _ = add_doc(session, "permit.docx", "docx", [
        dict(content="Sampling is permitted between 3 March 2025 and 21 March 2025.", content_type="paragraph", heading="Conditions"),
        dict(content="No more than 8 fragments may be collected per site.", content_type="paragraph", heading="Conditions"),
        dict(content="The decontamination log must be submitted with the final report.", content_type="paragraph", heading="Conditions"),
    ])
    rows = {1: [(None, "Dive"), (None, "Date"), (None, "Site"), (None, "Samples")]}
    data = [("D-01", "2025-03-04", "IR-01", 6), ("D-02", "2025-03-10", "IR-02", 7), ("D-03", "2025-03-22", "IR-03", 5), ("D-04", "2025-03-12", "IR-01", 9),
            ("D-05", "2025-03-14", "IR-02", 4), ("D-06", "2025-03-15", "IR-03", 3)]
    for i, (dive, date, site, n) in enumerate(data, start=2):
        rows[i] = [("Dive", dive), ("Date", date), ("Site", site), ("Samples", n)]
    register, _ = add_doc(session, "register.xlsx", "xlsx", [xlsx_chunk("Dives", rows)])
    lab, _ = add_doc(session, "lab.pdf", "pdf", [
        dict(content="Samples received: SPL-001 to SPL-006 and SPL-044-B. Sample SPL-044-B arrived without field metadata.", content_type="paragraph", page_number=1),
        dict(content="Mean sample count per dive was 5.7.", content_type="paragraph", page_number=2),
    ])
    session.commit()
    return permit, register, lab


def build_store(session, tmp_path, only=None, name="t.sqlite") -> TableStore:
    store = TableStore(tmp_path / name)
    store.build(session, only=only)
    return store


_fact_n = [0]


def make_fact(store, filename, quote, type="observation", **kw):
    """A Fact tied to the chunk of ``filename`` that holds ``quote``."""
    from app.investigator.models import Fact
    chunk_id = next(cid for cid, _where, content in store.chunks_of(filename) if quote in content)
    _fact_n[0] += 1
    defaults = dict(subject="", attribute="", operator="=", value="", unit=None, scope=None, when=None, modality=None, aggregate_of=None, value_type=None)
    defaults.update(kw)
    return Fact(id=f"f{_fact_n[0]:04d}", document_id=store.chunk(chunk_id)[0], chunk_id=chunk_id, type=type, quote=quote, **defaults)
