"""XLSX extraction with openpyxl.

The workbook is opened twice: once with formulas (``data_only=False``) and once with
cached results (``data_only=True``). openpyxl never calculates formulas, so a formula
cell's value is only what Excel last saved. If no cached value exists we say so
explicitly instead of treating the formula as ordinary text or guessing a result.

One unit is emitted per non-empty row. The first non-empty row of each sheet is used
as the column-header context for the rows below it (a heuristic, recorded in metadata).
"""
from __future__ import annotations

import time
from datetime import date, datetime
from datetime import time as time_of_day
from pathlib import Path

import openpyxl
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.formula import ArrayFormula, DataTableFormula

from app.core.exceptions import ExtractionError
from app.schemas.extraction import ExtractedUnit, ExtractionReport, Locator, XlsxCell
from app.services.extractors import lap


def count_sheets(path: Path) -> int | None:
    try:
        wb = openpyxl.load_workbook(path, read_only=True)
        try:
            return len(wb.sheetnames)
        finally:
            wb.close()
    except Exception:
        return None


def _formula_text(value: object) -> str | None:
    """The formula as text, for ordinary formulas ("=B2*C2") and Excel array / data-table formulas, else None."""
    if isinstance(value, str):
        return value if value.startswith("=") else None
    if isinstance(value, ArrayFormula):
        text = value.text or ""
        return text if text.startswith("=") else f"={text}" if text else f"{{array formula {value.ref}}}"
    if isinstance(value, DataTableFormula):
        return f"{{data table formula {value.ref}}}"
    return None


def _json_safe(value: object) -> object:
    """Cell values go into a JSON column: keep numbers, text and booleans, write dates as ISO text,
    anything else as its text."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (datetime, date, time_of_day)):
        return value.isoformat()
    return str(value)


def extract_xlsx(path: Path, document_id: str, filename: str, report: ExtractionReport | None = None) -> list[ExtractedUnit]:
    """``report`` (optional) receives milliseconds per part: load_formulas, load_values, read_rows."""
    try:
        t = time.perf_counter()
        wb_f = openpyxl.load_workbook(path, data_only=False)
        lap(report, "load_formulas", t)
        t = time.perf_counter()
        wb_v = openpyxl.load_workbook(path, data_only=True)
        lap(report, "load_values", t)
    except Exception as exc:
        raise ExtractionError(f"Cannot open XLSX '{filename}': {exc}") from exc

    units: list[ExtractedUnit] = []
    t_rows = time.perf_counter()
    try:
        for ws_f in wb_f.worksheets:
            ws_v = wb_v[ws_f.title]
            headers: dict[int, str] = {}
            header_row_no: int | None = None
            for row in ws_f.iter_rows():
                if all(c.value is None for c in row):
                    continue
                row_no = row[0].row
                if header_row_no is None:
                    header_row_no = row_no
                    headers = {c.column: str(c.value).strip() for c in row if c.value is not None}
                    # Header row is itself emitted so its text is searchable.
                cells: list[XlsxCell] = []
                parts: list[str] = []
                for c in row:
                    if c.value is None:
                        continue
                    cached = _json_safe(ws_v[c.coordinate].value)
                    formula = _formula_text(c.value)
                    plain = None if formula else _json_safe(c.value)
                    header = headers.get(c.column) if row_no != header_row_no else None
                    cell = XlsxCell(
                        ref=c.coordinate, column_header=header,
                        value=cached if formula else plain,
                        formula=formula,
                        cached_value_available=(cached is not None) if formula else True,
                    )
                    cells.append(cell)
                    label = f"{header}: " if header else f"{c.coordinate}: "
                    if formula:
                        shown = f"{cached}" if cached is not None else "cached value not available"
                        parts.append(f"{label}{formula} (= {shown})" if cached is not None
                                     else f"{label}{formula} ({shown})")
                    else:
                        parts.append(f"{label}{plain}")
                first = get_column_letter(row[0].column)
                last = get_column_letter(row[-1].column)
                ref = f"{first}{row_no}:{last}{row_no}"
                units.append(ExtractedUnit(
                    document_id=document_id, filename=filename, source_type="xlsx",
                    locator=Locator(sheet_name=ws_f.title, cell_reference=ref),
                    content=f"[{ws_f.title}!{ref}] " + "; ".join(parts),
                    content_type="table_row",
                    metadata={
                        "cells": [c.model_dump() for c in cells],
                        "is_header_row": row_no == header_row_no,
                        "header_row": header_row_no,
                    },
                ))
    finally:
        lap(report, "read_rows", t_rows)
        wb_f.close()
        wb_v.close()
    return units
