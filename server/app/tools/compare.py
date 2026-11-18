"""compare_values: the deterministic comparison of two values as written (scale words, units of one dimension, dates, names), so the agent never normalises by hand."""
from __future__ import annotations

from app.services.value_rules import comparison_note, compare_values
from app.tools.base import ToolContext
from app.tools.schemas import CompareValuesInput, CompareValuesOutput


def compare(_ctx: ToolContext, inp: CompareValuesInput) -> CompareValuesOutput:
    relation = compare_values(inp.a, inp.unit_a, inp.b, inp.unit_b, inp.value_type, inp.value_type)
    return CompareValuesOutput(relation=relation, note=comparison_note(inp.a, inp.unit_a, inp.b, inp.unit_b, inp.value_type, inp.value_type))
