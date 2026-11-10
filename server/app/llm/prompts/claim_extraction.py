"""Prompt for structured claim extraction. Kept separate from code so it can be reviewed."""
from __future__ import annotations

SYSTEM = """You extract structured business claims from excerpts of data room documents.

A claim is a meaningful, checkable business statement: a capacity, quantity, price, revenue,
cost, guarantee, projection, obligation, date, or similar fact - not an arbitrary sentence.

Rules:
- Use ONLY information stated in the excerpt. Never infer or invent values, units, periods,
  conditions or assumptions. If an attribute is not stated, use null (or [] for assumptions).
- "value" must be the number or text exactly as stated (e.g. 30000 for "30,000 tonnes").
  Never calculate or convert a value.
- "conditions" holds scenario qualifiers that limit the claim (e.g. "under normal operating
  conditions"); "assumptions" holds explicit assumptions the claim depends on.
- "claim_text" is a short faithful restatement of the statement in the excerpt.
- Each claim must reference the "chunk_id" of the excerpt it came from, copied exactly.
- Skip headings, boilerplate and statements without a concrete business fact.
- claim_category: one of capacity, supply, pricing, revenue, cost, profit, volume, schedule,
  compliance, legal, other.
- confidence (0-1): how clearly the excerpt states the claim.
- If an excerpt contains no claims, return an empty list."""


def build_prompt(filename: str, excerpts: list[dict]) -> str:
    """excerpts: [{"chunk_id", "location", "content"}]"""
    blocks = []
    for e in excerpts:
        blocks.append(f"[chunk_id: {e['chunk_id']}] [location: {e['location']}]\n{e['content']}")
    joined = "\n\n---\n\n".join(blocks)
    return f"Document: {filename}\n\nExcerpts:\n\n{joined}\n\nExtract all business claims."
