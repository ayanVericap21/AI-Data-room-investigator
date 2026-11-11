"""Prompt for structured claim extraction (domain agnostic: no business or industry wording). Kept separate from code so it can be reviewed."""
from __future__ import annotations

SYSTEM = """You extract structured claims from excerpts of documents. The documents can be about anything.

A claim is a checkable statement: something the excerpt says is true, with a stated value (a number,
quantity, amount, date, duration, name, yes/no, status or condition). Extract EVERY stated value that
could later be compared with another document, including values in lists, specification bullets, tables
and short fragments. Do not skip a value because it looks minor. Skip only headings, navigation text and
sentences that state no value.

Rules:
- Use ONLY information stated in the excerpt. Never infer or invent values, units, periods,
  conditions or assumptions. If an attribute is not stated, use null (or [] for assumptions).
- If the same quantity or attribute is stated more than once in an excerpt, extract EVERY mention as its own
  claim, even when the statements repeat or disagree. Never merge mentions and never keep only the first one.
- "subject" is what the value is about (the thing, item, party or section it belongs to). "metric" is the
  quantity or attribute that has the value. Copy both from the wording of the excerpt; do not translate
  them into your own categories.
- "value" must be the number or text exactly as stated (e.g. 30000 for "30,000"). Never calculate or
  convert a value. For a range, give the range as written.
- "conditions" holds qualifiers that limit the claim (a scope, scenario, version or "under ..."
  clause); "assumptions" holds explicit assumptions the claim depends on.
- "claim_text" is a short faithful restatement of the statement in the excerpt.
- Each claim must reference the "chunk_id" of the excerpt it came from, copied exactly.
- Label consistency: you may be shown subject and metric labels already used for earlier claims. When a claim
  is about the same thing as an existing label, use that label EXACTLY as written. Create a new label only
  when no existing label names the same thing. Never reuse a label for a different thing.
- claim_category: a short free-text word for the kind of statement, or null.
- confidence (0-1): how clearly the excerpt states the claim.
- If an excerpt contains no claims, return an empty list."""


def build_prompt(filename: str, excerpts: list[dict], subjects: list[str] | None = None,
                 metrics: list[str] | None = None) -> str:
    """excerpts: [{"chunk_id", "location", "content"}]. subjects / metrics: labels already used in this run."""
    blocks = []
    for e in excerpts:
        blocks.append(f"[chunk_id: {e['chunk_id']}] [location: {e['location']}]\n{e['content']}")
    joined = "\n\n---\n\n".join(blocks)
    vocabulary = ""
    if subjects or metrics:
        vocabulary = ("Labels already used for earlier claims (reuse one exactly when it names the same thing):\n"
                      f"subjects: {'; '.join(subjects or []) or '(none yet)'}\n"
                      f"metrics: {'; '.join(metrics or []) or '(none yet)'}\n\n")
    return f"Document: {filename}\n\n{vocabulary}Excerpts:\n\n{joined}\n\nExtract every claim."


def build_coverage_prompt(filename: str, chunk: dict, uncovered: list[tuple[str, str]]) -> str:
    """A follow-up for one excerpt: numbers it states that no extracted claim accounts for.
    chunk: {"chunk_id", "location", "content"}; uncovered: [(number as written, text around it)]."""
    numbers = "\n".join(f"- {n}: ...{context}..." for n, context in uncovered)
    return (f"Document: {filename}\n\nExcerpt:\n\n[chunk_id: {chunk['chunk_id']}] [location: {chunk['location']}]\n{chunk['content']}\n\n"
            "A first pass over this excerpt did not produce a claim for the numbers below. For each one that is part of a "
            "stated value (a quantity, amount, date, duration, measure or similar fact about something), extract a claim for it. "
            "Ignore numbers that are only labels, page or section numbers, or identifiers. If none of them states a value, "
            f"return an empty list.\n\nNumbers without a claim:\n{numbers}")
