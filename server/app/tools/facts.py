"""record_fact: the agent's running memory of what it has read.

Facts live in the run's context (in memory, this run only), not in the prompt, so they
survive however much older tool output is shortened. Every quote is verified against the
stored chunk text, exactly as in ``submit_finding``. Nothing here decides what a fact is or
how facts relate: the agent records, and a later step compares.
"""
from __future__ import annotations

from sqlalchemy import select

from app.models import Document, DocumentChunk
from app.tools.base import ToolContext, ToolError
from app.tools.reading import locator_text
from app.tools.schemas import RecordedFact, RecordFactInput, RecordFactOutput
from app.tools.submit import find_source_span


def record_fact(ctx: ToolContext, inp: RecordFactInput) -> RecordFactOutput:
    chunk_ids = {f.chunk_id for f in inp.facts}
    rows = {
        chunk.id: (chunk, filename)
        for chunk, filename in ctx.session.execute(
            select(DocumentChunk, Document.filename).join(Document, Document.id == DocumentChunk.document_id)
            .where(DocumentChunk.id.in_(chunk_ids))
        )
    }

    problems: list[str] = []
    accepted: list[RecordedFact] = []
    for i, fact in enumerate(inp.facts, start=1):
        if fact.chunk_id not in rows:
            problems.append(f"fact {i}: chunk '{fact.chunk_id}' does not exist (use ids from search or read_section)")
            continue
        chunk, filename = rows[fact.chunk_id]
        span = find_source_span(chunk.content, fact.quoted_text)
        if span is None:
            problems.append(f"fact {i}: the quote is not in chunk '{fact.chunk_id}' ({filename}, {locator_text(chunk)}). "
                            "Copy the text exactly from that chunk's content.")
            continue
        accepted.append(RecordedFact(
            fact_id=f"fact_{len(ctx.facts) + len(accepted) + 1}", label=fact.label, value=fact.value, unit=fact.unit,
            qualifier=fact.qualifier, document_id=chunk.document_id, filename=filename, chunk_id=chunk.id,
            locator=locator_text(chunk), quoted_text=span[0],
        ))
    if problems:
        raise ToolError("validation_failed", "No facts were recorded. " + "; ".join(problems))

    ctx.facts.extend(accepted)
    return RecordFactOutput(recorded=[f.fact_id for f in accepted], total_facts=len(ctx.facts))
