"""search: semantic passage search over the stored chunk vectors (pgvector)."""
from __future__ import annotations

from app.core.exceptions import AppError
from app.llm.errors import LLMError
from app.tools.base import ToolContext, ToolError
from app.tools.reading import locator_text
from app.tools.schemas import ChunkView, SearchInput, SearchOutput


def search(ctx: ToolContext, inp: SearchInput) -> SearchOutput:
    if ctx.retrieval is None:
        raise ToolError("search_unavailable", "Search is not configured for this run.")
    try:
        hits = ctx.retrieval.search(
            inp.query, top_k=inp.top_k, document_ids=inp.document_ids, exclude_document_ids=inp.exclude_document_ids,
        )
    except AppError as exc:
        raise ToolError("search_unavailable", str(exc)) from exc
    except LLMError as exc:
        raise ToolError("embedding_failed", f"Could not embed the query: {exc}") from exc
    ctx.recent_queries.append(inp.query)
    for h in hits:
        ctx.mark_read(h.document_id, [h.chunk_id])  # a hit shows the chunk's full text
    return SearchOutput(query=inp.query, hits=[
        ChunkView(
            chunk_id=h.chunk_id, document_id=h.document_id, filename=h.filename, locator=locator_text(h),
            content=h.content, content_type=h.content_type, page_number=h.page_number, sheet_name=h.sheet_name,
            cell_reference=h.cell_reference, heading=h.heading, score=round(h.score, 4),
        )
        for h in hits
    ])
