"""MCP server exposing the ingested data room as read-only tools (stdio transport).

Scope is set through environment variables so an investigation can only see its own documents:
  DRI_DATABASE_URL (+ optional DRI_DB_SCHEMA) - the database;  DRI_DOC_IDS - optional comma-separated document ids.
"""
from __future__ import annotations

import os
from typing import Any

from mcp.server import MCPServer

from ..config import get_settings
from ..db import Database
from ..tools import DataRoomTools


def build_server(tools: DataRoomTools) -> MCPServer:
    mcp = MCPServer("data-room", instructions=(
        "Read-only access to an ingested data room. Use search_chunks to find passages, get_neighbors for context, "
        "verify_quote to confirm a quotation really exists in a passage before citing it."))

    @mcp.tool()
    def list_documents() -> dict[str, Any]:
        """List processed documents with their summaries (overview, topics, key figures) - orientation, not evidence."""
        return tools.list_documents()

    @mcp.tool()
    def get_document_summary(document_id: int) -> dict[str, Any]:
        """Summary of one document. Summaries are NOT evidence: cite passages found with search_chunks/get_chunk."""
        return tools.get_document_summary(document_id)

    @mcp.tool()
    def search_chunks(query: str, limit: int = 5, document_id: int | None = None) -> dict[str, Any]:
        """Keyword (BM25) search over passages. Optionally restrict to one document."""
        return tools.search_chunks(query, limit, document_id)

    @mcp.tool()
    def get_chunk(chunk_id: int) -> dict[str, Any]:
        """Fetch one passage by id, with filename and page."""
        return tools.get_chunk(chunk_id)

    @mcp.tool()
    def get_neighbors(chunk_id: int, window: int = 1) -> dict[str, Any]:
        """Fetch the passages immediately before/after a passage (window <= 3) for surrounding context."""
        return tools.get_neighbors(chunk_id, window)

    @mcp.tool()
    def verify_quote(chunk_id: int, quote: str) -> dict[str, Any]:
        """Check that a quotation exists verbatim (whitespace/case-insensitive) in the cited passage."""
        return tools.verify_quote(chunk_id, quote)

    return mcp


def main() -> None:
    settings = get_settings()
    ids = [int(x) for x in os.environ.get("DRI_DOC_IDS", "").split(",") if x.strip()]
    db = Database(settings.require_database_url(), settings.db_schema, init=False)  # schema is owned by the API
    build_server(DataRoomTools(db, ids or None)).run("stdio")
