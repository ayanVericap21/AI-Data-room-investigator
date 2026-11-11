"""Deterministic data-room tools. The MCP server and the agent's local client both wrap this class,
so retrieval and citation verification have exactly one implementation.
"""
from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel

from .db import Database
from .models import Chunk

_WS = re.compile(r"\s+")
_QUOTE_CHARS = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"', "–": "-", "—": "-", " ": " "})


def normalize(text: str) -> str:
    """Whitespace/case/typographic-quote insensitive form used for quote matching."""
    return _WS.sub(" ", text.translate(_QUOTE_CHARS)).strip().casefold()


def quote_in_text(quote: str, text: str) -> bool:
    q = normalize(quote)
    return len(q) >= 8 and q in normalize(text)  # tiny quotes would match anything


class ChunkView(BaseModel):
    chunk_id: int
    document_id: int
    filename: str
    page: int
    ordinal: int
    text: str


class VerifyResult(BaseModel):
    found: bool
    chunk_id: int | None = None  # chunk where the quote actually lives
    relocated: bool = False      # True if the claimed chunk was wrong but the quote exists elsewhere in the same document
    reason: str


class DataRoomTools:
    def __init__(self, db: Database, allowed_document_ids: list[int] | None = None):
        self.db = db
        self.allowed = set(allowed_document_ids) if allowed_document_ids else None

    def _ok(self, doc_id: int) -> bool:
        return self.allowed is None or doc_id in self.allowed

    def _view(self, r) -> ChunkView:
        return ChunkView(chunk_id=r["id"], document_id=r["document_id"], filename=r["filename"], page=r["page"],
                         ordinal=r["ordinal"], text=r["text"])

    # ---- tools ----
    def _summary_of(self, r) -> dict[str, Any] | None:
        return json.loads(r["summary"]) if r["summary"] else None

    def list_documents(self) -> dict[str, Any]:
        """Processed documents with their summaries (orientation only - never cite a summary as evidence)."""
        rows = self.db.query("SELECT id, filename, page_count, chunk_count, summary, summary_status FROM documents "
                             "WHERE status='processed' ORDER BY id")
        return {"documents": [{"id": r["id"], "filename": r["filename"], "page_count": r["page_count"],
                               "chunk_count": r["chunk_count"], "summary_status": r["summary_status"],
                               "summary": self._summary_of(r)} for r in rows if self._ok(r["id"])]}

    def get_document_summary(self, document_id: int) -> dict[str, Any]:
        """Summary of one document (orientation only - never evidence)."""
        r =self.db.one("SELECT id, filename, summary, summary_status, summary_error FROM documents WHERE id=?", (document_id,))
        if not r or not self._ok(r["id"]):
            return {"error": f"document {document_id} not found"}
        return {"document_id": r["id"], "filename": r["filename"], "summary_status": r["summary_status"],
                "summary": self._summary_of(r), "note": "summaries are an orientation aid, not evidence; cite passages",
                **({"summary_error": r["summary_error"]} if r["summary_error"] else {})}

    def search_chunks(self, query: str, limit: int = 5, document_id: int | None = None) -> dict[str, Any]:
        """Keyword search over passages, ranked; optionally restricted to one document."""
        tokens = list(dict.fromkeys(t for t in re.findall(r"\w+", query.lower()) if len(t) > 1))
        if not tokens:
            return {"results": [], "note": "empty query"}
        limit = max(1, min(limit, 20))
        rows = [r for r in self.db.search_chunks(tokens, limit * 3, document_id) if self._ok(r["document_id"])][:limit]
        return {"results": [self._view(r).model_dump() for r in rows]}

    def get_chunk(self, chunk_id: int) -> dict[str, Any]:
        """Fetch one passage by id, with its filename and page."""
        r =self.db.one("SELECT c.*, d.filename FROM chunks c JOIN documents d ON d.id=c.document_id WHERE c.id=?", (chunk_id,))
        if not r or not self._ok(r["document_id"]):
            return {"error": f"chunk {chunk_id} not found"}
        return {"chunk": self._view(r).model_dump()}

    def get_neighbors(self, chunk_id: int, window: int = 1) -> dict[str, Any]:
        """Fetch the passages just before and after a passage, for surrounding context."""
        r =self.db.one("SELECT document_id, ordinal FROM chunks WHERE id=?", (chunk_id,))
        if not r or not self._ok(r["document_id"]):
            return {"error": f"chunk {chunk_id} not found"}
        w = max(0, min(window, 3))
        rows = self.db.query(
            "SELECT c.*, d.filename FROM chunks c JOIN documents d ON d.id=c.document_id "
            "WHERE c.document_id=? AND c.ordinal BETWEEN ? AND ? ORDER BY c.ordinal",
            (r["document_id"], r["ordinal"] - w, r["ordinal"] + w))
        return {"chunks": [self._view(x).model_dump() for x in rows]}

    def verify_quote(self, chunk_id: int, quote: str) -> dict[str, Any]:
        """Check that a quotation really exists in the cited passage (guards against invented evidence)."""
        r =self.db.one("SELECT * FROM chunks WHERE id=?", (chunk_id,))
        if not r or not self._ok(r["document_id"]):
            return VerifyResult(found=False, reason=f"chunk {chunk_id} does not exist").model_dump()
        if quote_in_text(quote, r["text"]):
            return VerifyResult(found=True, chunk_id=chunk_id, reason="quote found in cited chunk").model_dump()
        for o in self.db.query("SELECT id, text FROM chunks WHERE document_id=? AND id<>?", (r["document_id"], chunk_id)):
            if quote_in_text(quote, o["text"]):
                return VerifyResult(found=True, chunk_id=o["id"], relocated=True,
                                    reason="quote not in cited chunk but found elsewhere in same document").model_dump()
        if len(normalize(quote)) < 8:
            return VerifyResult(found=False, reason="quote too short to verify").model_dump()
        return VerifyResult(found=False, reason="quote not found in document").model_dump()

    def chunk_model(self, chunk_id: int) -> Chunk | None:
        r = self.db.one("SELECT * FROM chunks WHERE id=?", (chunk_id,))
        return Chunk(**{k: r[k] for k in r.keys()}) if r else None


TOOL_NAMES = ["list_documents", "get_document_summary", "search_chunks", "get_chunk", "get_neighbors", "verify_quote"]
