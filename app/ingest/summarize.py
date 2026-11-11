"""Document summarisation, run during ingest BEFORE the data room is served through MCP.

Summaries are an orientation aid (what is this document, what does it talk about) so the investigator can pick
what to search for. They are never evidence: a citation must be a verbatim quote of a source passage, and
`verify_quote` only checks passages - a quote copied from a summary fails verification.
Long documents are map-reduced: each batch of passages is summarised, then the partial summaries are merged.
"""
from __future__ import annotations

import logging

from ..config import Settings
from ..db import Database, dumps
from ..llm import LLM, LLMError
from ..models import DocSummary

log = logging.getLogger(__name__)

SUMMARY_SYSTEM = """You summarise one document from a due-diligence data room so that a reviewer can quickly see what it contains.
Be faithful: state only what the text says, keep numbers, units, dates and conditions exactly as written, and do not infer or add anything.
Note conditions and qualifications attached to figures (scope, period, assumptions) - they matter when documents are compared later.
Give a short overview, topic labels, and the key figures/commitments each with what it refers to."""

MERGE_SYSTEM = SUMMARY_SYSTEM + "\nYou are given partial summaries of consecutive parts of the same document; merge them into one summary of the whole document."


def _batches(rows: list, max_chars: int) -> list[list]:
    out, cur, size = [], [], 0
    for r in rows:
        if cur and size + len(r["text"]) > max_chars:
            out.append(cur)
            cur, size = [], 0
        cur.append(r)
        size += len(r["text"])
    if cur:
        out.append(cur)
    return out


def _fmt(s: DocSummary) -> str:
    return (f"Overview: {s.overview}\nTopics: {', '.join(s.key_topics)}\n"
            "Key figures:\n" + "\n".join(f"- {f}" for f in s.key_figures))


def summarize_document(db: Database, settings: Settings, llm: LLM, doc_id: int) -> str:
    """Summarise one processed document. Returns the summary_status. Never raises."""
    doc = db.one("SELECT filename FROM documents WHERE id=?", (doc_id,))
    rows = db.query("SELECT id, page, text FROM chunks WHERE document_id=? ORDER BY ordinal", (doc_id,))
    if not doc or not rows:
        _store(db, doc_id, "skipped", None, "no passages to summarise")
        return "skipped"
    try:
        partials = []
        for batch in _batches(rows, settings.summary_batch_chars):
            text = f"Document: {doc['filename']}\n\n" + "\n\n".join(f"[page {r['page']}]\n{r['text']}" for r in batch)
            partials.append(llm.structured(system=SUMMARY_SYSTEM, user=text, schema=DocSummary, purpose="summarize"))
        if len(partials) == 1:
            summary = partials[0]
        else:
            merged = "\n\n".join(f"Part {i + 1}:\n{_fmt(p)}" for i, p in enumerate(partials))
            summary = llm.structured(system=MERGE_SYSTEM, user=f"Document: {doc['filename']}\n\n{merged}",
                                     schema=DocSummary, purpose="summarize_merge")
        _store(db, doc_id, "done", summary, None)
        return "done"
    except LLMError as e:
        log.warning("summary failed for document %s: %s", doc_id, e)
        _store(db, doc_id, "failed", None, str(e)[:500])
        return "failed"
    except Exception as e:  # isolate: a summary problem must never fail the ingest
        log.exception("summary crashed for document %s", doc_id)
        _store(db, doc_id, "failed", None, f"{type(e).__name__}: {e}"[:500])
        return "failed"


def _store(db: Database, doc_id: int, status: str, summary: DocSummary | None, error: str | None) -> None:
    db.execute("UPDATE documents SET summary=?, summary_status=?, summary_error=? WHERE id=?",
               (summary.model_dump_json() if summary else None, status, error, doc_id))


def mark_skipped(db: Database, doc_ids: list[int], reason: str) -> None:
    for d in doc_ids:
        _store(db, d, "skipped", None, reason[:500])
