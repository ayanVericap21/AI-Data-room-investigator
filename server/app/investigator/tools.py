"""The tools a worker can ask for, and the resolution of its evidence references into verified chunk text.

Tools (each returns text, capped): sql (read-only, on the loaded tables), search (passages by words), read (a document's chunks),
find_entity (every mention of an identifier or alias), timeline (dated events, optionally for one entity and window), get_facts.
A worker never receives more than a few thousand characters per call; it narrows its query instead.
"""
from __future__ import annotations

import re
from typing import Any

from app.investigator.case_model import find_entity
from app.investigator.models import EvidenceRef, Fact, ResolvedEvidence, ToolCall
from app.investigator.tables import TableStore, ToolSqlError, norm_id, parse_when
from app.tools.submit import find_source_span

MAX_RESULT_CHARS = 2800
MAX_EVENTS = 6000
APPROX_SHARE = 0.75  # share of a reworded quote's words a passage must contain
APPROX_SHARE = 0.75  # share of a reworded quote's words that a passage must contain
EVIDENCE_ROLES = ("supporting", "contradicting", "contextual", "insufficient")


def _cap(text: str, n: int = MAX_RESULT_CHARS) -> str:
    return text if len(text) <= n else text[:n] + "\n... (cut)"


def build_timeline(store: TableStore, facts: list[Fact]) -> int:
    """Every dated event of the room in one table: event facts, and every row of a table that has a date column."""
    with store._lock:
        idents = {r[0] for r in store.conn.execute("SELECT DISTINCT ident FROM _mentions")}
    rows: list[tuple] = []
    for f in facts:
        iso = parse_when(f.when) if f.when else None
        if f.type == "event" and iso:
            rows.append((iso, f"{f.subject}: {f.attribute} {f.value}".strip(), f.subject, "fact", f.document_id, _filename(store, f.document_id), f.chunk_id, f.quote, f.when or ""))
    for t in store.table_list():
        date_cols = store.date_columns(t.name)
        if not date_cols:
            continue
        cols = [c.name for c in t.columns]
        sel = ", ".join(f'"{c}"' for c in cols)
        res = store.query(f'SELECT _chunk, _line, {sel} FROM "{t.name}" ORDER BY _rid', limit=100000)
        for row in res.rows:
            chunk_id, line, values = row[0], row[1], dict(zip(cols, row[2:]))
            label_col = next((c for c in cols if c not in date_cols and isinstance(values[c], str) and parse_when(values[c]) is None), None)
            label = str(values[label_col]) if label_col else ""
            entity = next((str(v) for v in values.values() if v is not None and norm_id(v) in idents), label)
            for dc in date_cols:
                iso = parse_when(values[dc])
                if iso:
                    header = next(c.header for c in t.columns if c.name == dc)
                    rows.append((iso, f"{label} {header}".strip(), entity, "table", t.document_id, t.filename, chunk_id, line, f"{t.title}: {header} = {values[dc]}"))
            if len(rows) > MAX_EVENTS:
                break
    store.execute("DELETE FROM _events")
    store.executemany("INSERT INTO _events VALUES (?,?,?,?,?,?,?,?,?)", rows[:MAX_EVENTS])
    store.commit()
    return min(len(rows), MAX_EVENTS)


def _filename(store: TableStore, document_id: str) -> str:
    with store._lock:
        row = store.conn.execute("SELECT filename FROM _chunks WHERE document_id = ? LIMIT 1", (document_id,)).fetchone()
    return row[0] if row else ""


class ToolBox:
    def __init__(self, store: TableStore, facts: list[Fact]):
        self.store = store
        self.facts = facts

    # -- the tools -------------------------------------------------------------------------------------------------------------------
    def run(self, call: ToolCall) -> str:
        try:
            handler = {"sql": self._sql, "search": self._search, "read": self._read, "find_entity": self._entity, "timeline": self._timeline, "get_facts": self._facts}.get(call.tool)
            if handler is None:
                return f"unknown tool {call.tool!r}; use sql, search, read, find_entity, timeline or get_facts"
            return _cap(handler(call))
        except ToolSqlError as exc:
            return f"SQL error: {exc}"
        except Exception as exc:  # a tool never takes a worker down; the worker sees the problem and corrects the call
            return f"tool error: {type(exc).__name__}: {str(exc)[:200]}"

    def _sql(self, c: ToolCall) -> str:
        res = self.store.query(c.sql or "")
        return f"{len(res.rows)} row(s)\n" + res.text(MAX_RESULT_CHARS - 40)

    def _search(self, c: ToolCall) -> str:
        hits = self.store.search(c.query or "", limit=5, document=c.document or None)
        if not hits:
            return "(no passages match)"
        return "\n".join(f"- [{h['chunk_id']}] {h['filename']}, {h['location']}: {h['snippet']}" for h in hits)

    def _read(self, c: ToolCall) -> str:
        loc = (c.location or "").strip()
        if loc.startswith("chk_"):
            row = self.store.chunk(loc)
            return f"[{loc}] {row[1]}, {row[2]}\n{row[3]}" if row else "(no such chunk)"
        chunks = self.store.chunks_of(c.document or "") if c.document else []
        if not chunks:
            return "(no such document: give its file name)"
        picked = [ch for ch in chunks if loc and loc.lower() in ch[1].lower()][:3] or chunks[:2]
        return "\n\n".join(f"[{cid}] {where}\n{content}" for cid, where, content in picked)

    def _entity(self, c: ToolCall) -> str:
        hits = find_entity(self.store, c.entity or c.query or "", limit=15)
        if not hits:
            return "(no mentions)"
        return "\n".join(f"- {h['ident']} ({h['entity_type']}) in {h['filename']} [{h['chunk_id']}]: {h['line'][:200]}" for h in hits)

    def _timeline(self, c: ToolCall) -> str:
        sql = "SELECT when_iso, label, entity, source, filename, chunk_id, detail FROM _events WHERE 1=1"
        params: list[Any] = []
        if c.entity:
            sql += " AND (norm_id(entity) LIKE ? OR norm_id(label) LIKE ? OR norm_id(detail) LIKE ?)"
            needle = f"%{re.sub(r'[^A-Za-z0-9]+', '', c.entity).upper()}%"
            params += [needle, needle, needle]
        if c.window_start:
            sql += " AND when_iso >= ?"
            params.append(parse_when(c.window_start) or c.window_start)
        if c.window_end:
            sql += " AND when_iso <= ?"
            params.append(parse_when(c.window_end) or c.window_end)
        res = self.store.query(sql + " ORDER BY when_iso LIMIT 40", params=tuple(params))
        if not res.rows:
            return "(no events)"
        return "\n".join(f"{r[0]} | {r[1]} | {r[2]} | {r[3]} {r[4]} [{r[5]}] | {r[6]}"[:260] for r in res.rows)

    def _facts(self, c: ToolCall) -> str:
        words = [w.lower() for w in re.findall(r"[A-Za-z0-9][\w\-.]*", c.query or "")]
        picked = [f for f in self.facts if all(w in (f.line() + " " + f.quote).lower() for w in words)] if words else self.facts
        if not picked:
            return "(no facts match)"
        return "\n".join(f"{f.line()} | {self.store.chunk(f.chunk_id)[1] if self.store.chunk(f.chunk_id) else ''}" for f in picked[:25])

    # -- evidence --------------------------------------------------------------------------------------------------------------------
    def resolve(self, ref: EvidenceRef) -> ResolvedEvidence | None:
        """Turn a worker's evidence reference into an exact quote from a stored chunk, or None when it cannot be found."""
        role = ref.role if ref.role in EVIDENCE_ROLES else "supporting"
        if ref.kind == "row" and ref.table and ref.ref:
            src = self.store.row_source(ref.table, ref.ref)
            if not src or not src[1]:
                return None
            chunk = self.store.chunk(src[0])
            if chunk is None:
                return None
            return ResolvedEvidence(src[0], chunk[0], chunk[1], chunk[2], src[1], role, ref.shows)
        quote = (ref.quote or "").strip()
        if len(quote) < 4:
            return None
        pool = self.store.chunks_of(ref.document) if ref.document else []
        if not pool:  # the document name was wrong or missing: look everywhere
            with self.store._lock:
                pool = [(r[0], r[1], r[2]) for r in self.store.conn.execute("SELECT chunk_id, location, content FROM _chunks")]
        for chunk_id, _where, content in pool:
            span = find_source_span(content, quote)
            if span:
                text = span[0]
                if len(text) < 8:  # a short quote is widened to its line, so the stored evidence names what it is about
                    text = next((ln.strip() for ln in content.split("\n") if text in ln), text)
                info = self.store.chunk(chunk_id)
                return ResolvedEvidence(chunk_id, info[0], info[1], info[2], text, role, ref.shows) if info and len(text) >= 8 else None
        return self._approximate(quote, pool, ref, role)

    def _approximate(self, quote: str, pool: list, ref: EvidenceRef, role: str) -> ResolvedEvidence | None:
        """A quote the model reworded slightly: the line or sentence of the room that holds at least APPROX_SHARE of its words. The stored quote is that
        real text, never the model's wording, and the evidence says it was matched approximately so the verifier and the reader know."""
        wanted = set(re.findall(r"[a-z0-9]+", quote.lower()))
        if len(wanted) < 4:
            return None
        best: tuple[float, int, str, str] | None = None
        for chunk_id, _where, content in pool:
            for piece in re.split(r"(?<=[.!?])\s+|\n", content):
                text = piece.strip()
                if len(text) < 8:
                    continue
                share = len(wanted & set(re.findall(r"[a-z0-9]+", text.lower()))) / len(wanted)
                if share >= APPROX_SHARE and (best is None or (share, -len(text)) > (best[0], -best[1])):
                    best = (share, len(text), chunk_id, text)
        if best is None:
            return None
        info = self.store.chunk(best[2])
        shows = (ref.shows + " " if ref.shows else "") + "(the quote was matched approximately to this text)"
        return ResolvedEvidence(best[2], info[0], info[1], info[2], best[3], role, shows) if info else None
