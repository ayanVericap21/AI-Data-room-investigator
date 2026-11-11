"""Paragraph-aware chunking that preserves source location.

Invariant (tested): for every chunk, page_text[char_start:char_end] == chunk.text.
"""
from __future__ import annotations

import re

from pydantic import BaseModel

from .parsers import PageText

_PARA = re.compile(r"\S[\s\S]*?(?=\n\s*\n|\Z)")


class RawChunk(BaseModel):
    page: int
    char_start: int
    char_end: int
    text: str


def _split_long(text: str, start: int, max_chars: int) -> list[tuple[int, int]]:
    spans, pos, n = [], 0, len(text)
    while pos < n:
        end = min(pos + max_chars, n)
        if end < n:
            ws = max(text.rfind(" ", pos, end), text.rfind("\n", pos, end))
            if ws > pos + max_chars // 2:
                end = ws
        spans.append((start + pos, start + end))
        pos = end
        while pos < n and text[pos].isspace():
            pos += 1
    return spans


def chunk_pages(pages: list[PageText], max_chars: int = 1000) -> list[RawChunk]:
    if max_chars < 50:
        raise ValueError("max_chars too small")
    out: list[RawChunk] = []
    for p in pages:
        spans: list[tuple[int, int]] = []
        for m in _PARA.finditer(p.text):
            s, e = m.start(), m.end()
            if e - s > max_chars:
                spans.extend(_split_long(p.text[s:e], s, max_chars))
            else:
                spans.append((s, e))
        # greedily merge adjacent small paragraphs
        cur: tuple[int, int] | None = None
        for s, e in spans:
            if cur and e - cur[0] <= max_chars:
                cur = (cur[0], e)
            else:
                if cur:
                    out.append(RawChunk(page=p.page, char_start=cur[0], char_end=cur[1], text=p.text[cur[0] : cur[1]]))
                cur = (s, e)
        if cur:
            out.append(RawChunk(page=p.page, char_start=cur[0], char_end=cur[1], text=p.text[cur[0] : cur[1]]))
    return out
