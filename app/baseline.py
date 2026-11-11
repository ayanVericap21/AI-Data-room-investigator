"""Approach A: direct single-pass LLM analysis of the whole data room. No retrieval, tools or verification
gating - the only extra is that passages carry ids so its citations can be *checked* afterwards for scoring.
"""
from __future__ import annotations

import time

from pydantic import BaseModel

from .agent.prompts import BASELINE_SYSTEM
from .agent.schemas import BaselineOut
from .agent.verify import verify_citations
from .config import Settings
from .db import Database
from .llm import LLM, Block, LLMError, Usage
from .mcp_server.client import LocalToolClient
from .models import Finding
from .tools import DataRoomTools


class BaselineResult(BaseModel):
    findings: list[Finding]
    usage: Usage
    truncated: bool = False
    error: str | None = None
    wall_s: float = 0.0


def run_baseline(db: Database, settings: Settings, llm: LLM, document_ids: list[int]) -> BaselineResult:
    t0 = time.time()
    parts, size, truncated = [], 0, False
    for did in document_ids:
        name = db.one("SELECT filename FROM documents WHERE id=?", (did,))["filename"]
        for r in db.query("SELECT id, page, text FROM chunks WHERE document_id=? ORDER BY ordinal", (did,)):
            block = f"[passage {r['id']} | {name} | p.{r['page']}]\n{r['text']}"
            if size + len(block) > settings.baseline_max_chars:
                truncated = True
                break
            parts.append(block)
            size += len(block)
    try:
        # The whole data room is the (large, identical-across-runs) prefix: cached, so repeat runs within the cache
        # TTL read it at the discounted rate. Same mechanism the agent uses - the comparison stays fair.
        out = llm.structured(system=BASELINE_SYSTEM, purpose="baseline", schema=BaselineOut,
                             prefix=[Block(text="Data room passages:\n\n" + "\n\n".join(parts), cache=True)],
                             user="Report the discrepancies.")
    except LLMError as e:
        return BaselineResult(findings=[], usage=llm.usage, truncated=truncated, error=str(e), wall_s=round(time.time() - t0, 2))
    tools = LocalToolClient(DataRoomTools(db, document_ids))
    findings = []
    for c in out.findings:
        cites = verify_citations(tools, c.citations)  # checked for scoring only; the baseline's claims are NOT gated
        findings.append(Finding(classification=c.classification, title=c.title, explanation=c.explanation, citations=cites,
                                uncertainties=c.uncertainties, assumptions=c.assumptions, confidence=c.confidence,
                                status="established"))
    return BaselineResult(findings=findings, usage=llm.usage, truncated=truncated, wall_s=round(time.time() - t0, 2))
