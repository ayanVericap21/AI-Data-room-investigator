"""Persistence for investigations: state, claims, hypotheses, findings, event history."""
from __future__ import annotations

from pydantic import BaseModel

from ..db import Database, dumps, loads
from ..models import Claim, Finding, utcnow


class InvestigationNotFound(LookupError):
    pass


class Hypothesis(BaseModel):
    id: int
    investigation_id: int
    claim_ids: list[int]
    question: str
    rationale: str
    status: str  # pending | running | done | failed
    attempts: int
    clarifications: list[str]


class Event(BaseModel):
    id: int
    investigation_id: int
    ts: str
    kind: str
    message: str
    data: dict


class Investigation(BaseModel):
    id: int
    status: str  # queued | running | completed | partial | failed | interrupted
    phase: str   # extract | hypothesize | investigate | done
    document_ids: list[int]
    error: str | None
    usage: dict
    created_at: str
    updated_at: str


def create(db: Database, document_ids: list[int]) -> Investigation:
    now = utcnow()
    i = db.execute("INSERT INTO investigations(status, phase, document_ids, created_at, updated_at) VALUES(?,?,?,?,?)",
                   ("queued", "extract", dumps(document_ids), now, now))
    return get(db, i)


def get(db: Database, inv_id: int) -> Investigation:
    r = db.one("SELECT * FROM investigations WHERE id=?", (inv_id,))
    if not r:
        raise InvestigationNotFound(f"investigation {inv_id} not found")
    return Investigation(id=r["id"], status=r["status"], phase=r["phase"], document_ids=loads(r["document_ids"], []),
                         error=r["error"], usage=loads(r["usage"], {}), created_at=r["created_at"], updated_at=r["updated_at"])


def list_all(db: Database) -> list[Investigation]:
    return [get(db, r["id"]) for r in db.query("SELECT id FROM investigations ORDER BY id DESC")]


def update(db: Database, inv_id: int, **fields) -> None:
    if "usage" in fields:
        fields["usage"] = dumps(fields["usage"])
    fields["updated_at"] = utcnow()
    cols = ", ".join(f"{k}=?" for k in fields)
    db.execute(f"UPDATE investigations SET {cols} WHERE id=?", (*fields.values(), inv_id))


def log_event(db: Database, inv_id: int, kind: str, message: str, data: dict | None = None) -> None:
    db.execute("INSERT INTO events(investigation_id, ts, kind, message, data) VALUES(?,?,?,?,?)",
               (inv_id, utcnow(), kind, message, dumps(data or {})))


def events(db: Database, inv_id: int, after_id: int = 0) -> list[Event]:
    return [Event(id=r["id"], investigation_id=r["investigation_id"], ts=r["ts"], kind=r["kind"], message=r["message"],
                  data=loads(r["data"], {})) for r in
            db.query("SELECT * FROM events WHERE investigation_id=? AND id>? ORDER BY id", (inv_id, after_id))]


# ---- claims ----
def add_claim(db: Database, inv_id: int, document_id: int, chunk_id: int, subject: str, value: str,
              unit: str | None, conditions: str | None, quote: str) -> int:
    return db.execute("INSERT INTO claims(investigation_id, document_id, chunk_id, subject, value, unit, conditions, quote) "
                      "VALUES(?,?,?,?,?,?,?,?)", (inv_id, document_id, chunk_id, subject, value, unit, conditions, quote))


def claims(db: Database, inv_id: int) -> list[Claim]:
    return [Claim(**{k: r[k] for k in r.keys()}) for r in
            db.query("SELECT * FROM claims WHERE investigation_id=? ORDER BY id", (inv_id,))]


# ---- hypotheses ----
def _hyp(r) -> Hypothesis:
    return Hypothesis(id=r["id"], investigation_id=r["investigation_id"], claim_ids=loads(r["claim_ids"], []),
                      question=r["question"], rationale=r["rationale"], status=r["status"], attempts=r["attempts"],
                      clarifications=loads(r["clarifications"], []))


def add_hypothesis(db: Database, inv_id: int, claim_ids: list[int], question: str, rationale: str = "") -> int:
    return db.execute("INSERT INTO hypotheses(investigation_id, claim_ids, question, rationale, status) VALUES(?,?,?,?, 'pending')",
                      (inv_id, dumps(claim_ids), question, rationale))


def hypotheses(db: Database, inv_id: int) -> list[Hypothesis]:
    return [_hyp(r) for r in db.query("SELECT * FROM hypotheses WHERE investigation_id=? ORDER BY id", (inv_id,))]


def get_hypothesis(db: Database, hyp_id: int) -> Hypothesis | None:
    r = db.one("SELECT * FROM hypotheses WHERE id=?", (hyp_id,))
    return _hyp(r) if r else None


def next_pending(db: Database, inv_id: int) -> Hypothesis | None:
    r = db.one("SELECT * FROM hypotheses WHERE investigation_id=? AND status='pending' ORDER BY id LIMIT 1", (inv_id,))
    return _hyp(r) if r else None


def set_hyp_status(db: Database, hyp_id: int, status: str, bump_attempt: bool = False) -> None:
    db.execute("UPDATE hypotheses SET status=?, attempts=attempts+? WHERE id=?", (status, 1 if bump_attempt else 0, hyp_id))


def add_clarification(db: Database, hyp_id: int, note: str) -> None:
    h = get_hypothesis(db, hyp_id)
    db.execute("UPDATE hypotheses SET clarifications=? WHERE id=?", (dumps([*h.clarifications, note]), hyp_id))


# ---- findings ----
def save_finding(db: Database, inv_id: int, hyp_id: int | None, f: Finding) -> Finding:
    f.investigation_id, f.hypothesis_id = inv_id, hyp_id
    fid = db.execute("INSERT INTO findings(investigation_id, hypothesis_id, data) VALUES(?,?,?)",
                     (inv_id, hyp_id, f.model_dump_json()))
    f.id = fid
    return f


def findings(db: Database, inv_id: int, include_superseded: bool = False) -> list[Finding]:
    sql = "SELECT * FROM findings WHERE investigation_id=?" + ("" if include_superseded else " AND superseded=0") + " ORDER BY id"
    out = []
    for r in db.query(sql, (inv_id,)):
        f = Finding.model_validate_json(r["data"])
        f.id, f.superseded = r["id"], bool(r["superseded"])
        out.append(f)
    return out


def supersede_findings_for(db: Database, hyp_id: int) -> None:
    db.execute("UPDATE findings SET superseded=1 WHERE hypothesis_id=?", (hyp_id,))
