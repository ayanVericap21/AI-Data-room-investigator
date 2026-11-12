"""Scoring a run against an answer key.

An answer key lists the planted issues of a data room (what a good run must find) and the traps (differences that must not be flagged as
problems). Each entry says how to recognise a finding about it: words that must all appear (``all``) and words of which at least one must
appear (``any``) in the finding's title, description or evidence quotes, compared in lower case. An issue may also name the classifications or
taxonomy types that count. An issue without a ``match`` is listed for manual checking and not counted.

The key is plain JSON so it can be edited by a person. Matching by words is blunt on purpose: it is reproducible and free; read the
report's matched titles to see whether the match is a real one.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class Match(BaseModel):
    all: list[str] = Field(default_factory=list)
    any: list[str] = Field(default_factory=list)


class KeyIssue(BaseModel):
    id: str
    title: str = ""
    match: Match | None = None  # None: checked by hand
    classes: list[str] = Field(default_factory=list)  # classification or taxonomy values that count; empty = any
    families: list[str] = Field(default_factory=list)  # the check families expected to find it (informational)


class KeyTrap(BaseModel):
    id: str
    title: str = ""
    match: Match
    flag_classes: list[str] = Field(default_factory=lambda: ["contradiction"])  # a finding of these classes about the trap is a false positive


class AnswerKey(BaseModel):
    room: str = ""
    provisional: bool = False
    note: str = ""
    issues: list[KeyIssue] = Field(default_factory=list)
    traps: list[KeyTrap] = Field(default_factory=list)


def load_key(path: Path) -> AnswerKey:
    return AnswerKey.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))


def load_findings(path: Path) -> list[dict]:
    """Findings from a ``.findings.json`` file, or from the FINDINGS block of a results text file (one file holds everything)."""
    raw = Path(path).read_text(encoding="utf-8")
    if Path(path).suffix == ".txt":
        a, b = raw.find("<<<FINDINGS\n"), raw.find("\nFINDINGS>>>")
        if a < 0 or b < 0:
            raise ValueError("no FINDINGS block in this results file")
        raw = raw[a + len("<<<FINDINGS\n"):b]
    return json.loads(raw)


def finding_text(f: dict) -> str:
    return " ".join([f.get("title", ""), f.get("description", ""), *(e.get("quote", "") for e in f.get("evidence", []))]).lower()


def matches(text: str, m: Match) -> bool:
    return all(t.lower() in text for t in m.all) and (not m.any or any(t.lower() in text for t in m.any))


def _class_ok(f: dict, classes: list[str]) -> bool:
    return not classes or f.get("classification") in classes or f.get("taxonomy") in classes


@dataclass
class IssueResult:
    issue: KeyIssue
    found: list[dict] = field(default_factory=list)
    counted: bool = True

    @property
    def hit(self) -> bool:
        return bool(self.found)


@dataclass
class Score:
    key: AnswerKey
    issues: list[IssueResult]
    trap_hits: list[tuple[KeyTrap, dict]]
    unmatched: list[dict]
    findings: int

    @property
    def counted(self) -> list[IssueResult]:
        return [r for r in self.issues if r.counted]

    @property
    def recall(self) -> float:
        c = self.counted
        return sum(r.hit for r in c) / len(c) if c else 0.0

    @property
    def precision(self) -> float:
        """Share of findings that match a key issue; findings that match nothing may be real extra finds or false alarms (see unmatched)."""
        return (self.findings - len(self.unmatched) - len(self.trap_hits)) / self.findings if self.findings else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0


def score(key: AnswerKey, findings: list[dict]) -> Score:
    texts = [(f, finding_text(f)) for f in findings]
    results: list[IssueResult] = []
    matched_ids: set[str] = set()
    for issue in key.issues:
        if issue.match is None:
            results.append(IssueResult(issue, [], counted=False))
            continue
        found = [f for f, t in texts if matches(t, issue.match) and _class_ok(f, issue.classes)]
        matched_ids.update(f["id"] for f in found)
        results.append(IssueResult(issue, found))
    trap_hits: list[tuple[KeyTrap, dict]] = []
    trap_ids: set[str] = set()
    for trap in key.traps:
        for f, t in texts:
            if matches(t, trap.match) and f.get("classification") in trap.flag_classes and f["id"] not in matched_ids:
                trap_hits.append((trap, f))
                trap_ids.add(f["id"])
    unmatched = [f for f in findings if f["id"] not in matched_ids and f["id"] not in trap_ids]
    return Score(key, results, trap_hits, unmatched, len(findings))


def render(score_: Score, findings_path: str = "") -> str:
    s = score_
    L = [f"SCORE  room: {s.key.room}" + ("   (PROVISIONAL KEY: derived from a reviewer's description, check the tokens)" if s.key.provisional else ""), "=" * 78,
         f"findings file : {findings_path}",
         f"issues found  : {sum(r.hit for r in s.counted)} of {len(s.counted)} counted ({s.recall:.0%} recall)" + (f"; {len(s.issues) - len(s.counted)} more need a manual check" if len(s.issues) > len(s.counted) else ""),
         f"findings      : {s.findings}; {s.findings - len(s.unmatched) - len(s.trap_hits)} match a key issue ({s.precision:.0%}), {len(s.unmatched)} match nothing in the key, {len(s.trap_hits)} flag a trap",
         f"trap false positives: {len(s.trap_hits)} of {len(s.key.traps)} traps hit" if s.key.traps else "trap false positives: no traps in this key",
         f"F1 (recall and key-match share): {s.f1:.2f}", "", "Issues:"]
    for r in s.issues:
        if not r.counted:
            L.append(f"  [manual] {r.issue.id}  {r.issue.title}")
            continue
        mark = "FOUND " if r.hit else "MISSED"
        L.append(f"  [{mark}] {r.issue.id}  {r.issue.title}")
        for f in r.found[:3]:
            L.append(f"           <- {f.get('classification')} / {f.get('taxonomy')} family {f.get('family')}: {f['title'][:110]}")
    if s.trap_hits:
        L += ["", "Traps flagged (false positives):"] + [f"  {t.id} {t.title}: {f['classification']}: {f['title'][:110]}" for t, f in s.trap_hits]
    if s.unmatched:
        L += ["", "Findings that match nothing in the key (extra finds or false alarms; read them):"]
        L += [f"  {f['classification']} / {f.get('taxonomy')} family {f.get('family')} {f['severity']}: {f['title'][:120]}" for f in s.unmatched]
    return "\n".join(L) + "\n"


def by_family(score_: Score) -> dict[str, Any]:
    """For each check family, how many issues it was expected to find and how many it matched."""
    out: dict[str, list[int]] = {}
    for r in score_.counted:
        fams = {f.get("family") for f in r.found}
        for fam in r.issue.families:
            row = out.setdefault(fam, [0, 0])
            row[0] += 1
            row[1] += fam in fams
    return {k: {"expected": v[0], "found_by_that_family": v[1]} for k, v in sorted(out.items())}
