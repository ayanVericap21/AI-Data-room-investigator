"""The case model: what the room is about, the role of each document, the kinds of identifier it uses, who is who, and how metrics are defined.

Code finds the candidate identifier shapes in the text (a shape is a token with its digit runs replaced by #, CF-IR06-03 -> CF-IR#-#);
the model names them, writes patterns for any it sees that code missed, groups aliases, and states roles and definitions. Code then
validates every pattern against the text, normalises identifiers (CF-IR06-3 and CF-IR06-03 are one), and scans every chunk for them, which
links documents deterministically. The result is saved as an editable JSON file.
"""
from __future__ import annotations

import json
import logging
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel, Field

from app.investigator.models import AliasGroup, AliasProposal, CaseModel, DocRole, IdPattern
from app.investigator.tables import TableStore

logger = logging.getLogger(__name__)

_TOKEN = re.compile(r"(?<![A-Za-z0-9])[A-Za-z][A-Za-z0-9]{0,7}(?:[-_/][A-Za-z0-9]{1,8}){1,3}(?![A-Za-z0-9])")
_NOT_AN_ID = {"table", "tab", "fig", "figure", "page", "pg", "section", "sec", "annex", "appendix", "version", "ver", "rev", "no", "ref", "item", "step", "ch",
              "chapter", "clause", "art", "para", "line", "row", "col", "iso", "pp", "vol", "eq", "ex", "and", "or", "of", "to", "in", "on", "at", "by"}
_TITLES = re.compile(r"\b(dr|mr|mrs|ms|miss|prof|professor|sir|capt|captain|eng)\b\.?", re.I)
_NAMEISH_HEADER = re.compile(r"name|author|prepared|collector|collected by|officer|scientist|technician|lead|diver|operator|signed|reviewer|approved", re.I)
MAX_SHAPES = 18
MAX_NAMES = 150
MIN_SHAPE_MEMBERS = 3
MAX_PATTERN_LENGTH = 240
_NESTED_QUANTIFIER = re.compile(r"\((?:[^()\\]|\\.)*[+*](?:[^()\\]|\\.)*\)[+*{]")


# ---- identifier shapes -------------------------------------------------------------------------------------------------------------------
def shape_of(token: str) -> str:
    """The family of an identifier: digit runs become #, a trailing one or two letter suffix after a number is dropped."""
    shape = re.sub(r"\d+", "#", token)
    return re.sub(r"(?<=#)[-_/][A-Za-z]{1,2}$", "", shape)


def shape_regex(shape: str) -> str:
    parts = shape.split("#")
    body = r"\d{1,8}".join(re.escape(p) for p in parts)
    return r"(?<![A-Za-z0-9])" + body + r"(?:[-_/][A-Za-z]{1,2})?(?![A-Za-z0-9])"


def candidate_shapes(store: TableStore) -> list[dict]:
    """Shapes seen at least MIN_SHAPE_MEMBERS different times, most common first, with examples and the documents they occur in."""
    members: dict[str, set[str]] = defaultdict(set)
    docs: dict[str, set[str]] = defaultdict(set)
    with store._lock:
        rows = store.conn.execute("SELECT filename, content FROM _chunks").fetchall()
    for filename, content in rows:
        for m in _TOKEN.finditer(content or ""):
            token = m.group(0)
            first = re.split(r"[-_/]", token)[0]
            if first.lower() in _NOT_AN_ID or not re.search(r"\d", token):
                continue
            shape = shape_of(token)
            if "#" not in shape:
                continue
            members[shape].add(token.upper())
            docs[shape].add(filename)
    out = [{"shape": s, "count": len(m), "examples": sorted(m)[:4], "documents": sorted(docs[s])} for s, m in members.items() if len(m) >= MIN_SHAPE_MEMBERS]
    out.sort(key=lambda d: (-len(d["documents"]), -d["count"]))
    return out[:MAX_SHAPES]


def regex_is_safe(pattern: str) -> re.Pattern | None:
    if not pattern or len(pattern) > MAX_PATTERN_LENGTH or _NESTED_QUANTIFIER.search(pattern):
        return None
    try:
        return re.compile(pattern, re.I)
    except re.error:
        return None


def normalise_ident(text: str, p: IdPattern) -> str:
    s = text.strip()
    if p.pad_last_number_to:
        s = re.sub(r"(\d+)(?!.*\d)", lambda m: m.group(1).zfill(p.pad_last_number_to), s)
    if p.strip_separators:
        s = re.sub(r"[\W_]+", "", s)
    return s.upper() if p.case_fold else s


# ---- people and aliases --------------------------------------------------------------------------------------------------------------------
def clean_name(text: str) -> str:
    return " ".join(_TITLES.sub(" ", re.sub(r"[().,;:]", " ", text)).split())


def person_key(name: str) -> str | None:
    """surname|initial for a name of two or more words (Dr Helena Marsh, H. Marsh, Marsh, H.); None for a single word."""
    words = clean_name(name).split()
    if len(words) < 2:
        return None
    return f"{words[-1].lower()}|{words[0][0].lower()}"


def deterministic_aliases(names: list[str]) -> list[AliasGroup]:
    groups: dict[str, list[str]] = defaultdict(list)
    for n in names:
        key = person_key(n)
        if key and n not in groups[key]:
            groups[key].append(n)
    out = []
    for members in groups.values():
        if len(members) >= 2:
            canonical = max(members, key=lambda m: len(clean_name(m)))
            out.append(AliasGroup(canonical=clean_name(canonical), entity_type="person", aliases=sorted(set(clean_name(m) for m in members))))
    return out


def candidate_names(store: TableStore) -> list[str]:
    """Distinct person-like strings: values of columns whose header names a person or role, and the subjects and values of name-typed facts."""
    names: Counter = Counter()
    for t in store.table_list():
        for col in t.columns:
            if _NAMEISH_HEADER.search(col.header):
                for v in store.column_values(t.name, col.name)[:2000]:
                    if isinstance(v, str) and 2 < len(v) < 60 and not re.search(r"\d{3,}", v):
                        names[v.strip()] += 1
    with store._lock:
        for subject, value in store.conn.execute("SELECT subject, value FROM _facts WHERE attribute LIKE '%name%' OR attribute LIKE '%author%' OR attribute LIKE '%prepared%' "
                                                 "OR attribute LIKE '%role%' OR attribute LIKE '%lead%'"):
            for v in (subject, value):
                if isinstance(v, str) and 2 < len(v) < 60 and not re.search(r"\d{3,}", v):
                    names[v.strip()] += 1
    return [n for n, _ in names.most_common(MAX_NAMES)]


# ---- builder ----------------------------------------------------------------------------------------------------------------------------------
class ShapeLabel(BaseModel):
    shape: str = ""
    entity_type: str = ""  # what kind of thing it identifies (sample, site, order ...), or "" when it is not an identifier of a thing
    is_entity: bool = True


class CaseModelDraft(CaseModel):
    shape_labels: list[ShapeLabel] = Field(default_factory=list)


SYSTEM = (
    "You model a data room before it is investigated. You are given a digest of every document (its kind, headings, the start of its text, the tables it holds), "
    "identifier shapes found in the text (a shape is a token with its digit runs replaced by #), and names that may be one person written several ways. "
    "Return: case_summary (two sentences on what the room is about); roles (one per document: authority = sets rules, limits or permissions; primary_record = raw observations, logs, "
    "registers; derived_analysis = processes primary data, such as a lab or statistical report; narrative = interprets for an audience, such as a summary report; "
    "event_register = logs exceptions, deviations or incidents; other), with a one-sentence summary; "
    "shape_labels (for each shape shown, the entity_type it identifies in plain words, or is_entity false when it is not an identifier of a thing, such as a document number); "
    "id_patterns only for identifiers you see in the digest that no shape covers (a regex, plus case_fold, strip_separators, pad_last_number_to); "
    "aliases (groups of strings that name the same person, organisation or thing: full name, initials, codes); definitions (how a metric is defined or computed, as a document states it); "
    "units (unit equivalences a document states); time_conventions (time zone or date format per document, where stated). "
    "Use only what the digest shows. Leave a list empty when there is nothing to put in it.")


@dataclass
class CaseResult:
    model: CaseModel
    patterns: list[tuple[IdPattern, re.Pattern]] = field(default_factory=list)
    shapes: list[dict] = field(default_factory=list)
    mentions: int = 0
    notes: list[str] = field(default_factory=list)

    def role_of(self, filename: str) -> str:
        return next((r.role for r in self.model.roles if r.document == filename), "other")

    def entity_types(self) -> list[str]:
        return sorted({p.entity_type for p, _ in self.patterns})


def digest(store: TableStore, max_per_doc: int = 1400) -> str:
    parts = []
    with store._lock:
        docs = store.conn.execute("SELECT DISTINCT filename FROM _chunks ORDER BY filename").fetchall()
    for (filename,) in docs:
        chunks = store.chunks_of(filename)
        text = " ".join(c[2] for c in chunks[:6])[:600]
        tables = [f"{t.name} {t.title}: " + ", ".join(c.header for c in t.columns[:10]) + f" ({t.rows} rows)" for t in store.table_list() if t.filename == filename][:6]
        block = f"## {filename} ({len(chunks)} chunks)\nStart: {' '.join(text.split())}\n" + ("Tables:\n  " + "\n  ".join(tables) if tables else "")
        parts.append(block[:max_per_doc])
    return "\n\n".join(parts)


class CaseModelBuilder:
    def __init__(self, store: TableStore, llm: Any, on_progress: Callable[[str], None] | None = None):
        self.store = store
        self.llm = llm
        self.on_progress = on_progress or (lambda _m: None)

    def build(self) -> CaseResult:
        shapes = candidate_shapes(self.store)
        names = candidate_names(self.store)
        prompt = ("Digest of the data room:\n\n" + digest(self.store)
                  + "\n\nIdentifier shapes found in the text (shape, how many distinct identifiers, examples, documents):\n"
                  + ("\n".join(f"- {s['shape']}  ({s['count']}; e.g. {', '.join(s['examples'])}; in {', '.join(s['documents'])})" for s in shapes) or "(none)")
                  + "\n\nNames that may be one person written several ways:\n" + ("\n".join(f"- {n}" for n in names) or "(none)"))
        notes: list[str] = []
        try:
            draft = self.llm.generate_structured_output(prompt, CaseModelDraft, system=SYSTEM)
        except Exception as exc:  # the model could not be reached: the deterministic parts still work
            notes.append(f"case model call failed ({str(exc)[:120]}); identifiers and aliases come from code only")
            draft = CaseModelDraft()
        model = CaseModel(**{k: getattr(draft, k) for k in CaseModel.model_fields})
        labels = {l.shape: l for l in draft.shape_labels}
        patterns: list[tuple[IdPattern, re.Pattern]] = []
        for s in shapes:
            label = labels.get(s["shape"])
            if label is not None and not label.is_entity:
                continue
            entity_type = (label.entity_type if label and label.entity_type else f"id {s['shape']}").strip()
            p = IdPattern(entity_type=entity_type, regex=shape_regex(s["shape"]), description=f"shape {s['shape']}", pad_last_number_to=_pad_width(s))
            patterns.append((p, re.compile(p.regex, re.I)))
        covered = {p.regex for p, _ in patterns}
        for p in model.id_patterns:
            compiled = regex_is_safe(p.regex)
            if compiled is None:
                notes.append(f"pattern for {p.entity_type!r} dropped: not a usable regex ({p.regex[:60]!r})")
            elif p.regex not in covered:
                patterns.append((p, compiled))
        model.id_patterns = [p for p, _ in patterns]
        merged = deterministic_aliases(names)
        have = {a.canonical.lower() for a in model.aliases}
        model.aliases += [a for a in merged if a.canonical.lower() not in have]
        return CaseResult(model=model, patterns=patterns, shapes=shapes, notes=notes)

    # -- scanning ---------------------------------------------------------------------------------------------------------------------
    def scan(self, result: CaseResult) -> int:
        """Find every mention of every identifier pattern and every alias in every chunk; link them in the store."""
        alias_lookup: dict[str, AliasGroup] = {}
        for g in result.model.aliases:
            for a in {g.canonical, *g.aliases}:
                if len(a) >= 4:
                    alias_lookup[clean_name(a).lower()] = g
        alias_re = re.compile(r"(?<![A-Za-z0-9])(" + "|".join(r"[\s.]+".join(re.escape(w) for w in a.split()) for a in sorted(alias_lookup, key=len, reverse=True))
                              + r")(?![A-Za-z0-9])", re.I) if alias_lookup else None
        rows: list[tuple] = []
        seen: set[tuple] = set()
        with self.store._lock:
            chunks = self.store.conn.execute("SELECT chunk_id, document_id, filename, content FROM _chunks").fetchall()
        for chunk_id, doc_id, filename, content in chunks:
            for line in (content or "").split("\n"):
                if not line.strip():
                    continue
                for p, rx in result.patterns:
                    for m in rx.finditer(line):
                        ident = normalise_ident(m.group(0), p)
                        key = (p.entity_type, ident, chunk_id, line)
                        if key not in seen:
                            seen.add(key)
                            rows.append((p.entity_type, ident, m.group(0), doc_id, filename, chunk_id, line))
                if alias_re:
                    for m in alias_re.finditer(line):
                        g = alias_lookup.get(clean_name(m.group(1)).lower())
                        if g is None:
                            continue
                        key = (g.entity_type or "alias", g.canonical, chunk_id, line)
                        if key not in seen:
                            seen.add(key)
                            rows.append((g.entity_type or "alias", g.canonical, m.group(0), doc_id, filename, chunk_id, line))
        self.store.execute("DELETE FROM _mentions")
        self.store.executemany("INSERT INTO _mentions VALUES (?,?,?,?,?,?,?)", rows)
        self.store.execute("DELETE FROM _aliases")
        self.store.executemany("INSERT INTO _aliases VALUES (?,?,?)", [(g.canonical, g.entity_type, a) for g in result.model.aliases for a in g.aliases])
        self.store.commit()
        result.mentions = len(rows)
        return len(rows)

    def save(self, result: CaseResult, path: Path) -> None:
        data = result.model.model_dump()
        data["shapes_seen"] = result.shapes
        data["notes"] = result.notes
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _pad_width(shape: dict) -> int:
    """Zero-padding width of the last number when every example writes it zero-padded to one width (SPL-001, SPL-044), else 0."""
    runs = [re.findall(r"\d+", e)[-1] for e in shape["examples"] if re.findall(r"\d+", e)]
    widths = {len(r) for r in runs}
    return widths.pop() if len(widths) == 1 and any(r.startswith("0") for r in runs) else 0


# ---- lookups used by the tools ---------------------------------------------------------------------------------------------------------------
def find_entity(store: TableStore, text: str, limit: int = 20) -> list[dict]:
    """Every mention of an identifier or alias, across the room. ``text`` may be an identifier (any spelling), an alias or a canonical name."""
    needle = re.sub(r"[\W_]+", "", text or "").upper()
    if not needle:
        return []
    with store._lock:
        canon = store.conn.execute("SELECT canonical FROM _aliases WHERE upper(replace(replace(alias,' ',''),'.','')) = ? LIMIT 1", (needle,)).fetchone()
        wanted = canon[0] if canon else None
        rows = store.conn.execute(
            "SELECT entity_type, ident, written, filename, chunk_id, line FROM _mentions WHERE ident = ? OR upper(replace(replace(ident,' ',''),'.','')) = ? OR ident = ? LIMIT ?",
            (needle, needle, wanted or "", limit)).fetchall()
    return [{"entity_type": r[0], "ident": r[1], "written": r[2], "filename": r[3], "chunk_id": r[4], "line": r[5]} for r in rows]


def ids_by_document(store: TableStore, entity_type: str) -> dict[str, dict[str, tuple[str, str]]]:
    """document filename -> identifier -> (chunk id, line of its first mention), for one kind of identifier."""
    out: dict[str, dict[str, tuple[str, str]]] = defaultdict(dict)
    with store._lock:
        rows = store.conn.execute("SELECT filename, ident, chunk_id, line FROM _mentions WHERE entity_type = ?", (entity_type,)).fetchall()
    for filename, ident, chunk_id, line in rows:
        out[filename].setdefault(ident, (chunk_id, line))
    return out
