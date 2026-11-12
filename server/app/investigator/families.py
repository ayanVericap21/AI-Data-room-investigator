"""The check families. Each builds the tasks for one kind of check from the facts, the tables and the case model, so coverage follows the
structure of the room: every constraint, every identifier kind found in more than one document, every dated table pair, every stated
aggregate, every universal or negative assertion, every table, every reference, every conclusion.

Families B (set differences) and F (duplicate series, duplicate keys) and the resolution part of G are deterministic; they produce
candidates directly. The rest produce worker tasks. Family I (value comparison) is the existing pairing and judging, run by the pipeline.
"""
from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any

from app.investigator.case_model import CaseResult, find_entity, ids_by_document
from app.investigator.models import Candidate, Fact, ReferenceKinds, ResolvedEvidence, TaskRecord
from app.investigator.tables import TableStore, norm_id
from app.investigator.workers import WorkerTask
from app.services.missing_items import _item_keys, _names

_STOP = {"the", "and", "for", "that", "with", "from", "this", "are", "was", "were", "has", "have", "per", "all", "any", "each", "must", "shall", "not", "may", "its", "than",
         "into", "over", "under", "between", "within", "after", "before", "when", "where", "which", "their", "them", "they", "will", "been", "being", "also"}
_UNIVERSAL = re.compile(r"\b(all|no|none|nobody|nothing|every|everyone|never|always|fully|entire(?:ly)?|only|zero|each|without any|not a single|neither|nowhere|any)\b", re.I)
_AGGREGATE = re.compile(r"\b(mean|average|avg|total|sum|count|number of|percentage|percent|proportion|share|maximum|minimum|max|min|median|rate|index|ratio|overall)\b", re.I)
_DECLARED_ABSENT = re.compile(r"not (?:provided|included|attached|available|received|submitted|enclosed|supplied|yet)|pending|outstanding|missing|to be (?:provided|submitted|supplied)|awaited", re.I)
MAX_TASKS_PER_FAMILY = 60


def words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]{3,}", (text or "").lower()) if w not in _STOP}


def related_tables(store: TableStore, text: str, k: int = 4) -> set[str] | None:
    """The tables whose headers, title or file name share words with ``text`` (best k), or None when none do (then every table is shown)."""
    want = words(text)
    scored = []
    for t in store.table_list():
        have = words(" ".join(c.header for c in t.columns) + " " + t.title + " " + t.filename)
        score = len(want & have)
        if score:
            scored.append((score, t.name))
    scored.sort(reverse=True)
    return {n for _, n in scored[:k]} or None


@dataclass
class Plan:
    tasks: list[WorkerTask] = field(default_factory=list)
    candidates: list[Candidate] = field(default_factory=list)  # from deterministic checks
    records: list[TaskRecord] = field(default_factory=list)  # audit lines for deterministic checks and skipped tasks
    quiet: list[str] = field(default_factory=list)  # things found and deliberately not reported (external references ...)


def _evidence(store: TableStore, table: str, ref: str, role: str, shows: str) -> ResolvedEvidence | None:
    src = store.row_source(table, ref)
    if not src or not src[1]:
        return None
    chunk = store.chunk(src[0])
    return ResolvedEvidence(src[0], chunk[0], chunk[1], chunk[2], src[1], role, shows) if chunk else None


def _cap(items: list, n: int, plan: Plan, family: str, what: str) -> list:
    if len(items) > n:
        plan.records.append(TaskRecord(f"{family}-cap", family, f"{what}: {len(items) - n} more than the per-family limit of {n}", "skipped", "raise --max-tasks to include them"))
    return items[:n]


# ---- A: constraint compliance ----------------------------------------------------------------------------------------------------------
def family_a(facts: list[Fact], store: TableStore, plan: Plan, limit: int) -> None:
    rules = [f for f in facts if f.type == "constraint"]
    for i, f in enumerate(_cap(rules, limit, plan, "A", "constraints"), start=1):
        plan.tasks.append(WorkerTask(
            f"A{i}", "A", f"Rule: {f.subject} | {f.attribute} {f.operator} {f.value}" + (f" {f.unit}" if f.unit else ""),
            "Find the records in the room that this rule governs (rows of the tables, or passages) and test them against it. Return each violating record as a "
            "constraint_violation candidate citing the rule and the violating record. If the rule requires a document, approval, log, form or record that should exist, check that the room "
            "holds it and return missing_evidence when it does not. If nothing in the room can be tested against the rule, answer untestable and say why. If every governed record complies, answer passed.",
            f"{f.line()}\nStated in {_doc(store, f)}: \"{f.quote}\"", tables=related_tables(store, f"{f.subject} {f.attribute} {f.scope or ''} {f.value}")))


# ---- B: set and chain reconciliation ----------------------------------------------------------------------------------------------------
def family_b(store: TableStore, case: CaseResult, plan: Plan, limit: int) -> None:
    pairs: list[tuple[int, str, str, str, list[str], list[str], int]] = []
    for etype in case.entity_types():
        by_doc = ids_by_document(store, etype)
        docs = [d for d, ids in by_doc.items() if len(ids) >= 3]
        for a, b in combinations(sorted(docs), 2):
            sa, sb = set(by_doc[a]), set(by_doc[b])
            shared = sa & sb
            if len(shared) < 2:
                continue
            only_a, only_b = sorted(sa - sb), sorted(sb - sa)
            if only_a or only_b:
                pairs.append((min(len(sa), len(sb)), etype, a, b, only_a, only_b, len(shared)))
    pairs.sort(reverse=True)
    for i, (_n, etype, a, b, only_a, only_b, shared) in enumerate(_cap(pairs, limit, plan, "B", "identifier set differences"), start=1):
        by_doc = ids_by_document(store, etype)

        def listing(ids: list[str], doc: str) -> str:
            return "\n".join(f"  {ident}: {by_doc[doc][ident][1].strip()[:160]}" for ident in ids[:25]) + (f"\n  ... {len(ids) - 25} more" if len(ids) > 25 else "")

        plan.tasks.append(WorkerTask(
            f"B{i}", "B", f"{etype} identifiers differ between {a} and {b}",
            f"The two documents name {shared} {etype} identifiers in common, and these differ. For each identifier, look for a documented reason in the room (excluded, failed quality control, "
            "discarded, lost, not collected, not yet received) with search and find_entity. Return the unexplained ones, grouped, as one reconciliation_gap candidate each way, citing a line that names the "
            "identifier and a line from the other document that shows its list. Explained differences are not findings: mention them in the reason of a passed answer. Consider numbers in a stated total "
            "(a count) against the identifiers actually listed.",
            f"In {a} only ({len(only_a)}):\n{listing(only_a, a)}\nIn {b} only ({len(only_b)}):\n{listing(only_b, b)}"))
    _outside_range(store, case, plan, limit)


def _number(ident: str) -> int | None:
    runs = re.findall(r"\d+", ident)
    return int(runs[-1]) if runs else None


def _outside_range(store: TableStore, case: CaseResult, plan: Plan, limit: int) -> None:
    """Without the model: an identifier one document names whose number lies outside the range of another document's list of the same kind
    (SPL-044-B against a register of SPL-001 to SPL-038) has no record there. Inside the range is not raised: documents hold different subsets of a series."""
    made = 0
    for etype in case.entity_types():
        by_doc = ids_by_document(store, etype)
        for list_doc, ids in by_doc.items():
            nums = {i: n for i in ids if (n := _number(i)) is not None}
            if len(nums) < 3:
                continue
            lo_id, hi_id = min(nums, key=nums.get), max(nums, key=nums.get)
            for doc, others in by_doc.items():
                if doc == list_doc or len(set(ids) & set(others)) < 2:
                    continue
                outside = sorted((i for i in others if (n := _number(i)) is not None and (n < nums[lo_id] or n > nums[hi_id])), key=lambda i: _number(i))
                if not outside or made >= limit:
                    continue
                made += 1
                edge = hi_id if _number(outside[0]) > nums[hi_id] else lo_id
                ev, seen = [], set()
                for ident, (chunk_id, line) in [(i, others[i]) for i in outside[:4]] + [(edge, ids[edge])]:
                    info, quote = store.chunk(chunk_id), line.strip()[:300]
                    if info and len(quote) >= 8 and (chunk_id, quote) not in seen:
                        seen.add((chunk_id, quote))
                        ev.append(ResolvedEvidence(chunk_id, info[0], info[1], info[2], quote, "insufficient",
                                                   f"names {ident}" if ident in others else f"the list in {list_doc} ends at {ident}"))
                if len(ev) >= 2:
                    names = ", ".join(outside[:4])
                    plan.candidates.append(Candidate(
                        "B", "reconciliation_gap", f"{names} named in {doc} has no record in {list_doc} ({lo_id} to {hi_id})",
                        f"{doc} names {names}, but {list_doc} lists {len(ids)} identifiers of the same kind ({lo_id} to {hi_id}) and none numbered {'that' if len(outside) == 1 else 'those'}.",
                        "medium", 0.5, "needs_investigation", [f"the record for {names} in {list_doc}, or the document that accounts for it"], ev, entities=outside[:4],
                        computed=(f"{etype} identifiers outside the range {lo_id} to {hi_id} of {list_doc}. Every identifier of this kind written in {list_doc} ({len(ids)}): "
                                  + ", ".join(sorted(ids, key=lambda i: _number(i) or 0)[:60]) + "."), source="code", task_id=f"B-range-{made}"))
    plan.records.append(TaskRecord("B-code", "B", "identifiers outside the range of another document's list", "finding" if made else "passed", f"{made} candidates", 0, made))


# ---- C: temporal consistency ------------------------------------------------------------------------------------------------------------
def family_c(facts: list[Fact], store: TableStore, plan: Plan, limit: int) -> None:
    with store._lock:
        idents = {r[0] for r in store.conn.execute("SELECT DISTINCT ident FROM _mentions")}
    dated = {t.name: store.date_columns(t.name) for t in store.table_list()}
    dated = {n: c for n, c in dated.items() if c}
    keys: dict[str, set[str]] = {}
    for name in dated:
        info = store.tables[name]
        found: set[str] = set()
        for col in info.columns:
            for v in store.column_values(name, col.name)[:3000]:
                if isinstance(v, str) and norm_id(v) in idents:
                    found.add(norm_id(v))
        keys[name] = found
    pairs = [(len(keys[a] & keys[b]), a, b) for a, b in combinations(sorted(dated), 2) if len(keys[a] & keys[b]) >= 3]
    pairs.sort(reverse=True)
    n = 0
    for shared, a, b in _cap(pairs, limit, plan, "C", "dated table pairs"):
        n += 1
        ta, tb = store.tables[a], store.tables[b]
        plan.tasks.append(WorkerTask(
            f"C{n}", "C", f"Order of events between {ta.title} ({ta.filename}) and {tb.title} ({tb.filename})",
            f"These two tables describe the same entities ({shared} identifiers in common) and both hold dates. Check causal ordering for entities present in both, using norm_id() to join and "
            "days_between() or hours_between() on the date columns: something is collected, started or dispatched before it is received, analysed, finished or reported; a recovery, end or "
            "closing date is not before its start. Return impossible orderings as temporal_inconsistency candidates (cite one row of each table), grouped by the kind of ordering broken. "
            "If every ordering is possible, answer passed.",
            f"{a}: date columns {', '.join(dated[a])}; {b}: date columns {', '.join(dated[b])}", tables={a, b}))
    events = sorted((f for f in facts if f.type == "event"), key=lambda f: f.when or "")
    if len(events) >= 3:
        listing = "\n".join(f"- {f.when or '?'} | {f.subject} | {f.attribute} {f.value} | {_doc(store, f)}" for f in events[:70])
        plan.tasks.append(WorkerTask(
            f"C{n + 1}", "C", "Events stated in the documents against each other and against the dated tables",
            "Check the stated events for impossible or conflicting states: the same item recorded in two incompatible states on the same date, an event dated while the document says the "
            "resource was unavailable, a time or date that cannot be right, a notification or report dated before the event it reports. Use timeline(entity, window) to see the dated rows of the "
            "tables for an entity, and read or search for the context. Return temporal_inconsistency candidates citing both statements. State a progression over time (in progress, then complete) is not a conflict.",
            "Events:\n" + listing))


# ---- D: recomputation -------------------------------------------------------------------------------------------------------------------
def _numeric(f: Fact) -> bool:
    return bool(re.search(r"\d", f.value or ""))


def family_d(facts: list[Fact], store: TableStore, plan: Plan, limit: int) -> None:
    aggregates = [f for f in facts if f.type in ("observation", "assertion") and _numeric(f) and (f.aggregate_of or _AGGREGATE.search(f"{f.attribute} {f.subject}"))]
    definitions = [f for f in facts if f.type == "definition"]
    for i, f in enumerate(_cap(aggregates, limit, plan, "D", "stated aggregates"), start=1):
        mine = [d for d in definitions if words(d.attribute + " " + d.subject + " " + d.value) & words(f.attribute + " " + f.subject)][:3]
        plan.tasks.append(WorkerTask(
            f"D{i}", "D", f"Stated figure: {f.subject} | {f.attribute} = {f.value}" + (f" {f.unit}" if f.unit else ""),
            "Recompute this figure from the data in the tables, following any definition of the metric stated in the room (see the definitions below, or use get_facts), and compare it with the stated "
            "value to the precision it is written with. If it does not match, return a direct_contradiction candidate citing the stated figure and the rows it rests on, with the recomputed value "
            "and the query in computed. If it matches, answer passed. If the underlying data is not in the tables, answer untestable and say what is missing.",
            f"{f.line()}" + (f"\nComputed from: {f.aggregate_of}" if f.aggregate_of else "") + f"\nStated in {_doc(store, f)}: \"{f.quote}\""
            + ("\nDefinitions:\n" + "\n".join(f"- {d.line()} | \"{d.quote[:200]}\"" for d in mine) if mine else ""),
            tables=related_tables(store, f"{f.subject} {f.attribute} {f.aggregate_of or ''} {f.scope or ''}")))


# ---- E: universal and negative assertions ------------------------------------------------------------------------------------------------
def family_e(facts: list[Fact], store: TableStore, plan: Plan, limit: int) -> None:
    claims = [f for f in facts if f.type in ("assertion", "observation", "conclusion") and _UNIVERSAL.search(f"{f.quote} {f.value}")]
    for i, f in enumerate(_cap(claims, limit, plan, "E", "universal or negative assertions"), start=1):
        plan.tasks.append(WorkerTask(
            f"E{i}", "E", f"Assertion: {f.subject} | {f.attribute}: {f.value}",
            "This statement is universal or negative (all, no, none, every, only, never, fully). Look for a counterexample in the registers and primary records: rows of tables that contradict it (sql), "
            "entries in any event, incident, deviation or exception register, and notes (search, find_entity). Return a direct_contradiction candidate citing the statement and the counterexample. "
            "If you checked the relevant records and found none, answer passed. If the room holds no record that could test it, answer untestable and say which record is missing.",
            f"{f.line()}\nStated in {_doc(store, f)}: \"{f.quote}\"", tables=related_tables(store, f"{f.subject} {f.attribute} {f.quote}")))


# ---- F: data integrity ------------------------------------------------------------------------------------------------------------------
def family_f(store: TableStore, plan: Plan, limit: int) -> None:
    series: dict[tuple, list[tuple[str, str]]] = defaultdict(list)
    dups: list[Candidate] = []
    for t in store.table_list():
        if t.rows < 5:
            continue
        for col in t.columns:
            vals = store.column_values(t.name, col.name)
            nonnull = [v for v in vals if v is not None]
            if len(nonnull) < 8:
                continue
            numbers = [v for v in nonnull if isinstance(v, (int, float)) and not isinstance(v, bool)]
            if len(numbers) == len(nonnull) and len(set(numbers)) >= 5 and not _is_counter(numbers):
                series[tuple(vals)].append((t.name, col.name))
            texts = [v for v in nonnull if isinstance(v, str)]
            if len(texts) == len(nonnull) and len(nonnull) >= 10 and len(set(texts)) >= 0.9 * len(texts) and len(set(texts)) < len(texts) and _id_like(texts):
                repeated = [v for v, n in _counts(texts).items() if n > 1][:3]
                for v in repeated[:2]:
                    rows = store.query(f'SELECT _ref FROM "{t.name}" WHERE "{col.name}" = ? ORDER BY _rid LIMIT 2', params=(v,)).rows
                    ev = [e for r in rows if (e := _evidence(store, t.name, r[0], "contradicting", f"{col.header} = {v}"))]
                    if len(ev) >= 2:
                        dups.append(Candidate("F", "data_integrity_issue", f"{col.header} value {v!r} is used for more than one row in {t.title}",
                                              f"In {t.filename} ({t.title}) the column {col.header} is otherwise unique, but the value {v!r} appears on {len(rows)} rows. "
                                              "A value that identifies one thing is assigned to two.", "medium", 0.6, "needs_investigation", ["confirmation that the repeated value is an error and not a shared code"],
                                              ev, computed=f"column {col.name} of {t.name}: duplicated value {v!r}", entities=[str(v)], source="code", task_id=f"F-{t.name}-{col.name}"))
    for key, members in series.items():
        distinct = {m for m in members}
        if len(distinct) >= 2 and len({m[0] for m in distinct}) >= 2:
            (ta, ca), (tb, cb) = sorted(distinct)[:2]
            la, lb = store.tables[ta], store.tables[tb]
            ev = []
            for t, c in ((ta, ca), (tb, cb)):
                for ordinal in (1, store.tables[t].rows):
                    res = store.query(f'SELECT _ref FROM "{t}" WHERE "{c}" IS NOT NULL ORDER BY _rid {"" if ordinal == 1 else "DESC"} LIMIT 1').rows
                    if res and (e := _evidence(store, t, res[0][0], "supporting", f"{store.tables[t].columns[[x.name for x in store.tables[t].columns].index(c)].header} in {store.tables[t].title}")):
                        if all((e.chunk_id, e.quote) != (x.chunk_id, x.quote) for x in ev):
                            ev.append(e)
            if len(ev) >= 2:
                dups.append(Candidate("F", "data_integrity_issue", f"Identical data series in {la.title} ({la.filename}) and {lb.title} ({lb.filename})",
                                      f"The column {ca} of {ta} and the column {cb} of {tb} hold exactly the same {len(key)} values in the same order, although they come from different "
                                      "places in the data room. Independent series do not match value for value; one may be a copy of the other.", "high", 0.6, "needs_investigation",
                                      ["the instrument or source records that show whether the two series are independent"], ev,
                                      computed=f"{ta}.{ca} == {tb}.{cb} for all {len(key)} rows", source="code", task_id=f"F-{ta}-{tb}"))
    for c in _cap(dups, limit, plan, "F", "deterministic integrity candidates"):
        plan.candidates.append(c)
    plan.records.append(TaskRecord("F-code", "F", "identical series and duplicated keys over every table", "finding" if dups else "passed", f"{len(dups)} candidates", 0, len(dups)))
    tables = [t for t in store.table_list() if t.rows >= 5]
    groups = [tables[i:i + 4] for i in range(0, len(tables), 4)]
    for n, group in enumerate(_cap(groups, max(1, limit // 4), plan, "F", "table groups"), start=1):
        plan.tasks.append(WorkerTask(
            f"F{n}", "F", "Plausibility of " + ", ".join(t.name for t in group),
            "Run plausibility checks with sql on these tables: values outside a plausible or stated range, a date earlier than another date in the same row where order matters, impossible dates, "
            "readings recorded after the date the same source says it ended or was recovered, negative or zero values where a measure cannot be, text in numeric columns, and suspicious rounding. "
            "Return data_integrity_issue candidates citing the rows. If nothing is implausible, answer passed.", "", tables={t.name for t in group}))


# ---- G: reference resolution ------------------------------------------------------------------------------------------------------------
@dataclass
class Reference:
    fact: Fact
    key: frozenset


def family_g(facts: list[Fact], store: TableStore, case: CaseResult, pool: list, llm: Any, plan: Plan, limit: int) -> None:
    refs = [f for f in facts if f.type == "reference" and f.value]
    unresolved: list[tuple[Fact, list[frozenset]]] = []
    present = 0
    for f in refs:
        keys = _item_keys(f.value)
        if not keys:
            continue
        own = " ".join(f.quote.split()).lower()
        if any(not (line and line in own) and not (not line and cid == f.chunk_id) and any(_names(w, k) for k in keys) for cid, w, line in pool):
            present += 1
            continue
        if _resolved_by_entity(store, f):
            present += 1
            continue
        unresolved.append((f, keys))
    items: list[tuple[frozenset, list[Fact]]] = []
    for f, keys in unresolved:
        for key, group in items:
            if key <= keys[0] or keys[0] <= key:
                group.append(f)
                break
        else:
            items.append((keys[0], [f]))
    kinds: dict[int, str] = {}
    ask = []
    for n, (_key, group) in enumerate(items):
        if any(_DECLARED_ABSENT.search(g.quote) for g in group):
            kinds[n] = "expected_artifact"
        else:
            ask.append(n)
    if ask:
        prompt = "References that no file, sheet, heading, caption, table row or identifier in the data room matches:\n" + "\n".join(
            f"{n}. {items[n][1][0].value!r}, in {_doc(store, items[n][1][0])}: \"{items[n][1][0].quote[:240]}\"" for n in ask)
        try:
            answer = llm.generate_structured_output(prompt, ReferenceKinds, system=(
                "For each numbered reference decide its kind: expected_artifact (a document, approval, record, log, form or annex that the submission, permit or report says belongs with it or "
                "that a data room for this matter should hold), external_standard (a published standard, guideline, law, public dataset or product, or an earlier version of such a document), "
                "internal_part (a table, figure or section of the same document), or unclear. Answer with number, kind and a one-line reason."))
            kinds.update({k.number: k.kind for k in answer.items})
        except Exception as exc:
            plan.quiet.append(f"reference kinds could not be decided ({str(exc)[:100]}); none of {len(ask)} undeclared references reported")
    made = 0
    for n, (_key, group) in enumerate(items):
        kind = kinds.get(n, "unclear")
        label = group[0].value
        if kind != "expected_artifact":
            plan.quiet.append(f"{kind}: {label} (in {_doc(store, group[0])})")
            continue
        if made >= limit:
            plan.records.append(TaskRecord("G-cap", "G", f"{label}", "skipped", "raise --max-tasks to include it"))
            continue
        ev, seen = [], set()
        for g in group:
            info = store.chunk(g.chunk_id)
            if info and (g.chunk_id, g.quote) not in seen and len(ev) < 4 and len(g.quote) >= 8:
                seen.add((g.chunk_id, g.quote))
                ev.append(ResolvedEvidence(g.chunk_id, info[0], info[1], info[2], g.quote, "insufficient", f"Refers to {g.value}"))
        if not ev:
            continue
        made += 1
        where = ", ".join(sorted({_doc(store, g) for g in group}))
        plan.candidates.append(Candidate(
            "G", "missing_evidence", f"Referenced item not in the data room: {label}",
            f"{where} refers to \"{label}\", but no file, sheet, heading, caption, table row or identifier in the data room is named like it, and every document was read.",
            "medium", 0.65, "needs_investigation", [f"the document or record \"{label}\""], ev, entities=[label], source="code", task_id=f"G{n}"))
    plan.records.append(TaskRecord("G-code", "G", f"{len(refs)} references resolved in a fixed order", "finding" if made else "passed",
                                   f"{present} found in the room, {len(items)} unresolved, {made} reported", 0, made))


def _resolved_by_entity(store: TableStore, f: Fact) -> bool:
    for token in re.findall(r"[A-Za-z]{1,6}[-_]\d+(?:[-_][A-Za-z0-9]+)?", f.value):
        for hit in find_entity(store, token, limit=10):
            if hit["chunk_id"] != f.chunk_id:
                return True
    return False


# ---- H: conclusion audit -----------------------------------------------------------------------------------------------------------------
def family_h(facts: list[Fact], store: TableStore, plan: Plan, limit: int) -> None:
    for i, f in enumerate(_cap([f for f in facts if f.type == "conclusion"], limit, plan, "H", "conclusions"), start=1):
        plan.tasks.append(WorkerTask(
            f"H{i}", "H", f"Conclusion: {f.subject} | {f.attribute}: {f.value[:120]}",
            "Audit this conclusion. Gather the data and statements it rests on (sql, search, get_facts, read), then check: is the comparison like for like (same population, method, scope and period)? "
            "is a causal claim backed by a design or a control? is the sample large enough? does a source in the room disclaim this inference? is the certainty (confirms, demonstrates, proves) larger than the "
            "evidence? does anything else in the room undermine it? Return an unsupported_conclusion candidate citing the conclusion and the evidence that falls short, with the reasons in the description. "
            "If it is supported, answer passed.",
            f"{f.line()}\nStated in {_doc(store, f)}: \"{f.quote}\"", tables=related_tables(store, f"{f.subject} {f.attribute} {f.quote}")))


def _doc(store: TableStore, f: Fact) -> str:
    info = store.chunk(f.chunk_id)
    return f"{info[1]}, {info[2]}" if info else f.document_id


def _counts(values: list) -> dict:
    out: dict[Any, int] = defaultdict(int)
    for v in values:
        out[v] += 1
    return out


def _is_counter(numbers: list[float]) -> bool:
    """1, 2, 3 ... or any strictly steady sequence: a row counter, not a measurement."""
    if len(numbers) < 3:
        return False
    steps = {round(b - a, 9) for a, b in zip(numbers, numbers[1:])}
    return len(steps) == 1


def _id_like(texts: list[str]) -> bool:
    share = sum(bool(re.search(r"\d", t)) and len(t) <= 24 and " " not in t.strip() for t in texts) / len(texts)
    return share >= 0.8
