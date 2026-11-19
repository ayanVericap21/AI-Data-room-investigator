"""Blocking by code: clusters found from the labels extraction gave the claims, so a comparison never depends on a model noticing it.

Two routes the LLM grouping and the nearest-neighbour matching miss:

* shared labels: claims in different documents (or numbers or dates in two places of one document) whose metric names share a word that few claims use ("purchase order", "headcount"), or whose subject and
  metric words together share two such words ("practical completion"), and that state the same kind of value, are one cluster. Common words (the metric "date") never match.
* negation: a claim that says there is none of something ("No subcontractors are engaged", "No safety incidents have occurred") and the claims
  that assert that thing are one rule cluster: the negation is the rule and the claims that assert the thing are the records.

``types_compatible`` also filters the nearest-neighbour matches: a date never goes with a yes/no status or an identifier.
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from itertools import combinations

from app.schemas.claim import ClaimView, Cluster
from app.services.claim_matching import different_records, one_sentence_in_one_document, same_table, table_rows_about_one_item, tokens
from app.services.clusters import clusters_from_matches

# words that name a kind of field, not what the field is about: a shared "date" or "value" says nothing
GENERIC = frozenset({"date", "value", "identifier", "number", "name", "type", "status", "document", "title", "amount", "description", "period", "item",
                     "reference", "detail", "information", "role"})
RARE_FLOOR, RARE_CAP, RARE_SHARE = 6, 12, 0.04  # a word is rare when at most this many claims use it: 4% of the claims, at least 6 and at most 12
SAME_METRIC_MAX = 8  # claims whose metrics are word for word the same go together when there are at most this many of them
PARTNERS = 6  # a claim is matched with at most this many others by a shared label
NEGATION_PARTNERS = 8  # at most this many claims are put with one negation
INSIDE_TYPES = frozenset({"quantity", "date"})  # the claims of one document that are compared with each other
KINDS = {"quantity": "number", "date": "time", "identifier": "id"}  # every other type is text (name, state, statement)
NEGATIVE_VALUES = frozenset({"no", "none", "never", "nobody", "nothing", "not any"})  # not "nil" or "zero": a balance of nil is a quantity
NEGATION = re.compile(r"\b(?:no|none of the|nothing|nobody|never any|not any|without any)\s+((?:[a-z][a-z-]*\s+){0,2}[a-z][a-z-]*)", re.IGNORECASE)
# words that end the thing a negation names ("no subcontractors ARE engaged"): the thing is what comes before them
ENDS_PHRASE = frozenset({"have", "has", "had", "are", "is", "was", "were", "been", "be", "being", "occurred", "occur", "occurs", "engaged", "reported", "recorded",
                         "used", "made", "taken", "found", "on", "in", "at", "to", "of", "the", "a", "an", "for", "and", "or", "will", "shall", "can", "may", "did",
                         "does", "do", "with", "by", "this", "that", "any", "from", "during", "under", "within", "since", "until"})


def _kind(claim: ClaimView) -> str:
    return KINDS.get(claim.value_type or "", "text")


def types_compatible(a: ClaimView, b: ClaimView) -> bool:
    """Two claims can state one attribute only when their values are the same kind of thing, or a count and a text (a status and a count) or a date and a
    count (a date and a duration). A date set against a yes/no, an identifier against a date, a number against an identifier are never one attribute."""
    ka, kb = _kind(a), _kind(b)
    return ka == kb or {ka, kb} in ({"number", "text"}, {"number", "time"})


def inside_document_ok(a: ClaimView, b: ClaimView) -> bool:
    """Whether two claims of ONE document are worth comparing: they come from different passages (one passage is one statement) and both state a number or a date,
    which can be checked against each other. Statements or names inside a document say too many different things for two of them to mean a conflict.
    Claims of different documents are always allowed here."""
    if a.document_id != b.document_id:
        return True
    return a.chunk_id != b.chunk_id and (a.value_type or "") in INSIDE_TYPES and (b.value_type or "") in INSIDE_TYPES


def rare_limit(claims: int) -> int:
    return max(RARE_FLOOR, min(RARE_CAP, int(claims * RARE_SHARE)))


def shared_label_clusters(views: list[ClaimView]) -> list[Cluster]:
    """Clusters of claims in different documents (or in different passages of one document) that state the same kind of value and share a rare word of their labels (see the module's text)."""
    limit = rare_limit(len(views))
    metric = [frozenset(t for t in tokens(v.metric) - GENERIC if len(t) > 2) for v in views]
    both = [metric[i] | frozenset(t for t in tokens(v.subject) - GENERIC if len(t) > 2) for i, v in enumerate(views)]
    metric_df = Counter(t for ts in metric for t in ts)
    both_df = Counter(t for ts in both for t in ts)
    index: dict[str, list[int]] = defaultdict(list)
    for i, ts in enumerate(both):
        for t in ts:
            if both_df[t] <= limit or (t in metric[i] and metric_df[t] <= limit):  # rare among all the labels, or among the metrics (a subject word many claims repeat)
                index[t].append(i)
    scored: dict[tuple[int, int], int] = {}
    for members in index.values():
        for i, j in combinations(sorted(set(members)), 2):
            if (i, j) in scored:
                continue
            a, b = views[i], views[j]
            if not inside_document_ok(a, b) or (a.value_type or "") != (b.value_type or ""):
                continue  # numbers or dates in two places of one document may go together; claims of one passage are parts of one statement
            shared_metric = {t for t in metric[i] & metric[j] if metric_df[t] <= limit}
            shared_both = {t for t in both[i] & both[j] if both_df[t] <= limit}
            if not shared_metric and len(shared_both) < 2:
                continue
            if same_table(a, b) or not table_rows_about_one_item(a, b) or one_sentence_in_one_document(a, b):
                continue
            if a.value_type != "identifier" and different_records(a, b):  # two ids of one kind (two orders) may be a mistake in one; two records are not
                continue
            scored[(i, j)] = len(shared_metric) * 2 + len(shared_both)
    by_metric: dict[frozenset, list[int]] = defaultdict(list)  # the same metric, word for word ("purchase order" in two documents): matched whatever the subjects
    for i, ts in enumerate(metric):
        if ts:
            by_metric[ts].append(i)
    for members in by_metric.values():
        if 2 <= len(members) <= SAME_METRIC_MAX:
            for i, j in combinations(members, 2):
                a, b = views[i], views[j]
                if (inside_document_ok(a, b) and (a.value_type or "") == (b.value_type or "") and not same_table(a, b) and table_rows_about_one_item(a, b)
                        and not one_sentence_in_one_document(a, b) and (a.value_type == "identifier" or not different_records(a, b))):
                    scored[(i, j)] = max(scored.get((i, j), 0), 10)
    taken: Counter = Counter()
    found = []
    for (i, j), score in sorted(scored.items(), key=lambda kv: (-kv[1], kv[0])):
        if taken[i] >= PARTNERS or taken[j] >= PARTNERS:
            continue
        taken[i] += 1
        taken[j] += 1
        found.append((views[i].id, views[j].id, 1.0))
    return clusters_from_matches(found, "lexical")


def negated_terms(claim: ClaimView) -> frozenset[str]:
    """The thing a claim says there is none of: the words after "no" up to the first verb or preposition ("No subcontractors are engaged" -> subcontractor),
    else, for a claim whose value is a negative, the words of its metric. Empty when the claim negates nothing."""
    match = NEGATION.search(claim.claim_text or "")
    words: list[str] = []
    if match:
        for w in re.findall(r"[a-z][a-z-]*", match.group(1).lower()):
            if w in ENDS_PHRASE:
                break
            words.append(w)
    elif str(claim.value or "").strip().lower() in NEGATIVE_VALUES:
        words = [w for w in re.findall(r"[a-z][a-z-]*", (claim.metric or "").lower()) if w not in ENDS_PHRASE]
    return frozenset(t for t in tokens(" ".join(words)) if t not in GENERIC and len(t) > 3)


def negation_clusters(views: list[ClaimView]) -> list[Cluster]:
    """For each claim that says there is none of something: a rule cluster of that claim (the rule) and the claims that mention that thing (the records)."""
    limit = rare_limit(len(views))
    texts = [tokens(f"{v.subject or ''} {v.metric or ''} {v.claim_text or ''} {v.value or ''}") for v in views]
    df = Counter(t for ts in texts for t in ts)
    found = []
    for i, negated in enumerate(views):
        terms = {t for t in negated_terms(negated) if df[t] <= limit * 2}
        if not terms:
            continue
        partners = []  # the claims that mention what this claim says there is none of
        for j, other in enumerate(views):
            hits = terms & texts[j]
            if j == i or not hits or one_sentence_in_one_document(negated, other) or negated.claim_text == other.claim_text:
                continue
            partners.append((-len(hits), j))
        records = [views[j].id for _score, j in sorted(partners)[:NEGATION_PARTNERS]]
        if records:
            found.append(Cluster(claim_ids=[negated.id, *records], matched_by="rule", rule_id=negated.id))
    return found
