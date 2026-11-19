"""Judges claim groups with the LLM and saves findings whose quotes are verified in code.

A group is a fact and every claim that may state it (the claims the grouping found, joined through their candidates). The whole group goes to
the model in one verdict: when all its claims state the same value they are consistent, when any one differs from the rest the group is a
contradiction, and a difference that the excerpts themselves explain is a contextual difference. Every claim and excerpt is shown once; the model
copies a quote for each claim it compared, and ``submit_finding`` verifies that each quote is an exact substring of its chunk, so a wrong or invented
quote never reaches the report. Groups whose claims code finds to state the same value are corroboration and never sent to the model.
"""
from __future__ import annotations

import logging
import re
import time
from decimal import Decimal
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from itertools import combinations
from typing import Callable

from pydantic import BaseModel, Field
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.llm.base import LLMClient
from app.llm.errors import LLMError, LLMInvalidOutputError, LLMTransientError
from app.models import Claim, Document, DocumentChunk, Evidence, Finding
from app.models.base import Classification, Severity
from app.schemas.claim import Candidate
from app.tools.submit import find_source_span
from app.services.blocking import GENERIC, KINDS
from app.services.claim_tags import EARLIER_BASES, LATER_BASES, tags_of
from app.services.cluster_split import COHERENCE_FLOOR, ClaimSimilarity
from app.services.candidate_matching import TABLE_TYPES, normalize_period, normalize_unit, subjects_overlap, tokens
from app.services.doc_profiles import render_cards
from app.services.identifiers import entity_labels
from app.services.internal_checks import clause_quote
from app.services.numbers import parse_range, value_supported_by_text
from app.services.severity import contradiction_severity, missing_severity
from app.services.supersession import superseded_documents
from app.services.value_rules import bound_direction, compare_values, comparison_note, normalise_text, one_contains_other, unit_info, within_limit
from app.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)
ProgressFn = Callable[[str, dict], None]

# Verdicts that become findings (with quotes). A group the model calls not comparable is only counted.
REPORTED = {Classification.contradiction, Classification.contextual_difference, Classification.missing_information,
            Classification.consistent, Classification.insufficient_evidence}
NEEDS_QUOTES_TO_SAVE = {Classification.consistent, Classification.insufficient_evidence, Classification.missing_information}  # without quotes: counted, not saved
ROLE = {Classification.contradiction: "contradicting", Classification.contextual_difference: "contextual",
        Classification.missing_information: "insufficient", Classification.consistent: "supporting",
        Classification.insufficient_evidence: "insufficient"}
MAX_RETRY_ROUNDS = 8  # rounds of asking again for groups with no verdict: enough to halve a call down to one group and ask again
MAX_EVIDENCE = 16  # claims quoted in one finding: the report stays readable
MAX_GROUP_CLAIMS = 15  # a set of claims joined by candidates that is larger than this is not one fact (a cluster of the grouping model, or claims linked by words): it is divided by meaning (cluster_split)
MIN_QUOTE = 8  # the shortest quote a finding accepts (submit_finding)
# wording that marks a figure as the later or actual basis of an earlier planned, contracted or original one: the two differ in basis, not in fact
LATER_BASIS = re.compile(r"\b(actual\w*|final|revised|amended|adjusted|as[- ]built|commenced|started|began|completed|achieved|including (?:approved )?variations?)\b", re.IGNORECASE)
EARLIER_BASIS = re.compile(r"\b(required|contracted|contractual|planned|scheduled|original(?:ly)?|budget(?:ed)?|forecast|target(?:ed)?|estimated|commences?|"
                           r"(?:shall|will) (?:start|begin|commence))\b", re.IGNORECASE)
HEDGE = re.compile(r"\b(about|approximately|approx\.?|around|circa|roughly|nearly|almost|more than|less than|at least|at most|up to)\b|~", re.IGNORECASE)
# a claim whose label names only the document it is in (its title, type, revision) identifies that document: two documents have different ones by nature
# metadata: a document's own date, as-of date, preparer or title, said about the document and not about what it covers
DATE_META = frozenset({"date", "as", "at", "dated", "prepared", "issued", "sent", "entries", "through", "extract"})
CONDITION_MET = re.compile(r"\b(?:with|under|has|had|have|obtained|given|received|per)\b[^.]{0,40}\b(?:consent|approval|authori[sz]ation|permission|notice)\b|"
                           r"\b(?:meets?|satisf(?:y|ies|ied)|compl(?:y|ies|ied)\s+with|fulfil\w*)\b[^.]{0,40}\b(?:rule|condition|requirement)\b", re.IGNORECASE)
CONDITION_NOT_MET = re.compile(r"\b(?:without|lacks?|lacking|no|not|absent|missing)\b[^.]{0,25}\b(?:consent|approval|authori[sz]ation|permission|notice)\b", re.IGNORECASE)
DOCUMENT_LABELS = frozenset({"revision", "version", "edition", "issue", "draft", "supersede", "supersedes", "title", "document", "type"})
CURRENCY = re.compile(r"\b(GBP|EUR|USD|INR|AUD|CAD|CHF|JPY|CNY|SGD|AED|ZAR|NZD)\b|(£)|(€)|(\$)")
SLASH_DATE = re.compile(r"^\s*(\d{1,2})[/.\-](\d{1,2})[/.\-](\d{4})\s*$")
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
WORDED = {None, "name", "state", "statement"}  # values written in words: the same words in another order are one value

SYSTEM = """You judge groups of claims taken from the documents of one collection. A group holds claims that may state the same attribute of the same subject. Give ONE verdict per group, about all of its claims together.

Rules:
1. Use ONLY the claims and excerpts shown. Never invent text, values or context.
2. Compare like with like: same subject, metric, unit, period and scope. Decide from what the excerpts themselves state. "Code check" lines say whether code found two claims' values to agree within their stated precision (scale words, units of one kind and rounding already allowed for), to differ (with the gap), or could not compare them. Trust them for the numbers; you decide what a difference means.
3. A group marked SAME RECORD is never set aside: its claims are about the same attribute of one record; compare their values. Otherwise, first set aside (set_aside, by claim number) every claim that is not about the same attribute of the same subject as the others: a different subject or metric, or explicit periods or scopes that neither overlap nor nest. Shared words, digits or phrasing do not make claims comparable. A claim whose own sentence names a record (a sample, an item, a row) as its basis states that record's value: it is about the same attribute as another claim that gives the same kind of value (a date, an amount) for that same record, whatever the two labels are. Claims that differ only in basis or version (planned and actual, original and final, as of two dates) are about the same attribute: do not set them aside. Judge the claims that remain (sets that each state another attribute: one verdict for each set, rule 7). When fewer than two remain, same_attribute is false and nothing else is judged.
4. CLASSIFICATION of the claims that remain (stop at the first step that applies):
   a. consistent: every value agrees under every reasonable reading (after units, scale, sign, format and spelling are normalised). If the readings split, it is not consistent.
   b. The values do not all agree. Even ONE claim that differs from the rest, however many agree with each other, makes the group a contradiction. BASIS: values of the same kind for the same item that differ only in basis (planned, budgeted, forecast, target, designed or estimated versus actual, installed, measured, reported or contracted) state the same attribute: name the bases and compare; different bases are a contextual_difference, the same basis a contradiction. contextual_difference: the excerpts themselves state a distinguishing context (a different period, scope, condition, definition, version or as-of date) that fully accounts for EVERY difference; name it in the description. contradiction: no such context is stated, or the stated one cannot account for the gap: it covers only part of it, it is logically impossible (a part larger than its whole), or the document's own stated basis or definition rules it out. Quote the sentence that states the context in the description. Two documents or editions giving different values for the same thing is a contradiction; never invent an explanation that is not in the excerpts. A version or as-of date explains a gap only when an excerpt states the change (an event, the date it took effect, or that a later edition supersedes the earlier one); different document dates alone never explain a gap. A different label is a context difference only when an excerpt states what each figure includes or excludes.
   c. PROGRESSION OVER TIME (an addition to b): when every claim states the date or stage its own value or status holds at and those differ, the claims are successive states of one thing: contextual_difference, naming the dates or stages and saying the later statement supersedes the earlier. If any claim does not state its own date or stage, apply b.
   d. missing_information: a specific, nameable item that the comparison needs is absent: the base of a percentage, an anchor date, a fiscal year end, or an item an excerpt itself says is not attached, available on request, on file or in process. Name the item. Whether it exists elsewhere in the collection is checked by code, not by you.
   e. insufficient_evidence: everything needed is shown but the claims cannot be settled: a vague or hedged value, ambiguous wording, an unclear format, a rounding that leaves the outcome open (5.5 against a limit of 5), a currency without a stated rate, or two readings that give different outcomes.
5. Evidence (full verdict): give one evidence entry for EACH claim that remains: claim (its number), value (exactly as shown) and quote, copied EXACTLY from that claim's excerpt: the shortest passage that shows the value with the words or unit next to it (at least 8 characters, never the bare number). For a claim read from a table, quote its own row including its label (its ID or name), never a neighbouring row that shows the same number.
6. A full verdict also gives title (one line: what differs), description (each distinct value, which document and which place (page, section or sheet) states it, which claims agree with which, whether they are in one document or several, why they conflict or what context separates them), severity, confidence (0 to 1), materiality_reasoning (one sentence) and attribute_reasoning (one short sentence on what the claims measure).
7. Never do arithmetic. Return one verdict per group, using the group number shown; when a group holds two separate issues (a revision that supersedes another, and a value that breaks a limit), return one verdict for each, with the same group number and only its own claims. Two claims read from one sentence are one statement.

SEVERITY: consistent is informational. insufficient_evidence is low. missing_information is medium, or high when another figure or obligation depends on the missing item. contextual_difference is low, or medium when the explaining context appears in only one document or only in a note. contradiction is critical when the gap is large (5% or more, or any identifier, name, yes/no or date that cannot both be true) AND the claim is prominent (a title, summary, total, definition, obligation or signature block); high when it is large OR prominent; low when the gap is under 1%; otherwise medium.

RULES BY TYPE OF VALUE
- Numbers: they agree when their stated precision allows it (10.2 million and 10,213,000; more than 500 and 612). "About", "approximately" or "around" before a number means within 10% of it, so a value inside that band agrees (about 0.5 million covers 450,000 to 550,000). Check scale words and table headings ("in thousands"), units (1.5 t and 1,500 kg agree) and sign (a bracketed table amount is negative). Compare currencies only in the same currency; two currencies without a stated rate are insufficient_evidence, never a contradiction; an amount with no currency takes the currency its document's other amounts use. A percentage against an absolute figure needs a stated base, else missing_information. "Rose 2%" against 10% and 12% can mean percent or percentage points: if the readings disagree, insufficient_evidence. A part larger than its whole (a half-year above its full year, a total that excludes something above the total that includes it) is a contradiction whatever context is named. A balance is compared only at the same date, a flow only over the same period.
- Dates: a date covers the whole period it names (2024 is the full year; Q3 2024 contains 1 July 2024). A single or approximate date inside a stated range agrees with it (a central estimate and its interval). Day and month order that the document does not settle is insufficient_evidence. A fiscal period equals a calendar one only when its year end is stated. A relative date ("30 days after completion") needs its anchor date. Years and months convert exactly; start plus term against an end date allows one day.
- Identifiers and phone numbers (certificate, order, account, serial numbers, codes): equal only when identical once spaces, hyphens, dots, slashes and case are ignored. Any other difference is a contradiction only when both claims label the same thing (the same certificate, order, account or item). Identifiers of two different things (two tools, two rows, two orders) do not conflict: set them aside. Never judge them by similarity.
- Names: equal when case, punctuation, word order and legal-form words (Ltd, Limited, Inc, GmbH) are ignored, or when an alias is stated in the excerpts. Any other differing word is a contradiction when both name the same role.
- Yes/no and status: compare polarity and quantifier at the same date. None against a count above zero, all against none, or opposite polarity is a contradiction; "not all" agrees with "none"; a status that differs between two stated dates is a contextual_difference.
- Statements: they agree if one implies the other and contradict if both cannot be true. A list differs from a longer or shorter list only when one claims to be complete (only, all, exclusively). A hedge (generally, mostly, significantly) set against specific exceptions is insufficient_evidence.

OUTPUT FORMAT (strict). Return exactly one JSON object and nothing else: {"verdicts": [ ... ]}, one verdict per group.
SHORT verdict, the only one allowed when same_attribute is false or the claims that remain are consistent: group, same_attribute, attribute_reasoning (at most 15 words), set_aside, classification: consistent. No other field.
FULL verdict, every other group, ALL of these fields: group; same_attribute: true; attribute_reasoning; set_aside (empty when none); classification: contradiction, contextual_difference, missing_information or insufficient_evidence; severity: critical, high, medium, low or informational; confidence: 0 to 1; title, description, materiality_reasoning: strings, never null; evidence: a list of {"claim": number, "value": string, "quote": string}, one per claim that remains.
Use no value outside these lists and add no other fields, comments or markdown."""


RULE_SYSTEM = """You check records against a rule. Each group holds a rule, a status or a universal or negative statement ("no subcontractors", "all complied": the first claim, marked in the group's heading), the records that may fall under it, and perhaps statements that claim compliance. Give ONE verdict per group.

Rules:
1. Use ONLY the claims and excerpts shown. Never invent text, values or context.
2. Read the rule for its scope (a place, a kind of item, some people, a period), for what it prohibits, requires, limits or suspends, and for any exception or condition it states ("unless approved").
3. A record is made of the claims about one item, row, event or sample: the claims that show what it is, where it was, when it happened and who did it (they place it inside or outside the rule's scope), and the claims that show what was done. Decide the scope by meaning: a rule about a kind of item covers an item whose own claims describe it as that kind; a rule about a place covers a record at that place; a rule about all trenches covers each trench; a rule about a period covers an event dated inside it, even when the date is only in a row's own columns. Use every claim of the group, not only the one that names the item.
4. A record BREAKS the rule when it falls inside the rule's scope and its action is what the rule prohibits, exceeds the limit, or happens where or when the rule says it cannot. A suspension, closure or ban of a scope for a period is broken by any record of an activity of that scope dated inside the period: a row of a log, or a claim that records an item, a depth, a count or a sample from a dated entry, shows that the activity took place on that date. Compare the action with the rule, not value with value. Only what the excerpts show counts: a record that does not mention a required step or document is not a break (that is a gap in the evidence, reported elsewhere), at most insufficient_evidence. A record that complies, or lies outside the scope or the period, does not break the rule.
5. If the excerpts themselves state an exception or an approval that covers a record that would otherwise break the rule, it is a contextual_difference: name it and quote it. Never infer an exception. A record that meets a condition the rule itself sets (the consent, approval or notice the rule requires) complies: that is consistent, never a contextual_difference.
6. A claim that asserts compliance (all activities complied, nothing was breached) is contradicted by a break: include it in the finding.
7. Start attribute_reasoning with one clause for each record: whether it falls inside the rule's scope and whether its action is what the rule prohibits. Then classify: contradiction when at least one record breaks the rule; consistent when the records inside the scope comply (including by meeting a condition the rule sets); insufficient_evidence when the excerpts cannot settle whether a record is inside the scope or breaks the rule (say what is missing). When no record falls under the rule at all, same_attribute is false.
8. In a full verdict (see OUTPUT FORMAT; a verdict for a consistent group or an unrelated one is short and has no evidence), for a contradiction, give an evidence entry for the rule claim, for each claim that places the breaking record inside the scope, for each claim that shows its action, and for each compliance statement. Set aside (set_aside, by claim number) every other claim: other records and details that do not matter for the rule. Evidence entries: claim (its number), value (exactly as shown) and quote copied EXACTLY from that claim's excerpt: the shortest passage that shows the value with the words or unit next to it (at least 8 characters). For a table row, quote the claim's own row including its label.
9. Also give title (the rule and what breaks it), description (the rule, the record, the facts that place it under the rule and the action that breaks it, with the document and the place of each, and any compliance statement it contradicts), severity, confidence (0 to 1), materiality_reasoning (one sentence) and attribute_reasoning (one short sentence on why the record falls under the rule). Never do arithmetic. Return one verdict per group, using the group number shown; when two different records break the rule separately, one verdict for each, with the same group number.

SEVERITY: a contradiction is critical when a prohibition or a legal condition is broken and a statement claims compliance, high when it is broken, medium when the break is small or doubtful. consistent is informational. insufficient_evidence is low. contextual_difference is low.

OUTPUT FORMAT (strict). Return exactly one JSON object and nothing else: {"verdicts": [ ... ]}, one verdict per group.
A SHORT verdict is the only one allowed when same_attribute is false (no record falls under the rule) or the records inside the scope comply; code reads nothing else from it. Its fields: group (the integer shown), same_attribute (true or false), attribute_reasoning (at most 15 words), set_aside (claim numbers, empty when none), classification: consistent. Write no severity, confidence, title, description, materiality_reasoning, evidence or quote in it.
A FULL verdict is for every other group and has ALL of these fields: group; same_attribute: true; attribute_reasoning; set_aside; classification: exactly one of contradiction, contextual_difference, insufficient_evidence; severity: exactly one of critical, high, medium, low, informational; confidence: a number from 0 to 1; title, description, materiality_reasoning: strings (an empty string, never null); evidence: a list of {"claim": number, "value": string, "quote": string}, one per claim that remains.
Never use a value outside the lists above, and never add other fields, comments or markdown."""


# wording by which a verdict that says "contradiction" denies it in its own reasoning ("the excerpts do not establish a contradiction")
SELF_DENIAL = re.compile(
    r"(?:do|does|did)(?:\s+not|n't)\s+(?:establish|show|demonstrate|indicate|state|amount\s+to)[^.]{0,80}?(?:contradiction|conflict|competing|mutually\s+exclusive)"
    r"|\bnot\s+(?:a|an)\s+(?:real\s+|true\s+|genuine\s+)?(?:contradiction|conflict)\b"
    r"|\b(?:are|is)\s+not\s+(?:in\s+conflict|conflicting|competing|mutually\s+exclusive)\b"
    r"|\bno\s+(?:real\s+|actual\s+)?(?:contradiction|conflict)\b"
    r"|\bnot\s+(?:competing|rival|mutually\s+exclusive)\s+values\b"
    r"|\bsame\s+words\b"
    r"|\bdiffer(?:s|ence|ences)?\s+(?:only\s+)?(?:in|by)\s+(?:the\s+)?(?:order|ordering|punctuation|wording|formatting|spelling|capitali[sz]ation)\b"
    r"|\b(?:ordering|order|punctuation|wording|formatting)\b[^.]{0,30}\bdiffer\b", re.IGNORECASE)


# wording by which a reason names a difference between the values themselves (a spelling, a mismatch, "X vs Y"), as opposed to claims about different things
DESCRIBES_A_DIFFERENCE = re.compile(r"\b(?:spelling|typo|misspell\w*|mismatch\w*|discrepanc\w+|inconsisten\w+|conflict\w*|disagree\w*|differs?|differences?|vs\.?|versus)\b", re.IGNORECASE)
ABOUT_DIFFERENT_THINGS = re.compile(r"\b(?:subjects?|metrics?|attributes?|scope|periods?|units?|measures?|topics?|entit(?:y|ies)|items?|things)\b", re.IGNORECASE)


def describes_a_difference(reason: str) -> bool:
    """True when a "not comparable" reason itself names a difference between the values (a spelling difference, 'Ashbey' vs 'Ashby') and is not about the
    claims measuring different things: such a verdict contradicts its own reason."""
    return bool(DESCRIBES_A_DIFFERENCE.search(reason or "")) and not ABOUT_DIFFERENT_THINGS.search(reason or "")


def denies_contradiction(*texts: str) -> bool:
    """True when the verdict's own wording says the claims are not in conflict."""
    return any(SELF_DENIAL.search(t or "") for t in texts)


class ClaimQuote(BaseModel):
    claim: int  # the number shown for the claim
    value: str = ""  # echoed by the model; checked against the claim
    quote: str = ""


class GroupVerdict(BaseModel):
    group: int
    same_attribute: bool = True  # false = the claims measure different things and are not compared
    attribute_reasoning: str = ""
    set_aside: list[int] = Field(default_factory=list)  # claims about something else, left out of the comparison
    classification: Classification
    severity: Severity = Severity.informational
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    title: str = ""
    description: str = ""
    materiality_reasoning: str | None = None
    evidence: list[ClaimQuote] = Field(default_factory=list)


class GroupVerdicts(BaseModel):
    verdicts: list[GroupVerdict] = Field(default_factory=list)


@dataclass
class Group:
    """A fact and the claims that may state it."""
    claim_ids: list[str]
    edges: list[Candidate]  # the candidates between its claims
    folded: int = 0  # claims left out of the call because another claim states the same value (only when the group was too large for one call)
    agreed: bool = False  # code found every claim to state the same value: not sent to the LLM
    kind: str = "fact"  # "fact": claims that may state one fact; "rule": a rule (the first claim) and the records that may fall under it; "record": claims that name one record


@dataclass
class CandidateRecord:
    """What happened to one candidate: both claims, how it was found, and the outcome (for the run report)."""
    claim: str
    candidate: str
    hint: str
    route: str
    outcome: str = ""


@dataclass
class SavedFinding:
    finding_id: str
    classification: str
    severity: str
    confidence: float
    claim_ids: tuple[str, ...]
    title: str



@dataclass
class JudgeStats:
    candidates_total: int = 0
    groups_total: int = 0
    groups_agreed: int = 0  # every claim states the same value: corroboration, not sent to the LLM
    clusters_split: int = 0  # sets of claims divided into smaller clusters by how alike the claims are (too large, or a grouping cluster of claims that are not alike)
    links_dropped: int = 0  # links that are in no cluster: they lie between two clusters, or join a claim that cannot be added to one
    groups_basis: int = 0  # two claims that differ in basis (planned and actual, original and final): a contextual difference decided in code
    currency_reclassified: int = 0  # contradictions between amounts in different currencies, saved as insufficient evidence
    base_missing: int = 0  # a percentage set against an absolute figure with no stated base: saved as missing information
    conflicts: list = field(default_factory=list)  # verdicts on the same pair of values that disagreed with a finding already saved
    value_seen: dict = field(default_factory=dict)  # (chunk, value) pairs of the findings saved -> (classification, title)
    groups_series: int = 0  # every claim is for a different period (a yearly series): not compared, not sent to the LLM
    groups_split: int = 0  # groups that held claims about different attributes (by type, unit, label or metadata), split before judging
    groups_unjudged: int = 0  # groups that got no verdict (a call failed or the run stopped)
    groups_labels: int = 0  # every claim only names its own document (title, type, revision): not compared
    groups_superseded: int = 0  # claims of a document and of the document it says it supersedes: a contextual difference decided in code
    groups_currency: int = 0  # amounts in different currencies: insufficient evidence decided in code
    merged: int = 0  # findings about one difference across documents folded into one
    groups_judged: int = 0
    llm_calls: int = 0
    batches_failed: int = 0
    verdict_counts: Counter = field(default_factory=Counter)
    findings_saved: int = 0
    findings_rejected: int = 0  # reported verdicts whose quotes failed verification (or were missing)
    rejections: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    corroborated_claims: set[str] = field(default_factory=set)  # claim ids in a group whose claims all agree
    grouped_claims: set[str] = field(default_factory=set)  # claim ids that are in any group
    findings_by_document: Counter = field(default_factory=Counter)
    finding_titles_by_document: dict[str, list[str]] = field(default_factory=dict)
    agreement_findings: int = 0  # `consistent` findings made from claims that state the same value in two or more documents
    duplicates_skipped: int = 0  # verdicts that restate a finding already saved (same quotes in the same chunks)
    seen: set = field(default_factory=set)
    records: dict[tuple[str, str], CandidateRecord] = field(default_factory=dict)  # one per candidate, in grouping order
    groups: list = field(default_factory=list)  # every group judged or agreed, for the run report
    misaligned: int = 0  # verdicts whose echoed values did not match their claims (the group is asked again, alone)
    escalated: int = 0  # groups the screening model called findings, decided by the bigger model (two-step judging)
    strong_calls: int = 0
    strong_failed: int = 0  # bigger-model calls that failed (the screen's verdict was kept for their groups)
    strong_confirmed: int = 0  # the bigger model also found a difference
    strong_overridden: int = 0  # the bigger model found the claims consistent or not comparable: nothing is saved for them
    strong_kept_screen: int = 0  # groups whose verdict is the screen's: the bigger model gave none that could be used, or the run was stopped
    reason_contradicts_verdict: int = 0  # "not comparable" verdicts whose own reason names a difference between the values: asked again once
    echo_missing: int = 0  # verdicts that did not echo any value
    retries: int = 0  # single-group calls for groups with no usable verdict
    quotes_from_code: int = 0  # consistent / insufficient-evidence verdicts saved with the claims' own sentences because the model gave no quotes
    quotes_replaced: int = 0  # quotes the model gave that were not word for word in the claim's chunk: the claim's own passage was stored instead
    saved: list = field(default_factory=list)  # every finding saved from a verdict, with its claims
    reclassified_names: int = 0  # contradictions between names where one contains the other, saved as consistent
    withdrawn: int = 0  # contradictions whose own wording says they are not one: dropped
    withdrawn_log: list = field(default_factory=list)
    consistent_not_saved: int = 0  # consistent verdicts and agreements not saved as findings (when consistent findings are off)
    grouped: int = 0  # consistent findings folded into one finding per set of documents and attribute
    claims_dropped: int = 0  # claims left out of a finding because their quote did not show their value
    deleted: set = field(default_factory=set)  # ids of findings removed by the steps above
    merge_log: list = field(default_factory=list)
    strong_log: list = field(default_factory=list)  # one entry for each group the bigger model was asked about: what the screen said, what the bigger model said, and what was done
    echo_log: list = field(default_factory=list)  # what the model repeated against what the claim holds, for each mismatch

    def note_edges(self, edges: list[Candidate], outcome: str) -> None:
        for e in edges:
            record = self.records.get((e.claim_id, e.candidate_id))
            if record is not None:
                record.outcome = outcome

    def append_edges(self, edges: list[Candidate], suffix: str) -> None:
        for e in edges:
            record = self.records.get((e.claim_id, e.candidate_id))
            if record is not None:
                record.outcome += suffix


def _location(chunk: DocumentChunk) -> str:
    if chunk.sheet_name:
        return f"sheet {chunk.sheet_name}, {chunk.cell_reference}"
    return ", ".join(p for p in (f"page {chunk.page_number}" if chunk.page_number else "",
                                 f"section {chunk.heading}" if chunk.heading else "") if p) or chunk.content_type


def _shown(claim: Claim) -> str:
    """The claim's value with its unit, unless the value already carries it ("12,000 WSPM" with unit WSPM)."""
    value, unit = str(claim.value), (claim.unit or "").strip()
    return value if not unit or unit.lower() in value.lower() else f"{value} {unit}"


def _claim_line(c: Claim) -> str:
    tags = ", ".join(f"{k}: {v}" for k, v in tags_of(c.assumptions).items())
    attrs = {k: v for k, v in (("subject", c.subject), ("metric", c.metric), ("value", c.value), ("unit", c.unit),
                               ("period", c.period), ("conditions", c.conditions), ("extraction notes", tags)) if v not in (None, "")}
    return f"{c.claim_text} | " + "; ".join(f"{k}={v}" for k, v in attrs.items())


def quote_for(claim: Claim, chunk: DocumentChunk) -> str | None:
    """An exact passage of the chunk that states the claim: its own sentence, else the line, else the words around its value."""
    text = chunk.content
    subject = " ".join((claim.subject or "").split()).casefold()
    if chunk.content_type in TABLE_TYPES and subject:
        # a table chunk holds several rows: quote the row labelled with the claim's subject, preferring the line that also holds its value
        rows = [line.strip() for line in text.splitlines() if len(line.strip()) >= 8 and subject in " ".join(line.split()).casefold()]
        value_text = str(claim.value or "").strip()
        own = [line for line in rows if value_text and value_text in line] or rows
        if own:
            return own[0]
    if claim.claim_text:
        span = find_source_span(text, claim.claim_text)
        if span and len(span[0]) >= 8:
            return span[0]
    value = str(claim.value or "").strip()
    if value:
        for line in text.splitlines():
            if value in line and len(line.strip()) >= 8:
                return line.strip()
        i = text.find(value)
        if i >= 0:
            around = text[max(0, i - 40): i + len(value) + 40].strip()
            return around if len(around) >= 8 else None
    return None


_NUMBER = re.compile(r"\d+(?:[.,]\d+)*")


def point_inside_range(a: object, b: object) -> bool:
    """One value is a range written 'low-high' and the other is a single number or an approximate one that lies inside it (about 1810 inside 1885-1740):
    a central estimate and its interval agree. A value with a plus-or-minus is not a range of this kind, and both must end in the same words (the unit)."""
    ta, tb = str(a), str(b)
    for rng_text, other_text in ((ta, tb), (tb, ta)):
        if "±" in rng_text or "+/-" in rng_text or not re.match(r"\D*\d+\s*[-–]\s*\d+", rng_text.strip()):
            continue
        low_high, other_range = parse_range(rng_text), parse_range(other_text)
        numbers = [Decimal(x.replace(",", "")) for x in _NUMBER.findall(other_text)]
        words = lambda t: set(re.findall(r"[a-z]{2,}", t.lower())) - {"approximately", "about", "around", "circa", "roughly", "approx"}
        if low_high is None or other_range is not None or len(numbers) != 1 or not words(rng_text) & words(other_text):
            continue
        if low_high[0] <= numbers[0] <= low_high[1]:
            return True
    return False


def _components(edges: list[tuple[str, str]]):
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for x, y in edges:
        parent[find(x)] = find(y)
    return find


class ClaimGroupJudge:
    def __init__(self, session: Session, llm: LLMClient, registry: ToolRegistry, max_chars_per_call: int = 12_000,
                 on_progress: ProgressFn | None = None, should_stop: Callable[[], bool] | None = None, workers: int = 1,
                 profiles: dict | None = None, save_consistent: bool = True, retry_pause_seconds: float = 0.0, sleep: Callable[[float], None] = time.sleep,
                 strong_llm: LLMClient | None = None, embedder=None):
        self.retry_pause = retry_pause_seconds  # the wait before groups are asked again after a failed call: a busy service needs more than the call's own short retries
        self._sleep = sleep
        self.profiles = profiles or {}  # document id -> (file name, profile): shown to the judge as document cards
        self.save_consistent = save_consistent  # False: agreements are counted but not saved as findings
        self.session = session
        self.llm = llm  # the model that judges every group (the screen, when ``strong_llm`` is given)
        self.strong_llm = strong_llm  # two-step judging: this model decides the groups the screen calls findings
        self.embedder = embedder  # divides a set of claims that is too large (or whose claims are not alike) by meaning; without it such a set falls back to groups of two
        self._escalate: list[tuple[Group, dict[int, str], list[GroupVerdict]]] = []
        self.workers = max(1, workers)  # model calls that run at the same time (1 = one after another)
        self.registry = registry
        self.max_chars = max_chars_per_call
        self.on_progress = on_progress or (lambda _m, _d: None)
        self.should_stop = should_stop or (lambda: False)

    # -- the run ------------------------------------------------------------------------------------------------------------------------
    def run(self, candidates: list[Candidate]) -> JudgeStats:
        stats = JudgeStats(candidates_total=len(candidates))
        claim_ids = {c.claim_id for c in candidates} | {c.candidate_id for c in candidates}
        claims = {c.id: c for c in self.session.scalars(select(Claim).where(Claim.id.in_(claim_ids)))} if claim_ids else {}
        chunk_ids = {c.chunk_id for c in claims.values()}
        chunks = {c.id: c for c in self.session.scalars(select(DocumentChunk).where(DocumentChunk.id.in_(chunk_ids)))} if chunk_ids else {}
        names = {d.id: d.filename for d in self.session.scalars(select(Document))}
        self._claims, self._chunks, self._names = claims, chunks, names
        counts: dict[str, Counter] = defaultdict(Counter)  # the currency most of a document's amounts use, for an amount that names none
        for c in claims.values():
            for m in CURRENCY.finditer(f"{c.value} {c.claim_text or ''}"):
                counts[c.document_id][next(x for x in m.groups() if x).upper().replace("£", "GBP").replace("€", "EUR").replace("$", "USD")] += 1
        self._doc_currency = {doc: c.most_common(1)[0][0] for doc, c in counts.items()}
        self._supersessions = superseded_documents(self.session)
        self._headings = {}  # document id -> the words that name the document: the parts of a header line "DOC-03 | Progress Report No. 3 | Revision A", else its heading and file name
        for doc_id, content, heading, filename in self.session.execute(
                select(DocumentChunk.document_id, DocumentChunk.content, DocumentChunk.heading, Document.filename).join(Document, Document.id == DocumentChunk.document_id)
                .where(DocumentChunk.chunk_index == 0)):
            first = next((" ".join(line.split()) for line in (content or "").splitlines() if line.strip()), "")
            header = re.split(r"(?<=[.!?])\s+(?=[A-Z])", first, maxsplit=1)[0]
            self._headings[doc_id] = tokens(header) if " | " in header else tokens(f"{heading or ''} {re.sub(r'[.][A-Za-z0-9]+$', '', filename)}")

        for c in candidates:
            stats.records[(c.claim_id, c.candidate_id)] = CandidateRecord(
                self._describe(claims.get(c.claim_id), names), self._describe(claims.get(c.candidate_id), names), c.relation_hint, c.matched_by)
        # a claim the code found is added to a cluster only when it can state the cluster's attribute AND its subject overlaps the one it links to: a shared word of the label ("maximum") is not enough
        attachable = lambda a, b, how: self._same_family(claims[a], claims[b], how, True) and (how == "identifier" or subjects_overlap(claims[a].subject, claims[b].subject))
        similarity = ClaimSimilarity(self.embedder, claims) if self.embedder is not None else None
        tally = {"split": 0, "dropped": 0}
        groups = [h for g in self._form_groups([c for c in candidates if c.claim_id in claims and c.candidate_id in claims], attachable, similarity=similarity, tally=tally)
                  for h in self._split_families(g, stats)]
        stats.clusters_split = tally["split"]
        stats.groups_total = len(groups)
        stats.groups = groups
        in_a_group = {(e.claim_id, e.candidate_id) for g in groups for e in g.edges}
        for c in candidates:
            record = stats.records.get((c.claim_id, c.candidate_id))
            if record is not None and (c.claim_id, c.candidate_id) not in in_a_group and not record.outcome:
                record.outcome = "in no cluster: the link lies between two clusters, or joins a claim that cannot be added to one"
                stats.links_dropped += 1
        to_judge: list[Group] = []
        for g in groups:
            stats.grouped_claims.update(g.claim_ids)
            if g.kind == "rule":
                to_judge.append(g)
            elif self._metadata_only(g):
                stats.groups_labels += 1
                stats.note_edges(g.edges, "metadata: every claim is about its own document (title, type, revision, its date), so none is a rival value of another")
            elif self._is_series(g):
                stats.groups_series += 1
                stats.note_edges(g.edges, "series: every claim is for a different period, so none is a rival value of another")
            elif self._all_agree(g):
                g.agreed = True
                stats.groups_agreed += 1
                stats.corroborated_claims.update(g.claim_ids)
                self._save_agreement(g, stats)
            elif self._decided_by_rule(g, stats):
                continue
            else:
                to_judge.append(g)
        stats.groups_judged = len(to_judge)

        unresolved = self._judge_round(to_judge, stats)
        if unresolved and not self.should_stop():
            self._retry(unresolved, stats, call_failed=stats.batches_failed > 0)
        if self._escalate:
            self._strong_pass(stats)
        self._group_consistent(stats, claims, names)
        self._merge_across_documents(stats, claims)
        for g in groups:
            lost = False
            for e in g.edges:
                record = stats.records[(e.claim_id, e.candidate_id)]
                if not record.outcome:
                    record.outcome = "no verdict returned (batch failed, run stopped, or the model skipped the group)"
                    lost = True
            stats.groups_unjudged += lost
        return stats

    def _judge_round(self, groups: list[Group], stats: JudgeStats) -> list[Group]:
        """Ask for a verdict for every group, in batches, and act on the verdicts. Returns the groups with no usable verdict (asked again by ``_retry``).
        A verdict that is a finding is held for the bigger model when two-step judging is on."""
        unresolved: list[Group] = []
        batches = list(self._batches(groups))
        answered = groups_answered = 0
        self.on_progress(f"Judging {len(groups)} groups in {len(batches)} batches", {"batches": len(batches), "total": len(groups)})
        for batch, numbering, result in self._ask(batches):
            if result is None:
                continue
            answered += 1
            groups_answered += len(batch)
            stats.llm_calls += 1
            if isinstance(result, LLMError):
                stats.batches_failed += 1
                stats.errors.append(str(result))
                logger.warning("Group judging batch failed", extra={"error": str(result)})
                unresolved.extend(batch)
                continue
            by_group: dict[int, list[GroupVerdict]] = defaultdict(list)
            for v in result.verdicts:
                by_group[v.group].append(v)
            for number, g in enumerate(batch, start=1):
                verdicts = by_group.get(number)
                if self.strong_llm is not None and verdicts and self._is_candidate(verdicts):
                    self._escalate.append((g, numbering[number - 1], verdicts))  # a candidate finding: the bigger model decides it
                    continue
                # a group that holds two separate issues has a verdict for each; if any of them cannot be used, the group is asked again
                if not verdicts or not all([self._handle(v, g, numbering[number - 1], stats, retrying=False) for v in verdicts]):
                    unresolved.append(g)
            judged = groups_answered - len(unresolved)  # groups with a usable verdict so far; failed ones are asked again afterwards
            self.on_progress(f"Judged batch {answered} of {len(batches)}: {judged} of {len(groups)} groups",
                             {"batch": answered, "batches": len(batches), "judged": judged, "total": len(groups)})
        if answered < len(batches):
            stats.errors.append("stop requested")
        return unresolved

    @staticmethod
    def _is_candidate(verdicts: list[GroupVerdict]) -> bool:
        """The screen found a difference in the group: the claims state one attribute and are not called consistent."""
        return any(v.same_attribute and v.classification != Classification.consistent for v in verdicts)

    def _strong_pass(self, stats: JudgeStats) -> None:
        """Two-step judging: the groups the screen called findings go to the bigger model, with the same prompt, and ITS verdict is the one acted on. When the
        bigger model fails, gives nothing usable for a group, or the run is stopped, the screen's own verdict for the group is used: no finding is lost to an
        unavailable model."""
        items, self._escalate = self._escalate, []
        stats.escalated = len(items)
        settled: set[int] = set()
        said: dict[int, list[GroupVerdict]] = {}  # the bigger model's verdicts for each group, for the report
        failure = ""
        groups = [g for g, _n, _v in items]
        if not self.should_stop():
            batches = list(self._batches(groups))
            self.on_progress(f"Bigger model: {len(groups)} candidate groups in {len(batches)} calls", {"strong_batches": len(batches), "strong_groups": len(groups)})
            for batch, numbering, result in self._ask(batches, strong=True):
                stats.strong_calls += 1
                if isinstance(result, LLMError):
                    stats.strong_failed += 1
                    stats.errors.append(f"bigger model: {result}")
                    failure = str(result)[:120]
                    continue
                by_group: dict[int, list[GroupVerdict]] = defaultdict(list)
                for v in result.verdicts:
                    by_group[v.group].append(v)
                for number, g in enumerate(batch, start=1):
                    verdicts = by_group.get(number)
                    if verdicts:
                        said[id(g)] = verdicts
                    if verdicts and all([self._handle(v, g, numbering[number - 1], stats, retrying=False) for v in verdicts]):
                        settled.add(id(g))
                        if self._is_candidate(verdicts):
                            stats.strong_confirmed += 1
                        else:
                            stats.strong_overridden += 1
        def summary(vs: list[GroupVerdict]) -> str:
            return " | ".join(f"{('not comparable' if not v.same_attribute else v.classification.value)}"
                              + (f"/{v.severity.value}" if v.same_attribute and v.classification != Classification.consistent else "")
                              + f": {(v.title or v.attribute_reasoning or '').strip()[:160]}"
                              + (f" [{v.description.strip()[:260]}]" if v.same_attribute and v.classification != Classification.consistent and v.description else "") for v in vs)

        for g, numbers, verdicts in items:
            if id(g) in settled:
                outcome = "confirmed" if self._is_candidate(said[id(g)]) else "overridden: nothing saved"
            else:
                stats.strong_kept_screen += 1
                outcome = "screen's verdict kept: " + ("the bigger model had no usable verdict for it" if id(g) in said else (failure or "the bigger model was not asked (run stopped)"))
                for v in verdicts:
                    self._handle(v, g, numbers, stats, retrying=True)
            stats.strong_log.append({"claims": [f"{(self._names.get(self._claims[i].document_id) or '?')[:6]} | {self._claims[i].subject or '-'} | {self._claims[i].metric or '-'} | {self._claims[i].value}" + (f" {self._claims[i].unit}" if self._claims[i].unit else "") for i in g.claim_ids][:8], "more": max(0, len(g.claim_ids) - 8),
                                     "screen": summary(verdicts), "bigger": summary(said[id(g)]) if id(g) in said else "(none)", "outcome": outcome})

    def _retry(self, unresolved: list[Group], stats: JudgeStats, call_failed: bool) -> None:
        """Ask again for every group that has no verdict, as many groups to a call as fit (the same batches as the first time), until all are judged.
        A call that fails again because the service is busy is asked again after a pause, as its two halves (smaller calls get through more often),
        down to a single group. A group the model answered but left out is asked again in smaller calls, each round, until it is judged or the rounds run out. A permanent error (a bad key, a bad request) is final:
        asking again would fail the same way. At most MAX_RETRY_ROUNDS rounds; what is still unjudged then is counted and reported."""
        queue = list(self._batches(unresolved))
        for round_number in range(1, MAX_RETRY_ROUNDS + 1):
            if not queue or self.should_stop():
                break
            if call_failed:
                self._pause()
                if self.should_stop():
                    break
            self.on_progress(f"Asking again for {sum(len(b) for b in queue)} groups in {len(queue)} calls (round {round_number})",
                             {"retry_round": round_number, "groups": sum(len(b) for b in queue)})
            next_queue: list[list[Group]] = []
            call_failed = False
            for batch, numbering, result in self._ask(queue):
                if result is None:
                    continue
                stats.llm_calls += 1
                stats.retries += 1
                if isinstance(result, LLMError):
                    stats.errors.append(str(result))
                    if isinstance(result, (LLMTransientError, LLMInvalidOutputError)):
                        call_failed = True
                        half = len(batch) // 2
                        next_queue += [batch[:half], batch[half:]] if half else [batch]
                    continue
                by_group: dict[int, list[GroupVerdict]] = defaultdict(list)
                for v in result.verdicts:
                    by_group[v.group].append(v)
                left_out: list[Group] = []
                for number, g in enumerate(batch, start=1):
                    if number not in by_group:
                        left_out.append(g)  # the model answered but left this group out: it is asked again, in smaller calls
                        continue
                    if self.strong_llm is not None and self._is_candidate(by_group[number]):
                        self._escalate.append((g, numbering[number - 1], by_group[number]))
                        continue
                    for verdict in by_group[number]:
                        self._handle(verdict, g, numbering[number - 1], stats, retrying=True)
                if left_out:
                    half = len(left_out) // 2
                    next_queue += [left_out[:half], left_out[half:]] if half else [left_out]
            queue = next_queue

    def _pause(self) -> None:
        waited = 0.0
        while waited < self.retry_pause and not self.should_stop():
            self._sleep(min(1.0, self.retry_pause - waited))
            waited += 1.0

    # -- groups -------------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def _form_groups(candidates: list[Candidate], attachable: Callable[[str, str, str], bool] | None = None, cap: int = MAX_GROUP_CLAIMS,
                     similarity: ClaimSimilarity | None = None, tally: dict | None = None) -> list[Group]:
        """The groups to judge: clusters of claims that state one attribute, never more than ``cap`` claims. The grouping model's clusters stay whole. A link the code
        found (shared words, a rare label word, nearest neighbours, a record id) attaches a claim to the cluster of the claim it links to when
        ``attachable(a, b, how)`` says the two can state one attribute. Claims linked only to each other form a cluster of their own. A set larger than ``cap``, and a
        cluster whose claims are not alike, is divided into clusters by meaning (``similarity``) or, without it, by how its claims are linked. A link that lies
        between two clusters, or joins a claim that cannot be added to a cluster, is in no group: only clusters are judged. Without ``attachable`` every link is allowed."""
        ok = attachable or (lambda a, b, how: True)
        rules = [c for c in candidates if c.matched_by == "rule"]
        model = [c for c in candidates if c.matched_by == "llm"]
        code = [c for c in candidates if c.matched_by not in ("rule", "llm")]
        groups: list[Group] = []

        def scattered(ids: list[str]) -> bool:
            return similarity is not None and 3 <= len(ids) <= cap and similarity.coherence(ids) < COHERENCE_FLOOR

        # 1. the grouping model's clusters: the claims it put together are one cluster
        find = _components([(c.claim_id, c.candidate_id) for c in model])
        clusters: dict[str, Group] = {}
        for c in model:
            g = clusters.setdefault(find(c.claim_id), Group([], []))
            for claim_id in (c.claim_id, c.candidate_id):
                if claim_id not in g.claim_ids:
                    g.claim_ids.append(claim_id)
            g.edges.append(c)
        cluster_of: dict[str, str] = {}
        for root, g in list(clusters.items()):
            if len(g.claim_ids) > cap or scattered(g.claim_ids):  # not one fact: divided into clusters by meaning
                del clusters[root]
                groups += ClaimGroupJudge._divide(g.claim_ids, g.edges, ok, cap, similarity, tally)
            else:
                for claim_id in g.claim_ids:
                    cluster_of[claim_id] = root
        # 2. what the code found: attach a claim to the cluster of the claim it links to
        pending = list(code)
        changed = True
        while changed:  # a claim that has just joined a cluster can attach others
            changed = False
            rest: list[Candidate] = []
            for c in pending:
                a, b = c.claim_id, c.candidate_id
                ca, cb = cluster_of.get(a), cluster_of.get(b)
                if ca and ca == cb:
                    clusters[ca].edges.append(c)  # both are in one cluster already: it is compared there
                elif ca and cb:
                    continue  # between two clusters: they are not one fact
                elif ca or cb:
                    root, new = (ca, b) if ca else (cb, a)
                    if len(clusters[root].claim_ids) < cap and ok(a, b, c.matched_by):
                        clusters[root].claim_ids.append(new)
                        clusters[root].edges.append(c)
                        cluster_of[new] = root
                        changed = True
                else:
                    rest.append(c)
            pending = rest
        groups += list(clusters.values())
        # 3. claims linked only to each other: each connected set is a cluster of its own (divided when it is too large or its claims are not alike)
        find = _components([(c.claim_id, c.candidate_id) for c in pending])
        sets: dict[str, Group] = {}
        for c in pending:
            g = sets.setdefault(find(c.claim_id), Group([], []))
            for claim_id in (c.claim_id, c.candidate_id):
                if claim_id not in g.claim_ids:
                    g.claim_ids.append(claim_id)
            g.edges.append(c)
        for g in sets.values():
            if len(g.claim_ids) > cap or scattered(g.claim_ids):
                groups += ClaimGroupJudge._divide(g.claim_ids, g.edges, ok, cap, similarity, tally)
            else:
                g.kind = "record" if all(c.matched_by == "identifier" for c in g.edges) else "fact"
                groups.append(g)
        by_rule: dict[str, list[Candidate]] = {}
        for c in rules:  # a rule and its records are a cluster of their own: two rules sharing a record are two clusters, not one
            by_rule.setdefault(c.claim_id, []).append(c)
        for rule_id, links in by_rule.items():
            groups.append(Group([rule_id] + [c.candidate_id for c in links if c.candidate_id != rule_id], links, kind="rule"))
        return groups

    @staticmethod
    def _pack(claim_ids: list[str], edges: list[Candidate], cap: int) -> list[list[str]]:
        """Without a similarity measure: ``claim_ids`` cut into clusters of at most ``cap`` claims that follow the links, each grown from the claim with the most
        links by taking its linked claims first."""
        from collections import deque
        ids = set(claim_ids)
        linked: dict[str, set[str]] = defaultdict(set)
        for c in edges:
            if c.claim_id in ids and c.candidate_id in ids:
                linked[c.claim_id].add(c.candidate_id)
                linked[c.candidate_id].add(c.claim_id)
        free = set(claim_ids)
        out: list[list[str]] = []
        while free:
            seed = max(sorted(free), key=lambda i: len(linked[i] & free))
            cluster, queue = [seed], deque([seed])
            free.discard(seed)
            while queue and len(cluster) < cap:
                for nb in sorted(linked[queue.popleft()] & free):
                    if len(cluster) < cap:
                        free.discard(nb)
                        cluster.append(nb)
                        queue.append(nb)
            out.append(cluster)
        return out

    @staticmethod
    def _divide(claim_ids: list[str], edges: list[Candidate], ok: Callable[[str, str, str], bool], cap: int,
                similarity: ClaimSimilarity | None = None, tally: dict | None = None) -> list[Group]:
        """A set of claims that is too large to be one fact, or whose claims are not alike, divided into clusters of at most ``cap`` claims: the claims joined by
        links that can state one attribute stay together; a joined set that is still too large or not alike is divided by meaning (``similarity``) or, without it,
        by how its claims are linked. A cluster keeps the links between its claims; the links between the clusters are dropped."""
        if tally is not None:
            tally["split"] += 1
        find = _components([(c.claim_id, c.candidate_id) for c in edges if ok(c.claim_id, c.candidate_id, c.matched_by)])
        sets: dict[str, list[str]] = defaultdict(list)
        for claim_id in claim_ids:
            sets[find(claim_id)].append(claim_id)
        out: list[Group] = []
        for ids in sets.values():
            if len(ids) < 2:
                continue
            if len(ids) <= cap and not (similarity is not None and len(ids) >= 3 and similarity.coherence(ids) < COHERENCE_FLOOR):
                parts = [ids]
            else:
                parts = similarity.split(ids, cap) if similarity is not None else ClaimGroupJudge._pack(ids, edges, cap)
            for part in parts:
                if len(part) < 2:
                    continue
                keep = set(part)
                held = [c for c in edges if c.claim_id in keep and c.candidate_id in keep]
                if held:
                    out.append(Group(part, held, kind="record" if all(c.matched_by == "identifier" for c in held) else "fact"))
        return out

    def _shared_record(self, g: Group) -> str:
        """The record every claim of the group names (RC-04), as the first claim writes it; empty when they share none."""
        def labels(i: str) -> dict:
            return entity_labels(f"{self._claims[i].subject or ''} {self._claims[i].claim_text or ''}")

        shared = set.intersection(*[set(labels(i)) for i in g.claim_ids]) if g.claim_ids else set()
        return ", ".join(sorted(labels(g.claim_ids[0])[k] for k in shared))

    def _all_agree(self, g: Group) -> bool:
        """Code finds every claim of the group to state the same value about the same subject. A hedged value ("about 0.5 million") is never decided here:
        the judge reads it."""
        if any(HEDGE.search(str(self._claims[i].value)) or "precision" in tags_of(self._claims[i].assumptions) for i in g.claim_ids):
            return False
        first = self._claims[g.claim_ids[0]]
        return all(subjects_overlap(first.subject, self._claims[i].subject)
                   and (compare_values(first.value, first.unit, self._claims[i].value, self._claims[i].unit, first.claim_category,
                                       self._claims[i].claim_category, normalize_unit) == "equal" or point_inside_range(first.value, self._claims[i].value))
                   for i in g.claim_ids[1:])

    def _is_series(self, g: Group) -> bool:
        """Every claim states a period and no two states the same one: a yearly series, one data point for each period, not rival values."""
        if len(g.claim_ids) < 3:
            return False  # two claims for two periods are a change over time, which the judge decides (a series has at least three points)
        periods = [normalize_period(self._claims[i].period) for i in g.claim_ids]
        return all(periods) and len(set(periods)) == len(periods)

    # -- groups of one attribute ----------------------------------------------------------------------------------------------------------------
    def _is_metadata(self, c: Claim) -> bool:
        """The claim is about its own document, not about what the document covers: a title, a type, a revision, or the document's own date (the claim's
        subject is words of the document's heading and its label says "date", "as of", "prepared by"). Documents differ in these by nature."""
        if tags_of(c.assumptions).get("about") == "document":
            return True  # the extraction call said so
        metric = tokens(c.metric)
        if not metric:
            return False
        return metric <= DOCUMENT_LABELS or (metric <= (DOCUMENT_LABELS | DATE_META) and bool(tokens(c.subject)) and tokens(c.subject) <= self._headings.get(c.document_id, frozenset()))

    def _metadata_only(self, g: Group) -> bool:
        return g.kind == "fact" and all(self._is_metadata(self._claims[i]) for i in g.claim_ids)

    @staticmethod
    def _percentage(c: Claim) -> bool:
        return str(c.value).strip().endswith("%") or (c.unit or "").strip().startswith("%") or "percent" in (c.unit or "").lower()

    def _same_family(self, a: Claim, b: Claim, how: str, strict: bool) -> bool:
        """Two claims linked by a candidate can state one attribute: both about a document or both about its content; no units of two different
        dimensions (an amount and a mass; amounts in two currencies are one dimension). In a group of more than two (``strict``) also not a percentage
        against an absolute figure, and a metric word in common unless the grouping model itself put them together. A candidate from a record id or a rule
        is trusted on all but the first; one the grouping model made is trusted on the label; values of different kinds only for the code's own matching."""
        if self._is_metadata(a) != self._is_metadata(b):
            return False
        if how in ("identifier", "rule"):
            return True
        ua, ub = unit_info(a.unit, normalize_unit), unit_info(b.unit, normalize_unit)
        if ua.dimension and ub.dimension and ua.dimension != ub.dimension:
            return False
        if strict and self._percentage(a) != self._percentage(b):
            return False
        if how == "llm":
            return True
        ka, kb = KINDS.get(a.claim_category or "", "text"), KINDS.get(b.claim_category or "", "text")
        if not (ka == kb or {ka, kb} in ({"number", "text"}, {"number", "time"})):
            return False
        ma, mb = tokens(a.metric) - GENERIC, tokens(b.metric) - GENERIC
        return not strict or not ma or not mb or bool(ma & mb) or bool(ma & tokens(b.subject)) or bool(mb & tokens(a.subject))

    def _split_families(self, g: Group, stats: JudgeStats) -> list[Group]:
        """A group chained together by candidates may hold claims about different attributes (a contract value and a liability cap, bays installed and a
        defect limit). They are separated before judging: the claims that can state one attribute stay together, one group each; a claim left alone has no
        comparison. A group of two is only dropped when its claims cannot state one attribute (the label check is not applied: synonyms)."""
        if g.kind != "fact":
            return [g]
        strict = len(g.claim_ids) > 2
        find = _components([(e.claim_id, e.candidate_id) for e in g.edges if self._same_family(self._claims[e.claim_id], self._claims[e.candidate_id], e.matched_by, strict)])
        sets: dict[str, list[str]] = defaultdict(list)
        for cid in g.claim_ids:
            sets[find(cid)].append(cid)
        if len(sets) == 1:
            return [g]
        stats.groups_split += 1
        out = []
        for ids in sets.values():
            keep = set(ids)
            if len(ids) >= 2:
                out.append(Group(ids, [e for e in g.edges if e.claim_id in keep and e.candidate_id in keep]))
        kept_edges = {(e.claim_id, e.candidate_id) for h in out for e in h.edges}
        for e in g.edges:
            if (e.claim_id, e.candidate_id) not in kept_edges:
                stats.records[(e.claim_id, e.candidate_id)].outcome = "family: the claims cannot state one attribute (a document's own details, another kind of value, another unit or another label), not compared"
        return out

    # -- decided in code: only where code is at least as strict as the judge --------------------------------------------------------------

    def _clause(self, c: Claim) -> str:
        """The label of the claim and the clause that holds its value (between the delimiters around it): where its basis is worded."""
        chunk = self._chunks.get(c.chunk_id)
        clause = clause_quote(chunk.content, c.value, fallback=c.claim_text or "") if chunk is not None else (c.claim_text or "")
        return f"{c.metric or ''} {c.conditions or ''} {clause or ''}"

    def _basis_tag(self, c: Claim) -> str | None:
        """"later" when the claim's own clause words its figure as the actual, final or revised one, "earlier" when as planned, required or contracted."""
        basis = tags_of(c.assumptions).get("basis")
        if basis in LATER_BASES:
            return "later"
        if basis in EARLIER_BASES:
            return "earlier"
        text = self._clause(c)
        later, earlier = LATER_BASIS.search(text), EARLIER_BASIS.search(text)
        return "later" if later and not earlier else "earlier" if earlier and not later else None

    def _agree(self, claims: list[Claim]) -> bool:
        first = claims[0]
        return all(compare_values(first.value, first.unit, c.value, c.unit, first.claim_category, c.claim_category, normalize_unit) == "equal"
                   or point_inside_range(first.value, c.value) for c in claims[1:])

    @staticmethod
    def _same_label(a: Claim, b: Claim) -> bool:
        """The two claims have the same attribute label: their metrics share a word that says what the attribute is (not "date", "value", "amount")."""
        return bool((tokens(a.metric) - GENERIC) & (tokens(b.metric) - GENERIC)) and subjects_overlap(a.subject, b.subject)

    def _basis_split(self, g: Group):
        """(rest, later): the claims of the group fall into two sets that each agree inside themselves, one worded as the actual, final or revised figure and
        the other not, and the later one has the same attribute label as one of the others (so a variation is never set against a contract value, nor a date
        against a yes/no). Only numbers and dates; both bases can be true."""
        if g.kind != "fact":
            return None
        claims = [self._claims[i] for i in g.claim_ids]
        if len({c.claim_category for c in claims}) != 1 or claims[0].claim_category not in ("quantity", "date"):
            return None
        later = [c for c in claims if self._basis_tag(c) == "later"]
        rest = [c for c in claims if c not in later and self._basis_tag(c) != "later"]
        if not later or not rest or not self._agree(later) or not self._agree(rest) or self._agree(rest + later):
            return None
        return (rest, later) if any(self._same_label(l, r) for l in later for r in rest) else None

    def _supersession_split(self, g: Group):
        """(old, new, supersession): the claims come from a document and from the document it says it supersedes, each side agrees inside itself and the
        two sides differ. The later edition states the position; the earlier one stated the position that was replaced."""
        if g.kind != "fact":
            return None
        claims = [self._claims[i] for i in g.claim_ids]
        docs = {c.document_id for c in claims}
        for s in self._supersessions:
            if docs == {s.new_document, s.old_document}:
                if any(self._is_metadata(c) for c in claims):
                    return None
                new = [c for c in claims if c.document_id == s.new_document]
                old = [c for c in claims if c.document_id == s.old_document]
                if self._agree(new) and self._agree(old) and not self._agree(new + old) and any(self._attributes_related(a, b) for a in new for b in old):
                    return old, new, s
        return None

    def _currency_split(self, g: Group):
        """The amounts are in different currencies, every currency is known, and two of them are about one attribute: they cannot be compared without a rate."""
        if g.kind != "fact":
            return None
        claims = [self._claims[i] for i in g.claim_ids]
        currency = {c.id: self._currency(c) for c in claims}
        if any(c.claim_category != "quantity" for c in claims) or not all(currency.values()) or len(set(currency.values())) < 2:
            return None
        return claims if any(currency[a.id] != currency[b.id] and self._attributes_related(a, b) and subjects_overlap(a.subject, b.subject)
                             for a, b in combinations(claims, 2)) else None

    def _decided_by_rule(self, g: Group, stats: JudgeStats) -> bool:
        """A group whose outcome code decides: saved as a finding with verified quotes, never sent to the model. False when no rule applies or the claims
        cannot be quoted from different passages."""
        names = self._names
        found = self._supersession_split(g)
        if found:
            old, new, s = found
            chunk = self._chunks[s.chunk_id]
            what = new[0].metric or new[0].subject or "value"
            ok = self._save_by_rule(
                g, new + old, "contextual_difference", "low", f"Superseded: {what}",
                f"{names.get(new[0].document_id)} states {new[0].value}; {names.get(old[0].document_id)} states {old[0].value}. {names.get(new[0].document_id)} says it "
                f"supersedes {names.get(old[0].document_id)} ('{s.sentence}'), so the later edition states the current position and the difference is a change of edition, "
                f"not a conflict.", stats, extra=[(chunk.id, s.sentence, "A later edition says it supersedes the earlier one")])
            stats.groups_superseded += ok
            return ok
        found = self._basis_split(g)
        if found:
            rest, later = found
            what = rest[0].metric or rest[0].subject or "value"
            word = (LATER_BASIS.search(self._clause(later[0])) or LATER_BASIS.search("final")).group(0)
            return self._save_by_rule(
                g, rest + later, "contextual_difference", "low", f"Basis differs: {what}",
                f"{names.get(rest[0].document_id)} states {rest[0].value}; {names.get(later[0].document_id)} states {later[0].value}, worded as the later or actual figure "
                f"('{word}'). They differ in basis, so both can be true.", stats, count="groups_basis")
        found = self._currency_split(g)
        if found:
            what = found[0].metric or found[0].subject or "amount"
            currencies = sorted({self._currency(c) for c in found})
            return self._save_by_rule(
                g, found, "insufficient_evidence", "low", f"Different currencies: {what}",
                f"The amounts for {what} are stated in different currencies ({', '.join(currencies)}) and no exchange rate is stated, so they cannot be compared.",
                stats, count="groups_currency")
        return False

    def _save_by_rule(self, g: Group, claims: list[Claim], classification: str, severity: str, title: str, description: str, stats: JudgeStats,
                      extra: list | None = None, count: str | None = None) -> bool:
        """Save a finding decided in code: one quote for each claim (the clause that holds its value), the passages must differ."""
        chunks, names = self._chunks, self._names
        quoted, seen = [], set()
        for c in claims:
            q = clause_quote(chunks[c.chunk_id].content, c.value, fallback=quote_for(c, chunks[c.chunk_id]) or "")
            if q and " ".join(q.split()) not in seen:
                seen.add(" ".join(q.split()))
                quoted.append((c, q))
        role = ROLE[Classification(classification)]
        evidence = [{"chunk_id": c.chunk_id, "quoted_text": q, "role": role, "relevance_reason": c.claim_text or title,
                     "support_assessment": f"{names.get(c.document_id)} states: {_claim_line(c)}"} for c, q in quoted[:MAX_EVIDENCE]]
        if len(evidence) < 2:
            return False
        for chunk_id, text, reason in extra or []:
            evidence.append({"chunk_id": chunk_id, "quoted_text": text, "role": role, "relevance_reason": reason, "support_assessment": reason})
        result = self.registry.execute("submit_finding", {
            "title": title[:500], "classification": classification, "severity": severity, "confidence": 0.8, "evidence_status": "adequate",
            "description": description + " " + self._where([(c.id, q) for c, q in quoted]), "evidence": evidence})
        if not result.ok:
            return False
        if count:
            setattr(stats, count, getattr(stats, count) + 1)
        stats.findings_saved += 1
        stats.saved.append(SavedFinding(result.output.finding_id, classification, severity, 0.8, tuple(c.id for c, _q in quoted), title))
        stats.note_edges(g.edges, f"decided in code: {classification}, saved: {title[:90]}")
        for doc_id in {c.document_id for c, _q in quoted}:
            stats.findings_by_document[doc_id] += 1
            stats.finding_titles_by_document.setdefault(doc_id, []).append(title)
        return True

    def _currency(self, claim: Claim) -> str | None:
        """The currency a claim's amount is in: written in its value or sentence, else the one most of its document's amounts use."""
        m = CURRENCY.search(f"{claim.value} {claim.unit or ''} {claim.claim_text or ''}")
        if m:
            return next(x for x in m.groups() if x).upper().replace("£", "GBP").replace("€", "EUR").replace("$", "USD")
        return self._doc_currency.get(claim.document_id)

    def _fold(self, g: Group) -> Group:
        """A group too large for one call keeps one claim for each distinct value (the others state what it states)."""
        kept: list[str] = []
        for cid in g.claim_ids:
            c = self._claims[cid]
            if not any(compare_values(c.value, c.unit, self._claims[k].value, self._claims[k].unit, c.claim_category, self._claims[k].claim_category,
                                      normalize_unit) == "equal" for k in kept):
                kept.append(cid)
        keep = set(kept)
        return Group(kept, [e for e in g.edges if e.claim_id in keep and e.candidate_id in keep], folded=len(g.claim_ids) - len(kept))

    def _chunk_ids(self, g: Group) -> set[str]:
        return {self._claims[i].chunk_id for i in g.claim_ids}

    def _size(self, g: Group, already_in_call: frozenset[str] | set[str] = frozenset()) -> int:
        """Characters the group adds to a call. A chunk the call already shows for an earlier group is printed once, so it is not counted again."""
        chunk_ids = self._chunk_ids(g) - already_in_call
        return sum(len(self._chunks[c].content) for c in chunk_ids) + sum(len(self._claims[i].claim_text) + 150 for i in g.claim_ids) + 300

    def _batches(self, groups: list[Group]):
        """Calls sized by characters: whole groups only, as many as fit. A group too large for one call is folded to its distinct values.
        A chunk shared by several groups of one call is counted, and printed (see ``_prompt``), once."""
        current: list[Group] = []
        size = 0
        shown: set[str] = set()  # the chunks printed in the call being filled
        for g in groups:
            if self._size(g) > self.max_chars and len(g.claim_ids) > 2 and g.kind == "fact":
                g = self._fold(g)
            n = self._size(g, shown)
            if current and (size + n > self.max_chars or current[-1].kind != g.kind):  # a call holds one kind of group: they are judged by different rules
                yield current
                current, size, shown = [], 0, set()
                n = self._size(g)
            current.append(g)
            size += n
            shown |= self._chunk_ids(g)
        if current:
            yield current

    # -- the calls ----------------------------------------------------------------------------------------------------------------------
    def _ask(self, batches: list[list[Group]], strong: bool = False):
        """(batch, claim numbering per group, answer) in batch order; an LLMError instead of an answer when the call failed. With several
        workers the calls run at the same time; only the model calls do: prompts are built and answers handled one at a time, here and by the
        caller, because the database session is not shared between threads."""
        built = [self._prompt(batch) for batch in batches]
        systems = [RULE_SYSTEM if batch[0].kind == "rule" else SYSTEM for batch in batches]

        def call(prompt: str, system: str):
            if self.should_stop():  # a stop request also skips the calls that are still waiting for a worker
                return None
            try:
                return (self.strong_llm if strong else self.llm).generate_structured_output(prompt, GroupVerdicts, system=system)
            except LLMError as exc:
                return exc

        if self.workers <= 1 or len(built) <= 1:
            for batch, (prompt, numbering), system in zip(batches, built, systems):
                answer = call(prompt, system)
                if answer is None:
                    return
                yield batch, numbering, answer
            return
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            futures = [(batch, numbering, pool.submit(call, prompt, system)) for batch, (prompt, numbering), system in zip(batches, built, systems)]
            for batch, numbering, future in futures:
                yield batch, numbering, future.result()

    def _header(self, batch: list[Group]) -> str:
        """The full card of every document of the collection, the same text in every call (sorted by file name): context for the judge, never a
        source of values. It is identical from call to call, so together with the system instruction it is a prefix the provider can cache; only the
        groups that follow differ. ``batch`` is not used: the header does not depend on what the call holds."""
        if not self.profiles:
            return ""
        return ("Documents in this collection (what each is about). Use them only to judge whether claims concern the same subject; they are not a source of "
                "values or quotes.\n" + render_cards(self.profiles) + "\n\n")

    @staticmethod
    def _limit_note(a: Claim, b: Claim) -> str:
        """When one claim's wording makes its value a limit (maximum, permitted, at least ...) and the other is a plain number, say whether
        that number satisfies the limit. A value inside a limit does not conflict with it."""
        da, db = bound_direction(f"{a.metric or ''} {a.claim_text or ''}"), bound_direction(f"{b.metric or ''} {b.claim_text or ''}")
        if bool(da) == bool(db):  # neither, or both are limits: nothing to say
            return ""
        limit, other, direction = (a, b, da) if da else (b, a, db)
        ok = within_limit(limit.value, limit.unit, direction, other.value, other.unit, normalize_unit)
        if ok is None:
            return ""
        word = "maximum" if direction == "upper" else "minimum"
        return (f"The claim with the {word} is compared with a plain number that {'is within' if ok else 'does not satisfy'} it. "
                + ("A value inside a limit is compliant, not a conflict." if ok else "A value outside a limit breaks it."))

    @staticmethod
    def _note(a: Claim, b: Claim) -> str:
        note = comparison_note(a.value, a.unit, b.value, b.unit, a.claim_category, b.claim_category, normalize_unit)
        limit = ClaimGroupJudge._limit_note(a, b)
        return f"{note} {limit}" if limit else note

    def _prompt(self, batch: list[Group]) -> tuple[str, list[dict[int, str]]]:
        """The call's text and, for each group, the claim number shown -> claim id (numbers run on through the whole call)."""
        blocks, numbering, number = [], [], 0
        printed: dict[str, tuple[int, int]] = {}  # chunk id -> (claim number, group number) where its text is printed in full: once for the whole call
        reused = False
        for g_index, g in enumerate(batch, start=1):
            nums: dict[int, str] = {}
            label: dict[str, int] = {}
            if g.kind == "rule":
                lines = [f"### Group {g_index} (RULE GROUP): claim [{number + 1}] is a rule or a status; the other {len(g.claim_ids) - 1} claims are records that may fall "
                         "under it, or statements about compliance with it. Every claim and its excerpt is shown once."]
            elif g.kind == "record" and self._shared_record(g):
                lines = [f"### Group {g_index} (SAME RECORD, {self._shared_record(g)}): the {len(g.claim_ids)} claims below each name {self._shared_record(g)} and give the same kind of "
                         "value for it. They ARE about the same attribute of the same record, whatever their labels: do not set any of them aside as a different attribute. "
                         "Say consistent when the values agree, a contradiction when they do not (a contextual_difference only when an excerpt states the context that explains "
                         "it). Every claim and its excerpt is shown once."]
            else:
                lines = [f"### Group {g_index}: {len(g.claim_ids)} claims that may state one fact. Every claim and its excerpt is shown once."]
            for cid in g.claim_ids:
                number += 1
                nums[number], label[cid] = cid, number
                c = self._claims[cid]
                ch = self._chunks[c.chunk_id]
                head = f"Claim [{number}] ({self._names.get(c.document_id)}, {_location(ch)}): {_claim_line(c)}"
                if ch.id in printed:
                    first_number, first_group = printed[ch.id]
                    where = "" if first_group == g_index else f" (printed under Group {first_group})"
                    reused = reused or first_group != g_index
                    lines.append(f"{head}\nExcerpt [{number}]: the same excerpt as [{first_number}]{where}")
                else:
                    printed[ch.id] = (number, g_index)
                    lines.append(f"{head}\nExcerpt [{number}]:\n{ch.content}")
            checks = [] if g.kind == "rule" else [
                f"[{label[e.claim_id]}] and [{label[e.candidate_id]}]: {self._note(self._claims[e.claim_id], self._claims[e.candidate_id])}"
                for e in g.edges if e.claim_id in label and e.candidate_id in label][:15]
            if g.kind == "fact":
                tags = {cid: self._basis_tag(self._claims[cid]) for cid in g.claim_ids}
                earlier = [f"[{label[c]}]" for c, t in tags.items() if t == "earlier"]
                later = [f"[{label[c]}]" for c, t in tags.items() if t == "later"]
                if earlier and later:
                    checks.append(f"Basis: claims {', '.join(earlier)} are worded as planned, required or contracted figures; claims {', '.join(later)} as actual, final or "
                                  "revised ones. Figures of different bases are about the same attribute: a contextual_difference, not a contradiction.")
            if checks:
                lines.append("Code checks:\n" + "\n".join(checks))
            blocks.append("\n".join(lines))
            numbering.append(nums)
        note = ("An excerpt marked \"the same excerpt as [n]\" is the text printed in full under claim [n], which may be in an earlier group of this call.\n\n"
                if reused else "")
        return self._header(batch) + "Judge each group below.\n\n" + note + "\n\n".join(blocks), numbering

    # -- verdicts -----------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def _echo_matches(echo: str, claim: Claim, chunk: DocumentChunk | None = None) -> bool:
        """The value the model repeated belongs to the claim: it is the claim's value, contains it (the model added the words next to it), or is
        written in the claim's own sentence or excerpt (a row that holds several values, such as an age and its calibrated range, gives one of them
        as the claim's value)."""
        def squash(text) -> str:
            return " ".join(str(text or "").split()).casefold()

        if not echo.strip():
            return True
        if compare_values(echo, claim.unit, claim.value, claim.unit, None, None, normalize_unit) == "equal" or normalise_text(echo) == normalise_text(claim.value):
            return True
        e, value = squash(echo), squash(claim.value)
        if len(e) < 3:
            return False
        return (bool(value) and value in e) or e in squash(claim.claim_text) or (chunk is not None and e in squash(chunk.content))

    @staticmethod
    def _ambiguous_date(group: list[Claim]):
        """(claim, readings) when a claim writes a date as day/month/year with both numbers up to 12 and the two readings compare differently with another
        claim's date (08/07/2025 is 8 July or 7 August; one agrees with "July 2025", the other does not)."""
        for c in group:
            m = SLASH_DATE.match(str(c.value))
            if not m or int(m.group(1)) > 12 or int(m.group(2)) > 12 or m.group(1) == m.group(2):
                continue
            readings = (f"{int(m.group(1))} {MONTHS[int(m.group(2)) - 1]} {m.group(3)}", f"{int(m.group(2))} {MONTHS[int(m.group(1)) - 1]} {m.group(3)}")
            for other in group:
                if other is not c and compare_values(readings[0], None, other.value, other.unit, "date", other.claim_category, normalize_unit) != \
                        compare_values(readings[1], None, other.value, other.unit, "date", other.claim_category, normalize_unit):
                    return c, readings
        return None

    @staticmethod
    def _attributes_related(a: Claim, b: Claim) -> bool:
        """The two claims can be about one attribute: their metric labels share a word, or one's metric appears in the other's subject (the
        attribute is sometimes the row label). Unknown (a claim without a metric) counts as related."""
        ma, mb = tokens(a.metric), tokens(b.metric)
        if not ma or not mb:
            return True
        return bool(ma & mb or ma & tokens(b.subject) or mb & tokens(a.subject))

    @staticmethod
    def _distinct_values(claims: list[Claim]) -> int:
        kept: list[Claim] = []
        for c in claims:
            if not any(compare_values(c.value, c.unit, k.value, k.unit, c.claim_category, k.claim_category, normalize_unit) == "equal" for k in kept):
                kept.append(c)
        return len(kept)

    @staticmethod
    def _describe(claim: Claim | None, names: dict[str, str]) -> str:
        if claim is None:
            return "(claim not found)"
        return f"{(names.get(claim.document_id) or '?')[:2]} | {claim.subject or '-'} | {claim.metric or '-'} | {claim.value}" + (f" {claim.unit}" if claim.unit else "")

    @staticmethod
    def _row_quote(claim: Claim, chunk: DocumentChunk, quote: str) -> tuple[str, str | None]:
        """A claim read from a table row must be quoted from that row. A quote from another row of the same chunk is refused; a quote from
        the right row that leaves out the row label is widened to the whole row, so the stored evidence names the row. (quote, error)."""
        subject = " ".join((claim.subject or "").split()).casefold()
        if chunk.content_type not in TABLE_TYPES or not subject:
            return quote, None
        content = chunk.content
        if subject not in " ".join(content.split()).casefold():
            return quote, None  # the subject is not a label in this chunk, so the row cannot be told
        span = find_source_span(content, quote)
        start = content.find(span[0]) if span else -1
        if start < 0:
            return quote, None
        end, position, lines = start + len(span[0]), 0, []
        for line in content.split("\n"):
            low, high = position, position + len(line)
            position = high + 1
            if low < end and high > start:
                lines.append(line)
        holding = [line for line in lines if subject in " ".join(line.split()).casefold()]
        if not holding:
            return quote, f"the quote is from a different row than the claim about '{claim.subject}'"
        if subject in " ".join(span[0].split()).casefold():
            return quote, None
        return holding[0].strip(), None

    def _handle(self, v: GroupVerdict, g: Group, nums: dict[int, str], stats: JudgeStats, retrying: bool) -> bool:
        """Act on one group's verdict. False when the verdict cannot be used (its values are not the claims' own): the group is asked again."""
        claims, chunks, names = self._claims, self._chunks, self._names
        aside = {nums[n] for n in v.set_aside if n in nums}
        compared = [cid for cid in g.claim_ids if cid not in aside]
        mine = [e for e in g.edges if e.claim_id in compared and e.candidate_id in compared]
        left_out = [e for e in g.edges if e not in mine]
        echoed = [(nums.get(ev.claim), ev) for ev in v.evidence]
        if v.same_attribute and v.classification != Classification.consistent and not any(ev.value.strip() for _c, ev in echoed):
            stats.echo_missing += 1  # a full verdict without echoed values; a short one (not comparable, consistent) carries no evidence by design
        for cid, ev in echoed:
            if cid is not None and not self._echo_matches(ev.value, claims[cid], chunks.get(claims[cid].chunk_id)):
                stats.misaligned += 1
                stats.echo_log.append(f"{'retry' if retrying else 'first call'}: claim holds {claims[cid].value!r}; model repeated {ev.value!r}; it said: {v.attribute_reasoning[:100]}")
                stats.note_edges(g.edges, "judge: verdict misaligned with its claims (echoed values differ)" + (", discarded" if retrying else ", asked again"))
                return retrying  # a second failure is final: nothing more to ask
        if left_out:
            stats.note_edges(left_out, "judge: set aside, not about the same attribute as the other claims")
        if not v.same_attribute and describes_a_difference(v.attribute_reasoning) and not retrying:
            stats.reason_contradicts_verdict += 1  # "not comparable", but the reason says the values differ ("spelling difference, 'Ashbey' vs 'Ashby'"): asked again
            stats.note_edges(g.edges, "judge: not comparable, but its reason names a difference between the values; asked again")
            return False
        if not v.same_attribute or len(compared) < 2:  # different attributes: not a comparison at all, so no classification applies
            stats.verdict_counts["not_comparable"] += 1
            stats.note_edges(mine, "judge: not comparable. " + (v.attribute_reasoning.strip() or "(no reason given)"))
            return True
        group = [claims[c] for c in compared]
        cls = v.classification
        if cls == Classification.insufficient_evidence and g.kind != "rule" and not any(self._attributes_related(a, b) for a, b in combinations(group, 2)):
            # "not enough evidence to compare" only makes sense for one attribute; different attributes of one item are not comparable
            stats.verdict_counts["not_comparable"] += 1
            stats.note_edges(mine, "judge: insufficient_evidence, dropped as not comparable (the metrics share no word)")
            return True
        if cls == Classification.contradiction and len({cur for cur in (self._currency(c) for c in group) if cur}) > 1:
            v = v.model_copy(update={"classification": Classification.insufficient_evidence, "severity": Severity.low,
                                     "description": (v.description + " The amounts are in different currencies and no exchange rate is stated, so they cannot be compared.").strip()})
            cls = Classification.insufficient_evidence
            stats.currency_reclassified += 1
        if (g.kind == "rule" and cls == Classification.contextual_difference and CONDITION_MET.search(f"{v.description} {v.attribute_reasoning}")
                and not CONDITION_NOT_MET.search(f"{v.description} {v.attribute_reasoning}")):
            v = v.model_copy(update={"classification": Classification.consistent, "severity": Severity.informational})  # the record meets the rule's own condition: it complies
            cls = Classification.consistent
        percent = [c for c in group if str(c.value).strip().endswith("%")]
        if (cls in (Classification.consistent, Classification.contradiction, Classification.contextual_difference, Classification.insufficient_evidence) and len(group) == 2 and len(percent) == 1
                and len(re.findall(r"\d[\d,.]*", percent[0].claim_text or "")) <= 1 and parse_range(next(c.value for c in group if c is not percent[0])) is None
                and not str(next(c.value for c in group if c is not percent[0])).strip().endswith("%")):
            absolute = next(c for c in group if c is not percent[0])
            v = v.model_copy(update={"classification": Classification.missing_information, "severity": Severity.medium,
                                     "title": f"Missing information: the base of the percentage for {percent[0].metric or percent[0].subject or 'the value'}",
                                     "description": (f"{names.get(percent[0].document_id)} states {percent[0].value} and {names.get(absolute.document_id)} states {absolute.value} "
                                                     f"for the same item, but the base the percentage is taken of is not stated, so the two cannot be compared.")})
            cls = Classification.missing_information
            stats.base_missing += 1
        ambiguous = self._ambiguous_date(group) if cls in (Classification.consistent, Classification.contradiction, Classification.contextual_difference) else None
        if ambiguous is not None:
            claim, readings = ambiguous
            v = v.model_copy(update={"classification": Classification.insufficient_evidence, "severity": Severity.low, "title": f"Ambiguous date format: {claim.value}",
                                     "description": (f"{names.get(claim.document_id)} writes {claim.value}, which reads as {readings[0]} or {readings[1]}, and no document settles the order of day and "
                                                     "month. The two readings compare differently with the other claim, so the dates cannot be compared.")})
            cls = Classification.insufficient_evidence
        stats.verdict_counts[cls.value] += 1
        if (cls in (Classification.contradiction, Classification.contextual_difference) and all(c.claim_category in WORDED for c in group)
                and all(one_contains_other(group[0].value, c.value) for c in group[1:])):
            # every value is the first with more or less detail, or in another order (a role with or without its qualifier): the same thing, not a conflict
            v = v.model_copy(update={"classification": Classification.consistent, "severity": Severity.informational,
                                     "title": f"Consistent: {group[0].metric or group[0].subject or 'value'} (the values are the same words with more or less detail)"})
            cls = Classification.consistent
            stats.reclassified_names += 1
        if cls == Classification.consistent and not self.save_consistent:
            stats.consistent_not_saved += 1
            stats.note_edges(mine, "judge: consistent, counted, not saved (consistent findings are off)")
            return True
        if cls == Classification.contradiction and denies_contradiction(v.description, v.attribute_reasoning, v.materiality_reasoning or ""):
            stats.withdrawn += 1
            stats.withdrawn_log.append(f"{v.title.strip() or group[0].metric or 'group'}: {(v.description or v.attribute_reasoning)[:200]}")
            stats.note_edges(mine, "judge: contradiction withdrawn, its own reasoning says the excerpts do not establish one")
            return True
        if cls not in REPORTED:
            stats.note_edges(mine, f"judge: {cls.value} (not a reported class)")
            return True

        quotes: dict[str, str] = {}
        for cid, ev in echoed:
            if cid in compared and ev.quote.strip():
                quotes.setdefault(cid, ev.quote)
        kept: list[tuple[str, str]] = []
        dropped: list[str] = []
        for cid in compared:
            c, ch = claims[cid], chunks[claims[cid].chunk_id]
            quote = quotes.get(cid, "")
            if len(quote.strip()) < MIN_QUOTE and (cls in NEEDS_QUOTES_TO_SAVE or quote.strip()) and subjects_overlap(c.subject, group[0].subject):
                quote = quote_for(c, ch) or ""  # no quote, or one too short to keep: the claim's own sentence is used, so the class is still reported with verified evidence
                stats.quotes_from_code += bool(quote)
            if not (quote and value_supported_by_text(c.value, quote)):
                quote = next((q for q in quotes.values() if value_supported_by_text(c.value, q)), "")  # the model attached the quote to another claim
            if not quote:
                dropped.append(f"{c.value!r} ({names.get(c.document_id)}): the quote does not show the value")
                continue
            if find_source_span(ch.content, quote) is None:
                # the model's quote is not in the claim's chunk word for word (a re-typed table row, a quote taken from another claim): the claim's own
                # passage is used instead, so the finding is saved with source text. The claim's value is known to be in this chunk (the claim gate).
                own = quote_for(c, ch) or ""
                if not (own and value_supported_by_text(c.value, own)):
                    dropped.append(f"{c.value!r} ({names.get(c.document_id)}): the quote is not in the excerpt")
                    continue
                quote = own
                stats.quotes_replaced += 1
            quote, error = self._row_quote(c, ch, quote)
            if error:
                dropped.append(f"{c.value!r} ({names.get(c.document_id)}): {error}")
                continue
            kept.append((cid, quote))
        stats.claims_dropped += len(dropped)
        if len(kept) < 2 or (cls == Classification.contradiction and self._distinct_values([claims[c] for c, _q in kept]) < 2):
            stats.findings_rejected += 1
            stats.rejections.append(f"{cls.value}: the quotes do not show the claims' values ({'; '.join(dropped)[:200] or 'fewer than two claims quoted'})")
            stats.note_edges(mine, f"judge: {cls.value}, REJECTED: the quotes do not show the claims' values")
            return True
        kept = kept[:MAX_EVIDENCE]
        if cls == Classification.consistent:  # an agreement is the same finding whichever claims it came from
            key = frozenset((claims[c].chunk_id, " ".join(q.split())) for c, q in kept)
        else:  # different facts can be stated by the same passages (a start date and a volume in one sentence and one row)
            key = frozenset((c, claims[c].chunk_id, " ".join(q.split())) for c, q in kept)
        if cls != Classification.consistent:
            values = frozenset((claims[c].chunk_id, " ".join(str(claims[c].value).split()).lower()) for c, _q in kept)
            earlier = stats.value_seen.get(values)
            if earlier is not None:  # the same pair of values from the same passages was already judged
                if earlier[0] == cls.value:
                    stats.duplicates_skipped += 1
                    stats.note_edges(mine, f"judge: {cls.value}, the same values from the same passages as a finding already saved")
                else:
                    stats.conflicts.append(f"{earlier[1]} ({earlier[0]}) against a later verdict of {cls.value} on the same values")
                    stats.note_edges(mine, f"judge: {cls.value}, CONFLICTS with the finding already saved for the same values ({earlier[0]}): the first is kept")
                return True
            stats.value_seen[values] = (cls.value, v.title.strip() or cls.value)
        if key in stats.seen:
            stats.duplicates_skipped += 1
            stats.note_edges(mine, f"judge: {cls.value}, same claims and quotes as a finding already saved")
            return True
        stats.seen.add(key)
        severity = v.severity.value
        if cls == Classification.contradiction:
            severity = contradiction_severity([claims[c] for c, _q in kept], [claims[i] for i in g.claim_ids] if g.kind == "rule" else ())
        elif cls == Classification.missing_information:
            severity = missing_severity(*[claims[c].claim_text or "" for c, _q in kept], v.description)
        first = claims[kept[0][0]]
        title = v.title.strip() or f"{cls.value.replace('_', ' ')}: {first.metric or first.subject or 'claims'}"
        description = v.description.strip() or " ".join(
            f"{names.get(claims[c].document_id)} states: {claims[c].claim_text} ({_location(chunks[claims[c].chunk_id])})." for c, _q in kept)
        description += " " + self._where(kept)
        if dropped:
            description += f" Not quoted (their quotes could not be verified): {"; ".join(dropped)}."
        role = ROLE[cls]
        result = self.registry.execute("submit_finding", {
            "title": title[:500], "description": description, "classification": cls.value, "severity": severity, "confidence": v.confidence,
            "evidence_status": "adequate", "materiality_reasoning": v.materiality_reasoning,
            "evidence": [{"chunk_id": claims[c].chunk_id, "quoted_text": q, "role": role, "relevance_reason": claims[c].claim_text,
                          "support_assessment": f"{names.get(claims[c].document_id)} states: {_claim_line(claims[c])}"} for c, q in kept]})
        if not result.ok:
            stats.findings_rejected += 1
            stats.rejections.append(f"{cls.value} '{title[:60]}': {result.error.code}: {result.error.message[:160]}")
            stats.note_edges(mine, f"judge: {cls.value}, REJECTED: {result.error.code}")
            return True
        stats.findings_saved += 1
        stats.saved.append(SavedFinding(result.output.finding_id, cls.value, severity, v.confidence, tuple(c for c, _q in kept), title))
        stats.note_edges([e for e in mine if e.relation_hint != "exact"], f"judge: {cls.value} ({severity}), saved: {title[:90]}")
        stats.note_edges([e for e in mine if e.relation_hint == "exact"], f"judge: these two claims state the same value; the group's finding ({cls.value}) is about "
                                                                         f"the claims that differ: {title[:70]}")
        for doc_id in {claims[c].document_id for c, _q in kept}:
            stats.findings_by_document[doc_id] += 1
            stats.finding_titles_by_document.setdefault(doc_id, []).append(title)
        return True

    def _where(self, kept: list[tuple[str, str]]) -> str:
        """Where each quoted value is stated: the places of one document when the claims all come from it, else the document and place of each."""
        claims, chunks, names = self._claims, self._chunks, self._names
        docs = {claims[c].document_id for c, _q in kept}
        places = [f"{claims[c].value} ({_location(chunks[claims[c].chunk_id])})" for c, _q in kept]
        if len(docs) == 1:
            return f"All of these are stated in {names.get(next(iter(docs)))}, in different places: " + "; ".join(places) + "."
        return "Stated in: " + "; ".join(f"{names.get(claims[c].document_id)}, {place}" for (c, _q), place in zip(kept, places)) + "."

    def _save_agreement(self, g: Group, stats: JudgeStats) -> None:
        """Every claim of the group states the same value: a `consistent` finding with a verified quote from each claim (no LLM call)."""
        claims, chunks, names = self._claims, self._chunks, self._names
        if not self.save_consistent:
            stats.consistent_not_saved += 1
            stats.note_edges(g.edges, "agreement: counted, not saved (consistent findings are off)")
            return
        if len({claims[c].document_id for c in g.claim_ids}) < 2:
            stats.note_edges(g.edges, "agreement inside one document: counted, not saved")
            return
        quoted = [(c, quote_for(claims[c], chunks[claims[c].chunk_id])) for c in g.claim_ids]
        quoted = [(c, q) for c, q in quoted if q][:MAX_EVIDENCE]
        if len(quoted) < 2:
            stats.note_edges(g.edges, "agreement not saved: no exact quote found for the claims")
            return
        key = frozenset((claims[c].chunk_id, " ".join(q.split())) for c, q in quoted)
        if key in stats.seen:
            stats.duplicates_skipped += 1
            stats.note_edges(g.edges, "agreement: same quotes as a finding already saved")
            return
        stats.seen.add(key)
        first = claims[quoted[0][0]]
        what = first.metric or first.subject or "value"
        title = f"Consistent: {what}"
        result = self.registry.execute("submit_finding", {
            "title": title[:500], "classification": "consistent", "severity": "informational", "confidence": 0.9, "evidence_status": "adequate",
            "description": f"{len(g.claim_ids)} claims in {len({claims[c].document_id for c in g.claim_ids})} documents state the same value for {what}: {_shown(first)}.",
            "evidence": [{"chunk_id": claims[c].chunk_id, "quoted_text": q, "role": "supporting", "relevance_reason": claims[c].claim_text or what,
                          "support_assessment": f"{names.get(claims[c].document_id)} states: {_claim_line(claims[c])}"} for c, q in quoted]})
        if not result.ok:
            stats.findings_rejected += 1
            stats.rejections.append(f"consistent '{title[:60]}': {result.error.code}: {result.error.message[:160]}")
            stats.note_edges(g.edges, f"agreement rejected: {result.error.code}")
            return
        stats.findings_saved += 1
        stats.agreement_findings += 1
        stats.saved.append(SavedFinding(result.output.finding_id, "consistent", "informational", 0.9, tuple(c for c, _q in quoted), title))
        stats.note_edges(g.edges, "agreement saved as a consistent finding")
        for doc_id in {claims[c].document_id for c, _q in quoted}:
            stats.findings_by_document[doc_id] += 1
            stats.finding_titles_by_document.setdefault(doc_id, []).append(title)

    # -- after the verdicts -------------------------------------------------------------------------------------------------------------
    def _same_passages_and_figure(self, first: str, second: str) -> bool:
        """The two findings quote exactly the same passages and their descriptions state a figure in common (not a year, not a small number): they are about one
        difference, built from different claims of the same sentences. Two facts that happen to sit in the same two sentences state different figures."""
        def passages(finding_id: str) -> frozenset:
            return frozenset((e.chunk_id, " ".join(e.quoted_text.split())) for e in self.session.scalars(select(Evidence).where(Evidence.finding_id == finding_id)))

        def figures(finding_id: str) -> set:
            text = (self.session.get(Finding, finding_id).description or "")
            return {n.replace(",", "") for n in re.findall(r"\d[\d,]*(?:\.\d+)?", text) if not re.fullmatch(r"(?:19|20)\d\d", n) and (len(n.replace(",", "")) >= 3 or "." in n)}

        return passages(first) == passages(second) and bool(passages(first)) and bool(figures(first) & figures(second))

    def _merge_across_documents(self, stats: JudgeStats, claims: dict[str, Claim]) -> None:
        """Two findings of one class that share a claim, and whose other claims all state one value, are one difference stated in several documents (the same
        final value set against a contract value and against the order for it): the first keeps the evidence of both."""
        live = [s for s in stats.saved if s.classification != "consistent" and s.finding_id not in stats.deleted]
        pair = {"contradiction", "contextual_difference"}
        for a, b in combinations(live, 2):
            if a.finding_id in stats.deleted or b.finding_id in stats.deleted or (a.classification != b.classification and {a.classification, b.classification} != pair):
                continue
            shared = set(a.claim_ids) & set(b.claim_ids)
            rest_a = [claims[c] for c in a.claim_ids if c not in shared and c in claims]
            rest_b = [claims[c] for c in b.claim_ids if c not in shared and c in claims]
            one_value = bool(shared) and bool(rest_a) and bool(rest_b) and self._distinct_values(rest_a + rest_b) == 1
            if not one_value and not self._same_passages_and_figure(a.finding_id, b.finding_id):
                continue
            main = self.session.get(Finding, a.finding_id)
            if main is None:
                continue
            if a.classification != b.classification:  # the same passages cannot be both: the verdict that found a stated context had the evidence of the other
                other = self.session.get(Finding, b.finding_id)
                if other is not None and other.classification == "contextual_difference":
                    main.title, main.classification, main.severity, main.description = other.title, "contextual_difference", other.severity, other.description
            have = {(e.chunk_id, " ".join(e.quoted_text.split())) for e in self.session.scalars(select(Evidence).where(Evidence.finding_id == main.id))}
            for e in list(self.session.scalars(select(Evidence).where(Evidence.finding_id == b.finding_id))):
                key = (e.chunk_id, " ".join(e.quoted_text.split()))
                if key not in have and len(have) < MAX_EVIDENCE:
                    have.add(key)
                    self.session.add(Evidence(finding_id=main.id, chunk_id=e.chunk_id, evidence_role=e.evidence_role, quoted_text=e.quoted_text,
                                              relevance_reason=e.relevance_reason, support_assessment=e.support_assessment))
            docs = sorted({self._names.get(claims[c].document_id, "?") for c in b.claim_ids if c in claims})
            main.description = f"{main.description} The same difference is also stated in {', '.join(docs)}."[:4000]
            self._remove_finding(b.finding_id, stats)
            stats.merged += 1
            stats.merge_log.append(f"'{b.title}' and '{a.title}' are one difference stated in two places (they share a claim and the others state one value): the first keeps both")
        self.session.commit()

    def _remove_finding(self, finding_id: str, stats: JudgeStats) -> None:
        self.session.execute(delete(Evidence).where(Evidence.finding_id == finding_id))
        gone = self.session.get(Finding, finding_id)
        if gone is not None:
            self.session.delete(gone)
        stats.deleted.add(finding_id)
        stats.findings_saved -= 1

    def _group_consistent(self, stats: JudgeStats, claims: dict[str, Claim], names: dict[str, str]) -> None:
        """One consistent finding per set of documents and attribute ("Tool_Category agrees for 12 of 12 tools"), not one per row."""
        groups: dict[tuple, list[SavedFinding]] = defaultdict(list)
        for s in stats.saved:
            if s.classification != "consistent" or s.finding_id in stats.deleted:
                continue
            members = [claims[c] for c in s.claim_ids if c in claims]
            if len(members) < 2:
                continue
            label = lambda c: " ".join(sorted(tokens(c.metric))) or "-"
            groups[(frozenset(c.document_id for c in members), frozenset(label(c) for c in members))].append(s)
        for (docs, _metrics), group in groups.items():
            if len(group) < 2:
                continue
            keep, others = group[0], group[1:]
            main = self.session.get(Finding, keep.finding_id)
            if main is None:
                continue
            first = claims[keep.claim_ids[0]]
            doc_names = sorted(names.get(d, "?") for d in docs)
            have = {(e.chunk_id, " ".join(e.quoted_text.split())) for e in self.session.scalars(select(Evidence).where(Evidence.finding_id == main.id))}
            kept_rows = len(have)
            values: list[str] = []
            for s in group:
                v = str(claims[s.claim_ids[0]].value)
                if v not in values:
                    values.append(v)
            for other in others:
                for e in list(self.session.scalars(select(Evidence).where(Evidence.finding_id == other.finding_id))):
                    key = (e.chunk_id, " ".join(e.quoted_text.split()))
                    if key not in have and kept_rows < 16:  # the report stays readable: the first 16 items are shown as evidence
                        have.add(key)
                        kept_rows += 1
                        self.session.add(Evidence(finding_id=main.id, chunk_id=e.chunk_id, evidence_role=e.evidence_role, quoted_text=e.quoted_text,
                                                  relevance_reason=e.relevance_reason, support_assessment=e.support_assessment))
                self._remove_finding(other.finding_id, stats)
            what = first.metric or "value"
            main.title = f"Consistent: {what} agrees in {len(group)} places"[:500]
            main.description = (f"{doc_names[0]} and {doc_names[-1]} state the same value for {what} in {len(group)} places "
                                f"(values: {'; '.join(values[:8])}{'; ...' if len(values) > 8 else ''}).")
            stats.grouped += len(others)
            stats.merge_log.append(f"{len(group)} consistent findings between {doc_names[0]} and {doc_names[-1]} about {what} became one")
        self.session.commit()
