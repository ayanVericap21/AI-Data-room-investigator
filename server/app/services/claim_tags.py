"""What the extraction call says about a claim beyond its value: about the document or its content, basis, precision, quantifier, anchor, completeness.

Stored as entries of the claim's existing ``assumptions`` list, each written "@name: value" (no new column). Entries without the leading "@" are ordinary
assumptions and are never touched here. Matching of claims ignores the tags, so a claim's tags never make two claims look as if they differ in context.
"""
from __future__ import annotations

from typing import Iterable

PREFIX = "@"
EARLIER_BASES = frozenset({"planned", "contracted", "forecast", "estimated"})
LATER_BASES = frozenset({"actual", "revised"})


def make_tags(about: str | None = None, basis: str | None = None, precision: str | None = None, quantifier: str | None = None,
              anchor: str | None = None, complete: bool | None = None, is_rule: bool | None = None) -> list[str]:
    """The tags worth keeping: a default (content, no basis, exact, no quantifier) says nothing and is not stored."""
    clean = lambda x: (x or "").strip().lower()
    tags = []
    if clean(about) == "document":
        tags.append(f"{PREFIX}about: document")
    if clean(basis) in EARLIER_BASES | LATER_BASES | {"limit"}:
        tags.append(f"{PREFIX}basis: {clean(basis)}")
    if clean(precision) in ("approximate", "vague"):
        tags.append(f"{PREFIX}precision: {clean(precision)}")
    if clean(quantifier) in ("all", "none", "some", "most", "not_all"):
        tags.append(f"{PREFIX}quantifier: {clean(quantifier)}")
    if (anchor or "").strip():
        tags.append(f"{PREFIX}anchor: {anchor.strip()}")
    if complete:
        tags.append(f"{PREFIX}complete: list")
    if is_rule:
        tags.append(f"{PREFIX}rule: yes")  # the extraction call says the claim states a rule (see rule_links)
    return tags


def is_rule(assumptions: Iterable[str] | None) -> bool:
    """The extraction call flagged the claim as a rule: it states what is permitted, prohibited, required, limited, scheduled or suspended for a scope."""
    return tags_of(assumptions).get("rule") == "yes"


def tags_of(assumptions: Iterable[str] | None) -> dict[str, str]:
    """name -> value of the tags among a claim's assumptions."""
    out = {}
    for item in assumptions or []:
        if isinstance(item, str) and item.startswith(PREFIX) and ":" in item:
            name, _, value = item[len(PREFIX):].partition(":")
            out[name.strip()] = value.strip()
    return out


def plain(assumptions: Iterable[str] | None) -> list[str]:
    """The assumptions that are not tags."""
    return [a for a in assumptions or [] if not (isinstance(a, str) and a.startswith(PREFIX))]
