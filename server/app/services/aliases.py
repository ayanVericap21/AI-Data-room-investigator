"""Names the documents define for themselves: Northwind Logistics Ltd ("the Supplier" or "NWL").

A contract names its parties once and then calls them by their roles, so a claim about "the Supplier" and a claim about "Northwind Logistics Ltd" are about
one party. The definitions are read from the text, and a subject that uses a defined name is rewritten to the full name for matching only (never shown).
"""
from __future__ import annotations

import re
from typing import Iterable

_NAME = r"[A-Z][\w&'’.-]*(?:\s+(?:of|and|&|the|de|von)?\s*[A-Z][\w&'’.-]*){0,5}"
_QUOTED = r"[\"“”]\s*(?:the\s+)?([^\"“”]{2,40}?)\s*[\"“”]"
DEFINITION = re.compile(rf"({_NAME})\s*\(\s*(?:hereinafter\s+)?{_QUOTED}(?:\s*(?:,|or|and)\s*{_QUOTED})?(?:\s*(?:,|or|and)\s*{_QUOTED})?\s*\)")


def defined_names(texts: Iterable[str]) -> dict[str, str]:
    """lower-cased defined name -> the full name, from definitions of the form Full Name ("the Role" or "SHORT")."""
    found: dict[str, str] = {}
    for text in texts:
        for m in DEFINITION.finditer(text or ""):
            full = " ".join(m.group(1).split())
            for alias in m.groups()[1:]:
                if alias and alias.lower() not in (full.lower(), ""):
                    found.setdefault(alias.strip().lower(), full)
    return found


def resolve(text: str | None, names: dict[str, str]) -> str | None:
    """The text with every defined name written as its full name (word boundaries, any case); the text itself when it uses none."""
    if not text or not names:
        return text
    out = text
    for alias in sorted(names, key=len, reverse=True):
        out = re.sub(rf"(?<![\w]){'(?:the\\s+)?' if ' ' not in alias else ''}{re.escape(alias)}(?![\w])", names[alias], out, flags=re.IGNORECASE)
    return out
