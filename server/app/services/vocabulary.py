"""Running vocabulary of the labels (subjects and metrics) used by claims extracted so far in a run.

It starts empty and is built only from the documents themselves, so it carries no domain
assumptions. Each extraction call is shown the labels already in use and asked to reuse one when
it names the same thing, so the same quantity gets the same label across chunks and documents.
The vocabulary is per run: a new ``ClaimService`` starts with an empty one.
"""
from __future__ import annotations

from collections import Counter

DEFAULT_MAX_CHARS = 2_000  # per list; keeps the prompt bounded however many labels exist


def _key(label: str) -> str:
    return " ".join(label.lower().split())


class RunningVocabulary:
    def __init__(self, max_chars: int = DEFAULT_MAX_CHARS):
        self.max_chars = max_chars
        self._counts: dict[str, Counter] = {"subject": Counter(), "metric": Counter()}
        self._display: dict[str, dict[str, str]] = {"subject": {}, "metric": {}}

    def add(self, kind: str, label: str | None) -> None:
        if not label or not label.strip():
            return
        key = _key(label)
        self._counts[kind][key] += 1
        self._display[kind].setdefault(key, " ".join(label.split()))  # first spelling seen is the one shown

    def labels(self, kind: str) -> list[str]:
        """Most used first, cut to ``max_chars`` in total."""
        out, size = [], 0
        for key, _ in sorted(self._counts[kind].items(), key=lambda kv: (-kv[1], kv[0])):
            shown = self._display[kind][key]
            if size + len(shown) + 2 > self.max_chars:
                break
            out.append(shown)
            size += len(shown) + 2
        return out

    def __len__(self) -> int:
        return sum(len(c) for c in self._counts.values())
