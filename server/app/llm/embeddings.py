"""Embedding provider interface (separate from the LLM so each can change independently)."""
from __future__ import annotations

import hashlib
import logging
import math
import re
import time
from abc import ABC, abstractmethod
from typing import Callable, Literal

from app.llm.errors import LLMTransientError

logger = logging.getLogger(__name__)
Task = Literal["document", "query"]


def l2_normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vector)) or 1.0
    return [v / norm for v in vector]


class EmbeddingProvider(ABC):
    MAX_BATCH = 100

    def __init__(self, dimensions: int = 1536, max_attempts: int = 3, base_delay: float = 1.0,
                 sleep: Callable[[float], None] = time.sleep):
        self.dimensions = dimensions
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self._sleep = sleep

    @abstractmethod
    def _embed_batch(self, texts: list[str], task: Task) -> list[list[float]]: ...

    def embed(self, texts: list[str], task: Task = "document") -> list[list[float]]:
        """Embed texts in provider-sized batches with bounded retries on transient errors."""
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.MAX_BATCH):
            batch = texts[start:start + self.MAX_BATCH]
            for attempt in range(1, self.max_attempts + 1):
                try:
                    vectors.extend(self._embed_batch(batch, task))
                    break
                except LLMTransientError as exc:
                    if attempt == self.max_attempts:
                        raise
                    delay = exc.retry_after if exc.retry_after is not None else self.base_delay * (2 ** (attempt - 1))
                    logger.warning("Transient embedding error, retrying", extra={"attempt": attempt, "delay": delay})
                    self._sleep(min(delay, 60))
        return vectors


class MockEmbeddingProvider(EmbeddingProvider):
    """Deterministic hashed bag-of-words embeddings for tests: texts sharing words are close."""

    def _embed_batch(self, texts: list[str], task: Task) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    def _vector(self, text: str) -> list[float]:
        vec = [0.0] * self.dimensions
        for token in re.findall(r"[a-z0-9]+", text.lower()):
            vec[int(hashlib.md5(token.encode()).hexdigest(), 16) % self.dimensions] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]
