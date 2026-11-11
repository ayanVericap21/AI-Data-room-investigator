"""Local embedding provider: a small sentence-transformers model run on the CPU through fastembed (ONNX).

After a one-time model download there is no network use, no quota and no rate limit, and document
text never leaves the machine for embedding. The default model, all-MiniLM-L6-v2, maps text to 384
numbers and reads at most 256 word pieces (about 1,000 characters): longer text is truncated by the
model, which is why chunks are limited to 1,000 characters.
"""
from __future__ import annotations

from typing import Callable

from app.llm.embeddings import EmbeddingProvider, Task, l2_normalize
from app.llm.errors import LLMConfigError, LLMPermanentError

DEFAULT_LOCAL_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class LocalEmbeddingProvider(EmbeddingProvider):
    MAX_BATCH = 64

    def __init__(self, model: str = DEFAULT_LOCAL_MODEL, dimensions: int = 384,
                 model_factory: Callable[[str], object] | None = None, **kwargs):
        super().__init__(dimensions=dimensions, **kwargs)
        self.model_name = model
        self._factory = model_factory  # lets tests supply a stand-in instead of downloading the model
        self._model = None

    def _load(self):
        if self._model is None:
            if self._factory is not None:
                self._model = self._factory(self.model_name)
            else:
                try:
                    from fastembed import TextEmbedding
                except ImportError as exc:
                    raise LLMConfigError("fastembed is not installed (pip install fastembed)") from exc
                self._model = TextEmbedding(model_name=self.model_name)
        return self._model

    def _embed_batch(self, texts: list[str], task: Task) -> list[list[float]]:
        model = self._load()
        try:
            vectors = [[float(x) for x in v] for v in model.embed(texts)]
        except Exception as exc:  # a model/runtime fault is not worth retrying
            raise LLMPermanentError(f"Local embedding failed: {exc}") from exc
        if vectors and len(vectors[0]) != self.dimensions:
            raise LLMConfigError(f"Model '{self.model_name}' produces {len(vectors[0])} dimensions but "
                                 f"EMBEDDING_DIMENSIONS is {self.dimensions}; they must match the database column")
        return [l2_normalize(v) for v in vectors]
