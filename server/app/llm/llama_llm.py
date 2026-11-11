"""An LLM client whose every call goes through LlamaExtract, so a whole run can use LlamaIndex's service only.

LlamaExtract reads a document and returns data that fits a JSON schema. A call here writes the prompt to a small
text document, uploads it, and asks LlamaExtract to fill the Pydantic model's schema from it, following the system
prompt as its instructions. The reply is then validated by Pydantic like any other provider's output, and the code
checks after it (exact quotes, values in chunks) still decide what is stored. Token counts are not reported by the
service, so credits are tracked instead.
"""
from __future__ import annotations

import json
import tempfile
import threading
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel

from app.llm.base import LLMClient, RawResponse
from app.llm.errors import LLMConfigError, LLMPermanentError, LLMTransientError
from app.llm.llama_extract import LlamaExtractClient, LlamaExtractError

T = TypeVar("T", bound=BaseModel)
SCHEMA_MARKER = "\n\nRespond with ONLY a JSON object that conforms to this JSON Schema"


def inline_refs(schema: dict[str, Any]) -> dict[str, Any]:
    """Pydantic puts nested models under $defs; replace each $ref by its definition so the schema stands alone."""
    defs = schema.get("$defs", {})

    def resolve(node: Any) -> Any:
        if isinstance(node, dict):
            if "$ref" in node:
                return resolve(defs[node["$ref"].split("/")[-1]])
            return {k: resolve(v) for k, v in node.items() if k != "$defs"}
        if isinstance(node, list):
            return [resolve(v) for v in node]
        return node

    return resolve(schema)


class LlamaExtractLLMClient(LLMClient):
    def __init__(self, client: LlamaExtractClient, **kwargs: Any):
        super().__init__(**kwargs)
        self._llama = client
        self.credits = 0.0  # LlamaExtract credits used by this client
        self._local = threading.local()  # the schema of the call this thread is making

    def generate_structured_output(self, prompt: str, schema: type[T], system: str | None = None) -> T:
        self._local.schema = inline_refs(schema.model_json_schema())
        try:
            return super().generate_structured_output(prompt, schema, system)
        finally:
            self._local.schema = None

    def _raw_generate(self, prompt: str, system: str | None, json_mode: bool) -> RawResponse:
        schema = getattr(self._local, "schema", None)
        if schema is None:
            raise LLMConfigError("LlamaExtract can only produce structured output (use generate_structured_output)")
        body = prompt.split(SCHEMA_MARKER)[0]  # the schema is sent as the extraction schema, not repeated in the text
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "input.txt"
            path.write_text(body, encoding="utf-8")
            try:
                job = self._llama.extract(path, schema, system_prompt=system)
            except LlamaExtractError as exc:
                if exc.code in (429, 500, 502, 503, 504) or exc.code is None:
                    raise LLMTransientError(str(exc)) from exc
                raise LLMPermanentError(str(exc)) from exc
        with self._lock:
            self.credits += job.credits
        if job.status != "COMPLETED" or not isinstance(job.result, dict):
            raise LLMTransientError(f"LlamaExtract job {job.status}: {job.error or 'no result'}")
        return RawResponse(text=json.dumps(job.result))
