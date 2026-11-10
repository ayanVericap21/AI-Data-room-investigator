"""Mock LLM for automated tests: scripted responses, no network."""
from __future__ import annotations

import json
from typing import Callable

from pydantic import BaseModel

from app.llm.base import LLMClient, RawResponse, TokenUsage
from app.llm.errors import LLMError

Response = str | dict | list | BaseModel | LLMError | Callable[[str], object]


class MockLLMClient(LLMClient):
    """Pops scripted responses in order. A response may be a model/dict (serialised to
    JSON), a raw string, an LLMError instance (raised), or a callable(prompt) -> any of these.
    ``prompts`` records every prompt sent, so tests can assert what the model was shown."""

    def __init__(self, responses: list[Response] | None = None, **kwargs):
        kwargs.setdefault("sleep", lambda _s: None)
        super().__init__(**kwargs)
        self.responses = list(responses or [])
        self.prompts: list[str] = []

    def _raw_generate(self, prompt: str, system: str | None, json_mode: bool) -> RawResponse:
        self.prompts.append(prompt)
        if not self.responses:
            raise AssertionError("MockLLMClient ran out of scripted responses")
        item = self.responses.pop(0)
        if callable(item):
            item = item(prompt)
        if isinstance(item, LLMError):
            raise item
        if isinstance(item, BaseModel):
            text = item.model_dump_json()
        elif isinstance(item, (dict, list)):
            text = json.dumps(item)
        else:
            text = str(item)
        return RawResponse(text=text, usage=TokenUsage(prompt_tokens=len(prompt) // 4, completion_tokens=len(text) // 4,
                                                       total_tokens=(len(prompt) + len(text)) // 4))
