"""Gemini takes the output schema as a request parameter; the other providers still read it in the prompt."""
import json

import httpx
import pytest
from pydantic import BaseModel, Field

from app.llm import google
from app.llm.google import GoogleLLMClient, gemini_schema
from app.llm.mock import MockLLMClient
from app.services.claim_group_judge import GroupVerdicts


class Inner(BaseModel):
    claim: int
    quote: str = ""


class Outer(BaseModel):
    group: int
    flag: bool = True
    note: str | None = None
    score: float = Field(default=0.5, ge=0.0, le=1.0)
    tags: list[int] = Field(default_factory=list)
    items: list[Inner] = Field(default_factory=list)


def test_a_pydantic_schema_becomes_a_gemini_response_schema():
    schema = gemini_schema(Outer.model_json_schema())
    assert schema["type"] == "OBJECT" and schema["required"] == ["group"]
    assert schema["propertyOrdering"] == ["group", "flag", "note", "score", "tags", "items"]  # the order the class declares
    props = schema["properties"]
    assert props["group"] == {"type": "INTEGER"} and props["flag"] == {"type": "BOOLEAN"}
    assert props["note"] == {"type": "STRING", "nullable": True}
    assert props["score"] == {"type": "NUMBER", "minimum": 0.0, "maximum": 1.0}
    assert props["tags"] == {"type": "ARRAY", "items": {"type": "INTEGER"}}
    assert props["items"]["items"]["properties"]["claim"] == {"type": "INTEGER"}  # the nested model is inlined, no $ref
    assert "$ref" not in json.dumps(schema) and "$defs" not in json.dumps(schema) and "title" not in json.dumps(schema) and "default" not in json.dumps(schema)


def test_the_judges_verdict_schema_converts_with_its_enums_and_its_optional_field():
    schema = gemini_schema(GroupVerdicts.model_json_schema())
    verdict = schema["properties"]["verdicts"]["items"]
    assert verdict["required"] == ["group", "classification"]  # everything else may be left out of a short verdict
    assert verdict["properties"]["classification"]["type"] == "STRING"
    assert set(verdict["properties"]["classification"]["enum"]) >= {"consistent", "contradiction", "contextual_difference"}
    assert verdict["properties"]["materiality_reasoning"]["nullable"] is True
    assert verdict["propertyOrdering"].index("attribute_reasoning") < verdict["propertyOrdering"].index("classification")  # reasoning is written first


@pytest.mark.parametrize("model_source", ["free_form", "union", "constant"])
def test_a_schema_gemini_cannot_express_is_left_to_the_prompt(model_source):
    from typing import Any, Literal

    class FreeForm(BaseModel):
        arguments: dict[str, Any]

    class Union(BaseModel):
        value: str | int

    class Constant(BaseModel):
        kind: Literal["only"]

    model = {"free_form": FreeForm, "union": Union, "constant": Constant}[model_source]
    assert gemini_schema(model.model_json_schema()) is None


class Capture:
    """Stands in for httpx.post and keeps what was sent."""

    def __init__(self):
        self.bodies = []

    def __call__(self, url, json=None, headers=None, timeout=None):
        self.bodies.append(json)
        payload = {"candidates": [{"content": {"parts": [{"text": '{"group": 1}'}]}}], "usageMetadata": {}}
        return httpx.Response(200, json=payload, request=httpx.Request("POST", url))


def test_gemini_gets_the_schema_as_a_parameter_and_the_prompt_does_not_carry_it(monkeypatch):
    capture = Capture()
    monkeypatch.setattr(google.httpx, "post", capture)
    client = GoogleLLMClient("gemini-test", "key")
    assert client.generate_structured_output("judge this", Outer, system="be brief").group == 1
    body = capture.bodies[0]
    assert body["generationConfig"]["responseSchema"]["type"] == "OBJECT" and body["generationConfig"]["responseMimeType"] == "application/json"
    sent = body["contents"][0]["parts"][0]["text"]
    assert sent == "judge this" and "JSON Schema" not in sent


def test_a_plain_text_call_after_a_structured_one_carries_no_schema(monkeypatch):
    capture = Capture()
    monkeypatch.setattr(google.httpx, "post", capture)
    client = GoogleLLMClient("gemini-test", "key")
    client.generate_structured_output("a", Outer)
    client.generate_text("b")
    assert "responseSchema" in capture.bodies[0]["generationConfig"] and "responseSchema" not in capture.bodies[1]["generationConfig"]


def test_a_schema_gemini_cannot_take_is_pasted_into_the_prompt_as_before(monkeypatch):
    from typing import Any

    class FreeForm(BaseModel):
        arguments: dict[str, Any] = {}

    capture = Capture()
    monkeypatch.setattr(google.httpx, "post", capture)
    GoogleLLMClient("gemini-test", "key").generate_structured_output("go", FreeForm)
    body = capture.bodies[0]
    assert "responseSchema" not in body["generationConfig"] and "conforms to this JSON Schema" in body["contents"][0]["parts"][0]["text"]


def test_providers_without_a_native_schema_still_get_it_in_the_prompt():
    llm = MockLLMClient(['{"group": 1}'])
    llm.generate_structured_output("q", Outer)
    assert "conforms to this JSON Schema" in llm.prompts[0]
