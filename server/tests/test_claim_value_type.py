"""Every extracted claim carries one of the six value types; anything else is dropped to none, never stored as a made-up label."""
from app.services.llama_claims import SCHEMA, SYSTEM_PROMPT, VALUE_TYPES, LlamaClaim
from app.services.value_rules import compare_values


def test_the_schema_and_prompt_allow_only_the_six_value_types():
    field = SCHEMA["properties"]["claims"]["items"]["properties"]["value_type"]
    assert field["enum"] == ["quantity", "date", "identifier", "name", "state", "statement"] == list(VALUE_TYPES)
    assert "value_type" in SCHEMA["properties"]["claims"]["items"]["required"]
    assert "exactly one of quantity, date, identifier, name, state or statement" in SYSTEM_PROMPT and "No other value_type is allowed" in SYSTEM_PROMPT
    for word in ("phone number", "A duration is a quantity", "even if all digits", "inside brackets or quotation marks"):
        assert word in field["description"]


def test_a_value_type_outside_the_allowed_ones_is_not_kept():
    assert LlamaClaim(value="5", value_type=" Quantity ").kind == "quantity"
    assert LlamaClaim(value="5", value_type="numerical").kind is None  # the old label is no longer allowed
    assert LlamaClaim(value="5", value_type="currency").kind is None
    assert LlamaClaim(value="5").kind is None


def test_identifiers_are_compared_character_by_character_and_names_after_normalising():
    assert compare_values("0481-2217", None, "04812271", None, "identifier", "identifier") == "different"  # not read as a number range
    assert compare_values("+91 80 1234 5678", None, "918012345678", None, "identifier", "identifier") == "equal"
    assert compare_values("SF-65-A", None, "sf 65 a", None, "identifier", "identifier") == "equal"
    assert compare_values("ACME Ltd", None, "Acme Limited", None, "name", "name") == "equal"
    assert compare_values("ACME Holdings Ltd", None, "ACME Ltd", None, "name", "name") == "different"
    assert compare_values("20240115", None, "2024-01-15", None, "identifier", "date") == "different"  # a code is not a date
    assert compare_values("412", None, "388", None, "quantity", "quantity") == "different"
