"""A claim is attached to the chunk that really states its value, and to the one holding its own row when several chunks hold the number."""
from types import SimpleNamespace

from app.services.llama_claims import pick_chunk
from app.services.numbers import value_supported_by_text


def chunk(id, content, page=None):
    return SimpleNamespace(id=id, content=content, page_number=page)


def test_a_number_is_found_only_as_a_whole_number_not_inside_a_date_or_a_longer_number():
    assert not value_supported_by_text("26", "Tool_ID: MET-003; Installed_Cost: 28; Commissioning_Date: 2026-03-14")
    assert not value_supported_by_text(26, "serial 1264 and 5260")
    assert value_supported_by_text("26", "Installed_Cost_INR_Crore: 26; Commissioning_Date: 2026-09-30")
    assert value_supported_by_text("24,000", "capacity of 24000 WSPM") and value_supported_by_text("92.4%", "Aggregate yield | 92.4% | x")
    assert value_supported_by_text("$1,250.50", "price $1,250.50 per unit")


def test_the_chunk_with_the_claims_own_row_is_picked_when_several_chunks_hold_the_number():
    first = chunk("c1", "[Sheet!A2:F2] Tool_ID: MET-003; Installed_Cost_INR_Crore: 28; Commissioning_Date: 2026-05-09")
    second = chunk("c2", "[Sheet!A13:F13] Tool_ID: MET-004; Installed_Cost_INR_Crore: 26; Commissioning_Date: 2026-09-30")
    assert pick_chunk([first, second], "26", "", None, "", "MET-004").id == "c2"  # only c2 holds 26 now
    both = chunk("c3", "Tool_ID: MET-003; cost 26\nTool_ID: MET-004; cost 26")
    one = chunk("c4", "Tool_ID: MET-005; cost 26")
    assert pick_chunk([one, both], "26", "", None, "", "MET-004").id == "c3"  # two chunks hold 26: the one with the label wins
    assert pick_chunk([one, both], "26", "", None, "", None).id == "c4"  # no subject: first chunk, as before
    assert pick_chunk([one], "26", "", None, "", "MET-009").id == "c4"  # a subject found nowhere does not stop the claim
