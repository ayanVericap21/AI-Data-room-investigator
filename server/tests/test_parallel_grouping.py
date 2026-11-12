"""Grouping calls can run at the same time (the same workers option as judging), without changing how answers are handled."""
import threading
import time

from app.llm.base import LLMClient, RawResponse
from app.llm.errors import LLMPermanentError
from app.models import Claim
from app.services.llm_pairing import LLMPairer
from tests.test_pair_judge import make_claims, room  # noqa: F401  (fixtures/helpers)


class SlowGroups(LLMClient):
    """Answers every call with no groups after a short wait, and records how many calls were in flight together."""

    def __init__(self):
        super().__init__(max_attempts=1)
        self.in_flight = 0
        self.peak = 0
        self._guard = threading.Lock()

    def _raw_generate(self, prompt, system, json_mode):
        with self._guard:
            self.in_flight += 1
            self.peak = max(self.peak, self.in_flight)
        time.sleep(0.08)
        with self._guard:
            self.in_flight -= 1
        return RawResponse(text='{"groups": []}')


class EveryOtherFails(LLMClient):
    def __init__(self):
        super().__init__(max_attempts=1)
        self.calls = 0

    def _raw_generate(self, prompt, system, json_mode):
        self.calls += 1
        if self.calls % 2 == 0:
            raise LLMPermanentError("boom")
        return RawResponse(text='{"groups": []}')


def several_batches(session, room, extra=4):  # noqa: F811
    a, b = make_claims(session, room)
    for k in range(extra):
        session.add(Claim(document_id=a.document_id, chunk_id=a.chunk_id, subject=f"item{k}", metric="m", value=str(k), claim_text=f"item{k} is {k}"))
    session.commit()


def test_workers_make_grouping_calls_at_the_same_time_and_one_worker_does_not(session, room):  # noqa: F811
    several_batches(session, room)
    fast, slow = SlowGroups(), SlowGroups()
    _, fast_stats = LLMPairer(session, fast, max_chars_per_call=1, workers=4).run()  # every claim in its own batch
    LLMPairer(session, slow, max_chars_per_call=1, workers=1).run()
    assert fast_stats.batches == 6 and fast_stats.llm_calls == 6 and fast.peak >= 2 and slow.peak == 1


def test_a_failed_call_does_not_stop_the_other_batches_when_calls_run_together(session, room):  # noqa: F811
    several_batches(session, room, extra=2)
    messages: list[str] = []
    _, stats = LLMPairer(session, EveryOtherFails(), max_chars_per_call=1, workers=1, on_progress=messages.append).run()  # calls 2 and 4 fail
    assert stats.batches == 4 and stats.batches_failed == 2
    assert len(messages) == 4 and "grouping call 1 of 4 done" in messages[0] and "failed" in messages[1]


def test_a_stop_request_skips_the_calls_still_waiting(session, room):  # noqa: F811
    several_batches(session, room)
    llm = SlowGroups()
    _, stats = LLMPairer(session, llm, max_chars_per_call=1, workers=2, should_stop=lambda: True).run()
    assert stats.llm_calls == 0 and "stop requested" in stats.errors
