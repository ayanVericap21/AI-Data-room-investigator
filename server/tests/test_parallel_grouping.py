"""Grouping calls can run at the same time (the same workers option as judging), without changing how answers are handled."""
import threading
import time

from app.llm.base import LLMClient, RawResponse
from app.llm.errors import LLMPermanentError
from app.models import Claim
from app.services.claim_grouping import ClaimGrouper
from tests.test_claim_group_judge import make_claims, room  # noqa: F401  (fixtures/helpers)


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
    _, fast_stats = ClaimGrouper(session, fast, max_chars_per_call=1, workers=4).run()  # every claim in its own batch
    ClaimGrouper(session, slow, max_chars_per_call=1, workers=1).run()
    assert fast_stats.batches == 6 and fast_stats.llm_calls == 6 and fast.peak >= 2 and slow.peak == 1


def test_a_failed_call_does_not_stop_the_other_batches_when_calls_run_together(session, room):  # noqa: F811
    several_batches(session, room, extra=2)
    messages: list[str] = []
    _, stats = ClaimGrouper(session, EveryOtherFails(), max_chars_per_call=1, workers=1, on_progress=messages.append).run()  # calls 2 and 4 fail
    assert stats.batches == 4 and stats.batches_failed == 2
    assert len(messages) == 4 and "grouping call 1 of 4 done" in messages[0] and "failed" in messages[1]


def test_a_stop_request_skips_the_calls_still_waiting(session, room):  # noqa: F811
    several_batches(session, room)
    llm = SlowGroups()
    _, stats = ClaimGrouper(session, llm, max_chars_per_call=1, workers=2, should_stop=lambda: True).run()
    assert stats.llm_calls == 0 and "stop requested" in stats.errors


# ---- a failed batch is asked again as two halves --------------------------------------------------------------------------------------------
from app.llm.errors import LLMTransientError  # noqa: E402


class BusyOnLargePrompts(LLMClient):
    """Answers 503 while a prompt holds more than ``limit`` claims, like a service that is overloaded by large requests; records the sizes it was asked."""

    def __init__(self, limit):
        super().__init__(max_attempts=1)
        self.limit, self.sizes = limit, []

    def _raw_generate(self, prompt, system, json_mode):
        claims = sum(1 for line in prompt.splitlines() if line.startswith("c") and " | " in line)
        self.sizes.append(claims)
        if claims > self.limit:
            raise LLMTransientError("503 high demand")
        return RawResponse(text='{"groups": []}')


def many_claims(session, room, count):  # noqa: F811
    a, _b = make_claims(session, room)
    for k in range(count):
        session.add(Claim(document_id=a.document_id, chunk_id=a.chunk_id, subject=f"item{k}", metric="m", value=str(k), claim_text=f"item{k} is {k}"))
    session.commit()


def test_a_batch_that_fails_is_asked_again_as_halves_and_its_claims_are_not_lost(session, room):  # noqa: F811
    many_claims(session, room, 598)  # 600 claims in one batch
    llm, pauses, messages = BusyOnLargePrompts(limit=350), [], []
    _, stats = ClaimGrouper(session, llm, max_chars_per_call=10_000_000, retry_pause_seconds=5, sleep=pauses.append, on_progress=messages.append).run()
    assert llm.sizes == [600, 300, 300]  # the whole batch failed, then each half went through
    assert (stats.batches_split, stats.batches_failed, stats.batches) == (1, 0, 2) and sum(pauses) == 5  # one pause before the halves
    assert any("asking again as two halves" in m for m in messages)


def test_a_half_that_fails_is_split_again_until_it_is_small_enough_or_too_small_to_split(session, room):  # noqa: F811
    many_claims(session, room, 598)
    llm = BusyOnLargePrompts(limit=150)
    _, stats = ClaimGrouper(session, llm, max_chars_per_call=10_000_000, retry_pause_seconds=0, sleep=lambda _s: None).run()
    assert llm.sizes[0] == 600 and max(llm.sizes[1:]) <= 300 and stats.batches_failed == 0 and stats.batches_split == 3  # 600 -> 300+300 -> 150+150 each
    never = BusyOnLargePrompts(limit=0)
    _, stats = ClaimGrouper(session, never, max_chars_per_call=10_000_000, retry_pause_seconds=0, sleep=lambda _s: None).run()
    assert stats.batches_failed > 0 and min(never.sizes) >= 100  # a piece under 200 claims is not split again: it fails and is counted


def test_a_permanent_error_is_not_split_because_smaller_would_fail_the_same_way(session, room):  # noqa: F811
    many_claims(session, room, 598)
    llm = EveryOtherFails()
    llm.calls = 1  # the first call fails (permanent)
    _, stats = ClaimGrouper(session, llm, max_chars_per_call=10_000_000, retry_pause_seconds=0, sleep=lambda _s: None).run()
    assert stats.batches_split == 0 and stats.batches_failed == 1 and llm.calls == 2


def test_a_stop_request_during_the_pause_ends_the_retry(session, room):  # noqa: F811
    many_claims(session, room, 598)
    state = {"stop": False}
    llm = BusyOnLargePrompts(limit=0)
    _, stats = ClaimGrouper(session, llm, max_chars_per_call=10_000_000, retry_pause_seconds=30, should_stop=lambda: state["stop"], sleep=lambda _s: state.update(stop=True)).run()
    assert llm.sizes == [600] and stats.batches_split == 0  # no half was asked
