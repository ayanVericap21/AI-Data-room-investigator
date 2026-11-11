"""Judge calls can run at the same time (a flag), without changing what is saved, and the client is safe to share between threads."""
import threading
import time

from sqlalchemy import select

from app.llm.base import LLMClient, RawResponse
from app.llm.mock import MockLLMClient
from app.models import Finding
from app.services.pair_judge import PairJudge
from app.tools import build_registry
from tests.test_pair_judge import make_claims, pair, verdict_for
from tests.test_tools import add_doc, room  # noqa: F401  (fixtures and helper)


class SlowFake(LLMClient):
    """Answers every call with no verdicts after a short wait, and records how many calls were in flight together."""

    def __init__(self):
        super().__init__(max_attempts=1)
        self.in_flight = 0
        self.peak = 0
        self.sizes: set[int] = set()
        self._guard = threading.Lock()

    def _raw_generate(self, prompt, system, json_mode):
        with self._guard:
            self.in_flight += 1
            self.peak = max(self.peak, self.in_flight)
        time.sleep(0.08)
        with self._guard:
            self.in_flight -= 1
        return RawResponse(text='{"verdicts": []}')


def many_pairs(session, room, n):  # noqa: F811
    a, b = make_claims(session, room)
    return [pair(a, b) for _ in range(n)], a, b


def test_workers_make_the_model_calls_at_the_same_time_and_one_worker_does_not(session, room):  # noqa: F811
    pairs, _, _ = many_pairs(session, room, 1)
    several = []
    for k in range(4):  # four different pairs, each alone in a batch
        extra, _ = add_doc(session, f"extra{k}.pdf", "pdf", [dict(content=f"Capacity is {k} tonnes. " * 20, page_number=1)])
        several.append(extra)
    session.commit()
    from app.models import Claim
    claims = []
    a = pairs[0]
    base_a = session.scalars(select(Claim)).first()
    for k, doc in enumerate(several):
        from app.models import DocumentChunk
        chunk = session.scalars(select(DocumentChunk).where(DocumentChunk.document_id == doc.id)).one()
        c = Claim(document_id=doc.id, chunk_id=chunk.id, subject="plant", metric="capacity", value=k, claim_text=f"Capacity is {k} tonnes.")
        session.add(c)
        claims.append(c)
    session.commit()
    all_pairs = [pair(base_a, c) for c in claims]

    fast = SlowFake()
    PairJudge(session, fast, build_registry(session, None, room.investigation.id), max_chars_per_call=1200, workers=4).run(all_pairs)
    slow = SlowFake()
    PairJudge(session, slow, build_registry(session, None, room.investigation.id), max_chars_per_call=1200, workers=1).run(all_pairs)
    assert fast.peak >= 2 and slow.peak == 1


def test_the_same_findings_are_saved_with_several_workers(session, room):  # noqa: F811
    a, b = make_claims(session, room)
    p = pair(a, b)
    verdict = verdict_for(p, a)
    sequential = PairJudge(session, MockLLMClient([verdict]), build_registry(session, None, room.investigation.id), workers=1).run([p])
    saved = session.scalars(select(Finding)).all()
    assert sequential.findings_saved == 1 and len(saved) == 1
    for f in saved:
        session.delete(f)
    session.commit()
    parallel = PairJudge(session, MockLLMClient([verdict]), build_registry(session, None, room.investigation.id), workers=4).run([p])
    assert parallel.findings_saved == 1 and len(session.scalars(select(Finding)).all()) == 1


def test_the_client_keeps_each_calls_own_details_when_threads_share_it():
    client = SlowFake()
    client._raw_generate = lambda prompt, system, json_mode: (time.sleep(0.05), RawResponse(text='{"verdicts": []}'))[1]
    from app.services.pair_judge import PairVerdictBatch

    def call(n):
        client.generate_structured_output("x" * n, PairVerdictBatch, system=None)

    threads = [threading.Thread(target=call, args=(100 * (i + 1),)) for i in range(4)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    lengths = sorted(m.prompt_chars for m in client.call_metrics)
    assert len(client.call_metrics) == 4 and len(set(lengths)) == 4  # four different sizes, none mixed up
    assert client.get_token_usage().calls == 4


def test_the_run_script_has_the_flag_with_one_worker_as_the_default():
    import pathlib
    text = (pathlib.Path(__file__).resolve().parent.parent / "scripts" / "ledger_run.py").read_text(encoding="utf-8")
    assert '"--judge-workers", type=int, default=1' in text and "workers=args.judge_workers" in text
