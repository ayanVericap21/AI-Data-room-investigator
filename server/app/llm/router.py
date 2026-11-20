"""Routing over an ordered chain of models: a call goes to the first model of the chain that can take it, and to the next when that one cannot.

A model that has used up its daily quota is left out for the rest of the run (its quota does not clear within a run); a model that fails a call after
its own retries (busy, timed out, an unusable answer) is skipped for that call only. Every model has its own limits (the free tier gives Flash Lite
models 15 requests a minute and 500 a day, Flash models 5 a minute and 20 a day), so each client of the chain paces and counts on its own. The clients share
one set of call metrics and token totals with the router, so a run's report covers every model; each metric names its model.
"""
from __future__ import annotations

import logging

from app.llm.base import LLMClient, RawResponse, T
from app.llm.errors import LLMError, LLMInvalidOutputError, LLMPermanentError, LLMQuotaError, LLMTransientError

logger = logging.getLogger(__name__)


class RoutedLLMClient(LLMClient):
    def __init__(self, clients: list[LLMClient]):
        if not clients:
            raise ValueError("a chain needs at least one model")
        super().__init__(max_attempts=clients[0].max_attempts)
        self.clients = clients
        self.unavailable: dict[int, str] = {}  # position in the chain -> why the model is left out for the rest of the run
        self.routed_calls = [0] * len(clients)  # calls answered by each model
        self.fell_back = 0  # calls that went to a later model of the chain
        for client in clients:  # one report for the chain
            client.call_metrics, client._usage, client._lock = self.call_metrics, self._usage, self._lock

    def share_with(self, main: LLMClient) -> None:
        """Count this chain's calls and tokens in ``main``'s report (one run, one set of totals)."""
        self.call_metrics, self._usage, self._lock = main.call_metrics, main._usage, main._lock
        for client in self.clients:
            client.call_metrics, client._usage, client._lock = main.call_metrics, main._usage, main._lock

    @property
    def models(self) -> list[str]:
        return [getattr(c, "model", type(c).__name__) for c in self.clients]

    def _raw_generate(self, prompt: str, system: str | None, json_mode: bool) -> RawResponse:  # pragma: no cover - the chain's clients make the calls
        raise NotImplementedError("a routed client calls the models of its chain")

    def generate_structured_output(self, prompt: str, schema: type[T], system: str | None = None, max_output_tokens: int | None = None) -> T:
        last: LLMError | None = None
        for position, client in enumerate(self.clients):
            if position in self.unavailable:
                continue
            try:
                answer = client.generate_structured_output(prompt, schema, system=system, max_output_tokens=max_output_tokens)
            except LLMQuotaError as exc:
                self.unavailable[position] = str(exc)[:200]
                logger.warning("Model %s has used up its quota; routing to the next model", self.models[position])
                last = exc
                continue
            except (LLMTransientError, LLMInvalidOutputError) as exc:  # busy or unusable after its own retries: the next model may answer
                logger.warning("Model %s failed this call (%s); routing to the next model", self.models[position], exc.__class__.__name__)
                last = exc
                continue
            except LLMPermanentError as exc:
                if "is not found" in str(exc) or "404" in str(exc)[:40]:  # a retired or unknown model: left out
                    self.unavailable[position] = str(exc)[:200]
                    last = exc
                    continue
                raise  # a bad request would fail on every model the same way
            self.routed_calls[position] += 1
            if position > 0:
                self.fell_back += 1
            return answer
        if len(self.unavailable) == len(self.clients):
            self.quota_exhausted = "every model of the chain is out: " + "; ".join(f"{m}: {self.unavailable[i][:120]}" for i, m in enumerate(self.models))
            raise LLMQuotaError(self.quota_exhausted)
        assert last is not None
        raise last
