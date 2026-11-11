# Technical retrospective (draft)

> **Status:** the experimental comparison has **not been run yet**. No `ANTHROPIC_API_KEY` was available in the build
> environment, so there are no baseline-vs-agent numbers, latency or cost measurements. Nothing below claims the agent is
> better; that must be established with `python -m eval.run_eval` and recorded here.

## What was built and why
* Baseline first (`app/baseline.py`): one call over the whole data room. It is deliberately un-gated so we can measure what
  retrieval, tools and verification add.
* Agent: LangGraph workflow with MCP tools. Complexity was added where it targets a specific failure of the baseline:
  fabricated/misquoted evidence (→ `verify_quote` + citation rules), overconfident contradictions on contextual differences
  (→ conditions captured at extraction, support check, downgrade), no way to say "can't tell" (→ `insufficient_evidence`,
  unresolved list), run-to-run variance (→ heuristic candidate generation, deterministic gating).

## Found and fixed during the build
* The first `AnthropicLLM` used forced `tool_choice` for structured output; the current Claude models reject that (HTTP 400), so it
  would have failed on the first live call. Caught while adding caching (API reference), fixed by switching to `messages.parse`.
  Lesson: the scripted-LLM tests could not have caught this - only a live call, or reading the API reference, does.

## Verified so far (169 automated tests incl. live Postgres 17; scripted LLM; Gemini + Anthropic clients faked)
Ingest/offset invariants, job idempotency/conflict/isolation, quote verification incl. fabricated and wrong-document quotes,
downgrade rules, step limits and loop guard, LLM failure → partial, belief revision logging, user follow-up + supersede,
resume, API flow, and the real MCP server over stdio; summaries (map-reduce, isolation, skipped/failed), the Anthropic request
shape and cache/cost accounting with a faked SDK client, the cache-friendly prompt layout, and Postgres SQL generation.
These prove the *mechanisms*, not model quality. Postgres: the same suite passes on a live Postgres 17 container, and the real
server + MCP subprocess ran ingest/search/verify on it. **Not verified:** any live Anthropic call, and real cache hit rates.
Setup gotcha found on the way: a password file written with a CRLF line ending silently produced a different password than clients
send - generate it without a trailing newline.

## To fill in after the live run
- [ ] Table from `eval/results/*.md` (3+ runs, both approaches); failures and unsupported findings, with examples.
- [ ] Which components moved which metric (ablate: no support check; no heuristics; local tools vs MCP).
- [ ] Which complexity did not help.
- [ ] Cost/latency per run (set `DRI_PRICE_IN/OUT` to real rates first - defaults are placeholders).
- [ ] How AI tools were used (this prototype was built with Claude Code; scoped to the brief, tests written alongside).

## Production changes needed
A job queue (arq/Redis) and a Postgres connection pool instead of in-process locks and connection-per-call; OCR and table/XLSX parsing; embeddings + hybrid retrieval;
per-run cancellation and a durable LangGraph checkpointer; auth/multi-tenancy; webhook/callback on completion; Secrets Manager
for keys; larger, independently labelled evaluation sets with significance testing.
