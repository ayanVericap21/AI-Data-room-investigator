# Agent tools

The tool layer is the only way the investigation agent reads documents or saves findings. It lives in `server/app/tools/`; this file is generated from the registry (`ToolRegistry.describe()`), so it matches the code.

Each turn the agent returns one JSON action (`server/app/agent/`):

```json
{"action": "call_tool", "tool": "search", "arguments": {"query": "plant capacity"}, "rationale": "find capacity statements"}
```

or `{"action": "finish", "rationale": "...", "summary": "..."}`. The model may only call the tools below.

## Tools

| # | Tool | Purpose |
|---|---|---|
| 1 | `list_documents` | List every document in the data room with its id, type, processing status and number of chunks. |
| 2 | `get_outline` | Show the structure of one document: its pages, sheets or headings with the number of chunks in each. |
| 3 | `search` | Semantic search over all stored passages. |
| 4 | `get_chunk` | Fetch one passage by id, optionally with neighbouring passages for context. |
| 5 | `read_section` | Read the passages of one page (PDF), sheet (XLSX) or heading (DOCX) of a document, in order. |
| 6 | `calculate` | Exact arithmetic on plain numbers (+ - * / % ** and parentheses). |
| 7 | `submit_finding` | Save a finding. |

### `list_documents`

List every document in the data room with its id, type, processing status and number of chunks. Start here.

No arguments.

**Returns:** `documents`: id, filename, file_type, status and chunk_count for each document. Documents marked missing are left out.

### `get_outline`

Show the structure of one document: its pages, sheets or headings with the number of chunks in each. Use it to decide what to read.

| Argument | Type | Required | Notes |
|---|---|---|---|
| `document_id` | string | yes | Id from list_documents. |

**Returns:** `sections`: for each page (PDF), sheet (XLSX) or heading (DOCX), its title, chunk range and chunk count.

**Tool errors:** `not_found`.

### `search`

Semantic search over all stored passages. Returns the closest passages with their exact text, source and location. Scores are similarity only and never prove a contradiction.

| Argument | Type | Required | Notes |
|---|---|---|---|
| `query` | string (min length 1, max length 500) | yes | What to look for, in natural language. |
| `top_k` | integer (min 1, max 10) | no | How many passages to return. Default 5. |
| `document_ids` | list of string | no | Only search these documents. |
| `exclude_document_ids` | list of string | no | Skip these documents. |

**Returns:** `hits`: closest passages with `chunk_id`, `filename`, `locator`, exact `content` and similarity `score` (best first).

**Tool errors:** `search_unavailable`, `embedding_failed`.

### `get_chunk`

Fetch one passage by id, optionally with neighbouring passages for context.

| Argument | Type | Required | Notes |
|---|---|---|---|
| `chunk_id` | string | yes |  |
| `neighbors` | integer (min 0, max 3) | no | Also return this many chunks before and after, in reading order. Default 0. |

**Returns:** `chunk` (exact text with locator) plus `before` and `after` neighbours in reading order.

**Tool errors:** `not_found`.

### `read_section`

Read the passages of one page (PDF), sheet (XLSX) or heading (DOCX) of a document, in order. Give exactly one of page_number, sheet_name or heading.

| Argument | Type | Required | Notes |
|---|---|---|---|
| `document_id` | string | yes |  |
| `page_number` | integer | no | PDF page. |
| `sheet_name` | string | no | XLSX sheet name. |
| `heading` | string | no | DOCX heading text, as shown by get_outline. |
| `max_chunks` | integer (min 1, max 40) | no | Default 15. |

**Returns:** `chunks` of the selected page, sheet or heading in order, `total_chunks`, and `truncated` when the reply was capped.

**Tool errors:** `not_found`.

### `calculate`

Exact arithmetic on plain numbers (+ - * / % ** and parentheses). Always use this for differences, ratios and percentages instead of computing them yourself.

| Argument | Type | Required | Notes |
|---|---|---|---|
| `expression` | string (min length 1, max length 200) | yes | Arithmetic on plain numbers: + - * / % ** and parentheses. No names, no commas. |
| `label` | string | no | What is being computed, for the trace. |

**Returns:** `result` as an exact decimal string.

**Tool errors:** `invalid_expression`.

### `submit_finding`

Save a finding. Every evidence quote is verified against the stored text and the finding is rejected if any quote is not found exactly. Comparing classifications (contradiction, contextual_difference, consistent) need evidence from at least two chunks. For missing_information or insufficient_evidence, cite what exists and describe in the description which searches found nothing.

| Argument | Type | Required | Notes |
|---|---|---|---|
| `title` | string (min length 1, max length 300) | yes |  |
| `description` | string (min length 1) | yes |  |
| `classification` | one of `contradiction`, `contextual_difference`, `missing_information`, `insufficient_evidence`, `consistent` | yes |  |
| `severity` | one of `critical`, `high`, `medium`, `low`, `informational` | yes |  |
| `confidence` | number (min 0.0, max 1.0) | yes |  |
| `evidence_status` | one of `adequate`, `partial`, `inadequate`, `unavailable` | yes |  |
| `materiality_reasoning` | string | no |  |
| `business_impact` | string | no |  |
| `unresolved_questions` | list of string | no |  |
| `evidence` | list of object (`chunk_id`, `quoted_text`, `role`, `relevance_reason`, `support_assessment`) (min items 1, max items 12) | yes |  |

**Returns:** `finding_id`, `verification_status`, and the stored `evidence` with the exact source text of each quote.

**Tool errors:** `validation_failed`, `no_investigation`, `finding_budget_exhausted`, `persist_failed`.

## Behaviour shared by every tool

- A call never raises. Failures come back as `{ok: false, error: {code, message}}` so the agent can correct itself.
- Common errors: `unknown_tool`, `invalid_arguments` (unknown or missing arguments are rejected, never ignored), `budget_exhausted`, `internal_error`.
- Every call counts against `max_tool_calls` (default 60), including failed ones, is logged with its latency, and is summarised in the run result.
- Every result carries `chunk_id` and a readable `locator` (page, sheet and cell, or heading), so any citation can be re-checked.

## What the code enforces, whatever the model does

- **Exact quotes.** `submit_finding` checks every quote against the stored chunk text. If any quote is not found, nothing is saved. Differences in whitespace only are accepted, and the exact source text is stored instead of the model's copy.
- **Two sides for comparisons.** `contradiction`, `contextual_difference` and `consistent` need evidence from at least two chunks.
- **Bounded runs.** Step cap, tool budget, finding budget and a stop flag are checked by code.
- **Arithmetic is code.** `calculate` uses exact decimal arithmetic and accepts only numbers and operators.

## Not built yet

- `list_topics`: topic grouping and where topics are stored are undecided (a schema decision).
- An MCP server wrapper. It would need the MCP Python SDK, a new dependency. The tools are plain functions, so a wrapper changes nothing here.
- Native provider function-calling. The agent uses a JSON action protocol that works with any provider through `generate_structured_output`.
