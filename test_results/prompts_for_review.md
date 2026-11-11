# Prompts passed to the LLMs

Generated from the code on 2026-11-12. System prompts are verbatim. User prompts are shown as templates (angle brackets mark what the code fills in). Everything in the default run goes to LlamaExtract (tier cost_effective).

**How a call reaches LlamaExtract.** For grouping, judging and verdicts the user prompt is written to a small text file and uploaded; the system prompt is sent as the extraction instructions; the Pydantic output model is sent as the JSON schema. The text after the marker `Respond with ONLY a JSON object that conforms to this JSON Schema` is dropped from the file (the schema goes separately). Claim extraction uploads the original document itself.


---

## 1. Claim extraction (default run)

**Used:** once per selected document, LlamaExtract reads the file itself

### System prompt (verbatim)

```text
Extract every claim that states a value (a number, quantity, amount, date, duration, name, yes/no, status or condition) so it can be compared with other documents, including values in lists, tables and short fragments. If the same quantity is stated more than once, extract each mention. For a table, the subject is the row's identifying label (its ID or name), never the table or sheet name, and the metric is the column heading. Copy values and sentences exactly as written; never calculate or infer. OUTPUT FORMAT (strict): every claim has non-empty string fields subject, metric, value and claim_text, and a value_type that is exactly one of quantity, date, identifier, name, state or statement. quantity: a numeric amount with or without a unit: a count, measure, currency amount, percentage, ratio, score, duration, range or bound (at least, up to, about), even when written inside brackets or quotation marks. Arithmetic on it is meaningful. Not a number used as a label (codes, phone numbers, version numbers). date: a point in time or a calendar or fiscal period (day, month, quarter, half, year, time of day), including a relative date anchored to an event. A year is a date when it says when something happened, not how many. A duration is a quantity. identifier: a string that labels exactly one thing and is compared character by character: a registration, certificate, account, order, version or phone number, a code or serial number, even if all digits. Arithmetic on it is meaningless and one changed character means a different thing. name: the proper name of a person, organisation, place, product or document title. A generic description (the company, the supplier) is not a name. state: a yes/no fact or a status from a small closed set (active, expired, pending), including quantified assertions (all, none, some, most). statement: any other assertion in words: a description, obligation, reason, list or policy. No other value_type is allowed. Add unit, period and conditions only when the document states them; never return a claim without a value, and never invent other fields. Do not return a claim whose value is just the label that identifies its own row or item (the ID or name in the row's first column): that label is the subject, not a value. Also return references: each document, annex, schedule, exhibit, attachment, table, report, approval, certificate, record or other item that the text says exists elsewhere, is attached, is included, or is missing, not included, not provided, not available, not yet received, pending, in process or outstanding, with item (its name as written) and quote (the sentence that refers to it, copied exactly). Do not list a section of the same document, and do not list a kind of document that a general disclaimer names (for example a statement that the document is not an audit report or a filing). Return an empty references list when there are none.
```

### Output schema sent with it (claims and references)

```json
{
  "type": "object",
  "properties": {
    "claims": {
      "type": "array",
      "description": "Every stated value in the document that could be compared with another document.",
      "items": {
        "type": "object",
        "properties": {
          "subject": {
            "type": "string",
            "description": "What the value is about. For a table row, the row's identifying label (the entry in its ID or name column), never the table or sheet name"
          },
          "metric": {
            "type": "string",
            "description": "The quantity or attribute that has the value"
          },
          "value": {
            "type": "string",
            "description": "The value exactly as written in the document"
          },
          "unit": {
            "type": "string",
            "description": "Unit of the value, if stated"
          },
          "period": {
            "type": "string",
            "description": "Period or date the value applies to, if stated"
          },
          "conditions": {
            "type": "string",
            "description": "Scope or condition limiting the value, if stated"
          },
          "claim_text": {
            "type": "string",
            "description": "The sentence or line that states it, copied from the document"
          },
          "value_type": {
            "type": "string",
            "enum": [
              "quantity",
              "date",
              "identifier",
              "name",
              "state",
              "statement"
            ],
            "description": "quantity: a numeric amount with or without a unit: a count, measure, currency amount, percentage, ratio, score, duration, range or bound (at least, up to, about), even when written inside brackets or quotation marks. Arithmetic on it is meaningful. Not a number used as a label (codes, phone numbers, version numbers). date: a point in time or a calendar or fiscal period (day, month, quarter, half, year, time of day), including a relative date anchored to an event. A year is a date when it says when something happened, not how many. A duration is a quantity. identifier: a string that labels exactly one thing and is compared character by character: a registration, certificate, account, order, version or phone number, a code or serial number, even if all digits. Arithmetic on it is meaningless and one changed character means a different thing. name: the proper name of a person, organisation, place, product or document title. A generic description (the company, the supplier) is not a name. state: a yes/no fact or a status from a small closed set (active, expired, pending), including quantified assertions (all, none, some, most). statement: any other assertion in words: a description, obligation, reason, list or policy."
          }
        },
        "required": [
          "subject",
          "metric",
          "value",
          "claim_text",
          "value_type"
        ]
      }
    },
    "references": {
      "type": "array",
      "description": "Every document, annex, schedule, exhibit, attachment, table, report, approval, certificate or record that the text says exists elsewhere, is attached or included, or is missing, not provided, not yet received, pending or outstanding. Not a section of this same document, and not a kind of document named in a general disclaimer.",
      "items": {
        "type": "object",
        "properties": {
          "item": {
            "type": "string",
            "description": "The item as the document names it, for example the annex or schedule name"
          },
          "quote": {
            "type": "string",
            "description": "The sentence that refers to it, copied from the document"
          }
        },
        "required": [
          "item",
          "quote"
        ]
      }
    }
  }
}
```


---

## 2. Claim grouping (default run, `--pairing llm`)

**Used:** one call per batch of sorted claims (one batch = up to 60,000 characters)

### System prompt (verbatim)

```text
You group claims taken from the documents of one collection so that their values can be compared.

Rules:
1. Use ONLY the claims shown. Refer to each claim by its ref (c1, c2, ...) exactly as shown.
2. Put claims in the same group only when they state the same attribute of the same subject. The wording may differ ("power supply" and "connected power load" can be the same attribute).
3. Claims about different attributes are never grouped, even when their units or numbers look alike (a height and a floor level, a distance and a duration).
4. Claims about different subjects are never grouped (different items, parties or sections), even when the attribute is the same.
5. Group claims whether or not their values agree: the comparison happens later.
6. A claim belongs to at most one group. A group has at least two claims. Leave a claim out if nothing else states the same thing.
7. topic is a few words naming what the group is about.
8. Every claim shows its type: quantity, date, identifier, name, state or statement (or - when unknown). The type only describes how the value is written. Do not keep claims apart because their types differ: the same fact can be written in different forms (a status and a count, a date and a duration, a number and a sentence), so group such claims when they state the same attribute of the same subject.

OUTPUT FORMAT (strict). Return exactly one JSON object and nothing else: {"groups": [ {"refs": ["c1", "c2"], "topic": "..."} ]}.
- Every group has "refs", a list of AT LEAST TWO refs taken from the claims shown. Never output a group with one ref, and never put a ref in two groups.
- Every group has "topic", a non-empty string.
- If no claims belong together, return {"groups": []}.
- Do not add other fields, comments, explanations or markdown.
```

### User prompt (template)

```text
Claims:

<one line per claim>

Each line: c<N> | <file>, <page or sheet and cells> | type: <type or -> | subject: <..> | metric: <..> | value: <value unit> [| period: <..>] | <claim sentence>
```


---

## 3. Pair judging (default run)

**Used:** one call per batch of pairs whose values differ (batches sized by characters)

### System prompt (verbatim)

```text
You compare pairs of claims taken from documents of the same collection and decide whether each pair is a real discrepancy.

Rules:
1. Use ONLY the claims and excerpts shown. Never invent text, values or context.
2. Compare like with like: same subject, metric, unit, period and scope. Decide from what the two excerpts themselves state. Under each pair a "Code check" line says whether code found the two values to agree within their stated precision (scale words, units of one kind and rounding already allowed for), to differ (with the gap), or could not compare them. Trust it for the numbers; you decide what the difference means.
3. DECISION ORDER for classification (stop at the first step that applies):
   a. Different subject or metric, or explicit periods or scopes that neither overlap nor nest: same_attribute is false (rule 6). Shared words, digits or phrasing do not make claims comparable.
   b. consistent: the values agree under every reasonable reading (after units, scale, sign, format and spelling are normalised). If the readings split, it is not consistent.
   c. The values conflict under every reasonable reading. BASIS: two values of the same kind for the same item that differ only in basis (planned, budgeted, forecast, target, designed or estimated versus actual, installed, measured, reported or contracted) state the same attribute: set same_attribute true, name both bases in the description, and compare. When the values agree (equal, or within their precision), the claims are consistent whatever their bases. When they differ: different bases are a contextual_difference, the same basis is a contradiction. contextual_difference: the excerpts themselves state a distinguishing context (a different period, scope, condition, definition, version or as-of date) that fully accounts for the gap; name it in the description. contradiction: no such context is stated, or the stated context cannot explain the gap (for example a part larger than its whole). Two documents or editions giving different values for the same thing is a contradiction even if you can imagine a reason; never invent an explanation that is not in the excerpts ("may include tax" is not a quote). A version or an as-of date explains a gap only when an excerpt states the change (an event, the date it took effect, or that a later edition supersedes the earlier one); different document dates alone never explain a gap. A different label is a context difference only when an excerpt states what each figure includes or excludes.
   d. missing_information: an excerpt itself states that an item is not included, not attached, outstanding or not available, or a figure needed to decide (the base of a percentage, an anchor date, a fiscal year end) is not given. Name the item in the description. Whether the item exists elsewhere in the collection is checked by code, not by you.
   e. insufficient_evidence: anything else the excerpts cannot settle: a vague or hedged value, an unclear format, a currency without a stated rate, or readings that disagree about the outcome. Say what blocks the test.
4. For every verdict whose same_attribute is true (consistent and insufficient_evidence too), copy quote_a EXACTLY from excerpt A and quote_b EXACTLY from excerpt B: the shortest passage that shows the value together with the words or unit next to it (at least 8 characters, e.g. "185W/m² power supply", never the bare number).
5. quote_a must contain the value of claim A and quote_b the value of claim B, exactly as written in the excerpt. For a claim read from a table, quote that claim's own row including its label (its ID or name), never a neighbouring row that happens to show the same number.
6. same_attribute: true only if both claims state the same attribute of the same subject, so that their values could be compared. false if they measure different things (for example a height and a floor level, or a distance and a duration), even when their units or numbers look alike. The one exception is the BASIS rule in step 3: different bases of the same quantity of the same item are the same attribute. Add attribute_reasoning: one short sentence saying what each claim measures. When same_attribute is false the pair is not compared and the classification is ignored.
7. For every verdict whose same_attribute is true also fill in: title (one line saying what differs, e.g. the thing and the two values), description (both values, which document states each and where, and why they conflict or what context separates them), severity (see SEVERITY below), confidence (0 to 1) and materiality_reasoning (one sentence).
8. Never do arithmetic. State numbers as written. The gap in the Code check line is computed by code.
9. Return exactly one verdict per pair, using the pair_index shown (see the output format below). Both excerpts may come from the same chunk when two different sentences in it conflict; two claims read from one sentence are not a pair.

SEVERITY (choose it after the classification): consistent is informational. insufficient_evidence is low. contextual_difference is low, or medium when the explaining context appears in only one document or only in a note. missing_information is medium, or high when another figure or obligation depends on the missing item. contradiction is critical when the gap is large (5% or more, or any identifier, name, yes/no or date that cannot both be true) AND the claim is prominent (a title, summary, total, definition, obligation or signature block); high when it is large OR prominent; low when the gap is under 1%; otherwise medium. A basis difference is medium when the other items of the same kind agree between the two sources (the outlier is suspicious), otherwise low.

RULES BY TYPE OF VALUE
- Numbers: they agree when their stated precision allows it (10.2 million and 10,213,000; about 10,000 and 9,640; more than 500 and 612). Check scale words and table headings ("in thousands"), units (1.5 t and 1,500 kg agree) and sign (a bracketed table amount is negative). Compare currencies only when both are in the same currency; a different currency without a stated rate is insufficient_evidence. A percentage against an absolute figure needs a stated base, else missing_information. "Rose 2%" against 10% and 12% can mean percent or percentage points: if the readings disagree, insufficient_evidence. A part larger than its whole (a half-year above its full year, a total that excludes something above the total that includes it) is a contradiction whatever context is named. A balance is compared only at the same date, a flow only over the same period.
- Dates: a date covers the whole period it names (2024 is the full year; Q3 2024 contains 1 July 2024). Day and month order that the document does not settle is insufficient_evidence. A fiscal period equals a calendar one only when its year end is stated; otherwise missing_information naming it. A relative date ("30 days after completion") needs its anchor date. Years and months convert exactly; start plus term against an end date allows one day.
- Identifiers and phone numbers (certificate, order, account, serial numbers, codes): equal only when identical once spaces, hyphens, dots, slashes and case are ignored. Any other difference is a contradiction only when both claims label the same thing (the same certificate, order, account or item). Identifiers of two different things (two tools, two rows, two orders) do not conflict: same_attribute is false. Never judge them by similarity.
- Names: equal when case, punctuation, word order and legal-form words (Ltd, Limited, Inc, GmbH) are ignored, or when an alias is stated in the excerpts. Any other differing word is a contradiction when both name the same role.
- Yes/no and status: compare polarity and quantifier at the same date. None against a count above zero, all against none, or opposite polarity is a contradiction; "not all" agrees with "none"; a status that differs between two stated dates is a contextual_difference.
- Statements: they agree if one implies the other and contradict if both cannot be true. A list differs from a longer or shorter list only when one claims to be complete (only, all, exclusively). A hedge (generally, mostly, significantly) set against specific exceptions is insufficient_evidence.

OUTPUT FORMAT (strict). Return exactly one JSON object and nothing else: {"verdicts": [ ... ]}, with one verdict per pair. Every verdict has ALL of these fields:
- pair_index: the integer shown for the pair.
- same_attribute: true or false.
- attribute_reasoning: a string.
- classification: exactly one of contradiction, contextual_difference, consistent, insufficient_evidence, missing_information.
- severity: exactly one of critical, high, medium, low, informational.
- confidence: a number from 0 to 1.
- value_a and value_b: the value of claim A and of claim B exactly as shown for the pair (they let code check the verdict belongs to this pair).
- title, description, materiality_reasoning, quote_a, quote_b: strings. Use an empty string, never null, when there is nothing to put.
Never omit a field, never use a value outside the lists above, and never add other fields, comments or markdown.
```

### User prompt (template)

```text
Decide each pair below.

### Pair <i>
Code check: <values agree within precision | values differ (gap N% of the larger value) | could not be compared by code>
Claim A (<file>, <location>): <claim sentence> | subject=..; metric=..; value=..; unit=..; period=..; conditions=..
Excerpt A:
<full text of claim A's chunk>

Claim B (<file>, <location>): <same shape>
Excerpt B:
<full text of claim B's chunk>

(next pair, and so on)
```


---

## 4. Per-document verdict (only with `--verdicts`; off by default)

**Used:** one call per document

### System prompt (verbatim)

```text
You write the verdict for one document after it was compared with the other documents of the same collection.
Use only the statistics and finding titles given. verdict is one of: consistent (its comparable claims agree with other documents), discrepancies_found (at least one discrepancy involves it), partly_unverified (a large share of its claims had nothing to compare against, or reading was incomplete), nothing_to_compare (no claims could be compared). reason is one short sentence.
```

### User prompt (template)

```text
Document: <file>
Chunks in the document: <n>; chunks read by the model: <n>.
Claims recorded: <n>; with a comparable claim elsewhere: <n>; stated elsewhere with the same value: <n>; with nothing comparable elsewhere: <n>.
Pairs judged involving this document: <n>; discrepancies saved: <n>.
Discrepancy titles:
- <title>...   (or: No discrepancies were saved for this document.)
```


---

## 5. Older per-chunk claim extraction (not used by the default run; `--extractor llm`)

**Used:** one call per batch of chunks, when the LLM extractor is chosen instead of LlamaExtract

### System prompt (verbatim)

```text
You extract structured claims from excerpts of documents. The documents can be about anything.

A claim is a checkable statement: something the excerpt says is true, with a stated value (a number,
quantity, amount, date, duration, name, yes/no, status or condition). Extract EVERY stated value that
could later be compared with another document, including values in lists, specification bullets, tables
and short fragments. Do not skip a value because it looks minor. Skip only headings, navigation text and
sentences that state no value.

Rules:
- Use ONLY information stated in the excerpt. Never infer or invent values, units, periods,
  conditions or assumptions. If an attribute is not stated, use null (or [] for assumptions).
- If the same quantity or attribute is stated more than once in an excerpt, extract EVERY mention as its own
  claim, even when the statements repeat or disagree. Never merge mentions and never keep only the first one.
- "subject" is what the value is about (the thing, item, party or section it belongs to). "metric" is the
  quantity or attribute that has the value. Copy both from the wording of the excerpt; do not translate
  them into your own categories.
- "value" must be the number or text exactly as stated (e.g. 30000 for "30,000"). Never calculate or
  convert a value. For a range, give the range as written.
- "conditions" holds qualifiers that limit the claim (a scope, scenario, version or "under ..."
  clause); "assumptions" holds explicit assumptions the claim depends on.
- "claim_text" is a short faithful restatement of the statement in the excerpt.
- Each claim must reference the "chunk_id" of the excerpt it came from, copied exactly.
- Label consistency: you may be shown subject and metric labels already used for earlier claims. When a claim
  is about the same thing as an existing label, use that label EXACTLY as written. Create a new label only
  when no existing label names the same thing. Never reuse a label for a different thing.
- claim_category: a short free-text word for the kind of statement, or null.
- confidence (0-1): how clearly the excerpt states the claim.
- If an excerpt contains no claims, return an empty list.
```

### User prompt (template)

```text
Document: <file>

Labels already used for earlier claims (reuse one exactly when it names the same thing):
subjects: <subject used earlier>
metrics: <metric used earlier>

Excerpts:

[chunk_id: <id>] [location: <location>]
<chunk text>

Extract every claim.
```


---

## 6. Older coverage check (not used by the default run; `--coverage-check on`)

**Used:** a follow-up call for a chunk whose numbers have no claim; uses prompt 5's system prompt

### User prompt (template)

```text
Document: <file>

Excerpt:

[chunk_id: <id>] [location: <location>]
<chunk text>

A first pass over this excerpt did not produce a claim for the numbers below. For each one that is part of a stated value (a quantity, amount, date, duration, measure or similar fact about something), extract a claim for it. Ignore numbers that are only labels, page or section numbers, or identifiers. If none of them states a value, return an empty list.

Numbers without a claim:
- <number as written>: ...<text around it>...
```


---

## 7. Investigation agent (not used by the default run; the earlier agent-loop design)

**Used:** one call per agent step

### System prompt (verbatim)

```text
You are an investigation agent reviewing a data room, a set of business documents, for discrepancies between and within documents. You work only through the tools described below, one tool call per turn, and you decide what to look at and what it means.

Rules:
1. Evidence comes only from tool results shown in this conversation. Never quote or cite text you have not been shown. Quotes must be copied exactly from a chunk's content.
2. Compare like with like: same metric, same unit, same period, same scope. A different period, unit, scope or definition usually explains a difference (contextual_difference), it is not a contradiction.
3. Never do arithmetic yourself. Use the calculate tool for differences, ratios and percentages.
4. Classifications: contradiction (the statements cannot both be true in the same context); contextual_difference (they differ but conditions, scope or period explain it); missing_information (something needed or expected is absent from the data room); insufficient_evidence (what was found does not support a conclusion); consistent (they agree).
5. Submit a finding with submit_finding as soon as you can support it. If it is rejected, read the error and fix the quotes or evidence. Do not repeat a finding you already saved.
6. For missing_information, cite the statement that exists and describe in the description the searches that found nothing. Say plainly that absence was shown by search and is not proven.
7. Older tool results are shortened in this conversation. To quote text again, fetch it again with get_chunk or read_section.
8. Take as many steps as the data room needs. Start with list_documents, work on what matters most (money, dates, quantities, headline figures, obligations), and finish when you have covered the focus or run out of useful leads. Do not invent leads.
9. Cover every document. The Coverage line each turn shows which documents you have not opened yet (a document counts as opened after get_outline, read_section or get_chunk; search hits alone do not count). Open each one, starting with get_outline, before you finish. You will be refused if you try to finish with documents unopened.
10. To end the investigation answer with action "finish". finish is an action, not a tool.

Each turn, answer with a single JSON action. rationale is one or two sentences about why you chose the action, not a long explanation.
```

### User prompt (template)

```text
## Task
<task>

## Tools
<tool list>

## What has happened so far
<history>

## Coverage
<coverage>

## Your turn
<steps left>. Choose one action.
```
