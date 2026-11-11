# Data Room Investigator: full context for a second opinion

Written 2026-11-12. This document is self-contained. The reader has not seen the code, the data, or any earlier conversation. Everything needed to judge the design is below: the goal, every pipeline step with its rules and thresholds, the actual test documents, real inputs and outputs at each step, the exact prompts sent to the models, results from three runs, a trace of every failure, and the questions I want answered.

Please be critical. I want gaps, wrong assumptions, cheaper or more reliable alternatives, and concrete improvements, not a summary.

---

## 0. What I am asking you to do

1. Read sections 1 to 8 (what it is, how it works, the data, the results).
2. Review the design and the prompts (sections 3 and 9).
3. Answer the questions in section 10, in the format given there.

---

## 1. The product

**An AI data room investigator.** A "data room" is a folder of documents that a company shares during due diligence: reports, presentations, contracts, spreadsheets. The system reads every document and finds discrepancies between them (for example, one document says capacity is 20,000 and another says 24,000), then reports each one with exact quotes, the document and page or cell, a classification and a severity.

**Hard requirements**
- **Domain-agnostic.** It must work on any data room (finance, legal, manufacturing, anything). No rule may depend on industry words. The test data room happens to be about a semiconductor plant, but nothing in the code or prompts may rely on that.
- **Verifiable.** Every reported finding must carry exact quotes that exist in the source text. No invented numbers, no invented quotes. Arithmetic and comparison of numbers and dates are done in code, never by the model.
- **Complete.** All five discrepancy types must be reported, including agreements ("consistent") and missing information.
- **Cost-aware.** All model calls go to one provider (LlamaExtract, a document extraction API from LlamaIndex Cloud) on its cheapest tier. The free plan gives about 10,000 credits a month. A full run costs about 1,200 credits, so about 8 runs a month.
- **No secrets in code.** Keys come from environment variables only. Model names are fixed in code.

**Five finding classes** (what a reader sees)
| Class | Meaning |
|---|---|
| `contradiction` | The two claims cannot both be true in the same context, and nothing quoted explains the difference. |
| `contextual_difference` | The values differ, but a quoted difference in period, scope, condition, definition, version or as-of date fully accounts for it. |
| `missing_information` | Something needed to compare or support a claim is absent from the data room (for example a referenced annex that is not included). |
| `insufficient_evidence` | The excerpts cannot settle it (vague wording, unclear format, a currency with no stated rate, readings that disagree). |
| `consistent` | The claims agree. Saved as a finding so a reader can see what matched. |

A sixth outcome, **not comparable** (the two claims are about different things), is counted and discarded; it is never a finding.

Severity is one of critical, high, medium, low, informational. Rules: consistent is informational; insufficient evidence is low; contextual difference is low (medium if the explaining context appears in one document only); missing information is medium (high if another figure depends on it); contradiction is critical when the gap is large (5% or more, or any identifier, name, yes/no or date that cannot both be true) and the claim is prominent (title, summary, total, definition, obligation, signature block), high when either, low when the gap is under 1%, otherwise medium.

---

## 2. System overview

**Stack.** Backend: Python, FastAPI, Pydantic, SQLAlchemy, PostgreSQL 16 with pgvector. Frontend: React, TypeScript, Vite, MUI. Models: LlamaExtract (REST) for everything that needs a language model; a small local embedding model (MiniLM, 384 dimensions, runs on the machine, free).

**How a run happens.** A UI page ("Test run") shows the data room's files as a tree with checkboxes, a Start and Stop button, and a live log. Starting a run launches a separate process (`scripts/ledger_run.py`) with the selected files. The process runs the whole pipeline and writes a text report and a metrics JSON file. The "Claims" page shows the claims, the candidate pairs, the discrepancies (findings) and the run metrics. A run on a selection of files scans, parses and reads only those files; the others are not touched.

**Database tables** (Postgres)
- `documents` (id, filename, file_hash unique, file_type, processing_status, ...)
- `document_chunks` (id, document_id, chunk_index, content, content_type paragraph/table/table_row, page_number, sheet_name, cell_reference, heading, embedding vector(384))
- `claims` (id, document_id, chunk_id, subject, metric, value, unit, period, conditions, assumptions, claim_text, claim_category, extraction_confidence). `claim_category` holds the claim's value type.
- `findings` (id, investigation_id, title, description, classification, severity, confidence, evidence_status, verification_status, materiality_reasoning, ...)
- `evidence` (id, finding_id, chunk_id required, claim_id optional, evidence_role, quoted_text, relevance_reason, support_assessment): every quote points to a stored chunk.
- `finding_claims`, `investigations`, `investigation_events`.

**Code map** (all under `server/`)
| Step | File |
|---|---|
| Run orchestration and report | `scripts/ledger_run.py` |
| Scan (hashing, registering files) | `app/services/document_service.py` |
| Parsing | `app/services/extractors/pdf.py`, `docx.py`, `xlsx.py` |
| Chunking | `app/services/chunking.py` |
| Processing (parse, chunk, embed, save) | `app/services/processing_service.py` |
| Claim extraction | `app/services/llama_claims.py` |
| Pairing | `app/services/llm_pairing.py`, `app/services/candidate_matching.py` |
| Value comparison rules | `app/services/value_rules.py`, `app/services/numbers.py` |
| Judge | `app/services/pair_judge.py` |
| Quote verification and saving findings | `app/tools/submit.py` |
| Missing-information check | `app/services/missing_items.py` |
| LlamaExtract client | `app/llm/llama_extract.py`, `app/llm/llama_llm.py` |

An earlier design used an autonomous agent that searched the documents with tools. It was abandoned for this claim-ledger pipeline because it was slower, costlier and harder to verify. That code still exists but is not used.

---

## 3. The pipeline, step by step

### 3.1 Scan
Every file in the data room folder is hashed (SHA-256) and registered as a document. The hash is the identity: renaming a file keeps its record; changing its content makes a new document; an unchanged hash means the file is not re-read. Supported: PDF, DOCX, XLSX. When a run is given a selection of files, only those files are scanned, parsed and read.

### 3.2 Parse and chunk (code, no model)
- **PDF** (PyMuPDF): text blocks with their page number; tables are detected and read cell by cell from each cell's own rectangle (an earlier version read table text from the table extractor, which dropped spaces and decimal points, for example "20000waferstartspermonth", and caused real misses).
- **DOCX** (python-docx): paragraphs and table rows in reading order, with the nearest heading.
- **XLSX** (openpyxl): every cell of every sheet; each row becomes one line like `[Installed_Tools!A13:F13] Tool_ID: MET-004; Tool_Category: Metrology; ...; Installed_Cost_INR_Crore: 26; ...` with its sheet and cell range. Formulas are read but never calculated.
- **Chunks** are at most 1,000 characters. Table rows are never merged into a paragraph chunk. A chunk keeps its type, page, sheet, cell range and heading.
- **Checked:** each of the 8 test documents was compared with an independent read of the source (every word, every numeric cell). All 8 score 100%: nothing is left out and no garbled text is produced.

### 3.3 Embed (local)
Each chunk gets a 384-number vector from a local MiniLM model and it is stored with the chunk. The current pipeline does not read chunk vectors (they were built for an earlier retrieval design). Claim text is embedded separately, on the fly, for pairing (3.6).

### 3.4 Extract claims (LlamaExtract, one job per document)
LlamaExtract receives the original file and a JSON schema, plus a system prompt (section 9.1). It returns a list of **claims**, each with:
- `subject` (what the value is about; for a table row, the row's identifying label such as an ID), `metric`, `value` (exactly as written), `unit`, `period`, `conditions`, `claim_text` (the source sentence), and
- `value_type`, exactly one of: `quantity` (a number with or without unit, currency, percentage, duration, range, bound), `date`, `identifier` (codes, certificate, order and phone numbers), `name`, `state` (yes/no, status, quantified assertions), `statement`.

It also returns **references**: every document, annex, schedule, exhibit, attachment, approval or other item the text says exists elsewhere, with the sentence that mentions it. This is used by the missing-information check (3.9).

**Attaching a claim to a stored chunk** (code). The service cites a page and text for each claim. Code finds the chunk that contains the claim's value: the value must appear as a whole standalone number (so "26" is not found inside "2026-03-14") or as the string; when several chunks hold it, the cited page is preferred, then the chunk that also contains the claim's subject, then the one containing the cited or claim sentence. A claim whose value is in no chunk is rejected and counted. Nothing unverified is stored.

Cost: 78 credits for all 8 documents.

### 3.5 Pair claims
The goal is to find which claims should be compared. Comparing every claim with every other would be 264 x 263 / 2 = about 34,700 pairs, so candidates come from three sources:

1. **LLM grouping.** All claims are sorted by metric words, then sent in batches of up to 60,000 characters (2 batches for 264 claims). The model groups claims that state the same attribute of the same subject (prompt in 9.2). Code then splits each group by distinct value (same value, unit and period = same cluster) and compares one claim per cluster with one per other cluster. Claims in different batches are never grouped together.
2. **Nearest neighbours.** Every claim is embedded as "subject: metric: claim sentence"; each claim is paired with its top 2 most similar claims in other documents, when cosine similarity is at least 0.72. This catches claims worded differently.
3. A word-overlap route exists but is not used in the default run.

**Gates** that remove pairs: rows of the same table are never paired (they are different items sharing column headings); a pair that includes an identifier claim is kept only if the two subjects overlap (token Jaccard at least 0.5, or one has no subject), because identifiers of two different things (two tools, two orders) do not conflict. The claim type is shown to the grouping model but does not stop two claims being paired (the same fact can be written as a number in one place and a sentence in another).

Each pair gets a **relation hint** from code (3.6): `exact` (values agree), `conflicting` (values differ), `contextual` (values differ and conditions or period differ), `partial` (units differ or a value is missing).

### 3.6 Compare values in code (`value_rules`)
For each pair code answers `equal`, `different` or `unknown`:

| Type | Rule | Examples |
|---|---|---|
| quantity | A value is an interval from its written precision ("10.2" is 10.15 to 10.25; "500" is 499.5 to 500.5, so 500 vs 512 differ). "about/around/approximately" widens by 10%. Bounds: "more than 500" is an open interval; two different bounds are `unknown`. A range equals only a range with the same ends, never a single number. Scale words (thousand, million, billion, lakh, crore) in the value or unit multiply. Units of one dimension convert (kg, tonne; metres; watts; months and years); other units are compared as written. A bracketed amount is negative. | "10.2 million" = "10,213,000" (equal); "about 10,000" vs 9,640 (equal); "more than 500" vs 612 (equal); "1.5 t" = "1,500 kg" (equal); 412 vs 388 (different); "36 months" = "3 years" (equal) |
| date | Parsed to an interval at its stated precision: day, month, quarter, half, year. Day/month order that cannot be told apart ("04/05/2024") is `unknown`, never guessed. A bare year counts as a date only when both claims are typed date. Fiscal periods are not guessed. | "1 July 2024" vs "Q3 2024" (equal); "14 March 2023" vs "2023-04-14" (different); "04/05/2024" vs "4 May 2024" (unknown) |
| identifier | Letters and digits must be identical; separators and case are ignored; never fuzzy; never read as a number. An identifier compared with another type is different. | "SF-65-A" = "sf 65 a" (equal); "0481-2217" vs "04812271" (different) |
| name | Equal after case, accents, punctuation and legal-form words (Ltd, Limited, Inc, GmbH...) are removed. | "ACME Ltd" = "Acme Limited"; "ACME Holdings Ltd" vs "ACME Ltd" (different) |
| state, statement | Equal only if the normalised text is identical; anything that needs meaning is left to the judge. | |

Limits: the words for approximation and bounds, and month names, are English. Number words ("three-year") are not parsed.

**Pairs whose values are `equal` are agreements.** They are never sent to the judge. Each agreement between two different documents becomes a `consistent` finding with a verified quote from each document (no model call).

### 3.7 Judge (LlamaExtract)
Every pair whose values differ goes to the judge, in batches sized by characters (up to 12,000 characters of excerpts per call; calls run one after another). For each pair the model sees (template in 9.3):
- a **Code check** line from step 3.6: "the values agree within their stated precision", or "the values differ (gap 5.8% of the larger value)", or "could not be compared by code";
- Claim A and Claim B (file, location, the sentence, subject, metric, value, unit, period, conditions) and the **full text of each claim's chunk**.

The model returns one verdict per pair: `same_attribute` (false = not comparable, discarded), class, severity, confidence, title, description, materiality reasoning, and one exact quote from each excerpt. The prompt gives a **decision order** (different attribute; consistent; contextual vs contradiction; missing information; insufficient evidence), the severity rules (section 1), and per-type rules (currency, percent vs percentage points, fiscal periods, relative dates, identifiers, names, yes/no polarity, statements).

### 3.8 Verify and save (code)
A verdict becomes a finding only if all of these pass (otherwise it is rejected, counted, and the reason is listed in the report):
1. each quote is an exact substring of its stored chunk after normalising Unicode forms, curly quotes, dash variants, zero-width characters and line-break hyphenation; the stored quote is always the source's own text;
2. each quote contains its claim's value (numbers compared as numbers; swapped quote order is tolerated and corrected);
3. for a claim read from a table, the quote comes from the claim's own row: a quote from a neighbouring row of the same chunk is rejected; a quote from the right row that leaves out the row label is widened to the whole row (when the claim's label appears in the chunk);
4. contradiction, contextual difference and consistent findings need two different quoted passages;
5. duplicates (same two quotes in the same two chunks) are skipped.
Consistent and insufficient-evidence verdicts are saved only when they carry both quotes.

### 3.9 Missing information from a single claim (code)
Each reference from step 3.4 is looked up in: every file name, every sheet name, every heading and the first line of every chunk (excluding the chunk that makes the reference). If it is nowhere, and every document in the room was fully processed, a `missing_information` finding is saved with the referencing sentence as its quote. If any document failed, is image-only or is still pending, the check stays silent and says why.

### 3.10 Report
A text report (findings with evidence, every candidate pair and what happened to it, per-phase model timing and credits, rejection reasons, timing per stage) and a metrics JSON that the UI shows.

---

## 4. The test data room (complete content)
It is synthetic and was seeded with discrepancies. Parsed chunks, exactly as the pipeline sees them, follow in Appendix A. Summary:

| # | File | Type | Content |
|---|---|---|---|
| 01 | Corporate_Overview.pdf | PDF, 1 page | company profile, facility table (capacity, area, status), ownership, limitations |
| 02 | Manufacturing_Operations_Report.pdf | PDF, 1 page | production table (wafer starts, yield), operations notes |
| 03 | Quality_and_Compliance_Report.pdf | PDF, 1 page | ISO readiness assessment, certification status, permits, audit date |
| 04 | Investor_Presentation.docx | DOCX | highlights, operating performance, certifications |
| 05 | Customer_Supply_Agreement.docx | DOCX | parties, forecast, qualification, delivery, schedules |
| 06 | Facility_and_Utilities_Plan.docx | DOCX | electrical and water demand, permits |
| 07 | Equipment_Register.xlsx | XLSX, 2 sheets | 12 installed tools with installed cost; 3 purchase orders |
| 08 | Financial_and_Capacity_Model.xlsx | XLSX, 3 sheets | capacity assumptions; the same 12 tools with budgeted cost; customer revenue plan |

29 chunks in total. The reader should treat the content as unknown domain data.

---

## 5. Ground truth: the discrepancies that were seeded

| # | Seeded item | Where |
|---|---|---|
| S1 | Yield: 92.4% vs 96.8% | 02 (table row "Aggregate yield") vs 04 and 08 |
| S2 | Designed nameplate capacity: 20,000 vs 24,000 WSPM | 01 vs 04 and 08 |
| S3 | Tool MET-004 cost: 26 (installed cost, file 07) vs 29 (budgeted cost, file 08) | 07 vs 08 |
| S4 | Customer forecast: 8,000 wafers per month (agreement, file 05) vs 10,000 (revenue plan, file 08 and deck 04) | 05 vs 08 |
| S5 | ISO 9001:2015 "certified" (deck, file 04) vs "no certificate had been received" (quality report, file 03) | 04 vs 03 |
| S6 | Schedule A (technical specification) referenced but not attached | 05 |

(Other differences exist in the data, for example installed vs qualified capacity, which are contextual rather than errors.)

---

## 6. Results

### 6.1 Three runs of the whole pipeline (all 8 documents)
| | Run A (first LLM pipeline) | Run B (after type, rules, missing-info work) | Run C (after row check and identifier gate) |
|---|---|---|---|
| Claims extracted | 204 | 264 | 264 |
| Claims rejected at extraction | | 1 | 1 |
| Candidate pairs | 145 | 180 | 183 |
| Agreements (values equal, no model call) | 59 | 59 | 59 |
| Pairs judged by the model | 86 | 121 | 124 |
| Verdicts: not comparable | 56 | 83 | 79 |
| Verdicts: contradiction | 1 | 7 | 4 |
| Verdicts: consistent | 17 | 24 | 28 |
| Verdicts: insufficient evidence | 7 | 5 | 8 |
| Verdicts: contextual difference | 1 | 0 | 1 |
| Pairs with no verdict returned | | | 4 |
| Findings saved | 24 | 38 | 40 |
| of which contradiction | 1 | 5 (2 false) | 3 |
| of which contextual difference | 1 | 0 | 1 |
| of which insufficient evidence | 0 | 1 | 0 |
| of which missing information | 0 | 1 | 1 |
| of which consistent | 22 | 31 | 35 |
| Findings rejected by verification | 20 | 17 | 13 |
| Wall time | 13 min 32 s | 9 min 12 s | 3 min 39 s |
| Model calls | 16 | 24 | 25 |
| LlamaExtract credits | 883 | 1,218 | 1,208 |
| Credits: extraction | 78 | 78 | 78 |
| Credits: grouping and judging | 805 | 1,140 | 1,130 |

Run C timing: scan 0 s, extraction 9.5 s, grouping 6.4 s, judging and verification 203 s (93% of the wall time, because judge calls run one after another), missing-information check 0 s.

Claim types in runs B and C (264 claims, none untyped): quantity 74, statement 65, state 51, identifier 34, date 30, name 10.

Other facts (run B): of 180 pairs, 45 came from the grouping route and 135 from nearest neighbours; relation hints were 59 exact, 15 contextual, 80 conflicting, 26 partial. Token counts are not reported by LlamaExtract.

### 6.2 Seeded items, run by run
| Item | Run A | Run B | Run C (latest) |
|---|---|---|---|
| S1 yield | missed (PDF table parsing bug had garbled the cell) | found | found, correct evidence |
| S2 capacity | missed (garbled cell) | found twice | found once, correct |
| S3 MET-004 | missed | missed | reported as a contradiction, but it is a false positive: it compares MET-004 with the MET-003 row, and the correct pair was discarded as not comparable (see 7.1). Real recall of the latest run is 3 of 6, and contradiction precision 2 of 3. |
| S4 customer volume | missed | missed | missed (see 7.3) |
| S5 ISO | found once as a contradiction (deck vs quality report) | missed | missed (see 7.2) |
| S6 Schedule A | not checked | found | found |

---

## 7. Trace of every miss and false positive in the latest run

### 7.1 S3, MET-004 cost 26 vs 29: found with wrong evidence
The pair log shows the correct pair (08 MET-004 budgeted cost 29 vs 07 MET-004 installed cost 26) judged **"not comparable"**, with the model saying claim B was the installed cost of MET-003. The cause was a chunk attachment bug: the 07 claim "MET-004 = 26" was attached to the chunk holding the MET-003 row because the value check used substring matching, and "26" occurs inside "2026-03-14". The model therefore read the wrong excerpt. A different, wrong pair (MET-004 29 vs MET-003 28) was saved as a contradiction titled "MET-004 budgeted cost: 29 versus 26", with the MET-003 row as evidence.
**Fixed** afterwards (whole-number matching; prefer the chunk that contains the claim's subject), tested offline, **not yet re-run on the full pipeline**. Open question even after the fix: the judge may still call "installed cost" and "budgeted cost" different attributes. A human reviewer would want them flagged.

### 7.2 S5, ISO: never paired
The two relevant claims are (04) "SiliconForge | ISO certification status | certified" and (03) "ISO 9001:2015 certificate | received status | No". Each is the other's closest match among claims of other documents (rank 1), with cosine similarity 0.618 to 0.663. The cut-off for nearest-neighbour pairing is 0.72, so the pair was never made, and the grouping model did not group them either. (In run A they were paired and the discrepancy was found.)

### 7.3 S4, customer volume: never paired
The claims are (05) "Customer rolling forecast | quantity | 8,000 wafers per month" and (08) "Northstar Controls GmbH | Monthly_Volume | 10000 Wafers". Cosine similarity is 0.39 to 0.46 (rank 9 to 20 among other-document claims), far below the cut-off, because the subjects and wording differ. The grouping model probably never saw both claims in one batch (264 claims were split into 2 batches, and claims in different batches are never grouped together), but the report cannot confirm this.

### 7.4 False positives seen earlier and since fixed
- "CLEAN-011 cost 19 vs 24" (run B): the quote with 24 came from a different row (TEST-021) of the same chunk, and the value check passed because 24 appeared in the quote. Fixed by the row check (3.8, rule 3).
- "Tool_ID differs: MET-004 vs MET-003" (run B): two different tools' identifiers were paired and called a contradiction. Fixed by the identifier subject gate (3.5).
- One contextual difference in run C ("9,200 average monthly starts vs 27,600 wafers") is arguably `consistent` after converting a quarterly total (27,600 / 3 = 9,200). The judge called it low-severity contextual.

### 7.5 Other observations
- 79 of 120 judged pairs are "not comparable". Many come from label columns of spreadsheets (category, status, ID) that produce claims whose "value" is just the row's own label; these generate many meaningless pairs, and many of the 35 consistent findings are of this kind (for example "Tool_Category: Lithography is the same in both files").
- 4 pairs received no verdict from the model, including two of the pairs about capacity (20,000 vs 24,000).
- In run B (17 rejected findings), most rejections were consistent findings where the model quoted a neighbouring cell or row instead of the claim's own; run C had 13 rejections (their reasons are in Appendix D).

---

## 8. Changes made so far, in order
1. Rows of one table are never paired.
2. All five classes are saved as findings; equal values across documents become consistent findings without a model call.
3. PDF table cells read per cell (fixed garbled text).
4. Nearest-neighbour pairing added (catches differently worded claims).
5. Strict output-format blocks in all prompts.
6. Six claim value types (replacing a first version with three) assigned at extraction.
7. Code value rules (section 3.6) wired into pair hints; judge sees a "Code check" line.
8. Judge prompt rewritten with a decision order, severity table and per-type rules.
9. Quote normalisation (Unicode, quotes, dashes, hyphenation).
10. Single-claim missing-information check (references).
11. Row check for table quotes; a pair log in the report listing every pair and its outcome.
12. Identifier gate (subject must overlap).
13. Whole-number value matching and subject-aware chunk choice.
14. A run now touches only the selected files.

**Proposed, not done:** (B) lower the nearest-neighbour cut-off from 0.72 to about 0.6 (would pair ISO; adds pairs and cost); (C) one grouping batch or overlapping batches (would let the customer volume claims meet; one large grouping call took 300 s and many credits in an early run); (D) retry pairs that returned no verdict, once; skip label-only claims at extraction; run judge calls in parallel (would cut time but needs thread-safe clients).

**Known weaknesses**
- Pairing recall is the main gap: a fixed similarity cut-off, only 2 neighbours per claim, and a grouping model that cannot see across batches.
- Judging is 93% of run time and most credits, and two thirds of judged pairs are discarded as not comparable.
- Code rules cover English wording only; currency conversion, percent vs absolute, fiscal-year mapping, relative-date anchors, locale detection, part-against-whole and yes/no polarity are in the judge prompt only (no code).
- The row check is skipped when the claim's label does not appear in the chunk.
- Extractor fields such as qualifier, scale, sign and as-of date would need new database columns; not decided.
- The missing-information check can flag an item as missing when the room names it only in body text, not in a file name, sheet name, heading or first line.
- Only one small synthetic data room has been used; there is no labelled test set.

---

## 9. The prompts sent to the model (verbatim)

### 9.0 How a call reaches the model

For grouping and judging, the user prompt is written to a small text file and uploaded to LlamaExtract; the system prompt is sent as the extraction instructions; the Pydantic output model is sent as the JSON schema. Claim extraction uploads the original document itself. All prompts below are verbatim from the code. Prompts 4 to 7 of the code (a per-document verdict, an older per-chunk extractor, an older coverage check, and the abandoned agent) are not used by the default run and are omitted.

### 9.1 Claim extraction (default run)

**Used:** once per selected document, LlamaExtract reads the file itself

**System prompt (verbatim)**

```text
Extract every claim that states a value (a number, quantity, amount, date, duration, name, yes/no, status or condition) so it can be compared with other documents, including values in lists, tables and short fragments. If the same quantity is stated more than once, extract each mention. For a table, the subject is the row's identifying label (its ID or name), never the table or sheet name, and the metric is the column heading. Copy values and sentences exactly as written; never calculate or infer. OUTPUT FORMAT (strict): every claim has non-empty string fields subject, metric, value and claim_text, and a value_type that is exactly one of quantity, date, identifier, name, state or statement. quantity: a numeric amount with or without a unit: a count, measure, currency amount, percentage, ratio, score, duration, range or bound (at least, up to, about), even when written inside brackets or quotation marks. Arithmetic on it is meaningful. Not a number used as a label (codes, phone numbers, version numbers). date: a point in time or a calendar or fiscal period (day, month, quarter, half, year, time of day), including a relative date anchored to an event. A year is a date when it says when something happened, not how many. A duration is a quantity. identifier: a string that labels exactly one thing and is compared character by character: a registration, certificate, account, order, version or phone number, a code or serial number, even if all digits. Arithmetic on it is meaningless and one changed character means a different thing. name: the proper name of a person, organisation, place, product or document title. A generic description (the company, the supplier) is not a name. state: a yes/no fact or a status from a small closed set (active, expired, pending), including quantified assertions (all, none, some, most). statement: any other assertion in words: a description, obligation, reason, list or policy. No other value_type is allowed. Add unit, period and conditions only when the document states them; never return a claim without a value, and never invent other fields. Also return references: each document, annex, schedule, exhibit, attachment, table, report, approval or other item that the text says exists elsewhere, is attached, is included or is still outstanding, with item (its name as written) and quote (the sentence that refers to it, copied exactly). Do not list a section of the same document, and return an empty references list when there are none.
```

**Output schema sent with the extraction request**

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
      "description": "Every document, annex, schedule, exhibit, attachment, table, report, approval or other item that the text says exists elsewhere, is attached, is included or is still outstanding. Not a section of this same document.",
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

### 9.2 Claim grouping (default run, `--pairing llm`)

**Used:** one call per batch of sorted claims (one batch = up to 60,000 characters)

**System prompt (verbatim)**

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

**User prompt (template)**

```text
Claims:

<one line per claim>

Each line: c<N> | <file>, <page or sheet and cells> | type: <type or -> | subject: <..> | metric: <..> | value: <value unit> [| period: <..>] | <claim sentence>
```

### 9.3 Pair judging (default run)

**Used:** one call per batch of pairs whose values differ (batches sized by characters)

**System prompt (verbatim)**

```text
You compare pairs of claims taken from documents of the same collection and decide whether each pair is a real discrepancy.

Rules:
1. Use ONLY the claims and excerpts shown. Never invent text, values or context.
2. Compare like with like: same subject, metric, unit, period and scope. Decide from what the two excerpts themselves state. Under each pair a "Code check" line says whether code found the two values to agree within their stated precision (scale words, units of one kind and rounding already allowed for), to differ (with the gap), or could not compare them. Trust it for the numbers; you decide what the difference means.
3. DECISION ORDER for classification (stop at the first step that applies):
   a. Different subject or metric, or explicit periods or scopes that neither overlap nor nest: same_attribute is false (rule 6). Shared words, digits or phrasing do not make claims comparable.
   b. consistent: the values agree under every reasonable reading (after units, scale, sign, format and spelling are normalised). If the readings split, it is not consistent.
   c. The values conflict under every reasonable reading. contextual_difference: the excerpts themselves state a distinguishing context (a different period, scope, condition, definition, version or as-of date) that fully accounts for the gap; name it in the description. contradiction: no such context is stated, or the stated context cannot explain the gap (for example a part larger than its whole). Two documents or editions giving different values for the same thing is a contradiction even if you can imagine a reason; never invent an explanation that is not in the excerpts ("may include tax" is not a quote).
   d. missing_information: a specific item needed to decide is absent from the excerpts (a referenced document, annex or table, the base of a percentage, an anchor date, a fiscal year end). Name the item in the description.
   e. insufficient_evidence: anything else the excerpts cannot settle: a vague or hedged value, an unclear format, a currency without a stated rate, or readings that disagree about the outcome. Say what blocks the test.
4. For every verdict whose same_attribute is true (consistent and insufficient_evidence too), copy quote_a EXACTLY from excerpt A and quote_b EXACTLY from excerpt B: the shortest passage that shows the value together with the words or unit next to it (at least 8 characters, e.g. "185W/m² power supply", never the bare number).
5. quote_a must contain the value of claim A and quote_b the value of claim B, exactly as written in the excerpt. For a claim read from a table, quote that claim's own row including its label (its ID or name), never a neighbouring row that happens to show the same number.
6. same_attribute: true only if both claims state the same attribute of the same subject, so that their values could be compared. false if they measure different things (for example a height and a floor level, or a distance and a duration), even when their units or numbers look alike. Add attribute_reasoning: one short sentence saying what each claim measures. When same_attribute is false the pair is not compared and the classification is ignored.
7. For every verdict whose same_attribute is true also fill in: title (one line saying what differs, e.g. the thing and the two values), description (both values, which document states each and where, and why they conflict or what context separates them), severity (see SEVERITY below), confidence (0 to 1) and materiality_reasoning (one sentence).
8. Never do arithmetic. State numbers as written. The gap in the Code check line is computed by code.
9. Return exactly one verdict per pair, using the pair_index shown (see the output format below). Both excerpts may be the same passage: a passage can contradict itself.

SEVERITY (choose it after the classification): consistent is informational. insufficient_evidence is low. contextual_difference is low, or medium when the explaining context appears in only one document or only in a note. missing_information is medium, or high when another figure or obligation depends on the missing item. contradiction is critical when the gap is large (5% or more, or any identifier, name, yes/no or date that cannot both be true) AND the claim is prominent (a title, summary, total, definition, obligation or signature block); high when it is large OR prominent; low when the gap is under 1%; otherwise medium.

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
- title, description, materiality_reasoning, quote_a, quote_b: strings. Use an empty string, never null, when there is nothing to put.
Never omit a field, never use a value outside the lists above, and never add other fields, comments or markdown.
```

**User prompt (template)**

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

### 9.4 A worked example: one pair through the judge
**Input for one pair** (what the model receives):
```text
### Pair 0
Code check: the values differ (gap 4.5% of the larger value).
Claim A (02_Manufacturing_Operations_Report.pdf, page 1): Aggregate yield | 92.4% | Good wafers out / wafer starts | subject=Aggregate yield; metric=Q3 2026 value; value=92.4; unit=%
Excerpt A:
Metric | Q3 2026 value | Basis
Wafer starts | 27,600 wafers | MES production log export
Average monthly wafer starts | 9,200 WSPM | 27,600 / 3 months
Good wafers out | 25,510 wafers | Final electrical test records
Aggregate yield | 92.4% | Good wafers out / wafer starts
Process node | 65 nm | Production routing

Claim B (04_Investor_Presentation.docx, section Operating performance): Current production yield: 96.8% on SF-65-A during Q3 2026. | subject=...; metric=...; value=96.8; unit=%
Excerpt B:
Operating performance
Current production yield: 96.8% on SF-65-A during Q3 2026.
Current installed nameplate capacity: 24,000 WSPM.
Management expects monthly shipments to reach 10,000 wafers by Q2 2027.
```
(The two claim lines are shortened here; the real ones also carry period and conditions when present.)

**Output the model returned for it, saved after verification** (real finding from the latest run):
```text
--- FINDING 24 of 40  [fnd_e78645be0d474be6] ------------------------------
title                 : Q3 2026 production yield: 92.4% vs 96.8%
classification        : contradiction
severity              : high
confidence            : 0.97
evidence_status       : adequate
verification_status   : verified
description           : The Manufacturing Operations Report states an aggregate yield of 92.4% for Q3 2026 in its production summary table. The Investor Presentation states a current production yield of 96.8% on SF-65-A during Q3 2026. The excerpts do not state a context that accounts for the differing values.
materiality_reasoning : The difference exceeds 1% and concerns a prominent operating performance metric.
business_impact       : -
unresolved_questions  : -
evidence:
  [1] role=contradicting
      document : 02_Manufacturing_Operations_Report.pdf
      chunk    : chk_52323f74df704f8c   (chunk_index 0)
      location : page 1
      quote    : "Aggregate yield | 92.4% | Good wafers out / wafer starts"
      relevance: | Aggregate yield              | 92.4%         | Good wafers out / wafer starts |
      assessment: 02_Manufacturing_Operations_Report.pdf states: | Aggregate yield              | 92.4%         | Good wafers out / wafer starts | | subject=Aggregate yield; metric=Q3 2026 value; value=92.4%; unit=%; period=Q3 2026
  [2] role=contradicting
      document : 04_Investor_Presentation.docx
      chunk    : chk_60861b5bbeb24d94   (chunk_index 2)
      location : heading Operating performance
      quote    : "Current production yield: 96.8% on SF-65-A during Q3 2026."
      relevance: Current production yield: 96.8% on SF-65-A during Q3 2026.
      assessment: 04_Investor_Presentation.docx states: Current production yield: 96.8% on SF-65-A during Q3 2026. | subject=SiliconForge; metric=current production yield; value=96.8%; unit=%; period=Q3 2026; conditions=on SF-65-A
```

---

## 10. Questions for you
Answer each, with concrete proposals. Where relevant, say what it would cost (model calls) and how you would test it.

1. **Architecture.** Is claim extraction, then pairing, then code comparison, then judging, then verification the right design? Is there a cheaper way to get high recall than comparing candidate pairs, for example extracting claims into a normalised ledger keyed by (subject, metric, period) and detecting conflicts by key? What would break in that approach?
2. **Pairing recall.** Two claims worded very differently ("Customer rolling forecast ... 8,000 wafers per month" vs "Northstar Controls GmbH ... Monthly_Volume 10000") are never paired, and a claim pair with similarity 0.62 to 0.66 (ISO) is below the 0.72 cut-off. How should pairing be made reliable and domain-agnostic? Consider cut-off tuning, more neighbours, a second embedding view (value plus unit plus period), a learned or LLM reranker, grouping all claims in one call, or having the extractor emit a normalised attribute key.
3. **Judge reliability.** Is "installed cost" vs "budgeted cost" of the same item a discrepancy to flag, or a different attribute? Where should the line be, and how should the prompt say it without domain words? About two thirds of judged pairs are discarded as not comparable: how do we avoid creating those pairs in the first place?
4. **Category definitions.** Review the five classes, the decision order and the severity rules in sections 1 and 9.3. Which cases are ambiguous? Give pairs that a model could label two ways, and tightened wording.
5. **Value rules.** Review section 3.6. Is interval overlap with a 10% "about" band reasonable? What should be added or removed? What should move from the prompt into code (currency, percent vs percentage points, fiscal periods, relative dates, polarity)?
6. **Verification.** We verify only by exact-quote substring (after normalisation), value presence, row membership and two distinct passages. What can still get through? What cannot be verified this way?
7. **Missing information.** Is the reference-lookup design (file names, sheet names, headings, first lines) sound? How would you reduce false "missing" without missing real ones?
8. **Evaluation.** There is one small synthetic data room and six seeded discrepancies. Design a test set and metrics (recall of seeded discrepancies, false-positive rate, cost per run, stability across runs). How many data rooms and what kinds of seeded errors?
9. **Cost and time.** Judging is 93% of run time and most credits (about 1,130 of 1,208). Suggest ways to cut it without losing recall (cheap pre-filter, parallel calls, caching verdicts for unchanged pairs).
10. **Anything we are missing.** Failure modes, risks or simpler alternatives we have not considered.

**Answer format:** (a) the three biggest problems you see, in order; (b) your answers to the ten questions; (c) a prioritised list of concrete changes with expected effect and cost; (d) anything in this document that looks wrong or inconsistent.

---

## Appendix A. The data room as the pipeline sees it (all 29 chunks)

Each chunk shows its type and location, then its exact content. Table rows are one line each, prefixed with their sheet and cell range.

### 01_Corporate_Overview.pdf

**chunk 0** (table; page 1)
```text
Item | Management-reported position
Facility status | Pilot production; commercial ramp in progress
Installed cleanroom area | 18,500 m²
Designed nameplate capacity | 20,000 wafer starts per month (WSPM)
Current qualified capacity | 12,000 WSPM as of 30 September 2026
Production format | 300 mm wafers
```

**chunk 1** (paragraph; page 1)
```text
SiliconForge Semiconductors Pvt. Ltd.
Corporate overview | Version 1.2 | 30 September 2026
Company profile
SiliconForge is a fictional Indian semiconductor manufacturer developing a 300 mm wafer fabrication facility in Nanjangud Industrial Area, Mysuru district, Karnataka.
Business focus: mature-node logic and mixed-signal components for industrial, automotive-support and power-management applications. Initial process platform: 65 nm.
Facility facts
Ownership and funding
Paid-up equity capital: INR 620 crore. Cumulative project expenditure: INR 2,480 crore. Both are management-reported and unaudited in this document.
Shareholding: SiliconForge Holdings Pvt. Ltd. 62%; institutional investors 28%; employee and other shareholders 10%.
Limitations
This is a management-prepared summary, not a statutory filing, audited financial statement, production certificate, environmental consent or customer acceptance record.
```

### 02_Manufacturing_Operations_Report.pdf

**chunk 0** (table; page 1)
```text
Metric | Q3 2026 value | Basis
Wafer starts | 27,600 wafers | MES production log export
Average monthly wafer starts | 9,200 WSPM | 27,600 / 3 months
Good wafers out | 25,510 wafers | Final electrical test records
Aggregate yield | 92.4% | Good wafers out / wafer starts
Process node | 65 nm | Production routing
```

**chunk 1** (paragraph; page 1)
```text
Manufacturing Operations Report
Reporting period: 1 July–30 September 2026 | Issued 15 October 2026
Production summary
Line performance
Fab line SF-65-A. Product mix: industrial controllers, power-management ICs and connectivity components.
Yield is calculated at final electrical test as good wafers out divided by wafer starts for the same cohort. Rework lots remain in the original cohort.
Capacity and ramp
Current qualified capacity is 12,000 WSPM. Actual average starts during Q3 2026 were 9,200 WSPM.
Operational target: 16,000 WSPM by Q2 2027, subject to tool qualification, customer qualification and stable process yield.
Open evidence
Independent third-party validation of Q3 yield was not included. Lot-level MES exports, uptime logs and customer acceptance records are available on request but are not attached.
```

### 03_Quality_and_Compliance_Report.pdf

**chunk 0** (table; page 1)
```text
Measure | Status
Customer-reported major escapes in Q3 2026 | 0 reported
Open internal corrective actions | 4 minor actions
Final test yield | 92.4% for Q3 2026; see Operations Report
Traceability | Lot-level traceability maintained in MES, per management
```

**chunk 1** (paragraph; page 1)
```text
Quality and Compliance Status
Internal status report | As of 31 October 2026
Quality management system
ISO 9001:2015 readiness assessment completed on 18 September 2026; four minor findings recorded.
External certification audit is scheduled for 12 December 2026. No ISO 9001:2015 certificate had been received as of this report date.
Product quality
Environmental and safety
Consolidated environmental consent application submitted to Karnataka State Pollution Control Board on 7 August 2026; acknowledgement KSPCB-ACK-26-0817.
No final Consent for Establishment amendment covering the proposed 22 MW connected load was present in records reviewed.
Fire NOC renewal is stated to be in process; renewed certificate was not included.
Limitations
This internal summary does not replace certificates, regulator orders, audit reports, test records or statutory filings.
```

### 04_Investor_Presentation.docx

**chunk 0** (paragraph; heading: SiliconForge | Investor Overview)
```text
SiliconForge | Investor Overview
Management presentation extract | 5 November 2026
```

**chunk 1** (paragraph; heading: Investment highlights)
```text
Investment highlights
SiliconForge is building a 300 mm, 65 nm semiconductor manufacturing platform in Karnataka for industrial, automotive-support and power-management markets.
Commercial ramp is underway, with planned expansion of qualified production capacity.
```

**chunk 2** (paragraph; heading: Operating performance)
```text
Operating performance
Current production yield: 96.8% on SF-65-A during Q3 2026.
Current installed nameplate capacity: 24,000 WSPM.
Management expects monthly shipments to reach 10,000 wafers by Q2 2027.
```

**chunk 3** (paragraph; heading: Certifications and readiness)
```text
Certifications and readiness
SiliconForge is ISO 9001:2015 certified.
Customer qualification is complete for the initial industrial controller product family.
```

**chunk 4** (paragraph; heading: Note)
```text
Note
Figures are management-provided and illustrative; this extract has not been independently audited.
```

### 05_Customer_Supply_Agreement.docx

**chunk 0** (paragraph; heading: Customer Supply Framework Agreement)
```text
Customer Supply Framework Agreement
Executed extract | 22 September 2026 | Agreement SF-CUS-026
```

**chunk 1** (paragraph; heading: Parties and scope)
```text
Parties and scope
Supplier: SiliconForge Semiconductors Pvt. Ltd., Karnataka, India.
Customer: Northstar Controls GmbH, Munich, Germany.
Products: 65 nm industrial controller wafers, 300 mm format, subject to product-specific qualification and agreed specifications.
```

**chunk 2** (paragraph; heading: Forecast and purchase commitments)
```text
Forecast and purchase commitments
Customer rolling forecast for the first six months is 8,000 wafers per month, commencing 1 April 2027.
Forecast is non-binding and for capacity planning only. Binding quantities arise only through purchase orders accepted by Supplier.
```

**chunk 3** (paragraph; heading: Qualification and delivery)
```text
Qualification and delivery
Initial product qualification is in progress. First engineering-lot review targeted for 15 January 2027.
Commercial deliveries may begin only after written customer product qualification and acceptance of the relevant purchase order.
Delivery: FCA SiliconForge facility, Incoterms 2020. Payment: net 45 days from invoice date.
```

**chunk 4** (paragraph; heading: Schedules and evidence)
```text
Schedules and evidence
Schedule A (wafer technical specification) is referenced but not attached to this extract.
No accepted purchase order or customer qualification approval is included in this data room.
```

### 06_Facility_and_Utilities_Plan.docx

**chunk 0** (paragraph; heading: Facility and Utilities Plan)
```text
Facility and Utilities Plan
Engineering planning basis | Revision C | 28 October 2026
```

**chunk 1** (paragraph; heading: Facility design basis)
```text
Facility design basis
Facility designed for 300 mm wafer manufacturing using a 65 nm process platform; cleanroom and support areas are being commissioned in phases.
```

**chunk 2** (paragraph; heading: Electrical demand)
```text
Electrical demand
Design connected electrical load: 22 MW at full planned equipment installation.
Initial commissioning demand: 14 MW. Full-load demand includes process tools, HVAC, UPW, abatement, compressed air and site utilities.
Utility design assumes a dedicated 22 MW supply connection. Final sanctioned-load approval was not included.
```

**chunk 3** (paragraph; heading: Water and waste utilities)
```text
Water and waste utilities
Projected average water demand at full ramp: 2,400 m³/day. Wastewater treatment design capacity: 2,100 m³/day.
These are engineering estimates, not measured operating averages. Final commissioning and discharge test reports were not included.
```

**chunk 4** (paragraph; heading: Permits and dependencies)
```text
Permits and dependencies
Final environmental approval, utility connection agreement, sanctioned-load letter and final discharge consent are outstanding evidence items.
```

### 07_Equipment_Register.xlsx

**chunk 0** (table_row; sheet Installed_Tools, cells A1:F5)
```text
[Installed_Tools!A1:F1] A1: Tool_ID; B1: Tool_Category; C1: Process_Area; D1: Status; E1: Installed_Cost_INR_Crore; F1: Commissioning_Date
[Installed_Tools!A2:F2] Tool_ID: LTH-001; Tool_Category: Lithography; Process_Area: Photolithography; Status: Installed; Installed_Cost_INR_Crore: 185; Commissioning_Date: 2026-03-14
[Installed_Tools!A3:F3] Tool_ID: ETCH-014; Tool_Category: Dry etch; Process_Area: Etch; Status: Installed; Installed_Cost_INR_Crore: 42; Commissioning_Date: 2026-04-02
[Installed_Tools!A4:F4] Tool_ID: DEP-008; Tool_Category: CVD deposition; Process_Area: Deposition; Status: Installed; Installed_Cost_INR_Crore: 36; Commissioning_Date: 2026-04-18
[Installed_Tools!A5:F5] Tool_ID: MET-003; Tool_Category: Metrology; Process_Area: Metrology; Status: Installed; Installed_Cost_INR_Crore: 28; Commissioning_Date: 2026-05-09
```

**chunk 1** (table_row; sheet Installed_Tools, cells A6:F10)
```text
[Installed_Tools!A6:F6] Tool_ID: CMP-006; Tool_Category: CMP; Process_Area: Planarization; Status: Installed; Installed_Cost_INR_Crore: 31; Commissioning_Date: 2026-06-11
[Installed_Tools!A7:F7] Tool_ID: IMPL-002; Tool_Category: Ion implantation; Process_Area: Implant; Status: Installed; Installed_Cost_INR_Crore: 54; Commissioning_Date: 2026-07-03
[Installed_Tools!A8:F8] Tool_ID: CLEAN-011; Tool_Category: Wet clean; Process_Area: Wet processing; Status: Installed; Installed_Cost_INR_Crore: 19; Commissioning_Date: 2026-07-22
[Installed_Tools!A9:F9] Tool_ID: TEST-021; Tool_Category: Wafer test; Process_Area: Electrical test; Status: Installed; Installed_Cost_INR_Crore: 24; Commissioning_Date: 2026-08-16
[Installed_Tools!A10:F10] Tool_ID: LTH-002; Tool_Category: Lithography; Process_Area: Photolithography; Status: Installed; Installed_Cost_INR_Crore: 176; Commissioning_Date: 2026-09-01
```

**chunk 2** (table_row; sheet Installed_Tools, cells A11:F13)
```text
[Installed_Tools!A11:F11] Tool_ID: ETCH-015; Tool_Category: Dry etch; Process_Area: Etch; Status: Installed; Installed_Cost_INR_Crore: 39; Commissioning_Date: 2026-09-18
[Installed_Tools!A12:F12] Tool_ID: DEP-009; Tool_Category: CVD deposition; Process_Area: Deposition; Status: Installed; Installed_Cost_INR_Crore: 34; Commissioning_Date: 2026-09-26
[Installed_Tools!A13:F13] Tool_ID: MET-004; Tool_Category: Metrology; Process_Area: Metrology; Status: Installed; Installed_Cost_INR_Crore: 26; Commissioning_Date: 2026-09-30
```

**chunk 3** (table_row; sheet Procurement_Pipeline, cells A1:E4)
```text
[Procurement_Pipeline!A1:E1] A1: PO_Number; B1: Tool_Category; C1: Quantity; D1: PO_Status; E1: Expected_Delivery
[Procurement_Pipeline!A2:E2] PO_Number: PO-2026-101; Tool_Category: Lithography; Quantity: 1; PO_Status: Purchase order issued; Expected_Delivery: 2027-01-15
[Procurement_Pipeline!A3:E3] PO_Number: PO-2026-108; Tool_Category: Etch; Quantity: 2; PO_Status: Quotation under review; Expected_Delivery: 2027-03-01
[Procurement_Pipeline!A4:E4] PO_Number: PO-2026-112; Tool_Category: Inspection; Quantity: 1; PO_Status: Purchase order issued; Expected_Delivery: 2027-02-10
```

### 08_Financial_and_Capacity_Model.xlsx

**chunk 0** (table_row; sheet Capacity_Assumptions, cells A1:D6)
```text
[Capacity_Assumptions!A1:D1] A1: Assumption; B1: Model_Value; C1: Unit; D1: Model_Note
[Capacity_Assumptions!A2:D2] Assumption: Designed nameplate capacity; Model_Value: 24000; Unit: WSPM; Model_Note: Full build-out capacity
[Capacity_Assumptions!A3:D3] Assumption: Current qualified capacity; Model_Value: 12000; Unit: WSPM; Model_Note: As of 30-Sep-2026
[Capacity_Assumptions!A4:D4] Assumption: Q3 2026 average wafer starts; Model_Value: 9200; Unit: WSPM; Model_Note: Average monthly starts
[Capacity_Assumptions!A5:D5] Assumption: Q3 2026 production yield; Model_Value: 96.8; Unit: %; Model_Note: SF-65-A
[Capacity_Assumptions!A6:D6] Assumption: Target monthly shipments by Q2 2027; Model_Value: 10000; Unit: Wafers/month; Model_Note: Management target
```

**chunk 1** (table_row; sheet Tool_Capex_Summary, cells A1:D7)
```text
[Tool_Capex_Summary!A1:D1] A1: Tool_ID; B1: Tool_Category; C1: Budgeted_Cost_INR_Crore; D1: Procurement_Status
[Tool_Capex_Summary!A2:D2] Tool_ID: LTH-001; Tool_Category: Lithography; Budgeted_Cost_INR_Crore: 185; Procurement_Status: Installed
[Tool_Capex_Summary!A3:D3] Tool_ID: ETCH-014; Tool_Category: Dry etch; Budgeted_Cost_INR_Crore: 42; Procurement_Status: Installed
[Tool_Capex_Summary!A4:D4] Tool_ID: DEP-008; Tool_Category: CVD deposition; Budgeted_Cost_INR_Crore: 36; Procurement_Status: Installed
[Tool_Capex_Summary!A5:D5] Tool_ID: MET-003; Tool_Category: Metrology; Budgeted_Cost_INR_Crore: 28; Procurement_Status: Installed
[Tool_Capex_Summary!A6:D6] Tool_ID: CMP-006; Tool_Category: CMP; Budgeted_Cost_INR_Crore: 31; Procurement_Status: Installed
[Tool_Capex_Summary!A7:D7] Tool_ID: IMPL-002; Tool_Category: Ion implantation; Budgeted_Cost_INR_Crore: 54; Procurement_Status: Installed
```

**chunk 2** (table_row; sheet Tool_Capex_Summary, cells A8:D13)
```text
[Tool_Capex_Summary!A8:D8] Tool_ID: CLEAN-011; Tool_Category: Wet clean; Budgeted_Cost_INR_Crore: 19; Procurement_Status: Installed
[Tool_Capex_Summary!A9:D9] Tool_ID: TEST-021; Tool_Category: Wafer test; Budgeted_Cost_INR_Crore: 24; Procurement_Status: Installed
[Tool_Capex_Summary!A10:D10] Tool_ID: LTH-002; Tool_Category: Lithography; Budgeted_Cost_INR_Crore: 176; Procurement_Status: Installed
[Tool_Capex_Summary!A11:D11] Tool_ID: ETCH-015; Tool_Category: Dry etch; Budgeted_Cost_INR_Crore: 39; Procurement_Status: Installed
[Tool_Capex_Summary!A12:D12] Tool_ID: DEP-009; Tool_Category: CVD deposition; Budgeted_Cost_INR_Crore: 34; Procurement_Status: Installed
[Tool_Capex_Summary!A13:D13] Tool_ID: MET-004; Tool_Category: Metrology; Budgeted_Cost_INR_Crore: 29; Procurement_Status: Installed
```

**chunk 3** (table_row; sheet Customer_Revenue_Plan, cells A1:E2)
```text
[Customer_Revenue_Plan!A1:E1] A1: Customer; B1: Monthly_Volume; C1: Unit; D1: Start_Date; E1: Status
[Customer_Revenue_Plan!A2:E2] Customer: Northstar Controls GmbH; Monthly_Volume: 10000; Unit: Wafers; Start_Date: 2027-04-01; Status: Forecast pipeline; not purchase-order backed
```

---

## Appendix B. Claims extracted (12 per document; all 264 are in the appendix file)

264 claims in the latest run. Format: [value type] subject | metric | value unit | period.

### 01_Corporate_Overview.pdf (20 claims)
```text
[name] subject: SiliconForge Semiconductors Pvt. Ltd. | metric: document title | value: SiliconForge Semiconductors Pvt. Ltd.
[identifier] subject: Corporate overview | metric: version | value: 1.2
[date] subject: Corporate overview | metric: date | value: 30 September 2026
[statement] subject: SiliconForge | metric: company profile | value: fictional Indian semiconductor manufacturer developing a 300 mm wafer fabrication facility in Nanjangud Industrial Area, Mysuru district, Karnataka.
[quantity] subject: SiliconForge | metric: wafer fabrication facility diameter | value: 300 mm mm
[statement] subject: SiliconForge | metric: facility location | value: Nanjangud Industrial Area, Mysuru district, Karnataka
[statement] subject: SiliconForge | metric: business focus | value: mature-node logic and mixed-signal components for industrial, automotive-support and power-management applications.
[quantity] subject: SiliconForge | metric: initial process platform | value: 65 nm nm
[state] subject: Facility status | metric: Management-reported position | value: Pilot production; commercial ramp in progress
[quantity] subject: Installed cleanroom area | metric: Management-reported position | value: 18,500 m² m²
[quantity] subject: Designed nameplate capacity | metric: Management-reported position | value: 20,000 wafer starts per month (WSPM) wafer starts per month (WSPM)
[quantity] subject: Current qualified capacity | metric: Management-reported position | value: 12,000 WSPM WSPM | period: 30 September 2026
```

### 02_Manufacturing_Operations_Report.pdf (19 claims)
```text
[date] subject: Manufacturing Operations Report | metric: Reporting period | value: 1 July–30 September 2026
[date] subject: Manufacturing Operations Report | metric: Issued | value: 15 October 2026
[quantity] subject: Wafer starts | metric: Q3 2026 value | value: 27,600 wafers wafers | period: Q3 2026
[quantity] subject: Average monthly wafer starts | metric: Q3 2026 value | value: 9,200 WSPM WSPM | period: Q3 2026
[quantity] subject: Good wafers out | metric: Q3 2026 value | value: 25,510 wafers wafers | period: Q3 2026
[quantity] subject: Aggregate yield | metric: Q3 2026 value | value: 92.4% % | period: Q3 2026
[quantity] subject: Process node | metric: Q3 2026 value | value: 65 nm nm | period: Q3 2026
[identifier] subject: Fab line SF-65-A | metric: Line | value: SF-65-A
[statement] subject: Fab line SF-65-A | metric: Product mix | value: industrial controllers, power-management ICs and connectivity components
[statement] subject: Yield | metric: Calculation | value: good wafers out divided by wafer starts for the same cohort
[statement] subject: Rework lots | metric: Cohort | value: remain in the original cohort
[quantity] subject: Current qualified capacity | metric: Capacity | value: 12,000 WSPM WSPM
```

### 03_Quality_and_Compliance_Report.pdf (17 claims)
```text
[date] subject: Quality and Compliance Status | metric: report date | value: 31 October 2026
[date] subject: ISO 9001:2015 readiness assessment | metric: completion date | value: 18 September 2026
[quantity] subject: ISO 9001:2015 readiness assessment | metric: minor findings recorded | value: four findings
[date] subject: External certification audit | metric: scheduled date | value: 12 December 2026
[state] subject: ISO 9001:2015 certificate | metric: received status | value: No | period: as of this report date
[quantity] subject: Customer-reported major escapes in Q3 2026 | metric: Status | value: 0 reported | period: Q3 2026
[quantity] subject: Open internal corrective actions | metric: Status | value: 4 minor actions
[quantity] subject: Final test yield | metric: Status | value: 92.4% % | period: Q3 2026
[name] subject: Final test yield | metric: referenced report | value: Operations Report
[statement] subject: Traceability | metric: Status | value: Lot-level traceability maintained in MES, per management
[date] subject: Consolidated environmental consent application | metric: submission date | value: 7 August 2026
[name] subject: Karnataka State Pollution Control Board | metric: name | value: Karnataka State Pollution Control Board
```

### 04_Investor_Presentation.docx (18 claims)
```text
[quantity] subject: SiliconForge | metric: manufacturing platform diameter | value: 300 mm mm
[quantity] subject: SiliconForge | metric: manufacturing platform process node | value: 65 nm nm
[name] subject: SiliconForge | metric: manufacturing platform location | value: Karnataka
[statement] subject: SiliconForge | metric: markets | value: industrial, automotive-support and power-management markets
[state] subject: SiliconForge | metric: commercial ramp status | value: underway
[state] subject: SiliconForge | metric: qualified production capacity expansion | value: planned
[quantity] subject: SiliconForge | metric: current production yield | value: 96.8% % | period: Q3 2026
[identifier] subject: SF-65-A | metric: product identifier | value: SF-65-A
[quantity] subject: SiliconForge | metric: current installed nameplate capacity | value: 24,000 WSPM WSPM
[quantity] subject: SiliconForge | metric: monthly shipments | value: 10,000 wafers wafers | period: by Q2 2027
[identifier] subject: SiliconForge | metric: ISO certification | value: ISO 9001:2015
[state] subject: SiliconForge | metric: ISO certification status | value: certified
```

### 05_Customer_Supply_Agreement.docx (26 claims)
```text
[name] subject: Customer Supply Framework Agreement | metric: document title | value: Customer Supply Framework Agreement
[date] subject: Agreement SF-CUS-026 | metric: execution date | value: 22 September 2026
[identifier] subject: Agreement SF-CUS-026 | metric: agreement number | value: SF-CUS-026
[name] subject: Supplier | metric: name | value: SiliconForge Semiconductors Pvt. Ltd.
[statement] subject: Supplier | metric: location | value: Karnataka, India
[name] subject: Customer | metric: name | value: Northstar Controls GmbH
[statement] subject: Customer | metric: location | value: Munich, Germany
[quantity] subject: Products | metric: process node | value: 65 nm nm
[statement] subject: Products | metric: product description | value: industrial controller wafers
[quantity] subject: Products | metric: format | value: 300 mm mm
[statement] subject: Products | metric: qualification condition | value: subject to product-specific qualification and agreed specifications
[quantity] subject: Customer rolling forecast | metric: duration | value: first six months months
```

### 06_Facility_and_Utilities_Plan.docx (20 claims)
```text
[name] subject: Facility and Utilities Plan | metric: document title | value: Facility and Utilities Plan
[statement] subject: Engineering planning basis | metric: document basis | value: Engineering planning basis
[identifier] subject: Revision C | metric: revision | value: Revision C
[date] subject: Facility and Utilities Plan | metric: date | value: 28 October 2026
[quantity] subject: Facility | metric: wafer diameter | value: 300 mm mm
[quantity] subject: Facility | metric: process platform | value: 65 nm nm
[statement] subject: cleanroom and support areas | metric: commissioning status | value: are being commissioned in phases
[quantity] subject: Design connected electrical load | metric: electrical load | value: 22 MW MW
[quantity] subject: Initial commissioning demand | metric: demand | value: 14 MW MW | period: Initial commissioning
[statement] subject: Full-load demand | metric: included items | value: process tools, HVAC, UPW, abatement, compressed air and site utilities
[quantity] subject: Utility design | metric: supply connection | value: dedicated 22 MW supply connection MW
[state] subject: Final sanctioned-load approval | metric: included | value: was not included
```

### 07_Equipment_Register.xlsx (87 claims)
```text
[identifier] subject: LTH-001 | metric: Tool_ID | value: LTH-001
[statement] subject: LTH-001 | metric: Tool_Category | value: Lithography
[statement] subject: LTH-001 | metric: Process_Area | value: Photolithography
[state] subject: LTH-001 | metric: Status | value: Installed
[quantity] subject: LTH-001 | metric: Installed_Cost_INR_Crore | value: 185 INR Crore
[date] subject: LTH-001 | metric: Commissioning_Date | value: 2026-03-14 | period: 2026-03-14
[identifier] subject: ETCH-014 | metric: Tool_ID | value: ETCH-014
[statement] subject: ETCH-014 | metric: Tool_Category | value: Dry etch
[statement] subject: ETCH-014 | metric: Process_Area | value: Etch
[state] subject: ETCH-014 | metric: Status | value: Installed
[quantity] subject: ETCH-014 | metric: Installed_Cost_INR_Crore | value: 42 INR Crore
[date] subject: ETCH-014 | metric: Commissioning_Date | value: 2026-04-02 | period: 2026-04-02
```

### 08_Financial_and_Capacity_Model.xlsx (57 claims)
```text
[state] subject: TEST-021 | metric: Procurement_Status | value: Installed
[quantity] subject: Designed nameplate capacity | metric: Model_Value | value: 24000 WSPM
[quantity] subject: Current qualified capacity | metric: Model_Value | value: 12000 WSPM | period: As of 30-Sep-2026
[quantity] subject: Q3 2026 average wafer starts | metric: Model_Value | value: 9200 WSPM | period: Q3 2026
[quantity] subject: Q3 2026 production yield | metric: Model_Value | value: 96.8 % | period: Q3 2026
[quantity] subject: Target monthly shipments by Q2 2027 | metric: Model_Value | value: 10000 Wafers/month | period: by Q2 2027
[identifier] subject: LTH-001 | metric: Tool_ID | value: LTH-001
[statement] subject: LTH-001 | metric: Tool_Category | value: Lithography
[quantity] subject: LTH-001 | metric: Budgeted_Cost_INR_Crore | value: 185 INR_Crore
[state] subject: LTH-001 | metric: Procurement_Status | value: Installed
[identifier] subject: ETCH-014 | metric: Tool_ID | value: ETCH-014
[statement] subject: ETCH-014 | metric: Tool_Category | value: Dry etch
```

---

## Appendix C. Every non-consistent finding of the latest run, in full

5 findings. The 35 consistent findings are in the appendix file.

```text
--- FINDING 24 of 40  [fnd_e78645be0d474be6] ------------------------------
title                 : Q3 2026 production yield: 92.4% vs 96.8%
classification        : contradiction
severity              : high
confidence            : 0.97
evidence_status       : adequate
verification_status   : verified
description           : The Manufacturing Operations Report states an aggregate yield of 92.4% for Q3 2026 in its production summary table. The Investor Presentation states a current production yield of 96.8% on SF-65-A during Q3 2026. The excerpts do not state a context that accounts for the differing values.
materiality_reasoning : The difference exceeds 1% and concerns a prominent operating performance metric.
business_impact       : -
unresolved_questions  : -
evidence:
  [1] role=contradicting
      document : 02_Manufacturing_Operations_Report.pdf
      chunk    : chk_52323f74df704f8c   (chunk_index 0)
      location : page 1
      quote    : "Aggregate yield | 92.4% | Good wafers out / wafer starts"
      relevance: | Aggregate yield              | 92.4%         | Good wafers out / wafer starts |
      assessment: 02_Manufacturing_Operations_Report.pdf states: | Aggregate yield              | 92.4%         | Good wafers out / wafer starts | | subject=Aggregate yield; metric=Q3 2026 value; value=92.4%; unit=%; period=Q3 2026
  [2] role=contradicting
      document : 04_Investor_Presentation.docx
      chunk    : chk_60861b5bbeb24d94   (chunk_index 2)
      location : heading Operating performance
      quote    : "Current production yield: 96.8% on SF-65-A during Q3 2026."
      relevance: Current production yield: 96.8% on SF-65-A during Q3 2026.
      assessment: 04_Investor_Presentation.docx states: Current production yield: 96.8% on SF-65-A during Q3 2026. | subject=SiliconForge; metric=current production yield; value=96.8%; unit=%; period=Q3 2026; conditions=on SF-65-A
```

```text
--- FINDING 29 of 40  [fnd_3a356df30f784b67] ------------------------------
title                 : Q3 2026 wafer starts: 9200 WSPM versus 27,600 wafers
classification        : contextual_difference
severity              : low
confidence            : 0.97
evidence_status       : adequate
verification_status   : verified
description           : Claim A states “9200” for Q3 2026 average wafer starts in Capacity_Assumptions, with the note “Average monthly starts.” Claim B states “27,600 wafers” for Q3 2026 wafer starts in the Manufacturing Operations Report. The report distinguishes the total from the monthly average, stating “Average monthly wafer starts | 9,200 WSPM | 27,600 / 3 months”; the different scopes account for the values.
materiality_reasoning : The figures are not directly comparable because one is an average monthly rate and the other is a quarterly total.
business_impact       : -
unresolved_questions  : -
evidence:
  [1] role=contextual
      document : 02_Manufacturing_Operations_Report.pdf
      chunk    : chk_52323f74df704f8c   (chunk_index 0)
      location : page 1
      quote    : "Wafer starts | 27,600 wafers | MES production log export"
      relevance: | Wafer starts                 | 27,600 wafers | MES production log export      |
      assessment: 02_Manufacturing_Operations_Report.pdf states: | Wafer starts                 | 27,600 wafers | MES production log export      | | subject=Wafer starts; metric=Q3 2026 value; value=27,600 wafers; unit=wafers; period=Q3 2026
  [2] role=contextual
      document : 08_Financial_and_Capacity_Model.xlsx
      chunk    : chk_b515bf4dc50b4194   (chunk_index 0)
      location : sheet Capacity_Assumptions, cell A1:D6
      quote    : "Assumption: Q3 2026 average wafer starts; Model_Value: 9200; Unit: WSPM; Model_Note: Average monthly starts"
      relevance: | Q3 2026 average wafer starts        | 9200        | WSPM         | Average monthly starts  |
      assessment: 08_Financial_and_Capacity_Model.xlsx states: | Q3 2026 average wafer starts        | 9200        | WSPM         | Average monthly starts  | | subject=Q3 2026 average wafer starts; metric=Model_Value; value=9200; unit=WSPM; period=Q3 2026; conditions=Average monthly starts
```

```text
--- FINDING 32 of 40  [fnd_fcf5f1bf1c5f4de7] ------------------------------
title                 : MET-004 budgeted cost: 29 versus 26
classification        : contradiction
severity              : medium
confidence            : 0.99
evidence_status       : adequate
verification_status   : verified
description           : The Tool_Capex_Summary excerpt states MET-004 has a Budgeted_Cost_INR_Crore of 29; the Installed_Tools excerpt lists MET-003 with an Installed_Cost_INR_Crore of 28, not MET-004. The excerpts do not provide a comparable installed-cost claim for MET-004, so they do not establish a conflicting value for the same attribute.
materiality_reasoning : The claims concern different cost measures and the second excerpt does not state an installed cost for MET-004.
business_impact       : -
unresolved_questions  : -
evidence:
  [1] role=contradicting
      document : 07_Equipment_Register.xlsx
      chunk    : chk_0ca7b3b1525e4ae8   (chunk_index 0)
      location : sheet Installed_Tools, cell A1:F5
      quote    : "Tool_ID: MET-003; Tool_Category: Metrology; Process_Area: Metrology; Status: Installed; Installed_Cost_INR_Crore: 28; Commissioning_Date: 2026-05-09"
      relevance: | MET-003   | Metrology        | Metrology        | Installed | 28                       | 2026-05-09         |
      assessment: 07_Equipment_Register.xlsx states: | MET-003   | Metrology        | Metrology        | Installed | 28                       | 2026-05-09         | | subject=MET-003; metric=Installed_Cost_INR_Crore; value=28; unit=INR Crore
  [2] role=contradicting
      document : 08_Financial_and_Capacity_Model.xlsx
      chunk    : chk_f909b2d40b464c8e   (chunk_index 2)
      location : sheet Tool_Capex_Summary, cell A8:D13
      quote    : "Tool_ID: MET-004; Tool_Category: Metrology; Budgeted_Cost_INR_Crore: 29; Procurement_Status: Installed"
      relevance: | MET-004   | Metrology        | 29                      | Installed          |
      assessment: 08_Financial_and_Capacity_Model.xlsx states: | MET-004   | Metrology        | 29                      | Installed          | | subject=MET-004; metric=Budgeted_Cost_INR_Crore; value=29; unit=INR_Crore
```

```text
--- FINDING 35 of 40  [fnd_7a4d4bf714544cc1] ------------------------------
title                 : Nameplate capacity: 20,000 versus 24,000 WSPM
classification        : contradiction
severity              : high
confidence            : 1.0
evidence_status       : adequate
verification_status   : verified
description           : The Corporate Overview states designed nameplate capacity of 20,000 wafer starts per month (WSPM). The Investor Presentation states current installed nameplate capacity of 24,000 WSPM. These are different stated values, and the excerpts do not provide a context that explains the gap.
materiality_reasoning : The values differ by more than 5%, and capacity is a prominent operating metric.
business_impact       : -
unresolved_questions  : -
evidence:
  [1] role=contradicting
      document : 01_Corporate_Overview.pdf
      chunk    : chk_cfbce9bf850f4f82   (chunk_index 0)
      location : page 1
      quote    : "Designed nameplate capacity | 20,000 wafer starts per month (WSPM)"
      relevance: | Designed nameplate capacity | 20,000 wafer starts per month (WSPM)          |
      assessment: 01_Corporate_Overview.pdf states: | Designed nameplate capacity | 20,000 wafer starts per month (WSPM)          | | subject=Designed nameplate capacity; metric=Management-reported position; value=20,000 wafer starts per month (WSPM); unit=wafer starts per month (WSPM)
  [2] role=contradicting
      document : 04_Investor_Presentation.docx
      chunk    : chk_60861b5bbeb24d94   (chunk_index 2)
      location : heading Operating performance
      quote    : "Current installed nameplate capacity: 24,000 WSPM."
      relevance: Current installed nameplate capacity: 24,000 WSPM.
      assessment: 04_Investor_Presentation.docx states: Current installed nameplate capacity: 24,000 WSPM. | subject=SiliconForge; metric=current installed nameplate capacity; value=24,000 WSPM; unit=WSPM
```

```text
--- FINDING 40 of 40  [fnd_c5201d67ad8c46f0] ------------------------------
title                 : Referenced item not in the data room: Schedule A (wafer technical specification)
classification        : missing_information
severity              : medium
confidence            : 0.7
evidence_status       : unavailable
verification_status   : verified
description           : 05_Customer_Supply_Agreement.docx (Schedules and evidence) refers to "Schedule A (wafer technical specification)", but no file, sheet or heading in the data room is named like it, and every document was read.
materiality_reasoning : -
business_impact       : -
unresolved_questions  : ['Where is Schedule A (wafer technical specification)?']
evidence:
  [1] role=insufficient
      document : 05_Customer_Supply_Agreement.docx
      chunk    : chk_b6f6488d0a5e455e   (chunk_index 4)
      location : heading Schedules and evidence
      quote    : "Schedule A (wafer technical specification) is referenced but not attached to this extract."
      relevance: Refers to Schedule A (wafer technical specification)
      assessment: The document points to this item; it is not in the data room.
```

---

## Appendix D. Pair outcomes and rejections in the latest run

**Verification rejections** (a finding the model proposed that failed the exact-quote, value or row check):

```text
Rejected findings (first 15):
  - consistent 'SiliconForge ISO 9001:2015 certification status agrees': validation_failed: Finding not saved. a consistent compares statements, so at least two different quoted passages are required
  - consistent pair 1: the quotes do not show the claims' values ('fictional Indian semiconductor manufacturer developing a 300 mm wafer fabrication facility in Nanjangud Industrial Area, Mysuru district, Karnataka.' and 'Karnataka')
  - consistent pair 4: the quotes do not show the claims' values ('Deposition' and 'Installed')
  - consistent pair 4: the quotes do not show the claims' values ('TEST-021' and 'Electrical test')
  - contradiction pair 1: the quotes do not show the claims' values ('176' and '185')
  - consistent pair 1: the quotes do not show the claims' values ('31' and '28')
  - consistent 'TEST-021 installed cost: 24 INR Crore in both claims': validation_failed: Finding not saved. evidence 2: the quote is not in chunk 'chk_b515bf4dc50b4194' (08_Financial_and_Capacity_Model.xlsx, sheet Capacity_Assumptions, cell A1:D6). 
  - consistent pair 2: the quotes do not show the claims' values ('19' and '24')
  - insufficient_evidence pair 4: the quotes do not show the claims' values ('Lithography' and 'LTH-002')
  - consistent pair 1: the quotes do not show the claims' values ('CMP' and 'Wet clean')
  - consistent 'LTH-002 commissioning date: 2026-09-01 in both excerpts': validation_failed: Finding not saved. evidence 2: the quote is not in chunk 'chk_7b768180bb8f49e7' (08_Financial_and_Capacity_Model.xlsx, sheet Tool_Capex_Summary, cell A1:D7). Co
  - consistent pair 3: the quotes do not show the claims' values ('fictional Indian semiconductor manufacturer developing a 300 mm wafer fabrication facility in Nanjangud Industrial Area, Mysuru district, Karnataka.' and '300 mm')
  - insufficient_evidence 'LTH-001 procurement status: Installed in both claims': validation_failed: Finding not saved. evidence 2: the quote is not in chunk 'chk_0ca7b3b1525e4ae8' (07_Equipment_Register.xlsx, sheet Installed_Tools, cell A1:F5). Copy the text e
```

**Candidate pairs that involve the seeded items, or that got no verdict** (31 of 184 pair lines; the complete log is in the appendix file). Columns: pair number, claim A (file | subject | metric | value), claim B, relation hint, source route, what happened.

```text
183 candidate pairs. Outcomes: {'agreement saved as a consistent finding': 23, 'agreement': 35, 'judge': 120, 'agreement inside one document': 1, 'no verdict returned (batch failed': 4}

  #    claim A                                                                 claim B                                                                 hint         route      what happened
  ---  ----------------------------------------------------------------------  ----------------------------------------------------------------------  -----------  ---------  ----------------------------------------------------------------------------------------------------------------------------------------------------------------
  5    08 | MET-004 | Tool_Category | Metrology                                07 | MET-004 | Tool_Category | Metrology                                exact        llm        agreement: same quotes as a finding already saved
  17   08 | MET-004 | Tool_ID | MET-004                                        07 | MET-004 | Tool_ID | MET-004                                        exact        llm        agreement saved as a consistent finding
  25   04 | SiliconForge | current production yield | 96.8% %                  08 | Q3 2026 production yield | Model_Value | 96.8 %                    exact        llm        agreement saved as a consistent finding
  26   02 | Aggregate yield | Q3 2026 value | 92.4% %                          04 | SiliconForge | current production yield | 96.8% %                  contextual   llm        judge: contradiction (high), saved: Q3 2026 production yield: 92.4% vs 96.8%
  29   02 | Current qualified capacity | Capacity | 12,000 WSPM WSPM           01 | Current qualified capacity | Management-reported position | 12...  exact        llm        agreement saved as a consistent finding
  30   02 | Current qualified capacity | Capacity | 12,000 WSPM WSPM           08 | Current qualified capacity | Model_Value | 12000 WSPM              contextual   llm        judge: consistent (informational), saved: Current qualified capacity agrees: 12,000 WSPM
  31   01 | Current qualified capacity | Management-reported position | 12...  08 | Current qualified capacity | Model_Value | 12000 WSPM              contextual   llm        no verdict returned (batch failed, run stopped, or the model skipped the pair)
  32   08 | Designed nameplate capacity | Model_Value | 24000 WSPM             01 | Designed nameplate capacity | Management-reported position | 2...  contextual   llm        no verdict returned (batch failed, run stopped, or the model skipped the pair)
  33   08 | Target monthly shipments by Q2 2027 | Model_Value | 10000 Wafe...  04 | SiliconForge | monthly shipments | 10,000 wafers wafers            contextual   llm        judge: not comparable. Claim A measures target monthly shipments, while Claim B measures designed nameplate capacity.
  34   04 | SiliconForge | ISO certification | ISO 9001:2015                   04 | SiliconForge | ISO certification status | certified                contextual   llm        judge: consistent, REJECTED: validation_failed
  45   08 | Northstar Controls GmbH | Customer | Northstar Controls GmbH       05 | Customer | name | Northstar Controls GmbH                          exact        llm        agreement saved as a consistent finding
  51   07 | LTH-002 | Process_Area | Photolithography                          08 | LTH-002 | Procurement_Status | Installed                           conflicting  embedding  no verdict returned (batch failed, run stopped, or the model skipped the pair)
  54   07 | MET-003 | Process_Area | Metrology                                 08 | MET-004 | Procurement_Status | Installed                           conflicting  embedding  judge: not comparable. Claim A measures the Process_Area of MET-003; Claim B states the Procurement_Status of MET-004.
  55   07 | MET-004 | Process_Area | Metrology                                 08 | MET-004 | Procurement_Status | Installed                           conflicting  embedding  judge: not comparable. Claim A measures the Process_Area of MET-004; Claim B provides no claim about MET-004 in its cited excerpt, which lists MET-003.
  56   08 | MET-003 | Procurement_Status | Installed                           07 | MET-004 | Process_Area | Metrology                                 conflicting  embedding  no verdict returned (batch failed, run stopped, or the model skipped the pair)
  76   08 | MET-003 | Budgeted_Cost_INR_Crore | 28 INR_Crore                   07 | MET-004 | Installed_Cost_INR_Crore | 26 INR Crore                  partial      embedding  judge: not comparable. Claim A measures the budgeted cost of MET-003, while Claim B measures the installed cost of MET-004.
  77   08 | MET-004 | Budgeted_Cost_INR_Crore | 29 INR_Crore                   07 | MET-004 | Installed_Cost_INR_Crore | 26 INR Crore                  partial      embedding  judge: not comparable. Claim A measures the budgeted cost of MET-004, while Claim B measures the installed cost of MET-003.
  78   08 | MET-004 | Budgeted_Cost_INR_Crore | 29 INR_Crore                   07 | MET-003 | Installed_Cost_INR_Crore | 28 INR Crore                  partial      embedding  judge: contradiction (medium), saved: MET-004 budgeted cost: 29 versus 26
  93   02 | Operational target | Capacity | 16,000 WSPM WSPM                   01 | Current qualified capacity | Management-reported position | 12...  contextual   embedding  judge: not comparable. Claim A measures the operational target capacity by Q2 2027; Claim B measures current qualified capacity as of 30 September 2026.
  94   08 | Designed nameplate capacity | Model_Value | 24000 WSPM             04 | SiliconForge | current installed nameplate capacity | 24,000 W...  contextual   embedding  judge: consistent (informational), saved: Nameplate capacity: 24000 WSPM and 24,000 WSPM
  95   01 | Designed nameplate capacity | Management-reported position | 2...  04 | SiliconForge | current installed nameplate capacity | 24,000 W...  partial      embedding  judge: contradiction (high), saved: Nameplate capacity: 20,000 versus 24,000 WSPM
  102  07 | MET-003 | Tool_Category | Metrology                                08 | MET-004 | Tool_Category | Metrology                                exact        embedding  agreement: same quotes as a finding already saved
  103  07 | MET-004 | Tool_Category | Metrology                                08 | MET-003 | Tool_Category | Metrology                                exact        embedding  agreement: same quotes as a finding already saved
  121  07 | MET-003 | Commissioning_Date | 2026-05-09                          08 | MET-004 | Procurement_Status | Installed                           conflicting  embedding  judge: not comparable. Claim A gives the commissioning date for MET-003; Claim B gives the procurement status for MET-004.
  122  07 | MET-004 | Commissioning_Date | 2026-09-30                          08 | MET-004 | Procurement_Status | Installed                           conflicting  embedding  judge: not comparable. Claim A gives the commissioning date for MET-004; Claim B gives the procurement status for MET-004.
  123  08 | MET-003 | Procurement_Status | Installed                           07 | MET-004 | Commissioning_Date | 2026-09-30                          conflicting  embedding  judge: not comparable. Claim A measures MET-003's procurement status; Claim B measures MET-004's commissioning date.
  144  08 | MET-004 | Tool_Category | Metrology                                07 | MET-004 | Tool_ID | MET-004                                        conflicting  embedding  judge: not comparable. Claim A states the Tool_Category of MET-004; Claim B states the Tool_ID of MET-004.
  145  08 | MET-004 | Tool_ID | MET-004                                        07 | MET-004 | Tool_Category | Metrology                                conflicting  embedding  judge: not comparable. Claim A states the Tool_ID of MET-004; Claim B states the Tool_ID of MET-004.
  166  07 | MET-004 | Status | Installed                                       08 | MET-003 | Procurement_Status | Installed                           exact        embedding  agreement: same quotes as a finding already saved
  167  07 | MET-004 | Status | Installed                                       08 | MET-004 | Procurement_Status | Installed                           exact        embedding  agreement: same quotes as a finding already saved
  168  07 | MET-003 | Status | Installed                                       08 | MET-004 | Procurement_Status | Installed                           exact        embedding  agreement: same quotes as a finding already saved
```
