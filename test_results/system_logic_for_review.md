# Data Room Investigator: logic and results, for a second opinion

Written 2026-11-12. For a reviewer who has not seen the code. The full prompts are in `prompts_for_review.md` (same folder). Please be critical: find gaps, wrong assumptions, and cheaper or more reliable ways to do each step.

## 1. What it does
Given a folder of documents (PDF, DOCX, XLSX) from any business domain, find discrepancies between them and report each one with exact quotes, the document, the page or cell, and a classification. It must work for any domain, so no rule may rely on industry words. Every finding must be verifiable: no invented numbers and no invented quotes.

Five classes: `contradiction`, `contextual_difference`, `missing_information`, `insufficient_evidence`, `consistent` (an agreement, saved so a reader can see what matched). A sixth outcome, "not comparable", is counted and discarded.

## 2. Pipeline
1. **Scan.** Hash each file (SHA-256). A file with an unchanged hash is not re-read. A run on a selection of files touches only those files.
2. **Parse and chunk (code).** PyMuPDF (tables read cell by cell), python-docx, openpyxl. Chunks are at most 1,000 characters. Each table row is its own line with its sheet and cell range. Parsing was checked against an independent read of each file: all 8 documents score 100% (every word and every numeric cell of the source appears in the chunks).
3. **Embed (local, free).** MiniLM, 384 dimensions, on the machine. Chunk vectors are stored but not used by the current pipeline. Claim text is embedded on the fly for pairing.
4. **Extract claims (LlamaExtract, one job per document).** The model reads the original file and returns claims: subject, metric, value (as written), unit, period, conditions, claim sentence, and a `value_type`. The same job also returns `references`: documents, annexes, schedules or approvals the text says exist elsewhere. Code attaches each claim to a stored chunk (page and text; the value must appear as a whole number or string in that chunk, and the chunk holding the claim's subject wins when several hold the value). A claim whose value is in no chunk is rejected and counted.
5. **Pair.** Three sources of candidate pairs:
   - an LLM groups claims about the same attribute (sorted claims, one call per batch of up to 60,000 characters; claims in different batches are never grouped together); code splits each group by distinct value;
   - nearest neighbours: each claim is paired with its top 2 most similar claims in other documents (cosine at least 0.72 on "subject: metric: claim sentence");
   - a word-overlap route exists but the default run uses the two above.
   Gates: rows of one table are never paired; a pair that includes an identifier claim is kept only if the subjects overlap (Jaccard at least 0.5); the claim type is shown to the grouping model but does not gate pairing.
6. **Compare values in code (`value_rules`).** Gives `equal`, `different` or `unknown` per pair:
   - quantity: interval from written precision ("10.2" = 10.15 to 10.25), "about" = plus or minus 10%, bounds ("more than 500"), ranges compared end to end, scale words (thousand, million, crore ...), units of one dimension converted (kg and tonne), bracketed amounts negative;
   - date: formats, intervals at stated precision ("1 July 2024" is inside "Q3 2024"), day/month order that cannot be told apart is `unknown`, bare years only when both claims are typed date;
   - identifier: letters and digits identical (separators and case ignored), never fuzzy, never read as a number; an identifier against another type is different;
   - name: equal after case, accents, punctuation and legal-form words (Ltd, GmbH ...) are removed;
   - state and statement: equal if the normalised text is identical, otherwise left to the judge.
   Equal values in two documents become a `consistent` finding without any model call.
7. **Judge (LlamaExtract, batches of up to 12,000 characters of excerpts).** For each pair whose values differ, the model sees both claims, both full source chunks, and a "Code check" line (values agree within precision, or differ with the gap, or could not be compared). It returns same_attribute, class, severity, confidence, title, description and one quote from each chunk. The prompt contains a decision order (different attribute, then consistent, then contextual vs contradiction, then missing information, then insufficient evidence), a severity table, and per-type rules (currency, percent vs points, fiscal periods, relative dates, identifiers, names, polarity, statements).
8. **Verify and save (code).** Each quote must be an exact substring of its chunk after normalising Unicode, quote and dash variants, zero-width characters and line-break hyphenation. It must contain the claim's value. For a claim read from a table, the quote must come from the claim's own row (a quote from a neighbouring row is rejected; a quote from the right row without the label is widened to the whole row). Contradictions, contextual differences and consistent findings need two different passages. A failed check rejects the finding whole and counts it.
9. **Single-claim missing information (code).** Each reference returned in step 4 is looked up in file names, sheet names, headings and each chunk's first line (excluding the chunk that makes the reference). If it is nowhere, and every document was fully processed, a `missing_information` finding is saved with the referencing sentence as its quote.
10. **Report.** A text report and a metrics JSON: findings with evidence, every candidate pair with its outcome, per-phase LLM timing and credits, rejection reasons.

Settings fixed in code: LlamaExtract tier `cost_effective` for everything; local embeddings; the environment supplies only keys, the database URL and paths.

## 3. Claim value types (assigned by the extractor)
`quantity` (number with or without unit, currency, percentage, duration, range, bound), `date`, `identifier` (codes, certificate, order and phone numbers), `name`, `state` (yes/no, status, quantified assertions), `statement`. Stored in an existing free-text column, no schema change. Last run, 264 claims: quantity 74, statement 65, state 51, identifier 34, date 30, name 10 (none untyped).

## 4. The data room used for testing
8 documents: 3 PDF (1 page each), 3 DOCX, 2 XLSX (5 sheets), 29 chunks in total. It is synthetic and was seeded with known discrepancies, which are the ground truth below.

## 5. Results by run
| | Run A (early) | Run B (after type, rule and missing-info work) | Run C (latest) |
|---|---|---|---|
| Claims | 204 | 264 | 264 |
| Candidate pairs | 145 | 180 | 183 |
| Exact agreements (no LLM) | 59 | 59 | 59 |
| Judged by the LLM | 86 | 121 | 124 |
| Verdicts "not comparable" | 56 of 86 | 83 of 121 | 79 of 120 |
| Findings saved | 24 | 38 | 40 |
| Contradictions | 1 | 5 (2 false) | 3 |
| Contextual differences | 1 | 0 | 1 |
| Missing information | 0 | 1 | 1 |
| Consistent | 22 | 31 | 35 |
| Findings rejected by the quote check | 20 | 17 | 13 |
| Wall time | 13 min 32 s | 9 min 12 s | 3 min 39 s |
| LLM calls | 16 | 24 | 25 |
| LlamaExtract credits | 883 | 1,218 | 1,208 |
| of which extraction | 78 | 78 | 78 |

Run C timing: extraction 9.5 s, grouping 6.4 s, judging and verification 203 s (93% of wall time; judge calls run one after another). The free plan allows about 10,000 credits a month, so roughly 8 full runs.

Run C chunk-level facts: all 264 claims typed; 1 claim rejected at extraction; 4 pairs got no verdict back from the model; 59 exact agreements, 23 saved as consistent findings (the rest were duplicates of an earlier quote or inside one document).

## 6. Ground truth: seeded discrepancies and what happened
| Seeded item | Latest status |
|---|---|
| Yield 92.4% (manufacturing report) vs 96.8% (investor deck) | Found as a contradiction, correct evidence. Was missed until a PDF table parsing bug was fixed. |
| Designed nameplate capacity 20,000 vs 24,000 WSPM | Found, correct. Earlier runs missed it or reported it twice. |
| Schedule A referenced but not attached | Found by the missing-information check. |
| Tool MET-004 cost: 26 (installed, equipment register) vs 29 (budgeted, financial model) | Reported as a contradiction, but with wrong evidence: the 26 was attached to the chunk of another row because the value check matched "26" inside "2026-03-14". Fixed since (whole-number match, subject-aware chunk choice) but not yet re-run. The correct pair was earlier judged "not comparable" because installed and budgeted cost may be different attributes. |
| ISO 9001 certified (deck) vs no certificate received (quality report) | Missed. The two claims are each other's closest match in the other document (similarity 0.62 to 0.66) but the cut-off is 0.72, so the pair was never made. |
| Customer forecast 8,000 (agreement) vs 10,000 (financial model) | Missed. Similarity about 0.4 (rank 9 to 20) because the wording and subjects differ ("Customer rolling forecast" vs "Northstar Controls GmbH"); the grouping model probably never saw both in one batch (2 batches in the run). |

False positives seen and fixed: a wrong-row quote ("CLEAN-011 cost 19 vs 24", fixed by the row check), and "Tool_ID MET-004 vs MET-003" (identifiers of two different tools, fixed by the identifier subject gate). One contextual difference in Run C (9,200 average monthly starts vs 27,600 quarterly total) is arguably `consistent` after converting 27,600 / 3 = 9,200.

## 7. Known weaknesses and open decisions
- **Recall of pairing is the main gap.** Pairs depend on an 0.72 similarity cut-off, two neighbours per claim, and an LLM grouping that cannot see across batches. Proposed: cut-off about 0.6; one grouping batch or overlapping batches; retry pairs that returned no verdict.
- **Judge calls are sequential and are 93% of run time and most credits.** About two thirds of judged pairs (79 of 120 in the latest run) come back "not comparable" (label columns such as category or status create many pairs). Options: skip label-only claims at extraction, parallel calls.
- **Judge quality depends on the prompt.** Installed vs budgeted cost is the kind of case where a model may say "different attribute" even though a human wants it flagged.
- **Rules assume English** for approximation and bound words and month names. Number words ("three-year") are not parsed.
- **Not in code, only in the judge prompt:** currency conversion, percent against absolute, percent vs percentage points, fiscal-year mapping, relative-date anchors, locale detection, part-against-whole checks, yes/no polarity.
- **Pending schema decisions (not made):** extractor fields for qualifier, scale, sign, as-of date and polarity would need new columns.
- **Missing-information check is conservative in one direction:** an item named only in body text (not a file name, sheet name, heading or first line) is reported missing.
- **Quote check does not prove the claim is about the quoted row when the subject is not a label in the chunk.**

## 8. What I would like your opinion on
1. Is the order (extract claims, pair, compare in code, judge, verify) the right architecture, or is there a cheaper way to get high recall without comparing everything with everything?
2. How should pairing be made reliable when two claims are worded very differently (customer forecast vs customer volume)?
3. The five categories plus "not comparable": are the definitions and the decision order sound? See `prompts_for_review.md`.
4. Which value-comparison rules would you add or remove? Is interval overlap with a 10% "about" band reasonable?
5. How would you measure quality? There is one small seeded data room. Suggest a test set and metrics (recall of seeded discrepancies, false-positive rate, cost per run).
6. Any risks in verifying only by exact-quote substring and value presence?
