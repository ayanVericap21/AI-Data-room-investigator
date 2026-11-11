# Todo: classification and rule-set work

Status: [ ] not started, [~] in progress, [x] done. 299 offline tests pass. No real LlamaExtract call has been made with any of the new prompts or schema fields.

## Task 1: help soon, fixes something real
- [x] 1a. Claim type is no longer a hard pairing gate (kept as a label; the grouping prompt shows it for information and says not to keep claims apart because of it)
- [x] 1b. Severity defined in the judge prompt (mapping per category; the gap comes from code)
- [x] 1c. Five category definitions rewritten as a decision order in the judge prompt
- [x] 1d. Quote check normalises Unicode, quote and dash variants, zero-width characters and line-break hyphenation (stored text is still the source's own)

## Task 2: rule sets per value type
- [x] 2a. Numerical in code (`value_rules.py`): precision intervals, "about" (+/-10%), bounds, ranges, scale words, units of one dimension, bracketed negatives
- [x] 2b. Date in code: formats, intervals at stated precision, quarters and halves, ambiguous day/month = unknown, bare year only when both claims are typed date, durations as numbers with units
- [x] 2c. Textual in code: identifiers, phone numbers and names compare equal after case, accents, spacing, punctuation and legal-form words are removed
- [x] 2d. Wired in: `compare_attributes` uses the rules (so exact / conflicting hints change); the judge sees a "Code check" line per pair with the gap
- [x] 2e. Per-type rules written into the judge prompt (currency, percent vs points, fiscal periods, relative dates, polarity, statements stay with the judge)

## Task 3: single-claim missing-information check
- [x] 3a. Extraction also returns references (item + quote); no extra call, same job
- [x] 3b. `missing_items.py`: item absent from every file name, sheet, heading and first line, all documents processed: missing_information finding, one verified quote
- [x] 3c. Wired into the run (stage 3b), report section and metrics counts

## Not done (by design or left for later)
- [ ] Code for: currency conversion, percent against absolute, percent vs points, fiscal-year mapping, relative-date anchors, locale detection, part/whole checks, stock vs flow, yes/no polarity. These are prompt rules for the judge only.
- [x] Six value types (quantity, date, identifier, name, state, statement) in extraction, grouping prompt and value rules; stored in the existing claim_category column (no migration)
- [ ] Extractor fields (qualifier, scale, sign, as-of date, polarity): need new columns, so the architect decides first
- [ ] Rules use English words ("about", "more than", month names)
- [ ] Number words ("three-year") are not parsed
- [ ] Run counts in the UI do not show missing-information findings separately (the totals include them)
