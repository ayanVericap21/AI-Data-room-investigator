# Parsing accuracy per document

Score = share of the source file's own words that appear in the chunks the parser produces (100 = nothing left out).
The source is read a second way, independent of the parser. Checked 2026-11-11 against a fresh parse, not stored chunks.
Not yet stored: the database has no chunks until the next run.

| # | Document | Type | Chunks | Words found | Score | Garbled or invented tokens | Status |
|---|---|---|---|---|---|---|---|
| 01 | 01_Corporate_Overview.pdf | PDF | 2 | 152 / 152 | 100.0 | 0 | checked, reviewed by eye |
| 02 | 02_Manufacturing_Operations_Report.pdf | PDF | 2 | 165 / 165 | 100.0 | 0 | checked |
| 03 | 03_Quality_and_Compliance_Report.pdf | PDF | 2 | 155 / 155 | 100.0 | 0 | checked |
| 04 | 04_Investor_Presentation.docx | DOCX | 5 | 101 / 101 | 100.0 | 0 | checked, reviewed by eye |
| 05 | 05_Customer_Supply_Agreement.docx | DOCX | 5 | 159 / 159 | 100.0 | 0 | checked, reviewed by eye |
| 06 | 06_Facility_and_Utilities_Plan.docx | DOCX | 5 | 138 / 138 | 100.0 | 0 | checked, reviewed by eye |
| 07 | 07_Equipment_Register.xlsx | XLSX | 4 | 98 / 98 words, 15 / 15 numeric cells | 100.0 | 37 cell references and numbers added by the parser | checked, every cell compared |
| 08 | 08_Financial_and_Capacity_Model.xlsx | XLSX | 4 | 104 / 104 words, 18 / 18 numeric cells | 100.0 | 43 cell references and numbers added by the parser | checked, every cell compared |

Notes
- The three PDFs have one page each, no images and one table each, so there is no image-only content to miss.
- The score covers text and cell values. It does not cover images, charts or shapes in any file type, and it does not check reading order.
- The old stored chunks for the PDFs scored lower (document 1: 125 of 152 words). They were deleted; the fixed parser produces the numbers above.
- The three DOCX files have only body paragraphs: no tables, images, text boxes, headers, footers, footnotes, comments, tracked changes or hidden text. Each chunk is one heading plus its paragraphs, in reading order. A "m3" shown as a broken character in some terminal output is a display issue only; the chunk holds the correct character.
- Both XLSX files: every cell of every sheet (5 sheets, all visible) is in the chunks. No formulas, merged cells, hidden rows or columns, comments, images, charts or defined names, so nothing of that kind can be lost. Each row is a `Header: value; ...` line with its sheet and cell range. The header row is also kept as its own line (`A1: Tool_ID; B1: ...`), which adds label-only text to the chunks.
- The XLSX "extra" tokens are the cell locators (for example `Installed_Tools!A2:F2`) and the numeric cell values written as plain numbers, which the parser adds on purpose.
