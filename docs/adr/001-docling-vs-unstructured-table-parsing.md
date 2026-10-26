# ADR-001: Docling for layout parsing of SEC filings (over Unstructured)

- **Status:** Accepted (2026-10-25). Updated 2026-10-25: serializer v2 built, benchmark ground truth corrected (see Update).
- **Deciders:** project owner, with Claude Code as pair
- **Related:** spec §4.3 (Strategy 2), §5.3 Exp-2 / Exp-3; ADR-002

## Context

Financial answers live in tables. A value is only useful with its **row label** (line item) and its **column header** (period): "29,998" means nothing; "Net income, three months ended December 31, 2022: $29,998 million" answers a question. Exp-1's naive HTML text extraction flattened tables one cell per line, and its failure analysis found the right filing in the top 5 for 65 of 76 questions but the right passage for only 18: retrieval matched company and year, not the line item.

Our filings are EDGAR HTML (iXBRL), not PDFs, so both candidates run their HTML paths (no OCR or vision models).

Options considered:
1. **Naive HTML text** (BeautifulSoup; Exp-1 baseline): no table structure.
2. **Unstructured** (`partition_html`): table elements with `text` and `metadata.text_as_html`.
3. **Docling** (HTML backend): a document tree with table grids (cell spans, offsets); serialized by us to compact Markdown.
4. **Our own parser**: rejected early; table spans, inline groups and page furniture are solved problems in 2 and 3.

## Decision

Use **Docling** to parse filings, with our own serializer turning each table grid into compact Markdown and our own SEC Item splitter (`ingestion/layout_parser.py`).

## Evidence

**Table fidelity** (`scripts/table_fidelity.py`, `scripts/dump_unstructured_tables.py`; 5 filings: AAPL, AMCR, TSLA 10-Q, AXP, MSFT 10-K). Ground truth comes from the source HTML, independent of every parser: each `<table>` becomes a grid with `colspan` expanded, and each number a (row label, column header, value) triple, 4,835 in total. An output keeps a triple if the value sits in a row with that label, under a column whose header contains the true header cell.

| Output | Triples kept | Figure recall | Table tokens |
| :--- | ---: | ---: | ---: |
| Docling, native Markdown export | **95.1%** | 100% | 403,622 (4.0×) |
| Docling + our compact serializer v1 (indexed in Exp-2/3) | 71.1% | 100% | 102,060 (1×) |
| Unstructured, table HTML | 16.8% | 100% | 248,838 |
| Unstructured, table text (default) | 0% | 98–99% | 65,910 |

- Unstructured's HTML drops `colspan`, so values slide out from under their period headers; its default text flattens each table into one line.
- Figure recall is ~100% for every output: a "are the numbers present?" check cannot tell these parsers apart. The triple check can.
- Parse time was similar (Docling 0.6–7.4 s, Unstructured 1.3–8.8 s per filing).
- Corpus-wide, Docling + our serializer lost 0 of 153,230 figures compared with the naive parser.

**End-to-end, on the 51 questions with numeric answers** (our proxy for table lookups; FinanceBench has no "table question" label):

| Run | Parser | Hit@5 | EM (lenient) | Refusals |
| :--- | :--- | ---: | ---: | ---: |
| Exp-1 | naive text | 0.27 | 0.20 | 0.67 |
| Exp-2 | Docling + serializer v1 (+ sections, header, semantic chunks) | 0.29 | 0.29 | 0.53 |
| Exp-3 | same parsing, parent-document retrieval | 0.41 | 0.45 | 0.31 |

Exp-1 → Exp-2 on these 51: EM +7 / −2 (sign test p = 0.18), Hit@5 +4 / −3. This change is confounded (parser, sections, header and chunker changed together) and not significant at n = 51, so the end-to-end numbers support the decision but do not isolate the parser. The fidelity benchmark above is the direct evidence.

**Storage and cost overhead**

| Item | Size |
| :--- | ---: |
| Qdrant `exp1_fixed512_bge_m3` (30,078 points) | 204 MB |
| Qdrant `exp2_semantic_bge_m3` (46,329 points) | 305 MB |
| Sentence-vector cache for semantic breakpoints (`data/cache/`) | 184 MB |

The extra Exp-2 storage comes mostly from smaller chunks (more points), not from the parser. Docling itself adds a large dependency (`docling==2.142.1`, which pulls in PyTorch-based layout models we do not use for HTML) and downgraded `websockets` 17.3 → 16.1.1.

## Consequences

**Positive**
- Tables reach the chunker with their structure; with Docling's native export 95% of label–period–value facts survive, versus 17% with Unstructured.
- The document tree also gives reading order, inline-group rejoining and table lead-in text, used by the Item splitter and the chunker.

**Negative / risks**
- **Our compact serializer v1 loses 24 points of the structure Docling provides** (71.1% vs 95.1%), worst on MSFT (23.6%). Two defects: (1) a glued "$" cell keeps the "$" column's position, so "$" rows and plain rows land in different columns; (2) header rows are detected only when column 0 is empty, missing "(In millions)" / "June 30," header rows. Exp-2 and Exp-3 were indexed with v1.
- Docling's native export keeps the structure but costs 4× the tokens (spanning headers repeated in every column), which would overflow 512-token chunks and the 1,500-token table limit.
- Heavy dependency footprint; Docling's API changes quickly (pinned exactly).

**Follow-up**
- **Serializer v2**: header rows = rows above the first numeric row; each value named by the header cells spanning its position; "$" glued at the number's position. Target: close to 95% triples kept at about 1× tokens, measured with `scripts/table_fidelity.py`.
- Re-index Exp-3 with v2 ("Exp-3b") before Exp-4, so the serializer's effect is measured on its own and Exp-4 changes only retrieval.
- Rejected for now: switching to Docling's native export (4× tokens).

## Update (2026-10-25): serializer v2 and a ground-truth fix

**Ground-truth fix.** Reading the American Express tables showed the benchmark's own rule was wrong: a header row of bare years ("2022 | 2021 | 2020") counted as a body row because years look like numbers, so those tables had period columns with no header. Bare 4-digit years now count as headers (SEC amounts of 1,000 and above carry commas). The fix applies to every output alike; the triple count rises from 4,835 to 6,794.

**Serializer v2** (`ingestion/layout_parser.py`):
1. A glued "$" cell keeps the number's position, so "$" rows and plain rows share a column.
2. Header rows = rows above the first row with a label and a value (bare years excluded), so "(In millions)" or "June 30," in column 0 no longer ends the header; column-0 header text names the label column; a label-only row directly above the first value row ("Revenues") stays in the body.
3. Columns start where values start, are named by every header cell spanning them, and neighbouring same-name columns that no row uses together are merged.

| Output (corrected ground truth, 6,794 triples) | Triples kept | Table tokens |
| :--- | ---: | ---: |
| Docling native Markdown export | 94.7% | 403,622 |
| **Docling + our serializer v2** | **85.0%** | **90,592** |
| Docling + our serializer v1 (indexed in Exp-2/3) | 67.6% | 102,060 |
| Unstructured, table HTML | 18.9% | 248,838 |
| Unstructured, table text | 0% | 65,910 |

Per filing, v1 → v2: AAPL 83.9% → 83.9%, AMCR 88.1% → 96.4%, AXP 73.0% → 76.4%, MSFT 28.5% → 91.4%, TSLA 84.0% → 95.8%. Figure recall stays 100%.

v2 closes most of the gap to Docling's native export (−9.7 points) at 0.22× its tokens, and is 11% smaller than v1 because merged columns drop the empty "$" columns. Caveat: v2's header rule and the benchmark's ground truth use the same heuristic (first row with a label and a value), so part of the gain is agreement by construction; the MSFT and AXP tables were also checked by eye. The decision stands; the next step is Exp-3b, re-indexing Exp-3 with v2 to measure the end-to-end effect.

**End-to-end (Exp-3b, 2026-10-25).** Re-indexing Exp-3 with v2 as the only change left EM unchanged on the 76 questions (0.45 / 0.39; +2 / −2 questions) and moved Hit@5 0.42 → 0.46 (+7 / −4, p = 0.55). The fidelity gain is real but did not show up in answer accuracy at this sample size; v2 is kept (no regression, fewer tokens). See spec §5.3 Exp-3b.
