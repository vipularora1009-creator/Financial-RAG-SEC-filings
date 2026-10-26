# ADR-004: Query router: metadata pre-filtering and HyDE adopted; decomposition built, not yet adopted

- **Status:** Accepted (2026-10-26). HyDE is provisional (see Consequences).
- **Deciders:** project owner, with Claude Code as pair
- **Related:** spec §4.5, §5.3 Exp-5a / Exp-5b; ADR-003; learning records 0023, 0024

## Context

After Exp-4 (hybrid dense + BM25, ADR-003), the failure analysis on the 76 FinanceBench questions showed two separate problems:

1. **Wrong filing in the context.** The top passage came from the right filing on 62/76 questions, but on average 34% of the 5 parents sent to Gemini came from other filings (other quarters or years of the same company). Numeric EM was 56% when all 5 parents came from the gold filing and 30% when filings were mixed.
2. **Wrong page inside the right filing.** 36 of the 39 Hit@5 misses already had the gold filing in the top 5: the search found the filing, not the evidence page.

The spec's query router has three paths: metadata pre-filtering (problem 1), HyDE / step-back rewriting (problem 2, vocabulary gap between questions and filing prose), and sub-query decomposition (comparative questions across companies or periods). All 76 evaluation questions are single-filing, so decomposition cannot be measured on this set.

Options considered:

- **Metadata extraction:** Gemini structured output with the ticker as an enum of the 30 indexed companies (chosen); a rules table of company aliases plus a year regex (free, deterministic, instant, but breaks on unexpected wording: "the iPhone maker", "Minnesota Mining"); rules first with an LLM fallback; the `instructor` library (not needed: google-genai accepts Pydantic schemas directly).
- **Filter strictness:** hard on ticker and year (strongest, but drops the gold filing on 2 questions); ticker only (safe, but leaves the same-company wrong-filing problem); hard on ticker, year ∈ {y−1, y} (chosen); a quarter filter in addition.
- **Where the filter applies:** pre-filtering inside each search (chosen) vs post-filtering the fused results (can leave few candidates).
- **HyDE:** hypothetical passage for dense only, BM25 keeps the question (chosen); passage for both (invented figures become exact BM25 tokens that match unrelated passages); averaging question and passage vectors (an anchor against off-topic passages; not run); HyDE only for questions classified as conceptual (a second LLM decision with its own errors; not run).
- **Decomposition merge:** round-robin interleave of parents (chosen) vs sorting by score (scores from differently filtered searches are not comparable) vs RRF over sub-lists (same order, less direct).

## Decision

1. **Adopt metadata pre-filtering.** Gemini extracts `QueryMetadata(ticker, fiscal_year, quarter)`; code builds the filter: ticker hard, `fiscal_year ∈ {y−1, y}`, quarter not used, no filter when nothing is extracted. The filter is applied inside both the dense and the BM25 search, backed by Qdrant payload indexes. y−1 covers forecasts (Boeing's FY2023 outlook is in the FY2022 10-K) and fiscal-year naming (Ulta's year ending January 2023).
2. **Adopt HyDE for the dense side (provisional).** Gemini writes a ~130-word 10-K/10-Q-style passage; dense search embeds it instead of the question; BM25 keeps the question. The passage never reaches the generator.
3. **Build decomposition, do not put it in the default pipeline yet.** `decompose()` → 1–4 `SubQuery` (question + filter fields); `search_decomposed()` runs the filtered searches in parallel (`asyncio.gather` + `asyncio.to_thread`), expands each to parents, then interleaves. To be measured on the Phase 2 multi-hop questions.
4. **Keep the LLM's job literal and the policy in code:** the model copies what the question states ("FY2023" → 2023); year widening and filter construction are deterministic and unit-tested.
5. **Evaluate router changes on cached router outputs** (`data/cache/exp5_*.jsonl`), so repeat runs differ only in retrieval and generation; charge their latency and cost to the totals.

## Evidence

n = 76 FinanceBench questions (EM on the 51 numeric); exact search; 5 parents to the generator; exact two-sided sign tests on paired questions.

**Extractor on its own** ($0.03): ticker right 75/76 (the 1 empty is a question that names no company); year equal to the gold filing's fiscal year 62/64 named, the 2 others covered by y−1; **filter keeps the gold filing 76/76** (with a quarter filter 75/76: "Q2 of FY2022" for CVS is answered by the FY2022 10-K, so the quarter filter was dropped). Median 1 filing left to search (exactly 1 for 48/76), median 528 of 76,788 children.

| | Hit@5 | Hit@10 | MRR@10 | All 5 parents from gold filing | EM lenient / strict | Refusals | Cost / 1k | Routing latency (median) | Total P95 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Exp-4 hybrid, run 1 / run 2 | 0.49 | 0.57 | 0.34 | 30/76 | 0.39 / 0.37; 0.43 / 0.39 | 0.33; 0.30 | $10.22; $10.64 | — | 15.3 s; 25.9 s |
| Exp-5a + pre-filter | 0.55 | 0.62 | 0.37 | 50/76 | **0.51 / 0.45** | 0.28 | $10.80 | 2.3 s | 24.7 s |
| Exp-5b + HyDE | **0.63** | **0.71** | **0.42** | — | 0.49 / 0.43 | **0.22** | $11.77 | 6.2 s | 23.3 s |

Paired comparisons:
- **Pre-filter vs hybrid:** Hit@5 +5 / −0 (p = 0.06); EM +6 / −0 (p = 0.031) vs run 1 and +5 / −1 (p = 0.22) vs run 2. Same direction against both identical baseline runs and larger than the ±0.04 noise floor, but significant against only one. The mechanism holds: numeric questions with a single-filing context rose from 18 to 33 of 51, and EM in that group stayed at 56–58%.
- **HyDE vs pre-filter:** Hit@5 +6 / −0 (p = 0.031), Hit@10 +8 / −1 (p = 0.039), MRR 22 better / 12 worse: the first ranking gains in the project with p < 0.05. Gains in both large question types (Hit@5 domain-relevant 0.48 → 0.56, novel-generated 0.65 → 0.75). EM +1 / −2.
- **Why HyDE's EM is flat, inspected:** three refusals became answers a reader would accept but EM scores wrong (Adobe FCF conversion 142.7% → 155.5% vs gold 143% → 156% plus an unstated "~13%"; Amcor "No, 19.4% → 18.5%" vs gold "decline of 0.8%"; American Water "No, a deficit" without the figure). On Boeing's tax rate the answer quotes the filing, (0.6)% vs 14.7%, while the gold has 0.62% vs −14.76%. EM's number matching under-credits yes/no answers with derived figures.
- **Decomposition** (smoke test only): "Compare 3M's and Johnson & Johnson's total revenue in FY2022" → 2 sub-queries; top-5 parents alternate MMM, JNJ, MMM, JNJ, MMM. Interleaving children before expanding gave 3 JNJ / 2 MMM (shared parents were deduplicated), hence expand-then-interleave.

**Cost.** Extraction ~$0.42 / 1k questions, HyDE ~$0.97 / 1k (together +15% on Exp-4's ~$10.4). Latency: extraction 2.3 s + HyDE 3.8 s median, run in sequence before search.

## Consequences

**Positive**
- The generator sees the right filing far more often (50/76 questions with an all-gold-filing context, from 30), the change most clearly linked to answer accuracy so far.
- HyDE closes the question-vs-filing vocabulary gap and finds the evidence page inside the right filing (Hit@10 0.71, the best in the project).
- A closed ticker enum and a filter-recall check (76/76) make the hard filter safe on this corpus; the policy is plain, tested code.
- Decomposition is ready for comparative questions, with parallel searches and balanced parents.

**Negative / risks**
- **Latency:** ~6 s of LLM calls before search (median; P95 9.3 s). Both calls need only the question and could run in parallel (~4 s, the slower one); not yet built.
- **An LLM in the query path:** a Gemini outage or a wrong ticker now affects retrieval, not just generation. A wrong hard filter removes every right passage. No fallback yet (e.g. unfiltered search on timeout or low confidence).
- **Corpus caveat:** 17 of the 30 companies have a single filing here, so the ticker filter alone often isolates the filing. With full filing histories the filter narrows less and the gain will likely be smaller.
- **HyDE is provisional:** its retrieval gain is significant, its answer gain is unmeasured because the EM judge under-credits the answers it unlocked. Adoption is revisited once the answer metric is fixed. If HyDE then shows no answer gain, the +9% cost and ~3.8 s are not justified.

**Open questions**
- A better answer metric. An LLM judge was built (`evals/judge.py`: rubric, closed verdict set; agreed with a reader on 6 hand-checked Exp-5b answers) but, by the owner's decision on 2026-10-26, not validated or adopted: EM stays the answer metric with its known blind spot, and HyDE stays provisional until Phase 2, whose custom questions can carry gold answers suited to EM (or revive the judge with human-label validation).
- Parallel extraction + HyDE, and a fallback when the router fails.
- Averaging question and HyDE vectors; several HyDE samples; HyDE only for some question types.
- A `form_type` filter for annual questions (the decomposition smoke test put 3M 10-Qs into an annual comparison).
- Decomposition measured on the Phase 2 multi-hop questions, ideally replacing the single-query extractor (a single-company question decomposes to one sub-query).
