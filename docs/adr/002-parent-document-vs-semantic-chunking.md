# ADR-002: Parent-document retrieval over semantic chunks alone

- **Status:** Accepted (2026-10-25). Open questions listed under Consequences.
- **Deciders:** project owner, with Claude Code as pair
- **Related:** spec §4.3 (Strategies 2 and 3), §5.3 Exp-2 / Exp-3; ADR-001; learning records 0021, 0022

## Context

After Exp-2 (Docling parsing, context headers, semantic chunking at 512 tokens) the right filing was in the top 5 for 73 of 76 questions, but the right passage for only 24. One chunk size had to serve two jobs that pull in opposite directions:

- **Matching** wants small, focused chunks: one paragraph or a few table rows, so the vector means one thing.
- **Answering** wants large context: the whole table with its header, units and footnotes, so the LLM can read and compute.

Options considered:
1. **Semantic chunks only** (Exp-2): one 512-token unit for both jobs.
2. **Parent-document retrieval** (spec Strategy 3): match on small children, pass their larger parents to the LLM.
3. **Larger chunks only** (e.g. 2,000 tokens, no children): more context, blunter vectors. Not run (see Open questions).
4. **Children plus LLM-generated table synopsis cards** (spec Strategy 3, full form): deferred to a separate ablation; ~16,600 tables means ~16,600 extra LLM calls, and adding it now would confound the result.

## Decision

Use **parent-document retrieval** (`chunking/parent_child.py`, `retrieval/doc_store.py`, `ingestion/indexer_exp3.py`):

- **Parents** (~1,600 tokens median, at most 2,000; tables up to 2,500 whole): consecutive Exp-2 units packed within one SEC Item, so a table shares its parent with its surrounding text and cuts prefer topic changes. Stored in Postgres (`parent_documents`) in the same transaction as the CDC ledger row.
- **Children** (~240 tokens median): each parent re-split at 384 tokens; tables bigger than that become row groups with the header row repeated. Embedded as context header + child; `parent_id` in the Qdrant payload.
- **Retrieval**: top-50 children → deduplicate by `parent_id` in rank order (a parent ranks at its best child: max aggregation) → top-5 parents to the generator.

## Evidence

Same 76 FinanceBench questions; judge ceiling 76/76 for children and parents; faithfulness = metric v2 for every row.

| Run | Hit@5 | MRR@10 | EM lenient / strict | Fully faithful | Refusals | Cost / 1k | Total P95 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Exp-2: semantic chunks | 0.32 | 0.24 | 0.29 / 0.27 | 0.79 (n=33) | 0.53 | $4.10 | 16.1 s |
| Exp-3: children only (ablation) | 0.34 | 0.22 | 0.25 / 0.22 | 0.91 (n=34) | 0.51 | $4.27 | 28.9 s |
| **Exp-3: children → parents** | **0.42** (child 0.34) | 0.29 (child 0.22) | **0.45 / 0.39** | 0.74 (n=46) | **0.32** | $10.90 | 21.8 s |

**Paired against Exp-2** (exact two-sided sign test on the questions that changed):

| Comparison | EM | Hit@5 |
| :--- | :--- | :--- |
| Exp-3 parents | +9 / −1, **p = 0.021** | +12 / −4, p = 0.077 |
| Exp-3 children only | +3 / −5, p = 0.73 | +8 / −6, p = 0.79 |

On the 51 numeric-answer questions (our table-lookup proxy), Hit@5 goes 0.29 → 0.41 and EM 0.29 → 0.45; every EM change is in this subset, since EM applies only to numeric answers.

**What the ablation shows.** The children-only run uses the same search as the full run, so the two differ only in what the LLM receives:
- Matching did not improve: child Hit@5 is 0.34 vs Exp-2's 0.32, a statistical tie.
- Small chunks alone do not help answering: EM 0.25, no better than Exp-2.
- Giving the LLM the parents lifts EM to 0.45 and cuts refusals from 53% to 32%: the model now sees whole tables and their notes instead of declining.
- Parent Hit@5 (0.42) is lenient by construction (a ~1,600-token parent contains evidence near the matched child), so the decision rests on EM, not on parent hits.

**Overhead**

| Item | Exp-2 | Exp-3 |
| :--- | ---: | ---: |
| Qdrant points | 46,329 | 73,385 (children) |
| Qdrant storage | 305 MB | 436 MB |
| Postgres doc store | none | 35 MB (10,711 parents) |
| Input tokens per question | 2,216 | 9,823 (4.4×) |
| Index time on the M1 (8 GB) | 158 min (reported across three runs; the true single-run cost is higher) | 260.8 min (one run) |

## Consequences

**Positive**
- The first statistically significant gain in the project (EM +9 / −1, p = 0.021), with a clean attribution: unlike Exp-2, one component changed, and the ablation separates matching from context.
- Indexing stays consistent under CDC: parents commit with the ledger row, old parents are deleted only after their old children, and an integrity check found 0 children without a parent and 0 orphan parents.

**Negative / risks**
- **2.7× cost per 1k questions and +5.7 s P95**, all from ~4× more context tokens.
- **Faithfulness**: fully grounded 0.74 on 46 answered numeric questions vs Exp-2's 0.79 on 33. More context means more answers attempted and more figures that can go wrong; at least one invented figure was found ("64 consecutive years" of dividends since 1916).
- More moving parts: two stores to keep in sync, ~1.6× more vectors, longer index time on the 8 GB Mac (swap reached ~8.7 of 9.2 GB).
- The tables inside parents still come from serializer v1, which keeps only 71% of label–period–value facts (ADR-001).

**Open questions / follow-ups**
- **Reranking to cut cost** (Exp-4): a cross-encoder choosing the best 2–3 parents could keep most of the EM gain at a fraction of the tokens; the risk is dropping the one parent with the answer.
- **Large chunks without children** (option 3) was not run, so "bigger context" and "parent-document" are not yet separated. Testing it means embedding the 10,711 parents as their own collection (10,711 vectors, but over the same total text in long sequences, so expect an Exp-2-sized index run) plus one eval; the saved reports cannot answer it.
- **Parent ranking**: max aggregation only; sum of child scores or reciprocal-rank fusion over children can be tested for free by re-sorting the saved 50 children per question.
- **Table synopsis cards** (option 4) as a separate ablation.
- n = 76: Hit@5 is not yet significant (p = 0.077); the 100 custom questions (Phase 2) will tighten this.
