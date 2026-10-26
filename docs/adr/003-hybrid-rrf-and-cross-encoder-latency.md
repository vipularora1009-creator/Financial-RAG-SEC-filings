# ADR-003: Hybrid retrieval (dense + BM25, RRF) kept; cross-encoder reranker not adopted

- **Status:** Accepted (2026-10-26)
- **Deciders:** project owner, with Claude Code as pair
- **Related:** spec §4.6, §5.3 Exp-4; ADR-002; learning records 0021, 0022

## Context

After Exp-3/3b, answers improved through richer context (parent documents), but matching barely moved: child-level Hit@5 0.34–0.38. The spec's standard remedy is two-stage retrieval:

1. **Hybrid first stage.** Dense vectors capture meaning but blur exact tokens (`ASC 606`, `EBITDA`, `$14,205`); BM25 matches exact tokens but not meaning. Reciprocal Rank Fusion (RRF) merges the two rankings by rank, not score, since cosine (~0.3–0.8) and BM25 (unbounded) are on different scales.
2. **Cross-encoder reranking.** A cross-encoder reads question and passage together, which judges relevance better than comparing two separate vectors, but needs one model pass per pair, so it only re-sorts the top candidates.

Options considered: dense only (Exp-3b); BM25 only; hybrid via RRF; hybrid + reranker (`BAAI/bge-reranker-v2-m3`); bge-m3 learned sparse or SPLADE instead of BM25 (not run: more correlated with the dense signal, or one more model).

## Decision

- **Keep hybrid retrieval** (fastembed BM25 + Qdrant IDF, RRF k = 60 over top-30 + top-30) as the first stage for the next experiments. It does not raise answer accuracy measurably, but it ranks evidence higher at no Gemini cost and only small retrieval cost, and it recovers exact-wording questions dense search misses.
- **Do not adopt the cross-encoder reranker** in the default pipeline. Keep the code (`retrieval/rerank.py`, Modal `Reranker`) for re-testing on a larger or differently judged evaluation set.
- **Evaluate with exact search** and judge every EM difference against the measured noise floor.

## Evidence

n = 76 FinanceBench questions; exact search; 5 parents to the generator; judge ceiling 76/76.

| | Hit@3 | Hit@5 | Hit@10 | MRR@10 | EM lenient / strict | Refusals | Retrieval P95 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Dense (Exp-3b vectors) | 0.37 | 0.47 | 0.58 | 0.31 | 0.45 / 0.39 (Exp-3b) | 0.33 | — |
| BM25 only | 0.21 | 0.26 | 0.39 | 0.18 | — | — | 0.12 s |
| Hybrid (RRF) | 0.43 | 0.49 | 0.57 | 0.34 | 0.39 / 0.37 (run 2: 0.43 / 0.39) | 0.33 | 0.88 s |
| Hybrid + rerank | 0.33 | 0.42 | 0.61 | 0.26 | 0.39 / 0.33 | 0.36 | 4.19 s |

**Noise floor.** The hybrid configuration was run twice unchanged. Retrieval was deterministic (identical context for 76/76 questions), but only 21/76 answers matched word for word; EM flipped on 4 questions (+3 / −1, 0.39 vs 0.43), refusals on 4, and total P95 moved from 15.3 s to 25.9 s.

**Paired comparisons** (exact two-sided sign test):
- Hybrid vs dense: Hit@3 +10 / −5 (p = 0.30), Hit@10 +5 / −6. Better ordering, no wider coverage.
- Exp-3b vs hybrid: EM +2 / −5 (p = 0.45), within the noise floor.
- Hybrid vs hybrid + rerank: EM +5 / −5, Hit@5 +4 / −9 (p = 0.27), within the noise floor for EM.
- BM25 finds 5 questions dense misses in its top 10; hybrid keeps 4.

**Why the reranker scores lower on Hit@k without hurting EM.** It keeps the top-1 from the right filing as often as hybrid (62/76) and puts more right-filing passages in the top 5 (3.74 vs 3.30), but often ranks another passage of that filing above FinanceBench's single evidence page, which the relevance judge does not credit. Hit@k with one gold page under-credits it; EM shows no loss and no gain.

**Cost.** Hybrid: no extra Gemini tokens, BM25 index +57 MB over Exp-3b's 448 MB (505 MB total, after optimization), build 5.6 min. Reranker: one Modal GPU call per question (25 pairs of up to 2,048 tokens), +3.3 s retrieval P95, ~2 min cold start per session.

## Consequences

**Positive**
- Exact-wording questions (cover-page boilerplate, named instruments) are recovered by BM25.
- The evaluation is now calibrated: retrieval is deterministic with exact search, and the generation noise floor (≈4 EM flips at n = 76) is known. In hindsight, Exp-3 vs Exp-2 (EM +9 / −1) is a real gain; Exp-3b vs Exp-3 and all Exp-4 differences are not distinguishable from noise.

**Negative / risks**
- Two retrieval systems to maintain (dense + sparse in one collection), a fastembed dependency, and a number-normalization rule that queries and documents must share.
- Rejecting the reranker rests on n = 76 and a single-evidence-page judge, both of which disadvantage it. The decision should be revisited with the Phase 2 custom questions (more n, possibly multiple acceptable passages).

**Open questions**
- Fewer parents to the generator (top-2/3) to cut the ~10,000-token context: not worth testing with this reranker, which does not rank the evidence page first more often.
- Reducing generation noise: several samples per question with a majority vote, or a model with deterministic decoding, would make small effects measurable.
- Metadata pre-filtering by company / period / form (Exp-5) targets the remaining failure: choosing the right filing among near-identical quarterly text.
