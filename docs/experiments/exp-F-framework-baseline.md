# Experiment F: Framework baseline (LlamaIndex) vs our custom pipeline

- **Status:** Done (run 2026-10-26, after Exp-5). Was deferred on 2026-10-26 so Exp-5 could run first.
- **Created:** 2026-10-25
- **Related:** spec §4.3–§4.6, §5.2; ADR-001, ADR-002; Exp-4

## Question

How does an off-the-shelf RAG framework compare with our custom pipeline on the same data, prompt and metrics? Quality, cost, speed, change handling and debuggability.

## Why

Our pipeline uses plain Python modules over focused libraries (Docling, sentence-transformers, fastembed, qdrant-client, psycopg, google-genai, Modal); no LangChain or LlamaIndex. Frameworks offer most generic parts ready-made (parent/child retrieval, hybrid search, reranking, ingestion dedup), but not the SEC-specific ones (Item splitter, iXBRL fiscal periods, table serializer v2, deterministic numeric metrics). Exp-F measures what the custom work buys, and gives an interview-ready answer to "why not just use LangChain?".

## Design

Change only the pipeline. Keep fixed: the 76 FinanceBench questions, the Gemini model and prompt (`generation/`), and our evaluation (`evals/runner.py` metrics, paired sign tests, exact search where applicable).

| Stage | Our pipeline (Exp-4) | Framework version (LlamaIndex) |
| :--- | :--- | :--- |
| Parsing | Docling + our table serializer v2 + SEC Item splitter | `DoclingReader` + `MarkdownNodeParser` (Docling's native table export) |
| Chunking | Parent/child built from table-aware units | `HierarchicalNodeParser` (or `SemanticSplitterNodeParser`) |
| Store / retrieval | Qdrant named vectors (dense + BM25), our RRF, parent expansion | `QdrantVectorStore` hybrid mode + `AutoMergingRetriever` |
| Reranking | bge-reranker-v2-m3 on Modal | `SentenceTransformerRerank` (same model) |
| Ingestion / CDC | Postgres ledger, per-section hashes | `IngestionPipeline` with docstore hashing (document level) |
| Embedding | bge-m3 on Modal (L4) | bge-m3, same backend, so vectors stay comparable |

Optional third arm: the same with LangChain (`ParentDocumentRetriever`, `EnsembleRetriever`, `CrossEncoderReranker`).

## Measurements

1. **Quality:** Hit@3/5/10, MRR@10, EM (lenient/strict), faithfulness (metric v2), refusal rate; paired against Exp-4 with sign tests; judge ceiling re-checked for the framework's chunk format.
2. **Table fidelity:** `scripts/table_fidelity.py` on the framework's table output (expected near Docling native: 94.7% facts kept at ~4.5× our v2's tokens).
3. **Cost and speed:** input tokens per question, cost per 1k questions, index time, retrieval and total P95 latency.
4. **Change handling:** edit one section of one filing; count what each pipeline re-embeds and deletes.
5. **Code and debuggability:** lines of code per stage; whether known issues are visible (numbers split by BM25 tokenization, approximate-search noise, table column misalignment).

## Expected results (hypotheses to test)

- Much less code to write for the framework version.
- Quality near Exp-2/Exp-3 levels, not Exp-4, because the SEC-specific parts are missing.
- Higher context cost from Docling's native table export.
- Coarser change detection (document-level, not section-level).
- Harder diagnosis, since steps run inside library classes.

## Cost estimate

One full index on Modal (~1 h, under $1 GPU) plus one or two Gemini eval runs (~$1 each). New dependencies: `llama-index-core`, `llama-index-vector-stores-qdrant`, `llama-index-readers-docling` (exact versions pinned when added).

## As run (changes from the plan)

| Stage | LlamaIndex configuration | Note |
| :--- | :--- | :--- |
| Parsing | `DoclingReader` (Markdown export), doc id = accession number | The default id is a random uuid4, which would make every re-run look like new documents to an `IngestionPipeline` |
| Chunking | `HierarchicalNodeParser`, `chunk_sizes=[2048, 384]` | Two levels sized like our parents/children (tiktoken tokens) instead of the default [2048, 512, 128] |
| Metadata | ticker, company, form, report date, fiscal year/period (our values) | Prepended to the embedding text by default, much like our header |
| Store | `QdrantVectorStore(enable_hybrid=True, fastembed_sparse_model="Qdrant/bm25")` | Same BM25 term weights + Qdrant IDF as ours; no number normalization |
| Retrieval | hybrid, 30 dense + 30 sparse, default relative-score fusion (alpha 0.5), exact search, `AutoMergingRetriever` | Our depths; exact search via `vector_store_kwargs` |
| Reranking | not used | Our default pipeline has no reranker either (ADR-003) |
| Generation, metrics | our prompt, `gemini-3.8-flash`, `evals.runner.evaluate` | Unchanged |
| Embedding | bge-m3 fp16 on Modal L4 (index), locally (queries), through a 20-line `BaseEmbedding` adapter | Same model and backend as ours |

Not run: the optional LangChain arm; a separate table-fidelity run (Docling's native Markdown export was already measured: 94.7% of facts at ~4.5× our v2's tokens).

Code: `src/sec_financial_rag/exp_f/pipeline.py`, `scripts/exp_f_index.py`, `scripts/exp_f_eval.py`, `scripts/exp_f_change_handling.py`, `tests/test_exp_f_pipeline.py`. Dependencies in the `exp-f` group: `llama-index-core==0.14.25`, `llama-index-vector-stores-qdrant==0.10.4`, `llama-index-readers-docling==0.5.0` (no existing pin changed).

Artifacts (parsed Markdown, 1 GB docstore, the `expf_llamaindex_bge_m3` Qdrant collection) were deleted after the run to free space; to rebuild: `uv sync --group exp-f`, then `scripts/exp_f_index.py` (~12 min parse + ~60 min on a Modal L4, under $1 GPU).

## Results

n = 76 FinanceBench questions (EM on 51); comparisons paired, exact two-sided sign tests.

| | Hit@5 | Hit@10 | MRR@10 | EM lenient / strict | Fully faithful | Refusals | Input tokens / q | Cost / 1k | Total P95 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Exp-1 naive baseline | 0.24 | — | 0.14 | 0.20 / 0.20 | 0.78 | 0.66 | — | $5.63 | 29.7 s |
| Exp-2 | 0.32 | — | 0.24 | 0.29 / 0.27 | 0.79 | 0.53 | — | $4.10 | 16.1 s |
| **Exp-F LlamaIndex** | **0.21** | 0.28 | 0.14 | **0.20 / 0.18** | 0.80 | **0.62** | 1,929 | **$4.27** | 28.5 s |
| Exp-4 hybrid (ours, same stage of the roadmap) | 0.49 | 0.57 | 0.34 | 0.39 / 0.37 (repeat 0.43 / 0.39) | 0.87 | 0.33 | 9,671 | $10.22 | 15.3 s |
| Exp-5b (ours, best) | 0.63 | 0.71 | 0.42 | 0.49 / 0.43 | 0.75 | 0.22 | — | $11.77 | 23.3 s |

**1. Quality: Exp-1 level.** vs Exp-4: Hit@5 +1 / −22, EM +2 / −12 (p = 0.013) and +1 / −13 vs the repeat (p = 0.002), far outside the noise floor. vs Exp-1: EM +6 / −6. The top passage comes from the gold filing as often as in ours (62/76), but the passage is rarely the evidence page.

**2. The metric was checked before trusting the drop.** The relevance judge recognizes the evidence in some framework leaf for 64/76 questions (76/76 for our indexes) and in a parent node for 47/76: Docling's native tables repeat cells, which changes word overlap. Restricted to the 64 judgeable questions, Hit@5 is still 0.25 vs 0.53 (+1 / −19); counting all 12 unjudgeable questions as hits would give at most 0.37. The gap is real.

**3. Where the loss comes from** (retrieval-only ablation on the framework index, Hit@5):

| Variant | Hit@5 | Our equivalent |
| :--- | ---: | :--- |
| Dense only, leaves | 0.26 | dense on our children ~0.38 (Exp-3b child level) |
| BM25 only, leaves | 0.11 | our BM25 0.26 |
| Hybrid, relative-score fusion | 0.21 | our RRF hybrid 0.49 |
| + AutoMergingRetriever | 0.21 (merged 1 of 380 top-5 slots) | our parent expansion: always parents |

- **Chunks match worse:** native Markdown tables repeat merged cells and keep empty layout tables (241 MB of Markdown, 226,203 leaves vs our 76,788 children), and there are no SEC Item sections.
- **BM25 is weaker** (0.11 vs 0.26): likely causes are the missing number normalization ("14,205" becomes "14" and "205") and table noise; not isolated by a separate run.
- **The default fusion hurts:** min-max scaling gives a weak sparse list equal weight, so hybrid scores below dense alone; RRF (ours) is robust to that.
- **Auto-merging almost never fires** (it needs more than half of a parent's leaves retrieved), so Gemini gets five ~384-token leaves (1,929 input tokens per question vs 9,671): cheaper, but the missing context shows as refusals (0.62). Parent context was the one clearly significant gain in our roadmap (Exp-3, EM +9 / −1).

**4. Change handling.** One sentence edited in 3M's FY2022 10-K, Item 1A: ours re-embeds 30 of 594 children (5%, that section); `IngestionPipeline` with a docstore re-embeds all 2,253 nodes (document-level hash; parent nodes are embedded too by default). An unchanged re-run embeds nothing in either.

**5. Cost and speed.** Parse 11.5 min (local) + index 59.9 min (Modal L4, 226,203 leaves; Exp-3b: 57.2 min in total for 76,788 children). Peak memory 1.5 GB indexing, 1.4 GB querying: the default `SimpleDocumentStore` is one 1.08 GB JSON file loaded whole at start (our parents live in Postgres). Answer cost $4.27 / 1k, less than half of ours, because the context is a fifth the size.

**6. Code and debuggability.** ~120 lines for the framework arm vs ~1,300 for the same stages of ours (parsing, chunking, store and retrieval, ingestion/CDC; the shared Modal embedder excluded). Every cause above was found by an ablation run, not by reading the pipeline: fusion, tokenization and merging happen inside library classes with defaults that are not visible from the calling code.

## Hypotheses, checked

| Hypothesis | Result |
| :--- | :--- |
| Much less code | Yes: ~11× fewer lines |
| Quality near Exp-2/Exp-3, not Exp-4 | Worse than expected: Exp-1 level (EM 0.20, Hit@5 0.21) |
| Higher context cost from native tables | No: lower cost per question, because auto-merging sends leaves; the native tables cost index size instead (2.9× the leaves) |
| Coarser change detection | Yes: whole document (100%) vs one section (5%) |
| Harder diagnosis | Yes: causes needed ablations; defaults (fusion, merge threshold, tokenization, uuid ids, 10-text embedding batches) are invisible until measured |

## Conclusion

The framework gives a working hybrid RAG pipeline in ~120 lines, but with default settings its quality on SEC filings is the naive baseline's. What our custom work buys, in order of measured effect: parent context for the generator (Exp-3), table-aware chunks with SEC sections (Exp-2/3), rank-based fusion with number-aware BM25 (Exp-4), and section-level change detection (5% vs 100% re-embedded). Each of these can be configured in LlamaIndex (custom node parser, RRF fusion function, our tokenizer, always-expand to parents, per-section documents), but each first has to be found, which is the work this roadmap did. The answer to "why not just use LlamaIndex?" is: use it for plumbing, not for the domain decisions, and measure its defaults before trusting them.
