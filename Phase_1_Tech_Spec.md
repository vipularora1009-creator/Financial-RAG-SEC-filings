# Phase 1: Production-Grade Financial RAG Pipeline — Baseline Architecture & Dataset Specification

**Project Status:** Baseline Architectural Blueprint  
**Domain:** SEC EDGAR Financial Filings (10-K Annual & 10-Q Quarterly Reports)  
**Primary Goal:** Build an enterprise-grade, evaluation-driven Retrieval-Augmented Generation (RAG) system designed around structural document hostility, continuous temporal updates, hybrid lexical-semantic retrieval, and strict latency/accuracy ablation studies.

---

## 1. Executive Summary & Engineering Philosophy

Standard portfolio RAG tutorials fail to impress in technical interviews because they index clean narrative prose (e.g., Wikipedia articles) using naive fixed-size chunking and evaluate via manual "vibe checks." On clean prose, a 10-line wrapper already achieves >85% recall, leaving no room for engineering depth.

This system is engineered around **SEC 10-K and 10-Q financial filings**—a corpus characterized by **structural hostility**:
1. **Multi-Column Financial Tables:** Where 40%+ of critical numerical answers reside, and where naive token splitting slices rows apart from their column headers and unit multipliers (`$ in millions` vs. `$ in thousands`).
2. **High Inter-Document Boilerplate Similarity:** An annual report from Apple in FY2023 is ~80% lexically identical to FY2024. Pure semantic search without temporal metadata routing or hybrid keyword weighting retrieves figures from the wrong fiscal period.
3. **Cross-Referenced Footnotes:** Line items on the Consolidated Statement of Operations routinely depend on accounting definitions buried 40 pages deeper in *Note 12 (Commitments and Contingencies)* or *Note 15 (Segment Reporting)*.
4. **Multi-Hop Comparative Queries:** Answering questions like *"How did NVIDIA's data center gross margin trend compare to AMD's across FY2023–FY2025, and what supply chain constraints were cited by each?"* requires decomposing a single user prompt into targeted sub-queries across distinct entities and timeframes.

---

## 2. The Initial 10-Company Seed Corpus

To maximize experimental signal during initial pipeline construction and ablation benchmarking, the seed corpus spans **10 public companies across 3 cross-comparable industry clusters**. This selection is deliberately overlapped with the **Patronus AI FinanceBench** ground-truth dataset while introducing fiscal year-end mismatches and accounting edge cases.

| # | Company | Ticker | CIK | Sector Cluster | FY End Month | Why Selected (Architectural Stress Test) |
| :- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1** | **Apple Inc.** | `AAPL` | `0000320193` | Semiconductors & Hardware | September (52/53-wk) | Non-standard 52/53-week fiscal calendar ending late September; clean hardware/services segment tables. |
| **2** | **NVIDIA Corp.** | `NVDA` | `0001045810` | Semiconductors & Hardware | January | Shifted fiscal calendar (FY2025 ended Jan 2025); tests temporal normalization (`Calendar Q4 2024` = `NVDA Q4 FY25`). |
| **3** | **Advanced Micro Devices** | `AMD` | `0000002488` | Semiconductors & Hardware | December | Direct competitor to NVDA for multi-hop comparative queries; heavy acquisition amortization tables (Xilinx). |
| **4** | **Microsoft Corp.** | `MSFT` | `0000789019` | Cloud & Hyperscalers | June | Fiscal year ends June 30; complex revenue reallocation across *Intelligent Cloud* and *Productivity* segments. |
| **5** | **Alphabet Inc.** | `GOOGL` | `0001652044` | Cloud & Hyperscalers | December | Deep nested tables separating Google Services, Google Cloud, and Other Bets; intricate hedging/SBC footnotes. |
| **6** | **Amazon.com, Inc.** | `AMZN` | `0001018724` | Cloud & Hyperscalers | December | Massive free-cash-flow and lease obligation reconciliations; high contrast between North America retail and AWS margins. |
| **7** | **Tesla, Inc.** | `TSLA` | `0001318605` | Automotive, Industrials & Energy | December | Automotive regulatory credits, energy generation/storage segment splits, and supply-chain risk factor evolution. |
| **8** | **Ford Motor Company** | `F` | `0000037996` | Automotive, Industrials & Energy | December | Segmented reporting (*Ford Blue*, *Ford Model e*, *Ford Pro*) + Ford Credit financial services balance sheet complexity. |
| **9** | **JPMorgan Chase & Co.** | `JPM` | `0000019617` | Financials & Regulated | December | **Schema Stress Test:** Banks do not use standard "Cost of Goods Sold" income statements; tests retrieval on Net Interest Income (NII), Tier 1 Capital (CET1) ratios, and credit loss provisions. |
| **10** | **3M Company** | `MMM` | `0000066740` | Conglomerate / FinanceBench Anchor | December | Heavy staple in **FinanceBench** ground-truth QA pairs; dense litigation reserves (PFAS/Combat Arms) and spin-off restatements (Solventum). |

### Corpus Scope for Initial Build (v1.0)
* **Time Horizon:** 3 Fiscal Years (`FY2023`, `FY2024`, `FY2025` + available `2026` 10-Q quarterly filings).
* **Document Volume per Company:** Three `10-K` filings (~100–180 pages each) + Nine `10-Q` filings (~50–80 pages each) = **~120 core documents (~10,000–12,000 pages total)**.
* **Why This Size:** Large enough to break naive vector search and require metadata filtering/hybrid search, yet compact enough to re-index in under 15 minutes during chunking ablation experiments without excessive embedding costs.

---

## 3. System Architecture & End-to-End Pipeline

```
+-----------------------------------------------------------------------------------+
|                        1. CONTINUOUS INGESTION & CDC LAYER                        |
|  SEC EDGAR API / FinanceBench PDFs --> SHA-256 Section Hashing (PostgreSQL/SQLite)|
|  --> Delta Detection (New/Modified Filings Only) --> Layout-Aware Parser (Docling)|
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
|                       2. CHUNKING & DUAL-STORAGE INDEXING                         |
|  [Strategy A: Fixed Token]   [Strategy B: Semantic]   [Strategy C: Parent-Child]  |
|  - Child Nodes (Sentences/Table Rows + Context Header) --> Qdrant Vector Store    |
|  - Parent Nodes (Full Markdown Tables / Full MD&A Sections) --> DocStore (SQLite) |
|  - Embeddings: Dense (BGE-M3 / text-embedding-3-small) + Sparse (BM25 / SPLADE)   |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
|                        3. AGENT QUERY TRANSFORMATION LAYER                        |
|  User Query --> Query Classifier / Router                                         |
|    |-- Simple Factoid --> Metadata Extractor (Ticker, FY, Form, Section)          |
|    |-- Ambiguous/Conceptual --> HyDE / Step-Back Prompting                        |
|    +-- Multi-Entity / Multi-Period --> Sub-Query Decomposer (Parallel Execution)  |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
|                         4. HYBRID RETRIEVAL & RERANKING                           |
|  Pre-Filter (Ticker + Fiscal Period) --> Dense Search + Sparse BM25 Search        |
|  --> Reciprocal Rank Fusion (RRF, top-k=50) --> Parent-Document Expansion         |
|  --> Cross-Encoder Reranker (bge-reranker-v2-m3 / Cohere, top-n=5 to 8)           |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
|                       5. GENERATION & OBSERVABILITY HARNESS                       |
|  Strict Citation Synthesizer (Page + Item + Table Ref) --> Arize Phoenix Traces   |
|  --> CI/CD Evaluation Suite (Ragas + DeepEval + FinanceBench Exact Match)         |
+-----------------------------------------------------------------------------------+
```
### Terms Explainations 
1. INGESTION AND CDC LAYER
    1. SHA-256 Section Hashing - Converts the part of text into a token. Any change made to the text is reflected in this Hash. It will be used to check for the delta in case new document arrives without have to check for it.
    2. Delta detection - What changed in the data from the past data? Powered through the SHA-256 in above case
2. CHUNKING AND DUAL STORAGE
    1. Qdrant - An open sourced Vector Store and Search Database
3. QUERY TRANSFORMATION
    1. Factoid - a false state
---

## 4. Deep-Dive Component Specifications & Tradeoffs

### 4.1. Document Parsing & Table Serialization
Standard PDF extractors (`pypdf`, `pdfplumber`) flatten two-column layouts and scramble financial tables.
* **Primary Parser:** **`Docling`** (IBM) or **`MinerU`** for PDF filings, alongside custom **SEC EDGAR HTML/iXBRL parsers** (`sec-parser` / `edgar-tools`).
* **Table Handling Strategy:**
  * Every extracted table is serialized into clean **GitHub-Flavored Markdown** (or structured HTML) with its preceding caption, unit declaration (e.g., *"(in millions, except per share data)"*), and column headers intact.
  * **Contextual Header Injection:** Before embedding any chunk or table row, prepend a deterministic metadata breadcrumb string:
    ```text
    [COMPANY: NVIDIA Corp (NVDA) | FILING: 10-K | PERIOD: FY2025 (Ended Jan 26, 2025) | SECTION: Item 8 - Consolidated Statements of Income]
    ```
    *Why defend this in interviews:* Without contextual header injection, a chunk containing `"Research and development: $8,675 | $7,339 | $5,268"` has zero semantic connection to NVIDIA or FY2025 in embedding space.
* **Understanding and Experimentation Strategy**
    Take PDF documents with different layouts (not necessarily from the above corpus just random PDFs from the internet) and process them using the various parsers and observe when one gives what results and keep that as a understanding study for later references.

### 4.2. Continuous Data Ingestion & Change Detection (CDC)
In production, new 10-Q and 8-K filings arrive daily, and companies occasionally issue **10-K/A (Amended)** restatements.
* **State Ledger (PostgreSQL / SQLite):** Maintain an `ingestion_ledger` table tracking:
  * `accession_number` (Unique SEC filing ID, e.g., `0001045810-25-000014`)
  * `company_ticker`, `form_type`, `filing_date`, `period_of_report`
  * `section_id` (e.g., `Item_1A`, `Item_7`, `Item_8_Note_4`)
  * `content_sha256` (SHA-256 hash of normalized section text)
  * `vector_point_ids` (Array of UUIDs stored in Qdrant)
* **Incremental Upsert Logic:**
  1. When a filing is ingested, parse it into canonical SEC Items (`Item 1` through `Item 15`).
  2. Compute `SHA-256` for each section. Compare against the state ledger.
  3. **Unchanged Hashes:** Skip embedding completely (0 token cost).
  4. **Modified/Amended Sections (e.g., 10-K/A restatement):** Delete old `vector_point_ids` from Qdrant, mark prior ledger entries as `is_superseded = True`, and upsert only the modified delta chunks.

### 4.3. Chunking Strategies (The Core Ablation Study)
You will implement a modular `Chunker` interface supporting three interchangeable strategies to benchmark against one another:

1. **Strategy 1 — Fixed-Size Token Chunking (Baseline):**
   * `chunk_size = 512 tokens`, `chunk_overlap = 64 tokens` using `tiktoken` (`cl100k_base`).
   * *Expected Failure Mode:* Slices mid-table, separates financial figures from column headers, and pollutes context with adjacent unrelated paragraphs.
2. **Strategy 2 — Semantic & Layout-Aware Chunking:**
   * Splits on Markdown structural boundaries (`#`, `##`, `###`, table blocks) first. For long narrative sections (like *Item 1A: Risk Factors* or *Item 7: MD&A*), computes cosine similarity between adjacent sentence embeddings and splits where similarity drops below the 85th percentile breakpoint threshold.
   * Tables remain atomic unless they exceed 1,500 tokens, in which case they are split into logical row groups with the **original table header row repeated at the top of every sub-chunk**.
3. **Strategy 3 — Hierarchical Parent-Document Retrieval:**
   * **Parent Nodes (1,500–2,500 tokens):** Full SEC Sub-sections or complete Markdown tables + footnotes stored in a fast key-value Document Store (SQLite/PostgreSQL).
   * **Child Nodes (256–400 tokens):** Granular paragraph chunks, individual table row summaries, and LLM-generated table synopsis cards indexed in Qdrant with a `parent_id` foreign key.
   * *Retrieval Mechanics:* Vector search matches high-precision Child Nodes; the system deduplicates `parent_id` references and passes the complete Parent Nodes to the Reranker/LLM.

### 4.4. Metadata Schema & Fiscal Calendar Normalization
A critical domain bug in financial RAG is the **Fiscal vs. Calendar Year mismatch** (e.g., Microsoft's FY2024 ended June 30, 2024; Apple's FY2024 ended September 28, 2024; NVIDIA's FY2025 ended January 26, 2025).
* Every Qdrant payload must enforce a strict Pydantic schema:
  ```python
  class FinancialChunkMetadata(BaseModel):
      chunk_id: str
      parent_id: Optional[str]
      ticker: str                    # e.g., "NVDA"
      cik: str                       # e.g., "0001045810"
      company_name: str
      form_type: Literal["10-K", "10-Q", "8-K"]
      fiscal_year: int               # e.g., 2025
      fiscal_quarter: Literal["Q1", "Q2", "Q3", "Q4", "FY"]
      calendar_year_mapped: int      # Normalized calendar year for cross-company comparison
      calendar_quarter_mapped: str   # e.g., "2024-Q4"
      filing_date: str               # ISO-8601 "2025-02-26"
      sec_item: str                  # e.g., "Item 7", "Item 8"
      contains_table: bool
      page_number: int
      content_hash: str
  ```

### 4.5. Query Transformation & Routing Layer
Before touching the vector database, an LLM-based structured query router (`instructor` / Pydantic structured outputs) classifies the incoming user query into one of three execution paths:

1. **Direct Metadata-Filtered Retrieval:**
   * *Query:* *"What was Apple's total revenue in FY2024?"*
   * *Action:* Extracts `{ticker: "AAPL", fiscal_year: 2024, form_type: "10-K", sec_item: ["Item 7", "Item 8"]}` and executes a single pre-filtered hybrid search.
2. **Step-Back / HyDE Rewriting (For Conceptual or Risk Queries):**
   * *Query:* *"How exposed is Tesla to lithium price volatility and tariff shifts?"*
   * *Action:* Generates a hypothetical SEC *Item 1A Risk Factor* / *Item 7A Quantitative Market Risk* disclosure paragraph (HyDE) or a broader step-back query to bridge vocabulary gaps between casual phrasing and formal SEC legalese.
3. **Sub-Query Decomposition (For Multi-Hop / Comparative Queries):**
   * *Query:* *"Compare Microsoft and Alphabet's capex growth from 2023 to 2025 and explain how each company justified the spend."*
   * *Action:* Decomposes into 4 parallel, isolated sub-queries:
     * `Sub-Q1:` MSFT Capital Expenditures FY2023, FY2024, FY2025 (`ticker="MSFT"`, `contains_table=True`)
     * `Sub-Q2:` GOOGL Capital Expenditures FY2023, FY2024, FY2025 (`ticker="GOOGL"`, `contains_table=True`)
     * `Sub-Q3:` MSFT MD&A commentary on capital expenditure drivers (`ticker="MSFT"`, `sec_item="Item 7"`)
     * `Sub-Q4:` GOOGL MD&A commentary on capital expenditure drivers (`ticker="GOOGL"`, `sec_item="Item 7"`)
   * Executes all 4 in parallel via `asyncio.gather`, reranks each sub-query's results independently (`top-k=3` per sub-query), and merges the context for final synthesis.

### 4.6. Hybrid Retrieval (Dense + Sparse RRF) & Cross-Encoder Reranking
* **Vector Database:** **Qdrant** (running via local Docker container).
* **Hybrid Search Configuration:**
  * **Dense Vector:** `BAAI/bge-m3` (1024-dim, local) or `text-embedding-3-small` (1536-dim). Captures semantic relationships.
  * **Sparse Vector:** BM25 / SPLADE sparse vectors natively indexed in Qdrant. Captures exact lexical tokens (`ASC 606`, `EBITDA`, `10-K`, `$14,205`, specific executive or subsidiary names).
  * **Fusion:** **Reciprocal Rank Fusion (RRF)** combining top-30 dense and top-30 sparse hits into a unified top-50 candidate pool:
    $$\text{RRF\_Score}(d \in D) = \sum_{m \in \{\text{dense, sparse}\}} \frac{1}{k + r_m(d)} \quad (k = 60)$$
* **Second-Stage Cross-Encoder Reranking:**
  * Pass the top-25 fused candidates (after Parent-Document expansion) through **`BAAI/bge-reranker-v2-m3`** (or Cohere `rerank-v3.5`).
  * Compress the final context window down to the **top-5 highest-scoring chunks**, applying a minimum relevance score threshold to trigger an explicit *"Insufficient filing data to answer"* fallback when appropriate.

---

## 5. Evaluation Strategy & Experimental Matrix

To defend this project as production-grade, every architectural choice is validated against a **200-Question Golden Evaluation Dataset**:
* **100 Curated Questions from `PatronusAI/financebench`** (covering your anchor companies like 3M, Apple, Microsoft, AMD, Amazon, Tesla, JPMorgan).
* **50 Adversarial Numerical & Table-Lookup Questions** (requiring exact cell extraction and unit verification across 10-K and 10-Q tables).
* **50 Multi-Hop Cross-Company & Temporal Questions** (requiring sub-query decomposition across 2+ companies or 3+ quarters).

### 5.1. Metrics Tracked (`evals/` Harness)
1. **Retrieval Metrics (Deterministic — Fast & Free):**
   * **Hit Rate @ K (`K=3, 5, 10`):** Did the ground-truth filing page/section appear in the top-K retrieved chunks?
   * **MRR @ 10 (Mean Reciprocal Rank):** How high up was the exact ground-truth table/section ranked?
   * **NDCG @ 10 (Normalized Discounted Cumulative Gain):** Measures graded ranking quality across multi-chunk answers.
2. **Generation & End-to-End Metrics (LLM-as-a-Judge via `DeepEval` / `Ragas`):**
   * **Context Precision & Context Recall**
   * **Faithfulness (Hallucination Rate):** Strict check that every dollar figure or percentage in the response exists verbatim or via explicit arithmetic from the retrieved chunks.
   * **Numerical Exact Match / F1:** Automated regex/value check against FinanceBench ground-truth numbers (accounting for rounding tolerances like `$14.2B` vs `$14,205M`).
3. **Systems & Operational Metrics:**
   * **P50 / P95 End-to-End Latency (ms)** broken down by stage (`Query Rewrite`, `Vector Search`, `Reranking`, `LLM Synthesis`).
   * **Mean Token Cost ($) per Query** and **Index Storage Footprint (MB)**.

### 5.2. The 5-Stage Ablation Study Table (To Populate in Your README)
Run `make evaluate` across all 5 configurations and commit the resulting table to your repository's root `README.md`:

| Exp ID | Pipeline Configuration | Hit Rate@5 | MRR@10 | Faithfulness | Numerical EM | P95 Latency | Cost / 1k Queries |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Exp-1** | **Baseline:** PyPDF + Fixed 512-token chunks + Dense-Only Cosine Search (`top-k=5`) — *as run: naive HTML text (stand-in for PyPDF), bge-m3 fp16, gemini-3.8-flash; see §5.3* | 0.24 | 0.14 | 0.96 mean / 78% fully grounded (n=23; metric v2, was 0.93 / 65%) | 0.20 lenient / 0.20 strict (n=51) | 29.7 s | $5.63 |
| **Exp-2** | **+ Layout Parsing:** Docling Markdown Tables + Contextual Header Injection + Semantic Chunking — *as run: Docling HTML + compact tables, SEC Item sections, bge-m3 fp16, gemini-3.8-flash; see §5.3* | 0.32 | 0.24 | 0.96 mean / 79% fully grounded (n=33; metric v2, was 0.94 / 73%) | 0.29 lenient / 0.27 strict (n=51) | 16.1 s | $4.10 |
| **Exp-3** | **+ Parent-Document Retrieval:** Child row/sentence embeddings -> Parent section/table expansion — *as run: ~300-token children in Qdrant, ~1,600-token parents in Postgres, top-50 children → top-5 parents; see §5.3* | 0.42 (child level 0.34) | 0.29 (child level 0.22) | 0.96 mean / 74% fully grounded (n=46; metric v2) | 0.45 lenient / 0.39 strict (n=51) | 21.8 s | $10.90 |
| Exp-3b | Exp-3 with table serializer v2 (ADR-001 update), embedded on a Modal L4 GPU — *single-change re-run; see §5.3* | 0.46 (child level 0.38) | 0.31 (child level 0.25) | 0.96 mean / 77% fully grounded (n=46; metric v2) | 0.45 lenient / 0.39 strict (n=51) | 24.9 s | $11.11 |
| **Exp-4** | **+ Hybrid & Reranking:** Dense + BM25 (RRF `top-25`) -> Cross-Encoder Reranker (`top-5`) — *as run: on Exp-3b's index, BM25 (fastembed + Qdrant IDF), RRF k=60 over top-30 + top-30, bge-reranker-v2-m3 on Modal over the top-25 parents, exact search; see §5.3* | 0.42 (child level 0.43); hybrid without reranker 0.49 | 0.26; hybrid without reranker 0.34 | 0.97 mean / 84% fully grounded (n=44) | 0.39 lenient / 0.33 strict (n=51); hybrid without reranker 0.39 / 0.37 | 17.6 s | $10.54 |
| **Exp-5** | **+ Agentic Query Router:** Metadata Pre-filtering + Sub-Query Decomposition + HyDE | *TBD* (Exp-5a below) | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* |
| Exp-5a | Exp-4 hybrid (no reranker) + metadata pre-filtering: Gemini structured-output extraction of ticker and year, hard filter on ticker, year ∈ {y−1, y} — *see §5.3* | 0.55 (child level 0.46) | 0.37 (child level 0.28) | 0.97 mean / 80% fully grounded (n=45) | 0.51 lenient / 0.45 strict (n=51) | 24.7 s | $10.80 |
| Exp-F | Framework baseline: LlamaIndex standard components (DoclingReader, HierarchicalNodeParser, QdrantVectorStore hybrid, AutoMergingRetriever) with our questions, prompt, model and metrics — *see `docs/experiments/exp-F-framework-baseline.md`* | 0.21 | 0.14 | 0.91 mean / 80% fully grounded | 0.20 lenient / 0.18 strict (n=51) | 28.5 s | $4.27 |
| Exp-5b | Exp-5a + HyDE: Gemini writes a ~130-word filing-style passage, embedded for the dense side only (BM25 keeps the question) — *see §5.3* | 0.63 (child level 0.47) | 0.42 (child level 0.30) | 0.97 mean / 75% fully grounded (n=53) | 0.49 lenient / 0.43 strict (n=51) | 23.3 s | $11.77 |

### 5.3. Experiment Log

#### Exp-1: Naive Baseline (run 2026-10-24)

**Pipeline as run**

| Stage | Choice | Code |
| :--- | :--- | :--- |
| Corpus | 209 filings: 10 seed companies (10-K/10-Q since 2022) + 30-filing FinanceBench slice | `ingestion/edgar_downloader.py` |
| Change detection | Postgres ledger: file SHA-256 per filing, text SHA-256 per (collection, filing, section) | `ingestion/cdc_ledger.py` |
| Parsing | Naive visible-text HTML extraction (BeautifulSoup); tables flattened one cell per line. Stands in for PyPDF (the filings are HTML) | `ingestion/naive_parser.py` |
| Chunking | Fixed 512 tokens, 64 overlap, tiktoken `cl100k_base` → 30,078 chunks | `chunking/fixed_token.py` |
| Embedding | `BAAI/bge-m3`, fp16, on an M1 (8 GB) via sentence-transformers; full index 117 min | `ingestion/indexer.py` |
| Vector store | Qdrant collection `exp1_fixed512_bge_m3`, cosine, deterministic uuid5 point ids | `retrieval/qdrant_store.py` |
| Retrieval | Dense only; top-10 for retrieval metrics, top-5 passed to the generator | `evals/runner.py` |
| Generation | `gemini-3.8-flash`, temperature 0, cite-or-refuse prompt ("Insufficient filing data to answer") | `generation/` |

**Evaluation set and metric definitions**
- 76 FinanceBench questions (FY2022+, 10-K/10-Q) with evidence passages. The 100 custom questions are not built yet.
- Relevant chunk = same filing AND overlap coefficient ≥ 0.5 on 5-word shingles with the evidence. Judge ceiling: 76/76 questions have at least one acceptable chunk.
- Numerical EM: every number in the gold answer is found in ours, with precision-aware rounding. Lenient also allows a missing unit; strict does not. Applies to the 51 questions with numeric gold answers.
- Faithfulness: share of the answer's numbers found in the retrieved chunks or the question, or shown as the result of arithmetic ("= x"). Negative-control chance-grounding rate is about 6% of answers.
- *Metric v2 (2026-10-25, during Exp-3):* a result written in bold (`= **36.8%**`) and a headline that repeats a shown result ("ratio is 0.54 … = 0.54") also count as grounded. All runs were re-scored from their saved contexts (`evals/rescore.py`); Exp-1 becomes 0.96 / 0.78. The results tables below keep the v1 numbers as run.

**Results**

| Hit@3 | Hit@5 | Hit@10 | MRR@10 | EM lenient / strict | Faithfulness mean / fully | Refusal rate | Cost / 1k | Retrieval P95 | Total P95 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 0.17 | 0.24 | 0.36 | 0.14 | 0.20 / 0.20 | 0.93 / 0.65 | 0.66 | $5.63 | 0.62 s | 29.7 s |

**Failure analysis**
1. **Right document, wrong passage.** The correct filing is in the top 5 for 65 of 76 questions, and 4.2 of the 5 top chunks come from the right company on average. Yet Hit@5 is only 0.24. Dense search matches company, year and form, but flattened tables lose their row labels, column periods and units, so the specific passage isn't found. One example: "Apple total net sales, quarter ended June 25, 2022" ranked an EPS/net-income table first because it matched the dates.
2. **Refusals track retrieval misses.** 47 of the 58 retrieval misses ended in "Insufficient filing data to answer" rather than a guess. 3 of the 18 retrieval hits were refused, and 7 answers were wrong.
3. **Faithfulness has to be read with the refusal rate.** 0.93 covers only the answered questions; 66% were refused.
4. **No unit-loss gap.** Lenient EM = strict EM, which goes against the hypothesis that naive chunks would make answers lose their "(in millions)" units.
5. **A known metric false negative.** Percentages whose "%" appears only in a column header (Amcor's "% of sales" column: 19.4, 18.5) count as ungrounded, so faithfulness is slightly under-stated.
6. **Cost and latency are driven by generation.** On average 3,361 input and 828 output tokens per question, mostly thinking tokens, which is 2.5× the pre-run estimate. Generation dominates the 29.7 s P95.

**Deviations from this spec**
- 76 FinanceBench questions instead of 100: too few in-scope FinanceBench questions exist without ingesting 8-Ks.
- HTML text extraction instead of PyPDF.
- Generation through the Gemini API instead of a local LLM, because of the 8 GB RAM limit.
- The "<2 s re-run" ingest target was dropped; a re-run takes about 6 s, mostly the SEC index requests.

**Expected to change in Exp-2:** layout-aware tables with headers kept, plus contextual header injection, should mainly improve failure 1 (passage selection inside the right filing).

#### Exp-2: Layout Parsing + Header Injection + Semantic Chunking (run 2026-10-25)

**Pipeline as run** (changes from Exp-1 only)

| Stage | Choice | Code |
| :--- | :--- | :--- |
| Parsing | Docling (HTML backend) reads the filing; our serializer turns each table grid into compact Markdown (spanning headers written once, spacer and "$" columns merged, one column per period). Link-split paragraphs rejoined; page footers, page numbers and "Table of Contents" removed. 0 of 153,230 figures lost vs the naive parser | `ingestion/layout_parser.py` |
| Sections | Regex SEC Item splitter (table of contents skipped, 10-Q Part I/II): 41/43 sample filings split; all 14 JPM 10-Qs land in `Preamble` (no Part I Item headings) | `ingestion/layout_parser.py` |
| Change detection | Ledger row per (collection, filing, Item section); hash covers header + section text | `ingestion/indexer_exp2.py` |
| Fiscal period | Read from each filing's iXBRL cover tags (`dei:DocumentFiscalYearFocus/PeriodFocus`); 209/209 tagged | `ingestion/fiscal_normalizer.py` |
| Header injection | `[COMPANY \| FILING \| PERIOD \| SECTION]` embedded with every chunk (median 52 tokens); payload `text` stores the body only | `chunking/header.py` |
| Chunking | Strategy 2: blocks packed to 416 tokens (512 − 96 header reserve); tables whole up to 1,500 tokens with their title/units lines, bigger tables split by rows with the header repeated; Item 1A / Item 7 (and 10-Q equivalents) cut at semantic breakpoints (bge-m3 sentence vectors, per-section 85th-percentile distance, topics < 100 tokens merged forward) → 46,329 chunks, median 328 tokens | `chunking/semantic_table.py`, `chunking/sentences.py` |
| Embedding / store / retrieval / generation | Unchanged: bge-m3 fp16, Qdrant `exp2_semantic_bge_m3`, dense top-10 / top-5 to the generator, same prompt and model | `ingestion/indexer_exp2.py` |

Index time: 158 min on the M1 (8 GB), across three runs. The first two were stopped by memory pressure (swap); releasing PyTorch's MPS cache after each filing kept GPU memory flat at 1.2 GB.

**Results** (same 76 questions and metric definitions as Exp-1; judge ceiling still 76/76)

| | Hit@3 | Hit@5 | Hit@10 | MRR@10 | EM lenient / strict | Faithfulness mean / fully | Refusal rate | Cost / 1k | Retrieval P95 | Total P95 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Exp-1 | 0.17 | 0.24 | 0.36 | 0.14 | 0.20 / 0.20 | 0.93 / 0.65 | 0.66 | $5.63 | 0.62 s | 29.7 s |
| Exp-2 | 0.30 | 0.32 | 0.50 | 0.24 | 0.29 / 0.27 | 0.94 / 0.73 | 0.53 | $4.10 | 0.49 s | 16.1 s |

**Analysis**
1. **Every metric moved the right way, but n is small.** Paired by question, Exp-2 gained 10 Hit@5 questions and lost 4; for EM it gained 7 and lost 2. A sign test on those discordant pairs gives p ≈ 0.18 for both, so the improvement is consistent but not yet statistically significant on 76 questions. The 100 custom questions are needed to confirm it.
2. **Finding the right filing improved; picking the passage is still the bottleneck.** The right filing is in the top 5 for 73/76 questions (Exp-1: 65), and 3.9 of the top-5 chunks come from it (Exp-1: 3.0), consistent with the header carrying company and period into the vector. Yet the right passage is in the top 5 for only 24 questions.
3. **Refusals still track retrieval misses**: 37 of 52 misses refused; 3 of 24 hits refused; 9 answered wrong (Exp-1: 7).
4. **Cost and latency fell** because chunks are smaller: 2,216 input / 651 output tokens per question (Exp-1: 3,362 / 828).
5. **A small unit-loss gap appeared** (EM lenient 0.29 vs strict 0.27, one question), where Exp-1 had none.
6. **Attribution:** as the spec defines it, Exp-2 changes parser, sections, header and chunker together, and chunks also got smaller (median 328 vs 512 tokens). Which component caused the gain is not separable from this run; a with/without-header run would isolate the header (re-embedding only, no generation cost).

**Expected to change in Exp-3:** parent-document retrieval (small child chunks for matching, whole sections or tables passed to the LLM) targets finding 2: the right filing is already found, and the LLM would see the full table or sub-section around a matched row.

#### Exp-3: Parent-Document Retrieval (run 2026-10-25)

**Pipeline as run** (changes from Exp-2 only)

| Stage | Choice | Code |
| :--- | :--- | :--- |
| Parents | Exp-2's chunker makes 512-token units (tables whole up to 2,500 tokens, topic breakpoints in narrative Items); consecutive units are packed up to 2,000 tokens, so cuts fall between units and a table shares its parent with the surrounding text. Never crosses an Item boundary → 10,711 parents, median 6,350 chars (~1,600 tokens) | `chunking/parent_child.py` |
| Children | Each parent re-split at 384 tokens (288 body + header reserve); tables bigger than that become row groups with the header row repeated (deterministic "row summaries"; LLM table synopsis cards deferred to a separate ablation) → 73,385 children, median ~240 tokens | `chunking/parent_child.py` |
| Doc store | Postgres table `parent_documents` next to the CDC ledger, committed in the same transaction as the ledger row; ids are uuid5 over (collection, filing, section, index, section sha). On a changed section, old parents are deleted only after the old children. Integrity check: 0 children without a parent, 0 orphan parents | `retrieval/doc_store.py` |
| Indexing | Children embedded as header + child (bge-m3 fp16), `parent_id` in the Qdrant payload, collection `exp3_parent_child_bge_m3` | `ingestion/indexer_exp3.py` |
| Retrieval | Top-50 children → deduplicate by `parent_id` in rank order (a parent ranks at its best child: max aggregation) → top-10 parents for metrics, top-5 to the generator | `retrieval/doc_store.py`, `evals/runner.py --parents` |
| Generation | Unchanged prompt and model | `generation/` |

Index time: 260.8 min on the M1 (8 GB) in one run (swap peaked at ~8.7 of 9.2 GB; GPU memory flat at 1.2 GB).

**Results** (same 76 questions; judge ceiling 76/76 for children and for parents; faithfulness = metric v2 for every row)

| | Hit@3 | Hit@5 | Hit@10 | MRR@10 | EM lenient / strict | Faithfulness mean / fully | Refusal rate | Cost / 1k | Retrieval P95 | Total P95 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Exp-1 | 0.17 | 0.24 | 0.36 | 0.14 | 0.20 / 0.20 | 0.96 / 0.78 (n=23) | 0.66 | $5.63 | 0.62 s | 29.7 s |
| Exp-2 | 0.30 | 0.32 | 0.50 | 0.24 | 0.29 / 0.27 | 0.96 / 0.79 (n=33) | 0.53 | $4.10 | 0.49 s | 16.1 s |
| Exp-3, children only (ablation) | 0.28 | 0.34 | 0.42 | 0.22 | 0.25 / 0.22 | 0.97 / 0.91 (n=34) | 0.51 | $4.27 | 0.62 s | 28.9 s |
| **Exp-3, parents** | 0.33 | **0.42** | 0.55 | 0.29 | **0.45 / 0.39** | 0.96 / 0.74 (n=46) | **0.32** | $10.90 | 0.65 s | 21.8 s |

Child-level metrics of the parents run equal the children-only run (same search).

**Analysis**
1. **First statistically significant gain.** Paired with Exp-2, EM gained 9 questions and lost 1 (exact sign test p = 0.021); Hit@5 gained 12 and lost 4 (p = 0.077).
2. **The gain comes from context, not matching.** The children-only ablation changes only what the LLM receives: matching is identical (child Hit@5 0.34 vs Exp-2's 0.32, +8/−6, p = 0.79), and its EM is no better than Exp-2 (0.25, +3/−5). Passing the parents lifts EM to 0.45 and cuts refusals from 53% to 32%: Gemini now sees whole tables and their notes instead of declining.
3. **Parent Hit@k is lenient by construction**: a ~1,600-token parent contains evidence near the matched child. The EM gain is what shows the extra context is used, not only counted.
4. **Cost of the context**: 9,823 input / 942 output tokens per question (Exp-2: 2,216 / 651), so 2.7× the cost per 1k questions and +5.7 s P95. The children-only run's 28.9 s P95 at similar token counts suggests total P95 at n=76 is dominated by Gemini latency variance.
5. **Faithfulness**: the v1 metric showed a drop to 0.54 fully grounded; 11 of the 21 flagged answers were EM-correct, mostly bold or repeated arithmetic results (fixed in metric v2). Under v2, Exp-3 is 0.74 fully grounded on 46 answers with numbers vs Exp-2's 0.79 on 33. Remaining flags include the known "%"-in-column-header false negative and at least one invented figure ("64 consecutive years" of dividends since 1916).
6. **Attribution**: unlike Exp-2, this experiment changes one component (the retrieval unit), and the ablation splits it into matching vs context.

#### Exp-3b: Table serializer v2 (run 2026-10-25)

**Change from Exp-3 (only):** tables serialized with v2 (ADR-001 update: "$" aligned with plain rows, header rows above the first value row, bare years as headers, same-name columns merged); 85.0% vs 67.6% of label–period–value facts kept on the table-fidelity benchmark, at 11% fewer table tokens. Prose breakpoints reuse Exp-3's cached sentence vectors, so tables are the only difference. Embedded on a Modal L4 GPU (CUDA vs MPS vectors: min cosine 0.9995 on 300 children, the same order as MPS run-to-run differences). Index: 76,788 children, 10,570 parents, 57.2 min (Exp-3 locally: 260.8 min). Judge ceiling 76/76.

| | Hit@5 (child) | MRR@10 (child) | EM lenient / strict | Fully faithful | Refusals | Cost / 1k | Retrieval P95 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Exp-3 | 0.42 (0.34) | 0.29 (0.22) | 0.45 / 0.39 | 0.74 (n=46) | 0.32 | $10.90 | 0.36 s* |
| Exp-3b | 0.46 (0.38) | 0.31 (0.25) | 0.45 / 0.39 | 0.77 (n=46) | 0.33 | $11.11 | 0.18 s* |

\* Re-measured back to back with `--retrieval-only`; the generation run logged 3.15 s for Exp-3b because the machine was still swapping after the index run (total P95 24.9 s includes that).

**Analysis**
1. **No end-to-end gain at n = 76.** EM is unchanged (+2 / −2 questions); Hit@5 +7 / −4 (p = 0.55) and child Hit@5 0.34 → 0.38 (+7 / −4) point the right way but are not significant.
2. **Offline fidelity and end-to-end accuracy disagree in size**: +17 points of table facts kept, ~0 EM. Likely reasons, untested: Gemini already reconstructs misaligned columns when the whole parent table is present, and most FinanceBench evidence tables were not the badly affected ones (MSFT-style layouts).
3. **v2 is kept**: no metric got worse, tables are smaller, and the structure is correct when read by eye. Exp-4 builds on the Exp-3b index.

**Expected to change in Exp-4:** hybrid dense + BM25 retrieval with a cross-encoder reranker targets matching, which Exp-3 left unchanged (child Hit@5 0.34): exact tokens such as line-item names, tickers and figures are where dense-only search is weakest. The reranker also lets fewer, better parents reach the LLM, which would bring the context cost down.

#### Exp-4: Hybrid retrieval + cross-encoder reranking (run 2026-10-25/26)

**Pipeline as run** (changes from Exp-3b only)

| Stage | Choice | Code |
| :--- | :--- | :--- |
| Index | `exp4_hybrid_bge_m3`: Exp-3b's 76,788 children copied (dense vectors, payloads, ids) with a BM25 sparse vector added; 5.6 min, 505 MB | `ingestion/build_exp4.py` |
| BM25 | fastembed `Qdrant/bm25` term frequencies (k1 1.2, b 0.75, avg_len 122 measured) + Qdrant IDF; figures kept as one token ("14,205" -> "14205", "16.52" -> "16_52") | `retrieval/bm25.py` |
| Fusion | Reciprocal Rank Fusion, k = 60, over dense top-30 + BM25 top-30, top-50 kept, then parent expansion | `retrieval/hybrid.py` |
| Reranker | `BAAI/bge-reranker-v2-m3` (fp16, Modal L4) scores "header + parent" for the top-25 parents, max 2,048 tokens | `retrieval/rerank.py`, `ingestion/modal_embedder.py` |
| Search | Exact (brute force) for all Exp-4 comparisons: identical copies of a collection gave identical top-50 lists 76/76 with exact search but 46/76 with approximate HNSW search | `--exact` |

**Results** (n = 76; judge ceiling 76/76; 5 parents to the generator)

| | Hit@3 | Hit@5 (child) | Hit@10 | MRR@10 | EM lenient / strict | Fully faithful | Refusals | Cost / 1k | Retrieval P95 | Total P95 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Exp-3b (dense) | 0.36 | 0.46 (0.38) | 0.59 | 0.31 | 0.45 / 0.39 | 0.77 (n=46) | 0.33 | $11.11 | — | 24.9 s |
| BM25 only (retrieval) | 0.21 | 0.26 (0.17) | 0.39 | 0.18 | — | — | — | — | 0.12 s | — |
| Hybrid, run 1 | 0.43 | 0.49 (0.43) | 0.57 | 0.34 | 0.39 / 0.37 | 0.87 (n=45) | 0.33 | $10.22 | 0.88 s | 15.3 s |
| Hybrid, run 2 (identical repeat) | 0.43 | 0.49 (0.43) | 0.57 | 0.34 | 0.43 / 0.39 | 0.85 (n=47) | 0.30 | $10.64 | 1.53 s | 25.9 s |
| **Hybrid + rerank** | 0.33 | 0.42 (0.43) | 0.61 | 0.26 | 0.39 / 0.33 | 0.84 (n=44) | 0.36 | $10.54 | 4.19 s | 17.6 s |

**Noise floor** (hybrid run twice, identical configuration): retrieval is deterministic (76/76 questions sent Gemini the same context), but only 21/76 answers were word-for-word identical. EM flipped on 4 questions (+3 / −1; 0.39 vs 0.43), refusals on 4 (+1 / −3), and total P95 latency moved from 15.3 s to 25.9 s. At n = 76, EM differences of about ±0.04 and latency differences of several seconds are within run-to-run noise.

**Analysis**
1. **Hybrid improves the top of the ranking, not coverage.** vs dense on the same index (exact): Hit@3 +10 / −5 (p = 0.30), MRR 0.31 -> 0.34, Hit@10 +5 / −6. BM25 alone is far weaker but finds 5 questions dense misses in its top 10 (exact boilerplate wording such as "debt securities registered to trade on a national securities exchange"); hybrid keeps 4 of them.
2. **No end-to-end gain.** Exp-3b -> hybrid EM +2 / −5 (p = 0.45); hybrid -> rerank EM +5 / −5. Both are the size of the noise floor (4 flips between identical runs). Neither component changes answer accuracy measurably on this set.
3. **The reranker lowers Hit@k but not EM.** It keeps the top-1 from the right filing as often (62/76) and puts more right-filing passages in the top 5 (3.74 vs 3.30), but often ranks a different passage of that filing above FinanceBench's single evidence page, which the judge does not credit. It adds ~3.3 s retrieval P95 and GPU cost for no measured gain.
4. **Earlier comparisons, re-read against the noise floor:** Exp-3 vs Exp-2 (EM +9 / −1, p = 0.021) is well above it and stands; Exp-3b vs Exp-3 (+2 / −2) and every Exp-4 comparison are within it. Total-P95 differences between runs are not meaningful at n = 76.
5. **Cost**: hybrid adds no Gemini cost (same 5 parents); the reranker adds GPU time. Cutting context (top-2/3 parents) was planned but not run, since the reranker did not rank the evidence page higher.

**Expected to change in Exp-5:** a query router with metadata pre-filtering (company, fiscal period, form type from the question) targets what neither dense, BM25 nor the reranker handles reliably: picking the right filing among near-identical quarterly text.

#### Exp-5a: Metadata pre-filtering (run 2026-10-26)

**Change from Exp-4 hybrid (only):** before searching, Gemini (`gemini-3.8-flash`, structured output, thinking LOW) reads the question into `QueryMetadata(ticker, fiscal_year, quarter)`; the ticker is an enum of the 30 indexed companies. Code turns it into a Qdrant filter applied inside both the dense and the BM25 search (pre-filtering, payload indexes on ticker / fiscal_year / form_type / fiscal_period): ticker hard; fiscal_year ∈ {y−1, y} (y−1 covers forecasts, e.g. Boeing's FY2023 outlook in the FY2022 10-K, and fiscal-year naming, e.g. Ulta's year ending January 2023); quarter not used; nothing extracted → no filter. Extractions are cached per question (`data/cache/exp5_extractions.jsonl`) so repeat runs differ only in retrieval and generation; their latency and cost are added to the totals. Code: `retrieval/query_router.py`, `retrieval/hybrid.py` (`where=`), `evals/runner.py --route`, `scripts/router_accuracy.py`.

**Extractor on its own** (n = 76, $0.03): ticker 75/76 right, 1 empty (a question that names no company); year named 64, equal to the gold filing's fiscal_year 62 (the 2 others: Boeing, Ulta); **the filter keeps the gold filing 76/76** (a quarter filter would keep 75/76: "Q2 of FY2022" for CVS is answered by the FY2022 10-K, so it was dropped). Search space: median 1 filing (exactly 1 for 48/76 questions), median 528 of 76,788 children. Latency median 2.3 s, P95 3.8 s.

**Results** (n = 76; exact search; 5 parents to the generator)

| | Hit@3 | Hit@5 (child) | Hit@10 | MRR@10 | All 5 parents from gold filing | EM lenient / strict | Fully faithful | Refusals | Cost / 1k | Retrieval P95 | Total P95 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Exp-4 hybrid, run 1 / run 2 | 0.43 | 0.49 (0.43) | 0.57 | 0.34 | 30/76 | 0.39 / 0.37; 0.43 / 0.39 | 0.87; 0.85 | 0.33; 0.30 | $10.22; $10.64 | 0.88 s; 1.53 s | 15.3 s; 25.9 s |
| **Exp-5a hybrid + filter** | 0.45 | **0.55** (0.46) | 0.62 | 0.37 | **50/76** | **0.51 / 0.45** | 0.80 (n=45) | **0.28** | $10.80 | 1.36 s (+2.3 s routing median) | 24.7 s |

**Analysis**
1. **Retrieval: the filter mostly changes the rest of the context, not the top hit.** Paired vs hybrid: Hit@5 +5 / −0 (p = 0.06), Hit@10 +4 / −0, Hit@3 +3 / −2, MRR 9 better / 4 worse; top-1 from the gold filing 62 → 65. The share of the 5 parents from the gold filing rose from 0.66 to 0.80; all 5 come from it on 50 questions instead of 30.
2. **EM: a consistent gain, at the edge of what n = 51 can show.** Lenient EM vs hybrid run 1: +6 / −0 (p = 0.031); vs run 2: +5 / −1 (p = 0.22); vs Exp-3b: +4 / −1. Refusals fell (5 removed, 1–3 added). Same direction against both noise runs, larger than the ±0.04 noise floor (0.51 vs 0.39 / 0.43), but only one of the two comparisons is significant.
3. **The mechanism checks out.** On the numeric questions, EM with a single-filing context was 56% (Exp-4) and 58% (Exp-5a); with mixed filings 30% and 39%. Exp-5a's gain comes mainly from moving questions into the single-filing group (18 → 33 of 51), as predicted from the Exp-4 failure analysis.
4. **What it cannot fix:** even with a context entirely from the right filing, 42% of numeric answers are wrong or refused (page-level retrieval misses and generation errors). That is the ceiling for any filing-level fix.
5. **Costs:** routing adds one small Gemini call (~$0.42 / 1k questions, +4%) and ~2.3 s median latency, a sequential step before search. Total P95 (24.7 s) is within the 15–26 s range of identical Exp-4 runs.
6. **Caveat:** 17 of the 30 companies have a single filing in this corpus, so the ticker filter alone isolates the filing for many questions. With full filing histories the filter narrows less and the gain is likely smaller.

**Expected to change in Exp-5b:** HyDE (a hypothetical filing passage used as the search query) targets the page-level misses that remain inside the right filing: 36 of Exp-4's 39 Hit@5 misses already had the gold filing in the top 5.

#### Exp-5b: HyDE on top of the filter (run 2026-10-26)

**Change from Exp-5a (only):** Gemini (`gemini-3.8-flash`, temperature 0, thinking LOW) writes a hypothetical 10-K/10-Q passage answering the question (~120 words requested; 114–133 produced, none empty; plausible figures allowed, refusals forbidden). The dense side embeds that passage instead of the question; BM25 keeps the question, because invented figures are exact tokens that would match unrelated passages. Applied to every question (no question-type routing). Passages cached per question (`data/cache/exp5_hyde.jsonl`, $0.075 for 76); latency and cost charged in sequence with the extraction. Code: `query_router.hypothetical_passage`, `hybrid.make_search(..., dense_text=)`, `runner --hyde`, `scripts/hyde_passages.py`.

**Results** (n = 76; exact search; filter on; 5 parents to the generator)

| | Hit@3 | Hit@5 (child) | Hit@10 | MRR@10 | EM lenient / strict | Fully faithful | Refusals | Cost / 1k | Routing latency (median) | Total P95 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Exp-5a filter | 0.45 | 0.55 (0.46) | 0.62 | 0.37 | 0.51 / 0.45 | 0.80 (n=45) | 0.28 | $10.80 | 2.3 s | 24.7 s |
| **Exp-5b filter + HyDE** | 0.46 | **0.63** (0.47) | **0.71** | **0.42** | 0.49 / 0.43 | 0.75 (n=53) | **0.22** | $11.77 | 6.2 s (P95 9.3 s) | 23.3 s |

**Analysis**
1. **Retrieval: the first significant ranking gains in the project.** Paired vs Exp-5a: Hit@5 +6 / −0 (p = 0.031), Hit@10 +8 / −1 (p = 0.039), MRR 22 better / 12 worse; Hit@3 +8 / −7 (churn). By question type, Hit@5 domain-relevant 0.48 → 0.56, novel-generated 0.65 → 0.75, metrics-generated unchanged (n = 6). HyDE finds the page inside the right filing, as targeted.
2. **EM did not move** (vs Exp-5a +1 / −2). vs the two Exp-4 hybrid runs: +6 / −1 (p = 0.13) and +5 / −2. Refusals fell 0.28 → 0.22 (5 removed, 1 added).
3. **Part of that is the EM judge.** Of the numeric questions whose outcome changed, three refusals became answers a reader would accept but EM scores wrong: Adobe FCF conversion (142.7% → 155.5% vs gold 143% → 156%; the gold's "~13%" change is not stated), Amcor gross margin ("No", 19.4% → 18.5% vs gold "decline of 0.8%"), American Water working capital ("No, a deficit" without the figure). On Boeing's tax rate the answer quotes the filing ((0.6)% vs 14.7%) while the gold has 0.62% vs −14.76%. EM's number matching under-credits yes/no answers that carry derived figures; Exp-5b's real gain on answers is likely understated, but by an amount this metric cannot show.
4. **Costs:** HyDE adds ~$0.97 / 1k questions (+9%) and ~3.8 s median before search (routing total 6.2 s median, run in sequence). Extraction and HyDE need only the question and could run in parallel (~4 s). Total P95 23.3 s is inside the 15–26 s range of identical runs; retrieval P95 (2.6 s in this run vs 0.53 s retrieval-only) is dominated by machine noise.
5. **Faithfulness** fell 0.80 → 0.75 fully grounded with more answered questions (n 45 → 53): the newly answered ones include derived calculations. The HyDE text never reaches the generator, so invented figures cannot enter answers.

#### Exp-F: Framework baseline, LlamaIndex (run 2026-10-26)

Full write-up: `docs/experiments/exp-F-framework-baseline.md`. Same questions, prompt, model and metrics; only the pipeline is LlamaIndex's standard components, sized like ours, with exact search. **Result: Exp-1 level** (Hit@5 0.21, EM 0.20 / 0.18, refusals 0.62); vs Exp-4 EM +2 / −12 (p = 0.013). The relevance judge's ceiling on the framework's chunks is 64/76 (ours 76/76), but on the judgeable 64 the gap holds (Hit@5 0.25 vs 0.53). Ablation: dense leaves 0.26, BM25 0.11, relative-score fusion 0.21 (below dense alone), auto-merging merged 1 of 380 slots, so the generator gets small leaves (1,929 input tokens per question vs 9,671; $4.27 / 1k). Change handling: a one-sentence edit re-embeds the whole filing (2,253 nodes) vs one section (30 of 594 children) in ours. ~120 lines of code vs ~1,300.

**Exp-5c (decomposition)** is built and unit-tested (`decompose`, `search_decomposed`: parallel filtered sub-searches, parents interleaved) but not evaluated: all 76 FinanceBench questions are single-filing. It is measured on the Phase 2 multi-hop questions. Decisions for Exp-5 as a whole: ADR-004.

#### Serving (built 2026-10-26)

`src/sec_financial_rag/api/`: FastAPI service with the Exp-5b pipeline. `POST /query` (router extraction and HyDE run concurrently, which ADR-004 listed as not yet built; optional `decompose`; `504` after 120 s), `POST /ingest/delta` + `GET /ingest/delta/{job_id}` (CDC re-index of changed sections into the dense collection, then `build_exp4.sync_with_bm25` copies new point ids into the hybrid collection and deletes superseded ones), `GET /metrics` (service counters + every experiment summary), `GET /health`. Live check: "What was 3M's total net sales in FY2022?" → "$34,229 million [1][3]", 9.0 s, $0.010; a delta run on Amcor found both filings unchanged (12 and 22 sections, nothing re-embedded). Serving uses approximate (HNSW) search with payload indexes; exact search was for experiment comparisons. Phoenix tracing (spec §6) is not wired in.

---

## 6. Recommended Repository Structure

```text
sec-financial-rag/
├── docker-compose.yml              # Qdrant + PostgreSQL (State Ledger) + Arize Phoenix (Tracing)
├── Makefile                        # CLI targets: make ingest, make evaluate, make serve
├── pyproject.toml                  # Dependencies managed via uv / poetry
├── docs/
│   └── adr/                        # Architecture Decision Records (Tradeoff documentation)
│       ├── 001-docling-vs-unstructured-table-parsing.md
│       ├── 002-parent-document-vs-semantic-chunking.md
│       ├── 003-hybrid-rrf-and-cross-encoder-latency.md
│       └── 004-sha256-cdc-incremental-ingestion.md
├── data/
│   ├── raw_filings/                # Downloaded 10-K / 10-Q PDFs & HTMLs (gitignored)
│   ├── parsed_markdown/            # Cached Docling structured outputs
│   └── golden_eval/                # FinanceBench subset + 100 custom multi-hop QA pairs (.jsonl)
├── src/
│   ├── config.py                   # Pydantic Settings & experiment YAML loader
│   ├── ingestion/
│   │   ├── edgar_downloader.py     # SEC EDGAR fetcher with rate-limiting & User-Agent compliance
│   │   ├── cdc_ledger.py           # SHA-256 section hashing & delta detection in SQLite/Postgres
│   │   ├── parser.py               # Docling / iXBRL table-preserving parser
│   │   └── fiscal_normalizer.py    # Maps shifted fiscal years (AAPL, MSFT, NVDA) to calendar periods
│   ├── chunking/
│   │   ├── base.py                 # Abstract Chunker interface
│   │   ├── fixed_token.py          # Exp-1 Baseline chunker
│   │   ├── semantic_table.py       # Exp-2 Layout & table-header preserving chunker
│   │   └── parent_child.py         # Exp-3 Hierarchical Parent-Document builder
│   ├── retrieval/
│   │   ├── qdrant_store.py         # Dense + Sparse (BM25) hybrid indexing & RRF search
│   │   ├── doc_store.py            # Parent document key-value store
│   │   ├── query_router.py         # Metadata extraction, HyDE, & Sub-Query Decomposition
│   │   └── reranker.py             # Cross-encoder reranking & score thresholding
│   ├── generation/
│   │   ├── prompts.py              # Strict financial analyst prompts with citation enforcement
│   │   └── synthesizer.py          # LLM synthesis + OpenTelemetry / Phoenix instrumentation
│   └── api/
│       └── main.py                 # FastAPI service exposing /query, /ingest/delta, and /metrics
└── evals/
    ├── runner.py                   # Automated benchmark runner across Exp-1 through Exp-5
    ├── metrics.py                  # HitRate, MRR, NDCG, Ragas/DeepEval wrappers, Numerical EM
    └── reports/                    # Auto-generated Markdown & CSV comparison reports
```

---

## 7. Step-by-Step Implementation Roadmap

### Week 1: Data Pipeline, CDC Ledger & Baseline (Exp-1)
1. Spin up `docker-compose.yml` with **Qdrant**, **PostgreSQL** (or local SQLite), and **Arize Phoenix**.
2. Build `edgar_downloader.py` and pull the FY2023–FY2025 `10-K` and `10-Q` filings for the 10 seed companies.
3. Curate `data/golden_eval/eval_dataset.jsonl` (pulling relevant questions from `PatronusAI/financebench` and augmenting with multi-hop comparisons).
4. Implement the `cdc_ledger.py` with SHA-256 section hashing and verify that re-running `make ingest` on unchanged filings executes in `<2 seconds` with `0` API calls.
5. Run **Exp-1 (Naive Baseline)** and log your baseline failure metrics (especially on table and multi-company queries).

### Week 2: Layout-Aware Parsing, Chunking Ablations & Parent-Document (Exp-2 & Exp-3)
1. Integrate `Docling` (or `sec-parser`) to extract clean Markdown tables and SEC Item boundaries (`Item 1A`, `Item 7`, `Item 8`).
2. Implement **Contextual Header Injection** (`[COMPANY | FILING | PERIOD | ITEM]`) and the `fiscal_normalizer.py` utility.
3. Build and benchmark **Strategy 2 (Semantic + Table Header Repetition)** and **Strategy 3 (Parent-Document Retrieval)**.
4. Write `ADR-001` and `ADR-002` documenting the exact recall gains on table-lookup questions vs. storage overhead.

### Week 3: Hybrid Search, Cross-Encoder Reranking & Query Routing (Exp-4 & Exp-5)
1. Upgrade the Qdrant collection to store both **Dense embeddings** and **Sparse BM25 vectors**. Benchmark RRF against dense-only search.
2. Add the **Cross-Encoder Reranker** (`bge-reranker-v2-m3`) and plot the **Latency vs. NDCG@10** Pareto curve across `top-k` candidate pool sizes (`k=15, 25, 50`).
3. Implement the **Query Router** (`query_router.py`) supporting metadata pre-filtering, HyDE for risk disclosures, and `asyncio` parallel sub-query decomposition for comparative prompts.
4. Run the full 5-stage ablation harness, populate the final comparison matrix, and export Arize Phoenix trace snapshots for your portfolio documentation.

---

## 8. Key Reference Repositories & Reading List

* **Datasets & Benchmarks:**
  * [`PatronusAI/financebench`](https://github.com/patronus-ai/financebench) — Open-source financial QA dataset with ground-truth SEC filing links and page-level evidence.
  * **SEC EDGAR API Documentation** — Official REST API (`data.sec.gov/submissions/CIK{cik}.json`) for real-time filing retrieval without third-party paid keys.
* **Parsing & Ingestion Tools:**
  * [`DS4SD/docling`](https://github.com/DS4SD/docling) — State-of-the-art layout analysis and table extraction into structured Markdown/JSON.
  * [`alphanome-ai/sec-parser`](https://github.com/alphanome-ai/sec-parser) — Parses SEC EDGAR HTML filings into semantic tree structures (`Item 1A`, `Item 7`, tables, titles).
* **Retrieval & Evaluation Frameworks:**
  * [`qdrant/qdrant`](https://github.com/qdrant/qdrant) — Native support for Hybrid Dense + Sparse vectors, RRF fusion, and payload indexing.
  * [`confident-ai/deepeval`](https://github.com/confident-ai/deepeval) & [`explodinggradients/ragas`](https://github.com/explodinggradients/ragas) — Evaluation frameworks for Context Precision, Recall, and Faithfulness.
  * [`Arize-ai/phoenix`](https://github.com/Arize-ai/phoenix) — OpenTelemetry-native local observability for inspecting retrieval scores, reranker drops, and sub-query traces.
