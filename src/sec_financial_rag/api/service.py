"""The query pipeline behind the API: the best configuration from the experiments.

Exp-5b (ADR-004): Gemini reads the question into a company/year filter and,
at the same time, writes a HyDE passage; hybrid dense + BM25 search runs inside
the filter (dense side embeds the HyDE passage, BM25 the question), children
expand to parents, and Gemini answers from the top parents with [n] citations.

The two router calls only need the question, so they run concurrently
(asyncio.gather): the router waits for the slower call (~4 s), not both (~6 s).
"""

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from sec_financial_rag.generation.prompts import INSUFFICIENT
from sec_financial_rag.generation.synthesizer import cost_usd
from sec_financial_rag.retrieval.query_router import (
    QueryMetadata,
    build_filter,
    search_decomposed,
)

TOP_N = 5  # parents given to the generator, as in every experiment
SOURCE_FIELDS = (
    "ticker",
    "company",
    "form_type",
    "report_date",
    "fiscal_year",
    "fiscal_period",
    "section_title",
    "accession_number",
)


@dataclass
class RagService:
    """The pipeline's parts are injected: from_env() wires the real ones, tests fakes."""

    search: Callable[..., list[dict]]  # hybrid.make_search(...): question -> children
    expand: Callable[[list[dict]], list[dict]]  # children -> ranked parents
    extract: Callable[[str], object]  # question -> Extraction
    hyde: Callable[[str], object]  # question -> Hypothetical
    decompose: Callable[[str], object]  # question -> Decomposed
    synthesize: Callable[[str, list[dict]], object]  # (question, parents) -> Answer
    checks: dict[str, Callable[[], object]] = field(default_factory=dict)  # /health

    async def answer(
        self,
        question: str,
        top_n: int = TOP_N,
        use_hyde: bool = True,
        decompose: bool = False,
    ) -> dict:
        t0 = time.perf_counter()
        router = self.decompose if decompose else self.extract
        calls = [asyncio.to_thread(router, question)]
        if use_hyde:
            calls.append(asyncio.to_thread(self.hyde, question))
        routed, *hypothetical = await asyncio.gather(*calls)
        t_route = time.perf_counter()

        if decompose and len(routed.decomposition.sub_queries) > 1:
            sub_queries = routed.decomposition.sub_queries
            parents = await search_decomposed(
                self.search, sub_queries, expand=self.expand
            )
            route = {"sub_queries": [q.model_dump(mode="json") for q in sub_queries]}
        else:
            meta: QueryMetadata = (
                routed.decomposition.sub_queries[0] if decompose else routed.metadata
            )
            dense_text = hypothetical[0].text if hypothetical else None
            children = await asyncio.to_thread(
                self.search, question, where=build_filter(meta), dense_text=dense_text
            )
            parents = self.expand(children)
            route = {
                "ticker": meta.ticker,
                "fiscal_year": meta.fiscal_year,
                "filtered": build_filter(meta) is not None,
            }
        t_search = time.perf_counter()

        sources = parents[:top_n]
        reply = await asyncio.to_thread(self.synthesize, question, sources)
        t_end = time.perf_counter()

        calls_made = [routed, *hypothetical, reply]
        return {
            "answer": reply.text,
            "refused": reply.text.startswith(INSUFFICIENT),
            "sources": [
                {"n": i, **{k: s.get(k) for k in SOURCE_FIELDS}, "text": s["text"]}
                for i, s in enumerate(sources, start=1)
            ],
            "route": {**route, "hyde": use_hyde, "decomposed": decompose},
            "timings_ms": {
                "route": round((t_route - t0) * 1000, 1),
                "retrieval": round((t_search - t_route) * 1000, 1),
                "generation": round((t_end - t_search) * 1000, 1),
                "total": round((t_end - t0) * 1000, 1),
            },
            "cost_usd": round(sum(cost_usd(c) for c in calls_made), 6),
        }

    @classmethod
    def from_env(cls) -> "RagService":
        """Real parts: bge-m3 locally, Qdrant + Postgres from .env, Gemini."""
        import torch
        from qdrant_client import QdrantClient
        from sentence_transformers import SentenceTransformer

        from sec_financial_rag.generation.synthesizer import make_client, synthesize
        from sec_financial_rag.ingestion import indexer
        from sec_financial_rag.ingestion.build_exp4 import COLLECTION
        from sec_financial_rag.ingestion.cdc_ledger import connect
        from sec_financial_rag.retrieval import hybrid, query_router
        from sec_financial_rag.retrieval.bm25 import Bm25Encoder
        from sec_financial_rag.retrieval.doc_store import expand_to_parents, get_parents

        model = SentenceTransformer(
            indexer.MODEL_NAME, model_kwargs={"dtype": torch.float16}
        )
        qdrant = QdrantClient(url=indexer.QDRANT_URL)
        conn = connect()
        conn.autocommit = True  # read-only lookups
        client = make_client()
        # Serving uses approximate (HNSW) search with payload indexes; exact search
        # was for comparing experiments, where HNSW noise mattered.
        search = hybrid.make_search(qdrant, model, Bm25Encoder(), COLLECTION, "hybrid")
        search("warm-up")
        return cls(
            search=search,
            expand=lambda children: expand_to_parents(
                children, lambda ids: get_parents(conn, ids)
            ),
            extract=lambda q: query_router.extract_metadata(client, q),
            hyde=lambda q: query_router.hypothetical_passage(client, q),
            decompose=lambda q: query_router.decompose(client, q),
            synthesize=lambda q, chunks: synthesize(client, q, chunks),
            checks={
                "qdrant": lambda: qdrant.count(COLLECTION).count,
                "postgres": lambda: conn.execute("SELECT 1").fetchone()[0],
            },
        )
