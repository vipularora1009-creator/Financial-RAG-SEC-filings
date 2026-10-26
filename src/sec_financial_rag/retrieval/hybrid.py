"""Hybrid retrieval (spec §4.6): dense + BM25, fused with Reciprocal Rank Fusion.

RRF uses ranks only, never raw scores: cosine similarities (~0.3-0.8) and BM25
scores (unbounded) are on different scales, so adding them would let one method
dominate. Each ranked list adds 1 / (k + rank) to every id it contains.
"""

from collections.abc import Callable

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, SearchParams

from sec_financial_rag.ingestion.build_exp4 import DENSE, SPARSE
from sec_financial_rag.retrieval.bm25 import Bm25Encoder

RRF_K = 60  # spec; the standard value from Cormack et al. (2009)
DEPTH = 30  # hits taken from each method (spec: top-30 dense + top-30 sparse)
FUSED = 50  # fused candidates kept (spec: unified top-50)
MODES = ("dense", "bm25", "hybrid")


def rrf(rankings: list[list], k: int = RRF_K) -> list[tuple]:
    """Fuse ranked id lists -> [(id, score)] best first. Rank starts at 1."""
    scores: dict = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
    # Ties keep first-seen order (sorted is stable): the earlier list wins.
    return sorted(scores.items(), key=lambda pair: -pair[1])


def make_search(
    qdrant: QdrantClient,
    model,  # anything with SentenceTransformer's encode()
    encoder: Bm25Encoder,
    collection: str,
    mode: str = "hybrid",
    exact: bool = False,
) -> Callable[..., list[dict]]:
    """(question, where=None, dense_text=None) -> ranked payloads (each with "score").

    where is a Qdrant Filter applied inside each search (pre-filtering), so every
    method still returns its full depth from the matching filings only.
    dense_text (HyDE) is embedded instead of the question for the dense side;
    BM25 always uses the question, whose exact terms it needs.

    exact=True scans every vector instead of walking the HNSW graph. Use it to
    compare experiments: each collection builds its own graph, so approximate
    top-50 lists differ between identical copies (46/76 questions measured).
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")

    def ranked(query, using: str, limit: int, where: Filter | None) -> list:
        return qdrant.query_points(
            collection,
            query=query,
            using=using,
            query_filter=where,
            limit=limit,
            with_payload=True,
            search_params=SearchParams(exact=exact),
        ).points

    def search(
        question: str, where: Filter | None = None, dense_text: str | None = None
    ) -> list[dict]:
        hits = []
        if mode in ("dense", "hybrid"):
            text = dense_text or question
            vector = model.encode(text, normalize_embeddings=True).tolist()
            hits.append(
                ranked(vector, DENSE, DEPTH if mode == "hybrid" else FUSED, where)
            )
        if mode in ("bm25", "hybrid"):
            hits.append(
                ranked(
                    encoder.query(question),
                    SPARSE,
                    DEPTH if mode == "hybrid" else FUSED,
                    where,
                )
            )
        payloads = {p.id: p.payload for points in hits for p in points}
        if mode != "hybrid":
            return [{**p.payload, "score": p.score} for p in hits[0]]
        fused = rrf([[p.id for p in points] for points in hits])[:FUSED]
        return [{**payloads[pid], "score": score} for pid, score in fused]

    return search
