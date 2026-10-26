"""Second-stage reranking (spec §4.6): a cross-encoder re-sorts the expanded parents.

A cross-encoder reads question and passage together, so it judges relevance far
better than comparing two separately computed vectors, but it costs one model
pass per pair; so it only sees the first-stage candidates.
"""

from collections.abc import Callable

CANDIDATES = 25  # spec: top-25 fused candidates after parent expansion

Score = Callable[[str, list[str]], list[float]]  # (question, passages) -> scores


def rerank(
    question: str, parents: list[dict], score: Score, candidates: int = CANDIDATES
) -> list[dict]:
    """Re-sort the first `candidates` parents by cross-encoder score, best first.

    The reranker sees "header + parent" (company, filing, period, section), as the
    dense model saw "header + child". `score` becomes the reranker score; the
    first-stage score is kept as `retrieval_score`.
    """
    pool = parents[:candidates]
    scores = score(question, [f"{p.get('header', '')}\n{p['text']}" for p in pool])
    ranked = sorted(zip(pool, scores), key=lambda pair: -pair[1])  # stable on ties
    return [{**p, "retrieval_score": p.get("score"), "score": s} for p, s in ranked]
