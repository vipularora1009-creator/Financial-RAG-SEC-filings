"""Retrieval metrics (spec §5.1): Hit Rate@K and MRR@10, judged against FinanceBench evidence.

A retrieved chunk counts as relevant when it comes from the evidence's filing AND
shares enough 5-word shingles with the evidence text. Evidence passages range from
11 to 641 words while chunks are ~350, so overlap is measured against the SMALLER
of the two (overlap coefficient): a short passage inside a chunk and a chunk inside
a long evidence page both score 1.0.
"""

from sec_financial_rag.evals.financebench import shingles, tokens

# At least half of the smaller text's shingles must appear in the other.
RELEVANT_OVERLAP = 0.5


def overlap_coefficient(evidence: str, chunk: str) -> float:
    """|E ∩ C| / min(|E|, |C|) over 5-word shingles (Szymkiewicz–Simpson)."""
    e, c = shingles(tokens(evidence)), shingles(tokens(chunk))
    if not e or not c:
        return 0.0
    return len(e & c) / min(len(e), len(c))


def is_relevant(payload: dict, evidence: list[dict]) -> bool:
    """Does this retrieved chunk contain (part of) any evidence passage, in the same filing?"""
    return any(
        payload["accession_number"] == ev["accession_number"]
        and overlap_coefficient(ev["text"], payload["text"]) >= RELEVANT_OVERLAP
        for ev in evidence
    )


def hit_at_k(relevant: list[bool], k: int) -> float:
    """1.0 if any of the top-k results is relevant, else 0.0."""
    return 1.0 if any(relevant[:k]) else 0.0


def reciprocal_rank(relevant: list[bool], k: int = 10) -> float:
    """1 / rank of the first relevant result within the top k (rank starts at 1), else 0."""
    for rank, is_hit in enumerate(relevant[:k], start=1):
        if is_hit:
            return 1.0 / rank
    return 0.0
