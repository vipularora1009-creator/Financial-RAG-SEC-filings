import pytest

from sec_financial_rag.evals.metrics import (
    hit_at_k,
    is_relevant,
    overlap_coefficient,
    reciprocal_rank,
)

SHORT = "Total net sales were 90,753 million dollars in the quarter"
LONG_PAGE = (
    "Consolidated statements of operations " + SHORT + " and services grew strongly"
)


def test_overlap_coefficient_handles_short_and_long_evidence():
    assert overlap_coefficient(SHORT, "intro text " + SHORT + " more text") == 1.0
    assert (
        overlap_coefficient(LONG_PAGE, SHORT) == 1.0
    )  # chunk inside a long evidence page
    assert (
        overlap_coefficient(SHORT, "completely different words about weather today")
        == 0.0
    )


def test_relevant_needs_same_filing_and_overlapping_text():
    evidence = [{"accession_number": "acc-1", "text": SHORT}]
    assert is_relevant({"accession_number": "acc-1", "text": SHORT}, evidence)
    # Same text in another filing (e.g. next year's 10-Q repeats a sentence) is not a hit.
    assert not is_relevant({"accession_number": "acc-2", "text": SHORT}, evidence)


def test_hit_at_k():
    ranks = [False, False, True]
    assert hit_at_k(ranks, 2) == 0.0
    assert hit_at_k(ranks, 3) == 1.0


def test_reciprocal_rank_uses_first_relevant_within_k():
    assert reciprocal_rank([False, False, True, True]) == pytest.approx(1 / 3)
    assert reciprocal_rank([False] * 10) == 0.0
    assert reciprocal_rank([False] * 10 + [True]) == 0.0  # rank 11 is outside @10
