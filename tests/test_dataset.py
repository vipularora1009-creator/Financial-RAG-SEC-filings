import pytest

from sec_financial_rag.evals.dataset import merge_records


def test_merge_rejects_duplicate_ids():
    with pytest.raises(ValueError, match="fb-1"):
        merge_records([{"id": "fb-1"}], [{"id": "fb-1"}])


def test_merge_keeps_order_financebench_then_custom():
    assert [r["id"] for r in merge_records([{"id": "fb-1"}], [{"id": "c-1"}])] == [
        "fb-1",
        "c-1",
    ]
