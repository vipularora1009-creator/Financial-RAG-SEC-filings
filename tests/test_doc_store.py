import pytest

from sec_financial_rag.ingestion.cdc_ledger import apply_schema, record_filing
from sec_financial_rag.retrieval.doc_store import (
    delete_stale_parents,
    expand_to_parents,
    get_parents,
    parent_id,
    put_parents,
)

EXP3 = "exp3_parent_child_bge_m3"
FILING = {
    "form": "10-K",
    "accession_number": "0000320193-25-000079",
    "report_date": "2025-09-27",
    "primary_document": "aapl-20250927.htm",
}
ACC = FILING["accession_number"]


def setup(db):
    apply_schema(db)
    record_filing(db, "AAPL", 320193, FILING, "file-hash")


def test_parent_id_is_deterministic_and_changes_with_sha():
    a = parent_id(EXP3, ACC, "Item_7", 0, "sha1")
    assert a == parent_id(EXP3, ACC, "Item_7", 0, "sha1")
    assert a != parent_id(EXP3, ACC, "Item_7", 0, "sha2")
    assert a != parent_id(EXP3, ACC, "Item_7", 1, "sha1")


def test_put_then_get_round_trip_and_retry_is_harmless(db):
    setup(db)
    ids = put_parents(db, EXP3, ACC, "Item_7", "sha1", ["first parent", "second"])
    assert (
        put_parents(db, EXP3, ACC, "Item_7", "sha1", ["first parent", "second"]) == ids
    )
    assert get_parents(db, ids) == {ids[0]: "first parent", ids[1]: "second"}
    count = db.execute("SELECT count(*) FROM parent_documents").fetchone()[0]
    assert count == 2  # the retry inserted nothing


def test_get_parents_skips_unknown_ids(db):
    setup(db)
    ids = put_parents(db, EXP3, ACC, "Item_7", "sha1", ["only"])
    unknown = parent_id(EXP3, ACC, "Item_7", 9, "nope")
    assert get_parents(db, [*ids, unknown]) == {ids[0]: "only"}


def test_delete_stale_parents_keeps_current_version_and_other_sections(db):
    setup(db)
    old = put_parents(db, EXP3, ACC, "Item_7", "sha1", ["old a", "old b"])
    new = put_parents(db, EXP3, ACC, "Item_7", "sha2", ["new a"])
    other = put_parents(db, EXP3, ACC, "Item_1A", "sha1", ["risk"])
    assert delete_stale_parents(db, EXP3, ACC, "Item_7", "sha2") == 2
    assert get_parents(db, old + new + other) == {new[0]: "new a", other[0]: "risk"}


def test_expand_to_parents_dedupes_in_rank_order():
    p1, p2 = (
        parent_id(EXP3, ACC, "Item_7", 0, "s"),
        parent_id(EXP3, ACC, "Item_7", 1, "s"),
    )
    children = [
        {"parent_id": str(p2), "text": "row 7", "score": 0.9},
        {"parent_id": str(p1), "text": "para 1", "score": 0.8},
        {"parent_id": str(p2), "text": "row 9", "score": 0.7},  # same parent: dropped
    ]
    store = {p1: "PARENT ONE", p2: "PARENT TWO"}
    parents = expand_to_parents(children, lambda ids: {i: store[i] for i in ids})
    assert [p["text"] for p in parents] == ["PARENT TWO", "PARENT ONE"]
    assert [p["score"] for p in parents] == [0.9, 0.8]  # the best child's score
    assert parents[0]["child_text"] == "row 7"


def test_expand_to_parents_fails_loudly_on_a_missing_parent():
    child = {"parent_id": str(parent_id(EXP3, ACC, "Item_7", 0, "s")), "text": "x"}
    with pytest.raises(KeyError):
        expand_to_parents([child], lambda ids: {})
