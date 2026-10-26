import numpy as np
from qdrant_client import QdrantClient

from sec_financial_rag.retrieval.qdrant_store import (
    delete_points,
    ensure_collection,
    point_id,
    upsert_chunks,
)

C = "exp1_test"
META = {"ticker": "AAPL", "accession_number": "acc-1", "form_type": "10-K"}


def test_point_id_is_deterministic_and_version_specific():
    a = point_id(C, "acc-1", "FULL", 0, "sha-v1")
    assert a == point_id(C, "acc-1", "FULL", 0, "sha-v1")  # retry -> same id
    assert a != point_id(C, "acc-1", "FULL", 0, "sha-v2")  # new version -> new id
    assert a != point_id(C, "acc-1", "FULL", 1, "sha-v1")  # next chunk
    assert a != point_id("exp2_test", "acc-1", "FULL", 0, "sha-v1")  # other experiment


def vectors(n, dim=4):
    v = np.random.default_rng(0).random((n, dim)).astype("float32")
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def test_upsert_is_idempotent_and_delete_removes_only_given_ids():
    client = QdrantClient(":memory:")
    ensure_collection(client, C, dim=4)
    ensure_collection(client, C, dim=4)  # second call is a no-op

    old = upsert_chunks(
        client, C, "acc-1", "FULL", "v1", ["a", "b", "c"], vectors(3), META
    )
    upsert_chunks(client, C, "acc-1", "FULL", "v1", ["a", "b", "c"], vectors(3), META)
    assert client.count(C).count == 3  # retry overwrote, no duplicates

    new = upsert_chunks(client, C, "acc-1", "FULL", "v2", ["x", "y"], vectors(2), META)
    delete_points(client, C, old)  # cleanup after the ledger commit
    assert client.count(C).count == 2
    hit = client.retrieve(C, [str(new[0])], with_payload=True)[0]
    assert hit.payload == {**META, "section_id": "FULL", "chunk_index": 0, "text": "x"}
