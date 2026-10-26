import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, SparseVector, VectorParams

from sec_financial_rag.ingestion import build_exp4
from sec_financial_rag.retrieval.bm25 import Bm25Encoder


def test_copy_keeps_ids_payload_dense_vector_and_adds_bm25(monkeypatch):
    monkeypatch.setattr(build_exp4, "DIM", 4)
    client = QdrantClient(":memory:")
    client.create_collection(
        "src", vectors_config=VectorParams(size=4, distance=Distance.COSINE)
    )
    payloads = [
        {
            "header": "[AAPL 10-Q]",
            "text": "iPhone net sales $14,205",
            "parent_id": "p1",
        },
        {"header": "[AAPL 10-Q]", "text": "Services revenue grew", "parent_id": "p2"},
    ]
    vectors = np.eye(4)[:2].tolist()
    client.upsert(
        "src",
        [
            PointStruct(id=i + 1, vector=v, payload=p)
            for i, (v, p) in enumerate(zip(vectors, payloads))
        ],
    )
    build_exp4.create_collection(client, "dst")

    assert build_exp4.copy_with_bm25(client, Bm25Encoder(), "src", "dst") == 2
    points = {
        p.id: p
        for p in client.retrieve("dst", [1, 2], with_payload=True, with_vectors=True)
    }
    assert points[1].payload == payloads[0]
    assert np.allclose(points[1].vector["dense"], vectors[0])
    assert isinstance(points[1].vector["bm25"], SparseVector)

    hits = client.query_points(
        "dst", query=Bm25Encoder().query("iPhone 14,205"), using="bm25"
    ).points
    assert hits[0].id == 1  # the exact figure finds the right child


def test_sync_copies_new_points_and_deletes_superseded_ones(monkeypatch):
    monkeypatch.setattr(build_exp4, "DIM", 4)
    client = QdrantClient(":memory:")
    client.create_collection(
        "src", vectors_config=VectorParams(size=4, distance=Distance.COSINE)
    )
    payload = {"header": "[AAPL 10-Q]", "text": "Services revenue grew"}
    client.upsert(  # source after a CDC run: point 1 is new, point 2 unchanged
        "src",
        [
            PointStruct(id=i, vector=np.eye(4)[i].tolist(), payload=payload)
            for i in (1, 2)
        ],
    )
    build_exp4.create_collection(client, "dst")
    build_exp4.copy_with_bm25(client, Bm25Encoder(), "src", "dst")
    client.delete("dst", [1])  # target is behind: lacks 1 ...
    client.upsert(  # ... and still has 3, which the CDC run superseded
        "dst",
        [PointStruct(id=3, vector={"dense": np.eye(4)[3].tolist()}, payload=payload)],
    )

    copied, deleted = build_exp4.sync_with_bm25(client, Bm25Encoder(), "src", "dst")

    assert (copied, deleted) == (1, 1)
    ids = {p.id for p in client.scroll("dst", limit=10)[0]}
    assert ids == {1, 2}
    (one,) = client.retrieve("dst", [1], with_vectors=True)
    assert isinstance(one.vector["bm25"], SparseVector)
