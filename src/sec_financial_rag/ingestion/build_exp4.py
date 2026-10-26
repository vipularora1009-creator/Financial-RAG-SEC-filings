"""Exp-4 collection (spec §4.6): Exp-3b's children with a BM25 vector next to the dense one.

Run: uv run --env-file .env python -m sec_financial_rag.ingestion.build_exp4

A derived collection: dense vectors, payloads and point ids are copied from Exp-3b
(no re-embedding), so Exp-3b is the exact dense-only baseline. Parents are shared
(looked up by parent_id). Rebuild after re-indexing Exp-3b; CDC lives upstream.
"""

import time

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    Modifier,
    PayloadSchemaType,
    PointStruct,
    SparseVectorParams,
    VectorParams,
)

from sec_financial_rag.ingestion.indexer import DIM, QDRANT_URL
from sec_financial_rag.retrieval.bm25 import Bm25Encoder

SOURCE = "exp3b_parent_child_tablesv2_bge_m3"
COLLECTION = "exp4_hybrid_bge_m3"
DENSE, SPARSE = "dense", "bm25"  # named vectors in one point
PAGE = 1000
# Fields the query router filters on (Exp-5). A payload index lets Qdrant filter
# inside the HNSW search instead of checking every candidate's payload.
INDEXED = {
    "ticker": PayloadSchemaType.KEYWORD,
    "form_type": PayloadSchemaType.KEYWORD,
    "fiscal_period": PayloadSchemaType.KEYWORD,
    "fiscal_year": PayloadSchemaType.INTEGER,
}


def bm25_text(payload: dict) -> str:
    """The same text the dense vector was built from: header + child."""
    return f"{payload['header']}\n{payload['text']}"


def create_collection(client: QdrantClient, name: str = COLLECTION) -> None:
    client.create_collection(
        name,
        vectors_config={DENSE: VectorParams(size=DIM, distance=Distance.COSINE)},
        # IDF is computed by Qdrant from the stored documents, kept up to date.
        sparse_vectors_config={SPARSE: SparseVectorParams(modifier=Modifier.IDF)},
    )


def create_payload_indexes(client: QdrantClient, name: str = COLLECTION) -> None:
    for field, schema in INDEXED.items():
        client.create_payload_index(name, field, field_schema=schema, wait=True)


def _upsert_hybrid(
    client: QdrantClient, encoder: Bm25Encoder, target: str, points
) -> None:
    """Source points (one dense vector) -> target points (dense + BM25), same ids."""
    sparse = encoder.documents([bm25_text(p.payload) for p in points])
    client.upsert(
        target,
        points=[
            PointStruct(id=p.id, vector={DENSE: p.vector, SPARSE: s}, payload=p.payload)
            for p, s in zip(points, sparse)
        ],
        wait=True,
    )


def copy_with_bm25(
    client: QdrantClient, encoder: Bm25Encoder, source: str, target: str
) -> int:
    offset, copied = None, 0
    while True:
        points, offset = client.scroll(
            source, limit=PAGE, offset=offset, with_payload=True, with_vectors=True
        )
        _upsert_hybrid(client, encoder, target, points)
        copied += len(points)
        if offset is None:
            return copied


def point_ids(client: QdrantClient, collection: str) -> set:
    ids, offset = set(), None
    while True:
        points, offset = client.scroll(
            collection, limit=10_000, offset=offset, with_payload=False
        )
        ids.update(p.id for p in points)
        if offset is None:
            return ids


def sync_with_bm25(
    client: QdrantClient, encoder: Bm25Encoder, source: str, target: str
) -> tuple[int, int]:
    """Bring the hybrid collection up to date after a CDC run on the source.

    Point ids are content-addressed (they include the section's sha), so a
    changed section has new ids: ids only in the source are copied (with BM25),
    ids only in the target were superseded and are deleted. Copy first, then
    delete: search may briefly see both versions, never neither.
    """
    source_ids, target_ids = point_ids(client, source), point_ids(client, target)
    new, stale = sorted(source_ids - target_ids, key=str), list(target_ids - source_ids)
    for i in range(0, len(new), PAGE):
        points = client.retrieve(
            source, new[i : i + PAGE], with_payload=True, with_vectors=True
        )
        _upsert_hybrid(client, encoder, target, points)
    if stale:
        client.delete(target, stale, wait=True)
    return len(new), len(stale)


def main() -> None:
    start = time.perf_counter()
    client = QdrantClient(url=QDRANT_URL)
    if client.collection_exists(COLLECTION):
        client.delete_collection(COLLECTION)  # derived: always rebuilt whole
    create_collection(client)
    copied = copy_with_bm25(client, Bm25Encoder(), SOURCE, COLLECTION)
    create_payload_indexes(client)
    source_count = client.count(SOURCE).count
    print(
        f"{COLLECTION}: {copied} points copied (source {source_count})"
        f"   {(time.perf_counter() - start) / 60:.1f} min"
    )


if __name__ == "__main__":
    main()
