"""Qdrant side of indexing: collections, deterministic point ids, upsert, delete.

One collection per experiment config (e.g. exp1_fixed512_bge_m3), matching the
ledger's `collection` column, so experiments never overwrite each other.
"""

import uuid
from uuid import UUID

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointIdsList, PointStruct, VectorParams

# Fixed namespace for uuid5: the same name always maps to the same UUID.
NAMESPACE = uuid.uuid5(uuid.NAMESPACE_DNS, "sec-financial-rag")


def point_id(
    collection: str, accession: str, section_id: str, chunk_index: int, sha: str
) -> UUID:
    """Deterministic id: retries overwrite (same id); a new text version never
    shares ids with the old one (sha differs), so deleting old ids is always safe."""
    return uuid.uuid5(
        NAMESPACE, f"{collection}|{accession}|{section_id}|{chunk_index}|{sha}"
    )


def ensure_collection(client: QdrantClient, name: str, dim: int) -> None:
    """Create a cosine-distance collection if it doesn't exist yet (idempotent)."""
    if not client.collection_exists(name):
        client.create_collection(
            name, vectors_config=VectorParams(size=dim, distance=Distance.COSINE)
        )


def upsert_chunks(
    client: QdrantClient,
    collection: str,
    accession: str,
    section_id: str,
    sha: str,
    chunks: list[str],
    vectors: np.ndarray,
    metadata: dict,
    extra: list[dict] | None = None,
) -> list[UUID]:
    """Store one section's chunks + vectors. Returns their point ids (for the ledger).

    `metadata` is shared by every chunk; `extra` (optional) is one dict per chunk,
    e.g. Exp-3's parent_id.
    """
    ids = [
        point_id(collection, accession, section_id, i, sha) for i in range(len(chunks))
    ]
    points = [
        PointStruct(
            id=str(pid),
            vector=vector.tolist(),
            # Payload: metadata stored with the vector, returned with search hits.
            payload={
                **metadata,
                "section_id": section_id,
                "chunk_index": i,
                "text": chunk,
                **(extra[i] if extra else {}),
            },
        )
        for i, (pid, chunk, vector) in enumerate(zip(ids, chunks, vectors))
    ]
    # wait=True: return only once Qdrant has applied the write, so the ledger never
    # records point ids that aren't stored yet.
    client.upsert(collection, points=points, wait=True)
    return ids


def delete_points(client: QdrantClient, collection: str, ids: list[UUID]) -> None:
    """Remove a superseded version's vectors (call only AFTER the ledger commit)."""
    if ids:
        client.delete(
            collection,
            points_selector=PointIdsList(points=[str(i) for i in ids]),
            wait=True,
        )
