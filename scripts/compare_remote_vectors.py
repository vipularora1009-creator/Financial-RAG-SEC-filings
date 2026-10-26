"""Do Modal (CUDA) vectors match the local (Apple MPS) ones? Run before using Modal
for an index that will be compared with locally-built ones.

Usage: uv run --env-file .env python scripts/compare_remote_vectors.py [N]
Takes N (default 300) Exp-3 children from Qdrant, re-embeds "header + text" as
indexed, locally and on Modal, and compares both with the stored vectors.
"""

import random
import sys
import time

import numpy as np
import torch
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from sec_financial_rag.ingestion import indexer, indexer_exp3
from sec_financial_rag.ingestion.modal_embedder import RemoteModel, app


def cosine_stats(a: np.ndarray, b: np.ndarray) -> str:
    cos = np.sum(a * b, axis=1)  # rows are unit vectors
    return f"min {cos.min():.5f}  mean {cos.mean():.5f}  p1 {np.percentile(cos, 1):.5f}"


def main(n: int) -> None:
    qdrant = QdrantClient(url=indexer.QDRANT_URL)
    points, _ = qdrant.scroll(
        indexer_exp3.COLLECTION, limit=5000, with_payload=True, with_vectors=True
    )
    random.seed(0)
    sample = random.sample(points, n)
    texts = [f"{p.payload['header']}\n{p.payload['text']}" for p in sample]
    stored = np.array([p.vector for p in sample], dtype=np.float32)

    local_model = SentenceTransformer(
        indexer.MODEL_NAME, model_kwargs={"dtype": torch.float16}
    )
    t0 = time.perf_counter()
    local = local_model.encode(texts, batch_size=8, normalize_embeddings=True)
    local_s = time.perf_counter() - t0

    with app.run():
        remote_model = RemoteModel()
        t0 = time.perf_counter()
        remote_model.encode(texts[:2])  # cold start: container + model load
        cold_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        remote = remote_model.encode(texts, batch_size=32)
        remote_s = time.perf_counter() - t0

    print(f"{n} children, median {np.median([len(t) for t in texts]):.0f} chars")
    print(f"local vs stored : {cosine_stats(local, stored)}")
    print(f"remote vs stored: {cosine_stats(remote, stored)}")
    print(f"remote vs local : {cosine_stats(remote, local)}")
    print(
        f"time: local {local_s:.1f} s, remote {remote_s:.1f} s (cold start {cold_s:.1f} s)"
    )


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 300)
