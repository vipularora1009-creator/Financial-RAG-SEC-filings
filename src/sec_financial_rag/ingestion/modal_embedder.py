"""GPU models on Modal: bge-m3 embeddings (encode()) and the bge-reranker-v2-m3
cross-encoder (Exp-4 reranking).

Why: embedding the corpus on the 8 GB M1 swaps heavily (Exp-3: 4.3 h, swap at
~8.7 of 9.2 GB). Only the GPU work moves; parsing, chunking, Qdrant and Postgres
stay local, and only chunk texts (public SEC filings) are sent.

Use (auth: MODAL_TOKEN_ID / MODAL_TOKEN_SECRET in .env):
    with app.run():
        model = RemoteModel()
        vectors = model.encode(texts, batch_size=32, normalize_embeddings=True)

Self-contained on purpose: the container runs this file, so it imports nothing
heavy from the project. Library versions match pyproject.toml, so the GPU is the
only difference from local embedding.
"""

import modal
import numpy as np

MODEL_NAME = "BAAI/bge-m3"  # = indexer.MODEL_NAME (asserted in tests)
RERANKER_NAME = "BAAI/bge-reranker-v2-m3"
RERANK_MAX_LENGTH = 2048  # question + header + parent (median parent ~1,600 tokens)
DIM = 1024
GPU = "L4"  # 24 GB; ~$0.80/h
CALL_BATCH = 256  # texts per remote call: amortizes the network round trip


def _download_weights() -> None:
    from huggingface_hub import snapshot_download

    snapshot_download(MODEL_NAME)
    snapshot_download(RERANKER_NAME)


image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.14.1",
        "sentence-transformers==6.2.0",
        "transformers==5.20.2",
        "numpy==2.5.4",
    )
    .run_function(_download_weights)  # weights baked in: no download per cold start
    .add_local_python_source("sec_financial_rag")
)
app = modal.App("sec-rag-embed", image=image)


@app.cls(gpu=GPU, scaledown_window=120, timeout=900)
class Embedder:
    @modal.enter()
    def load(self) -> None:
        import torch
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(
            MODEL_NAME, device="cuda", model_kwargs={"dtype": torch.float16}
        )

    @modal.method()
    def encode(self, texts: list[str], batch_size: int) -> np.ndarray:
        vectors = self.model.encode(
            texts, batch_size=batch_size, normalize_embeddings=True
        )
        return np.asarray(vectors, dtype=np.float32)


class RemoteModel:
    """Adapter: SentenceTransformer's encode(), computed on Modal.

    Only valid inside `with app.run():`. Always returns unit vectors, as every
    caller in this project asks for them.
    """

    def __init__(self):
        self.embedder = Embedder()

    def encode(
        self, texts, batch_size: int = 32, normalize_embeddings: bool = True, **_
    ):
        if not normalize_embeddings:
            raise ValueError("RemoteModel always returns normalized embeddings")
        if isinstance(texts, str):
            return self.encode([texts], batch_size)[0]
        parts = [
            self.embedder.encode.remote(texts[i : i + CALL_BATCH], batch_size)
            for i in range(0, len(texts), CALL_BATCH)
        ]
        return np.concatenate(parts) if parts else np.zeros((0, DIM), np.float32)


@app.cls(gpu=GPU, scaledown_window=120, timeout=900)
class Reranker:
    @modal.enter()
    def load(self) -> None:
        import torch
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(
            RERANKER_NAME,
            device="cuda",
            max_length=RERANK_MAX_LENGTH,
            model_kwargs={"dtype": torch.float16},
        )

    @modal.method()
    def score(self, question: str, passages: list[str]) -> list[float]:
        """Relevance of each passage to the question, 0-1 (sigmoid), input order."""
        scores = self.model.predict(
            [(question, p) for p in passages], batch_size=8, show_progress_bar=False
        )
        return [float(x) for x in scores]


class RemoteReranker:
    """score(question, passages) on Modal. Only valid inside `with app.run():`."""

    def __init__(self):
        self.reranker = Reranker()

    def score(self, question: str, passages: list[str]) -> list[float]:
        return self.reranker.score.remote(question, passages) if passages else []
