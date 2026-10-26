"""Exp-F framework arm, end to end on two tiny documents (in-memory Qdrant)."""

import numpy as np
import pytest

pytest.importorskip("llama_index.core")  # only with: uv sync --group exp-f

from llama_index.core import Document
from qdrant_client import QdrantClient

from sec_financial_rag.exp_f import pipeline


class FakeModel:
    """Two directions: text about interest vs everything else."""

    def encode(self, texts, normalize_embeddings=True):
        return np.array(
            [[1.0, 0.0] if "interest" in t.lower() else [0.0, 1.0] for t in texts]
        )


def test_adapter_embeds_batches_through_encode():
    embed = pipeline.EncodeEmbedding(model=FakeModel())
    assert embed.get_text_embedding_batch(["Interest expense", "Revenue"]) == [
        [1.0, 0.0],
        [0.0, 1.0],
    ]
    assert embed.get_query_embedding("interest?") == [1.0, 0.0]


def test_index_and_search_return_payload_shaped_hits(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "DOCSTORE", tmp_path / "docstore.json")
    meta = {"ticker": "AAPL", "form_type": "10-K", "report_date": "2024-09-28"}
    docs = [
        Document(
            id_="acc-1",
            text="Interest expense was $14,205 million.",
            metadata={"accession_number": "acc-1", **meta},
        ),
        Document(
            id_="acc-2",
            text="Net sales grew in Greater China.",
            metadata={"accession_number": "acc-2", **meta},
        ),
    ]
    qdrant, embed = (
        QdrantClient(":memory:"),
        pipeline.EncodeEmbedding(model=FakeModel()),
    )
    nodes, leaves = pipeline.build_index(docs, qdrant, embed)
    assert nodes >= leaves >= 2

    hits = pipeline.make_search(qdrant, embed)("What was interest expense?")
    assert hits[0]["accession_number"] == "acc-1"
    assert {"ticker", "form_type", "report_date", "text", "score"} <= set(hits[0])
