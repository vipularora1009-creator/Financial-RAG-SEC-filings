import numpy as np
import pytest
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue, PointStruct

from sec_financial_rag.ingestion import build_exp4
from sec_financial_rag.retrieval.bm25 import Bm25Encoder
from sec_financial_rag.retrieval.hybrid import make_search, rrf


def test_rrf_rewards_agreement_over_one_high_rank():
    # b: 1/62 + 1/61 (found by both); a: 1/61 (top of one list only); c: 1/62
    fused = rrf([["a", "b"], ["b", "c"]], k=60)
    assert [i for i, _ in fused] == ["b", "a", "c"]


def test_rrf_scores_are_sums_of_reciprocal_ranks():
    (top, score), *_ = rrf([["x"], ["x"]], k=60)
    assert top == "x" and score == pytest.approx(2 / 61)


class FakeModel:
    def encode(self, text, **_):
        return np.array([1.0, 0.0, 0.0, 0.0])  # always "about" child 1


def collection_with_two_children(monkeypatch) -> QdrantClient:
    monkeypatch.setattr(build_exp4, "DIM", 4)
    client = QdrantClient(":memory:")
    build_exp4.create_collection(client, "c")
    enc = Bm25Encoder()
    texts = [
        "Total net sales grew in Greater China",
        "Interest expense was $14,205 million",
    ]
    sparse = enc.documents(texts)
    client.upsert(
        "c",
        [
            PointStruct(
                id=i + 1,
                vector={"dense": np.eye(4)[i].tolist(), "bm25": sparse[i]},
                payload={"text": t, "ticker": ticker},
            )
            for i, (t, ticker) in enumerate(zip(texts, ["AAPL", "MSFT"]))
        ],
    )
    return client


def test_modes_find_by_meaning_by_figure_and_both(monkeypatch):
    client = collection_with_two_children(monkeypatch)
    enc, model = Bm25Encoder(), FakeModel()
    question = "What was interest expense of 14,205?"
    dense = make_search(client, model, enc, "c", "dense")(question)
    bm25 = make_search(client, model, enc, "c", "bm25")(question)
    hybrid = make_search(client, model, enc, "c", "hybrid")(question)
    assert dense[0]["text"].startswith("Total net sales")  # the fake dense model
    assert bm25[0]["text"].startswith("Interest expense")  # exact figure match
    assert {h["text"] for h in hybrid} == {d["text"] for d in dense} | {
        b["text"] for b in bm25
    }
    assert all("score" in h for h in hybrid)


@pytest.mark.parametrize("mode", ["dense", "bm25", "hybrid"])
def test_filter_keeps_only_matching_payloads(monkeypatch, mode):
    client = collection_with_two_children(monkeypatch)
    search = make_search(client, FakeModel(), Bm25Encoder(), "c", mode)
    apple = Filter(must=[FieldCondition(key="ticker", match=MatchValue(value="AAPL"))])
    # The figure is only in MSFT's text, so unfiltered BM25 would rank MSFT first.
    hits = search("What was interest expense of 14,205?", where=apple)
    assert {h["ticker"] for h in hits} <= {"AAPL"}  # never MSFT
    # BM25 alone finds nothing: AAPL's text shares no term with the question.
    assert len(hits) == (0 if mode == "bm25" else 1)


class RecordingClient:
    """In-memory Qdrant ignores payload indexes, so record the calls instead."""

    def __init__(self):
        self.indexed = {}

    def create_payload_index(self, collection, field, field_schema, wait):
        self.indexed[field] = field_schema


def test_payload_indexes_are_created():
    client = RecordingClient()
    build_exp4.create_payload_indexes(client, "c")
    assert client.indexed == build_exp4.INDEXED


class RecordingModel(FakeModel):
    def __init__(self):
        self.seen = []

    def encode(self, text, **kwargs):
        self.seen.append(text)
        return super().encode(text, **kwargs)


def test_dense_text_replaces_the_question_for_dense_only(monkeypatch):
    client = collection_with_two_children(monkeypatch)
    model = RecordingModel()
    search = make_search(client, model, Bm25Encoder(), "c", "hybrid")
    hits = search("What was interest expense of 14,205?", dense_text="HyDE passage")
    assert model.seen == ["HyDE passage"]  # dense embedded the passage
    # BM25 still matched the question's exact figure.
    assert any(h["text"].startswith("Interest expense") for h in hits)


def test_unknown_mode_is_refused():
    with pytest.raises(ValueError):
        make_search(None, None, None, "c", "splade")
