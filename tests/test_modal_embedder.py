import numpy as np
import pytest

from sec_financial_rag.ingestion import indexer, modal_embedder
from sec_financial_rag.ingestion.modal_embedder import RemoteModel


def test_remote_model_name_matches_local_indexer():
    assert modal_embedder.MODEL_NAME == indexer.MODEL_NAME
    assert modal_embedder.DIM == indexer.DIM


class FakeRemote:
    """Stands in for Embedder().encode.remote: records batch sizes."""

    def __init__(self):
        self.calls: list[int] = []

    def remote(self, texts, batch_size):
        self.calls.append(len(texts))
        return np.ones((len(texts), 3), dtype=np.float32)


def remote_model(monkeypatch) -> tuple[RemoteModel, FakeRemote]:
    fake = FakeRemote()
    model = RemoteModel.__new__(RemoteModel)  # skip Embedder(): no Modal call
    model.embedder = type("E", (), {"encode": fake})()
    monkeypatch.setattr(modal_embedder, "CALL_BATCH", 4)
    return model, fake


def test_encode_splits_into_call_batches_and_keeps_order(monkeypatch):
    model, fake = remote_model(monkeypatch)
    vectors = model.encode([f"t{i}" for i in range(10)], batch_size=8)
    assert fake.calls == [4, 4, 2]
    assert vectors.shape == (10, 3)


def test_single_string_gives_one_vector_and_empty_list_gives_none(monkeypatch):
    model, _ = remote_model(monkeypatch)
    assert model.encode("one question").shape == (3,)
    assert model.encode([]).shape == (0, modal_embedder.DIM)


def test_unnormalized_request_is_refused(monkeypatch):
    model, _ = remote_model(monkeypatch)
    with pytest.raises(ValueError):
        model.encode(["x"], normalize_embeddings=False)
