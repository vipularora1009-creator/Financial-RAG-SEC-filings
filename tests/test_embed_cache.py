import numpy as np

from sec_financial_rag.chunking.embed_cache import CachedEmbedder


class FakeModel:
    """Deterministic vectors from the text; records what it was asked to embed."""

    def __init__(self):
        self.seen: list[str] = []

    def encode(self, sentences, normalize_embeddings=True):
        self.seen += sentences
        vecs = np.array([[len(s), s.count("a") + 1, 1.0] for s in sentences])
        return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)


def test_second_call_hits_cache(tmp_path):
    model = FakeModel()
    embed = CachedEmbedder(model, "fake", tmp_path / "c.sqlite")
    first = embed(["alpha one.", "beta two."])
    second = embed(["beta two.", "alpha one."])
    assert model.seen == ["alpha one.", "beta two."]  # nothing re-embedded
    np.testing.assert_allclose(second, first[::-1], atol=1e-3)  # order follows input


def test_repeated_sentence_embedded_once(tmp_path):
    model = FakeModel()
    vectors = CachedEmbedder(model, "fake", tmp_path / "c.sqlite")(["same.", "same."])
    assert model.seen == ["same."]
    assert vectors.shape == (2, 3) and vectors.dtype == np.float32


def test_cache_survives_restart_but_not_model_change(tmp_path):
    path = tmp_path / "c.sqlite"
    CachedEmbedder(FakeModel(), "fake", path)(["kept."])
    model = FakeModel()
    CachedEmbedder(model, "fake", path)(["kept."])
    assert model.seen == []  # read from disk
    other = FakeModel()
    CachedEmbedder(other, "other-model", path)(["kept."])
    assert other.seen == ["kept."]  # different model -> different key
