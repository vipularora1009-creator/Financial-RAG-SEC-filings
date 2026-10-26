"""BM25 sparse vectors for hybrid retrieval (spec §4.6).

fastembed computes the term-frequency half of BM25 (stemming, stop words, length
normalization); Qdrant applies the IDF half from its own corpus statistics
(collection created with `modifier=IDF`), so adding documents never needs a refit.
"""

import re

from fastembed.sparse.bm25 import Bm25
from qdrant_client.models import SparseVector

MODEL = "Qdrant/bm25"
# Mean BM25 tokens per Exp-3b child, header included (measured: 122.1 on 5,000 children).
AVG_LEN = 122.0

THOUSANDS = re.compile(r"(?<=\d),(?=\d{3}\b)")
DECIMAL = re.compile(r"(?<=\d)\.(?=\d)")


def normalize_numbers(text: str) -> str:
    """Keep each figure one token: "$14,205" -> "$14205", "16.52%" -> "16_52%".

    fastembed splits on punctuation, which turns "14,205" into "14" and "205":
    two common tokens that match thousands of unrelated figures. "_" counts as a
    word character, so it survives.
    """
    return DECIMAL.sub("_", THOUSANDS.sub("", text))


class Bm25Encoder:
    def __init__(self, avg_len: float = AVG_LEN):
        self.model = Bm25(MODEL, avg_len=avg_len)

    def documents(self, texts: list[str]) -> list[SparseVector]:
        return [
            SparseVector(indices=e.indices.tolist(), values=e.values.tolist())
            for e in self.model.embed([normalize_numbers(t) for t in texts])
        ]

    def query(self, text: str) -> SparseVector:
        e = next(self.model.query_embed(normalize_numbers(text)))
        return SparseVector(indices=e.indices.tolist(), values=e.values.tolist())

    def tokens(self, text: str) -> list[str]:
        """The stemmed tokens BM25 sees (for tests and debugging)."""
        from fastembed.sparse.bm25 import remove_non_alphanumeric

        doc = remove_non_alphanumeric(normalize_numbers(text))
        return self.model._stem(self.model.tokenizer.tokenize(doc))
