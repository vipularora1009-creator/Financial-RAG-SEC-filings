import numpy as np
from qdrant_client import QdrantClient

from sec_financial_rag.chunking.fixed_token import FixedTokenChunker
from sec_financial_rag.ingestion import edgar_downloader as ed
from sec_financial_rag.ingestion import indexer
from sec_financial_rag.ingestion.cdc_ledger import apply_schema, record_filing
from sec_financial_rag.retrieval.qdrant_store import delete_points, ensure_collection

ROW = {
    "form": "10-K",
    "accession_number": "0000320193-25-000079",
    "report_date": "2025-09-27",
    "primary_document": "aapl-20250927.htm",
}
C = indexer.COLLECTION


class FakeModel:
    """Stands in for bge-m3: same encode() shape, no 2 GB download. Counts calls."""

    calls = 0

    def encode(self, chunks, **kwargs):
        self.calls += 1
        v = (
            np.random.default_rng(len(chunks))
            .random((len(chunks), 4))
            .astype("float32")
        )
        return v / np.linalg.norm(v, axis=1, keepdims=True)


def setup(db, tmp_path, monkeypatch, html):
    monkeypatch.setattr(ed, "RAW_DIR", tmp_path / "raw")
    apply_schema(db)
    path = ed.filing_path("AAPL", ROW)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html)
    record_filing(db, "AAPL", 320193, ROW, "file-hash")
    qdrant = QdrantClient(":memory:")
    ensure_collection(qdrant, C, dim=4)
    return qdrant, indexer.filings_to_index(db)[0]


HTML_V1 = "<p>" + " the" * 40 + "</p>"
HTML_V2 = "<p>" + " a" * 20 + "</p>"
CHUNKER = FixedTokenChunker(size=8, overlap=2)  # tiny chunks -> several per filing


def test_new_then_unchanged_without_embedding_again(db, tmp_path, monkeypatch):
    qdrant, filing = setup(db, tmp_path, monkeypatch, HTML_V1)
    model = FakeModel()

    status, old = indexer.index_filing(db, qdrant, model, CHUNKER, filing)
    assert (status, old) == ("new", [])
    assert qdrant.count(C).count == len(CHUNKER.split(" the" * 40))
    assert (
        qdrant.retrieve(C, [qdrant.scroll(C)[0][0].id])[0].payload["ticker"] == "AAPL"
    )

    status, _ = indexer.index_filing(db, qdrant, model, CHUNKER, filing)
    assert status == "unchanged"
    assert model.calls == 1  # unchanged -> no second embedding


def test_changed_text_replaces_old_vectors(db, tmp_path, monkeypatch):
    qdrant, filing = setup(db, tmp_path, monkeypatch, HTML_V1)
    indexer.index_filing(db, qdrant, FakeModel(), CHUNKER, filing)

    ed.filing_path("AAPL", ROW).write_text(HTML_V2)
    status, old = indexer.index_filing(db, qdrant, FakeModel(), CHUNKER, filing)
    assert status == "changed"
    delete_points(qdrant, C, old)  # what main() does after the commit
    assert qdrant.count(C).count == len(CHUNKER.split(" a" * 20))
