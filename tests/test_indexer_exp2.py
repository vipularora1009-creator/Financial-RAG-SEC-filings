from datetime import date

import numpy as np
from qdrant_client import QdrantClient

from sec_financial_rag.chunking.semantic_table import SemanticTableChunker
from sec_financial_rag.ingestion import edgar_downloader as ed
from sec_financial_rag.ingestion import indexer_exp2
from sec_financial_rag.ingestion.cdc_ledger import apply_schema, record_filing
from sec_financial_rag.ingestion.indexer import filings_to_index
from sec_financial_rag.retrieval.qdrant_store import delete_points, ensure_collection

C = indexer_exp2.COLLECTION
ROW = {
    "form": "10-K",
    "accession_number": "0000320193-25-000079",
    "report_date": date(2025, 9, 27),
    "primary_document": "aapl-20250927.htm",
}
COVER = """<div style="display:none"><ix:header><ix:hidden>
<ix:nonNumeric name="dei:DocumentFiscalYearFocus">2025</ix:nonNumeric>
<ix:nonNumeric name="dei:DocumentFiscalPeriodFocus">FY</ix:nonNumeric>
<ix:nonNumeric name="dei:EntityRegistrantName">Apple Inc.</ix:nonNumeric>
</ix:hidden></ix:header></div>"""  # hidden, as in real filings


def html(mdna: str) -> str:
    return f"""<html><body>{COVER}
<p>Item 1. Business</p><p>The Company designs smartphones and computers.</p>
<p>Item 7. Management's Discussion and Analysis</p><p>{mdna}</p>
</body></html>"""


class FakeModel:
    """Records what it embeds; same encode() shape as bge-m3."""

    def __init__(self):
        self.inputs: list[str] = []

    def encode(self, texts, **kwargs):
        self.inputs += texts
        v = np.random.default_rng(len(texts)).random((len(texts), 4)).astype("float32")
        return v / np.linalg.norm(v, axis=1, keepdims=True)


def setup(db, tmp_path, monkeypatch, page):
    monkeypatch.setattr(ed, "RAW_DIR", tmp_path / "raw")
    apply_schema(db)
    path = ed.filing_path("AAPL", ROW)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(page)
    record_filing(db, "AAPL", 320193, ROW, "file-hash")
    qdrant = QdrantClient(":memory:")
    ensure_collection(qdrant, C, dim=4)
    return qdrant, filings_to_index(db)[0]


def payloads(qdrant):
    return {p.payload["section_id"]: p.payload for p in qdrant.scroll(C, limit=100)[0]}


def test_sections_indexed_with_header_embedded_not_stored(db, tmp_path, monkeypatch):
    qdrant, filing = setup(db, tmp_path, monkeypatch, html("Net sales grew 2%."))
    model = FakeModel()
    statuses, old = indexer_exp2.index_filing(
        db, qdrant, model, SemanticTableChunker(), filing
    )
    assert statuses == {"new": 2} and old == []
    item7 = payloads(qdrant)["Item_7"]
    assert item7["header"] == (
        "[COMPANY: Apple Inc. (AAPL) | FILING: 10-K | PERIOD: FY2025 (Ended Sep 27, 2025)"
        " | SECTION: Item 7 - Management's Discussion and Analysis]"
    )
    assert item7["text"].endswith("Net sales grew 2%.")  # body only
    assert "[COMPANY" not in item7["text"]
    assert (item7["fiscal_year"], item7["fiscal_period"]) == (2025, "FY")
    assert all(text.startswith("[COMPANY: Apple Inc.") for text in model.inputs)


def test_rerun_unchanged_embeds_nothing(db, tmp_path, monkeypatch):
    qdrant, filing = setup(db, tmp_path, monkeypatch, html("Net sales grew 2%."))
    indexer_exp2.index_filing(db, qdrant, FakeModel(), SemanticTableChunker(), filing)
    model = FakeModel()
    statuses, old = indexer_exp2.index_filing(
        db, qdrant, model, SemanticTableChunker(), filing
    )
    assert statuses == {"unchanged": 2} and old == [] and model.inputs == []


def test_only_the_changed_section_is_reembedded(db, tmp_path, monkeypatch):
    qdrant, filing = setup(db, tmp_path, monkeypatch, html("Net sales grew 2%."))
    indexer_exp2.index_filing(db, qdrant, FakeModel(), SemanticTableChunker(), filing)
    old_item7_ids = [
        p.id for p in qdrant.scroll(C)[0] if p.payload["section_id"] == "Item_7"
    ]

    ed.filing_path("AAPL", ROW).write_text(html("Net sales grew 3% (restated)."))
    model = FakeModel()
    statuses, old = indexer_exp2.index_filing(
        db, qdrant, model, SemanticTableChunker(), filing
    )
    assert statuses == {"unchanged": 1, "changed": 1}
    assert sorted(map(str, old)) == sorted(map(str, old_item7_ids))
    assert all("Item 7" in text for text in model.inputs)  # Item 1 not re-embedded
    delete_points(qdrant, C, old)
    assert payloads(qdrant)["Item_7"]["text"].endswith("(restated).")
