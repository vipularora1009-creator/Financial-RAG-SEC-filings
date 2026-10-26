from uuid import UUID

from qdrant_client import QdrantClient
from test_indexer_exp2 import ROW, FakeModel, html

from sec_financial_rag.chunking.parent_child import ParentChildChunker
from sec_financial_rag.ingestion import edgar_downloader as ed
from sec_financial_rag.ingestion import indexer_exp3
from sec_financial_rag.ingestion.cdc_ledger import apply_schema, record_filing
from sec_financial_rag.ingestion.indexer import filings_to_index
from sec_financial_rag.retrieval.doc_store import delete_stale_parents, get_parents
from sec_financial_rag.retrieval.qdrant_store import delete_points, ensure_collection

C = indexer_exp3.COLLECTION


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


def children(qdrant):
    return [p.payload for p in qdrant.scroll(C, limit=100)[0]]


def every_child_has_its_parent(db, qdrant) -> bool:
    kids = children(qdrant)
    parents = get_parents(db, [UUID(k["parent_id"]) for k in kids])
    return all(k["text"] in parents[UUID(k["parent_id"])] for k in kids)


def test_children_point_to_stored_parents(db, tmp_path, monkeypatch):
    qdrant, filing = setup(db, tmp_path, monkeypatch, html("Net sales grew 2%."))
    model = FakeModel()
    statuses, old, reindexed = indexer_exp3.index_filing(
        db, qdrant, model, ParentChildChunker(), filing
    )
    assert statuses == {"new": 2} and old == []
    assert {sid for sid, _ in reindexed} == {"Item_1", "Item_7"}
    assert every_child_has_its_parent(db, qdrant)
    assert all(text.startswith("[COMPANY: Apple Inc.") for text in model.inputs)
    assert all("[COMPANY" not in k["text"] for k in children(qdrant))  # body only


def test_changed_section_keeps_old_parents_until_old_children_are_gone(
    db, tmp_path, monkeypatch
):
    qdrant, filing = setup(db, tmp_path, monkeypatch, html("Net sales grew 2%."))
    indexer_exp3.index_filing(db, qdrant, FakeModel(), ParentChildChunker(), filing)

    ed.filing_path("AAPL", ROW).write_text(html("Net sales grew 3% (restated)."))
    statuses, old, reindexed = indexer_exp3.index_filing(
        db, qdrant, FakeModel(), ParentChildChunker(), filing
    )
    assert statuses == {"unchanged": 1, "changed": 1}
    # Before cleanup both versions coexist, and every child still finds its parent.
    assert every_child_has_its_parent(db, qdrant)

    delete_points(qdrant, C, old)  # main(): old children first...
    acc = ROW["accession_number"]
    deleted = sum(delete_stale_parents(db, C, acc, sid, sha) for sid, sha in reindexed)
    assert deleted == 1  # ...then the one old Item 7 parent
    assert every_child_has_its_parent(db, qdrant)
    item7 = [k["text"] for k in children(qdrant) if k["section_id"] == "Item_7"]
    assert len(item7) == 1 and item7[0].endswith("(restated).")
