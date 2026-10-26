import httpx

from sec_financial_rag.ingestion import edgar_downloader as ed
from sec_financial_rag.ingestion.cdc_ledger import apply_schema

ROW = {
    "form": "10-K",
    "accession_number": "0000320193-25-000079",
    "report_date": "2025-09-27",
    "primary_document": "aapl-20250927.htm",
}


def fake_client(calls: list) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        return httpx.Response(200, content=b"<html>10-K</html>")

    return httpx.Client(transport=httpx.MockTransport(handler))


def setup(db, tmp_path, monkeypatch):
    monkeypatch.setattr(ed, "RAW_DIR", tmp_path / "raw")  # never touch the real corpus
    monkeypatch.setattr(ed.time, "sleep", lambda s: None)
    apply_schema(db)


def test_first_run_downloads_and_records_then_rerun_is_unchanged(
    db, tmp_path, monkeypatch
):
    setup(db, tmp_path, monkeypatch)
    calls = []
    assert ed.ingest_filing(fake_client(calls), db, "AAPL", 320193, ROW) == "new"
    assert len(calls) == 1

    rerun_calls = []
    status = ed.ingest_filing(fake_client(rerun_calls), db, "AAPL", 320193, ROW)
    assert status == "unchanged"
    assert rerun_calls == []  # file already on disk: no download


def test_file_edited_on_disk_is_reported_changed(db, tmp_path, monkeypatch):
    # Always hashing catches changes made outside the pipeline.
    setup(db, tmp_path, monkeypatch)
    ed.ingest_filing(fake_client([]), db, "AAPL", 320193, ROW)
    ed.filing_path("AAPL", ROW).write_bytes(b"<html>edited</html>")
    assert ed.ingest_filing(fake_client([]), db, "AAPL", 320193, ROW) == "changed"


def test_file_on_disk_but_not_in_ledger_is_backfilled(db, tmp_path, monkeypatch):
    # The 209 files downloaded before the ledger existed.
    setup(db, tmp_path, monkeypatch)
    path = ed.filing_path("AAPL", ROW)
    path.parent.mkdir(parents=True)
    path.write_bytes(b"<html>10-K</html>")

    calls = []
    assert ed.ingest_filing(fake_client(calls), db, "AAPL", 320193, ROW) == "new"
    assert calls == []  # recorded from the file on disk, not downloaded again
    stored = db.execute("SELECT file_sha256 FROM filings").fetchone()[0]
    assert len(stored) == 64
