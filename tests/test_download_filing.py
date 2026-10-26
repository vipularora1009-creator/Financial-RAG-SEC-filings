import json

import httpx

from sec_financial_rag.ingestion.edgar_downloader import download_filing

URL = "https://www.sec.gov/Archives/edgar/data/320193/000032019325000079/aapl-20250927.htm"


def fake_client(calls: list) -> httpx.Client:
    """A client that never touches the network: every request returns fixed HTML."""

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        return httpx.Response(200, content=b"<html>10-K</html>")

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_downloads_and_saves_bytes(tmp_path):
    calls = []
    path = tmp_path / "AAPL" / "aapl-20250927.htm"  # AAPL/ folder doesn't exist yet

    assert download_filing(fake_client(calls), URL, path) is True
    assert path.read_bytes() == b"<html>10-K</html>"
    assert calls == [URL]


def test_skips_file_already_on_disk(tmp_path):
    calls = []
    path = tmp_path / "aapl-20250927.htm"
    path.write_bytes(b"old copy")

    assert download_filing(fake_client(calls), URL, path) is False
    assert path.read_bytes() == b"old copy"  # untouched
    assert calls == []  # no request made at all


def test_download_mapped_filings_only_verified(db, tmp_path, monkeypatch):
    from sec_financial_rag.ingestion import edgar_downloader as ed
    from sec_financial_rag.ingestion.cdc_ledger import apply_schema

    apply_schema(db)

    monkeypatch.setattr(ed, "RAW_DIR", tmp_path / "raw")  # never touch the real corpus
    row = {
        "form": "10-K",
        "accession_number": "0000764478-23-000006",
        "report_date": "2023-01-28",
        "primary_document": "bby.htm",
    }
    mapping = {
        "BESTBUY_2023_10K": {
            **row,
            "ticker": "BBY",
            "cik": 764478,
            "status": "verified",
        },
        "UNSURE_DOC": {**row, "ticker": "XYZ", "cik": 1, "status": "review"},
    }
    mapping_path = tmp_path / "mapping.json"
    mapping_path.write_text(json.dumps(mapping))
    calls = []

    assert ed.download_mapped_filings(fake_client(calls), db, mapping_path) == {
        "new": 1
    }
    assert (tmp_path / "raw" / "BBY" / "0000764478-23-000006" / "bby.htm").exists()
    assert not (tmp_path / "raw" / "XYZ").exists()  # `review` entries are left out
