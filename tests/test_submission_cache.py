import httpx

from sec_financial_rag.ingestion import edgar_downloader as ed
from sec_financial_rag.ingestion.cdc_ledger import apply_schema

OLDER = {
    "name": "CIK0000019617-submissions-001.json",
    "filingFrom": "2022-01-03",
    "filingTo": "2023-06-30",
    "filingCount": 1,
}
RECENT = {"form": [], "accessionNumber": [], "reportDate": [], "primaryDocument": []}
OLDER_BODY = {
    "form": ["10-K"],
    "accessionNumber": ["0000019617-23-000231"],
    "reportDate": ["2022-12-31"],
    "primaryDocument": ["jpm-20221231.htm"],
}


def fake_edgar(calls: list, older: dict) -> httpx.Client:
    """Fake data.sec.gov: one submissions index listing one older file."""

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        if request.url.path.endswith(older["name"]):
            return httpx.Response(200, json=OLDER_BODY)
        return httpx.Response(
            200, json={"filings": {"recent": RECENT, "files": [older]}}
        )

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_older_files_are_fetched_once_then_served_from_ledger(db, monkeypatch):
    monkeypatch.setattr(ed.time, "sleep", lambda s: None)  # no real waiting in tests
    apply_schema(db)

    first_calls, second_calls = [], []
    first = ed.fetch_all_filings(
        fake_edgar(first_calls, OLDER), 19617, "2022-01-01", db
    )
    second = ed.fetch_all_filings(
        fake_edgar(second_calls, OLDER), 19617, "2022-01-01", db
    )

    assert len(first_calls) == 2  # index + older file
    assert len(second_calls) == 1  # index only; older file came from Postgres
    assert first == second
    assert first[0]["accession_number"] == "0000019617-23-000231"


def test_older_file_is_refetched_when_index_metadata_changes(db, monkeypatch):
    monkeypatch.setattr(ed.time, "sleep", lambda s: None)
    apply_schema(db)
    ed.fetch_all_filings(fake_edgar([], OLDER), 19617, "2022-01-01", db)

    calls = []
    rewritten = {**OLDER, "filingCount": 2}  # SEC rewrote the file
    ed.fetch_all_filings(fake_edgar(calls, rewritten), 19617, "2022-01-01", db)
    assert len(calls) == 2  # cache miss -> fetched again
