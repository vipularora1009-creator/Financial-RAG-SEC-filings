from sec_financial_rag.ingestion.edgar_downloader import (
    RAW_DIR,
    filing_path,
    filing_url,
)


def test_filing_url_matches_known_apple_10k():
    # Apple FY2025 10-K; this URL was verified to return HTTP 200 (lesson 0001).
    row = {
        "form": "10-K",
        "accession_number": "0000320193-25-000079",
        "report_date": "2025-09-27",
        "primary_document": "aapl-20250927.htm",
    }
    assert filing_url(320193, row) == (
        "https://www.sec.gov/Archives/edgar/data/320193/000032019325000079/aapl-20250927.htm"
    )


def test_filing_path_unique_when_amendment_reuses_file_name():
    # Real case: AMD's FY2025 10-K and its 10-K/A share the primaryDocument name.
    # Keying on file name alone saved one and silently skipped the other.
    original = {
        "form": "10-K",
        "accession_number": "0000002488-26-000018",
        "report_date": "2025-12-27",
        "primary_document": "amd-20251227.htm",
    }
    amendment = {
        **original,
        "form": "10-K/A",
        "accession_number": "0000002488-26-000021",
    }

    assert filing_path("AMD", original) != filing_path("AMD", amendment)
    assert filing_path("AMD", original) == (
        RAW_DIR / "AMD" / "0000002488-26-000018" / "amd-20251227.htm"
    )
