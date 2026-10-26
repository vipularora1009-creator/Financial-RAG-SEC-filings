"""Download 10-K / 10-Q filings from SEC EDGAR (spec §7, Week 1 step 2).

EDGAR rules: no API key, max 10 requests/second, and a declared User-Agent
("Name email"). See https://www.sec.gov/os/accessing-edgar-data
"""

import json
import os
import time
from collections import Counter
from pathlib import Path

import httpx
import psycopg

from sec_financial_rag.ingestion.cdc_ledger import (
    apply_schema,
    cache_rows,
    connect,
    get_cached_rows,
    record_filing,
    sha256_of_file,
)

# The submissions API wants the CIK as 10 digits with leading zeros.
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
# Older filings beyond `recent` live in extra files listed under filings.files.
OLDER_FILES_URL = "https://data.sec.gov/submissions/{name}"
# Filing documents: unpadded CIK, then the accession number without dashes.
ARCHIVES_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{document}"
# Keeping all as we need to download all of them. What we do with it is next step
FORMS_TO_KEEP = {"10-K", "10-Q", "10-K/A", "10-Q/A"}
# Since Date
SINCE = "2022-01-01"
# Saved Data Directory
RAW_DIR = Path("data/raw_filings")  # Path, so `RAW_DIR / ticker` joins paths
# Extra filings behind the FinanceBench questions (Option A), verified by evals.financebench.
FINANCEBENCH_MAPPING = Path("data/golden_eval/financebench_docs.json")
# CIK ID Dictionary
COMPANIES = {
    "AAPL": 320193,
    "NVDA": 1045810,
    "AMD": 2488,
    "MSFT": 789019,
    "GOOGL": 1652044,
    "AMZN": 1018724,
    "TSLA": 1318605,
    "F": 37996,
    "JPM": 19617,
    "MMM": 66740,
}


def make_client(user_agent: str) -> httpx.Client:
    """One shared client: sends our User-Agent on every request and reuses connections."""
    return httpx.Client(headers={"User-Agent": user_agent}, timeout=30.0)


def fetch_submissions(client: httpx.Client, cik: int) -> dict:
    """Return a company's EDGAR submissions JSON (metadata + filing history)."""
    response = client.get(SUBMISSIONS_URL.format(cik=cik))
    response.raise_for_status()  # turn 403/404/5xx into an exception instead of bad data
    return response.json()


def filter_filings(recent: dict) -> list[dict]:
    """Turn EDGAR's parallel arrays into rows for the forms we index."""
    rows = []
    for form, acc, period, doc in zip(
        recent["form"],
        recent["accessionNumber"],
        recent["reportDate"],
        recent["primaryDocument"],
    ):
        if form in FORMS_TO_KEEP:
            rows.append(
                {
                    "form": form,
                    "accession_number": acc,
                    "report_date": period,
                    "primary_document": doc,
                }
            )
    return rows


def older_file_rows(
    client: httpx.Client, file: dict, conn: psycopg.Connection | None = None
) -> list[dict]:
    """Kept rows from one older submissions file: from the ledger cache if fresh, else SEC."""
    if conn is not None:
        cached = get_cached_rows(conn, file)
        if cached is not None:
            return cached
    time.sleep(0.15)  # stay well under SEC's 10 requests/second
    response = client.get(OLDER_FILES_URL.format(name=file["name"]))
    response.raise_for_status()
    # Older files use the same parallel-array layout as `recent`.
    rows = filter_filings(response.json())
    if conn is not None:
        cache_rows(conn, file, rows)
    return rows


def fetch_all_filings(
    client: httpx.Client, cik: int, since: str, conn: psycopg.Connection | None = None
) -> list[dict]:
    """All kept filings with report_date >= since (ISO date), incl. EDGAR's older files.

    With a ledger connection, older files are cached in Postgres (fetched once).
    """
    data = fetch_submissions(client, cik)
    rows = filter_filings(data["filings"]["recent"])
    for file in data["filings"]["files"]:  # ordered newest first
        # Newest filing in this file is before the cutoff -> every later file is older still.
        if file["filingTo"] < since:
            break
        rows.extend(older_file_rows(client, file, conn))
    # ISO dates (YYYY-MM-DD) compare correctly as plain text.
    return [row for row in rows if row["report_date"] >= since]


def filing_url(cik: int, row: dict) -> str:
    """URL of a filing's primary document in the EDGAR Archives."""
    return ARCHIVES_URL.format(
        cik=cik,
        accession=row["accession_number"].replace("-", ""),
        document=row["primary_document"],
    )


def filing_path(ticker: str, row: dict) -> Path:
    """Local path for a filing. Keyed on the accession number (unique per filing):
    an amendment can reuse the original's file name (e.g. AMD FY2025 10-K and 10-K/A)."""
    return RAW_DIR / ticker / row["accession_number"] / row["primary_document"]


def download_filing(client: httpx.Client, url: str, path: Path) -> bool:
    """Save one document to `path`. Returns False if it was already on disk (skipped)."""
    if path.exists():
        return False
    time.sleep(0.15)  # rate limit: stay under SEC's 10 requests/second
    response = client.get(url)
    response.raise_for_status()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".part")
    tmp_path.write_bytes(response.content)
    tmp_path.replace(path)

    return True


def ingest_filing(
    client: httpx.Client, conn: psycopg.Connection, ticker: str, cik: int, row: dict
) -> str:
    """Download one filing if needed and record it in the ledger.

    Always hashes the file on disk, so any change to it (re-download or edit) is
    caught. Returns 'new', 'changed' or 'unchanged'.
    """
    path = filing_path(ticker, row)
    download_filing(client, filing_url(cik, row), path)
    return record_filing(conn, ticker, cik, row, sha256_of_file(path))


def download_company(
    client: httpx.Client, conn: psycopg.Connection, ticker: str, cik: int, since: str
) -> Counter:
    """Ingest all kept filings for one company. Counts 'new' / 'changed' / 'unchanged'."""
    counts = Counter()
    for row in fetch_all_filings(client, cik, since, conn):
        counts[ingest_filing(client, conn, ticker, cik, row)] += 1
    return counts


def download_mapped_filings(
    client: httpx.Client, conn: psycopg.Connection, mapping_path: Path
) -> Counter:
    """Ingest the specific filings listed in a mapping file (only verified entries)."""
    counts = Counter()
    for row in json.loads(mapping_path.read_text()).values():
        if not row["status"].startswith("verified"):
            continue  # unresolved docs stay out of the corpus until a human decides
        counts[ingest_filing(client, conn, row["ticker"], row["cik"], row)] += 1
    return counts


def summary(label: str, counts: Counter) -> str:
    return (
        f"{label:6} new {counts['new']:3}  changed {counts['changed']:3}"
        f"  unchanged {counts['unchanged']:3}"
    )


def main() -> None:
    """Ingest the seed corpus + FinanceBench slice into the ledger and print a summary."""
    start = time.perf_counter()
    total = Counter()
    sec_requests = []
    with connect() as conn, make_client(os.environ["SEC_USER_AGENT"]) as client:
        client.event_hooks["request"].append(sec_requests.append)  # count every request
        apply_schema(conn)
        for ticker, cik in COMPANIES.items():
            counts = download_company(client, conn, ticker, cik, SINCE)
            conn.commit()  # checkpoint: a crash later keeps this company's work
            print(summary(ticker, counts))
            total.update(counts)
        counts = download_mapped_filings(client, conn, FINANCEBENCH_MAPPING)
        conn.commit()
        print(summary("FB", counts) + "   (FinanceBench slice)")
        total.update(counts)
    print(summary("TOTAL", total))
    print(
        f"SEC requests {len(sec_requests)}   time {time.perf_counter() - start:.1f} s"
    )


if __name__ == "__main__":
    main()
