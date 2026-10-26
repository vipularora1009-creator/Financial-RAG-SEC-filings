"""CDC ledger in PostgreSQL (spec §4.2, Week 1 step 4).

Records which filings and sections have been processed, with a SHA-256 of
their content, so re-runs skip unchanged work.
Run: uv run --env-file .env python -m sec_financial_rag.ingestion.cdc_ledger
"""

import hashlib
import os
from pathlib import Path
from uuid import UUID

import psycopg
from psycopg.types.json import Jsonb

SCHEMA_PATH = Path(__file__).with_name("schema.sql")


def content_sha256(text: str) -> str:
    """Fingerprint of a section's text; whitespace-only differences don't count as changes."""
    # Any run of spaces/newlines/tabs -> one space; leading/trailing whitespace dropped.
    normalized = " ".join(text.split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def sha256_of_file(path: Path) -> str:
    """Fingerprint of a file's raw bytes, read in chunks (no full copy in memory)."""
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def database_url() -> str:
    """Build the connection URL from the same .env values docker-compose uses."""
    user, password, db = (
        os.environ[k] for k in ("POSTGRES_USER", "POSTGRES_PASSWORD", "POSTGRES_DB")
    )
    host = os.environ.get("POSTGRES_HOST", "127.0.0.1")
    return f"postgresql://{user}:{password}@{host}:5432/{db}"


def connect() -> psycopg.Connection:
    """Open a connection. Unlike pg_isready, this really logs in (wrong .env fails here)."""
    return psycopg.connect(database_url(), connect_timeout=5)


def apply_schema(conn: psycopg.Connection) -> None:
    """Create the ledger tables if they don't exist yet."""
    conn.execute(SCHEMA_PATH.read_text())


def record_filing(
    conn: psycopg.Connection, ticker: str, cik: int, row: dict, file_sha256: str
) -> str:
    """Upsert one filing. Returns 'new', 'unchanged' or 'changed' (by raw-file hash)."""
    previous = conn.execute(
        "SELECT file_sha256 FROM filings WHERE accession_number = %s",
        (row["accession_number"],),
    ).fetchone()
    conn.execute(
        """
        INSERT INTO filings (accession_number, ticker, cik, form_type,
                             report_date, primary_document, file_sha256)
        VALUES (%(accession_number)s, %(ticker)s, %(cik)s, %(form)s,
                %(report_date)s, %(primary_document)s, %(file_sha256)s)
        ON CONFLICT (accession_number) DO UPDATE SET file_sha256 = EXCLUDED.file_sha256
        """,
        {**row, "ticker": ticker, "cik": cik, "file_sha256": file_sha256},
    )
    if previous is None:
        return "new"
    return "unchanged" if previous[0] == file_sha256 else "changed"


def classify_section(
    conn: psycopg.Connection,
    collection: str,
    accession_number: str,
    section_id: str,
    sha: str,
) -> str:
    """'new' if no current row, 'unchanged' if the current hash matches, else 'changed'."""
    current = conn.execute(
        "SELECT content_sha256 FROM ingestion_ledger WHERE collection = %s"
        " AND accession_number = %s AND section_id = %s AND NOT is_superseded",
        (collection, accession_number, section_id),
    ).fetchone()
    if current is None:
        return "new"
    return "unchanged" if current[0] == sha else "changed"


def record_section(
    conn: psycopg.Connection,
    collection: str,
    accession_number: str,
    section_id: str,
    sha: str,
    point_ids: list[UUID],
) -> list[UUID]:
    """Make this the current version of a section (call it only for new/changed ones).

    Returns the superseded row's point ids; delete those from Qdrant AFTER commit.
    """
    with conn.transaction():  # both statements succeed, or neither does
        old = conn.execute(
            "UPDATE ingestion_ledger SET is_superseded = true WHERE collection = %s"
            " AND accession_number = %s AND section_id = %s AND NOT is_superseded"
            " RETURNING vector_point_ids",
            (collection, accession_number, section_id),
        ).fetchone()
        # Supersede first: inserting first would briefly make two current rows,
        # which the one_current_section_per_collection index rejects.
        conn.execute(
            "INSERT INTO ingestion_ledger (collection, accession_number,"
            " section_id, content_sha256, vector_point_ids)"
            " VALUES (%s, %s, %s, %s, %s)",
            (collection, accession_number, section_id, sha, point_ids),
        )
    return old[0] if old else []


def get_cached_rows(conn: psycopg.Connection, file: dict) -> list[dict] | None:
    """Cached rows for an older submissions file, or None if missing or stale.

    `file` is one entry of EDGAR's filings.files; every field must match the cache.
    """
    hit = conn.execute(
        "SELECT rows FROM submission_files WHERE name = %s"
        " AND filing_from = %s AND filing_to = %s AND filing_count = %s",
        (file["name"], file["filingFrom"], file["filingTo"], file["filingCount"]),
    ).fetchone()
    return hit[0] if hit else None  # psycopg turns JSONB back into Python lists/dicts


def cache_rows(conn: psycopg.Connection, file: dict, rows: list[dict]) -> None:
    """Store (or replace) the filtered rows of an older submissions file."""
    conn.execute(
        """
        INSERT INTO submission_files (name, filing_from, filing_to, filing_count, rows)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (name) DO UPDATE SET
            filing_from = EXCLUDED.filing_from, filing_to = EXCLUDED.filing_to,
            filing_count = EXCLUDED.filing_count, rows = EXCLUDED.rows,
            fetched_at = now()
        """,
        (
            file["name"],
            file["filingFrom"],
            file["filingTo"],
            file["filingCount"],
            Jsonb(rows),
        ),
    )


def main() -> None:
    with connect() as conn:  # commits on success, rolls back on error
        apply_schema(conn)
        tables = conn.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'public' ORDER BY table_name"
        ).fetchall()
    print("ledger tables:", [t[0] for t in tables])


if __name__ == "__main__":
    main()
