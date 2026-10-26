import psycopg
import pytest

from sec_financial_rag.ingestion.cdc_ledger import database_url


@pytest.fixture
def db():
    """A real Postgres connection inside a transaction that is always rolled back."""
    try:
        conn = psycopg.connect(database_url(), connect_timeout=2)
    except (KeyError, psycopg.OperationalError) as exc:
        pytest.skip(
            f"Postgres unavailable (docker compose up -d; uv run --env-file .env pytest): {exc}"
        )
    # Isolate from the real ledger: tables are created in an empty schema that only
    # exists inside this (rolled-back) transaction.
    conn.execute("CREATE SCHEMA test_ledger")
    conn.execute("SET LOCAL search_path TO test_ledger")
    yield conn
    conn.rollback()  # DDL and rows from the test vanish; Postgres DDL is transactional
    conn.close()
