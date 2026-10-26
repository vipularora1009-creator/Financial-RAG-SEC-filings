"""Exp-1 indexing: filings in the ledger -> naive text -> 512/64 chunks -> bge-m3 -> Qdrant.

Run: uv run --env-file .env python -m sec_financial_rag.ingestion.indexer [--limit N]
Re-runs skip filings whose text is unchanged in this collection (CDC ledger).
"""

import sys
import time

import psycopg
import torch
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from sec_financial_rag.chunking.base import Chunker
from sec_financial_rag.chunking.fixed_token import FixedTokenChunker
from sec_financial_rag.ingestion.cdc_ledger import (
    classify_section,
    connect,
    content_sha256,
    record_section,
)
from sec_financial_rag.ingestion.edgar_downloader import filing_path
from sec_financial_rag.ingestion.naive_parser import html_to_text
from sec_financial_rag.retrieval.qdrant_store import (
    delete_points,
    ensure_collection,
    upsert_chunks,
)

COLLECTION = "exp1_fixed512_bge_m3"
MODEL_NAME = "BAAI/bge-m3"
DIM = 1024
SECTION = "FULL"  # Week 1: one section per filing; Items come with the Week 2 parser
QDRANT_URL = "http://127.0.0.1:6333"


def filings_to_index(conn: psycopg.Connection) -> list[dict]:
    """Every recorded filing, as dicts shaped like the downloader's rows."""
    rows = conn.execute(
        "SELECT ticker, cik, form_type, accession_number, report_date, primary_document"
        " FROM filings ORDER BY ticker, report_date"
    ).fetchall()
    keys = (
        "ticker",
        "cik",
        "form",
        "accession_number",
        "report_date",
        "primary_document",
    )
    return [dict(zip(keys, row)) for row in rows]


def index_filing(
    conn: psycopg.Connection,
    qdrant: QdrantClient,
    model,  # anything with SentenceTransformer's encode(); tests pass a fake
    chunker: Chunker,
    filing: dict,
) -> tuple[str, list]:
    """Index one filing unless unchanged. Returns (status, superseded point ids).

    Does NOT commit: the caller commits, then deletes the returned ids.
    """
    text = html_to_text(filing_path(filing["ticker"], filing).read_text())
    sha = content_sha256(text)
    acc = filing["accession_number"]
    status = classify_section(conn, COLLECTION, acc, SECTION, sha)
    if status == "unchanged":
        return status, []
    chunks = chunker.split(text)
    vectors = model.encode(chunks, batch_size=16, normalize_embeddings=True)
    metadata = {
        "ticker": filing["ticker"],
        "cik": filing["cik"],
        "form_type": filing["form"],
        "accession_number": acc,
        "report_date": str(filing["report_date"]),  # date -> "YYYY-MM-DD" for JSON
    }
    ids = upsert_chunks(
        qdrant, COLLECTION, acc, SECTION, sha, chunks, vectors, metadata
    )
    old_ids = record_section(conn, COLLECTION, acc, SECTION, sha, ids)
    return status, old_ids


def main(limit: int | None = None) -> None:
    start = time.perf_counter()
    model = SentenceTransformer(MODEL_NAME, model_kwargs={"dtype": torch.float16})
    chunker = FixedTokenChunker()
    qdrant = QdrantClient(url=QDRANT_URL)
    ensure_collection(qdrant, COLLECTION, DIM)
    with connect() as conn:
        filings = filings_to_index(conn)[:limit]
        for n, filing in enumerate(filings, 1):
            t0 = time.perf_counter()
            status, old_ids = index_filing(conn, qdrant, model, chunker, filing)
            conn.commit()  # checkpoint per filing (a crash keeps finished work)
            delete_points(qdrant, COLLECTION, old_ids)  # only after the commit
            print(
                f"[{n:3}/{len(filings)}] {filing['ticker']:5} {filing['form']:7}"
                f" {filing['report_date']}  {status:9} {time.perf_counter() - t0:6.1f}s",
                flush=True,
            )
    total = qdrant.count(COLLECTION).count
    print(
        f"{COLLECTION}: {total} points   total {(time.perf_counter() - start) / 60:.1f} min"
    )


if __name__ == "__main__":
    main(
        int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None
    )
