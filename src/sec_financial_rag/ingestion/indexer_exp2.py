"""Exp-2 indexing (spec §5.2): Docling layout parse -> SEC Item sections -> table-aware
+ semantic chunks with a context header -> bge-m3 -> Qdrant.

Run: uv run --env-file .env python -m sec_financial_rag.ingestion.indexer_exp2 [--limit N]

Differences from Exp-1 (indexer.py, kept unchanged so Exp-1 stays reproducible):
- one ledger row per Item section, so an amendment re-embeds only changed sections;
- each chunk is embedded as "header + body"; the payload `text` is the body only
  (the generator prompt labels passages itself, and the relevance judge compares
  `text` with the evidence), and the header is stored beside it.
"""

import gc
import sys
import time
from collections import Counter

import psycopg
import torch
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from sec_financial_rag.chunking.embed_cache import CachedEmbedder
from sec_financial_rag.chunking.header import context_header, section_label
from sec_financial_rag.chunking.semantic_table import SemanticTableChunker, is_narrative
from sec_financial_rag.ingestion.cdc_ledger import (
    classify_section,
    connect,
    content_sha256,
    record_section,
)
from sec_financial_rag.ingestion.edgar_downloader import filing_path
from sec_financial_rag.ingestion.fiscal_normalizer import (
    dei_value,
    fiscal_period,
    period_label,
)
from sec_financial_rag.ingestion.indexer import (
    DIM,
    MODEL_NAME,
    QDRANT_URL,
    filings_to_index,
)
from sec_financial_rag.ingestion.layout_parser import parse_filing
from sec_financial_rag.retrieval.qdrant_store import (
    delete_points,
    ensure_collection,
    upsert_chunks,
)

COLLECTION = "exp2_semantic_bge_m3"
BATCH_SIZE = (
    8  # chunks can reach ~1,600 tokens (whole tables); 16 risks MPS memory on 8 GB
)


def index_filing(
    conn: psycopg.Connection,
    qdrant: QdrantClient,
    model,  # anything with SentenceTransformer's encode(); tests pass a fake
    chunker: SemanticTableChunker,
    filing: dict,
    converter=None,
) -> tuple[Counter, list]:
    """Index a filing section by section. Returns (status counts, superseded point ids).

    Does NOT commit: the caller commits, then deletes the returned ids.
    """
    path = filing_path(filing["ticker"], filing)
    html = path.read_text()
    acc = filing["accession_number"]
    fp = fiscal_period(html)
    period = period_label(fp, filing["report_date"])
    company = dei_value(html, "EntityRegistrantName") or filing["ticker"]
    statuses: Counter = Counter()
    old_ids: list = []
    for section in parse_filing(path, filing["form"], converter):
        header = context_header(
            company,
            filing["ticker"],
            filing["form"],
            period,
            section_label(section.section_id, section.title),
        )
        # Hash what determines the vectors: header + text (learning record 0016).
        sha = content_sha256(f"{header}\n{section.text}")
        status = classify_section(conn, COLLECTION, acc, section.section_id, sha)
        statuses[status] += 1
        if status == "unchanged":
            continue
        bodies = chunker.split(section.text, narrative=is_narrative(section.section_id))
        if not bodies:
            continue
        vectors = model.encode(
            [f"{header}\n{body}" for body in bodies],
            batch_size=BATCH_SIZE,
            normalize_embeddings=True,
        )
        metadata = {
            "ticker": filing["ticker"],
            "cik": filing["cik"],
            "form_type": filing["form"],
            "accession_number": acc,
            "report_date": str(filing["report_date"]),
            "company": company,
            "fiscal_year": fp.year if fp else None,
            "fiscal_period": fp.period if fp else None,
            "section_title": section.title,
            "header": header,
        }
        ids = upsert_chunks(
            qdrant, COLLECTION, acc, section.section_id, sha, bodies, vectors, metadata
        )
        old_ids += record_section(conn, COLLECTION, acc, section.section_id, sha, ids)
    return statuses, old_ids


def release_memory() -> float:
    """Free memory between filings on an 8 GB Mac; returns GPU memory still held (GB).

    PyTorch's MPS backend keeps freed GPU memory cached for reuse. With chunk
    lengths varying from 50 to 1,600 tokens the cache keeps growing, pushing the
    machine into swap (the first full run slowed from ~12 s to ~280 s per filing).
    """
    gc.collect()  # drop unreachable Python objects (Docling documents, arrays)
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
        return torch.mps.driver_allocated_memory() / 1e9
    return 0.0


def main(limit: int | None = None) -> None:
    from docling.document_converter import DocumentConverter

    start = time.perf_counter()
    model = SentenceTransformer(MODEL_NAME, model_kwargs={"dtype": torch.float16})
    chunker = SemanticTableChunker(embed=CachedEmbedder(model, f"{MODEL_NAME}|fp16"))
    converter = DocumentConverter()
    qdrant = QdrantClient(url=QDRANT_URL)
    ensure_collection(qdrant, COLLECTION, DIM)
    with connect() as conn:
        filings = filings_to_index(conn)[:limit]
        for n, filing in enumerate(filings, 1):
            t0 = time.perf_counter()
            statuses, old_ids = index_filing(
                conn, qdrant, model, chunker, filing, converter
            )
            conn.commit()  # checkpoint per filing (a crash keeps finished work)
            delete_points(qdrant, COLLECTION, old_ids)  # only after the commit
            gpu_gb = release_memory()
            summary = " ".join(f"{k}={v}" for k, v in sorted(statuses.items()))
            print(
                f"[{n:3}/{len(filings)}] {filing['ticker']:5} {filing['form']:7}"
                f" {filing['report_date']}  {summary:30} {time.perf_counter() - t0:6.1f}s"
                f"  gpu {gpu_gb:.1f} GB",
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
