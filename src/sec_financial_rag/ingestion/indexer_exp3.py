"""Exp-3 indexing (spec §4.3 Strategy 3): children -> Qdrant, parents -> Postgres.

Run: uv run --env-file .env python -m sec_financial_rag.ingestion.indexer_exp3
         [--limit N] [--collection NAME] [--modal]
--collection: write a variant (e.g. Exp-3b) next to Exp-3 without touching it.
--modal: embed on a Modal GPU (ingestion/modal_embedder.py); parsing stays local.

Same parsing, header and per-section CDC as Exp-2 (indexer_exp2.py). Differences:
- each section becomes parents (~2,000 tokens) with children (~300 tokens);
- children are embedded as "header + child" and carry `parent_id` in their payload;
- parents go to the doc store in the same transaction as the ledger row.
"""

import sys
import time
from collections import Counter

import psycopg
import torch
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from sec_financial_rag.chunking.embed_cache import CachedEmbedder
from sec_financial_rag.chunking.header import context_header, section_label
from sec_financial_rag.chunking.parent_child import ParentChildChunker
from sec_financial_rag.chunking.semantic_table import is_narrative
from sec_financial_rag.ingestion.cdc_ledger import (
    apply_schema,
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
from sec_financial_rag.ingestion.indexer_exp2 import BATCH_SIZE, release_memory
from sec_financial_rag.ingestion.layout_parser import parse_filing
from sec_financial_rag.retrieval.doc_store import delete_stale_parents, put_parents
from sec_financial_rag.retrieval.qdrant_store import (
    delete_points,
    ensure_collection,
    upsert_chunks,
)

COLLECTION = "exp3_parent_child_bge_m3"


def index_filing(
    conn: psycopg.Connection,
    qdrant: QdrantClient,
    model,  # anything with SentenceTransformer's encode(); tests pass a fake
    chunker: ParentChildChunker,
    filing: dict,
    converter=None,
    collection: str = COLLECTION,
) -> tuple[Counter, list, list[tuple[str, str]]]:
    """Index a filing section by section.

    Returns (status counts, superseded point ids, (section_id, sha) of re-indexed
    sections). Does NOT commit: the caller commits, deletes the old points, and
    only then deletes the old parents (see doc_store).
    """
    path = filing_path(filing["ticker"], filing)
    html = path.read_text()
    acc = filing["accession_number"]
    fp = fiscal_period(html)
    period = period_label(fp, filing["report_date"])
    company = dei_value(html, "EntityRegistrantName") or filing["ticker"]
    statuses: Counter = Counter()
    old_ids: list = []
    reindexed: list[tuple[str, str]] = []
    for section in parse_filing(path, filing["form"], converter):
        header = context_header(
            company,
            filing["ticker"],
            filing["form"],
            period,
            section_label(section.section_id, section.title),
        )
        sha = content_sha256(f"{header}\n{section.text}")
        status = classify_section(conn, collection, acc, section.section_id, sha)
        statuses[status] += 1
        if status == "unchanged":
            continue
        parents = chunker.split(
            section.text, narrative=is_narrative(section.section_id)
        )
        if not parents:
            continue
        parent_ids = put_parents(
            conn, collection, acc, section.section_id, sha, [p.text for p in parents]
        )
        children, extra = [], []
        for index, (parent, pid) in enumerate(zip(parents, parent_ids)):
            children += parent.children
            extra += [{"parent_id": str(pid), "parent_index": index}] * len(
                parent.children
            )
        vectors = model.encode(
            [f"{header}\n{child}" for child in children],
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
            qdrant,
            collection,
            acc,
            section.section_id,
            sha,
            children,
            vectors,
            metadata,
            extra,
        )
        old_ids += record_section(conn, collection, acc, section.section_id, sha, ids)
        reindexed.append((section.section_id, sha))
    return statuses, old_ids, reindexed


def main(
    limit: int | None = None, collection: str = COLLECTION, use_modal: bool = False
) -> None:
    if use_modal:
        from sec_financial_rag.ingestion.modal_embedder import RemoteModel, app

        with app.run():
            run(RemoteModel(), limit, collection)
    else:
        model = SentenceTransformer(MODEL_NAME, model_kwargs={"dtype": torch.float16})
        run(model, limit, collection)


def run(
    model, limit: int | None, collection: str, tickers: set[str] | None = None
) -> None:
    """CDC-index every filing in the ledger (or only `tickers`)."""
    from docling.document_converter import DocumentConverter

    start = time.perf_counter()
    # Same cache key on both backends: cached sentence vectors (and so the semantic
    # breakpoints) are reused exactly; Modal vs MPS vectors measured equivalent.
    chunker = ParentChildChunker(embed=CachedEmbedder(model, f"{MODEL_NAME}|fp16"))
    converter = DocumentConverter()
    qdrant = QdrantClient(url=QDRANT_URL)
    ensure_collection(qdrant, collection, DIM)
    with connect() as conn:
        apply_schema(conn)  # creates parent_documents on the first run
        conn.commit()
        filings = [
            f for f in filings_to_index(conn) if not tickers or f["ticker"] in tickers
        ][:limit]
        for n, filing in enumerate(filings, 1):
            t0 = time.perf_counter()
            statuses, old_ids, reindexed = index_filing(
                conn, qdrant, model, chunker, filing, converter, collection
            )
            conn.commit()  # parents + ledger rows together (a crash keeps finished work)
            delete_points(qdrant, collection, old_ids)  # old children first...
            for section_id, sha in reindexed:  # ...then their parents
                delete_stale_parents(
                    conn, collection, filing["accession_number"], section_id, sha
                )
            conn.commit()
            gpu_gb = release_memory()
            summary = " ".join(f"{k}={v}" for k, v in sorted(statuses.items()))
            print(
                f"[{n:3}/{len(filings)}] {filing['ticker']:5} {filing['form']:7}"
                f" {filing['report_date']}  {summary:30} {time.perf_counter() - t0:6.1f}s"
                f"  gpu {gpu_gb:.1f} GB",
                flush=True,
            )
    total = qdrant.count(collection).count
    print(
        f"{collection}: {total} points   total {(time.perf_counter() - start) / 60:.1f} min"
    )


if __name__ == "__main__":
    args = sys.argv
    main(
        int(args[args.index("--limit") + 1]) if "--limit" in args else None,
        args[args.index("--collection") + 1] if "--collection" in args else COLLECTION,
        use_modal="--modal" in args,
    )
