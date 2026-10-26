"""Exp-F: parse all filings with DoclingReader and index them with LlamaIndex.

Run: uv run --group exp-f --env-file .env python scripts/exp_f_index.py [--limit N] [--parse-only]

Parsed Markdown is cached in data/cache/exp_f_markdown, so a re-run only embeds.
Embedding runs on Modal (bge-m3, L4) like Exp-3b.
"""

import sys
import time

from qdrant_client import QdrantClient

from sec_financial_rag.exp_f import pipeline
from sec_financial_rag.ingestion.build_exp4 import COLLECTION as OUR_COLLECTION
from sec_financial_rag.ingestion.indexer import QDRANT_URL


def main(limit: int | None, parse_only: bool) -> None:
    qdrant = QdrantClient(url=QDRANT_URL)
    start = time.perf_counter()
    metadata = pipeline.filing_metadata(qdrant, OUR_COLLECTION)
    documents = pipeline.load_documents(metadata, limit)
    parsed = time.perf_counter()
    print(
        f"{len(documents)} documents parsed/loaded in {(parsed - start) / 60:.1f} min"
    )
    if parse_only:
        return
    if qdrant.collection_exists(pipeline.COLLECTION):
        qdrant.delete_collection(pipeline.COLLECTION)  # derived: always rebuilt whole

    from sec_financial_rag.ingestion.modal_embedder import RemoteModel, app

    with app.run():
        embed = pipeline.EncodeEmbedding(model=RemoteModel())
        nodes, leaves = pipeline.build_index(documents, qdrant, embed)
    print(
        f"{nodes} nodes, {leaves} leaves embedded"
        f"   index {(time.perf_counter() - parsed) / 60:.1f} min"
    )


if __name__ == "__main__":
    args = sys.argv
    main(
        int(args[args.index("--limit") + 1]) if "--limit" in args else None,
        "--parse-only" in args,
    )
