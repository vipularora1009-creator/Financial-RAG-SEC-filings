"""Exp-5 step 3: how accurate is the query metadata extractor, on its own?

Run: uv run --env-file .env python scripts/router_accuracy.py

Extracts metadata for every eval question (cached in data/cache, so re-runs
and the Exp-5 evals reuse the same extractions at no cost), then checks each
field against the gold filing's payload and whether the filter still contains
the gold filing. A filter that drops the gold filing is a guaranteed miss.
"""

import json
from collections import Counter

from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

from sec_financial_rag.evals.runner import DATASET_PATH, read_jsonl
from sec_financial_rag.generation.synthesizer import cost_usd, make_client
from sec_financial_rag.ingestion.build_exp4 import COLLECTION
from sec_financial_rag.ingestion.indexer import QDRANT_URL
from sec_financial_rag.retrieval.query_router import (
    EXTRACTIONS,
    QueryMetadata,
    build_filter,
    extract_metadata,
    load_cache,
)


def gold_payload(qdrant: QdrantClient, accession: str) -> dict:
    points, _ = qdrant.scroll(
        COLLECTION,
        scroll_filter=Filter(
            must=[
                FieldCondition(
                    key="accession_number", match=MatchValue(value=accession)
                )
            ]
        ),
        limit=1,
        with_payload=["ticker", "fiscal_year", "fiscal_period"],
    )
    return points[0].payload


def keeps_gold(qdrant: QdrantClient, where: Filter | None, accession: str) -> bool:
    gold = FieldCondition(key="accession_number", match=MatchValue(value=accession))
    combined = Filter(must=[gold, *(where.must if where else [])])
    return qdrant.count(COLLECTION, count_filter=combined, exact=True).count > 0


def main() -> None:
    qdrant, client, cache = (
        QdrantClient(url=QDRANT_URL),
        make_client(),
        load_cache(EXTRACTIONS),
    )
    tally, spend, latencies, misses = Counter(), 0.0, [], []
    with EXTRACTIONS.open("a") as out:
        for rec in read_jsonl(DATASET_PATH):
            if rec["id"] not in cache:
                ex = extract_metadata(client, rec["question"])
                cache[rec["id"]] = {
                    "id": rec["id"],
                    "metadata": ex.metadata.model_dump(mode="json"),
                    "latency_ms": ex.latency_ms,
                    "cost_usd": cost_usd(ex),
                }
                out.write(json.dumps(cache[rec["id"]]) + "\n")
                out.flush()
                spend += cost_usd(ex)
            row = cache[rec["id"]]
            meta = QueryMetadata(**row["metadata"])
            latencies.append(row["latency_ms"])
            accession = rec["filings"][0]["accession_number"]
            gold = gold_payload(qdrant, accession)
            gold_quarter = (
                None if gold["fiscal_period"] == "FY" else gold["fiscal_period"]
            )

            tally["ticker right"] += meta.ticker == gold["ticker"]
            tally["ticker empty"] += meta.ticker is None
            tally["year named"] += meta.fiscal_year is not None
            tally["year = gold"] += meta.fiscal_year == gold["fiscal_year"]
            tally["quarter = gold"] += meta.quarter == gold_quarter
            for name, quarter in (
                ("filter keeps gold", False),
                ("+ quarter keeps gold", True),
            ):
                kept = keeps_gold(qdrant, build_filter(meta, quarter), accession)
                tally[name] += kept
                if not kept and not quarter:
                    misses.append((rec["id"], row["metadata"], gold))

    n = len(latencies)
    for name, count in tally.items():
        print(f"{name:22s} {count:3d}/{n}")
    latencies.sort()
    print(
        f"latency median {latencies[n // 2]:.0f} ms, P95 {latencies[int(n * 0.95)]:.0f} ms"
    )
    print(f"Gemini spend this run: ${spend:.4f}")
    for miss in misses:
        print("DROPS GOLD:", *miss)


if __name__ == "__main__":
    main()
