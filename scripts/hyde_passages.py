"""Exp-5b: generate and cache one HyDE passage per eval question.

Run: uv run --env-file .env python scripts/hyde_passages.py

Cached in data/cache (git-ignored), so every Exp-5b run embeds the same passages
and differs only in retrieval and generation. Re-runs skip cached questions.
"""

import json

from sec_financial_rag.evals.runner import DATASET_PATH, read_jsonl
from sec_financial_rag.generation.synthesizer import cost_usd, make_client
from sec_financial_rag.retrieval.query_router import (
    HYDE,
    hypothetical_passage,
    load_cache,
)


def main() -> None:
    client, cache, spend, words = make_client(), load_cache(HYDE), 0.0, []
    with HYDE.open("a") as out:
        for rec in read_jsonl(DATASET_PATH):
            if rec["id"] not in cache:
                hypo = hypothetical_passage(client, rec["question"])
                cache[rec["id"]] = {
                    "id": rec["id"],
                    "text": hypo.text,
                    "latency_ms": hypo.latency_ms,
                    "cost_usd": cost_usd(hypo),
                }
                out.write(json.dumps(cache[rec["id"]]) + "\n")
                out.flush()
                spend += cost_usd(hypo)
            words.append(len(cache[rec["id"]]["text"].split()))
    latencies = sorted(r["latency_ms"] for r in cache.values())
    n = len(latencies)
    print(f"{n} passages; words min {min(words)} median {sorted(words)[n // 2]}")
    print(
        f"latency median {latencies[n // 2]:.0f} ms, P95 {latencies[int(n * 0.95)]:.0f} ms"
    )
    print(f"empty passages: {sum(w == 0 for w in words)}")
    print(f"Gemini spend this run: ${spend:.4f}")


if __name__ == "__main__":
    main()
