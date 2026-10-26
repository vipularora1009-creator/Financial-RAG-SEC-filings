"""Exp-F: evaluate the LlamaIndex pipeline with our questions, prompt and metrics.

Run: uv run --group exp-f --env-file .env python scripts/exp_f_eval.py [--retrieval-only]

Same as evals.runner, except `search`: LlamaIndex hybrid retrieval (exact search)
followed by AutoMergingRetriever. Its results are already final passages (leaves
or merged parents), so there is no separate parent expansion.
"""

import json
import sys

import torch
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from sec_financial_rag.evals.runner import (
    DATASET_PATH,
    REPORT_DIR,
    Answer,
    evaluate,
    read_jsonl,
    with_retries,
)
from sec_financial_rag.exp_f import pipeline
from sec_financial_rag.generation.synthesizer import make_client, synthesize
from sec_financial_rag.ingestion import indexer


def main(generate: bool) -> None:
    model = SentenceTransformer(
        indexer.MODEL_NAME, model_kwargs={"dtype": torch.float16}
    )
    embed = pipeline.EncodeEmbedding(model=model)
    search = pipeline.make_search(QdrantClient(url=indexer.QDRANT_URL), embed)
    search("warm-up")  # first call loads models; keep it out of the latency numbers

    answer_fn = None
    if generate:
        client = make_client()

        def answer_fn(question: str, chunks: list[dict]) -> Answer:
            return with_retries(lambda: synthesize(client, question, chunks))

    name = f"{pipeline.COLLECTION}_hybrid_exact"
    partial = REPORT_DIR / f"{name}_eval.partial.jsonl"
    done = {}
    if partial.exists():
        done = {r["id"]: r for r in map(json.loads, partial.read_text().splitlines())}
        print(f"resuming: {len(done)} questions already scored")
    with partial.open("a") as checkpoint:

        def save(row: dict) -> None:
            checkpoint.write(json.dumps(row) + "\n")
            checkpoint.flush()

        rows, summary = evaluate(
            read_jsonl(DATASET_PATH),
            search,
            answer_fn=answer_fn,
            done=done,
            on_row=save,
        )
    with (REPORT_DIR / f"{name}_eval.jsonl").open("w") as f:
        f.writelines(json.dumps(row) + "\n" for row in rows)
    (REPORT_DIR / f"{name}_eval_summary.json").write_text(json.dumps(summary, indent=2))
    partial.unlink()
    print(json.dumps(summary["all"], indent=2))
    if generate:
        print(f"Gemini spend: ${sum(r['cost_usd'] for r in rows):.2f}")


if __name__ == "__main__":
    main(generate="--retrieval-only" not in sys.argv)
