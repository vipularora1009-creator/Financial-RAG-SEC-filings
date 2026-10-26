"""Measure an embedding model on real Exp-1 chunks: speed, peak memory, truncation.

Usage: uv run python scripts/benchmark_embeddings.py BAAI/bge-small-en-v1.5 [--fp16]
Runs one model per process so peak memory belongs to that model alone.
"""

import random
import resource
import sys
import time
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer

from sec_financial_rag.chunking.fixed_token import FixedTokenChunker
from sec_financial_rag.ingestion.naive_parser import html_to_text

CORPUS_CHUNKS = 33_000  # ≈ 15 M tokens / 448 new tokens per chunk (Step 5a/5b)


def sample_chunks(n: int = 128, seed: int = 0) -> list[str]:
    """n chunks from 8 random filings (fixed seed, so every model sees the same text)."""
    rng = random.Random(seed)
    files = sorted(Path("data/raw_filings").rglob("*.htm"))
    chunker = FixedTokenChunker()
    chunks = []
    for path in rng.sample(files, 8):
        chunks += chunker.split(html_to_text(path.read_text()))
    return rng.sample(chunks, n)


def main(model_name: str, fp16: bool = False) -> None:
    chunks = sample_chunks()
    t0 = time.perf_counter()
    dtype = torch.float16 if fp16 else torch.float32
    model = SentenceTransformer(model_name, model_kwargs={"dtype": dtype})
    load_s = time.perf_counter() - t0

    # Truncation: count each chunk with the MODEL's own tokenizer, not tiktoken.
    limit = model.max_seq_length
    lengths = [len(model.tokenizer(c)["input_ids"]) for c in chunks]
    truncated = sum(n > limit for n in lengths)

    model.encode(chunks[:8], batch_size=8)  # warm-up (first GPU call is slow)
    t0 = time.perf_counter()
    vectors = model.encode(chunks, batch_size=16, normalize_embeddings=True)
    per_s = len(chunks) / (time.perf_counter() - t0)

    peak_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9  # bytes on macOS
    print(
        f"{model_name:24} {str(dtype)[6:]:7} dim {vectors.shape[1]:4}  device {model.device}  load {load_s:5.1f}s"
        f"  {per_s:6.1f} chunks/s  full corpus ≈ {CORPUS_CHUNKS / per_s / 60:5.1f} min"
        f"  peak RAM {peak_gb:.1f} GB  model-tokens max {max(lengths)}"
        f"  truncated {truncated}/{len(chunks)} (limit {limit})"
    )


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(
            "Usage: uv run python scripts/benchmark_embeddings.py <model_name> [--fp16]"
        )
        sys.exit(0 if len(sys.argv) > 1 and sys.argv[1] in ("-h", "--help") else 1)
    main(sys.argv[1], fp16="--fp16" in sys.argv)
