"""Experiment evaluation (spec §5.1): retrieval + generation metrics per question.

Run: uv run --env-file .env python -m sec_financial_rag.evals.runner
         [--retrieval-only] [--collection exp2_semantic_bge_m3]   (default: Exp-1's)
         [--parents]   (Exp-3: search children, score and answer with their parents)
         [--mode dense|bm25|hybrid]   (Exp-4: named vectors + RRF, retrieval/hybrid.py)
         [--exact]   (with --mode: exact instead of approximate (HNSW) search)
         [--rerank]   (with --parents: cross-encoder on Modal re-sorts the top-25 parents)
         [--top-n N]  (passages given to the generator; default 5)
         [--tag NAME] (appended to the report name, e.g. a repeat run for noise)
         [--route]    (with --mode, Exp-5a: filter by company/year from the cached
                       router extractions; run scripts/router_accuracy.py first)
         [--hyde]     (with --mode, Exp-5b: dense search embeds the cached HyDE
                       passage; run scripts/hyde_passages.py first)
Writes a per-question report (incl. answers and their context, so metrics can be
re-scored later without paying for new answers) and a summary to evals/reports/.
"""

import json
import sys
import time
from collections.abc import Callable
from pathlib import Path

from sec_financial_rag.evals.dataset import DATASET_PATH, read_jsonl
from sec_financial_rag.evals.faithfulness import numeric_faithfulness
from sec_financial_rag.evals.metrics import hit_at_k, is_relevant, reciprocal_rank
from sec_financial_rag.evals.numeric import numeric_match
from sec_financial_rag.generation.prompts import INSUFFICIENT
from sec_financial_rag.generation.synthesizer import Answer, cost_usd

K = 10  # retrieved for the retrieval metrics
TOP_N = 5  # passed to the generator (spec Exp-1: top-k=5)
CHILD_POOL = 50  # Exp-3: children searched before deduplicating to parents
REPORT_DIR = Path("evals/reports")

Search = Callable[[str], list[dict]]  # question -> ranked payloads (each with "score")
AnswerFn = Callable[[str, list[dict]], Answer]  # (question, chunks) -> Answer
# (question, ranked children) -> ranked parents; the question is for reranking
Expand = Callable[[str, list[dict]], list[dict]]
# record -> (search keyword args such as where / dense_text,
#            extra row fields such as route_latency_ms / route_cost_usd)
Route = Callable[[dict], tuple[dict, dict]]


def percentile(values: list[float], p: float) -> float:
    """Nearest-rank percentile, p in [0, 1]."""
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(p * len(ordered)))]


def mean_of(rows: list[dict], key: str) -> float | None:
    """Mean over rows where the metric applies (None = not applicable)."""
    values = [r[key] for r in rows if r.get(key) is not None]
    return sum(values) / len(values) if values else None


def share_fully_faithful(rows: list[dict]) -> float | None:
    """Share of answers (with numbers) where every number is grounded."""
    scores = [r["faithfulness"] for r in rows if r["faithfulness"] is not None]
    return sum(score == 1.0 for score in scores) / len(scores) if scores else None


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    summary = {
        "n": n,
        **{f"hit@{k}": sum(r[f"hit@{k}"] for r in rows) / n for k in (3, 5, 10)},
        "mrr@10": sum(r["rr"] for r in rows) / n,
        "retrieval_p95_ms": percentile([r["latency_ms"] for r in rows], 0.95),
    }
    if "child_rr" in rows[0]:  # Exp-3: matching quality before parent expansion
        summary |= {
            **{
                f"child_hit@{k}": sum(r[f"child_hit@{k}"] for r in rows) / n
                for k in (3, 5, 10)
            },
            "child_mrr@10": sum(r["child_rr"] for r in rows) / n,
        }
    if "answer" in rows[0]:
        summary |= {
            "em_lenient": mean_of(rows, "em_lenient"),
            "em_strict": mean_of(rows, "em_strict"),
            "em_n": sum(r["em_lenient"] is not None for r in rows),
            "faithfulness": mean_of(rows, "faithfulness"),
            "fully_faithful": share_fully_faithful(rows),
            "refusal_rate": sum(r["refused"] for r in rows) / n,
            # Routing (Exp-5) is part of the price and the wait, when present.
            "cost_per_1k_usd": 1000
            * sum(r["cost_usd"] + r.get("route_cost_usd", 0) for r in rows)
            / n,
            "p95_total_ms": percentile(
                [
                    r.get("route_latency_ms", 0) + r["latency_ms"] + r["gen_latency_ms"]
                    for r in rows
                ],
                0.95,
            ),
        }
    return summary


def retrieval_row(rec: dict, hits: list[dict], latency_ms: float, k: int) -> dict:
    relevant = [is_relevant(h, rec["evidence"]) for h in hits]
    return {
        "id": rec["id"],
        "question": rec["question"],
        "question_type": rec["meta"]["question_type"],
        "first_relevant_rank": next(
            (rank for rank, ok in enumerate(relevant, start=1) if ok), None
        ),
        **{f"hit@{n}": hit_at_k(relevant, n) for n in (3, 5, 10)},
        "rr": reciprocal_rank(relevant, k),
        "latency_ms": round(latency_ms, 1),
        "top5": [  # what was retrieved instead, for failure analysis
            {
                "ticker": h.get("ticker"),
                "accession_number": h["accession_number"],
                "report_date": h.get("report_date"),
                "chunk_index": h.get("chunk_index"),
                "score": round(h["score"], 3),
                "relevant": ok,
            }
            for h, ok in zip(hits[:5], relevant)
        ],
    }


def child_fields(rec: dict, children: list[dict], k: int) -> dict:
    """Hit@k / RR on the children themselves: comparable with Exp-1/2 chunk hits."""
    relevant = [is_relevant(c, rec["evidence"]) for c in children]
    return {
        **{f"child_hit@{n}": hit_at_k(relevant, n) for n in (3, 5, 10)},
        "child_rr": reciprocal_rank(relevant, k),
    }


def generation_fields(rec: dict, chunks: list[dict], answer_fn: AnswerFn) -> dict:
    """Answer from the top chunks and score it: EM (lenient/strict), faithfulness, cost."""
    answer = answer_fn(rec["question"], chunks)
    lenient, strict = numeric_match(rec["answer"], answer.text)
    return {
        "answer": answer.text,
        "gold_answer": rec["answer"],
        "refused": answer.text.startswith(INSUFFICIENT),
        "em_lenient": lenient,
        "em_strict": strict,
        "faithfulness": numeric_faithfulness(answer.text, chunks, rec["question"]),
        "input_tokens": answer.input_tokens,
        "output_tokens": answer.output_tokens,
        "cost_usd": cost_usd(answer),
        "gen_latency_ms": round(answer.latency_ms, 1),
        "context": [c["text"] for c in chunks],  # kept so metrics can be re-scored free
    }


def evaluate(
    records: list[dict],
    search: Search,
    k: int = K,
    answer_fn: AnswerFn | None = None,
    expand: Expand | None = None,
    top_n: int = TOP_N,
    done: dict[str, dict] | None = None,
    on_row: Callable[[dict], None] | None = None,
    route: Route | None = None,
) -> tuple[list[dict], dict]:
    """Score every question; return per-question rows and summaries (overall + by type).

    With `expand` (Exp-3), `search` returns children: the standard metrics and the
    answer use the parents, and child_hit@k / child_rr score the children.
    `done` (id -> row) are reused without searching or answering again (resume);
    `on_row` is called with each new row as soon as it is scored (checkpoint).
    With `route` (Exp-5), each search gets that question's keyword arguments.
    """
    rows = []
    for rec in records:
        if done and rec["id"] in done:
            rows.append(done[rec["id"]])
            continue
        kwargs, route_fields = route(rec) if route else ({}, {})
        t0 = time.perf_counter()
        children = search(rec["question"], **kwargs)
        hits = (expand(rec["question"], children) if expand else children)[:k]
        row = retrieval_row(rec, hits, (time.perf_counter() - t0) * 1000, k)
        row |= route_fields
        if expand:
            row |= child_fields(rec, children[:k], k)
        if answer_fn is not None:
            row |= generation_fields(rec, hits[:top_n], answer_fn)
        if on_row:
            on_row(row)
        rows.append(row)
    return rows, summaries(rows)


def summaries(rows: list[dict]) -> dict:
    """Summary over all rows and per question type."""
    groups = {"all": rows}
    for row in rows:
        groups.setdefault(row["question_type"], []).append(row)
    return {name: summarize(group) for name, group in groups.items()}


def make_search(qdrant, model, collection: str, k: int = K) -> Search:
    """Real search: embed the question with the same model, cosine top-k in Qdrant."""

    def search(question: str) -> list[dict]:
        vector = model.encode(question, normalize_embeddings=True)
        points = qdrant.query_points(
            collection, query=vector.tolist(), limit=k, with_payload=True
        ).points
        return [{**p.payload, "score": p.score} for p in points]

    return search


def with_retries(fn: Callable[[], Answer], attempts: int = 4) -> Answer:
    """Retry temporary API failures (rate limits, server errors) with backoff."""
    from google.genai import errors

    for attempt in range(attempts):
        try:
            return fn()
        except (errors.ServerError, errors.ClientError) as exc:
            retryable = isinstance(exc, errors.ServerError) or exc.code == 429
            if not retryable or attempt == attempts - 1:
                raise
            time.sleep(5 * 2**attempt)  # 5, 10, 20 s
    raise AssertionError("unreachable")


def main(
    generate: bool = True,
    collection: str | None = None,
    parents: bool = False,
    mode: str | None = None,
    exact: bool = False,
    rerank: bool = False,
    top_n: int = TOP_N,
    tag: str | None = None,
    routed: bool = False,
    hyde: bool = False,
) -> None:
    import contextlib

    import torch

    if rerank and not parents:
        raise SystemExit("--rerank needs --parents (it re-sorts the expanded parents)")
    if (routed or hyde) and not mode:
        raise SystemExit("--route / --hyde need --mode (hybrid.make_search takes them)")
    from qdrant_client import QdrantClient
    from sentence_transformers import SentenceTransformer

    from sec_financial_rag.generation.synthesizer import make_client, synthesize
    from sec_financial_rag.ingestion import indexer

    model = SentenceTransformer(
        indexer.MODEL_NAME, model_kwargs={"dtype": torch.float16}
    )
    qdrant = QdrantClient(url=indexer.QDRANT_URL)
    collection = collection or indexer.COLLECTION
    if mode:
        from sec_financial_rag.retrieval import hybrid
        from sec_financial_rag.retrieval.bm25 import Bm25Encoder

        search = hybrid.make_search(
            qdrant, model, Bm25Encoder(), collection, mode, exact
        )
    else:
        search = make_search(qdrant, model, collection, CHILD_POOL if parents else K)
    search("warm-up")  # first GPU call is slow; keep it out of the latency numbers

    expand = None
    if parents:
        from sec_financial_rag.ingestion.cdc_ledger import connect
        from sec_financial_rag.retrieval.doc_store import expand_to_parents, get_parents

        conn = connect()
        conn.autocommit = True  # read-only lookups; no transaction left open

        score = None
        if rerank:
            from sec_financial_rag.ingestion.modal_embedder import RemoteReranker, app
            from sec_financial_rag.retrieval.rerank import rerank as rerank_parents

        def expand(question: str, children: list[dict]) -> list[dict]:
            found = expand_to_parents(children, lambda ids: get_parents(conn, ids))
            return rerank_parents(question, found, score) if score else found

    route = None
    if routed or hyde:
        from sec_financial_rag.retrieval.query_router import (
            EXTRACTIONS,
            HYDE,
            QueryMetadata,
            build_filter,
            load_cache,
        )

        caches = {}
        if routed:
            caches["extraction"] = load_cache(EXTRACTIONS)
        if hyde:
            caches["hyde"] = load_cache(HYDE)

        def route(rec: dict) -> tuple[dict, dict]:
            rows = {}
            for name, cache in caches.items():
                if rec["id"] not in cache:
                    raise SystemExit(
                        f"no cached {name} for {rec['id']}: run its script"
                    )
                rows[name] = cache[rec["id"]]
            kwargs, fields = (
                {},
                {
                    # Charged in sequence, as the calls are made (not run in parallel).
                    "route_latency_ms": round(
                        sum(r["latency_ms"] for r in rows.values()), 1
                    ),
                    "route_cost_usd": sum(r["cost_usd"] for r in rows.values()),
                },
            )
            if "extraction" in rows:
                meta = rows["extraction"]["metadata"]
                kwargs["where"] = build_filter(QueryMetadata(**meta))
                fields["route"] = meta
            if "hyde" in rows:
                kwargs["dense_text"] = rows["hyde"]["text"]
            return kwargs, fields

    answer_fn = None
    if generate:
        client = make_client()

        def answer_fn(question: str, chunks: list[dict]) -> Answer:
            return with_retries(lambda: synthesize(client, question, chunks))

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    # Reports are per collection (+ mode): earlier experiments are never overwritten.
    name = "_".join(
        [
            collection,
            *([mode] if mode else []),
            *(["exact"] if exact else []),
            *(["route"] if routed else []),
            *(["hyde"] if hyde else []),
            *(["parents"] if parents else []),
            *(["rerank"] if rerank else []),
            *([f"top{top_n}"] if top_n != TOP_N else []),
            *([tag] if tag else []),
        ]
    )
    # Checkpoint: every scored row is appended here at once, so a failure (credits,
    # network) loses at most the question in flight; a re-run resumes from it.
    partial = REPORT_DIR / f"{name}_eval.partial.jsonl"
    done = {}
    if partial.exists():
        done = {r["id"]: r for r in map(json.loads, partial.read_text().splitlines())}
        print(f"resuming: {len(done)} questions already scored")

    gpu = app.run() if rerank else contextlib.nullcontext()
    with gpu, partial.open("a") as checkpoint:  # Modal app lives only for the eval
        if rerank:
            score = RemoteReranker().score
            score("warm-up", ["cold start: container + model load"])

        def save(row: dict) -> None:
            checkpoint.write(json.dumps(row) + "\n")
            checkpoint.flush()

        rows, summary = evaluate(
            read_jsonl(DATASET_PATH),
            search,
            answer_fn=answer_fn,
            expand=expand,
            top_n=top_n,
            done=done,
            on_row=save,
            route=route,
        )
    with (REPORT_DIR / f"{name}_eval.jsonl").open("w") as f:
        f.writelines(json.dumps(row) + "\n" for row in rows)
    (REPORT_DIR / f"{name}_eval_summary.json").write_text(json.dumps(summary, indent=2))
    partial.unlink()  # complete: the full report replaces the checkpoint
    print(json.dumps(summary["all"], indent=2))
    if generate:
        spend = sum(r["cost_usd"] + r.get("route_cost_usd", 0) for r in rows)
        print(f"Gemini spend for this report (incl. resumed rows): ${spend:.2f}")


if __name__ == "__main__":
    args = sys.argv
    if "-h" in args or "--help" in args:
        print("Usage: uv run python -m sec_financial_rag.evals.runner [OPTIONS]")
        print("\nOptions:")
        print("  --collection NAME   Qdrant collection to evaluate")
        print("  --mode MODE         Search mode: dense, bm25, or hybrid")
        print("  --exact             Use exact (brute-force) search")
        print(
            "  --parents           Expand matched child chunks to parents via Postgres doc store"
        )
        print(
            "  --route             Use query router to filter by ticker and fiscal year"
        )
        print(
            "  --hyde              Embed a hypothetical filing passage for dense retrieval"
        )
        print("  --rerank            Rerank candidates using cross-encoder")
        print("  --retrieval-only    Skip answer generation (no LLM cost)")
        print(
            "  --top-n N           Number of passages sent to the generator (default: 5)"
        )
        print("  --tag TAG           Optional tag appended to the report filename")
        print("  -h, --help          Show this message and exit")
        sys.exit(0)
    main(
        generate="--retrieval-only" not in args,
        collection=args[args.index("--collection") + 1]
        if "--collection" in args
        else None,
        parents="--parents" in args,
        mode=args[args.index("--mode") + 1] if "--mode" in args else None,
        exact="--exact" in args,
        rerank="--rerank" in args,
        top_n=int(args[args.index("--top-n") + 1]) if "--top-n" in args else TOP_N,
        tag=args[args.index("--tag") + 1] if "--tag" in args else None,
        routed="--route" in args,
        hyde="--hyde" in args,
    )
