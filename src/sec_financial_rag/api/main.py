"""FastAPI service (spec §6): /query, /ingest/delta, /metrics, /health.

Run (Qdrant + Postgres up, index built):
    uv run --env-file .env uvicorn sec_financial_rag.api.main:app --port 8000
Interactive docs at http://localhost:8000/docs.
"""

import asyncio
import json
import logging
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from sec_financial_rag.api.service import TOP_N, RagService

REPORTS = Path("evals/reports")
log = logging.getLogger("sec_financial_rag.api")
QUERY_TIMEOUT_S = 120  # a stuck pipeline answers 504 instead of hanging the client
EXPERIMENT_FIELDS = ("hit@5", "mrr@10", "em_lenient", "em_strict", "refusal_rate")


class QueryRequest(BaseModel):
    question: str = Field(min_length=5, max_length=1000)
    top_n: int = Field(
        TOP_N, ge=1, le=10, description="Parents given to the generator."
    )
    hyde: bool = Field(True, description="Embed a HyDE passage for the dense search.")
    decompose: bool = Field(
        False, description="Split comparative questions into filtered sub-queries."
    )


class Source(BaseModel):
    n: int
    ticker: str | None
    company: str | None
    form_type: str | None
    report_date: str | None
    fiscal_year: int | None
    fiscal_period: str | None
    section_title: str | None
    accession_number: str | None
    text: str


class QueryResponse(BaseModel):
    answer: str
    refused: bool
    sources: list[Source]
    route: dict
    timings_ms: dict[str, float]
    cost_usd: float


class DeltaRequest(BaseModel):
    tickers: list[str] | None = Field(
        None, description="Only these companies' filings (default: every filing)."
    )
    use_modal: bool = Field(
        False, description="Embed on a Modal GPU instead of locally."
    )


def ingest_delta(tickers: set[str] | None, use_modal: bool) -> dict:
    """CDC run on the dense collection, then sync the served hybrid collection.

    Only sections whose sha changed are re-embedded (cdc_ledger); the hybrid
    collection then copies new point ids and deletes superseded ones.
    """
    from qdrant_client import QdrantClient

    from sec_financial_rag.ingestion import build_exp4, indexer, indexer_exp3
    from sec_financial_rag.retrieval.bm25 import Bm25Encoder

    if use_modal:
        from sec_financial_rag.ingestion.modal_embedder import RemoteModel, app

        with app.run():
            indexer_exp3.run(RemoteModel(), None, build_exp4.SOURCE, tickers)
    else:
        import torch
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(
            indexer.MODEL_NAME, model_kwargs={"dtype": torch.float16}
        )
        indexer_exp3.run(model, None, build_exp4.SOURCE, tickers)
    copied, deleted = build_exp4.sync_with_bm25(
        QdrantClient(url=indexer.QDRANT_URL),
        Bm25Encoder(),
        build_exp4.SOURCE,
        build_exp4.COLLECTION,
    )
    return {"points_copied": copied, "points_deleted": deleted}


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(p * len(ordered)))]


def experiment_results(reports: Path = REPORTS) -> dict[str, dict]:
    """Headline metrics of every evaluation summary (aggregates only, no questions)."""
    results = {}
    for path in sorted(reports.glob("*_eval_summary.json")):
        overall = json.loads(path.read_text())["all"]
        name = path.name.removesuffix("_eval_summary.json")
        results[name] = {k: overall.get(k) for k in EXPERIMENT_FIELDS}
    return results


def create_app(
    service_factory=RagService.from_env,
    delta_runner=ingest_delta,
    query_timeout_s: float = QUERY_TIMEOUT_S,
) -> FastAPI:
    """App factory: tests pass a fake service and a fake ingestion runner."""
    state: dict = {"service": None}
    stats = {"queries": 0, "errors": 0, "refusals": 0, "cost_usd": 0.0, "latency": []}
    jobs: dict[str, dict] = {}
    lock = threading.Lock()  # one ingestion job at a time

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        state["service"] = service_factory()  # models load once, at startup
        yield

    app = FastAPI(
        title="SEC Financial RAG",
        description="Question answering over SEC 10-K / 10-Q filings with cited sources.",
        version="1.0.0",
        lifespan=lifespan,
    )

    @app.post("/query", response_model=QueryResponse)
    async def query(request: QueryRequest) -> dict:
        try:
            result = await asyncio.wait_for(
                state["service"].answer(
                    request.question, request.top_n, request.hyde, request.decompose
                ),
                timeout=query_timeout_s,
            )
        except TimeoutError as exc:
            log.error("query timed out after %s s", query_timeout_s)
            stats["errors"] += 1
            raise HTTPException(504, "pipeline timed out") from exc
        except Exception as exc:
            log.exception("query failed")  # full traceback server-side only
            stats["errors"] += 1
            raise HTTPException(502, f"pipeline failed: {type(exc).__name__}") from exc
        stats["queries"] += 1
        stats["refusals"] += result["refused"]
        stats["cost_usd"] += result["cost_usd"]
        stats["latency"].append(result["timings_ms"]["total"])
        return result

    @app.post("/ingest/delta", status_code=202)
    def start_delta(request: DeltaRequest) -> dict:
        if not lock.acquire(blocking=False):
            raise HTTPException(409, "an ingestion job is already running")
        job_id = uuid.uuid4().hex[:12]
        jobs[job_id] = {"status": "running", "started": time.time()}

        def work() -> None:
            try:
                tickers = (
                    {t.upper() for t in request.tickers} if request.tickers else None
                )
                jobs[job_id] |= {
                    "status": "done",
                    "result": delta_runner(tickers, request.use_modal),
                }
            except Exception as exc:  # noqa: BLE001 - job boundary: report any failure
                jobs[job_id] |= {
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            finally:
                jobs[job_id]["finished"] = time.time()
                lock.release()

        threading.Thread(target=work, daemon=True).start()
        return {"job_id": job_id, "status": "running"}

    @app.get("/ingest/delta/{job_id}")
    def delta_status(job_id: str) -> dict:
        if job_id not in jobs:
            raise HTTPException(404, "unknown job")
        return {"job_id": job_id, **jobs[job_id]}

    @app.get("/metrics")
    def metrics() -> dict:
        latency = stats["latency"]
        return {
            "service": {
                "queries": stats["queries"],
                "errors": stats["errors"],
                "refusal_rate": stats["refusals"] / stats["queries"]
                if stats["queries"]
                else None,
                "latency_ms_p50": percentile(latency, 0.5),
                "latency_ms_p95": percentile(latency, 0.95),
                "cost_usd_total": round(stats["cost_usd"], 6),
            },
            "experiments": experiment_results(),
        }

    @app.get("/health")
    def health() -> dict:
        checks = {}
        for name, check in state["service"].checks.items():
            try:
                check()
                checks[name] = "ok"
            except Exception as exc:  # noqa: BLE001 - health boundary: report, don't crash
                checks[name] = f"error: {type(exc).__name__}"
        healthy = all(v == "ok" for v in checks.values())
        if not healthy:
            raise HTTPException(503, checks)
        return {"status": "ok", **checks}

    return app


def __getattr__(name: str):
    """`uvicorn sec_financial_rag.api.main:app` builds the real app on first access,
    so importing this module (tests) never loads models or connects to services."""
    if name == "app":
        globals()["app"] = create_app()
        return globals()["app"]
    raise AttributeError(name)
