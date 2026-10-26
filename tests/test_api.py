import asyncio
import threading
import time
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from sec_financial_rag.api.main import create_app, experiment_results
from sec_financial_rag.api.service import RagService
from sec_financial_rag.retrieval.query_router import (
    Decomposition,
    QueryMetadata,
    SubQuery,
)

PARENT = {
    "ticker": "MMM",
    "company": "3M COMPANY",
    "form_type": "10-K",
    "report_date": "2022-12-31",
    "fiscal_year": 2022,
    "fiscal_period": "FY",
    "section_title": "Risk Factors.",
    "accession_number": "acc-1",
    "text": "Net sales were $34,229 million.",
}


def call(text="", **extra):
    """Stands in for an Extraction / Hypothetical / Answer: tokens for cost_usd()."""
    return SimpleNamespace(text=text, input_tokens=1000, output_tokens=100, **extra)


def fake_service(seen: dict, delay: float = 0.0) -> RagService:
    def slow(result):
        def fn(*args):
            time.sleep(delay)
            return result

        return fn

    def search(question, where=None, dense_text=None):
        seen["search"] = (question, where, dense_text)
        return [{**PARENT, "parent_id": "p1"}]

    meta = QueryMetadata(ticker="MMM", fiscal_year=2022)
    return RagService(
        search=search,
        expand=lambda children: [dict(PARENT)],
        extract=slow(call(metadata=meta)),
        hyde=slow(call("Item 7. Net sales increased ...")),
        decompose=lambda q: call(
            decomposition=Decomposition(
                sub_queries=[
                    SubQuery(question="3M sales", ticker="MMM", fiscal_year=2022),
                    SubQuery(question="J&J sales", ticker="JNJ", fiscal_year=2022),
                ]
            )
        ),
        synthesize=lambda q, chunks: call("Net sales were $34,229 million [1]."),
        checks={"qdrant": lambda: 1},
    )


def test_router_calls_run_concurrently_and_feed_the_search():
    seen = {}
    service = fake_service(seen, delay=0.3)
    t0 = time.perf_counter()
    result = asyncio.run(service.answer("What were 3M's FY2022 net sales?"))
    assert time.perf_counter() - t0 < 0.5  # two 0.3 s calls overlapped
    question, where, dense_text = seen["search"]
    assert question == "What were 3M's FY2022 net sales?"  # BM25 gets the question
    assert where.must[0].match.value == "MMM"  # the extracted filter
    assert dense_text.startswith("Item 7.")  # HyDE for the dense side
    assert result["route"] == {
        "ticker": "MMM",
        "fiscal_year": 2022,
        "filtered": True,
        "hyde": True,
        "decomposed": False,
    }
    assert result["sources"][0]["n"] == 1 and result["sources"][0]["ticker"] == "MMM"
    assert result["cost_usd"] == pytest.approx(3 * (1000 * 0.75 + 100 * 3.75) / 1e6)


def test_comparative_question_is_decomposed_into_filtered_sub_queries():
    seen = {}
    result = asyncio.run(
        fake_service(seen).answer("Compare 3M and J&J sales", decompose=True)
    )
    assert [q["ticker"] for q in result["route"]["sub_queries"]] == ["MMM", "JNJ"]


@pytest.fixture
def client():
    started = threading.Event()
    release = threading.Event()

    def delta_runner(tickers, use_modal):
        started.set()
        release.wait(5)
        return {"points_copied": 3, "points_deleted": 1, "tickers": sorted(tickers)}

    app = create_app(
        service_factory=lambda: fake_service({}), delta_runner=delta_runner
    )
    with TestClient(app) as test_client:
        test_client.started, test_client.release = started, release
        yield test_client


def test_query_returns_answer_sources_route_and_cost(client):
    response = client.post(
        "/query", json={"question": "What were 3M's FY2022 net sales?"}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["answer"].endswith("[1].") and not body["refused"]
    assert body["sources"][0]["accession_number"] == "acc-1"
    assert set(body["timings_ms"]) == {"route", "retrieval", "generation", "total"}


def test_query_validates_input(client):
    assert client.post("/query", json={"question": "hi"}).status_code == 422
    assert (
        client.post(
            "/query", json={"question": "Valid question?", "top_n": 50}
        ).status_code
        == 422
    )


def test_metrics_count_queries(client):
    client.post("/query", json={"question": "What were 3M's FY2022 net sales?"})
    service = client.get("/metrics").json()["service"]
    assert service["queries"] == 1 and service["refusal_rate"] == 0.0
    assert service["latency_ms_p95"] is not None


def test_ingest_delta_runs_one_job_at_a_time(client):
    first = client.post("/ingest/delta", json={"tickers": ["mmm"]})
    assert first.status_code == 202
    client.started.wait(5)
    assert client.post("/ingest/delta", json={}).status_code == 409  # still running
    client.release.set()
    job_id = first.json()["job_id"]
    for _ in range(50):
        status = client.get(f"/ingest/delta/{job_id}").json()
        if status["status"] != "running":
            break
        time.sleep(0.05)
    assert status["status"] == "done"
    assert status["result"] == {
        "points_copied": 3,
        "points_deleted": 1,
        "tickers": ["MMM"],
    }
    assert client.get("/ingest/delta/nope").status_code == 404


def test_health_reports_each_dependency(client):
    assert client.get("/health").json() == {"status": "ok", "qdrant": "ok"}


def test_experiment_results_read_only_aggregates(tmp_path):
    (tmp_path / "exp9_eval_summary.json").write_text(
        '{"all": {"hit@5": 0.5, "em_lenient": 0.4, "n": 76}}'
    )
    assert experiment_results(tmp_path) == {
        "exp9": {
            "hit@5": 0.5,
            "mrr@10": None,
            "em_lenient": 0.4,
            "em_strict": None,
            "refusal_rate": None,
        }
    }


def test_a_stuck_pipeline_times_out_instead_of_hanging():
    class Stuck:
        def __init__(self):
            self.checks = {}

        async def answer(self, *args):
            await asyncio.sleep(5)

    app = create_app(service_factory=Stuck, query_timeout_s=0.2)
    with TestClient(app) as test_client:
        response = test_client.post("/query", json={"question": "Will this hang?"})
    assert response.status_code == 504
