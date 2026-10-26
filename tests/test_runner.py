import pytest

from sec_financial_rag.evals.runner import evaluate

EVIDENCE = "Total net sales were 90,753 million dollars in the quarter ended March"
RIGHT = {
    "accession_number": "acc-1",
    "text": "intro " + EVIDENCE + " outro",
    "score": 0.8,
}
WRONG = {
    "accession_number": "acc-9",
    "text": "unrelated risk factor boilerplate text",
    "score": 0.7,
}


def record(qid, qtype="metrics-generated"):
    return {
        "id": qid,
        "question": f"question {qid}",
        "evidence": [{"accession_number": "acc-1", "text": EVIDENCE}],
        "meta": {"question_type": qtype},
    }


def fake_search(results_by_question):
    """Stands in for embed + Qdrant search: question text -> ranked payloads."""
    return lambda question: results_by_question[question]


def test_evaluate_scores_each_question_and_averages():
    records = [record("q1"), record("q2"), record("q3", "novel-generated")]
    search = fake_search(
        {
            "question q1": [RIGHT, WRONG],  # rank 1
            "question q2": [WRONG, WRONG, WRONG, RIGHT],  # rank 4
            "question q3": [WRONG] * 10,  # miss
        }
    )
    rows, summary = evaluate(records, search)

    assert [r["first_relevant_rank"] for r in rows] == [1, 4, None]
    assert summary["all"]["hit@3"] == pytest.approx(1 / 3)
    assert summary["all"]["hit@5"] == pytest.approx(2 / 3)
    assert summary["all"]["mrr@10"] == pytest.approx((1 + 1 / 4 + 0) / 3)
    assert summary["all"]["n"] == 3
    assert summary["novel-generated"]["hit@10"] == 0.0  # breakdown by question type


def test_evaluate_with_generation_scores_answers():
    from sec_financial_rag.generation.prompts import INSUFFICIENT
    from sec_financial_rag.generation.synthesizer import Answer

    records = [record("q1"), record("q2")]
    records[0]["answer"] = "$90,753 million"
    records[1]["answer"] = "$90,753 million"
    search = fake_search({"question q1": [RIGHT], "question q2": [WRONG]})
    replies = {
        "question q1": "Total net sales were $90,753 million [1].",
        "question q2": INSUFFICIENT,
    }

    def answer_fn(question, chunks):
        return Answer(
            replies[question], input_tokens=1000, output_tokens=200, latency_ms=50
        )

    rows, summary = evaluate(records, search, answer_fn=answer_fn)

    assert [r["em_lenient"] for r in rows] == [True, False]
    assert rows[0]["faithfulness"] == 1.0  # 90,753 is in the RIGHT chunk
    assert rows[1]["refused"] and rows[1]["faithfulness"] is None
    s = summary["all"]
    assert s["em_lenient"] == 0.5 and s["refusal_rate"] == 0.5
    assert s["faithfulness"] == 1.0  # only the non-refused answer counts
    assert s["cost_per_1k_usd"] > 0


def test_evaluate_with_parent_expansion_scores_both_levels():
    # The child is too small to contain the evidence; its parent contains all of it.
    child = {
        "accession_number": "acc-1",
        "parent_id": "p1",
        "text": "90,753",
        "score": 0.9,
    }
    parent = {**RIGHT, "parent_id": "p1", "child_text": "90,753"}
    search = fake_search({"question q1": [child, WRONG]})
    rows, summary = evaluate(
        [record("q1")], search, expand=lambda question, children: [parent, WRONG]
    )
    assert rows[0]["hit@3"] == 1.0  # parent level: what the LLM sees
    assert rows[0]["child_hit@3"] == 0.0  # child level: matching alone missed
    assert summary["all"]["child_mrr@10"] == 0.0


def test_evaluate_checkpoints_each_row_and_resumes_without_reasking():
    records = [record("q1"), record("q2")]
    search = fake_search({"question q1": [RIGHT], "question q2": [WRONG]})
    saved = []
    evaluate(records, search, on_row=saved.append)
    assert [r["id"] for r in saved] == ["q1", "q2"]

    # Resume: q1 is done, so only q2 is searched (and would be answered) again.
    asked = []

    def counting_search(question):
        asked.append(question)
        return search(question)

    resumed, summary = evaluate(records, counting_search, done={"q1": saved[0]})
    assert asked == ["question q2"]
    assert [r["id"] for r in resumed] == ["q1", "q2"]
    assert summary["all"]["hit@3"] == 0.5


def test_evaluate_with_route_filters_search_and_charges_its_cost():
    from sec_financial_rag.generation.synthesizer import Answer

    record_ = record("q1")
    record_["answer"] = "$90,753 million"
    seen = {}

    def search(question, where=None, dense_text=None):
        seen[question] = (where, dense_text)
        return [RIGHT]

    def route(rec):
        kwargs = {"where": "only acc-1", "dense_text": "HyDE"}
        return kwargs, {"route_latency_ms": 2000.0, "route_cost_usd": 0.5}

    def answer_fn(question, chunks):
        return Answer("$90,753 million [1].", 0, 0, latency_ms=50)

    rows, summary = evaluate([record_], search, answer_fn=answer_fn, route=route)

    assert seen == {"question q1": ("only acc-1", "HyDE")}  # kwargs reached search
    assert rows[0]["route_latency_ms"] == 2000.0
    s = summary["all"]
    assert s["p95_total_ms"] >= 2050  # routing + generation (+ search)
    assert s["retrieval_p95_ms"] < 2000  # search time only
    assert s["cost_per_1k_usd"] == pytest.approx(500)  # $0.5 routing, $0 answer
