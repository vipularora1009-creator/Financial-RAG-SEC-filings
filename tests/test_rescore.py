from sec_financial_rag.evals.rescore import rescore


def test_rescore_recomputes_faithfulness_from_saved_context():
    row = {
        "question": "What was the growth?",
        "answer": "Growth = **1.9%**",
        "context": ["Net sales 82,959"],
        "faithfulness": 0.0,  # scored by the old metric
        "em_lenient": True,
    }
    (new,) = rescore([row])
    assert new["faithfulness"] == 1.0
    assert new["em_lenient"] is True  # other fields untouched
    assert row["faithfulness"] == 0.0  # input not mutated
