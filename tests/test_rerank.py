from sec_financial_rag.retrieval.rerank import rerank


def keyword_scorer(question, passages):
    """Fake cross-encoder: 1.0 if the passage mentions 'iPhone'."""
    return [1.0 if "iPhone" in p else 0.1 for p in passages]


def test_rerank_resorts_by_score_and_keeps_first_stage_score():
    parents = [
        {"text": "Services revenue grew.", "header": "[AAPL]", "score": 0.9},
        {"text": "iPhone net sales were $50,570.", "header": "[AAPL]", "score": 0.5},
    ]
    out = rerank("iPhone sales?", parents, keyword_scorer)
    assert [p["text"][:6] for p in out] == ["iPhone", "Servic"]
    assert out[0]["score"] == 1.0 and out[0]["retrieval_score"] == 0.5


def test_only_the_first_candidates_are_scored():
    seen = []

    def spy(question, passages):
        seen.extend(passages)
        return [0.0] * len(passages)

    parents = [{"text": str(i), "header": "[H]", "score": 0} for i in range(30)]
    assert len(rerank("q", parents, spy, candidates=25)) == 25
    assert len(seen) == 25 and seen[0] == "[H]\n0"  # header + text
