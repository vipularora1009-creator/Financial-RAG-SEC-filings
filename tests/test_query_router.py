from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from sec_financial_rag.retrieval.query_router import (
    COMPANIES,
    Decomposition,
    QueryMetadata,
    SubQuery,
    build_filter,
    decompose,
    extract_metadata,
    hypothetical_passage,
    interleave,
    search_decomposed,
)


class FakeGemini:
    """Records the request and replies with a fixed parsed object."""

    def __init__(self, parsed: QueryMetadata | None = None, text: str = ""):
        self.parsed, self.text, self.request = parsed, text, None
        self.models = self

    def generate_content(self, model, contents, config):
        self.request = SimpleNamespace(model=model, contents=contents, config=config)
        usage = SimpleNamespace(
            prompt_token_count=400, candidates_token_count=20, thoughts_token_count=5
        )
        return SimpleNamespace(parsed=self.parsed, text=self.text, usage_metadata=usage)


def test_schema_accepts_known_tickers_and_nulls():
    meta = QueryMetadata(ticker="MMM", fiscal_year=2022, quarter=None)
    assert meta.ticker == "MMM" and meta.quarter is None


@pytest.mark.parametrize("bad", [{"ticker": "XYZ"}, {"quarter": "Q5"}])
def test_schema_rejects_values_outside_the_closed_sets(bad):
    with pytest.raises(ValidationError):
        QueryMetadata(**{"ticker": None, "fiscal_year": None, "quarter": None, **bad})


def test_extract_sends_schema_at_temperature_zero_and_counts_tokens():
    reply = QueryMetadata(ticker="ULTA", fiscal_year=2023, quarter=None)
    fake = FakeGemini(reply)
    result = extract_metadata(fake, "Ulta Beauty's acquisitions in FY2023?")
    assert result.metadata == reply
    assert fake.request.config.response_schema is QueryMetadata
    assert fake.request.config.temperature == 0
    assert "Ulta Beauty's acquisitions" in fake.request.contents
    assert result.input_tokens == 400 and result.output_tokens == 25


def test_prompt_lists_every_company():
    fake = FakeGemini(QueryMetadata(ticker=None, fiscal_year=None, quarter=None))
    extract_metadata(fake, "Anything")
    instruction = fake.request.config.system_instruction
    assert all(t in instruction and n in instruction for t, n in COMPANIES.items())


def conditions(where) -> dict:
    return {c.key: c.match for c in where.must}


def test_filter_is_hard_on_ticker_and_soft_on_year():
    where = build_filter(QueryMetadata(ticker="ULTA", fiscal_year=2023, quarter=None))
    match = conditions(where)
    assert match["ticker"].value == "ULTA"
    assert sorted(match["fiscal_year"].any) == [2022, 2023]  # y-1: forecasts, naming
    assert "fiscal_period" not in match


def test_quarter_filters_only_when_asked():
    meta = QueryMetadata(ticker="MMM", fiscal_year=2024, quarter="Q2")
    assert "fiscal_period" not in conditions(build_filter(meta))
    assert conditions(build_filter(meta, quarter=True))["fiscal_period"].value == "Q2"


def test_nothing_extracted_means_no_filter():
    assert (
        build_filter(QueryMetadata(ticker=None, fiscal_year=None, quarter=None)) is None
    )


def test_hypothetical_passage_asks_for_filing_prose_and_returns_it():
    fake = FakeGemini(text="  Net sales increased 8% to $4.2 billion...  ")
    result = hypothetical_passage(fake, "How did 3M's sales change in FY2022?")
    assert result.text == "Net sales increased 8% to $4.2 billion..."
    assert "How did 3M's sales change in FY2022?" in fake.request.contents
    instruction = fake.request.config.system_instruction
    assert "10-K" in instruction and "do not refuse" in instruction.lower()
    assert fake.request.config.temperature == 0
    assert result.input_tokens == 400 and result.output_tokens == 25


def sub(question, ticker=None, year=None):
    return SubQuery(question=question, ticker=ticker, fiscal_year=year)


def test_decomposition_allows_one_to_four_sub_queries():
    assert len(Decomposition(sub_queries=[sub("q")]).sub_queries) == 1
    for bad in ([], [sub("q")] * 5):
        with pytest.raises(ValidationError):
            Decomposition(sub_queries=bad)


def test_decompose_sends_the_schema_and_returns_sub_queries():
    reply = Decomposition(
        sub_queries=[sub("3M revenue FY2022", "MMM", 2022), sub("JNJ revenue", "JNJ")]
    )
    fake = FakeGemini(reply)
    result = decompose(fake, "Compare 3M's and J&J's FY2022 revenue")
    assert result.decomposition == reply
    assert fake.request.config.response_schema is Decomposition
    assert all(t in fake.request.config.system_instruction for t in COMPANIES)


def test_interleave_takes_turns_and_keeps_leftovers():
    assert interleave([["a1", "a2", "a3"], ["b1"]]) == ["a1", "b1", "a2", "a3"]


def test_sub_queries_are_filtered_separately_and_run_in_parallel():
    import asyncio
    import time

    def search(question, where=None):
        time.sleep(0.3)  # blocking, like an HTTP call to Qdrant
        ticker = where.must[0].match.value
        return [{"q": question, "ticker": ticker, "rank": r} for r in (1, 2)]

    subs = [sub("3M revenue", "MMM", 2022), sub("JNJ revenue", "JNJ", 2022)]
    t0 = time.perf_counter()
    hits = asyncio.run(search_decomposed(search, subs))
    elapsed = time.perf_counter() - t0
    assert [(h["ticker"], h["rank"]) for h in hits] == [
        ("MMM", 1),
        ("JNJ", 1),
        ("MMM", 2),
        ("JNJ", 2),
    ]
    assert elapsed < 0.5  # two 0.3 s searches overlapped, not 0.6 s in sequence


def test_expanding_before_interleaving_keeps_parents_balanced():
    import asyncio

    def search(question, where=None):
        ticker = where.must[0].match.value
        if ticker == "MMM":  # three children of one 3M parent, then a second parent
            return [
                {"parent": "m1"},
                {"parent": "m1"},
                {"parent": "m1"},
                {"parent": "m2"},
            ]
        return [{"parent": "j1"}, {"parent": "j2"}, {"parent": "j3"}]

    def expand(children):  # dedupe to parents, first-seen order
        return [{"parent": p} for p in dict.fromkeys(c["parent"] for c in children)]

    subs = [sub("3M", "MMM"), sub("JNJ", "JNJ")]
    parents = asyncio.run(search_decomposed(search, subs, expand=expand))
    assert [p["parent"] for p in parents][:4] == ["m1", "j1", "m2", "j2"]
