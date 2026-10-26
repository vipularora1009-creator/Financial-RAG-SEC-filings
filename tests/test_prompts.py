from sec_financial_rag.generation.prompts import (
    INSUFFICIENT,
    SYSTEM_PROMPT,
    build_prompt,
)

CHUNKS = [
    {
        "ticker": "AAPL",
        "form_type": "10-Q",
        "report_date": "2022-06-25",
        "text": "Total net sales 82,959",
    },
    {
        "ticker": "AAPL",
        "form_type": "10-K",
        "report_date": "2022-09-24",
        "text": "Net income 99,803",
    },
]


def test_prompt_numbers_and_labels_each_chunk_in_rank_order():
    prompt = build_prompt("What were Apple's net sales?", CHUNKS)
    first = prompt.index("[1] AAPL 10-Q, period ended 2022-06-25")
    second = prompt.index("[2] AAPL 10-K, period ended 2022-09-24")
    assert first < second  # retrieval order is kept
    assert "Total net sales 82,959" in prompt
    assert prompt.rstrip().endswith("Question: What were Apple's net sales?")


def test_system_prompt_states_the_rules():
    assert INSUFFICIENT in SYSTEM_PROMPT  # exact refusal string the code can detect
    assert "[n]" in SYSTEM_PROMPT  # citation format


def test_system_prompt_asks_for_detectable_arithmetic():
    # numeric_faithfulness treats a number right after "=" as computed (5f.4).
    assert "= 1.9%" in SYSTEM_PROMPT
