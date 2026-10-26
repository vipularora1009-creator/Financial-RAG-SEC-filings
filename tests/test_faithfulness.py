from sec_financial_rag.evals.faithfulness import numeric_faithfulness
from sec_financial_rag.generation.prompts import INSUFFICIENT

CONTEXT = [{"text": "Total net sales | 82,959 | 81,434 | Net income | 19,442"}]


def test_numbers_found_in_context_are_grounded():
    # The chunk lost "(in millions)", so the context number has no unit: still grounded.
    assert numeric_faithfulness("Net sales were $82,959 million [1].", CONTEXT) == 1.0


def test_invented_number_is_not_grounded():
    answer = "Net sales were $82,959 million [1], up from $75,000 million."
    assert numeric_faithfulness(answer, CONTEXT) == 0.5


def test_result_of_shown_arithmetic_counts_as_grounded():
    answer = "Growth: (82,959 - 81,434) / 81,434 = 1.9% [1]."
    assert numeric_faithfulness(answer, CONTEXT) == 1.0


def test_computed_number_without_shown_arithmetic_is_not_grounded():
    assert numeric_faithfulness("Net sales grew 1.9% [1].", CONTEXT) == 0.0


def test_bold_result_of_shown_arithmetic_counts_as_grounded():
    answer = "Growth: (82,959 - 81,434) / 81,434 = **1.9%** [1]."
    assert numeric_faithfulness(answer, CONTEXT) == 1.0


def test_headline_repeating_a_shown_result_is_grounded():
    answer = (
        "Net sales grew **1.9%**. Calculation: (82,959 - 81,434) / 81,434 = 1.9% [1]."
    )
    assert numeric_faithfulness(answer, CONTEXT) == 1.0


def test_headline_differing_from_the_shown_result_is_not_grounded():
    answer = "Net sales grew 2.5%. Calculation: (82,959 - 81,434) / 81,434 = 1.9% [1]."
    assert (
        numeric_faithfulness(answer, CONTEXT) == 4 / 5
    )  # 5 numbers; only 2.5% is ungrounded


def test_numbers_restated_from_the_question_are_grounded():
    question = "Is the quick ratio above 1.0?"
    context = [{"text": "Quick ratio 0.96"}]
    assert numeric_faithfulness("No: 0.96 [1], below 1.0.", context, question) == 1.0


def test_not_applicable_for_refusals_and_answers_without_numbers():
    assert numeric_faithfulness(INSUFFICIENT, CONTEXT) is None
    assert numeric_faithfulness("Data Center [1]", CONTEXT) is None
