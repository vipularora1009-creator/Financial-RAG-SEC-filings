import pytest

from sec_financial_rag.evals.numeric import extract_numbers, numeric_match


def values(text):
    return [n.value for n in extract_numbers(text)]


def test_extracts_values_with_units_and_signs():
    assert values("$82.96 billion") == [82.96e9]
    assert values("negative working capital of -$1561M") == [-1561e6]
    assert values("Best Buy ($1.8 bn)") == [1.8e9]
    assert values("declined from 36.8% to 34.6%") == [36.8, 34.6]
    assert values("an expense of (1,234) million") == [-1234e6]
    assert values("$9068.00") == [9068.0]


def test_ignores_years_and_labels():
    # Years, fiscal labels and "1x" are not answer values.
    assert values("from 24.6% in FY 2021 to 21.6% in FY2022") == [24.6, 21.6]
    assert values("0.96 by Jun'23 close, to touch the 1x mark in Q2") == [0.96]
    # Found by calibrating on real gold answers (fb-01107, fb-00669):
    assert values("legal areas: (1) pricing (2) opioids, $4.3 billion") == [4.3e9]
    assert values("COVID-19 vaccine costs of $1.1 billion") == [1.1e9]


@pytest.mark.parametrize(
    "gold, ours, lenient, strict",
    [
        ("$82.96 billion", "net sales were $82,959 million [1]", True, True),
        ("$82.96 billion", "net sales were $82,959 [1]", True, False),  # unit lost
        (
            "$14,205M",
            "about $14.2B",
            True,
            True,
        ),  # spec example: coarser precision wins
        ("from 36.8% in FY 2021 to 34.6%", "36.8% then 34.6% [2]", True, True),
        (
            "from 36.8% in FY 2021 to 34.6%",
            "it fell to 34.6% [2]",
            False,
            False,
        ),  # all needed
        ("$1.8 bn", "$1,824 million", True, True),  # rounds to 1.8 bn
        ("$1.8 bn", "$2.1 billion", False, False),
        (
            "-$1561M",
            "working capital was negative $1,561 million",
            True,
            True,
        ),  # sign in words
    ],
)
def test_numeric_match(gold, ours, lenient, strict):
    assert numeric_match(gold, ours) == (lenient, strict)


def test_not_applicable_when_gold_has_no_number():
    assert numeric_match("Data Center", "Data Center [1]") == (None, None)
