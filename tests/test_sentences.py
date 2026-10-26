import numpy as np

from sec_financial_rag.chunking.sentences import breakpoints, segments, split_sentences


def test_abbreviations_and_decimals_do_not_end_sentences():
    text = (
        "Apple Inc. reported $1.2 billion in U.S. Treasury gains. "
        "Rates rose in Jan. 2023 per J. Powell. Was it material? Yes."
    )
    assert split_sentences(text) == [
        "Apple Inc. reported $1.2 billion in U.S. Treasury gains.",
        "Rates rose in Jan. 2023 per J. Powell.",
        "Was it material?",
        "Yes.",
    ]


def test_cross_references_do_not_end_sentences():
    text = "See Part II, Item 7. of this Form 10-K and Note 5. Debt for details. Next."
    assert split_sentences(text) == [
        "See Part II, Item 7. of this Form 10-K and Note 5. Debt for details.",
        "Next.",
    ]


def test_joining_sentences_restores_text():
    text = "First one.  Second\none.\n\nThird (3) one."
    assert " ".join(split_sentences(text)) == "First one. Second one. Third (3) one."


def unit(*xs):
    v = np.array(xs, dtype=float)
    return v / np.linalg.norm(v)


def test_breakpoint_at_biggest_topic_change():
    rates_a, rates_b = unit(1, 0.1, 0), unit(1, 0.2, 0)
    cyber_a, cyber_b = unit(0, 0.1, 1), unit(0, 0.2, 1)
    vectors = np.stack([rates_a, rates_b, cyber_a, cyber_b])
    assert breakpoints(vectors) == [2]  # cut before the first cyber sentence


def test_no_breakpoints_when_too_short_or_uniform():
    assert breakpoints(np.stack([unit(1, 0), unit(0, 1)])) == []
    assert breakpoints(np.stack([unit(1, 0)] * 5)) == []  # all distances equal


def test_segments():
    assert segments(["a", "b", "c", "d"], [2]) == [["a", "b"], ["c", "d"]]
    assert segments(["a", "b"], []) == [["a", "b"]]
