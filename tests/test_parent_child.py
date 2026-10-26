from collections import Counter

from sec_financial_rag.chunking.parent_child import ParentChildChunker
from sec_financial_rag.chunking.semantic_table import HEADER_RESERVE

HEADER = "| | 2023 | 2022 |\n|---|---|---|"
ROWS = [f"| Segment {i} revenue | ${i},100 | ${i},050 |" for i in range(40)]
TABLE = HEADER + "\n" + "\n".join(ROWS)


def small_chunker() -> ParentChildChunker:
    # Body budgets: parents 200 tokens (tables up to 800), units 80, children 60.
    return ParentChildChunker(
        parent_size=HEADER_RESERVE + 200,
        parent_table_max=800,
        child_size=HEADER_RESERVE + 60,
        unit_size=HEADER_RESERVE + 80,
    )


def test_children_cover_their_parent_without_losing_words():
    chunker = small_chunker()
    paragraphs = [f"Paragraph {i} explains liquidity and cash flow." for i in range(30)]
    parents = chunker.split("\n\n".join(paragraphs))
    assert len(parents) > 1
    for parent in parents:
        assert len(parent.children) > 1
        children_words = Counter(" ".join(parent.children).split())
        assert children_words == Counter(parent.text.split())  # exact cover, no overlap


def test_table_is_one_parent_and_children_repeat_its_header():
    chunker = small_chunker()
    parents = chunker.split(f"(In millions)\n\n{TABLE}")
    assert len(parents) == 1  # 660 tokens <= parent table_max 800
    assert "Segment 39" in parents[0].text
    children = parents[0].children
    assert len(children) > 1
    assert all(c.startswith(f"(In millions)\n\n{HEADER}") for c in children)
    assert all(chunker.children.tokens(c) <= 60 + 10 for c in children)  # ~ budget


def test_children_stay_within_budget():
    chunker = small_chunker()
    text = "\n\n".join(f"Risk {i} could harm our results. " * 3 for i in range(20))
    for parent in chunker.split(text):
        assert chunker.parents.tokens(parent.text) <= 200
        assert all(chunker.children.tokens(c) <= 60 for c in parent.children)


def test_small_table_shares_a_parent_with_its_surrounding_text():
    chunker = small_chunker()
    small_table = HEADER + "\n" + "\n".join(ROWS[:3])
    text = f"Revenue rose in every segment.\n\n{small_table}\n\n(1) Excludes hedging gains."
    parents = chunker.split(text)
    assert len(parents) == 1
    assert parents[0].text.endswith("(1) Excludes hedging gains.")
