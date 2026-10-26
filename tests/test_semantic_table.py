import numpy as np

from sec_financial_rag.chunking.semantic_table import (
    HEADER_RESERVE,
    SemanticTableChunker,
    units_of,
)

TABLE = "| | 2023 | 2022 |\n|---|---|---|\n| Net sales | $383,285 | $394,328 |"


def test_table_takes_its_title_and_units_line():
    text = f"A long paragraph about results. {'x ' * 120}\n\nIncome Statement\n\n(In millions)\n\n{TABLE}"
    units = units_of(text)
    assert len(units) == 2
    assert units[1].lead == ["Income Statement", "(In millions)"]
    assert units[1].text.endswith("$394,328 |")


def test_long_paragraph_is_not_a_lead_in():
    long_para = "Revenue grew. " * 30  # > 200 chars
    units = units_of(f"{long_para}\n\n{TABLE}")
    assert units[1].lead == []


def test_packs_blocks_up_to_budget_and_keeps_order():
    chunker = SemanticTableChunker(size=HEADER_RESERVE + 20)  # 20-token body budget
    paragraphs = [f"Paragraph {i} has exactly some words." for i in range(6)]
    chunks = chunker.split("\n\n".join(paragraphs))
    assert len(chunks) > 1
    assert all(chunker.tokens(c) <= 20 for c in chunks)
    assert "\n\n".join(chunks) == "\n\n".join(paragraphs)  # nothing lost or reordered


def test_table_never_separated_from_lead_in():
    chunker = SemanticTableChunker(size=HEADER_RESERVE + 30)
    earlier = "Earlier paragraph that is too long to be a lead-in. " * 5
    text = f"{earlier}\n\n(In millions)\n\n{TABLE}"
    chunks = chunker.split(text)
    assert any(c.startswith("(In millions)\n\n| | 2023") for c in chunks)


def test_big_table_split_by_rows_with_header_repeated():
    rows = "\n".join(f"| Row {i} | {i},000 | {i},500 |" for i in range(200))
    table = f"| | 2023 | 2022 |\n|---|---|---|\n{rows}"
    chunker = SemanticTableChunker(table_max=300)
    chunks = chunker.split(f"(In millions)\n\n{table}")
    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.startswith(
            "(In millions)\n\n| | 2023 | 2022 |\n|---|---|---|\n| Row"
        )
        assert chunker.tokens(chunk) <= 300
    assert sum(c.count("| Row ") for c in chunks) == 200  # every row exactly once


def test_over_long_paragraph_cut_into_windows():
    chunker = SemanticTableChunker(size=HEADER_RESERVE + 50)
    text = " ".join(
        ["Adequacy of capital; counterparties and clients"] * 60
    )  # no full stop
    chunks = chunker.split(text)
    assert all(chunker.tokens(c) <= 50 for c in chunks)
    assert " ".join(chunks) == text  # cut at spaces: no word split in half


def test_empty_text():
    assert SemanticTableChunker().split("  \n\n ") == []


def fake_embed(sentences):
    """Direction by topic: rates, the "Business Risks" heading, everything else.
    A small jitter by length keeps neighbouring distances from tying."""

    def direction(s):
        if "rate" in s.lower():
            return [1.0, 0.0, 0.0]
        if s == "Business Risks":
            return [0.0, 1.0, 0.0]
        return [0.0, 0.0, 1.0]

    v = np.array([np.array(direction(s)) + len(s) / 1000 for s in sentences])
    return v / np.linalg.norm(v, axis=1, keepdims=True)


RATES = "Interest rates rose this year. Higher rates raise our funding cost. Rate moves hit margins."
CYBER = "We depend on IT systems. A cyberattack could disrupt operations. Breaches cost money."


def test_narrative_cuts_at_topic_change_even_when_it_fits():
    chunker = SemanticTableChunker(embed=fake_embed, min_topic=0)
    text = f"{RATES} {CYBER}"  # one paragraph, well under the budget
    assert chunker.split(text) == [text]  # structure only: one chunk
    assert chunker.split(text, narrative=True) == [RATES, CYBER]


def test_narrative_keeps_paragraph_breaks_and_tables():
    chunker = SemanticTableChunker(embed=fake_embed, min_topic=0)
    # > 200 chars, so it is a paragraph, not a lead-in line for the table below.
    rates = (
        RATES
        + " Rates may keep rising next year and raise deposit costs and loan losses across all of our regional banking businesses."
    )
    text = f"Rate risk\n\n{rates}\n\n(In millions)\n\n{TABLE}\n\n{CYBER}"
    chunks = chunker.split(text, narrative=True)
    assert chunks == [f"Rate risk\n\n{rates}", f"(In millions)\n\n{TABLE}", CYBER]


def test_small_topic_joins_the_next():
    # "Business Risks" differs from the sentence after it, but a heading belongs
    # with the text it introduces, so the tiny topic merges forward.
    chunker = SemanticTableChunker(embed=fake_embed, percentile=50, min_topic=10)
    text = f"Rate risk\n\n{RATES}\n\nBusiness Risks\n\n{CYBER}"
    assert chunker.split(text, narrative=True) == [
        f"Rate risk\n\n{RATES}",
        f"Business Risks\n\n{CYBER}",
    ]


def test_narrative_without_embedder_falls_back_to_structure():
    chunker = SemanticTableChunker()
    assert chunker.split(f"{RATES} {CYBER}", narrative=True) == [f"{RATES} {CYBER}"]


def test_long_paragraph_packed_by_whole_sentences():
    chunker = SemanticTableChunker(size=HEADER_RESERVE + 40)
    paragraph = " ".join(
        f"Sentence number {i} talks about revenue growth." for i in range(30)
    )
    chunks = chunker.split(paragraph)
    assert len(chunks) > 1
    assert all(c.endswith("growth.") and c.startswith("Sentence") for c in chunks)
    assert " ".join(chunks) == paragraph
