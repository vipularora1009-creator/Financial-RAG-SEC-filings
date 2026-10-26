from sec_financial_rag.chunking.header import context_header, section_label


def test_section_labels():
    assert (
        section_label("Item_7", "Management's Discussion")
        == "Item 7 - Management's Discussion"
    )
    assert (
        section_label("PartII_Item1A", "Risk Factors")
        == "Part II, Item 1A - Risk Factors"
    )
    assert section_label("Preamble", "") == "Cover page and contents"


def test_header_matches_spec_format():
    header = context_header(
        "Apple Inc.",
        "AAPL",
        "10-Q",
        "Q2 FY2022 (Ended Mar 26, 2022)",
        "Part I, Item 1 - Financial Statements",
    )
    assert header == (
        "[COMPANY: Apple Inc. (AAPL) | FILING: 10-Q | PERIOD: Q2 FY2022 (Ended Mar 26, 2022)"
        " | SECTION: Part I, Item 1 - Financial Statements]"
    )
