from types import SimpleNamespace

from sec_financial_rag.ingestion.layout_parser import (
    row_cells,
    split_items,
    table_to_markdown,
)


def cell(text, start, span=1):
    """Fake Docling TableCell: only the fields the serializer reads."""
    return SimpleNamespace(text=text, start_col_offset_idx=start, col_span=span)


def spanning(text, start, span):
    """Docling repeats a spanning cell in every grid position it covers."""
    return [cell(text, start, span)] * span


def test_dollar_and_closing_paren_glued_to_their_number():
    row = [
        cell("Products", 0),
        cell("$", 1),
        cell("77,457", 2),
        cell("(9", 3),
        cell(")", 4),
    ]
    assert [c.text for c in row_cells(row)] == ["Products", "$77,457", "(9)"]


def test_spanning_cell_kept_once():
    row = [cell("", 0), *spanning("Three Months Ended", 1, 4)]
    assert [c.text for c in row_cells(row)] == ["Three Months Ended"]


def test_period_headers_stack_into_column_names():
    grid = [
        [cell("", 0), *spanning("Three Months Ended", 1, 4)],
        [
            cell("", 0),
            *spanning("March 26, 2022", 1, 2),
            *spanning("March 27, 2021", 3, 2),
        ],
        [
            cell("Products", 0),
            cell("$", 1),
            cell("77,457", 2),
            cell("$", 3),
            cell("72,683", 4),
        ],
    ]
    assert table_to_markdown(grid).splitlines() == [
        "| | Three Months Ended March 26, 2022 | Three Months Ended March 27, 2021 |",
        "|---|---|---|",
        "| Products | $77,457 | $72,683 |",
    ]


def test_full_width_units_row_does_not_drop_columns():
    # Regression: PayPal facilities table lost "Other Countries" and "Total".
    grid = [
        [cell("", 0), cell("US", 1), cell("Other", 2)],
        [cell("", 0), *spanning("(In millions)", 1, 2)],
        [cell("Owned", 0), cell("1.0", 1), cell("0.1", 2)],
    ]
    lines = table_to_markdown(grid).splitlines()
    assert lines[0] == "| | US (In millions) | Other (In millions) |"
    assert lines[2] == "| Owned | 1.0 | 0.1 |"


def test_value_without_header_still_kept():
    grid = [
        [cell("", 0), cell("", 1), cell("Page", 2)],
        [cell("Item 1.", 0), cell("Business", 1), cell("4", 2)],
    ]
    assert table_to_markdown(grid).splitlines()[-1] == "| Item 1. | Business | 4 |"


def test_empty_layout_table_gives_empty_string():
    assert table_to_markdown([[cell("", 0), cell(" ", 1)]]) == ""


def ids(sections):
    return [s.section_id for s in sections]


def test_contents_lines_skipped_real_headings_split():
    blocks = [
        "Apple Inc. Form 10-K",
        "Item 1.",  # table of contents: bare, followed by another Item
        "Item 7.",
        "Item 1. Business",
        "We design smartphones.",
        "Item 7. Management's Discussion and Analysis",
        "| | 2022 |\n|---|---|\n| Net sales | $394,328 |",
    ]
    sections = split_items(blocks, "10-K")
    assert ids(sections) == ["Preamble", "Item_1", "Item_7"]
    assert sections[2].title == "Management's Discussion and Analysis"
    assert sections[2].blocks[-1].startswith("| | 2022")


def test_10q_items_restart_in_part_ii():
    blocks = [
        "PART I - FINANCIAL INFORMATION",
        "Item 1. Financial Statements",
        "PART II - OTHER INFORMATION",
        "Item 1. Legal Proceedings",
        "ITEM 1A. RISK FACTORS",
    ]
    assert ids(split_items(blocks, "10-Q"))[1:] == [
        "PartI_Item1",
        "PartII_Item1",
        "PartII_Item1A",
    ]


def test_bare_item_takes_title_from_next_block():
    # AMD 10-Q: "Item 2" and its title are separate blocks.
    sections = split_items(
        ["Item 2", "Management's Discussion", "Revenue grew."], "10-Q"
    )
    assert ids(sections) == ["PartI_Item2"]
    assert sections[0].title == "Management's Discussion"


def test_spaced_item_letter_and_long_paragraph():
    long_para = "Item 3. Quantitative disclosures. " + "Rates rose. " * 20
    sections = split_items(["Item 1 C . Cybersecurity", long_para], "10-K")
    assert ids(sections) == ["Item_1C"]  # the long paragraph stays inside Item 1C
    assert sections[0].blocks == ["Item 1 C . Cybersecurity", long_para]


def test_repeated_heading_continues_its_section():
    blocks = [
        "Item 7. MD&A",
        "Part one.",
        "Item 8. Statements",
        "Item 7. MD&A",
        "Part two.",
    ]
    sections = split_items(blocks, "10-K")
    assert ids(sections) == ["Item_7", "Item_8"]
    assert sections[0].blocks[-1] == "Part two."


MINI_10Q = """<html><body>
<p>Example Corp Form 10-Q</p>
<table>
  <tr><td>Item 1.</td><td>Financial Statements</td><td>3</td></tr>
  <tr><td>Item 1A.</td><td>Risk Factors</td><td>9</td></tr>
</table>
<p>PART I - FINANCIAL INFORMATION</p>
<table><tr><td></td></tr><tr><td>Item 1.</td><td>Financial Statements</td></tr></table>
<p>(In millions)</p>
<table>
  <tr><td></td><td colspan="2">Three Months Ended</td></tr>
  <tr><td></td><td>June 30, 2024</td><td>June 30, 2023</td></tr>
  <tr><td>Net sales</td><td>$ 1,234</td><td>$ 1,100</td></tr>
</table>
<p>PART II - OTHER INFORMATION</p>
<p>Item 1A. Risk Factors</p>
<p>No material changes.</p>
</body></html>"""


def test_parse_filing_end_to_end(tmp_path):
    from sec_financial_rag.ingestion.layout_parser import parse_filing

    path = tmp_path / "mini.htm"
    path.write_text(MINI_10Q)
    sections = {s.section_id: s for s in parse_filing(path, "10-Q")}

    assert list(sections) == ["Preamble", "PartI_Item1", "PartII_Item1A"]
    item1 = sections["PartI_Item1"].text
    assert "(In millions)" in item1
    assert (
        "| Three Months Ended June 30, 2024 | Three Months Ended June 30, 2023 |"
        in item1
    )
    assert "| Net sales | $ 1,234 | $ 1,100 |" in item1
    assert item1.count("1,234") == 1  # table cells not repeated as text items
    assert sections["PartII_Item1A"].text.endswith("No material changes.")


def test_paragraph_with_link_stays_one_block(tmp_path):
    # Docling splits at <a>; the pieces share an "inline" group and are rejoined.
    from docling.document_converter import DocumentConverter

    from sec_financial_rag.ingestion.layout_parser import document_blocks

    path = tmp_path / "link.htm"
    path.write_text(
        '<html><body><p>See <a href="#x">Part II, Item 8</a> of this Form 10-K.</p>'
        "<p>Next paragraph.</p></body></html>"
    )
    doc = DocumentConverter().convert(path).document
    assert document_blocks(doc) == [
        "See Part II, Item 8 of this Form 10-K.",
        "Next paragraph.",
    ]


def test_page_furniture_dropped_real_lines_kept():
    from sec_financial_rag.ingestion.layout_parser import drop_page_furniture

    footers = [f"Apple Inc. | 2025 Form 10-K | {page}" for page in range(5, 11)]
    keep = [
        "Year Ended December 31, 2023",  # repeats with the same number: a real lead-in
        "Year Ended December 31, 2023",
        "Year Ended December 31, 2023",
        "(In millions)",
        "Note 3",  # short label, too few letters to be a footer
        "Note 4",
        "Note 5",
        "Note 6",
        "Note 7",
        "Item 2",
        "| | 2023 |\n|---|---|\n| Sales | 5 |",
    ]
    junk = ["29", "F-12", "Table of Contents", "Table of Conten t s", *footers]
    assert drop_page_furniture(junk + keep) == keep


def test_dollar_rows_align_with_plain_rows_and_units_header_in_column_0():
    # Regression (MSFT intangibles): "$" rows and plain rows landed in different
    # columns, and "(In millions)" in column 0 hid the header rows.
    grid = [
        [cell("(In millions)", 0), *spanning("Gross Carrying Amount", 1, 2)],
        [cell("June 30,", 0), *spanning("2023", 1, 2)],
        [cell("Technology-based", 0), cell("$", 1), cell("11,245", 2)],
        [cell("Customer-related", 0), cell("7,281", 2)],
    ]
    assert table_to_markdown(grid).splitlines() == [
        "| (In millions) June 30, | Gross Carrying Amount 2023 |",
        "|---|---|",
        "| Technology-based | $11,245 |",
        "| Customer-related | 7,281 |",
    ]


def test_label_only_row_above_first_value_row_stays_in_body():
    grid = [
        [cell("", 0), cell("2022", 1)],
        [cell("Revenues", 0)],
        [cell("Interest income", 0), cell("23", 1)],
    ]
    assert table_to_markdown(grid).splitlines() == [
        "| | 2022 |",
        "|---|---|",
        "| Revenues |  |",
        "| Interest income | 23 |",
    ]


def test_same_name_columns_merge_only_when_no_row_uses_both():
    # Values at two positions under one spanning header (seen in real filings).
    merged = [
        [cell("", 0), *spanning("2022", 1, 2)],
        [cell("Revenue", 0), cell("10", 1)],
        [cell("Cost", 0), cell("4", 2)],
    ]
    assert table_to_markdown(merged).splitlines() == [
        "| | 2022 |",
        "|---|---|",
        "| Revenue | 10 |",
        "| Cost | 4 |",
    ]
    kept_apart = [
        [cell("", 0), *spanning("Change", 1, 2)],
        [cell("Revenue", 0), cell("$10", 1), cell("25%", 2)],
    ]
    assert table_to_markdown(kept_apart).splitlines()[0] == "| | Change | Change |"
