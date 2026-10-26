"""Contextual header injection (spec §4.1): a fixed label in front of every chunk.

A table row like "| Research and development | $8,675 |" says nothing about the
company, period or statement. The embedding model only sees text, so the label
puts those facts into the text it embeds. Built from our metadata, no LLM call.
"""

import re


def section_label(section_id: str, title: str) -> str:
    """ "PartII_Item1A", "Risk Factors" -> "Part II, Item 1A - Risk Factors"."""
    if section_id == "Preamble":
        return "Cover page and contents"
    m = re.fullmatch(r"(?:Part(\w+)_)?Item_?(\w+)", section_id)
    if not m:
        return section_id
    part, item = m.groups()
    item_name = f"Part {part}, Item {item}" if part else f"Item {item}"
    return f"{item_name} - {title}" if title else item_name


def context_header(
    company: str, ticker: str, form_type: str, period: str, section: str
) -> str:
    """[COMPANY: Apple Inc. (AAPL) | FILING: 10-Q | PERIOD: Q2 FY2022 (Ended Mar 26, 2022) | SECTION: ...]"""
    return (
        f"[COMPANY: {company} ({ticker}) | FILING: {form_type} | "
        f"PERIOD: {period} | SECTION: {section}]"
    )
