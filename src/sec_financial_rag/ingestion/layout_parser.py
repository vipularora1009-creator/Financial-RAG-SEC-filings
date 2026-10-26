"""Layout-aware parsing for Exp-2 (spec §4.1): Docling reads the HTML, we compact its tables.

Docling's own Markdown repeats a spanning header ("Three Months Ended") in every
grid column it covers and keeps spacer and "$" columns, which bloats a chunk's
token budget. Here each table becomes one row per line, one column per period.
"""

import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from docling_core.types.doc import (
    DoclingDocument,
    GroupLabel,
    NodeItem,
    TableItem,
)

# "Item 7. Management's Discussion..." / "ITEM 1A: RISK FACTORS" / "Item 1 C ."
# The title may be empty: then it is in the next block, or this is a contents line.
ITEM_HEADING = re.compile(
    r"item\s*(\d{1,2}(?:\s?[a-c])?)\s*[.:\-–—]?\s*(.*)", re.IGNORECASE
)
PART_HEADING = re.compile(r"part\s+(iv|i{1,3})\b", re.IGNORECASE)
MAX_HEADING_CHARS = 150  # longer = a paragraph that merely starts with "Item 3."
PAGE_NUMBER = re.compile(r"(?:page\s+)?[A-Z]?-?\d{1,3}", re.IGNORECASE)  # "29", "F-12"
# A table cell holding one number: "$77,457", "(9)", "12.5%", "1.0".
NUMBER_CELL = re.compile(r"\$?\(?-?\d[\d,]*(?:\.\d+)?\)?%?")
# A bare year is a period header, not a value: SEC amounts >= 1,000 carry commas.
YEAR_CELL = re.compile(r"(?:19|20)\d\d")


@dataclass
class Cell:
    text: str
    start: int  # first grid column the cell covers
    span: int  # number of grid columns it covers

    @property
    def end(self) -> int:
        return self.start + self.span


def row_cells(grid_row) -> list[Cell]:
    """Distinct non-empty cells of one Docling grid row, left to right.

    Docling repeats a spanning cell in every position it covers, so keep the
    first copy only. "$" and "%" / ")" cells are glued to their number.
    """
    cells: list[Cell] = []
    seen: set[int] = set()
    for c in grid_row:
        text = " ".join(c.text.split())
        if not text or c.start_col_offset_idx in seen:
            continue
        seen.add(c.start_col_offset_idx)
        cell = Cell(text, c.start_col_offset_idx, c.col_span)
        if cells and cells[-1].text == "$":  # "$" | "77,457" -> "$77,457"
            cells.pop()  # keep the NUMBER's position: rows without "$" align with it
            cell = Cell("$" + text, cell.start, cell.span)
        elif cells and text in ("%", ")", ")%"):  # "12.5" | "%" -> "12.5%"
            prev = cells.pop()
            cell = Cell(prev.text + text, prev.start, cell.end - prev.start)
        cells.append(cell)
    return cells


def _is_value(text: str) -> bool:
    """A number that can be a table value (not a bare year like "2023")."""
    t = text.replace(" ", "")
    return bool(NUMBER_CELL.fullmatch(t)) and not YEAR_CELL.fullmatch(t)


def _header_count(rows: list[list[Cell]]) -> int:
    """Header rows = rows above the first body row (a label in column 0 and a value),
    so "(In millions)" or "June 30, | 2023 | 2022" no longer end the header. A table
    without values falls back to: leading rows with nothing in column 0."""
    for i, cells in enumerate(rows):
        if cells[0].start == 0 and any(_is_value(c.text) for c in cells[1:]):
            # A label-only row right above it ("Revenues") opens the body, not the header.
            while i > 0 and len(rows[i - 1]) == 1 and rows[i - 1][0].start == 0:
                i -= 1
            return i
    n = 0
    while n < len(rows) and rows[n][0].start > 0:
        n += 1
    return n


def _merge_columns(names: list[str], values: list[list[str]]) -> tuple[list, list]:
    """Merge neighbouring columns with the same name that no row uses both of
    (a header spanning a "$" column and a number column becomes one column)."""
    groups: list[list[int]] = []
    for j, name in enumerate(names):
        prev = groups[-1] if groups else None
        if (
            prev
            and names[prev[0]] == name
            and not any(row[j] and any(row[k] for k in prev) for row in values)
        ):
            prev.append(j)
        else:
            groups.append([j])
    merged = [["".join(row[k] for k in g) for g in groups] for row in values]
    return [names[g[0]] for g in groups], merged


def table_to_markdown(grid) -> str:
    """Compact Markdown for one table grid (list of rows of Docling TableCells).

    Header rows sit above the first body row. Each value column is named by every
    header cell spanning it ("Three Months Ended March 26, 2022"); header text in
    column 0 ("(In millions)") names the label column.
    """
    rows = [cells for cells in (row_cells(r) for r in grid) if cells]
    if not rows:
        return ""  # layout-only table with no text
    n_header = _header_count(rows)
    header_rows, body = rows[:n_header], rows[n_header:]
    label_header = " ".join(hr[0].text for hr in header_rows if hr[0].start == 0)
    headers = [h for hr in header_rows for h in hr if h.start > 0]

    # Columns start where values start; a header cell covering no value keeps
    # its own column, so no text is lost.
    starts = {c.start for row in body for c in row if c.start > 0}
    starts |= {
        h.start for h in headers if not any(h.start <= s < h.end for s in starts)
    }
    starts = sorted(starts)
    names = [" ".join(h.text for h in headers if h.start <= s < h.end) for s in starts]

    values = []
    labels = []
    for cells in body:
        label = cells[0].text if cells[0].start == 0 else ""
        row = [""] * len(starts)
        for cell in cells[1:] if label else cells:
            row[starts.index(cell.start)] = cell.text
        labels.append(label)
        values.append(row)
    names, values = _merge_columns(names, values)

    lines = [
        (f"| {label_header} | " if label_header else "| | ") + " | ".join(names) + " |",
        "|---" * (len(names) + 1) + "|",
    ]
    lines += [
        f"| {label} | " + " | ".join(row) + " |" for label, row in zip(labels, values)
    ]
    return "\n".join(lines)


@dataclass
class Section:
    section_id: str  # "Item_7" (10-K) or "PartII_Item1A" (10-Q)
    title: str
    blocks: list[str] = field(default_factory=list)  # Markdown, reading order

    @property
    def text(self) -> str:
        return "\n\n".join(self.blocks)


def item_heading(block: str) -> tuple[str, str] | None:
    """(item number, title) if the block is an Item heading, else None."""
    if len(block) > MAX_HEADING_CHARS:
        return None
    m = ITEM_HEADING.fullmatch(" ".join(block.split()))
    return (m[1].replace(" ", "").upper(), m[2]) if m else None


def _title_after(blocks: list[str], i: int) -> str:
    """Title for a bare "Item 1" at blocks[i]: the next block, if it reads like one.

    In a table of contents the next block is another Item, a Part or a page
    number, so this returns "" and the line is not treated as a heading.
    """
    nxt = blocks[i + 1].strip() if i + 1 < len(blocks) else ""
    is_title = (
        len(nxt) <= MAX_HEADING_CHARS
        and re.search(r"[a-z]", nxt, re.IGNORECASE)
        and not nxt.startswith("|")  # a table
        and not item_heading(nxt)
        and not PART_HEADING.match(nxt)
    )
    return nxt if is_title else ""


def section_id(item: str, part: str, form_type: str) -> str:
    """10-Q items restart in Part II, so their id carries the part."""
    if form_type.startswith("10-Q"):
        return f"Part{part}_Item{item}"
    return f"Item_{item}"


def split_items(blocks: list[str], form_type: str) -> list[Section]:
    """Group a filing's Markdown blocks into SEC Item sections, in order.

    Text before the first heading goes to "Preamble" (cover page, contents).
    A repeated heading continues its earlier section, so nothing is dropped.
    """
    sections = {"Preamble": Section("Preamble", "")}
    current = sections["Preamble"]
    part = "I"
    for i, block in enumerate(blocks):
        if len(block) <= 60 and (m := PART_HEADING.match(block.strip())):
            part = m[1].upper()
        elif heading := item_heading(block):
            item, title = heading
            title = title or _title_after(blocks, i)
            if title:
                sid = section_id(item, part, form_type)
                current = sections.setdefault(sid, Section(sid, title))
        current.blocks.append(block)
    return [s for s in sections.values() if s.blocks]


def table_block(grid) -> str:
    """A data table as compact Markdown; a layout table (one row of text,
    e.g. Amazon's "Item 1. | Financial Statements" headings) as a text line."""
    rows = [cells for cells in (row_cells(r) for r in grid) if cells]
    if len(rows) <= 1:
        return " ".join(c.text for c in rows[0]) if rows else ""
    return table_to_markdown(grid)


def _inside_table(item: NodeItem, doc: DoclingDocument) -> bool:
    """Docling also lists every table cell as a text item (table > group > text)."""
    while item.parent is not None:
        if item.parent.cref.startswith("#/tables/"):
            return True
        item = item.parent.resolve(doc)
    return False


def document_blocks(doc: DoclingDocument) -> list[str]:
    """The document as Markdown blocks in reading order, each table once."""
    blocks = []
    open_inline = None  # the inline group the last block came from
    for item, _level in doc.iterate_items():
        if isinstance(item, TableItem):
            blocks.append(table_block(item.data.grid))
            open_inline = None
        elif not _inside_table(item, doc):
            text = " ".join(getattr(item, "text", "").split())
            group = _inline_group(item, doc)
            if group is not None and group == open_inline:
                # Docling splits a paragraph at each link; its pieces share an
                # "inline" group: "...in Part II, Item 8" + "of this Form 10-K."
                blocks[-1] = f"{blocks[-1]} {text}".strip()
                continue
            blocks.append(text)
            open_inline = group
    return [b for b in blocks if b]


def _inline_group(item: NodeItem, doc: DoclingDocument) -> str | None:
    """Reference of the item's parent if that parent is an inline group."""
    if item.parent is None:
        return None
    parent = item.parent.resolve(doc)
    return (
        parent.self_ref if getattr(parent, "label", None) == GroupLabel.INLINE else None
    )


def parse_filing(path: Path, form_type: str, converter=None) -> list[Section]:
    """Filing HTML -> SEC Item sections of Markdown text and tables.

    Pass one DocumentConverter when parsing many filings; building it is slow.
    """
    if converter is None:
        from docling.document_converter import DocumentConverter

        converter = DocumentConverter()
    doc = converter.convert(path).document
    return split_items(drop_page_furniture(document_blocks(doc)), form_type)


def drop_page_furniture(blocks: list[str]) -> list[str]:
    """Remove page headers/footers, which split sentences and topics apart.

    Page numbers, "Table of Contents" links, and short lines that repeat with a
    changing number ("Apple Inc. | 2025 Form 10-K | 7"): a page counter. A line
    repeating with the same number ("Year Ended December 31, 2023") is kept.
    """

    def shape(block: str) -> str:
        return re.sub(r"\d+", "#", block)

    variants = defaultdict(set)
    for b in blocks:
        if len(b) <= 100 and not b.startswith("|") and re.search(r"\d", b):
            variants[shape(b)].add(b)

    def is_furniture(b: str) -> bool:
        if b.startswith("|") or item_heading(b):
            return False
        letters = re.sub(r"[\d#\s]", "", shape(b))
        return bool(
            PAGE_NUMBER.fullmatch(b)
            or b.replace(" ", "").lower() == "tableofcontents"
            or (len(variants.get(shape(b), ())) >= 5 and len(letters) >= 12)
        )

    return [b for b in blocks if not is_furniture(b)]
