"""ADR-001 evidence: how much table structure each parser keeps.

Usage: uv run python scripts/table_fidelity.py unstructured_dump.json
(the dump comes from scripts/dump_unstructured_tables.py; its keys are the filings)

Ground truth is read from the source HTML, independent of both parsers: every
<table> becomes a grid with colspan expanded, and each number becomes a
(row label, column header, value) triple. A parser "keeps" a triple when its
output has the same value in a row with the same label, under a column whose
header contains the true header cell (e.g. "December 31, 2022").

Reported per output: triple recall (can a value still be tied to its line item
AND its period?), figure recall (is the number there at all?) and table tokens.
Outputs: our compact serializer (layout_parser.table_to_markdown, as indexed in
Exp-2/3), Docling's own Markdown export, Unstructured's table HTML and text.
"""

import json
import re
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import tiktoken
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)  # iXBRL is XHTML

NUMBER = re.compile(r"\(?\$?\s*\d[\d,]*(?:\.\d+)?\s*\)?%?")

Triple = tuple[str, str, str]  # (label, header, value), all normalized


def norm(text: str) -> str:
    return " ".join(text.split()).lower().rstrip(":").strip()


def value_key(text: str) -> str | None:
    """'$ 24,977' -> '24977'; '( 317 )' -> '-317'; not a number -> None."""
    t = text.replace("$", "").replace(" ", "")
    if not re.fullmatch(r"\(?\d[\d,]*(?:\.\d+)?\)?%?", t):
        return None
    digits = re.sub(r"[^\d.]", "", t)
    return f"-{digits}" if t.startswith("(") else digits


def html_grid(table, expand_colspan: bool = True) -> list[list[str]]:
    rows = []
    for tr in table.find_all("tr"):
        row = []
        for cell in tr.find_all(["td", "th"]):
            text = " ".join(cell.get_text(" ").split())
            span = int(cell.get("colspan", 1) or 1) if expand_colspan else 1
            row += [text] * span
        rows.append(row)
    return rows


def is_year(text: str) -> bool:
    """A bare year ("2022") is a period header: SEC amounts >= 1,000 carry commas."""
    return bool(re.fullmatch(r"(?:19|20)\d\d", text.strip()))


def triples(grid: list[list[str]]) -> set[Triple]:
    """Header rows = rows above the first row with a label and a value (bare years
    don't count). A value's header = the header cell in exactly its column (lowest
    one wins)."""
    first_body = next(
        (
            i
            for i, row in enumerate(grid)
            if row
            and row[0].strip()
            and any(value_key(c) and not is_year(c) for c in row[1:])
        ),
        None,
    )
    if first_body is None:
        return set()
    headers: dict[int, str] = {}
    for row in grid[:first_body]:
        for j, cell in enumerate(row):
            if cell.strip() and j > 0:
                headers[j] = norm(cell)
    out = set()
    for row in grid[first_body:]:
        if not row or not row[0].strip():
            continue
        for j, cell in enumerate(row[1:], start=1):
            v = value_key(cell)
            if v and j in headers:
                out.add((norm(row[0]), headers[j], v))
    return out


def markdown_grid(md: str) -> list[list[str]]:
    lines = [ln for ln in md.splitlines() if ln.startswith("|")]
    rows = [[c.strip() for c in ln.strip("|").split("|")] for ln in lines]
    return [r for r in rows if not all(set(c) <= set("-: ") for c in r)]


def kept(truth: set[Triple], found: set[Triple]) -> int:
    """Triples whose label and value match and whose header contains the true header."""
    by_row: dict[tuple[str, str], list[str]] = defaultdict(list)
    for label, header, value in found:
        by_row[(label, value)].append(header)
    return sum(
        any(h in fh for fh in by_row.get((label, value), []))
        for label, h, value in truth
    )


def figure_recall(truth: set[Triple], text: str) -> float:
    present = {value_key(m.group()) for m in NUMBER.finditer(text)}
    values = {value for _, _, value in truth}
    return len(values & present) / len(values) if values else 1.0


def main(dump_path: str) -> None:
    from docling.document_converter import DocumentConverter

    from sec_financial_rag.ingestion.layout_parser import table_to_markdown

    converter = DocumentConverter()
    encoding = tiktoken.get_encoding("cl100k_base")
    dump = json.loads(Path(dump_path).read_text())
    names = ("ours", "docling_native", "unstructured_html", "unstructured_text")
    kept_total: dict[str, int] = defaultdict(int)
    tokens: dict[str, int] = defaultdict(int)
    n_truth = 0
    print(f"{'filing':22} {'triples':>7}  " + "  ".join(f"{n:>17}" for n in names))
    for path, unstr in dump.items():
        soup = BeautifulSoup(Path(path).read_text(), "lxml")
        truth = set().union(*(triples(html_grid(t)) for t in soup.find_all("table")))
        n_truth += len(truth)
        doc = converter.convert(path).document
        texts = {
            "ours": [table_to_markdown(t.data.grid) for t in doc.tables],
            "docling_native": [t.export_to_markdown(doc=doc) for t in doc.tables],
            "unstructured_html": [t["html"] for t in unstr["tables"]],
            "unstructured_text": [t["text"] for t in unstr["tables"]],
        }
        grids = {
            "ours": [markdown_grid(md) for md in texts["ours"]],
            "docling_native": [markdown_grid(md) for md in texts["docling_native"]],
            "unstructured_html": [
                html_grid(BeautifulSoup(h, "lxml"), expand_colspan=False)
                for h in texts["unstructured_html"]
            ],
            "unstructured_text": [],  # flattened text: no rows or columns left
        }
        cells = []
        for name in names:
            k = kept(truth, set().union(set(), *(triples(g) for g in grids[name])))
            kept_total[name] += k
            tokens[name] += sum(len(encoding.encode(t)) for t in texts[name])
            fig = figure_recall(truth, "\n".join(texts[name]))
            cells.append(f"{k / len(truth):7.1%} (fig {fig:4.0%})")
        print(f"{Path(path).name:22} {len(truth):7}  " + "  ".join(cells))
    for name in names:
        print(
            f"TOTAL {name:18} triples kept {kept_total[name]:5}/{n_truth}"
            f" = {kept_total[name] / n_truth:5.1%}   table tokens {tokens[name]:,}"
        )


if __name__ == "__main__":
    main(sys.argv[1])
