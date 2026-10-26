"""Naive HTML -> plain text for the Exp-1 baseline (spec §5.2: "PyPDF" row).

Our filings are HTML, not PDF, so this plays PyPDF's role: keep only the text a
reader would see, with no structure. Tables come out one cell per line, so numbers
get separated from their row and column labels. That's the failure mode Exp-2's
layout-aware parser is meant to fix.
"""

import re
import warnings

from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

# Inline-XBRL filings start with <?xml ...?> but are XHTML: parsing as HTML is intended.
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

HIDDEN = re.compile(r"display\s*:\s*none", re.IGNORECASE)


def html_to_text(html: str) -> str:
    """Visible text of an HTML filing: one text node per line, whitespace collapsed."""
    soup = BeautifulSoup(html, "lxml")
    # Not visible to a reader: scripts, styles, the inline-XBRL metadata header,
    # and anything hidden with CSS.
    for tag in soup(["script", "style", "ix:header"]):
        tag.decompose()
    for tag in soup.find_all(style=HIDDEN):
        tag.decompose()
    # Line breaks inside the HTML source aren't visible, so collapse them per text node.
    return "\n".join(" ".join(s.split()) for s in soup.stripped_strings)
