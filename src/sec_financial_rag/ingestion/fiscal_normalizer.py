"""Fiscal periods (spec §4.4): "Q1 FY2023" for NVIDIA means the quarter ended May 1, 2022.

Read from the filing's own inline-XBRL cover tags (dei = Document and Entity
Information), so no per-company fiscal calendar table is needed. Regex rather
than an HTML parser: the tags are machine-generated with a fixed shape, and a
second full parse of a 10 MB filing would cost about a second.
"""

import re
from dataclasses import dataclass
from datetime import date
from html import unescape

DEI_TAG = r'<ix:nonNumeric[^>]*name="dei:{}"[^>]*>(.*?)</ix:nonNumeric>'


@dataclass(frozen=True)
class FiscalPeriod:
    year: int  # fiscal year, e.g. 2023
    period: str  # "FY" (annual) or "Q1".."Q3"


def dei_value(html: str, name: str) -> str | None:
    """Text of one dei cover tag, e.g. dei_value(html, "DocumentFiscalYearFocus")."""
    m = re.search(DEI_TAG.format(name), html, re.DOTALL | re.IGNORECASE)
    if not m:
        return None
    text = unescape(re.sub(r"<[^>]+>", "", m[1]))  # "JOHNSON &amp; JOHNSON"
    return " ".join(text.split())  # also turns &#160; (no-break space) into " "


def fiscal_period(html: str) -> FiscalPeriod | None:
    """The filing's fiscal year and period as the company tagged them."""
    year = dei_value(html, "DocumentFiscalYearFocus")
    period = dei_value(html, "DocumentFiscalPeriodFocus")
    if not (year and period):
        return None
    return FiscalPeriod(int(year), period.upper())


def period_label(fp: FiscalPeriod | None, report_date: date) -> str:
    """Header text: "Q2 FY2022 (Ended Mar 26, 2022)". The end date pins the
    fiscal label to the calendar; without tags, only the date is given."""
    ended = f"Ended {report_date:%b} {report_date.day}, {report_date.year}"
    if fp is None:
        return ended
    name = f"FY{fp.year}" if fp.period == "FY" else f"{fp.period} FY{fp.year}"
    return f"{name} ({ended})"
