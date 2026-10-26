from datetime import date

from sec_financial_rag.ingestion.fiscal_normalizer import (
    FiscalPeriod,
    fiscal_period,
    period_label,
)

# Shape of a real iXBRL cover (NVIDIA 10-Q for the quarter ended May 1, 2022).
NVDA_COVER = """<ix:header><ix:hidden>
<ix:nonNumeric contextRef="c-1" name="dei:DocumentFiscalYearFocus" id="f-4">2023</ix:nonNumeric>
<ix:nonNumeric contextRef="c-1" name="dei:DocumentFiscalPeriodFocus" id="f-5">Q1</ix:nonNumeric>
</ix:hidden></ix:header>"""


def test_reads_shifted_fiscal_year_from_tags():
    assert fiscal_period(NVDA_COVER) == FiscalPeriod(2023, "Q1")


def test_nested_markup_inside_tag_is_stripped():
    html = '<ix:nonNumeric name="dei:DocumentFiscalYearFocus"><span>2022</span></ix:nonNumeric>'
    html += '<ix:nonNumeric name="dei:DocumentFiscalPeriodFocus">fy</ix:nonNumeric>'
    assert fiscal_period(html) == FiscalPeriod(2022, "FY")


def test_missing_tags_give_none():
    assert fiscal_period("<html>no xbrl</html>") is None


def test_labels():
    assert period_label(FiscalPeriod(2023, "Q1"), date(2022, 5, 1)) == (
        "Q1 FY2023 (Ended May 1, 2022)"
    )
    assert period_label(FiscalPeriod(2022, "FY"), date(2022, 9, 24)) == (
        "FY2022 (Ended Sep 24, 2022)"
    )
    assert period_label(None, date(2022, 9, 24)) == "Ended Sep 24, 2022"


def test_entities_and_no_break_spaces_decoded():
    from sec_financial_rag.ingestion.fiscal_normalizer import dei_value

    html = '<ix:nonNumeric name="dei:EntityRegistrantName">BEST&#160;BUY CO., INC. &amp; Co</ix:nonNumeric>'
    assert dei_value(html, "EntityRegistrantName") == "BEST BUY CO., INC. & Co"
