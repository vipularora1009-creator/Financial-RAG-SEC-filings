from sec_financial_rag.ingestion.naive_parser import html_to_text

HTML = """<html><body>
<div style="display:none"><ix:header><ix:hidden>dei:EntityCentralIndexKey 0000320193</ix:hidden></ix:header></div>
<script>var x = 1;</script>
<p>Total   net
   sales increased.</p>
<table><tr><td>Revenue</td><td>$</td><td>391,035</td></tr></table>
</body></html>"""


def test_keeps_visible_text_and_drops_hidden_metadata():
    text = html_to_text(HTML)
    assert "Total net sales increased." in text  # whitespace collapsed within a line
    assert "EntityCentralIndexKey" not in text  # hidden inline-XBRL header
    assert "var x" not in text  # scripts


def test_tables_are_flattened_one_cell_per_line():
    # Naive baseline on purpose: the number loses its row label (Exp-1's failure mode).
    lines = html_to_text(HTML).splitlines()
    assert lines[-3:] == ["Revenue", "$", "391,035"]
