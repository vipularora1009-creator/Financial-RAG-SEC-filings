from sec_financial_rag.retrieval.bm25 import Bm25Encoder, normalize_numbers


def test_figures_stay_one_token():
    assert normalize_numbers("$14,205 and 1,234,567") == "$14205 and 1234567"
    assert normalize_numbers("margin 16.52%") == "margin 16_52%"
    assert normalize_numbers("Item 7. Results, 2023") == "Item 7. Results, 2023"


def test_tokens_keep_numbers_and_drop_stop_words():
    tokens = Bm25Encoder().tokens(
        "Net sales were $14,205 million (16.52%) under ASC 606."
    )
    assert tokens == ["net", "sale", "14205", "million", "16_52", "asc", "606"]


def test_query_and_document_share_token_ids():
    enc = Bm25Encoder()
    doc = enc.documents(["iPhone net sales were $14,205 million"])[0]
    query = enc.query("iPhone sales 14,205")
    assert set(query.indices) <= set(doc.indices)
    assert all(
        v == 1.0 for v in query.values
    )  # Qdrant's IDF modifier weights the query
