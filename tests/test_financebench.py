from sec_financial_rag.evals.financebench import evidence_score, tokens

PDF_EVIDENCE = (
    "Revenue\n$\n 46,298 \n \n$\n 51,761 \n \n$\n 47,262 \nCost of sales\n 36,386"
)


def test_tokens_ignore_html_layout_and_entities():
    assert tokens("<td>Revenue</td><td>$&nbsp;46,298</td>") == ["revenue", "46", "298"]


def test_same_table_scores_high_despite_different_layout():
    html_doc = (
        "<tr><td>Revenue</td><td>$</td><td>46,298</td><td>$</td><td>51,761</td>"
        "<td>$</td><td>47,262</td></tr><tr><td>Cost of sales</td><td>36,386</td></tr>"
    )
    assert evidence_score(PDF_EVIDENCE, html_doc) == 1.0


def test_next_years_filing_scores_low_even_with_shared_numbers():
    # The following 10-K repeats 46,298 and 51,761, but in a shifted column order.
    next_year = "<tr><td>Revenue</td><td>43,452</td><td>46,298</td><td>51,761</td></tr>"
    assert evidence_score(PDF_EVIDENCE, next_year) < 0.5


def test_to_eval_record_uses_our_accession_and_schema():
    from sec_financial_rag.evals.financebench import to_eval_record

    question = {
        "financebench_id": "financebench_id_00685",
        "doc_name": "BESTBUY_2023_10K",
        "question_type": "domain-relevant",
        "question_reasoning": "Logical reasoning",
        "question": "Are Best Buy's gross margins historically consistent?",
        "answer": "Yes, a minor decline of 1.1%.",
        "justification": "9912/46298",
        "evidence": [
            {
                "evidence_text": "Consolidated Statements of Earnings",
                "evidence_page_num": 39,
            }
        ],
    }
    filing = {
        "ticker": "BBY",
        "accession_number": "0000764478-23-000006",
        "form": "10-K",
        "report_date": "2023-01-28",
        "status": "verified",
    }
    record = to_eval_record(question, filing)

    assert record["id"] == "fb-00685"
    assert record["source"] == "financebench"
    assert record["category"] == "single-filing"
    assert record["filings"] == [
        {
            "ticker": "BBY",
            "accession_number": "0000764478-23-000006",
            "form": "10-K",
            "report_date": "2023-01-28",
        }
    ]
    assert record["evidence"] == [
        {
            "accession_number": "0000764478-23-000006",
            "text": "Consolidated Statements of Earnings",
        }
    ]
    assert record["meta"]["pdf_page"] == 39
