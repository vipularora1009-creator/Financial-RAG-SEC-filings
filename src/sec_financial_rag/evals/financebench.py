"""Map FinanceBench documents to SEC filings (Week 1 step 3, Option A).

FinanceBench's year labels don't follow one rule (e.g. Ulta's fiscal 2022 is
labelled "2023"), so candidate filings are verified by content: does the
question's evidence text actually appear in the downloaded document?
"""

import html
import json
import os
import re
from pathlib import Path

import httpx

from sec_financial_rag.ingestion.edgar_downloader import (
    download_filing,
    fetch_all_filings,
    filing_url,
    make_client,
)

TAG = re.compile(r"<[^>]+>")
NON_CONTENT = re.compile(r"<(script|style)\b.*?</\1>", re.IGNORECASE | re.DOTALL)
TOKEN = re.compile(r"[a-z0-9]+")


def tokens(text: str) -> list[str]:
    """Lowercase letter/digit runs from HTML or plain text; layout, tags and `$` disappear."""
    text = NON_CONTENT.sub(" ", text)
    text = html.unescape(TAG.sub(" ", text))
    return TOKEN.findall(text.lower())


def shingles(toks: list[str], n: int = 5) -> set[tuple[str, ...]]:
    """All runs of n consecutive tokens (word n-grams)."""
    return {tuple(toks[i : i + n]) for i in range(len(toks) - n + 1)}


def evidence_score(evidence: str, document: str, n: int = 5) -> float:
    """Fraction of the evidence's n-token shingles that also occur in the document."""
    wanted = shingles(tokens(evidence), n)
    if not wanted:
        return 0.0
    return len(wanted & shingles(tokens(document), n)) / len(wanted)


# --- Resolving FinanceBench documents to SEC filings -------------------------

FB_URL = "https://raw.githubusercontent.com/patronus-ai/financebench/main/data/{name}"
FB_DIR = Path("data/golden_eval/financebench")  # cached copy (git-ignored: no license)
CANDIDATE_DIR = Path("data/golden_eval/candidates")  # never mixed into the corpus
MAPPING_PATH = Path("data/golden_eval/financebench_docs.json")  # committed result
TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
FORMS = {"10k": "10-K", "10q": "10-Q"}
MIN_SCORE, MIN_MARGIN = 0.8, 0.2
# Human-reviewed decisions for docs the matcher flags (kept in code so re-runs reproduce them).
MANUAL_OVERRIDES = {
    # Only evidence is the boilerplate company overview (scores 0.70-0.95 everywhere).
    # Top score is the FY2022 10-K; JPM's fiscal year = calendar year; answer is year-independent.
    "JPMORGAN_2022_10K": "0000019617-23-000231",
}

# Explicit on purpose: fuzzy name matching could silently pick the wrong company.
FINANCEBENCH_TICKERS = {
    "3M": "MMM", "AES Corporation": "AES", "AMD": "AMD", "Adobe": "ADBE",
    "Amcor": "AMCR", "American Express": "AXP", "American Water Works": "AWK",
    "Best Buy": "BBY", "Boeing": "BA", "CVS Health": "CVS", "Coca-Cola": "KO",
    "Corning": "GLW", "General Mills": "GIS", "Johnson & Johnson": "JNJ",
    "JPMorgan": "JPM", "Lockheed Martin": "LMT", "MGM Resorts": "MGM",
    "Microsoft": "MSFT", "Nike": "NKE", "Paypal": "PYPL", "PepsiCo": "PEP",
    "Pfizer": "PFE", "Ulta Beauty": "ULTA", "Verizon": "VZ",
}  # fmt: skip


def load_financebench(client: httpx.Client) -> tuple[list[dict], dict[str, dict]]:
    """Questions and document info, downloaded once into FB_DIR."""
    loaded = []
    for name in (
        "financebench_open_source.jsonl",
        "financebench_document_information.jsonl",
    ):
        path = FB_DIR / name
        download_filing(client, FB_URL.format(name=name), path)  # skips if cached
        loaded.append([json.loads(line) for line in path.read_text().splitlines()])
    questions, doc_info = loaded
    return questions, {d["doc_name"]: d for d in doc_info}


def target_docs(
    questions: list[dict], doc_info: dict[str, dict]
) -> dict[str, list[str]]:
    """FY2022+ 10-K/10-Q docs → all their evidence passages (Option A scope)."""
    targets: dict[str, list[str]] = {}
    for q in questions:
        info = doc_info[q["doc_name"]]
        if int(info["doc_period"]) >= 2022 and info["doc_type"] in FORMS:
            targets.setdefault(q["doc_name"], []).extend(
                e["evidence_text"] for e in q["evidence"]
            )
    return targets


def candidates(client: httpx.Client, cik: int, form: str, year: int) -> list[dict]:
    """Filings of `form` whose period ends within about a year of the labelled year."""
    rows = fetch_all_filings(client, cik, since=f"{year - 1}-01-01")
    return [
        r for r in rows if r["form"] == form and r["report_date"] <= f"{year + 1}-06-30"
    ]


def resolve_doc(
    client: httpx.Client, ticker: str, cik: int, rows: list[dict], evidence: list[str]
) -> dict:
    """Score each candidate by mean evidence score; accept only a clear winner."""
    scored = []
    for row in rows:
        path = (
            CANDIDATE_DIR / ticker / row["accession_number"] / row["primary_document"]
        )
        download_filing(client, filing_url(cik, row), path)
        document = path.read_text(errors="ignore")
        mean = sum(evidence_score(e, document) for e in evidence) / len(evidence)
        scored.append((round(mean, 3), row))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    best, row = scored[0] if scored else (0.0, {})
    runner_up = scored[1][0] if len(scored) > 1 else 0.0
    ok = best >= MIN_SCORE and best - runner_up >= MIN_MARGIN
    return {**row, "ticker": ticker, "cik": cik, "score": best,
            "runner_up_score": runner_up, "status": "verified" if ok else "review"}  # fmt: skip


def to_eval_record(question: dict, filing: dict) -> dict:
    """One FinanceBench question → our eval schema (same shape as custom questions)."""
    accession = filing["accession_number"]
    return {
        "id": "fb-" + question["financebench_id"].rsplit("_", 1)[-1],
        "source": "financebench",
        "question": question["question"],
        "answer": question["answer"],
        "category": "single-filing",
        "filings": [
            {
                k: filing[k]
                for k in ("ticker", "accession_number", "form", "report_date")
            }
        ],
        "evidence": [
            {"accession_number": accession, "text": e["evidence_text"]}
            for e in question["evidence"]
        ],
        "meta": {
            "financebench_id": question["financebench_id"],
            "question_type": question["question_type"],
            "question_reasoning": question["question_reasoning"],
            "justification": question["justification"],
            "pdf_page": question["evidence"][0][
                "evidence_page_num"
            ],  # their PDF, zero-indexed
        },
    }


def main() -> None:
    with make_client(os.environ["SEC_USER_AGENT"]) as client:
        questions, doc_info = load_financebench(client)
        targets = target_docs(questions, doc_info)
        ciks = {
            r["ticker"]: r["cik_str"] for r in client.get(TICKERS_URL).json().values()
        }
        mapping = {}
        for doc_name, evidence in sorted(targets.items()):
            info = doc_info[doc_name]
            ticker = FINANCEBENCH_TICKERS[info["company"]]
            form, year = FORMS[info["doc_type"]], int(info["doc_period"])
            rows = candidates(client, ciks[ticker], form, year)
            result = resolve_doc(client, ticker, ciks[ticker], rows, evidence)
            if MANUAL_OVERRIDES.get(doc_name) == result.get("accession_number"):
                result["status"] = "verified-manual"
            mapping[doc_name] = result
            print(f"{doc_name:28} {result.get('accession_number', '-'):22} "
                  f"best {result['score']:.2f}  next {result['runner_up_score']:.2f}  "
                  f"{len(rows)} cand.  {result['status']}")  # fmt: skip
    MAPPING_PATH.write_text(json.dumps(mapping, indent=2) + "\n")
    n_ok = sum(m["status"].startswith("verified") for m in mapping.values())
    print(f"\n{n_ok}/{len(mapping)} verified → {MAPPING_PATH}")


if __name__ == "__main__":
    main()
