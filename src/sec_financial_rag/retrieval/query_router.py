"""Query router (spec §4.5), Exp-5 step 2: metadata extraction.

Gemini reads the question and fills `QueryMetadata` through structured output:
the schema goes in the request and the reply is constrained to match it. The
LLM extracts what the question says, literally; the filtering policy (which
years to allow) is plain code, so it stays deterministic and testable.
"""

import asyncio
import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from itertools import zip_longest
from pathlib import Path
from typing import Literal

from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue

from sec_financial_rag.generation.synthesizer import MODEL, token_counts

# Cached extractions per eval question id (git-ignored): written by
# scripts/router_accuracy.py, reused by the Exp-5 eval runs, so runs differ only
# in retrieval and generation, not in extraction.
EXTRACTIONS = Path("data/cache/exp5_extractions.jsonl")
HYDE = Path("data/cache/exp5_hyde.jsonl")  # scripts/hyde_passages.py

# The 30 companies in the index (ticker -> company name as in the filings).
COMPANIES = {
    "AAPL": "Apple Inc.",
    "ADBE": "ADOBE INC.",
    "AES": "THE AES CORPORATION",
    "AMCR": "AMCOR PLC",
    "AMD": "ADVANCED MICRO DEVICES, INC",
    "AMZN": "AMAZON.COM, INC.",
    "AWK": "AMERICAN WATER WORKS COMPANY, INC.",
    "AXP": "American Express Co",
    "BA": "BOEING CO",
    "BBY": "BEST BUY CO., INC.",
    "CVS": "CVS HEALTH CORPORATION",
    "F": "Ford Motor Co",
    "GIS": "GENERAL MILLS, INC.",
    "GLW": "CORNING INC /NY",
    "GOOGL": "Alphabet Inc.",
    "JNJ": "JOHNSON & JOHNSON",
    "JPM": "JPMorgan Chase & Co",
    "KO": "COCA COLA CO",
    "LMT": "LOCKHEED MARTIN CORPORATION",
    "MGM": "MGM Resorts International",
    "MMM": "3M COMPANY",
    "MSFT": "MICROSOFT CORPORATION",
    "NKE": "NIKE, Inc.",
    "NVDA": "NVIDIA CORP",
    "PEP": "PepsiCo, Inc.",
    "PFE": "PFIZER INC",
    "PYPL": "PayPal Holdings, Inc.",
    "TSLA": "Tesla, Inc.",
    "ULTA": "ULTA BEAUTY, INC.",
    "VZ": "Verizon Communications Inc.",
}

# A closed set: the model can only answer with a ticker that exists in the index.
Ticker = StrEnum("Ticker", {t: t for t in COMPANIES})


class QueryMetadata(BaseModel):
    ticker: Ticker | None = Field(
        None, description="The company the question is about, or null if none is named."
    )
    fiscal_year: int | None = Field(
        None,
        description="The year the question names, as written ('FY2023' -> 2023),"
        " or null if none.",
    )
    quarter: Literal["Q1", "Q2", "Q3", "Q4"] | None = Field(
        None, description="The fiscal quarter the question names, or null if none."
    )


COMPANY_LIST = "Companies (ticker: name):\n" + "\n".join(
    f"{t}: {name}" for t, name in COMPANIES.items()
)
SYSTEM = (
    "Extract search filters from a question about SEC filings. Copy what the"
    " question states; do not infer a year or quarter it does not mention.\n"
    + COMPANY_LIST
)


@dataclass
class Extraction:
    metadata: QueryMetadata
    input_tokens: int  # same fields as synthesizer.Answer, so cost_usd() works
    output_tokens: int
    latency_ms: float


def extract_metadata(client: genai.Client, question: str) -> Extraction:
    t0 = time.perf_counter()
    response = client.models.generate_content(
        model=MODEL,
        contents=question,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM,
            temperature=0,
            response_mime_type="application/json",
            response_schema=QueryMetadata,
            # A lookup, not reasoning: low thinking keeps it fast and cheap (this model
            # rejects MINIMAL with a 400).
            thinking_config=types.ThinkingConfig(
                thinking_level=types.ThinkingLevel.LOW
            ),
        ),
    )
    return Extraction(
        response.parsed, *token_counts(response), (time.perf_counter() - t0) * 1000
    )


HYDE_SYSTEM = (
    "Write the passage of a 10-K or 10-Q filing that answers the question, as it"
    " would appear in the filing: formal SEC disclosure language, about 120 words,"
    " with the line items, section wording and figures such a passage would"
    " contain. Plausible figures are fine. Do not refuse, explain or mention that"
    " the passage is hypothetical."
)


@dataclass
class Hypothetical:
    text: str
    input_tokens: int
    output_tokens: int
    latency_ms: float


def hypothetical_passage(client: genai.Client, question: str) -> Hypothetical:
    """HyDE: a fake filing passage to embed instead of the question.

    Used for dense search only: its vocabulary and structure match real filing
    text, but its figures are invented, which would mislead exact-token BM25.
    """
    t0 = time.perf_counter()
    response = client.models.generate_content(
        model=MODEL,
        contents=question,
        config=types.GenerateContentConfig(
            system_instruction=HYDE_SYSTEM,
            temperature=0,
            thinking_config=types.ThinkingConfig(
                thinking_level=types.ThinkingLevel.LOW
            ),
        ),
    )
    return Hypothetical(
        (response.text or "").strip(),
        *token_counts(response),
        (time.perf_counter() - t0) * 1000,
    )


def build_filter(meta: QueryMetadata, quarter: bool = False) -> Filter | None:
    """Filtering policy: hard on company, soft on year, quarter optional.

    The year also allows y-1: a question about FY2023 can be answered by the
    FY2022 filing (Boeing's forecast for 2023; Ulta's fiscal 2022 ends in Jan
    2023). Returns None when nothing was extracted: search everything.
    """
    must = []
    if meta.ticker:
        must.append(FieldCondition(key="ticker", match=MatchValue(value=meta.ticker)))
    if meta.fiscal_year:
        years = [meta.fiscal_year - 1, meta.fiscal_year]
        must.append(FieldCondition(key="fiscal_year", match=MatchAny(any=years)))
    if quarter and meta.quarter:
        must.append(
            FieldCondition(key="fiscal_period", match=MatchValue(value=meta.quarter))
        )
    return Filter(must=must) if must else None


def load_cache(path: Path) -> dict[str, dict]:
    """id -> cached row ({"metadata" or "text", "latency_ms", "cost_usd"})."""
    if not path.exists():
        return {}
    return {r["id"]: r for r in map(json.loads, path.open())}


class SubQuery(QueryMetadata):
    """One self-contained search: its own text and its own filter fields."""

    question: str = Field(description="A self-contained search question.")


class Decomposition(BaseModel):
    sub_queries: list[SubQuery] = Field(min_length=1, max_length=4)


DECOMPOSE_SYSTEM = (
    "Split a question about SEC filings into self-contained search questions,"
    " one per company and fiscal year whose filing is needed (at most 4). A"
    " question about one company and period becomes one search question: the"
    " question itself. For each, fill the company and the year it states; do"
    " not infer a year it does not mention.\n" + COMPANY_LIST
)


@dataclass
class Decomposed:
    decomposition: Decomposition
    input_tokens: int
    output_tokens: int
    latency_ms: float


def decompose(client: genai.Client, question: str) -> Decomposed:
    t0 = time.perf_counter()
    response = client.models.generate_content(
        model=MODEL,
        contents=question,
        config=types.GenerateContentConfig(
            system_instruction=DECOMPOSE_SYSTEM,
            temperature=0,
            response_mime_type="application/json",
            response_schema=Decomposition,
            thinking_config=types.ThinkingConfig(
                thinking_level=types.ThinkingLevel.LOW
            ),
        ),
    )
    return Decomposed(
        response.parsed, *token_counts(response), (time.perf_counter() - t0) * 1000
    )


def interleave(lists: list[list]) -> list:
    """Round-robin merge: a1, b1, a2, b2, ... so every sub-query reaches the top.

    Scores from differently filtered searches are not comparable, so sorting by
    score could let one company fill every slot.
    """
    gap = object()  # marks a list that has run out
    return [
        x for row in zip_longest(*lists, fillvalue=gap) for x in row if x is not gap
    ]


async def search_decomposed(
    search: Callable[..., list[dict]],
    sub_queries: list[SubQuery],
    expand: Callable[[list[dict]], list[dict]] | None = None,
) -> list[dict]:
    """Search every sub-query with its own filter, in parallel, then interleave.

    `search` is blocking (HTTP to Qdrant, embedding), so each call runs in a
    worker thread; gather() waits for all of them: total time ~ the slowest one.
    `expand` (children -> parents) runs per sub-query before interleaving:
    interleaving children first would let deduplication take slots from a
    sub-query whose children share a parent.
    """
    lists = await asyncio.gather(
        *(
            asyncio.to_thread(search, q.question, where=build_filter(q))
            for q in sub_queries
        )
    )
    return interleave([expand(hits) for hits in lists] if expand else lists)
