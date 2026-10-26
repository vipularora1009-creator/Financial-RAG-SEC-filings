"""Answer synthesis with the Gemini API (spec §4.6). One call per question.

The model is pinned and the temperature is 0, so every experiment uses the same,
as-repeatable-as-possible generator and score changes come from retrieval.
"""

import os
import time
from dataclasses import dataclass

from google import genai
from google.genai import types

from sec_financial_rag.generation.prompts import SYSTEM_PROMPT, build_prompt

MODEL = "gemini-3.8-flash"  # pinned; changing it is its own experiment
# USD per 1M tokens for MODEL (ai.google.dev/gemini-api/docs/pricing, checked
# 2026-10-24). These double on 2027-01-01: re-check before quoting costs.
PRICE_IN, PRICE_OUT = 0.75, 3.75


@dataclass
class Answer:
    text: str
    input_tokens: int
    output_tokens: int  # visible answer + thinking tokens (both billed as output)
    latency_ms: float


def cost_usd(answer: Answer) -> float:
    return (answer.input_tokens * PRICE_IN + answer.output_tokens * PRICE_OUT) / 1e6


def token_counts(response) -> tuple[int, int]:
    """(input, output) tokens; thinking is billed as output, so it counts there."""
    usage = response.usage_metadata
    output = (usage.candidates_token_count or 0) + (usage.thoughts_token_count or 0)
    return usage.prompt_token_count or 0, output


def make_client() -> genai.Client:
    """Gemini client using the key from .env (never printed or logged)."""
    return genai.Client(api_key=os.environ["GEMINI_API_KEY"])


def synthesize(client: genai.Client, question: str, chunks: list[dict]) -> Answer:
    """Answer `question` from the retrieved `chunks` only, with [n] citations."""
    t0 = time.perf_counter()
    response = client.models.generate_content(
        model=MODEL,
        contents=build_prompt(question, chunks),
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0,
        ),
    )
    latency_ms = (time.perf_counter() - t0) * 1000
    input_tokens, output_tokens = token_counts(response)
    return Answer(
        text=(response.text or "").strip(),  # .text is None if the reply was empty
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency_ms=latency_ms,
    )
