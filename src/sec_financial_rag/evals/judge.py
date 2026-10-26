"""Answer correctness by an LLM judge (replaces numeric EM as the headline metric).

EM checks that the gold answer's numbers appear in the answer. That fails on
gold numbers that are not the answer (an intermediate "~13%"), on rounding
(142.7% vs 143%), and on yes/no answers; and it cannot score non-numeric
questions. The judge compares meaning against a written rubric instead.

The judge sees the question, the gold answer and the generated answer, not the
retrieved context: it scores correctness; faithfulness is scored separately.
It is the same pinned model as the generator (self-preference bias is the
known risk), so it must be validated against human labels before it is trusted.

Status (2026-10-26): built and smoke-tested only; NOT used by the runner. The
project owner chose to keep EM as the answer metric for Phase 1 (ADR-004).
"""

import time
from dataclasses import dataclass
from typing import Literal

from google import genai
from google.genai import types
from pydantic import BaseModel, Field

from sec_financial_rag.generation.synthesizer import MODEL, token_counts

VERDICTS = ("correct", "partially_correct", "incorrect", "refused")


class Verdict(BaseModel):
    # Reason first: the model writes its comparison before it commits to a verdict.
    reason: str = Field(
        description="One or two sentences comparing the answer with the gold answer."
    )
    verdict: Literal["correct", "partially_correct", "incorrect", "refused"]


JUDGE_SYSTEM = """You grade answers to questions about SEC filings against a gold answer.

- correct: the answer's conclusion matches the gold (yes/no, direction, the item
  asked for) and every key figure the gold gives as the answer matches, allowing
  rounding (142.7% matches 143%; $1,561 million matches $1.56 billion) and
  equivalent units or sign conventions ((0.6)% means -0.6%).
- partially_correct: the conclusion matches but a key figure is missing or wrong,
  or only part of a multi-part question is answered correctly.
- incorrect: the conclusion or the key figures contradict the gold.
- refused: the answer declines or says the data is insufficient.

Extra correct detail is not penalised. Figures the gold only uses along the way
(an intermediate step or a rounded difference) are not required if the
requested figures are right. Judge only against the gold; do not use your own
knowledge of the company."""


@dataclass
class Judgement:
    verdict: Verdict
    input_tokens: int  # same fields as synthesizer.Answer, so cost_usd() works
    output_tokens: int
    latency_ms: float


def judge(client: genai.Client, question: str, gold: str, answer: str) -> Judgement:
    t0 = time.perf_counter()
    response = client.models.generate_content(
        model=MODEL,
        contents=f"Question:\n{question}\n\nGold answer:\n{gold}\n\nAnswer:\n{answer}",
        config=types.GenerateContentConfig(
            system_instruction=JUDGE_SYSTEM,
            temperature=0,
            response_mime_type="application/json",
            response_schema=Verdict,
            thinking_config=types.ThinkingConfig(
                thinking_level=types.ThinkingLevel.LOW
            ),
        ),
    )
    return Judgement(
        response.parsed, *token_counts(response), (time.perf_counter() - t0) * 1000
    )
