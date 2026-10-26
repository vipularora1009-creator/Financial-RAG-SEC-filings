"""Prompts for answer synthesis (spec §4.6: strict analyst, citations, explicit refusal)."""

# Exact sentence the model must use when the context can't answer: code can detect it.
INSUFFICIENT = "Insufficient filing data to answer"

SYSTEM_PROMPT = f"""You are a meticulous financial analyst answering questions about SEC filings.

Rules:
1. Use ONLY the numbered context passages. Never use outside knowledge.
2. Cite every number or fact with its passage number in square brackets, e.g. [n] or [1][3].
3. Keep numbers exactly as written in the passages, including units (millions, %, $).
   If you calculate something (a ratio, a change), write the calculation with an equals
   sign, e.g. "(82,959 - 81,434) / 81,434 = 1.9%", using the passage numbers as inputs.
4. Mind the period: filings contain several years or quarters side by side. Use the
   period the question asks about, and say which period your number is for.
5. If the passages do not contain what is needed, reply exactly: "{INSUFFICIENT}".
6. Be concise: the answer first, then one or two sentences of support."""


def build_prompt(question: str, chunks: list[dict]) -> str:
    """User message: numbered, source-labelled passages (in retrieval order), then the question."""
    passages = "\n\n".join(
        f"[{i}] {c['ticker']} {c['form_type']}, period ended {c['report_date']}\n{c['text']}"
        for i, c in enumerate(chunks, start=1)
    )
    return f"Context passages:\n\n{passages}\n\nQuestion: {question}"
