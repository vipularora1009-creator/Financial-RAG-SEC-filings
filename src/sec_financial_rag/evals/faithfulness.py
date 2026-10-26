"""Numeric faithfulness (spec §5.1): is every figure in the answer backed by the context?

A number in the answer is grounded if it appears in a retrieved chunk, in the
question itself, or right after "=" / "≈" (the shown result of arithmetic, which
the prompt asks for), or repeats such a shown result ("The ratio is 0.54 ...
= 0.54"). Deterministic: free, repeatable, no judge bias. It checks that
arithmetic is shown, not that it is correct.
"""

from sec_financial_rag.evals.numeric import UNIT_GUESSES, Num, extract_numbers
from sec_financial_rag.generation.prompts import INSUFFICIENT


def _supports(source: Num, claim: Num) -> bool:
    """Does a context number back the claim to the precision the claim states?

    Directional, unlike Numerical EM's symmetric rule: "25" must not ground "24.6%",
    and a stray "1" must not ground "$831 million". Unit guesses only for a context
    number without a unit (tables often lose their "(in millions)" header).
    """
    if source.is_pct != claim.is_pct:
        return False
    guesses = UNIT_GUESSES if not source.has_unit and not claim.is_pct else (1,)
    tolerance = claim.step / 2 + 1e-9 * abs(claim.value)
    return any(
        abs(abs(source.value * g) - abs(claim.value)) <= tolerance for g in guesses
    )


def _is_computed(answer: str, num: Num) -> bool:
    """True if the number is the result of shown arithmetic: "... = 1.9%",
    also when Markdown bold sits in between: "... = **1.9%**"."""
    return answer[: num.start].rstrip().rstrip("*").rstrip().endswith(("=", "≈"))


def numeric_faithfulness(
    answer: str, chunks: list[dict], question: str = ""
) -> float | None:
    """Share of the answer's numbers that are grounded (1.0 = no invented figures).

    None when not applicable: a refusal, or an answer with no numbers.
    """
    if answer.startswith(INSUFFICIENT):
        return None
    claims = extract_numbers(answer)
    if not claims:
        return None
    sources = extract_numbers(question)
    for chunk in chunks:
        sources += extract_numbers(chunk["text"])
    results = [claim for claim in claims if _is_computed(answer, claim)]
    grounded = [
        claim in results
        or any(_supports(source, claim) for source in sources)
        or any(_supports(result, claim) for result in results)  # headline repeat
        for claim in claims
    ]
    return sum(grounded) / len(grounded)
