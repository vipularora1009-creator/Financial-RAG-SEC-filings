"""Numerical exact match (spec §5.1): do the gold answer's numbers appear in ours?

Two versions, reported side by side:
- lenient: a number with no unit may be in thousands/millions/billions ("$82,959"
  matches "$82.96 billion"), which is the spec's "$14.2B vs $14,205M" tolerance.
- strict: units must agree as written, so an answer that lost "(in millions)"
  fails. The lenient-strict gap measures unit loss (expected large in Exp-1).
"""

import re
from dataclasses import dataclass

SCALES = {
    "trillion": 1e12, "tn": 1e12,
    "billion": 1e9, "bn": 1e9, "b": 1e9,
    "million": 1e6, "mn": 1e6, "m": 1e6,
    "thousand": 1e3, "k": 1e3,
}  # fmt: skip
UNIT_GUESSES = (1, 1e3, 1e6, 1e9, 1e-3, 1e-6, 1e-9)  # lenient only

NUMBER = re.compile(
    # Not inside a word (FY2022, Q2, COVID-19), an 'year ('23) or a citation [1].
    r"(?<![\w'.\[-])"
    r"(?P<neg>-|−)?\s?(?P<dollar>\$)?\s?"
    r"(?P<paren>\((?=\d))?"  # accounting negative "(1,234)" / "$(1,234)", not "($1.8 bn)"
    r"(?P<int>\d{1,3}(?:,\d{3})+|\d+)(?P<dec>\.\d+)?\)?"
    r"(?:\s?(?P<unit>%|percent|trillion|tn|billion|bn|b|million|mn|m|thousand|k)(?![a-z]))?"
    r"(?![\w%\]])",  # not followed by letters/digits: rejects "1x", "777X"
    re.IGNORECASE,
)


@dataclass
class Num:
    value: float  # in base units (dollars, or percentage points)
    is_pct: bool
    has_unit: bool  # % or a scale word was written
    step: float  # precision: 1.8 bn -> 0.1e9; 82,959 -> 1
    start: int = 0  # position in the text


def extract_numbers(text: str) -> list[Num]:
    """Numbers in `text` with their units, skipping years and labels."""
    nums = []
    for m in NUMBER.finditer(text):
        unit = (m["unit"] or "").lower()
        digits = m["int"].replace(",", "")
        decimals = len(m["dec"]) - 1 if m["dec"] else 0
        is_year = not (unit or m["dollar"] or m["dec"] or "," in m["int"]) and (
            1990 <= int(digits) <= 2035
        )
        # "(1) pricing... (2) ..." in prose is a list marker, not a negative number.
        is_list_marker = (
            m["paren"] and not (m["dollar"] or m["dec"] or unit) and len(digits) <= 2
        )
        if is_year or is_list_marker:
            continue
        scale = SCALES.get(unit, 1.0)
        sign = -1 if m["neg"] or m["paren"] else 1
        nums.append(
            Num(
                value=sign * float(digits + (m["dec"] or "")) * scale,
                is_pct=unit in ("%", "percent"),
                has_unit=bool(unit),
                step=10.0**-decimals * scale,
                start=m.start(),
            )
        )
    return nums


def same_number(gold: Num, ours: Num, lenient: bool) -> bool:
    """Same kind and equal within rounding at the coarser of the two precisions.

    Signs are ignored: answers often say "negative $1,561 million" in words.
    """
    if gold.is_pct != ours.is_pct:
        return False
    unit_unknown = not (gold.has_unit and ours.has_unit)
    guesses = UNIT_GUESSES if lenient and unit_unknown and not gold.is_pct else (1,)
    for g in guesses:
        tolerance = max(gold.step, ours.step * g) / 2 + 1e-9 * abs(gold.value)
        if abs(abs(ours.value * g) - abs(gold.value)) <= tolerance:
            return True
    return False


def numeric_match(gold_answer: str, answer: str) -> tuple[bool | None, bool | None]:
    """(lenient, strict): every gold number found in our answer. (None, None) if the
    gold answer has no number, so the question doesn't count for this metric."""
    gold_nums, ours = extract_numbers(gold_answer), extract_numbers(answer)
    if not gold_nums:
        return None, None
    lenient = all(any(same_number(g, o, True) for o in ours) for g in gold_nums)
    strict = all(any(same_number(g, o, False) for o in ours) for g in gold_nums)
    return lenient, strict
