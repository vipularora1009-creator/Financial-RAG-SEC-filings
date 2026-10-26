"""Sentence splitting and semantic breakpoints for Strategy 2 (spec §4.3).

A breakpoint is a place where the topic changes: the cosine distance between
two neighbouring sentences is in the top 15% for that section (above the 85th
percentile), so the threshold adapts to each section.
"""

import re
from itertools import pairwise

import numpy as np

# Words that end with "." without ending a sentence: "U.S. Treasury", "Apple Inc. reported".
ABBREVIATIONS = {
    "u.s", "inc", "corp", "co", "ltd", "llc", "plc", "no", "nos", "vs", "approx",
    "mr", "ms", "mrs", "dr", "st", "e.g", "i.e", "jan", "feb", "mar", "apr",
    "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec",
}  # fmt: skip

# A candidate boundary: ".", "!" or "?" then space, then something that can start a sentence.
SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9“\"'(\[$])")


# Cross-references: "see Item 7. of this report", "in Note 5. Debt".
CROSS_REFERENCE = re.compile(r"\b(?:item|note|part)\s+\w+\.$", re.IGNORECASE)


def _ends_with_abbreviation(sentence: str) -> bool:
    """Does the "." at the end belong to a word, not to the sentence?"""
    last = sentence.rsplit(maxsplit=1)[-1].rstrip(".").lower()
    return (
        last in ABBREVIATIONS
        or len(last) == 1  # initials: "J. Smith"
        or bool(CROSS_REFERENCE.search(sentence))
    )


def split_sentences(text: str) -> list[str]:
    """Sentences in order; joining them with " " gives back the (collapsed) text."""
    sentences: list[str] = []
    for part in SENTENCE_END.split(" ".join(text.split())):
        if sentences and _ends_with_abbreviation(sentences[-1]):
            sentences[-1] += " " + part  # the "." belonged to an abbreviation
        else:
            sentences.append(part)
    return [s for s in sentences if s]


def breakpoints(vectors: np.ndarray, percentile: float = 85) -> list[int]:
    """Indices of sentences that start a new topic.

    `vectors` are unit-length sentence embeddings (one row per sentence), so a
    dot product is the cosine similarity. Distance = 1 - similarity.
    """
    if len(vectors) < 3:
        return []  # too few gaps for a percentile to mean anything
    distances = 1 - np.sum(vectors[:-1] * vectors[1:], axis=1)
    threshold = np.percentile(distances, percentile)
    return [i + 1 for i, d in enumerate(distances) if d > threshold]


def segments(sentences: list[str], starts: list[int]) -> list[list[str]]:
    """Cut `sentences` before each index in `starts`."""
    bounds = [0, *starts, len(sentences)]
    return [sentences[a:b] for a, b in pairwise(bounds) if a < b]
