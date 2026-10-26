"""Chunker interface (spec §4.3): every strategy turns one document's text into chunks."""

from abc import ABC, abstractmethod


class Chunker(ABC):
    """Exp-1..3 swap strategies behind this one method, so the rest of the pipeline never changes."""

    @abstractmethod
    def split(self, text: str) -> list[str]:
        """Split `text` into chunks, in document order."""
