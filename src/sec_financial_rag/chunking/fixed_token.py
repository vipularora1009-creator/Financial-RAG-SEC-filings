"""Exp-1 baseline chunker (spec §4.3 Strategy 1): fixed 512-token windows, 64-token overlap."""

import tiktoken

from sec_financial_rag.chunking.base import Chunker


class FixedTokenChunker(Chunker):
    def __init__(self, size: int = 512, overlap: int = 64):
        self.size = size
        self.overlap = overlap
        self.encoding = tiktoken.get_encoding("cl100k_base")

    def split(self, text: str) -> list[str]:
        """Windows of `size` tokens; each starts `overlap` tokens before the previous one ends."""
        tokens = self.encoding.encode(text)
        if not tokens:
            return []
        step = self.size - self.overlap
        last_start = max(len(tokens) - self.overlap, 1)
        return [
            self.encoding.decode(tokens[start : start + self.size])
            for start in range(0, last_start, step)
        ]
