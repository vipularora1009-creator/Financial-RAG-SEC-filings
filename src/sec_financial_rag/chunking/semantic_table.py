"""Exp-2 chunker (spec §4.3 Strategy 2): structure first, tables kept whole,
narrative prose cut where the topic changes.

Input is one section's Markdown from the layout parser: blocks separated by a
blank line, tables as "| ... |" lines (a table never contains a blank line).
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from itertools import groupby

import numpy as np
import tiktoken

from sec_financial_rag.chunking.base import Chunker
from sec_financial_rag.chunking.sentences import breakpoints, split_sentences

HEADER_RESERVE = 96  # tokens left free for the context header (measured max 80)
TABLE_MAX = 1500  # a table up to this size stays one chunk (spec)
LEAD_IN_BLOCKS = 2  # short lines above a table (title, "(In millions)") travel with it
LEAD_IN_MAX_CHARS = 200
# Long narrative sections get semantic breakpoints (spec: Item 1A, Item 7; 10-Q equivalents).
NARRATIVE_SECTIONS = {"Item_1A", "Item_7", "PartI_Item2", "PartII_Item1A"}
# A topic smaller than this joins the next one: a heading goes with the text it
# introduces, and a ~52-token header would otherwise dominate a tiny chunk's vector.
MIN_TOPIC_TOKENS = 100

Embed = Callable[[list[str]], np.ndarray]  # sentences -> unit vectors (CachedEmbedder)
Item = tuple[str, str]  # (text, separator to use if joined to the previous text)


@dataclass
class Unit:
    """A piece the packer never splits: a prose block, or a table with its lead-in."""

    body: str
    lead: list[str] = field(default_factory=list)

    @property
    def is_table(self) -> bool:
        return self.body.startswith("|")

    @property
    def text(self) -> str:
        return "\n\n".join([*self.lead, self.body])


def units_of(text: str) -> list[Unit]:
    """Blocks in order; each table takes the short text lines right above it."""
    units: list[Unit] = []
    for block in (b.strip() for b in text.split("\n\n")):
        if not block:
            continue
        unit = Unit(block)
        if unit.is_table:
            while (
                len(unit.lead) < LEAD_IN_BLOCKS
                and units
                and not units[-1].is_table
                and len(units[-1].body) <= LEAD_IN_MAX_CHARS
            ):
                unit.lead.insert(0, units.pop().body)
        units.append(unit)
    return units


class SemanticTableChunker(Chunker):
    def __init__(
        self,
        size: int = 512,
        table_max: int = TABLE_MAX,
        embed: Embed | None = None,
        percentile: float = 85,
        min_topic: int = MIN_TOPIC_TOKENS,
    ):
        self.budget = size - HEADER_RESERVE  # body tokens; the header fills the rest
        self.table_max = table_max
        self.embed = embed  # None: narrative prose is packed by sentences only
        self.percentile = percentile
        self.min_topic = min_topic
        self.encoding = tiktoken.get_encoding("cl100k_base")

    def tokens(self, text: str) -> int:
        return len(self.encoding.encode(text))

    def split(self, text: str, narrative: bool = False) -> list[str]:
        """Chunks in document order. `narrative=True` adds topic breakpoints."""
        units = units_of(text)
        if not (narrative and self.embed):
            return self.pack([(p, "\n\n") for u in units for p in self.pieces(u)])
        starts = self.topic_starts(units)
        chunks: list[str] = []
        for is_table, run in groupby(enumerate(units), key=lambda x: x[1].is_table):
            run = list(run)
            if is_table:
                chunks += self.pack(
                    [(p, "\n\n") for _, u in run for p in self.pieces(u)]
                )
            else:
                chunks += self.pack_topics(run, starts)
        return chunks

    def pack(self, items: list[Item]) -> list[str]:
        """Greedy packing: add items to the current chunk until the next won't fit."""
        chunks: list[str] = []
        current, used = "", 0
        for text, sep in items:
            n = self.tokens(text) + 1  # + roughly one token for the separator
            if current and used + n > self.budget:
                chunks.append(current)
                current, used = "", 0
            current = f"{current}{sep}{text}" if current else text
            used += n
        if current:
            chunks.append(current)
        return chunks

    def pieces(self, unit: Unit) -> list[str]:
        """The unit as-is if it fits; otherwise cut it (tables by rows, prose by sentences)."""
        if unit.is_table:
            if self.tokens(unit.text) <= self.table_max:
                return [unit.text]
            return self.row_groups(unit)
        if self.tokens(unit.text) <= self.budget:
            return [unit.text]
        return self.pack(self.sentence_items(split_sentences(unit.body)))

    def sentence_items(self, sentences: list[str], sep: str = " ") -> list[Item]:
        """Sentences as packable items; a sentence longer than the budget is cut."""
        items: list[Item] = []
        for s in sentences:
            parts = self.windows(s) if self.tokens(s) > self.budget else [s]
            items += [(part, sep) for part in parts]
        return items

    def topic_starts(self, units: list[Unit]) -> set[tuple[int, int]]:
        """(unit index, sentence index) of every sentence that starts a new topic.

        One embedding call for the whole section, so the 85th-percentile
        threshold is computed per section, as the spec says.
        """
        where, sentences = [], []
        for ui, unit in enumerate(units):
            if not unit.is_table:
                for si, sentence in enumerate(split_sentences(unit.body)):
                    where.append((ui, si))
                    sentences.append(sentence)
        if not sentences:
            return set()
        return {where[i] for i in breakpoints(self.embed(sentences), self.percentile)}

    def pack_topics(self, run: list[tuple[int, Unit]], starts: set) -> list[str]:
        """Prose run -> chunks: a topic start always opens a new chunk."""
        topics: list[list[Item]] = [[]]
        for ui, unit in run:
            for si, sentence in enumerate(split_sentences(unit.body)):
                if (ui, si) in starts and topics[-1]:
                    topics.append([])
                sep = " " if si > 0 else "\n\n"  # new paragraph -> blank line
                topics[-1] += self.sentence_items([sentence], sep)
        return [
            chunk for topic in self.merge_small(topics) for chunk in self.pack(topic)
        ]

    def merge_small(self, topics: list[list[Item]]) -> list[list[Item]]:
        """Join each topic under `min_topic` tokens onto the next (the last onto the previous)."""
        merged: list[list[Item]] = []
        carry: list[Item] = []
        for topic in topics:
            topic = carry + topic
            carry = []
            if sum(self.tokens(text) for text, _ in topic) < self.min_topic:
                carry = topic
            else:
                merged.append(topic)
        if carry:
            if merged:
                merged[-1] += carry
            else:
                merged.append(carry)
        return merged

    def row_groups(self, unit: Unit) -> list[str]:
        """Split a big table into row groups, repeating lead-in and header row in each."""
        header, separator, *rows = unit.body.split("\n")
        prefix = "\n\n".join([*unit.lead, f"{header}\n{separator}"])
        room = self.table_max - self.tokens(prefix)
        groups: list[list[str]] = [[]]
        used = 0
        for row in rows:
            n = self.tokens(row) + 1  # + the newline
            if groups[-1] and used + n > room:
                groups.append([])
                used = 0
            groups[-1].append(row)
            used += n
        return [prefix + "\n" + "\n".join(group) for group in groups]

    def windows(self, text: str) -> list[str]:
        """Last resort for one "sentence" longer than the budget (a bullet list with
        no full stops): pack whole words, so no word is cut in half."""
        return self.pack([(word, " ") for word in text.split(" ")])


def is_narrative(section_id: str) -> bool:
    return section_id in NARRATIVE_SECTIONS
