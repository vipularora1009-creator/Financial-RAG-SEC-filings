"""Exp-3 chunker (spec §4.3 Strategy 3): parent-document hierarchy.

Parents (~2,000 tokens) are what the LLM reads; children (~300 tokens) are what
the vector search matches. Built in two stages with SemanticTableChunker:
1. units: 512-token Exp-2 chunks (topic-aware, tables whole up to 2,500 tokens);
2. parents: consecutive units packed up to 2,000 tokens, so a cut falls only
   between units (preferably at a topic change) and a table shares its parent
   with the text around it. Parents never cross an Item boundary.
Each child is cut from one parent's text, so it belongs to exactly that parent.
"""

from dataclasses import dataclass

from sec_financial_rag.chunking.semantic_table import (
    HEADER_RESERVE,
    Embed,
    SemanticTableChunker,
)

PARENT_SIZE = 2000  # spec: parents 1,500-2,500 tokens
PARENT_TABLE_MAX = 2500  # a table up to this size is one parent
UNIT_SIZE = 512  # Exp-2's chunk size: the building blocks of a parent
CHILD_SIZE = 384  # spec: children 256-400 tokens (header + body)


@dataclass
class Parent:
    text: str
    children: list[str]


class ParentChildChunker:
    def __init__(
        self,
        embed: Embed | None = None,
        parent_size: int = PARENT_SIZE,
        parent_table_max: int = PARENT_TABLE_MAX,
        child_size: int = CHILD_SIZE,
        unit_size: int = UNIT_SIZE,
    ):
        self.units = SemanticTableChunker(
            size=unit_size, table_max=parent_table_max, embed=embed
        )
        self.parents = SemanticTableChunker(size=parent_size)  # only its pack() is used
        # table_max = the child body budget: any bigger table becomes row groups
        # with its header repeated (our "table row summaries").
        self.children = SemanticTableChunker(
            size=child_size, table_max=child_size - HEADER_RESERVE
        )

    def split(self, text: str, narrative: bool = False) -> list[Parent]:
        """Parents in document order, each with its children in order."""
        units = self.units.split(text, narrative)
        return [
            Parent(p, self.children.split(p))
            for p in self.parents.pack([(u, "\n\n") for u in units])
        ]
