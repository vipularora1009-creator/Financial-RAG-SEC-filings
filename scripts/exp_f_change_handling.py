"""Exp-F measurement 4: what does each pipeline re-embed when one sentence changes?

Run (after exp_f_index.py has parsed the filings):
    uv run --group exp-f python scripts/exp_f_change_handling.py

One sentence in 3M's FY2022 10-K, Item 1A, gets edited.
- Ours: the CDC ledger hashes each SEC Item section, so only the changed
  section's children are re-embedded (counted from our index).
- LlamaIndex: IngestionPipeline with a docstore (UPSERTS strategy), run on the
  original and then the edited document, with a counting fake embedder.
"""

from llama_index.core import Document
from llama_index.core.ingestion import DocstoreStrategy, IngestionPipeline
from llama_index.core.node_parser import HierarchicalNodeParser
from llama_index.core.storage.docstore import SimpleDocumentStore
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

from sec_financial_rag.exp_f import pipeline
from sec_financial_rag.ingestion.build_exp4 import COLLECTION as OURS
from sec_financial_rag.ingestion.indexer import QDRANT_URL

ACCESSION = "0000066740-23-000014"  # 3M 10-K, FY2022
SECTION = "Item_1A"


class CountingModel:
    def __init__(self):
        self.texts = 0

    def encode(self, texts, normalize_embeddings=True):
        import numpy as np

        self.texts += len(texts)
        return np.ones((len(texts), 4))


def our_children() -> list[dict]:
    points, _ = QdrantClient(url=QDRANT_URL).scroll(
        OURS,
        scroll_filter=Filter(
            must=[
                FieldCondition(
                    key="accession_number", match=MatchValue(value=ACCESSION)
                )
            ]
        ),
        limit=10_000,
        with_payload=["section_id", "text"],
    )
    return [p.payload for p in points]


def edited_snippet(children: list[dict], markdown: str) -> str:
    """A 60-character piece of Item 1A prose present in both parsings."""
    for child in children:
        if child["section_id"] != SECTION:
            continue
        for line in child["text"].splitlines():
            snippet = line.strip()[:60]
            if len(snippet) == 60 and snippet in markdown:
                return snippet
    raise SystemExit("no shared Item 1A sentence found")


def main() -> None:
    children = our_children()
    markdown = (pipeline.MARKDOWN_DIR / f"{ACCESSION}.md").read_text()
    snippet = edited_snippet(children, markdown)
    edited = markdown.replace(snippet, snippet + " (edited)", 1)
    ours = sum(c["section_id"] == SECTION for c in children)
    print(
        f"ours: {ours} of {len(children)} children re-embedded ({ours / len(children):.0%})"
    )

    counter = CountingModel()
    ingest = IngestionPipeline(
        transformations=[
            HierarchicalNodeParser.from_defaults(chunk_sizes=pipeline.CHUNK_SIZES),
            pipeline.EncodeEmbedding(model=counter),
        ],
        docstore=SimpleDocumentStore(),
        docstore_strategy=DocstoreStrategy.UPSERTS,
    )
    for label, text in (
        ("first run", markdown),
        ("edited", edited),
        ("unchanged", edited),
    ):
        before = counter.texts
        nodes = ingest.run(documents=[Document(id_=ACCESSION, text=text)])
        print(
            f"llamaindex {label:9s}: {counter.texts - before} texts embedded, {len(nodes)} nodes"
        )


if __name__ == "__main__":
    main()
