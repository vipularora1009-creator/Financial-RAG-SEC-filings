"""Exp-F: the same task built from LlamaIndex's standard components.

Install: uv sync --group exp-f   (kept out of the main dependencies)

Only the pipeline changes; the questions, the Gemini prompt and model, and the
metrics are ours. Sizes follow our pipeline where the framework has a size to
set (parents ~2,048, children ~384 tokens; 30 dense + 30 sparse hits), so the
comparison is about our SEC-specific parts, not about chunk size.
"""

from pathlib import Path
from typing import Any

from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.node_parser import HierarchicalNodeParser, get_leaf_nodes
from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.vector_stores.types import VectorStoreQueryMode
from llama_index.vector_stores.qdrant import QdrantVectorStore
from pydantic import Field
from qdrant_client import QdrantClient

COLLECTION = "expf_llamaindex_bge_m3"
MARKDOWN_DIR = Path("data/cache/exp_f_markdown")  # parsed filings (git-ignored)
DOCSTORE = Path("data/cache/exp_f_docstore.json")  # all nodes, for auto-merging
RAW = Path("data/raw_filings")
CHUNK_SIZES = [2048, 384]  # parent, leaf (tiktoken tokens, the framework default)
TOP_K = 30  # dense hits and sparse hits, as our hybrid DEPTH
METADATA = (
    "ticker",
    "company",
    "form_type",
    "report_date",
    "fiscal_year",
    "fiscal_period",
)


class EncodeEmbedding(BaseEmbedding):
    """Adapter: any object with SentenceTransformer's encode() as a LlamaIndex model.

    The same bge-m3 runs on Modal for indexing and locally for queries, as in
    our pipeline, so vectors stay comparable.
    """

    model: Any = Field(exclude=True)
    embed_batch_size: int = 256  # LlamaIndex default is 10: one Modal call per 10 texts

    def _encode(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts, normalize_embeddings=True).tolist()

    def _get_query_embedding(self, query: str) -> list[float]:
        return self._encode([query])[0]

    async def _aget_query_embedding(self, query: str) -> list[float]:
        return self._get_query_embedding(query)

    def _get_text_embedding(self, text: str) -> list[float]:
        return self._encode([text])[0]

    def _get_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        return self._encode(texts)


def filing_metadata(qdrant: QdrantClient, source: str) -> dict[str, dict]:
    """accession -> the filing metadata our pipeline computed (same for both arms)."""
    found, offset = {}, None
    while True:
        points, offset = qdrant.scroll(
            source,
            limit=10_000,
            offset=offset,
            with_payload=["accession_number", *METADATA],
        )
        for p in points:
            found.setdefault(p.payload["accession_number"], p.payload)
        if offset is None:
            return found


def load_documents(
    metadata: dict[str, dict], limit: int | None = None
) -> list[Document]:
    """DoclingReader per filing (Markdown export), cached as Markdown on disk.

    The reader's default id is a random uuid4, which would make every re-run
    look like new documents to an IngestionPipeline; the accession is stable.
    """
    from llama_index.readers.docling import DoclingReader

    MARKDOWN_DIR.mkdir(parents=True, exist_ok=True)
    reader = DoclingReader(id_func=lambda doc, file_path: Path(file_path).parent.name)
    documents = []
    for path in sorted(RAW.glob("*/*/*.htm"))[:limit]:
        accession = path.parent.name
        cached = MARKDOWN_DIR / f"{accession}.md"
        if not cached.exists():
            (doc,) = reader.load_data(path)
            cached.write_text(doc.text)
        documents.append(
            Document(
                id_=accession,
                text=cached.read_text(),
                metadata={"accession_number": accession, **metadata[accession]},
                excluded_embed_metadata_keys=["accession_number"],
                excluded_llm_metadata_keys=["accession_number"],
            )
        )
    return documents


def vector_store(qdrant: QdrantClient) -> QdrantVectorStore:
    return QdrantVectorStore(
        client=qdrant,
        collection_name=COLLECTION,
        enable_hybrid=True,
        fastembed_sparse_model="Qdrant/bm25",  # Qdrant applies IDF, as ours
        batch_size=256,
    )


def build_index(
    documents: list[Document], qdrant: QdrantClient, embed_model: BaseEmbedding
) -> tuple[int, int]:
    """Hierarchical nodes; leaves embedded into Qdrant, every node in the docstore."""
    nodes = HierarchicalNodeParser.from_defaults(
        chunk_sizes=CHUNK_SIZES
    ).get_nodes_from_documents(documents)
    leaves = get_leaf_nodes(nodes)
    docstore = SimpleDocumentStore()
    docstore.add_documents(nodes)
    storage = StorageContext.from_defaults(
        docstore=docstore, vector_store=vector_store(qdrant)
    )
    VectorStoreIndex(
        leaves, storage_context=storage, embed_model=embed_model, show_progress=True
    )
    docstore.persist(str(DOCSTORE))
    return len(nodes), len(leaves)


def make_search(qdrant: QdrantClient, embed_model: BaseEmbedding):
    """question -> ranked dicts shaped like our payloads (for evals.runner.evaluate)."""
    docstore = SimpleDocumentStore.from_persist_path(str(DOCSTORE))
    storage = StorageContext.from_defaults(
        docstore=docstore, vector_store=vector_store(qdrant)
    )
    index = VectorStoreIndex.from_vector_store(
        storage.vector_store, embed_model=embed_model
    )
    base = index.as_retriever(
        vector_store_query_mode=VectorStoreQueryMode.HYBRID,
        similarity_top_k=TOP_K,
        sparse_top_k=TOP_K,
        vector_store_kwargs={"search_params": {"exact": True}},
    )
    retriever = AutoMergingRetriever(base, storage)

    def search(question: str) -> list[dict]:
        return [
            {**hit.node.metadata, "text": hit.node.text, "score": hit.score or 0.0}
            for hit in retriever.retrieve(question)
        ]

    return search
