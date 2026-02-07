"""
Haystack Pipeline

RAG pipeline using Haystack 2.x with Qdrant vector store.
Provides semantic search over document chunks with proper
sentence-aware splitting and document-level embedding.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4
import json
import logging
import re

from src.config import get_settings

MOCK_STORE_PATH = Path(__file__).resolve().parent.parent.parent / ".mock_chunks.json"

logger = logging.getLogger(__name__)


@dataclass
class ChunkResult:
    """A chunk retrieved from Haystack."""
    chunk_id: str
    content: str
    score: float
    doc_id: str
    metadata: dict[str, Any] = field(default_factory=dict)


class HaystackPipeline:
    """
    Haystack 2.x RAG pipeline with Qdrant.

    Features:
    - Sentence-aware document splitting via DocumentSplitter
    - SentenceTransformersDocumentEmbedder for indexing
    - SentenceTransformersTextEmbedder  for queries
    - Qdrant vector store with proper filter API
    - Mock fallback when dependencies are unavailable
    """

    def __init__(
        self,
        qdrant_host: Optional[str] = None,
        qdrant_port: Optional[int] = None,
        collection_name: str = "causeway_chunks",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        settings = get_settings()
        self.qdrant_host = qdrant_host or settings.qdrant_host
        self.qdrant_port = qdrant_port or settings.qdrant_port
        self.collection_name = collection_name
        self.embedding_model = embedding_model

        self._document_store = None
        self._retriever = None
        self._text_embedder = None       # for queries
        self._doc_embedder = None        # for documents
        self._splitter = None
        self._initialized = False
        self._mock_mode = False

        # Mock storage for testing (file-backed so chunks survive restarts)
        self._mock_chunks: dict[str, ChunkResult] = {}
        self._mock_embeddings: dict[str, list[float]] = {}
        self._load_mock_store()

    async def initialize(self) -> None:
        """Initialize the Haystack pipeline components."""
        try:
            from haystack_integrations.document_stores.qdrant import QdrantDocumentStore
            from haystack.components.embedders import (
                SentenceTransformersTextEmbedder,
                SentenceTransformersDocumentEmbedder,
            )
            from haystack.components.preprocessors import DocumentSplitter
            from haystack_integrations.components.retrievers.qdrant import QdrantEmbeddingRetriever

            # Try to connect to Qdrant — fall back to mock on failure
            try:
                self._document_store = QdrantDocumentStore(
                    host=self.qdrant_host,
                    port=self.qdrant_port,
                    index=self.collection_name,
                    embedding_dim=384,  # MiniLM-L6-v2
                    recreate_index=False,
                )
                self._document_store.count_documents()
            except Exception:
                logger.warning("Qdrant unavailable — falling back to mock mode")
                self._mock_mode = True
                self._initialized = True
                return

            # Sentence-aware splitter (replaces naive char chunking)
            self._splitter = DocumentSplitter(
                split_by="sentence",
                split_length=3,          # 3 sentences per chunk
                split_overlap=1,         # 1 sentence overlap
            )

            # Document embedder — used during indexing (takes Document objects)
            self._doc_embedder = SentenceTransformersDocumentEmbedder(
                model=self.embedding_model,
            )
            self._doc_embedder.warm_up()

            # Text embedder — used during queries (takes plain text string)
            self._text_embedder = SentenceTransformersTextEmbedder(
                model=self.embedding_model,
            )
            self._text_embedder.warm_up()

            self._retriever = QdrantEmbeddingRetriever(
                document_store=self._document_store,
            )

            self._initialized = True
            self._mock_mode = False
            logger.info("Haystack pipeline initialised (Qdrant live)")

        except ImportError:
            logger.warning("Haystack / Qdrant packages missing — mock mode")
            self._mock_mode = True
            self._initialized = True

    # ------------------------------------------------------------------ #
    # Document indexing
    # ------------------------------------------------------------------ #

    async def add_document(
        self,
        doc_id: str,
        content: str,
        metadata: Optional[dict[str, Any]] = None,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ) -> list[str]:
        """
        Add a document by splitting, embedding, and storing chunks.

        Args:
            doc_id:        Document identifier
            content:       Full document text (already extracted to str)
            metadata:      Optional metadata to attach to chunks
            chunk_size:    Chars per chunk (used only in mock mode)
            chunk_overlap: Overlap (used only in mock mode)

        Returns:
            List of chunk IDs created
        """
        if not self._initialized:
            await self.initialize()

        if self._mock_mode:
            return self._mock_add_document(
                doc_id, content, metadata, chunk_size, chunk_overlap,
            )

        from haystack import Document
        from haystack.document_stores.types import DuplicatePolicy

        base_meta = {"doc_id": doc_id, **(metadata or {})}

        # 1. Create a single Haystack Document
        source_doc = Document(content=content, meta=base_meta, id=doc_id)

        # 2. Split into sentence-aware chunks
        split_result = self._splitter.run(documents=[source_doc])
        split_docs: list[Document] = split_result["documents"]

        # Assign deterministic IDs and enrich meta
        for idx, doc in enumerate(split_docs):
            doc.id = f"{doc_id}_chunk_{idx}"
            doc.meta.update({
                "chunk_index": idx,
                "total_chunks": len(split_docs),
            })

        # 3. Embed all chunks in one batch via DocumentEmbedder
        embed_result = self._doc_embedder.run(documents=split_docs)
        embedded_docs: list[Document] = embed_result["documents"]

        # 4. Write to Qdrant
        self._document_store.write_documents(
            embedded_docs,
            policy=DuplicatePolicy.OVERWRITE,
        )

        return [d.id for d in embedded_docs]

    # ------------------------------------------------------------------ #
    # Search
    # ------------------------------------------------------------------ #

    async def search(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[dict[str, Any]] = None,
    ) -> list[ChunkResult]:
        """
        Semantic search for relevant chunks.

        Args:
            query:   Search query
            top_k:   Number of results
            filters: Optional metadata filters (Haystack 2.x format)

        Returns:
            Ranked chunk results
        """
        if not self._initialized:
            await self.initialize()

        if self._mock_mode:
            return self._mock_search(query, top_k, filters)

        # Embed query with the TEXT embedder (correct for queries)
        embed_result = self._text_embedder.run(text=query)
        query_embedding = embed_result["embedding"]

        # Build Qdrant-compatible filter if doc_id supplied
        qdrant_filters = None
        if filters and "doc_id" in filters:
            qdrant_filters = {
                "operator": "AND",
                "conditions": [
                    {
                        "field": "meta.doc_id",
                        "operator": "==",
                        "value": filters["doc_id"],
                    }
                ],
            }

        results = self._retriever.run(
            query_embedding=query_embedding,
            top_k=top_k,
            filters=qdrant_filters,
        )

        return [
            ChunkResult(
                chunk_id=doc.id or "",
                content=doc.content or "",
                score=doc.score or 0.0,
                doc_id=doc.meta.get("doc_id", ""),
                metadata=doc.meta,
            )
            for doc in results["documents"]
        ]

    # ------------------------------------------------------------------ #
    # Delete
    # ------------------------------------------------------------------ #

    async def delete_document(self, doc_id: str) -> int:
        """Delete all chunks for a document."""
        if self._mock_mode:
            to_delete = [
                cid for cid in self._mock_chunks if cid.startswith(f"{doc_id}_")
            ]
            for cid in to_delete:
                del self._mock_chunks[cid]
                self._mock_embeddings.pop(cid, None)
            self._save_mock_store()
            return len(to_delete)

        if self._document_store:
            self._document_store.delete_documents(
                filters={
                    "operator": "AND",
                    "conditions": [
                        {
                            "field": "meta.doc_id",
                            "operator": "==",
                            "value": doc_id,
                        }
                    ],
                }
            )
        return 0  # Qdrant doesn't return count

    # ------------------------------------------------------------------ #
    # Mock helpers
    # ------------------------------------------------------------------ #

    def _mock_add_document(
        self,
        doc_id: str,
        content: str,
        metadata: Optional[dict[str, Any]],
        chunk_size: int,
        chunk_overlap: int,
    ) -> list[str]:
        chunks = self._chunk_text(content, chunk_size, chunk_overlap)

        # Pre-compute page boundaries: list of (char_offset, page_number)
        page_markers = [
            (m.start(), int(m.group(1)))
            for m in re.finditer(r"\[Page\s+(\d+)\]", content)
        ]

        # Pre-compute section boundaries: list of (char_offset, section_name)
        section_markers = [
            (m.start(), m.group(1).strip())
            for m in re.finditer(
                r"(?:^|\n)\s*((?:[IVX]+\.|[A-Z][A-Za-z ]+(?:Plan|Statement|Summary|Details|Goals|Objectives|Analysis|Background))(?:\s+[^\n]{0,80})?)",
                content,
            )
        ]

        chunk_ids: list[str] = []
        char_offset = 0
        for i, chunk_text in enumerate(chunks):
            chunk_id = f"{doc_id}_chunk_{i}"
            chunk_ids.append(chunk_id)

            # Determine which page this chunk belongs to
            page_num = None
            # Check for [Page N] inside the chunk itself first
            page_in_chunk = re.findall(r"\[Page\s+(\d+)\]", chunk_text)
            if page_in_chunk:
                page_num = int(page_in_chunk[0])  # first page in chunk
            else:
                # Fall back to the last page marker before this chunk
                for offset, pn in reversed(page_markers):
                    if offset <= char_offset:
                        page_num = pn
                        break

            # Determine which section this chunk belongs to
            section_name = None
            # Check for section header inside the chunk itself
            section_in_chunk = re.findall(
                r"(?:^|\n)\s*((?:[IVX]+\.\s+[A-Z][^\n]+))", chunk_text
            )
            if section_in_chunk:
                section_name = section_in_chunk[0].strip()
            else:
                # Fall back to the last section marker before this chunk
                for offset, sn in reversed(section_markers):
                    if offset <= char_offset:
                        section_name = sn
                        break

            chunk_meta = {
                "doc_id": doc_id,
                "chunk_index": i,
                "total_chunks": len(chunks),
                **(metadata or {}),
            }
            if page_num is not None:
                chunk_meta["page_number"] = page_num
            if section_name is not None:
                chunk_meta["section_name"] = section_name

            self._mock_chunks[chunk_id] = ChunkResult(
                chunk_id=chunk_id,
                content=chunk_text,
                score=0.0,
                doc_id=doc_id,
                metadata=chunk_meta,
            )
            self._mock_embeddings[chunk_id] = [0.1] * 384
            char_offset += len(chunk_text)
        self._save_mock_store()
        return chunk_ids

    @staticmethod
    def _chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
        """Fallback character chunking for mock mode."""
        chunks: list[str] = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end])
            start = end - overlap
            if start >= len(text):
                break
        return chunks

    def _mock_search(
        self,
        query: str,
        top_k: int,
        filters: Optional[dict[str, Any]],
    ) -> list[ChunkResult]:
        results: list[ChunkResult] = []
        query_lower = query.lower()
        query_words = set(query_lower.split())

        for chunk in self._mock_chunks.values():
            if filters:
                doc_id_filter = filters.get("doc_id")
                if doc_id_filter and chunk.doc_id != doc_id_filter:
                    continue

            chunk_words = set(chunk.content.lower().split())
            overlap = len(chunk_words & query_words)
            if overlap > 0 or not query:
                score = overlap / max(len(query_words), 1)
                results.append(
                    ChunkResult(
                        chunk_id=chunk.chunk_id,
                        content=chunk.content,
                        score=score,
                        doc_id=chunk.doc_id,
                        metadata=chunk.metadata,
                    )
                )

        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]

    # ------------------------------------------------------------------ #
    # Mock persistence
    # ------------------------------------------------------------------ #

    def _save_mock_store(self) -> None:
        """Persist mock chunks to disk so they survive server restarts."""
        try:
            data = {
                cid: {
                    "chunk_id": c.chunk_id,
                    "content": c.content,
                    "score": c.score,
                    "doc_id": c.doc_id,
                    "metadata": c.metadata,
                }
                for cid, c in self._mock_chunks.items()
            }
            MOCK_STORE_PATH.write_text(json.dumps(data), encoding="utf-8")
        except Exception as exc:
            logger.debug("Mock store save failed: %s", exc)

    def _load_mock_store(self) -> None:
        """Load previously persisted mock chunks from disk."""
        if not MOCK_STORE_PATH.exists():
            return
        try:
            data = json.loads(MOCK_STORE_PATH.read_text(encoding="utf-8"))
            for cid, cdata in data.items():
                self._mock_chunks[cid] = ChunkResult(
                    chunk_id=cdata["chunk_id"],
                    content=cdata["content"],
                    score=cdata.get("score", 0.0),
                    doc_id=cdata["doc_id"],
                    metadata=cdata.get("metadata", {}),
                )
            if self._mock_chunks:
                logger.info("Loaded %d mock chunks from disk", len(self._mock_chunks))
        except Exception as exc:
            logger.debug("Mock store load failed: %s", exc)

    # ------------------------------------------------------------------ #
    # Properties
    # ------------------------------------------------------------------ #

    @property
    def is_mock_mode(self) -> bool:
        return self._mock_mode

    @property
    def chunk_count(self) -> int:
        return len(self._mock_chunks)
