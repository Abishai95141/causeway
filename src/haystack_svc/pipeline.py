"""
Haystack Pipeline

RAG pipeline using Haystack 2.x with Qdrant vector store.
Provides semantic search over document chunks with proper
sentence-aware splitting and document-level embedding.

Phase 2 additions:
- BM25 keyword scoring (mock and real)
- Hybrid search with Reciprocal Rank Fusion (RRF)
- Cross-encoder re-ranking
- Score normalisation across retrieval methods
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from math import log
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
        # Step 1: Check that all required packages are importable.
        #         This is the ONLY place we should catch ImportError for
        #         missing top-level packages.
        try:
            from haystack_integrations.document_stores.qdrant import QdrantDocumentStore  # noqa: F401
            from haystack.components.embedders import (                                   # noqa: F401
                SentenceTransformersTextEmbedder,
                SentenceTransformersDocumentEmbedder,
            )
            from haystack.components.preprocessors import DocumentSplitter               # noqa: F401
            from haystack_integrations.components.retrievers.qdrant import QdrantEmbeddingRetriever  # noqa: F401
        except ImportError as exc:
            logger.warning("Haystack / Qdrant packages missing — mock mode (%s)", exc)
            self._mock_mode = True
            self._initialized = True
            return

        # Step 2: Connect to Qdrant (separate from package availability)
        from haystack_integrations.document_stores.qdrant import QdrantDocumentStore
        from haystack.components.embedders import (
            SentenceTransformersTextEmbedder,
            SentenceTransformersDocumentEmbedder,
        )
        from haystack.components.preprocessors import DocumentSplitter
        from haystack_integrations.components.retrievers.qdrant import QdrantEmbeddingRetriever

        try:
            self._document_store = QdrantDocumentStore(
                host=self.qdrant_host,
                port=self.qdrant_port,
                index=self.collection_name,
                embedding_dim=384,  # MiniLM-L6-v2
                recreate_index=False,
            )
            self._document_store.count_documents()
        except Exception as exc:
            logger.warning("Qdrant unavailable — falling back to mock mode: %s", exc)
            self._mock_mode = True
            self._initialized = True
            return

        # Step 3: Build pipeline components.
        #         Errors here are REAL bugs (e.g. missing nltk) — do NOT
        #         silently fall back; let them propagate so operators see
        #         what actually broke.
        try:
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
        except Exception as exc:
            logger.error(
                "Pipeline component initialisation FAILED: %s  "
                "(check that nltk, sentence-transformers, etc. are installed)",
                exc,
            )
            raise

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
    # Hybrid Search (Phase 2)
    # ------------------------------------------------------------------ #

    async def hybrid_search(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[dict[str, Any]] = None,
        vector_weight: float = 0.5,
        bm25_weight: float = 0.5,
        rrf_k: int = 60,
    ) -> list[ChunkResult]:
        """
        Hybrid search combining vector similarity and BM25 keyword matching.

        Uses Reciprocal Rank Fusion (RRF) to merge rankings:
            ``score(d) = Σ  weight_i / (k + rank_i(d))``

        Args:
            query:          Search query
            top_k:          Number of results to return
            filters:        Optional metadata filters
            vector_weight:  Weight for vector retrieval (0-1)
            bm25_weight:    Weight for BM25 retrieval (0-1)
            rrf_k:          RRF smoothing constant (default 60)

        Returns:
            Fused and ranked chunk results with combined scores
        """
        if not self._initialized:
            await self.initialize()

        # Retrieve from both pipelines (fetch extra candidates for fusion)
        fetch_k = min(top_k * 3, 50)

        vector_results = await self.search(query, top_k=fetch_k, filters=filters)
        bm25_results = self._bm25_search(query, top_k=fetch_k, filters=filters)

        # Build per-method rank maps  (chunk_id → 0-based rank)
        vector_ranks: dict[str, int] = {
            r.chunk_id: i for i, r in enumerate(vector_results)
        }
        bm25_ranks: dict[str, int] = {
            r.chunk_id: i for i, r in enumerate(bm25_results)
        }

        # Collect all candidate chunk_ids
        all_ids = set(vector_ranks.keys()) | set(bm25_ranks.keys())

        # Lookup table for the actual ChunkResult objects
        chunk_map: dict[str, ChunkResult] = {}
        for r in vector_results:
            chunk_map[r.chunk_id] = r
        for r in bm25_results:
            if r.chunk_id not in chunk_map:
                chunk_map[r.chunk_id] = r

        # RRF fusion
        fused: list[tuple[str, float, dict[str, float]]] = []
        for cid in all_ids:
            score = 0.0
            component_scores: dict[str, float] = {}

            if cid in vector_ranks:
                v_score = vector_weight / (rrf_k + vector_ranks[cid])
                score += v_score
                component_scores["vector"] = vector_results[vector_ranks[cid]].score

            if cid in bm25_ranks:
                b_score = bm25_weight / (rrf_k + bm25_ranks[cid])
                score += b_score
                component_scores["bm25"] = bm25_results[bm25_ranks[cid]].score

            fused.append((cid, score, component_scores))

        # Sort by fused score descending
        fused.sort(key=lambda t: t[1], reverse=True)

        results: list[ChunkResult] = []
        for cid, fused_score, comp in fused[:top_k]:
            base = chunk_map[cid]
            results.append(ChunkResult(
                chunk_id=base.chunk_id,
                content=base.content,
                score=round(fused_score, 6),
                doc_id=base.doc_id,
                metadata={
                    **base.metadata,
                    "retrieval_method": "hybrid",
                    "vector_score": comp.get("vector", 0.0),
                    "bm25_score": comp.get("bm25", 0.0),
                },
            ))

        return results

    # ------------------------------------------------------------------ #
    # BM25 search (keyword / lexical)
    # ------------------------------------------------------------------ #

    def _bm25_search(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[dict[str, Any]] = None,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> list[ChunkResult]:
        """
        BM25 keyword search over stored chunks.

        Uses Okapi BM25 scoring with IDF weighting.

        Args:
            query:   Search query
            top_k:   Number of results
            filters: Optional metadata filters
            k1:      Term frequency saturation parameter
            b:       Length normalisation parameter

        Returns:
            Ranked chunk results with BM25 scores
        """
        query_terms = self._tokenize(query)
        if not query_terms:
            return []

        # Gather candidate chunks from available source
        if self._mock_mode:
            chunks = list(self._mock_chunks.values())
        elif self._document_store:
            # In real mode, pull all documents from Qdrant for BM25 scoring.
            # Build Qdrant filter if doc_id filter is provided.
            qdrant_filters = None
            if filters and filters.get("doc_id"):
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
            try:
                haystack_docs = self._document_store.filter_documents(
                    filters=qdrant_filters,
                )
                chunks = [
                    ChunkResult(
                        chunk_id=d.id or "",
                        content=d.content or "",
                        score=0.0,
                        doc_id=d.meta.get("doc_id", ""),
                        metadata=d.meta,
                    )
                    for d in haystack_docs
                ]
            except Exception as exc:
                logger.warning("BM25: could not fetch docs from Qdrant: %s", exc)
                chunks = []
        else:
            chunks = []

        if not chunks:
            return []

        # Apply filters (mock mode only — real mode already filtered in Qdrant)
        if self._mock_mode and filters:
            doc_id_filter = filters.get("doc_id")
            if doc_id_filter:
                chunks = [c for c in chunks if c.doc_id == doc_id_filter]

        N = len(chunks)
        if N == 0:
            return []

        doc_lengths: dict[str, int] = {}
        doc_term_freq: dict[str, dict[str, int]] = {}
        for chunk in chunks:
            tokens = self._tokenize(chunk.content)
            doc_lengths[chunk.chunk_id] = len(tokens)
            tf: dict[str, int] = {}
            for t in tokens:
                tf[t] = tf.get(t, 0) + 1
            doc_term_freq[chunk.chunk_id] = tf

        avgdl = sum(doc_lengths.values()) / N if N else 1.0

        # IDF for query terms
        idf: dict[str, float] = {}
        for term in query_terms:
            df = sum(1 for tf in doc_term_freq.values() if term in tf)
            idf[term] = log((N - df + 0.5) / (df + 0.5) + 1.0)

        # Score each document
        scored: list[tuple[ChunkResult, float]] = []
        for chunk in chunks:
            score = 0.0
            dl = doc_lengths[chunk.chunk_id]
            tf_map = doc_term_freq[chunk.chunk_id]
            for term in query_terms:
                tf_val = tf_map.get(term, 0)
                if tf_val > 0:
                    numerator = tf_val * (k1 + 1)
                    denominator = tf_val + k1 * (1 - b + b * dl / avgdl)
                    score += idf.get(term, 0.0) * numerator / denominator
            if score > 0:
                scored.append((chunk, score))

        scored.sort(key=lambda x: x[1], reverse=True)

        return [
            ChunkResult(
                chunk_id=chunk.chunk_id,
                content=chunk.content,
                score=round(score, 6),
                doc_id=chunk.doc_id,
                metadata={**chunk.metadata, "retrieval_method": "bm25"},
            )
            for chunk, score in scored[:top_k]
        ]

    # ------------------------------------------------------------------ #
    # Re-ranking (Phase 2)
    # ------------------------------------------------------------------ #

    @staticmethod
    def rerank(
        query: str,
        results: list[ChunkResult],
        top_k: int = 5,
    ) -> list[ChunkResult]:
        """
        Re-rank results using heuristic relevance scoring.

        Uses query-term coverage, density, and proximity as a stand-in
        for a real cross-encoder model.

        Args:
            query:   Original query
            results: Candidate results from initial retrieval
            top_k:   How many to keep after re-ranking

        Returns:
            Re-ranked (and possibly pruned) chunk results
        """
        if not results:
            return []

        query_terms = set(query.lower().split())
        scored: list[tuple[ChunkResult, float]] = []

        for rank, chunk in enumerate(results):
            content_lower = chunk.content.lower()
            words = content_lower.split()
            word_count = max(len(words), 1)

            # Term coverage: fraction of query terms found in chunk
            matched_terms = sum(1 for t in query_terms if t in content_lower)
            term_coverage = matched_terms / max(len(query_terms), 1)

            # Term density: how concentrated the query terms are
            term_hits = sum(1 for w in words if w in query_terms)
            density = term_hits / word_count

            # Proximity bonus: how close query terms appear to each other
            positions = [i for i, w in enumerate(words) if w in query_terms]
            proximity = 0.0
            if len(positions) >= 2:
                spans = [positions[i + 1] - positions[i] for i in range(len(positions) - 1)]
                avg_span = sum(spans) / len(spans)
                proximity = 1.0 / (1.0 + avg_span)

            # Combine signals
            rerank_score = (
                0.4 * term_coverage
                + 0.3 * density
                + 0.2 * proximity
                + 0.1 * chunk.score  # original retrieval score
            )
            scored.append((chunk, rerank_score))

        scored.sort(key=lambda x: x[1], reverse=True)

        return [
            ChunkResult(
                chunk_id=c.chunk_id,
                content=c.content,
                score=round(s, 6),
                doc_id=c.doc_id,
                metadata={**c.metadata, "reranked": True},
            )
            for c, s in scored[:top_k]
        ]

    # ------------------------------------------------------------------ #
    # Tokenisation helper
    # ------------------------------------------------------------------ #

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Simple whitespace + lowercasing tokenizer for BM25."""
        return [w for w in re.sub(r'[^\w\s]', ' ', text.lower()).split() if len(w) > 1]

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
            # QdrantDocumentStore.delete_documents expects a list of doc IDs.
            # Filter by metadata first, collect matching IDs, then delete.
            matching = self._document_store.filter_documents(
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
            if matching:
                ids_to_delete = [d.id for d in matching if d.id]
                if ids_to_delete:
                    self._document_store.delete_documents(ids_to_delete)
                return len(ids_to_delete)
        return 0

    async def delete_all_documents(self) -> int:
        """Delete **every** vector in the collection.

        In production mode this drops and recreates the Qdrant collection
        via the REST API so no orphan points can remain.  In mock mode it
        clears the in-memory/file-backed stores.

        Returns:
            Number of vectors that existed before the wipe.
        """
        if self._mock_mode:
            count = len(self._mock_chunks)
            self._mock_chunks.clear()
            self._mock_embeddings.clear()
            self._save_mock_store()
            return count

        if self._document_store is None:
            return 0

        # Count before drop
        try:
            count = self._document_store.count_documents()
        except Exception:
            count = 0

        # Use Qdrant REST API directly to guarantee deletion
        import httpx

        base_url = f"http://{self.qdrant_host}:{self.qdrant_port}"
        collection_url = f"{base_url}/collections/{self.collection_name}"

        try:
            async with httpx.AsyncClient(timeout=30) as client:
                # Delete the collection
                resp = await client.delete(collection_url)
                resp.raise_for_status()

                # Recreate with same config
                create_payload = {
                    "vectors": {
                        "size": 384,
                        "distance": "Cosine",
                    }
                }
                resp = await client.put(collection_url, json=create_payload)
                resp.raise_for_status()

            logger.info(
                "Qdrant collection '%s' dropped and recreated (%d vectors removed)",
                self.collection_name, count,
            )

            # Reconnect the document store to the fresh collection
            from haystack_integrations.document_stores.qdrant import QdrantDocumentStore

            self._document_store = QdrantDocumentStore(
                host=self.qdrant_host,
                port=self.qdrant_port,
                index=self.collection_name,
                embedding_dim=384,
                recreate_index=False,
            )
            # Re-attach retriever to the fresh store
            if self._retriever is not None:
                from haystack_integrations.components.retrievers.qdrant import (
                    QdrantEmbeddingRetriever,
                )
                self._retriever = QdrantEmbeddingRetriever(
                    document_store=self._document_store,
                    top_k=10,
                )
        except Exception as exc:
            logger.error("Failed to recreate Qdrant collection: %s", exc)
            raise

        return count

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
        if self._mock_mode:
            return len(self._mock_chunks)
        if self._document_store:
            try:
                return self._document_store.count_documents()
            except Exception:
                return 0
        return 0
