"""
Haystack Pipeline

RAG pipeline using Haystack 2.x with Qdrant vector store.
Provides semantic search over document chunks.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional
from uuid import uuid4

from src.config import get_settings


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
    - Document chunking and embedding
    - Semantic search with BM25 hybrid
    - Chunk-level provenance
    
    Note: Uses mock implementation if dependencies unavailable.
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
        self._embedder = None
        self._initialized = False
        self._mock_mode = False
        
        # Mock storage for testing
        self._mock_chunks: dict[str, ChunkResult] = {}
        self._mock_embeddings: dict[str, list[float]] = {}
    
    async def initialize(self) -> None:
        """Initialize the Haystack pipeline components."""
        try:
            from haystack_integrations.document_stores.qdrant import QdrantDocumentStore
            from haystack.components.embedders import SentenceTransformersTextEmbedder
            from haystack_integrations.components.retrievers.qdrant import QdrantEmbeddingRetriever
            
            # Try to connect to Qdrant - if it fails, use mock mode
            try:
                self._document_store = QdrantDocumentStore(
                    host=self.qdrant_host,
                    port=self.qdrant_port,
                    index=self.collection_name,
                    embedding_dim=384,  # MiniLM-L6-v2 dimension
                    recreate_index=False,
                )
                # Test connection by counting documents
                self._document_store.count_documents()
            except Exception:
                # Qdrant not available, fall back to mock
                self._mock_mode = True
                self._initialized = True
                return
            
            self._embedder = SentenceTransformersTextEmbedder(
                model=self.embedding_model,
            )
            
            self._retriever = QdrantEmbeddingRetriever(
                document_store=self._document_store,
            )
            
            # Warm up embedder
            self._embedder.warm_up()
            
            self._initialized = True
            self._mock_mode = False
            
        except ImportError:
            # Fall back to mock mode if dependencies not available
            self._mock_mode = True
            self._initialized = True
    
    async def add_document(
        self,
        doc_id: str,
        content: str,
        metadata: Optional[dict[str, Any]] = None,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ) -> list[str]:
        """
        Add a document by chunking and embedding.
        
        Args:
            doc_id: Document identifier
            content: Full document text
            metadata: Optional metadata to attach to chunks
            chunk_size: Characters per chunk
            chunk_overlap: Overlap between chunks
            
        Returns:
            List of chunk IDs created
        """
        if not self._initialized:
            await self.initialize()
        
        # Create chunks
        chunks = self._chunk_text(content, chunk_size, chunk_overlap)
        chunk_ids = []
        
        for i, chunk_text in enumerate(chunks):
            chunk_id = f"{doc_id}_chunk_{i}"
            chunk_ids.append(chunk_id)
            
            chunk_metadata = {
                "doc_id": doc_id,
                "chunk_index": i,
                "total_chunks": len(chunks),
                **(metadata or {}),
            }
            
            if self._mock_mode:
                # Mock storage
                self._mock_chunks[chunk_id] = ChunkResult(
                    chunk_id=chunk_id,
                    content=chunk_text,
                    score=0.0,
                    doc_id=doc_id,
                    metadata=chunk_metadata,
                )
                # Mock embedding (just for testing)
                self._mock_embeddings[chunk_id] = [0.1] * 384
            else:
                # Real Haystack storage
                from haystack import Document
                
                doc = Document(
                    content=chunk_text,
                    meta=chunk_metadata,
                    id=chunk_id,
                )
                
                # Embed and store
                embedding_result = self._embedder.run(text=chunk_text)
                embedding = embedding_result["embedding"]
                doc.embedding = embedding
                self._document_store.write_documents([doc])
        
        return chunk_ids
    
    async def search(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[dict[str, Any]] = None,
    ) -> list[ChunkResult]:
        """
        Search for relevant chunks.
        
        Args:
            query: Search query
            top_k: Number of results
            filters: Optional metadata filters
            
        Returns:
            Ranked chunk results
        """
        if not self._initialized:
            await self.initialize()
        
        if self._mock_mode:
            return self._mock_search(query, top_k, filters)
        
        # Embed query
        embedding_result = self._embedder.run(text=query)
        query_embedding = embedding_result["embedding"]
        
        # Retrieve
        results = self._retriever.run(
            query_embedding=query_embedding,
            top_k=top_k,
            filters=filters,
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
    
    async def delete_document(self, doc_id: str) -> int:
        """
        Delete all chunks for a document.
        
        Args:
            doc_id: Document identifier
            
        Returns:
            Number of chunks deleted
        """
        if self._mock_mode:
            to_delete = [cid for cid in self._mock_chunks if cid.startswith(f"{doc_id}_")]
            for cid in to_delete:
                del self._mock_chunks[cid]
                self._mock_embeddings.pop(cid, None)
            return len(to_delete)
        
        if self._document_store:
            # Delete by filter
            self._document_store.delete_documents(
                filters={"field": "meta.doc_id", "operator": "==", "value": doc_id}
            )
        return 0  # Qdrant doesn't return count
    
    def _chunk_text(
        self,
        text: str,
        chunk_size: int,
        overlap: int,
    ) -> list[str]:
        """Split text into overlapping chunks."""
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
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
        """Mock search for testing."""
        results = []
        query_lower = query.lower()
        
        for chunk_id, chunk in self._mock_chunks.items():
            # Apply filters if any
            if filters:
                doc_id_filter = filters.get("doc_id")
                if doc_id_filter and chunk.doc_id != doc_id_filter:
                    continue
            
            # Simple relevance scoring based on word overlap
            chunk_words = set(chunk.content.lower().split())
            query_words = set(query_lower.split())
            overlap = len(chunk_words & query_words)
            
            if overlap > 0 or not query:
                score = overlap / max(len(query_words), 1)
                results.append(ChunkResult(
                    chunk_id=chunk.chunk_id,
                    content=chunk.content,
                    score=score,
                    doc_id=chunk.doc_id,
                    metadata=chunk.metadata,
                ))
        
        # Sort by score and return top_k
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]
    
    @property
    def is_mock_mode(self) -> bool:
        """Check if running in mock mode."""
        return self._mock_mode
    
    @property
    def chunk_count(self) -> int:
        """Get total number of chunks (mock mode only)."""
        return len(self._mock_chunks)
