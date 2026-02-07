"""
Haystack Service

High-level service for Haystack operations, converting results
to canonical EvidenceBundle format.
"""

from datetime import datetime, timezone
from typing import Any, Optional

from src.models.evidence import (
    EvidenceBundle,
    SourceReference,
    LocationMetadata,
    RetrievalTrace,
)
from src.models.enums import RetrievalMethod
from src.haystack_svc.pipeline import HaystackPipeline, ChunkResult


class HaystackService:
    """
    High-level Haystack service.
    
    Provides:
    - Document indexing with chunking
    - Evidence retrieval with provenance
    - Unified output format
    """
    
    def __init__(self, pipeline: Optional[HaystackPipeline] = None):
        self.pipeline = pipeline or HaystackPipeline()
    
    async def initialize(self) -> None:
        """Initialize the underlying pipeline."""
        await self.pipeline.initialize()
    
    async def index_document(
        self,
        doc_id: str,
        content: str,
        filename: Optional[str] = None,
        chunk_size: int = 500,
    ) -> list[str]:
        """
        Index a document for semantic search.
        
        Args:
            doc_id: Document identifier
            content: Document text content
            filename: Optional filename for metadata
            chunk_size: Characters per chunk
            
        Returns:
            List of chunk IDs created
        """
        metadata = {}
        if filename:
            metadata["filename"] = filename
        
        return await self.pipeline.add_document(
            doc_id=doc_id,
            content=content,
            metadata=metadata,
            chunk_size=chunk_size,
        )
    
    async def retrieve_evidence(
        self,
        query: str,
        doc_id: Optional[str] = None,
        doc_title: Optional[str] = None,
        top_k: int = 5,
    ) -> list[EvidenceBundle]:
        """
        Retrieve evidence using semantic search.
        
        Args:
            query: Search query
            doc_id: Optional filter to specific document
            doc_title: Optional document title for provenance
            top_k: Maximum results
            
        Returns:
            List of EvidenceBundle with full provenance
        """
        filters = None
        if doc_id:
            filters = {"doc_id": doc_id}
        
        results = await self.pipeline.search(query, top_k, filters)
        
        return [
            self._chunk_to_evidence(chunk, query, doc_title)
            for chunk in results
        ]
    
    async def delete_document(self, doc_id: str) -> int:
        """Delete all chunks for a document."""
        return await self.pipeline.delete_document(doc_id)
    
    def _chunk_to_evidence(
        self,
        chunk: ChunkResult,
        query: str,
        doc_title: Optional[str] = None,
    ) -> EvidenceBundle:
        """Convert Haystack chunk to canonical EvidenceBundle."""
        metadata = chunk.metadata
        
        return EvidenceBundle(
            content=chunk.content,
            source=SourceReference(
                doc_id=chunk.doc_id,
                doc_title=doc_title or metadata.get("filename", chunk.doc_id),
            ),
            location=LocationMetadata(
                chunk_id=chunk.chunk_id,
            ),
            retrieval_trace=RetrievalTrace(
                method=RetrievalMethod.HAYSTACK,
                query=query,
                timestamp=datetime.now(timezone.utc),
                scores={"vector": chunk.score},
            ),
        )
    
    @property
    def is_mock_mode(self) -> bool:
        """Check if running in mock mode."""
        return self.pipeline.is_mock_mode
