"""
Retrieval Router

Routes retrieval queries to the appropriate backend (PageIndex or Haystack)
and merges results into unified EvidenceBundle format.

Routing logic:
- PageIndex: Policy docs, postmortems, specs (when provenance matters)
- Haystack: Semantic search across all content
- Both: Comprehensive search with deduplication
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional
from uuid import UUID

from src.models.evidence import EvidenceBundle
from src.models.enums import RetrievalMethod
from src.pageindex.service import PageIndexService
from src.haystack_svc.service import HaystackService


class RetrievalStrategy(str, Enum):
    """Strategy for retrieval routing."""
    PAGEINDEX_ONLY = "pageindex_only"
    HAYSTACK_ONLY = "haystack_only"
    BOTH_MERGE = "both_merge"
    AUTO = "auto"


@dataclass
class RetrievalRequest:
    """Request for evidence retrieval."""
    query: str
    doc_ids: Optional[list[str]] = None  # Filter to specific documents
    doc_titles: Optional[dict[str, str]] = None  # doc_id -> title mapping
    strategy: RetrievalStrategy = RetrievalStrategy.AUTO
    max_results: int = 10
    require_provenance: bool = False  # If True, prefer PageIndex


class RetrievalRouter:
    """
    Routes retrieval requests to appropriate backends.
    
    Features:
    - Automatic strategy selection
    - Result merging and deduplication
    - Unified EvidenceBundle output
    """
    
    def __init__(
        self,
        pageindex_service: Optional[PageIndexService] = None,
        haystack_service: Optional[HaystackService] = None,
    ):
        self.pageindex = pageindex_service or PageIndexService()
        self.haystack = haystack_service or HaystackService()
    
    async def initialize(self) -> None:
        """Initialize both services."""
        await self.haystack.initialize()
    
    async def close(self) -> None:
        """Close both services."""
        await self.pageindex.close()
    
    async def retrieve(
        self,
        request: RetrievalRequest,
    ) -> list[EvidenceBundle]:
        """
        Retrieve evidence based on request strategy.
        
        Args:
            request: RetrievalRequest with query and options
            
        Returns:
            List of EvidenceBundle, deduplicated and ranked
        """
        strategy = self._determine_strategy(request)
        
        if strategy == RetrievalStrategy.PAGEINDEX_ONLY:
            return await self._retrieve_pageindex(request)
        elif strategy == RetrievalStrategy.HAYSTACK_ONLY:
            return await self._retrieve_haystack(request)
        else:  # BOTH_MERGE
            return await self._retrieve_both(request)
    
    async def retrieve_simple(
        self,
        query: str,
        max_results: int = 5,
    ) -> list[EvidenceBundle]:
        """
        Simple retrieval interface.
        
        Args:
            query: Search query
            max_results: Maximum results
            
        Returns:
            Evidence bundles from best available source
        """
        request = RetrievalRequest(
            query=query,
            strategy=RetrievalStrategy.AUTO,
            max_results=max_results,
        )
        return await self.retrieve(request)
    
    def _determine_strategy(self, request: RetrievalRequest) -> RetrievalStrategy:
        """Determine best strategy for request."""
        if request.strategy != RetrievalStrategy.AUTO:
            return request.strategy
        
        # If provenance required, prefer PageIndex
        if request.require_provenance:
            return RetrievalStrategy.PAGEINDEX_ONLY
        
        # If specific documents requested, use both
        if request.doc_ids:
            return RetrievalStrategy.BOTH_MERGE
        
        # Default to Haystack for semantic search
        return RetrievalStrategy.HAYSTACK_ONLY
    
    async def _retrieve_pageindex(
        self,
        request: RetrievalRequest,
    ) -> list[EvidenceBundle]:
        """Retrieve from PageIndex only."""
        all_bundles = []
        
        if request.doc_ids:
            for doc_id in request.doc_ids:
                doc_title = (request.doc_titles or {}).get(doc_id, doc_id)
                bundles = await self.pageindex.retrieve_evidence(
                    doc_id=doc_id,
                    doc_title=doc_title,
                    query=request.query,
                    max_results=request.max_results // max(len(request.doc_ids), 1),
                )
                all_bundles.extend(bundles)
        
        return all_bundles[:request.max_results]
    
    async def _retrieve_haystack(
        self,
        request: RetrievalRequest,
    ) -> list[EvidenceBundle]:
        """Retrieve from Haystack only."""
        doc_id = request.doc_ids[0] if request.doc_ids else None
        doc_title = None
        if doc_id and request.doc_titles:
            doc_title = request.doc_titles.get(doc_id)
        
        return await self.haystack.retrieve_evidence(
            query=request.query,
            doc_id=doc_id,
            doc_title=doc_title,
            top_k=request.max_results,
        )
    
    async def _retrieve_both(
        self,
        request: RetrievalRequest,
    ) -> list[EvidenceBundle]:
        """Retrieve from both and merge results."""
        # Get from both sources
        pi_bundles = await self._retrieve_pageindex(request)
        hs_bundles = await self._retrieve_haystack(request)
        
        # Merge and deduplicate
        all_bundles = pi_bundles + hs_bundles
        return self._deduplicate(all_bundles, request.max_results)
    
    def _deduplicate(
        self,
        bundles: list[EvidenceBundle],
        max_results: int,
    ) -> list[EvidenceBundle]:
        """Deduplicate evidence bundles by content hash."""
        seen_hashes: set[str] = set()
        unique_bundles: list[EvidenceBundle] = []
        
        for bundle in bundles:
            if bundle.content_hash not in seen_hashes:
                seen_hashes.add(bundle.content_hash)
                unique_bundles.append(bundle)
                
                if len(unique_bundles) >= max_results:
                    break
        
        return unique_bundles
    
    # ----- Document Management -----
    
    async def register_document_pageindex(
        self,
        doc_id: str,
        filename: str,
        content: bytes,
        content_type: str,
    ) -> str:
        """Register document with PageIndex."""
        return await self.pageindex.register_document(
            doc_id=doc_id,
            filename=filename,
            content=content,
            content_type=content_type,
        )
    
    async def index_document_haystack(
        self,
        doc_id: str,
        content: str,
        filename: Optional[str] = None,
    ) -> list[str]:
        """Index document with Haystack."""
        return await self.haystack.index_document(
            doc_id=doc_id,
            content=content,
            filename=filename,
        )
    
    async def ingest_document(
        self,
        doc_id: str,
        filename: str,
        content: bytes,
        content_type: str,
    ) -> dict[str, any]:
        """
        Ingest document into both retrieval systems.
        
        Returns:
            Dict with pageindex_doc_id and haystack_chunk_ids
        """
        # Register with PageIndex
        pi_doc_id = await self.register_document_pageindex(
            doc_id=doc_id,
            filename=filename,
            content=content,
            content_type=content_type,
        )
        
        # Index with Haystack
        text_content = content.decode("utf-8", errors="ignore")
        hs_chunk_ids = await self.index_document_haystack(
            doc_id=doc_id,
            content=text_content,
            filename=filename,
        )
        
        return {
            "pageindex_doc_id": pi_doc_id,
            "haystack_chunk_ids": hs_chunk_ids,
        }
