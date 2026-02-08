"""
Retrieval Router

Routes retrieval queries to the appropriate backend (PageIndex or Haystack)
and merges results into unified EvidenceBundle format.

Phase 2 additions:
- HYBRID_MERGE strategy (BM25 + vector fusion within Haystack)
- Hypothesis-aware retrieval for causal claim testing
- Evidence sufficiency thresholds
- Re-ranking support

Routing logic:
- PageIndex: Policy docs, postmortems, specs (when provenance matters)
- Haystack: Semantic search across all content
- Hybrid: BM25 + vector fusion with re-ranking
- Both: Comprehensive search with deduplication
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
from uuid import UUID

from src.models.evidence import EvidenceBundle
from src.models.enums import RetrievalMethod
from src.pageindex.service import PageIndexService
from src.haystack_svc.service import (
    HaystackService,
    HypothesisQuery,
    EvidenceAssessment,
)


class RetrievalStrategy(str, Enum):
    """Strategy for retrieval routing."""
    PAGEINDEX_ONLY = "pageindex_only"
    HAYSTACK_ONLY = "haystack_only"
    HYBRID = "hybrid"          # Phase 2: BM25 + vector inside Haystack
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
    use_reranking: bool = False       # Phase 2: apply re-ranking
    vector_weight: float = 0.5        # Phase 2: weight for vector score
    bm25_weight: float = 0.5          # Phase 2: weight for BM25 score


class RetrievalRouter:
    """
    Routes retrieval requests to appropriate backends.
    
    Features:
    - Automatic strategy selection
    - Hybrid BM25 + vector retrieval (Phase 2)
    - Hypothesis-aware retrieval (Phase 2)
    - Evidence sufficiency checks (Phase 2)
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
        elif strategy == RetrievalStrategy.HYBRID:
            return await self._retrieve_hybrid(request)
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
        
        # Phase 2: default to hybrid for better recall
        return RetrievalStrategy.HYBRID
    
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
        all_bundles: list[EvidenceBundle] = []

        if request.doc_ids:
            # Query per doc_id so we don't lose any
            per_doc = max(1, request.max_results // len(request.doc_ids))
            for doc_id in request.doc_ids:
                doc_title = (request.doc_titles or {}).get(doc_id, doc_id)
                bundles = await self.haystack.retrieve_evidence(
                    query=request.query,
                    doc_id=doc_id,
                    doc_title=doc_title,
                    top_k=per_doc,
                )
                all_bundles.extend(bundles)
        else:
            all_bundles = await self.haystack.retrieve_evidence(
                query=request.query,
                top_k=request.max_results,
            )

        return all_bundles[:request.max_results]
    
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

    async def _retrieve_hybrid(
        self,
        request: RetrievalRequest,
    ) -> list[EvidenceBundle]:
        """Retrieve using hybrid BM25 + vector fusion."""
        all_bundles: list[EvidenceBundle] = []

        if request.doc_ids:
            per_doc = max(1, request.max_results // len(request.doc_ids))
            for doc_id in request.doc_ids:
                doc_title = (request.doc_titles or {}).get(doc_id, doc_id)
                bundles = await self.haystack.retrieve_hybrid(
                    query=request.query,
                    doc_id=doc_id,
                    doc_title=doc_title,
                    top_k=per_doc,
                    vector_weight=request.vector_weight,
                    bm25_weight=request.bm25_weight,
                    rerank=request.use_reranking,
                )
                all_bundles.extend(bundles)
        else:
            all_bundles = await self.haystack.retrieve_hybrid(
                query=request.query,
                top_k=request.max_results,
                vector_weight=request.vector_weight,
                bm25_weight=request.bm25_weight,
                rerank=request.use_reranking,
            )

        return all_bundles[:request.max_results]

    # ------------------------------------------------------------------ #
    # Hypothesis-aware retrieval (Phase 2)
    # ------------------------------------------------------------------ #

    async def retrieve_for_hypothesis(
        self,
        cause: str,
        effect: str,
        mechanism: str = "",
        domain: str = "",
        top_k: int = 5,
    ) -> EvidenceAssessment:
        """
        Test a causal hypothesis against the evidence corpus.

        Convenience method that delegates to HaystackService.

        Args:
            cause:     Cause variable name
            effect:    Effect variable name
            mechanism: Optional mechanism description
            domain:    Optional domain context
            top_k:     Max results per direction

        Returns:
            EvidenceAssessment with supporting/contradicting evidence
        """
        hypothesis = HypothesisQuery(
            cause=cause,
            effect=effect,
            mechanism=mechanism,
            domain=domain,
        )
        return await self.haystack.retrieve_for_hypothesis(
            hypothesis=hypothesis,
            top_k=top_k,
        )

    async def check_evidence_sufficiency(
        self,
        cause: str,
        effect: str,
        existing_evidence: list[EvidenceBundle] | None = None,
        min_supporting: int = 2,
    ) -> EvidenceAssessment:
        """
        Check whether sufficient evidence exists for a causal claim.

        Args:
            cause:              Cause variable
            effect:             Effect variable
            existing_evidence:  Previously retrieved evidence (optional)
            min_supporting:     Minimum supporting evidence count

        Returns:
            EvidenceAssessment with sufficiency judgment
        """
        return await self.haystack.check_evidence_sufficiency(
            cause=cause,
            effect=effect,
            existing_evidence=existing_evidence,
            min_supporting=min_supporting,
        )
    
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
        from src.extraction.extractor import DocumentExtractor

        extractor = DocumentExtractor()
        text_content = extractor.extract(content, content_type, filename)

        if not text_content.strip():
            raise ValueError(
                f"Text extraction produced no content for {filename} "
                f"(content_type={content_type}, {len(content)} bytes)"
            )

        # Register with PageIndex
        pi_doc_id = await self.register_document_pageindex(
            doc_id=doc_id,
            filename=filename,
            content=content,
            content_type=content_type,
        )
        
        # Index with Haystack (always text from here on)
        hs_chunk_ids = await self.index_document_haystack(
            doc_id=doc_id,
            content=text_content,
            filename=filename,
        )
        
        return {
            "pageindex_doc_id": pi_doc_id,
            "haystack_chunk_ids": hs_chunk_ids,
        }
