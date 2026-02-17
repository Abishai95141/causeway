"""
Haystack Service

High-level service for Haystack operations, converting results
to canonical EvidenceBundle format.

Phase 2 additions:
- Hybrid retrieval (BM25 + vector with RRF fusion)
- Hypothesis-aware retrieval (support & refutation queries)
- Evidence sufficiency assessment
- Re-ranking with cross-encoder heuristic
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from src.models.evidence import (
    EvidenceBundle,
    SourceReference,
    LocationMetadata,
    RetrievalTrace,
)
from src.models.enums import RetrievalMethod, EvidenceStrength
from src.haystack_svc.pipeline import HaystackPipeline, ChunkResult


# ---------------------------------------------------------------------------
# Data classes for hypothesis-aware retrieval (Phase 2)
# ---------------------------------------------------------------------------

@dataclass
class HypothesisQuery:
    """A causal hypothesis to test against the evidence corpus."""
    cause: str
    effect: str
    mechanism: str = ""
    domain: str = ""


@dataclass
class EvidenceAssessment:
    """Assessment of evidence sufficiency for a causal claim."""
    supporting: list[EvidenceBundle] = field(default_factory=list)
    contradicting: list[EvidenceBundle] = field(default_factory=list)
    strength: EvidenceStrength = EvidenceStrength.HYPOTHESIS
    confidence: float = 0.3
    is_sufficient: bool = False
    reason: str = ""


class HaystackService:
    """
    High-level Haystack service.
    
    Provides:
    - Document indexing with chunking
    - Evidence retrieval with provenance
    - Hybrid search (BM25 + vector)
    - Hypothesis-aware retrieval
    - Evidence sufficiency assessment
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

    # ------------------------------------------------------------------ #
    # Hybrid retrieval (Phase 2)
    # ------------------------------------------------------------------ #

    async def retrieve_hybrid(
        self,
        query: str,
        doc_id: Optional[str] = None,
        doc_title: Optional[str] = None,
        top_k: int = 5,
        vector_weight: float = 0.5,
        bm25_weight: float = 0.5,
        rerank: bool = True,
    ) -> list[EvidenceBundle]:
        """
        Hybrid retrieval using BM25 + vector with RRF fusion.

        Args:
            query:          Search query
            doc_id:         Optional document filter
            doc_title:      Optional document title for provenance
            top_k:          Maximum results
            vector_weight:  Weight for vector retrieval (0-1)
            bm25_weight:    Weight for BM25 retrieval (0-1)
            rerank:         Whether to apply re-ranking after fusion

        Returns:
            List of EvidenceBundle with full provenance and component scores
        """
        filters = {"doc_id": doc_id} if doc_id else None

        results = await self.pipeline.hybrid_search(
            query=query,
            top_k=top_k * 2 if rerank else top_k,
            filters=filters,
            vector_weight=vector_weight,
            bm25_weight=bm25_weight,
        )

        if rerank:
            results = HaystackPipeline.rerank(query, results, top_k=top_k)

        return [
            self._chunk_to_evidence(
                chunk, query, doc_title,
                extra_scores={
                    "vector": chunk.metadata.get("vector_score", 0.0),
                    "bm25": chunk.metadata.get("bm25_score", 0.0),
                    "fused": chunk.score,
                },
            )
            for chunk in results
        ]

    # ------------------------------------------------------------------ #
    # Hypothesis-aware retrieval (Phase 2)
    # ------------------------------------------------------------------ #

    async def retrieve_for_hypothesis(
        self,
        hypothesis: HypothesisQuery,
        top_k: int = 5,
        doc_id: Optional[str] = None,
        doc_title: Optional[str] = None,
    ) -> EvidenceAssessment:
        """
        Test a causal hypothesis against the evidence corpus.

        Generates both supporting and refuting queries, retrieves evidence
        for each, and classifies the overall strength.

        Args:
            hypothesis: The causal hypothesis to test
            top_k:      Max results per direction (support + refute)
            doc_id:     Optional document filter
            doc_title:  Optional title for provenance

        Returns:
            EvidenceAssessment with supporting/contradicting evidence
        """
        # Build support query
        support_query = (
            f"{hypothesis.cause} causes {hypothesis.effect}"
        )
        if hypothesis.mechanism:
            support_query += f" through {hypothesis.mechanism}"

        # Build refutation query
        refute_query = (
            f"{hypothesis.cause} does not cause {hypothesis.effect} OR "
            f"no relationship between {hypothesis.cause} and {hypothesis.effect} OR "
            f"{hypothesis.cause} has no effect on {hypothesis.effect}"
        )

        filters = {"doc_id": doc_id} if doc_id else None

        # Retrieve both directions using hybrid search
        support_results = await self.pipeline.hybrid_search(
            query=support_query, top_k=top_k, filters=filters,
        )
        refute_results = await self.pipeline.hybrid_search(
            query=refute_query, top_k=top_k, filters=filters,
        )

        # Convert to evidence bundles
        supporting = [
            self._chunk_to_evidence(c, support_query, doc_title)
            for c in support_results
        ]

        # Filter refutation results for actual negation signals
        negation_signals = [
            "not cause", "no effect", "no relationship",
            "does not affect", "no significant", "no evidence",
            "contrary", "disprove", "refute", "negative",
            "no correlation", "unrelated",
        ]
        contradicting: list[EvidenceBundle] = []
        for c in refute_results:
            content_lower = c.content.lower()
            if any(sig in content_lower for sig in negation_signals):
                contradicting.append(
                    self._chunk_to_evidence(c, refute_query, doc_title)
                )

        # Classify strength
        from src.causal.pywhyllm_bridge import CausalGraphBridge
        strength, confidence = CausalGraphBridge.classify_edge_strength(
            len(supporting), len(contradicting),
        )

        # Determine sufficiency
        is_sufficient = (
            len(supporting) >= 2
            and strength in (EvidenceStrength.STRONG, EvidenceStrength.MODERATE)
        )

        reason = self._build_sufficiency_reason(
            len(supporting), len(contradicting), strength, is_sufficient,
        )

        return EvidenceAssessment(
            supporting=supporting,
            contradicting=contradicting,
            strength=strength,
            confidence=confidence,
            is_sufficient=is_sufficient,
            reason=reason,
        )

    # ------------------------------------------------------------------ #
    # Evidence sufficiency check (Phase 2)
    # ------------------------------------------------------------------ #

    async def check_evidence_sufficiency(
        self,
        cause: str,
        effect: str,
        existing_evidence: list[EvidenceBundle] | None = None,
        min_supporting: int = 2,
        max_contradicting_ratio: float = 0.5,
    ) -> EvidenceAssessment:
        """
        Check whether there is sufficient evidence for a causal claim.

        Can operate on pre-existing evidence or retrieve fresh evidence.

        Args:
            cause:                   Cause variable
            effect:                  Effect variable
            existing_evidence:       Previously retrieved evidence (optional)
            min_supporting:          Minimum supporting evidence count
            max_contradicting_ratio: Max ratio of contradicting/total evidence

        Returns:
            EvidenceAssessment with sufficiency judgment
        """
        if existing_evidence is not None:
            # Classify pre-existing evidence
            supporting = existing_evidence
            contradicting: list[EvidenceBundle] = []
            # Simple classification: check if any evidence is actually contradicting
            negation_signals = [
                "not cause", "no effect", "no relationship",
                "does not affect", "no significant",
            ]
            actual_support: list[EvidenceBundle] = []
            for eb in existing_evidence:
                content_lower = eb.content.lower()
                if any(sig in content_lower for sig in negation_signals):
                    contradicting.append(eb)
                else:
                    actual_support.append(eb)
            supporting = actual_support
        else:
            # Retrieve fresh evidence
            hypothesis = HypothesisQuery(cause=cause, effect=effect)
            assessment = await self.retrieve_for_hypothesis(hypothesis)
            return assessment

        from src.causal.pywhyllm_bridge import CausalGraphBridge
        strength, confidence = CausalGraphBridge.classify_edge_strength(
            len(supporting), len(contradicting),
        )

        total = len(supporting) + len(contradicting)
        contra_ratio = len(contradicting) / total if total > 0 else 0.0

        is_sufficient = (
            len(supporting) >= min_supporting
            and contra_ratio <= max_contradicting_ratio
        )

        reason = self._build_sufficiency_reason(
            len(supporting), len(contradicting), strength, is_sufficient,
        )

        return EvidenceAssessment(
            supporting=supporting,
            contradicting=contradicting,
            strength=strength,
            confidence=confidence,
            is_sufficient=is_sufficient,
            reason=reason,
        )

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    async def delete_document(self, doc_id: str) -> int:
        """Delete all chunks for a document."""
        return await self.pipeline.delete_document(doc_id)

    async def delete_all_documents(self) -> int:
        """Delete every vector in the collection (drop + recreate)."""
        return await self.pipeline.delete_all_documents()

    def _chunk_to_evidence(
        self,
        chunk: ChunkResult,
        query: str,
        doc_title: Optional[str] = None,
        extra_scores: Optional[dict[str, float]] = None,
    ) -> EvidenceBundle:
        """Convert Haystack chunk to canonical EvidenceBundle."""
        metadata = chunk.metadata
        
        page_number = metadata.get("page_number")
        section_name = metadata.get("section_name")

        scores: dict[str, float] = {"vector": chunk.score}
        if extra_scores:
            scores.update(extra_scores)

        return EvidenceBundle(
            content=chunk.content,
            source=SourceReference(
                doc_id=chunk.doc_id,
                doc_title=doc_title or metadata.get("filename", chunk.doc_id),
            ),
            location=LocationMetadata(
                chunk_id=chunk.chunk_id,
                page_number=int(page_number) if page_number is not None else None,
                section_name=section_name,
            ),
            retrieval_trace=RetrievalTrace(
                method=RetrievalMethod.HAYSTACK,
                query=query,
                timestamp=datetime.now(timezone.utc),
                scores=scores,
            ),
        )

    @staticmethod
    def _build_sufficiency_reason(
        supporting_count: int,
        contradicting_count: int,
        strength: EvidenceStrength,
        is_sufficient: bool,
    ) -> str:
        """Build a human-readable sufficiency explanation."""
        parts: list[str] = []
        parts.append(f"{supporting_count} supporting, {contradicting_count} contradicting")
        parts.append(f"strength={strength.value}")
        if is_sufficient:
            parts.append("SUFFICIENT — enough converging evidence")
        else:
            reasons: list[str] = []
            if supporting_count < 2:
                reasons.append(f"need ≥2 supporting (have {supporting_count})")
            if contradicting_count > 0 and strength == EvidenceStrength.CONTESTED:
                reasons.append("evidence is contested")
            parts.append("INSUFFICIENT — " + "; ".join(reasons) if reasons else "INSUFFICIENT")
        return "; ".join(parts)

    @property
    def is_mock_mode(self) -> bool:
        """Check if running in mock mode."""
        return self.pipeline.is_mock_mode
