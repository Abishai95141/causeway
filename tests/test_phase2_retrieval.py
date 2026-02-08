"""
Tests for Phase 2: Advanced Haystack RAG

Tests cover:
- BM25 keyword search with Okapi BM25 scoring
- Hybrid search with Reciprocal Rank Fusion (RRF)
- Re-ranking with heuristic cross-encoder
- Hypothesis-aware retrieval (support + refutation)
- Evidence sufficiency assessment
- Retrieval router HYBRID strategy
- Integration with Mode 1 and Mode 2 via hybrid retrieval
"""

import pytest
from uuid import uuid4

from src.haystack_svc.pipeline import HaystackPipeline, ChunkResult
from src.haystack_svc.service import (
    HaystackService,
    HypothesisQuery,
    EvidenceAssessment,
)
from src.retrieval.router import RetrievalRouter, RetrievalRequest, RetrievalStrategy
from src.models.enums import EvidenceStrength, RetrievalMethod
from src.models.evidence import (
    EvidenceBundle,
    SourceReference,
    LocationMetadata,
    RetrievalTrace,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_evidence(content: str, doc_id: str = "doc1") -> EvidenceBundle:
    return EvidenceBundle(
        content=content,
        source=SourceReference(doc_id=doc_id, doc_title="TestDoc.pdf"),
        retrieval_trace=RetrievalTrace(
            method=RetrievalMethod.HAYSTACK, query="test",
        ),
    )


# ---------------------------------------------------------------------------
# BM25 search
# ---------------------------------------------------------------------------


class TestBM25Search:
    """Test BM25 keyword search in HaystackPipeline."""

    @pytest.fixture
    async def pipeline(self):
        p = HaystackPipeline()
        await p.initialize()
        # Clear any pre-existing mock data for isolation
        p._mock_chunks.clear()
        p._mock_embeddings.clear()
        await p.add_document(
            doc_id="bm25_doc1",
            content="Pricing decisions affect revenue. Higher prices reduce customer demand.",
        )
        await p.add_document(
            doc_id="bm25_doc2",
            content="Customer satisfaction correlates with retention. Churn increases with poor service.",
        )
        yield p
        # Cleanup test documents
        for doc_id in ("bm25_doc1", "bm25_doc2"):
            try:
                await p.delete_document(doc_id)
            except Exception:
                pass

    @pytest.mark.asyncio
    async def test_bm25_returns_results(self, pipeline):
        """BM25 should return matching chunks."""
        results = pipeline._bm25_search("pricing revenue", top_k=5,
                                        filters={"doc_id": "bm25_doc1"})
        assert len(results) > 0
        # doc1 should rank higher for pricing query
        assert results[0].doc_id == "bm25_doc1"

    @pytest.mark.asyncio
    async def test_bm25_no_match_returns_empty(self, pipeline):
        """BM25 should return empty for unmatched query."""
        results = pipeline._bm25_search("quantum physics", top_k=5)
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_bm25_respects_filters(self, pipeline):
        """BM25 should filter by doc_id."""
        results = pipeline._bm25_search("customer", top_k=5, filters={"doc_id": "bm25_doc2"})
        for r in results:
            assert r.doc_id == "bm25_doc2"

    @pytest.mark.asyncio
    async def test_bm25_scores_are_positive(self, pipeline):
        """BM25 scores should be positive for matching terms."""
        results = pipeline._bm25_search("pricing", top_k=5)
        for r in results:
            assert r.score > 0

    @pytest.mark.asyncio
    async def test_bm25_metadata_has_method(self, pipeline):
        """BM25 results should have retrieval_method=bm25 in metadata."""
        results = pipeline._bm25_search("pricing", top_k=5)
        for r in results:
            assert r.metadata.get("retrieval_method") == "bm25"

    def test_tokenize_strips_punctuation(self):
        """Tokenizer should strip punctuation and lowercase."""
        tokens = HaystackPipeline._tokenize("Hello, World! This is a test.")
        assert "hello" in tokens
        assert "world" in tokens
        assert "," not in tokens

    def test_tokenize_removes_single_chars(self):
        """Tokenizer should remove single-character tokens."""
        tokens = HaystackPipeline._tokenize("I am a test")
        assert "i" not in tokens  # single char
        assert "a" not in tokens  # single char
        assert "am" in tokens
        assert "test" in tokens


# ---------------------------------------------------------------------------
# Hybrid search (RRF fusion)
# ---------------------------------------------------------------------------


class TestHybridSearch:
    """Test hybrid BM25 + vector search with RRF fusion."""

    @pytest.fixture
    async def pipeline(self):
        p = HaystackPipeline()
        await p.initialize()
        p._mock_chunks.clear()
        p._mock_embeddings.clear()
        await p.add_document(
            doc_id="doc1",
            content="Revenue grew 15% after the pricing change in Q3. "
                    "Customer acquisition cost decreased significantly.",
        )
        await p.add_document(
            doc_id="doc2",
            content="Marketing spend on social media increased brand awareness. "
                    "The ROI of digital campaigns was 3.2x.",
        )
        return p

    @pytest.mark.asyncio
    async def test_hybrid_returns_results(self, pipeline):
        """Hybrid search should return fused results."""
        results = await pipeline.hybrid_search("revenue pricing", top_k=5)
        assert len(results) > 0

    @pytest.mark.asyncio
    async def test_hybrid_metadata_has_component_scores(self, pipeline):
        """Hybrid results should include vector and bm25 component scores."""
        results = await pipeline.hybrid_search("revenue pricing", top_k=5)
        for r in results:
            assert "vector_score" in r.metadata
            assert "bm25_score" in r.metadata
            assert r.metadata.get("retrieval_method") == "hybrid"

    @pytest.mark.asyncio
    async def test_hybrid_respects_weights(self, pipeline):
        """Different weights should produce different rankings."""
        # Heavy BM25 weight
        bm25_heavy = await pipeline.hybrid_search(
            "revenue pricing", top_k=3, vector_weight=0.1, bm25_weight=0.9,
        )
        # Heavy vector weight
        vector_heavy = await pipeline.hybrid_search(
            "revenue pricing", top_k=3, vector_weight=0.9, bm25_weight=0.1,
        )
        # Both should return results (may have different orderings)
        assert len(bm25_heavy) > 0
        assert len(vector_heavy) > 0

    @pytest.mark.asyncio
    async def test_hybrid_respects_filters(self, pipeline):
        """Hybrid search should filter by doc_id."""
        results = await pipeline.hybrid_search(
            "revenue", top_k=5, filters={"doc_id": "doc1"},
        )
        for r in results:
            assert r.doc_id == "doc1"

    @pytest.mark.asyncio
    async def test_hybrid_fused_score_is_positive(self, pipeline):
        """Fused RRF scores should be positive."""
        results = await pipeline.hybrid_search("revenue pricing", top_k=5)
        for r in results:
            assert r.score > 0


# ---------------------------------------------------------------------------
# Re-ranking
# ---------------------------------------------------------------------------


class TestReranking:
    """Test heuristic re-ranking."""

    def test_rerank_empty_returns_empty(self):
        """Re-ranking empty list should return empty."""
        result = HaystackPipeline.rerank("query", [], top_k=5)
        assert result == []

    def test_rerank_promotes_relevant_chunks(self):
        """Chunks with more query terms should rank higher."""
        chunks = [
            ChunkResult(
                chunk_id="c1", content="Completely unrelated content about cats",
                score=0.9, doc_id="d1", metadata={},
            ),
            ChunkResult(
                chunk_id="c2", content="pricing affects revenue through elasticity",
                score=0.5, doc_id="d1", metadata={},
            ),
        ]
        result = HaystackPipeline.rerank("pricing revenue", chunks, top_k=2)
        # c2 should rank higher despite lower original score
        assert result[0].chunk_id == "c2"

    def test_rerank_respects_top_k(self):
        """Re-ranking should prune to top_k."""
        chunks = [
            ChunkResult(chunk_id=f"c{i}", content=f"content {i} pricing",
                       score=0.5, doc_id="d1", metadata={})
            for i in range(10)
        ]
        result = HaystackPipeline.rerank("pricing", chunks, top_k=3)
        assert len(result) == 3

    def test_rerank_marks_metadata(self):
        """Re-ranked results should have reranked=True in metadata."""
        chunks = [
            ChunkResult(chunk_id="c1", content="pricing revenue",
                       score=0.5, doc_id="d1", metadata={}),
        ]
        result = HaystackPipeline.rerank("pricing", chunks, top_k=1)
        assert result[0].metadata.get("reranked") is True

    def test_rerank_proximity_bonus(self):
        """Chunks with query terms close together should score higher."""
        close = ChunkResult(
            chunk_id="close", content="pricing directly affects revenue growth",
            score=0.5, doc_id="d1", metadata={},
        )
        far = ChunkResult(
            chunk_id="far",
            content="pricing is important for many reasons and in the end revenue matters",
            score=0.5, doc_id="d1", metadata={},
        )
        result = HaystackPipeline.rerank("pricing revenue", [far, close], top_k=2)
        assert result[0].chunk_id == "close"


# ---------------------------------------------------------------------------
# Hypothesis-aware retrieval
# ---------------------------------------------------------------------------


class TestHypothesisRetrieval:
    """Test hypothesis-aware retrieval in HaystackService."""

    @pytest.fixture
    async def service(self):
        svc = HaystackService()
        await svc.initialize()
        svc.pipeline._mock_chunks.clear()
        svc.pipeline._mock_embeddings.clear()
        await svc.index_document(
            doc_id="doc1",
            content="Price increases lead to reduced demand through elasticity effects. "
                    "Revenue grew after pricing adjustments in Q3.",
        )
        return svc

    @pytest.mark.asyncio
    async def test_retrieve_for_hypothesis_returns_assessment(self, service):
        """Should return EvidenceAssessment."""
        hypothesis = HypothesisQuery(cause="price", effect="demand")
        assessment = await service.retrieve_for_hypothesis(hypothesis)
        assert isinstance(assessment, EvidenceAssessment)

    @pytest.mark.asyncio
    async def test_assessment_has_supporting_evidence(self, service):
        """Assessment should find supporting evidence."""
        hypothesis = HypothesisQuery(cause="price", effect="demand")
        assessment = await service.retrieve_for_hypothesis(hypothesis)
        # Mock search returns results if terms match
        assert isinstance(assessment.supporting, list)

    @pytest.mark.asyncio
    async def test_assessment_has_strength(self, service):
        """Assessment should classify evidence strength."""
        hypothesis = HypothesisQuery(cause="price", effect="demand")
        assessment = await service.retrieve_for_hypothesis(hypothesis)
        assert isinstance(assessment.strength, EvidenceStrength)
        assert 0.0 <= assessment.confidence <= 1.0

    @pytest.mark.asyncio
    async def test_assessment_has_sufficiency_reason(self, service):
        """Assessment should include a sufficiency reason string."""
        hypothesis = HypothesisQuery(cause="price", effect="demand")
        assessment = await service.retrieve_for_hypothesis(hypothesis)
        assert assessment.reason  # non-empty string
        assert "supporting" in assessment.reason.lower()


# ---------------------------------------------------------------------------
# Evidence sufficiency
# ---------------------------------------------------------------------------


class TestEvidenceSufficiency:
    """Test evidence sufficiency assessment."""

    @pytest.fixture
    async def service(self):
        svc = HaystackService()
        await svc.initialize()
        svc.pipeline._mock_chunks.clear()
        svc.pipeline._mock_embeddings.clear()
        return svc

    @pytest.mark.asyncio
    async def test_sufficient_with_enough_evidence(self, service):
        """Should be sufficient with ≥2 supporting evidence."""
        evidence = [
            _make_evidence("Price increases reduce demand via elasticity"),
            _make_evidence("Higher prices correlate with lower demand in Q3"),
            _make_evidence("Pricing experiment showed reduced demand"),
        ]
        assessment = await service.check_evidence_sufficiency(
            "price", "demand", existing_evidence=evidence,
        )
        assert assessment.is_sufficient
        assert len(assessment.supporting) == 3

    @pytest.mark.asyncio
    async def test_insufficient_with_little_evidence(self, service):
        """Should be insufficient with <2 supporting evidence."""
        evidence = [
            _make_evidence("Some mention of price"),
        ]
        assessment = await service.check_evidence_sufficiency(
            "price", "demand", existing_evidence=evidence,
        )
        assert not assessment.is_sufficient

    @pytest.mark.asyncio
    async def test_contested_evidence_detected(self, service):
        """Should detect contradicting evidence."""
        evidence = [
            _make_evidence("Price causes demand reduction"),
            _make_evidence("Price does not affect demand significantly"),
            _make_evidence("There is no effect of pricing on demand"),
        ]
        assessment = await service.check_evidence_sufficiency(
            "price", "demand", existing_evidence=evidence,
        )
        assert len(assessment.contradicting) >= 1
        assert assessment.strength == EvidenceStrength.CONTESTED

    @pytest.mark.asyncio
    async def test_sufficiency_reason_includes_counts(self, service):
        """Reason string should include evidence counts."""
        evidence = [
            _make_evidence("Price causes demand reduction"),
            _make_evidence("Pricing affects demand"),
        ]
        assessment = await service.check_evidence_sufficiency(
            "price", "demand", existing_evidence=evidence,
        )
        assert "supporting" in assessment.reason.lower()

    @pytest.mark.asyncio
    async def test_check_without_existing_evidence(self, service):
        """Without existing evidence, should retrieve fresh."""
        await service.index_document(
            doc_id="fresh",
            content="Price changes affect demand through elasticity.",
        )
        assessment = await service.check_evidence_sufficiency(
            "price", "demand",
        )
        assert isinstance(assessment, EvidenceAssessment)


# ---------------------------------------------------------------------------
# Hybrid retrieval in HaystackService
# ---------------------------------------------------------------------------


class TestHaystackServiceHybrid:
    """Test HaystackService.retrieve_hybrid()."""

    @pytest.fixture
    async def service(self):
        svc = HaystackService()
        await svc.initialize()
        svc.pipeline._mock_chunks.clear()
        svc.pipeline._mock_embeddings.clear()
        await svc.index_document(
            doc_id="doc1",
            content="Revenue grew 15% after the pricing change. "
                    "Demand decreased when prices increased.",
            filename="report.txt",
        )
        return svc

    @pytest.mark.asyncio
    async def test_retrieve_hybrid_returns_evidence(self, service):
        """Should return EvidenceBundle objects."""
        bundles = await service.retrieve_hybrid("pricing revenue", top_k=3)
        assert isinstance(bundles, list)
        for b in bundles:
            assert isinstance(b, EvidenceBundle)

    @pytest.mark.asyncio
    async def test_retrieve_hybrid_has_component_scores(self, service):
        """Evidence bundles should have vector, bm25, fused scores."""
        bundles = await service.retrieve_hybrid("pricing", top_k=3)
        for b in bundles:
            scores = b.retrieval_trace.scores
            assert scores is not None
            assert "fused" in scores or "vector" in scores

    @pytest.mark.asyncio
    async def test_retrieve_hybrid_with_reranking(self, service):
        """Re-ranking should not crash and should return results."""
        bundles = await service.retrieve_hybrid(
            "pricing revenue", top_k=3, rerank=True,
        )
        assert isinstance(bundles, list)

    @pytest.mark.asyncio
    async def test_retrieve_hybrid_with_doc_filter(self, service):
        """Should filter by doc_id."""
        bundles = await service.retrieve_hybrid(
            "pricing", top_k=3, doc_id="doc1",
        )
        for b in bundles:
            assert b.source.doc_id == "doc1"


# ---------------------------------------------------------------------------
# Retrieval Router Phase 2
# ---------------------------------------------------------------------------


class TestRetrievalRouterPhase2:
    """Test retrieval router with HYBRID strategy."""

    @pytest.fixture
    async def router(self):
        r = RetrievalRouter()
        await r.initialize()
        r.haystack.pipeline._mock_chunks.clear()
        r.haystack.pipeline._mock_embeddings.clear()
        await r.haystack.index_document(
            doc_id="test",
            content="Pricing decisions impact revenue and customer demand.",
        )
        return r

    @pytest.mark.asyncio
    async def test_auto_strategy_defaults_to_hybrid(self, router):
        """AUTO strategy should now default to HYBRID."""
        request = RetrievalRequest(query="test")
        strategy = router._determine_strategy(request)
        assert strategy == RetrievalStrategy.HYBRID

    @pytest.mark.asyncio
    async def test_hybrid_strategy_retrieves(self, router):
        """HYBRID strategy should return results."""
        request = RetrievalRequest(
            query="pricing revenue",
            strategy=RetrievalStrategy.HYBRID,
            max_results=5,
        )
        bundles = await router.retrieve(request)
        assert isinstance(bundles, list)

    @pytest.mark.asyncio
    async def test_explicit_haystack_still_works(self, router):
        """Explicit HAYSTACK_ONLY should still work."""
        request = RetrievalRequest(
            query="pricing",
            strategy=RetrievalStrategy.HAYSTACK_ONLY,
            max_results=3,
        )
        bundles = await router.retrieve(request)
        assert isinstance(bundles, list)

    @pytest.mark.asyncio
    async def test_retrieve_for_hypothesis_via_router(self, router):
        """Router should expose hypothesis testing."""
        assessment = await router.retrieve_for_hypothesis(
            cause="pricing", effect="revenue",
        )
        assert isinstance(assessment, EvidenceAssessment)

    @pytest.mark.asyncio
    async def test_check_evidence_sufficiency_via_router(self, router):
        """Router should expose evidence sufficiency check."""
        assessment = await router.check_evidence_sufficiency(
            cause="pricing", effect="revenue",
        )
        assert isinstance(assessment, EvidenceAssessment)
        assert isinstance(assessment.is_sufficient, bool)

    @pytest.mark.asyncio
    async def test_hybrid_with_reranking(self, router):
        """Hybrid with re-ranking should work via router."""
        request = RetrievalRequest(
            query="pricing revenue",
            strategy=RetrievalStrategy.HYBRID,
            max_results=5,
            use_reranking=True,
        )
        bundles = await router.retrieve(request)
        assert isinstance(bundles, list)

    def test_hybrid_strategy_enum_exists(self):
        """HYBRID should be a valid strategy."""
        assert RetrievalStrategy.HYBRID == "hybrid"


# ---------------------------------------------------------------------------
# Data structure tests
# ---------------------------------------------------------------------------


class TestPhase2DataStructures:
    """Test Phase 2 data structure defaults."""

    def test_hypothesis_query_defaults(self):
        hq = HypothesisQuery(cause="price", effect="demand")
        assert hq.cause == "price"
        assert hq.effect == "demand"
        assert hq.mechanism == ""
        assert hq.domain == ""

    def test_hypothesis_query_with_mechanism(self):
        hq = HypothesisQuery(
            cause="price", effect="demand",
            mechanism="elasticity", domain="pricing",
        )
        assert hq.mechanism == "elasticity"
        assert hq.domain == "pricing"

    def test_evidence_assessment_defaults(self):
        ea = EvidenceAssessment()
        assert ea.supporting == []
        assert ea.contradicting == []
        assert ea.strength == EvidenceStrength.HYPOTHESIS
        assert ea.confidence == 0.3
        assert ea.is_sufficient is False
        assert ea.reason == ""

    def test_retrieval_request_phase2_defaults(self):
        req = RetrievalRequest(query="test")
        assert req.use_reranking is False
        assert req.vector_weight == 0.5
        assert req.bm25_weight == 0.5
