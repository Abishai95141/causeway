"""
Tests for Modules 4-6: Retrieval Infrastructure

Tests cover:
- PageIndex client (mock mode)
- PageIndex service (evidence conversion)
- Haystack pipeline (mock mode)
- Haystack service (evidence conversion)
- Retrieval router (strategy and deduplication)
"""

import pytest
from datetime import datetime

from src.pageindex.client import PageIndexClient, PageIndexDocument, PageIndexSection
from src.pageindex.service import PageIndexService
from src.haystack_svc.pipeline import HaystackPipeline, ChunkResult
from src.haystack_svc.service import HaystackService
from src.retrieval.router import RetrievalRouter, RetrievalRequest, RetrievalStrategy
from src.models.enums import RetrievalMethod


class TestPageIndexClient:
    """Test PageIndex client in mock mode."""
    
    @pytest.fixture
    def client(self):
        """Create client in mock mode (no API key)."""
        return PageIndexClient(api_key=None)
    
    def test_mock_mode_enabled(self, client):
        """Client should be in mock mode without API key."""
        assert client.is_mock_mode
    
    @pytest.mark.asyncio
    async def test_register_document(self, client):
        """Should register document and return ID."""
        result = await client.register_document(
            doc_id="test_123",
            filename="test.pdf",
            content=b"Test document content here",
            content_type="application/pdf",
        )
        
        assert isinstance(result, PageIndexDocument)
        assert result.doc_id.startswith("pi_")
        assert result.filename == "test.pdf"
        assert result.page_count >= 1
    
    @pytest.mark.asyncio
    async def test_list_sections(self, client):
        """Should list sections from document."""
        # Register first
        doc = await client.register_document(
            doc_id="test",
            filename="test.md",
            content=b"# Introduction\n\nSome content\n\n# Methods\n\nMore content",
            content_type="text/markdown",
        )
        
        sections = await client.list_sections(doc.doc_id)
        assert isinstance(sections, list)
    
    @pytest.mark.asyncio
    async def test_search(self, client):
        """Should search and return sections."""
        doc = await client.register_document(
            doc_id="test",
            filename="test.txt",
            content=b"Content about pricing decisions",
            content_type="text/plain",
        )
        
        results = await client.search(doc.doc_id, "pricing", max_results=3)
        assert isinstance(results, list)
    
    @pytest.mark.asyncio
    async def test_read_page(self, client):
        """Should read page content."""
        doc = await client.register_document(
            doc_id="test",
            filename="test.txt",
            content=b"Page content here",
            content_type="text/plain",
        )
        
        content = await client.read_page(doc.doc_id, 1)
        assert isinstance(content, str)


class TestPageIndexService:
    """Test PageIndex service."""
    
    @pytest.fixture
    def service(self):
        """Create service with mock client."""
        return PageIndexService()
    
    @pytest.mark.asyncio
    async def test_retrieve_evidence(self, service):
        """Should return EvidenceBundle objects."""
        # Register a document first
        doc_id = await service.register_document(
            doc_id="test",
            filename="policy.pdf",
            content=b"Pricing policy content",
            content_type="application/pdf",
        )
        
        bundles = await service.retrieve_evidence(
            doc_id=doc_id,
            doc_title="Pricing Policy",
            query="price",
            max_results=3,
        )
        
        assert isinstance(bundles, list)
        for bundle in bundles:
            assert bundle.retrieval_trace.method == RetrievalMethod.PAGEINDEX
            assert bundle.source.doc_title == "Pricing Policy"


class TestHaystackPipeline:
    """Test Haystack pipeline in mock mode."""
    
    @pytest.fixture
    async def pipeline(self):
        """Create pipeline and clean up afterwards."""
        p = HaystackPipeline()
        await p.initialize()
        yield p
        # Cleanup: delete any test documents left behind
        for doc_id in ("doc1", "to_delete", "test"):
            try:
                await p.delete_document(doc_id)
            except Exception:
                pass
    
    @pytest.mark.asyncio
    async def test_initialize_mock_mode(self, pipeline):
        """Should initialize without error."""
        assert pipeline._initialized
    
    @pytest.mark.asyncio
    async def test_add_document_creates_chunks(self, pipeline):
        """Should chunk document and store."""
        chunk_ids = await pipeline.add_document(
            doc_id="doc1",
            content="This is test content. " * 100,  # ~2000 chars
            metadata={"filename": "test.txt"},
        )
        
        assert len(chunk_ids) > 0
        assert all(cid.startswith("doc1_chunk_") for cid in chunk_ids)
    
    @pytest.mark.asyncio
    async def test_search_returns_results(self, pipeline):
        """Should search and return ranked chunks."""
        await pipeline.add_document(
            doc_id="doc1",
            content="Pricing decisions affect revenue significantly. "
                    "Customer churn increases with price hikes.",
        )
        
        results = await pipeline.search("pricing", top_k=3,
                                        filters={"doc_id": "doc1"})
        
        assert isinstance(results, list)
        assert len(results) > 0
        for result in results:
            assert isinstance(result, ChunkResult)
            assert result.doc_id == "doc1"
    
    @pytest.mark.asyncio
    async def test_delete_document(self, pipeline):
        """Should delete all chunks for a document."""
        await pipeline.add_document(
            doc_id="to_delete",
            content="This is test content for deletion. It should be removed afterwards.",
        )
        
        # Verify document was indexed
        results_before = await pipeline.search("test deletion", top_k=5,
                                               filters={"doc_id": "to_delete"})
        assert len(results_before) > 0
        
        await pipeline.delete_document("to_delete")
        
        # After deletion, search scoped to that doc should return nothing
        remaining = await pipeline.search("test deletion", top_k=5,
                                          filters={"doc_id": "to_delete"})
        assert len(remaining) == 0


class TestHaystackService:
    """Test Haystack service."""
    
    @pytest.fixture
    async def service(self):
        """Create service and clean up afterwards."""
        svc = HaystackService()
        await svc.initialize()
        yield svc
        # Cleanup
        for doc_id in ("doc1", "svc_test"):
            try:
                await svc.pipeline.delete_document(doc_id)
            except Exception:
                pass
    
    @pytest.mark.asyncio
    async def test_retrieve_evidence(self, service):
        """Should return EvidenceBundle objects."""
        await service.index_document(
            doc_id="doc1",
            content="Revenue grew 15% after the pricing change.",
            filename="report.txt",
        )
        
        bundles = await service.retrieve_evidence(
            query="revenue pricing",
            doc_id="doc1",
            doc_title="Q3 Report",
        )
        
        assert isinstance(bundles, list)
        for bundle in bundles:
            assert bundle.retrieval_trace.method == RetrievalMethod.HAYSTACK
            assert bundle.retrieval_trace.scores is not None


class TestRetrievalRouter:
    """Test retrieval router."""
    
    @pytest.fixture
    async def router(self):
        """Create router and clean up afterwards."""
        r = RetrievalRouter()
        await r.initialize()
        yield r
        # Cleanup test documents
        for doc_id in ("test", "test_doc"):
            try:
                await r.haystack.pipeline.delete_document(doc_id)
            except Exception:
                pass
    
    @pytest.mark.asyncio
    async def test_retrieve_simple(self, router):
        """Simple retrieval should work."""
        # Index a document first
        await router.haystack.index_document(
            doc_id="test",
            content="Test content about pricing and revenue",
        )
        
        bundles = await router.retrieve_simple("pricing", max_results=3)
        assert isinstance(bundles, list)
    
    @pytest.mark.asyncio
    async def test_strategy_selection(self, router):
        """Should select appropriate strategy."""
        # Provenance required -> PageIndex
        request = RetrievalRequest(
            query="test",
            require_provenance=True,
        )
        strategy = router._determine_strategy(request)
        assert strategy == RetrievalStrategy.PAGEINDEX_ONLY
        
        # Default -> Hybrid (Phase 2: upgraded from HAYSTACK_ONLY)
        request = RetrievalRequest(query="test")
        strategy = router._determine_strategy(request)
        assert strategy == RetrievalStrategy.HYBRID
        
        # With doc_ids -> Both
        request = RetrievalRequest(query="test", doc_ids=["doc1"])
        strategy = router._determine_strategy(request)
        assert strategy == RetrievalStrategy.BOTH_MERGE
    
    @pytest.mark.asyncio
    async def test_deduplication(self, router):
        """Should deduplicate by content hash."""
        from src.models.evidence import EvidenceBundle, SourceReference, RetrievalTrace
        
        bundles = [
            EvidenceBundle(
                content="Same content",
                source=SourceReference(doc_id="doc1", doc_title="Test"),
                retrieval_trace=RetrievalTrace(
                    method=RetrievalMethod.HAYSTACK,
                    query="test",
                ),
            ),
            EvidenceBundle(
                content="Same content",  # Duplicate
                source=SourceReference(doc_id="doc2", doc_title="Test2"),
                retrieval_trace=RetrievalTrace(
                    method=RetrievalMethod.PAGEINDEX,
                    query="test",
                ),
            ),
            EvidenceBundle(
                content="Different content",
                source=SourceReference(doc_id="doc3", doc_title="Test3"),
                retrieval_trace=RetrievalTrace(
                    method=RetrievalMethod.HAYSTACK,
                    query="test",
                ),
            ),
        ]
        
        deduped = router._deduplicate(bundles, max_results=10)
        assert len(deduped) == 2  # "Same content" deduplicated
    
    @pytest.mark.asyncio
    async def test_ingest_document(self, router):
        """Should ingest into both systems."""
        
        result = await router.ingest_document(
            doc_id="test_doc",
            filename="test.txt",
            content=b"Test content for ingestion",
            content_type="text/plain",
        )
        
        assert "pageindex_doc_id" in result
        assert "haystack_chunk_ids" in result
        assert result["pageindex_doc_id"].startswith("pi_")
