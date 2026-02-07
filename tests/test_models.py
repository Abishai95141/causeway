"""
Tests for Module 1: Data Models & Schema Layer

Tests validate:
- Model serialization/deserialization
- Enum validation
- Evidence bundle required fields + hash formatting
- Validation rules
"""

import pytest
from datetime import datetime
from uuid import uuid4

from src.models import (
    # Enums
    IngestionStatus,
    EvidenceStrength,
    ModelStatus,
    RetrievalMethod,
    VariableType,
    MeasurementStatus,
    ConfidenceLevel,
    OperatingMode,
    # Documents
    DocumentRecord,
    # Evidence
    SourceReference,
    LocationMetadata,
    RetrievalTrace,
    EvidenceBundle,
    # Causal
    VariableDefinition,
    EdgeMetadata,
    CausalEdge,
    CausalPath,
    WorldModelVersion,
    # Decision
    DecisionQuery,
    DecisionRecommendation,
    EscalationResponse,
    # Audit
    AuditEntry,
)


class TestEnums:
    """Test all enumeration types."""
    
    def test_ingestion_status_values(self):
        """Verify IngestionStatus enum values."""
        assert IngestionStatus.PENDING.value == "pending"
        assert IngestionStatus.INDEXING.value == "indexing"
        assert IngestionStatus.INDEXED.value == "indexed"
        assert IngestionStatus.FAILED.value == "failed"
    
    def test_evidence_strength_values(self):
        """Verify EvidenceStrength enum values."""
        assert EvidenceStrength.STRONG.value == "strong"
        assert EvidenceStrength.MODERATE.value == "moderate"
        assert EvidenceStrength.HYPOTHESIS.value == "hypothesis"
        assert EvidenceStrength.CONTESTED.value == "contested"
    
    def test_model_status_values(self):
        """Verify ModelStatus enum values."""
        assert ModelStatus.DRAFT.value == "draft"
        assert ModelStatus.REVIEW.value == "review"
        assert ModelStatus.ACTIVE.value == "active"
        assert ModelStatus.DEPRECATED.value == "deprecated"
    
    def test_retrieval_method_values(self):
        """Verify RetrievalMethod enum values."""
        assert RetrievalMethod.PAGEINDEX.value == "pageindex"
        assert RetrievalMethod.HAYSTACK.value == "haystack"
        assert RetrievalMethod.BOTH.value == "both"
    
    def test_operating_mode_values(self):
        """Verify OperatingMode enum values."""
        assert OperatingMode.MODE_1.value == "world_model_construction"
        assert OperatingMode.MODE_2.value == "decision_support"


class TestDocumentRecord:
    """Test DocumentRecord model."""
    
    def test_create_valid_document(self):
        """Create a valid DocumentRecord."""
        doc = DocumentRecord(
            filename="test.pdf",
            content_type="application/pdf",
            size_bytes=1024,
            sha256="a" * 64,
            storage_uri="s3://bucket/test.pdf"
        )
        assert doc.filename == "test.pdf"
        assert doc.ingestion_status == IngestionStatus.PENDING  # default
        assert doc.pageindex_doc_id is None
        assert doc.haystack_doc_ids == []
    
    def test_sha256_validation_lowercase(self):
        """SHA256 should be normalized to lowercase."""
        doc = DocumentRecord(
            filename="test.pdf",
            content_type="application/pdf",
            size_bytes=1024,
            sha256="A" * 64,  # uppercase
            storage_uri="s3://bucket/test.pdf"
        )
        assert doc.sha256 == "a" * 64  # lowercase
    
    def test_sha256_validation_invalid_chars(self):
        """Invalid SHA256 characters should raise error."""
        with pytest.raises(ValueError, match="hexadecimal"):
            DocumentRecord(
                filename="test.pdf",
                content_type="application/pdf",
                size_bytes=1024,
                sha256="g" * 64,  # 'g' is not hex
                storage_uri="s3://bucket/test.pdf"
            )
    
    def test_unsupported_content_type(self):
        """Unsupported content types should raise error."""
        with pytest.raises(ValueError, match="Unsupported content type"):
            DocumentRecord(
                filename="test.doc",
                content_type="application/msword",  # not supported
                size_bytes=1024,
                sha256="a" * 64,
                storage_uri="s3://bucket/test.doc"
            )
    
    def test_supported_content_types(self):
        """All prototype content types should work."""
        supported = [
            ("application/pdf", "test.pdf"),
            ("text/plain", "test.txt"),
            ("text/markdown", "test.md"),
            ("application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "test.xlsx"),
        ]
        for content_type, filename in supported:
            doc = DocumentRecord(
                filename=filename,
                content_type=content_type,
                size_bytes=1024,
                sha256="a" * 64,
                storage_uri=f"s3://bucket/{filename}"
            )
            assert doc.content_type == content_type
    
    def test_serialization_roundtrip(self):
        """Model should serialize and deserialize correctly."""
        doc = DocumentRecord(
            filename="test.pdf",
            content_type="application/pdf",
            size_bytes=1024,
            sha256="a" * 64,
            storage_uri="s3://bucket/test.pdf"
        )
        json_str = doc.model_dump_json()
        restored = DocumentRecord.model_validate_json(json_str)
        assert restored.filename == doc.filename
        assert restored.doc_id == doc.doc_id


class TestEvidenceModels:
    """Test evidence-related models."""
    
    def test_source_reference_creation(self):
        """Create a valid SourceReference."""
        source = SourceReference(
            doc_id="doc_123",
            doc_title="Pricing Policy v7"
        )
        assert source.doc_id == "doc_123"
        assert source.url is None
    
    def test_location_metadata_optional_fields(self):
        """LocationMetadata should allow all optional fields."""
        loc = LocationMetadata()
        assert loc.section_name is None
        assert loc.page_number is None
        
        loc_with_page = LocationMetadata(page_number=5, section_name="Overview")
        assert loc_with_page.page_number == 5
    
    def test_retrieval_trace_creation(self):
        """Create a valid RetrievalTrace."""
        trace = RetrievalTrace(
            method=RetrievalMethod.HAYSTACK,
            query="price elasticity"
        )
        assert trace.method == RetrievalMethod.HAYSTACK
        assert trace.timestamp is not None
    
    def test_evidence_bundle_content_hash(self):
        """EvidenceBundle should compute content hash."""
        bundle = EvidenceBundle(
            content="Test content for hashing",
            source=SourceReference(doc_id="doc_1", doc_title="Test Doc"),
            retrieval_trace=RetrievalTrace(
                method=RetrievalMethod.PAGEINDEX,
                query="test query"
            )
        )
        # Hash should be 64 char hex string
        assert len(bundle.content_hash) == 64
        assert all(c in "0123456789abcdef" for c in bundle.content_hash)
    
    def test_evidence_bundle_hash_consistency(self):
        """Same content should produce same hash."""
        trace = RetrievalTrace(method=RetrievalMethod.HAYSTACK, query="test")
        source = SourceReference(doc_id="doc_1", doc_title="Test")
        
        bundle1 = EvidenceBundle(
            content="Same content here",
            source=source,
            retrieval_trace=trace
        )
        bundle2 = EvidenceBundle(
            content="Same content here",
            source=source,
            retrieval_trace=trace
        )
        assert bundle1.content_hash == bundle2.content_hash
    
    def test_evidence_bundle_whitespace_normalization(self):
        """Content whitespace should be normalized."""
        bundle = EvidenceBundle(
            content="  Multiple   spaces   here  ",
            source=SourceReference(doc_id="doc_1", doc_title="Test"),
            retrieval_trace=RetrievalTrace(method=RetrievalMethod.HAYSTACK, query="test")
        )
        assert bundle.content == "Multiple spaces here"
    
    def test_evidence_bundle_matches_hash(self):
        """matches_hash method should work correctly."""
        bundle = EvidenceBundle(
            content="Test content",
            source=SourceReference(doc_id="doc_1", doc_title="Test"),
            retrieval_trace=RetrievalTrace(method=RetrievalMethod.HAYSTACK, query="test")
        )
        assert bundle.matches_hash(bundle.content_hash)
        assert bundle.matches_hash(bundle.content_hash.upper())  # case insensitive
        assert not bundle.matches_hash("different" + "0" * 56)
    
    def test_evidence_bundle_serialization(self):
        """EvidenceBundle should serialize with computed hash."""
        bundle = EvidenceBundle(
            content="Test content",
            source=SourceReference(doc_id="doc_1", doc_title="Test"),
            retrieval_trace=RetrievalTrace(method=RetrievalMethod.HAYSTACK, query="test")
        )
        data = bundle.model_dump()
        assert "content_hash" in data
        assert len(data["content_hash"]) == 64


class TestCausalModels:
    """Test causal models."""
    
    def test_variable_definition_snake_case_validation(self):
        """variable_id must be snake_case."""
        var = VariableDefinition(
            variable_id="price_elasticity",
            name="Price Elasticity",
            definition="Measure of demand sensitivity to price changes",
            type=VariableType.CONTINUOUS,
            measurement_status=MeasurementStatus.MEASURED
        )
        assert var.variable_id == "price_elasticity"
        
        with pytest.raises(ValueError, match="snake_case"):
            VariableDefinition(
                variable_id="PriceElasticity",  # camelCase not allowed
                name="Price Elasticity",
                definition="Test",
                type=VariableType.CONTINUOUS,
                measurement_status=MeasurementStatus.MEASURED
            )
    
    def test_edge_metadata_update_strength(self):
        """EdgeMetadata.update_strength_from_evidence_count should work."""
        meta = EdgeMetadata(mechanism="Direct effect")
        
        meta.update_strength_from_evidence_count(supporting=3)
        assert meta.evidence_strength == EvidenceStrength.STRONG
        
        meta.update_strength_from_evidence_count(supporting=2)
        assert meta.evidence_strength == EvidenceStrength.MODERATE
        
        meta.update_strength_from_evidence_count(supporting=1)
        assert meta.evidence_strength == EvidenceStrength.HYPOTHESIS
        
        meta.update_strength_from_evidence_count(supporting=5, contradicting=1)
        assert meta.evidence_strength == EvidenceStrength.CONTESTED
    
    def test_causal_edge_creation(self):
        """Create a valid CausalEdge."""
        edge = CausalEdge(
            from_var="price",
            to_var="revenue",
            metadata=EdgeMetadata(mechanism="Higher price increases revenue per unit")
        )
        assert edge.edge_id == "price->revenue"
    
    def test_causal_path_mediators(self):
        """CausalPath.mediators should return middle variables."""
        path = CausalPath(
            path=["price", "churn", "revenue"],
            edges=[],
            mechanism_chain="price increases churn which decreases revenue",
            strength="moderate"
        )
        assert path.mediators == ["churn"]
        assert path.length == 2
        
        direct_path = CausalPath(
            path=["price", "revenue"],
            edges=[],
            mechanism_chain="direct effect",
            strength="strong"
        )
        assert direct_path.mediators == []
    
    def test_world_model_version_validation(self):
        """WorldModelVersion version_id must start with 'wm_'."""
        wm = WorldModelVersion(
            version_id="wm_pricing_20260206_120000",
            domain="pricing",
            description="Test world model"
        )
        assert wm.version_id.startswith("wm_")
        
        with pytest.raises(ValueError, match="wm_"):
            WorldModelVersion(
                version_id="pricing_20260206",  # missing prefix
                domain="pricing",
                description="Test"
            )
    
    def test_world_model_get_all_evidence_refs(self):
        """get_all_evidence_refs should collect all edge evidence."""
        id1, id2, id3 = uuid4(), uuid4(), uuid4()
        
        wm = WorldModelVersion(
            version_id="wm_test_123",
            domain="test",
            description="Test",
            edges=[
                CausalEdge(
                    from_var="a", to_var="b",
                    metadata=EdgeMetadata(
                        mechanism="test",
                        evidence_refs=[id1, id2]
                    )
                ),
                CausalEdge(
                    from_var="b", to_var="c",
                    metadata=EdgeMetadata(
                        mechanism="test",
                        evidence_refs=[id2, id3]  # id2 is duplicate
                    )
                ),
            ]
        )
        refs = wm.get_all_evidence_refs()
        assert refs == {id1, id2, id3}  # deduplicated


class TestDecisionModels:
    """Test decision query and recommendation models."""
    
    def test_decision_query_creation(self):
        """Create a valid DecisionQuery."""
        query = DecisionQuery(
            text="Should we raise prices 10%?",
            objective="Maximize revenue",
            levers=["price"],
            constraints=["Market conditions"]
        )
        assert query.text == "Should we raise prices 10%?"
        assert query.query_id is not None
    
    def test_decision_recommendation_creation(self):
        """Create a valid DecisionRecommendation."""
        rec = DecisionRecommendation(
            recommendation="Hold prices steady",
            confidence=ConfidenceLevel.MEDIUM,
            expected_outcome="Maintain market share",
            risks=["Competitor may gain advantage"]
        )
        assert rec.recommendation == "Hold prices steady"
        assert rec.evidence_refs == []
    
    def test_escalation_response_creation(self):
        """Create a valid EscalationResponse."""
        esc = EscalationResponse(
            reason="Model stale - not updated in 45 days",
            suggested_mode1_scope={"domain": "pricing", "outcomes": ["revenue"]}
        )
        assert "World model update" in esc.message


class TestAuditEntry:
    """Test audit log entry model."""
    
    def test_audit_entry_creation(self):
        """Create a valid AuditEntry."""
        entry = AuditEntry(
            mode=OperatingMode.MODE_1,
            trace_id="trace_abc123",
            input_query="Build pricing model",
            output_type="WorldModelVersion",
            output_id="wm_pricing_20260206"
        )
        assert entry.mode == OperatingMode.MODE_1
        assert entry.success is True  # default
        assert entry.audit_id is not None
    
    def test_audit_entry_serialization(self):
        """AuditEntry should serialize correctly."""
        entry = AuditEntry(
            mode=OperatingMode.MODE_2,
            trace_id="trace_xyz",
            input_query="Should we raise prices?",
            output_type="DecisionRecommendation",
            execution_time_ms=5000
        )
        data = entry.model_dump()
        assert data["mode"] == "decision_support"
        assert data["execution_time_ms"] == 5000
        
        # Roundtrip
        json_str = entry.model_dump_json()
        restored = AuditEntry.model_validate_json(json_str)
        assert restored.trace_id == entry.trace_id
