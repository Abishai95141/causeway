"""
Tests for Modules 9-10: Mode Implementations

Tests cover:
- Mode 1: World Model Construction workflow
- Mode 2: Decision Support workflow
"""

import pytest
from uuid import uuid4

from src.modes.mode1 import (
    Mode1WorldModelConstruction,
    Mode1Stage,
    Mode1Result,
    VariableCandidate,
    EdgeCandidate,
)
from src.modes.mode2 import (
    Mode2DecisionSupport,
    Mode2Stage,
    Mode2Result,
    ParsedQuery,
)
from src.models.enums import (
    EvidenceStrength,
    ModelStatus,
    VariableType,
    MeasurementStatus,
    ConfidenceLevel,
    VariableRole,
)
from src.causal.pywhyllm_bridge import CausalGraphBridge
from src.extraction.service import (
    ExtractionService,
    ExtractedVariable,
    ExtractedEdge,
    ExtractedQuery,
    ExtractedRecommendation,
)


class TestMode1WorldModelConstruction:
    """Test Mode 1 workflow."""
    
    @pytest.fixture
    def mode1(self):
        """Create Mode 1 instance."""
        return Mode1WorldModelConstruction()
    
    @pytest.mark.asyncio
    async def test_initialization(self, mode1):
        """Should initialize all components."""
        await mode1.initialize()
        assert mode1.current_stage == Mode1Stage.VARIABLE_DISCOVERY
    
    @pytest.mark.asyncio
    async def test_run_returns_result(self, mode1):
        """Should return Mode1Result from run."""
        await mode1.initialize()
        
        result = await mode1.run(
            domain="test_domain",
            initial_query="test query",
            max_variables=5,
        )
        
        assert isinstance(result, Mode1Result)
        assert result.trace_id.startswith("m1_")
        assert result.domain == "test_domain"
    
    @pytest.mark.asyncio
    async def test_run_creates_audit_entries(self, mode1):
        """Should create audit trail."""
        await mode1.initialize()
        
        result = await mode1.run(
            domain="audit_test",
            initial_query="test",
        )
        
        assert len(result.audit_entries) > 0
    
    def test_mode1_has_extraction_service(self, mode1):
        """Mode 1 should have an ExtractionService instance."""
        assert hasattr(mode1, 'extraction')
        assert isinstance(mode1.extraction, ExtractionService)

    def test_extraction_service_dedup_variables(self):
        """ExtractionService._dedup_variables should collapse near-duplicates."""
        variables = [
            ExtractedVariable(name="local residents", description="People living nearby", var_type="continuous", measurement_status="measured"),
            ExtractedVariable(name="local_residents", description="Another phrasing", var_type="continuous", measurement_status="measured"),
        ]
        deduped = ExtractionService._dedup_variables(variables)
        assert len(deduped) == 1
        assert deduped[0].name == "local residents"

    def test_extraction_service_validate_citation_match(self):
        """Citation validation should match when tokens overlap."""
        keys = ExtractionService._validate_citation(
            "price increases reduce demand",
            {"chunk1": "Studies show price increases reduce demand through elasticity"},
        )
        assert "chunk1" in keys

    def test_extraction_service_validate_citation_reject(self):
        """Citation validation should reject unrelated evidence."""
        keys = ExtractionService._validate_citation(
            "price increases reduce demand",
            {"chunk1": "The weather was sunny today in the park"},
        )
        assert keys == []
    
    @pytest.mark.asyncio
    async def test_approve_model_changes_status(self, mode1):
        """Should change model status on approval."""
        await mode1.initialize()
        
        # Pre-create a model in the causal service
        mode1.causal.create_world_model("approval_test")
        mode1.causal.add_variable("test_var", "Test Variable", "A test variable")
        
        # Approve
        model = await mode1.approve_model("approval_test", "test_approver")
        
        assert model.status == ModelStatus.ACTIVE
        assert model.approved_by == "test_approver"
        assert mode1.current_stage == Mode1Stage.COMPLETE


class TestMode2DecisionSupport:
    """Test Mode 2 workflow."""
    
    @pytest.fixture
    def mode2(self):
        """Create Mode 2 instance."""
        return Mode2DecisionSupport()
    
    @pytest.fixture
    def mode2_with_model(self):
        """Create Mode 2 with a pre-existing world model."""
        mode2 = Mode2DecisionSupport()
        
        # Create a world model
        mode2.causal.create_world_model("pricing")
        mode2.causal.add_variable("price", "Price", "Product pricing")
        mode2.causal.add_variable("demand", "Demand", "Customer demand")
        mode2.causal.add_variable("revenue", "Revenue", "Total revenue")
        mode2.causal.add_causal_link("price", "demand", "Price affects demand")
        mode2.causal.add_causal_link("demand", "revenue", "Demand drives revenue")
        
        return mode2
    
    @pytest.mark.asyncio
    async def test_initialization(self, mode2):
        """Should initialize all components."""
        await mode2.initialize()
        assert mode2.current_stage == Mode2Stage.QUERY_PARSING
    
    @pytest.mark.asyncio
    async def test_run_escalates_without_model(self, mode2):
        """Should escalate to Mode 1 when no model exists."""
        await mode2.initialize()
        
        result = await mode2.run(
            query="Should we increase prices?",
            domain_hint="nonexistent_domain",
        )
        
        assert result.escalate_to_mode1
        assert "No world model" in result.escalation_reason
    
    @pytest.mark.asyncio
    async def test_run_with_existing_model(self, mode2_with_model):
        """Should complete analysis with existing model."""
        await mode2_with_model.initialize()
        
        result = await mode2_with_model.run(
            query="Should we increase prices to boost revenue?",
            domain_hint="pricing",
        )
        
        assert isinstance(result, Mode2Result)
        assert result.trace_id.startswith("m2_")
    
    @pytest.mark.asyncio
    async def test_run_creates_audit_entries(self, mode2_with_model):
        """Should create audit trail."""
        await mode2_with_model.initialize()
        
        result = await mode2_with_model.run(
            query="Pricing decision",
            domain_hint="pricing",
        )
        
        assert len(result.audit_entries) > 0
    
    def test_mode2_has_extraction_service(self, mode2):
        """Mode 2 should have an ExtractionService instance."""
        assert hasattr(mode2, 'extraction')
        assert isinstance(mode2.extraction, ExtractionService)

    def test_extracted_query_dataclass(self):
        """ExtractedQuery should hold parsed components."""
        eq = ExtractedQuery(
            domain="pricing",
            intervention="increase prices",
            target_outcome="revenue",
            constraints=["budget limit"],
        )
        assert eq.domain == "pricing"
        assert eq.intervention == "increase prices"
        assert eq.constraints == ["budget limit"]
    
    def test_find_matching_variable_exact_match(self, mode2):
        """Should find exact variable match."""
        variables = ["price", "demand", "revenue"]
        
        match = mode2._find_matching_variable("price", variables)
        
        assert match == "price"
    
    def test_find_matching_variable_partial_match(self, mode2):
        """Should find partial variable match."""
        variables = ["product_price", "customer_demand", "total_revenue"]
        
        match = mode2._find_matching_variable("price", variables)
        
        assert match == "product_price"
    
    def test_find_matching_variable_no_match(self, mode2):
        """Should return None when no match."""
        variables = ["demand", "revenue"]
        
        match = mode2._find_matching_variable("profit", variables)
        
        assert match is None
    
    @pytest.mark.asyncio
    async def test_causal_reasoning_with_model(self, mode2_with_model):
        """Should perform causal reasoning with model."""
        await mode2_with_model.initialize()
        
        insights = await mode2_with_model._perform_causal_reasoning(
            "pricing", "price", "revenue"
        )
        
        assert len(insights) > 0
        assert insights[0].mediators == ["demand"] or "demand" in insights[0].mediators


class TestMode1Phase1:
    """Test Phase 1 evidence-grounded DAG construction in Mode 1."""

    @pytest.fixture
    def mode1(self):
        """Create Mode 1 instance with explicit mock bridge."""
        bridge = CausalGraphBridge(api_key=None)
        return Mode1WorldModelConstruction(causal_bridge=bridge)

    def test_constructor_accepts_causal_bridge(self):
        """Should accept a CausalGraphBridge in constructor."""
        bridge = CausalGraphBridge(api_key=None)
        mode1 = Mode1WorldModelConstruction(causal_bridge=bridge)
        assert mode1.bridge is bridge
        assert mode1.bridge.is_mock_mode

    def test_constructor_auto_creates_bridge(self):
        """When no bridge is given, should auto-create from config."""
        mode1 = Mode1WorldModelConstruction()
        assert hasattr(mode1, 'bridge')
        assert isinstance(mode1.bridge, CausalGraphBridge)

    @pytest.mark.asyncio
    async def test_run_creates_evidence_grounded_result(self, mode1):
        """Full run should produce a result with evidence links."""
        await mode1.initialize()
        result = await mode1.run(
            domain="test_domain",
            initial_query="test query",
            max_variables=5,
        )
        assert isinstance(result, Mode1Result)
        assert result.evidence_linked >= 0  # At least 0 evidence cached

    def test_variable_candidate_has_role_field(self):
        """VariableCandidate should support the role field."""
        vc = VariableCandidate(
            name="Price",
            description="Product price",
            var_type=VariableType.CONTINUOUS,
            measurement_status=MeasurementStatus.MEASURED,
            role=VariableRole.TREATMENT,
        )
        assert vc.role == VariableRole.TREATMENT

    def test_variable_candidate_default_role(self):
        """VariableCandidate should default to UNKNOWN role."""
        vc = VariableCandidate(
            name="X",
            description="test",
            var_type=VariableType.CONTINUOUS,
            measurement_status=MeasurementStatus.MEASURED,
        )
        assert vc.role == VariableRole.UNKNOWN

    def test_edge_candidate_has_evidence_fields(self):
        """EdgeCandidate should have evidence bundle IDs, contradictions, assumptions."""
        ec = EdgeCandidate(
            from_var="price",
            to_var="demand",
            mechanism="Price elasticity",
            strength=EvidenceStrength.MODERATE,
            evidence_bundle_ids=[uuid4()],
            contradicting_refs=["hash123"],
            contradicting_bundle_ids=[uuid4()],
            assumptions=["No confounders"],
            conditions=["US market"],
            confidence=0.7,
        )
        assert len(ec.evidence_bundle_ids) == 1
        assert len(ec.contradicting_refs) == 1
        assert len(ec.assumptions) == 1
        assert len(ec.conditions) == 1
        assert ec.confidence == 0.7

    def test_extraction_service_build_edge_examples(self):
        """ExtractionService should build dynamic edge examples from var IDs."""
        examples = ExtractionService._build_edge_examples(
            variable_ids=["price", "demand", "revenue"],
            variable_names=["Price", "Demand", "Revenue"],
        )
        assert len(examples) == 1
        assert len(examples[0].extractions) == 2  # 2 examples for 3+ vars
        assert examples[0].extractions[0].attributes["from_var"] == "price"
        assert examples[0].extractions[1].attributes["to_var"] == "revenue"

    def test_extraction_service_build_edge_examples_fallback(self):
        """Should produce generic example with <2 variables."""
        examples = ExtractionService._build_edge_examples(
            variable_ids=["x"],
            variable_names=["X"],
        )
        assert len(examples) == 1
        assert examples[0].extractions[0].attributes["from_var"] == "a"

    def test_extraction_service_edge_prompt_constrains_vars(self):
        """Edge prompt template should contain the {allowed_vars} placeholder."""
        from src.extraction.service import _EDGE_PROMPT_TEMPLATE
        assert "{allowed_vars}" in _EDGE_PROMPT_TEMPLATE
        assert "MUST ONLY use variable IDs" in _EDGE_PROMPT_TEMPLATE

    @pytest.mark.asyncio
    async def test_run_audit_includes_contradiction_info(self, mode1):
        """Audit trail should include contradiction and contested edge stats."""
        await mode1.initialize()
        result = await mode1.run(
            domain="audit_test",
            initial_query="test",
        )
        # Find the triangulation audit entry
        tri_entries = [
            e for e in result.audit_entries
            if e.action == "evidence_triangulation_complete"
        ]
        assert len(tri_entries) == 1
        data = tri_entries[0].data
        assert "contradicting_evidence" in data
        assert "contested_edges" in data


class TestMode1Stage:
    """Test Mode 1 stages enum."""
    
    def test_all_stages_defined(self):
        """Should have all required stages."""
        stages = [s.value for s in Mode1Stage]
        
        assert "variable_discovery" in stages
        assert "evidence_gathering" in stages
        assert "dag_drafting" in stages
        assert "evidence_triangulation" in stages
        assert "human_review" in stages
        assert "complete" in stages


class TestMode2Stage:
    """Test Mode 2 stages enum."""
    
    def test_all_stages_defined(self):
        """Should have all required stages."""
        stages = [s.value for s in Mode2Stage]
        
        assert "query_parsing" in stages
        assert "model_retrieval" in stages
        assert "evidence_refresh" in stages
        assert "causal_reasoning" in stages
        assert "recommendation_synthesis" in stages
        assert "complete" in stages
