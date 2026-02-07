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
    
    def test_parse_variables_valid_json(self, mode1):
        """Should parse variables from valid JSON."""
        content = '''```json
        [
            {"variable_id": "price", "name": "Price", "description": "Product price", "type": "continuous", "measurement_status": "measured"},
            {"variable_id": "demand", "name": "Demand", "description": "Customer demand", "type": "continuous", "measurement_status": "measured"}
        ]
        ```'''
        
        variables = mode1._parse_variables(content)
        
        assert len(variables) == 2
        assert variables[0].name == "Price"
        assert variables[1].name == "Demand"
    
    def test_parse_edges_valid_json(self, mode1):
        """Should parse edges from valid JSON."""
        content = '''```json
        [
            {"from_var": "price", "to_var": "demand", "mechanism": "Price affects demand", "strength": "strong"}
        ]
        ```'''
        
        edges = mode1._parse_edges(content)
        
        assert len(edges) == 1
        assert edges[0].from_var == "price"
        assert edges[0].to_var == "demand"
        assert edges[0].strength == EvidenceStrength.STRONG
    
    def test_parse_variables_handles_invalid_json(self, mode1):
        """Should handle invalid JSON gracefully."""
        content = "Not valid JSON"
        variables = mode1._parse_variables(content)
        assert variables == []
    
    def test_parse_edges_handles_invalid_json(self, mode1):
        """Should handle invalid JSON gracefully."""
        content = "Not valid JSON"
        edges = mode1._parse_edges(content)
        assert edges == []
    
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
    
    def test_extract_parsed_query_valid_json(self, mode2):
        """Should extract parsed query from valid JSON."""
        content = '''```json
        {"domain": "pricing", "intervention": "increase prices", "target_outcome": "revenue", "constraints": ["budget limit"]}
        ```'''
        
        parsed = mode2._extract_parsed_query(content)
        
        assert parsed.domain == "pricing"
        assert parsed.intervention == "increase prices"
        assert parsed.target_outcome == "revenue"
    
    def test_extract_parsed_query_handles_invalid_json(self, mode2):
        """Should handle invalid JSON gracefully."""
        content = "Not valid JSON"
        parsed = mode2._extract_parsed_query(content)
        
        assert parsed.domain == "general"  # Default
    
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
