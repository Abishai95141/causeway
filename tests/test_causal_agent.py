"""
Tests for Modules 7-8: Causal Intelligence & Agent Runtime

Tests cover:
- DAG engine (node/edge CRUD, cycle detection, validation)
- Path finder (paths, confounders, mediators)
- Causal service (high-level operations)
- LLM client (mock mode)
- Context manager (token budgeting)
- Agent orchestrator (tool dispatch)
"""

import pytest
from uuid import uuid4

from src.causal.dag_engine import (
    DAGEngine,
    CycleDetectedError,
    NodeNotFoundError,
)
from src.causal.path_finder import CausalPathFinder
from src.causal.service import CausalService
from src.agent.llm_client import LLMClient, LLMModel
from src.agent.context_manager import ContextManager, MessageRole
from src.agent.orchestrator import AgentOrchestrator
from src.models.enums import EvidenceStrength, ModelStatus, VariableType, MeasurementStatus, VariableRole


class TestDAGEngine:
    """Test DAG engine operations."""
    
    @pytest.fixture
    def engine(self):
        """Create fresh engine."""
        return DAGEngine()
    
    def test_add_variable(self, engine):
        """Should add variable as node."""
        var = engine.add_variable(
            "price", 
            "Product Price", 
            "The price of the product",
            unit="USD"
        )
        assert var.variable_id == "price"
        assert engine.node_count == 1
    
    def test_add_duplicate_variable_raises(self, engine):
        """Should reject duplicate variable names."""
        engine.add_variable("price", "Product Price", "The price")
        with pytest.raises(ValueError, match="already exists"):
            engine.add_variable("price", "Another Price", "Another price")
    
    def test_add_edge(self, engine):
        """Should add edge between nodes."""
        engine.add_variable("price", "Product Price", "The price")
        engine.add_variable("demand", "Customer Demand", "The demand")
        
        edge = engine.add_edge(
            "price", "demand",
            mechanism="Higher prices reduce demand",
            strength=EvidenceStrength.STRONG,
        )
        
        assert edge.from_var == "price"
        assert edge.to_var == "demand"
        assert engine.edge_count == 1
    
    def test_add_edge_missing_node_raises(self, engine):
        """Should reject edge with missing node."""
        engine.add_variable("price", "Product Price", "The price")
        
        with pytest.raises(NodeNotFoundError):
            engine.add_edge("price", "demand", "Test mechanism")
    
    def test_cycle_detection(self, engine):
        """Should detect and reject cycles."""
        engine.add_variable("var_a", "Variable A", "Variable A")
        engine.add_variable("var_b", "Variable B", "Variable B")
        engine.add_variable("var_c", "Variable C", "Variable C")
        
        engine.add_edge("var_a", "var_b", "A causes B")
        engine.add_edge("var_b", "var_c", "B causes C")
        
        # This would create A -> B -> C -> A cycle
        with pytest.raises(CycleDetectedError):
            engine.add_edge("var_c", "var_a", "C causes A")
    
    def test_remove_variable(self, engine):
        """Should remove variable and connected edges."""
        engine.add_variable("var_a", "A", "Variable A")
        engine.add_variable("var_b", "B", "Variable B")
        engine.add_edge("var_a", "var_b", "test")
        
        engine.remove_variable("var_a")
        
        assert engine.node_count == 1
        assert engine.edge_count == 0
    
    def test_validation(self, engine):
        """Should validate DAG structure."""
        engine.add_variable("var_a", "A", "Variable A")
        engine.add_variable("var_b", "B", "Variable B")
        engine.add_edge("var_a", "var_b", "test")
        
        result = engine.validate()
        
        assert result.is_valid
        assert result.node_count == 2
        assert result.edge_count == 1
    
    def test_validation_warns_no_evidence(self, engine):
        """Should warn about edges without evidence."""
        engine.add_variable("var_a", "A", "Variable A")
        engine.add_variable("var_b", "B", "Variable B")
        engine.add_edge("var_a", "var_b", "test")  # No evidence_refs
        
        result = engine.validate()
        assert len(result.warnings) > 0
        assert "no evidence" in result.warnings[0].lower()
    
    def test_to_world_model(self, engine):
        """Should serialize to WorldModelVersion."""
        engine.add_variable("price", "Price", "Product price")
        engine.add_variable("demand", "Demand", "Customer demand")
        engine.add_edge("price", "demand", "Price affects demand")
        
        model = engine.to_world_model("pricing", "Pricing model")
        
        assert model.domain == "pricing"
        assert len(model.variables) == 2
        assert len(model.edges) == 1
        assert model.status == ModelStatus.DRAFT
    
    def test_from_world_model(self, engine):
        """Should deserialize from WorldModelVersion."""
        engine.add_variable("var_a", "A", "Variable A")
        engine.add_variable("var_b", "B", "Variable B")
        engine.add_edge("var_a", "var_b", "test")
        
        model = engine.to_world_model("test", "Test model")
        
        # Create new engine from model
        new_engine = DAGEngine.from_world_model(model)
        
        assert new_engine.node_count == 2
        assert new_engine.edge_count == 1


class TestDAGEnginePhase1:
    """Test Phase 1 extensions: variable roles, edge assumptions, conditions, contradictions."""

    @pytest.fixture
    def engine(self):
        return DAGEngine()

    def test_add_variable_with_role(self, engine):
        """Should accept a VariableRole for the variable."""
        var = engine.add_variable(
            "price", "Price", "Product price",
            role=VariableRole.TREATMENT,
        )
        assert var.role == VariableRole.TREATMENT

    def test_add_variable_default_role_is_unknown(self, engine):
        """Default role should be UNKNOWN."""
        var = engine.add_variable("x", "X", "Variable X")
        assert var.role == VariableRole.UNKNOWN

    def test_add_edge_with_assumptions(self, engine):
        """Should store assumptions on the edge metadata."""
        engine.add_variable("a", "A", "Variable A")
        engine.add_variable("b", "B", "Variable B")

        edge = engine.add_edge(
            "a", "b", "A causes B",
            assumptions=["No unmeasured confounders", "Temporal ordering"],
        )
        assert edge.metadata.assumptions == ["No unmeasured confounders", "Temporal ordering"]

    def test_add_edge_with_conditions(self, engine):
        """Should store conditions on the edge metadata."""
        engine.add_variable("a", "A", "Variable A")
        engine.add_variable("b", "B", "Variable B")

        edge = engine.add_edge(
            "a", "b", "A causes B",
            conditions=["Only in US market", "When demand > 100"],
        )
        assert edge.metadata.conditions == ["Only in US market", "When demand > 100"]

    def test_add_edge_with_contradicting_refs(self, engine):
        """Should store contradicting evidence refs on the edge metadata."""
        engine.add_variable("a", "A", "Variable A")
        engine.add_variable("b", "B", "Variable B")

        contra_id = uuid4()
        edge = engine.add_edge(
            "a", "b", "A causes B",
            contradicting_refs=[contra_id],
        )
        assert contra_id in edge.metadata.contradicting_refs

    def test_add_edge_with_confidence(self, engine):
        """Should store confidence score on the edge."""
        engine.add_variable("a", "A", "Variable A")
        engine.add_variable("b", "B", "Variable B")

        edge = engine.add_edge(
            "a", "b", "A causes B",
            confidence=0.85,
        )
        assert edge.metadata.confidence == 0.85

    def test_add_edge_all_phase1_fields(self, engine):
        """Should handle all Phase 1 fields together."""
        engine.add_variable("price", "Price", "Product price", role=VariableRole.TREATMENT)
        engine.add_variable("demand", "Demand", "Customer demand", role=VariableRole.OUTCOME)

        ev_id = uuid4()
        contra_id = uuid4()
        edge = engine.add_edge(
            "price", "demand", "Price elasticity",
            strength=EvidenceStrength.MODERATE,
            evidence_refs=[ev_id],
            confidence=0.72,
            assumptions=["No unmeasured confounders"],
            conditions=["US market only"],
            contradicting_refs=[contra_id],
        )

        assert edge.from_var == "price"
        assert edge.to_var == "demand"
        assert edge.metadata.evidence_strength == EvidenceStrength.MODERATE
        assert ev_id in edge.metadata.evidence_refs
        assert contra_id in edge.metadata.contradicting_refs
        assert edge.metadata.assumptions == ["No unmeasured confounders"]
        assert edge.metadata.conditions == ["US market only"]
        assert edge.metadata.confidence == 0.72

    def test_to_world_model_preserves_roles(self, engine):
        """WorldModelVersion serialization should preserve variable roles."""
        engine.add_variable("price", "Price", "Price", role=VariableRole.TREATMENT)
        engine.add_variable("demand", "Demand", "Demand", role=VariableRole.OUTCOME)
        engine.add_edge("price", "demand", "Price affects demand")

        model = engine.to_world_model("test", "Test model")
        new_engine = DAGEngine.from_world_model(model)

        variables_by_id = {v.variable_id: v for v in new_engine.variables}
        assert variables_by_id["price"].role == VariableRole.TREATMENT
        assert variables_by_id["demand"].role == VariableRole.OUTCOME

    def test_to_world_model_preserves_edge_metadata(self, engine):
        """WorldModelVersion serialization should preserve edge assumptions and conditions."""
        engine.add_variable("a", "A", "Variable A")
        engine.add_variable("b", "B", "Variable B")

        contra_id = uuid4()
        engine.add_edge(
            "a", "b", "A causes B",
            assumptions=["Assumption 1"],
            conditions=["Condition 1"],
            contradicting_refs=[contra_id],
            confidence=0.65,
        )

        model = engine.to_world_model("test", "Test model")
        new_engine = DAGEngine.from_world_model(model)

        edges = new_engine.edges
        assert len(edges) == 1
        assert edges[0].metadata.assumptions == ["Assumption 1"]
        assert edges[0].metadata.conditions == ["Condition 1"]
        assert contra_id in edges[0].metadata.contradicting_refs
        assert edges[0].metadata.confidence == 0.65


class TestCausalServicePhase1:
    """Test Phase 1 CausalService extensions."""

    @pytest.fixture
    def service(self):
        svc = CausalService()
        svc.create_world_model("test")
        return svc

    def test_add_variable_with_role(self, service):
        """CausalService.add_variable should pass role through."""
        var = service.add_variable(
            "price", "Price", "Product price",
            role=VariableRole.TREATMENT,
        )
        assert var.role == VariableRole.TREATMENT

    def test_add_causal_link_with_assumptions(self, service):
        """CausalService.add_causal_link should pass assumptions through."""
        service.add_variable("a", "A", "Variable A")
        service.add_variable("b", "B", "Variable B")

        edge = service.add_causal_link(
            "a", "b", "A causes B",
            assumptions=["No confounders"],
        )
        assert edge.metadata.assumptions == ["No confounders"]

    def test_add_causal_link_with_conditions(self, service):
        """CausalService.add_causal_link should pass conditions through."""
        service.add_variable("a", "A", "Variable A")
        service.add_variable("b", "B", "Variable B")

        edge = service.add_causal_link(
            "a", "b", "A causes B",
            conditions=["US market"],
        )
        assert edge.metadata.conditions == ["US market"]

    def test_add_causal_link_with_contradicting_refs(self, service):
        """CausalService.add_causal_link should pass contradicting_refs through."""
        service.add_variable("a", "A", "Variable A")
        service.add_variable("b", "B", "Variable B")

        contra_id = uuid4()
        edge = service.add_causal_link(
            "a", "b", "A causes B",
            contradicting_refs=[contra_id],
        )
        assert contra_id in edge.metadata.contradicting_refs


class TestCausalPathFinder:
    """Test causal path finding."""
    
    @pytest.fixture
    def finder(self):
        """Create finder with test graph."""
        engine = DAGEngine()
        engine.add_variable("var_a", "A", "Variable A")
        engine.add_variable("var_b", "B", "Variable B")
        engine.add_variable("var_c", "C", "Variable C")
        engine.add_variable("var_d", "D", "Variable D")
        engine.add_variable("var_e", "E", "Variable E")
        
        # A -> B -> D
        # A -> C -> D
        # E -> B (confounder)
        # E -> D (confounder)
        engine.add_edge("var_a", "var_b", "A to B")
        engine.add_edge("var_a", "var_c", "A to C")
        engine.add_edge("var_b", "var_d", "B to D")
        engine.add_edge("var_c", "var_d", "C to D")
        engine.add_edge("var_e", "var_b", "E to B")
        engine.add_edge("var_e", "var_d", "E to D")
        
        return CausalPathFinder(engine.graph)
    
    def test_find_all_paths(self, finder):
        """Should find all paths between nodes."""
        paths = finder.find_all_paths("var_a", "var_d")
        
        assert len(paths) == 2  # A->B->D and A->C->D
        path_nodes = [p.path for p in paths]
        assert ["var_a", "var_b", "var_d"] in path_nodes
        assert ["var_a", "var_c", "var_d"] in path_nodes
    
    def test_find_shortest_path(self, finder):
        """Should find shortest path."""
        path = finder.find_shortest_path("var_a", "var_d")
        
        assert path is not None
        assert path.length == 2
    
    def test_no_path_returns_none(self, finder):
        """Should return None when no path exists."""
        path = finder.find_shortest_path("var_d", "var_a")  # Reverse direction
        assert path is None
    
    def test_find_confounders(self, finder):
        """Should identify confounders."""
        result = finder.find_confounders("var_b", "var_d")
        
        # E affects both B and D, so it's a confounder
        assert "var_e" in result.confounders
    
    def test_find_mediators(self, finder):
        """Should identify mediators."""
        result = finder.find_mediators("var_a", "var_d")
        
        # B and C are mediators between A and D
        assert "var_b" in result.mediators
        assert "var_c" in result.mediators
        assert not result.direct_path_exists  # No direct A->D edge
    
    def test_analyze_relationship(self, finder):
        """Should provide complete analysis."""
        analysis = finder.analyze("var_a", "var_d")
        
        assert analysis.cause == "var_a"
        assert analysis.effect == "var_d"
        assert analysis.total_paths == 2
        assert len(analysis.mediators) == 2
    
    def test_get_descendants(self, finder):
        """Should get all descendants."""
        descendants = finder.get_descendants("var_a")
        assert "var_b" in descendants
        assert "var_c" in descendants
        assert "var_d" in descendants
    
    def test_topological_sort(self, finder):
        """Should return valid topological order."""
        order = finder.topological_sort()
        
        # A and E should come before B, C, D
        a_idx = order.index("var_a")
        d_idx = order.index("var_d")
        assert a_idx < d_idx


class TestCausalService:
    """Test causal service operations."""
    
    @pytest.fixture
    def service(self):
        """Create service with test model."""
        svc = CausalService()
        svc.create_world_model("pricing")
        svc.add_variable("price", "Product Price", "The product price")
        svc.add_variable("demand", "Customer Demand", "The customer demand")
        svc.add_variable("revenue", "Total Revenue", "Total revenue")
        svc.add_causal_link("price", "demand", "Price affects demand")
        svc.add_causal_link("demand", "revenue", "Demand affects revenue")
        return svc
    
    def test_create_world_model(self, service):
        """Should create new world model."""
        assert "pricing" in service.list_domains()
    
    def test_analyze_relationship(self, service):
        """Should analyze causal relationship."""
        analysis = service.analyze_relationship("price", "revenue")
        
        assert analysis.total_paths == 1
        assert "demand" in analysis.mediators
    
    def test_get_variable_effects(self, service):
        """Should get all effects of a variable."""
        effects = service.get_variable_effects("price")
        
        assert "demand" in effects
        assert "revenue" in effects
    
    def test_link_evidence(self, service):
        """Should link evidence to edge."""
        evidence_uuid = uuid4()
        service.link_evidence("price", "demand", evidence_uuid)
        
        engine = service.get_engine()
        edge = [e for e in engine.edges if e.from_var == "price"][0]
        assert evidence_uuid in edge.metadata.evidence_refs
    
    def test_export_import_model(self, service):
        """Should export and import model."""
        model = service.export_world_model()
        
        # Create new service and import
        new_service = CausalService()
        new_service.import_world_model(model)
        
        assert "pricing" in new_service.list_domains()
        summary = new_service.get_model_summary("pricing")
        assert summary["node_count"] == 3


class TestLLMClient:
    """Test LLM client in mock mode."""
    
    @pytest.fixture
    def client(self):
        """Create client in mock mode."""
        return LLMClient(api_key=None)  # No API key = mock mode
    
    def test_mock_mode_enabled(self, client):
        """Should be in mock mode without API key."""
        assert client.is_mock_mode
    
    @pytest.mark.asyncio
    async def test_generate(self, client):
        """Should generate mock response."""
        response = await client.generate("Test prompt")
        
        assert response.content is not None
        assert response.model == "mock"
        assert response.latency_ms >= 0
    
    @pytest.mark.asyncio
    async def test_generate_with_custom_mock(self, client):
        """Should use custom mock responses."""
        client.set_mock_responses(['{"answer": "42"}'])
        
        response = await client.generate("What is the answer?")
        
        assert '"answer"' in response.content
    
    @pytest.mark.asyncio
    async def test_generate_with_tools(self, client):
        """Should generate with tool definitions."""
        from src.agent.llm_client import ToolDefinition
        
        tools = [
            ToolDefinition(
                name="search",
                description="Search for information",
                parameters={"query": {"type": "string"}},
            )
        ]
        
        response = await client.generate_with_tools("Find pricing data", tools)
        
        assert response.content is not None


class TestContextManager:
    """Test context manager."""
    
    @pytest.fixture
    def ctx(self):
        """Create context manager."""
        return ContextManager(max_tokens=1000)
    
    def test_add_messages(self, ctx):
        """Should add messages to context."""
        ctx.add_user_message("Hello")
        ctx.add_assistant_message("Hi there!")
        
        assert len(ctx.messages) == 2
    
    def test_token_tracking(self, ctx):
        """Should track token usage."""
        ctx.add_user_message("This is a test message.")
        
        assert ctx.used_tokens > 0
        assert ctx.available_tokens < 1000
    
    def test_build_prompt(self, ctx):
        """Should build formatted prompt."""
        ctx.set_system_prompt("You are a helpful assistant.")
        ctx.add_user_message("Hello")
        ctx.add_assistant_message("Hi!")
        
        prompt = ctx.build_prompt()
        
        assert "System:" in prompt
        assert "User: Hello" in prompt
        assert "Assistant: Hi!" in prompt
    
    def test_evidence_tracking(self, ctx):
        """Should track evidence context."""
        ctx.add_evidence("ev1", "Price data from Q3 report")
        ctx.add_evidence("ev2", "Customer survey results")
        
        evidence_ctx = ctx.get_evidence_context()
        
        assert "[ev1]" in evidence_ctx
        assert "Price data" in evidence_ctx
    
    def test_auto_trim(self, ctx):
        """Should auto-trim when approaching limit."""
        ctx = ContextManager(max_tokens=100, summarize_threshold=0.8)
        
        # Add many messages
        for i in range(20):
            ctx.add_user_message(f"Message {i} " * 10)
        
        # Should have trimmed old messages
        assert ctx.used_tokens <= 100
    
    def test_get_stats(self, ctx):
        """Should return context statistics."""
        ctx.add_user_message("Test 1")
        ctx.add_assistant_message("Response 1")
        ctx.add_user_message("Test 2")
        
        stats = ctx.get_stats()
        
        assert stats.total_messages == 3
        assert stats.user_messages == 2
        assert stats.assistant_messages == 1


class TestAgentOrchestrator:
    """Test agent orchestrator."""
    
    @pytest.fixture
    def orchestrator(self):
        """Create orchestrator."""
        return AgentOrchestrator()
    
    @pytest.mark.asyncio
    async def test_run_simple_query(self, orchestrator):
        """Should run simple query."""
        await orchestrator.initialize()
        
        result = await orchestrator.run("What is 2+2?")
        
        assert result.trace_id.startswith("orch_")
        assert result.query == "What is 2+2?"
        assert result.response is not None
    
    @pytest.mark.asyncio
    async def test_context_accessible(self, orchestrator):
        """Should provide access to context."""
        await orchestrator.initialize()
        await orchestrator.run("Test query")
        
        assert len(orchestrator.context.messages) > 0
    
    def test_register_custom_tool(self, orchestrator):
        """Should register custom tools."""
        from src.agent.llm_client import ToolDefinition
        
        async def custom_handler(x: int) -> int:
            return x * 2
        
        orchestrator.register_tool(
            ToolDefinition(
                name="double",
                description="Double a number",
                parameters={"x": {"type": "integer"}},
            ),
            custom_handler,
        )
        
        assert "double" in orchestrator._tool_handlers
