"""
Tests for CausewayAgent — Unified Agentic Entrypoint

Tests cover:
- Mode routing → tool selection
- PageIndex tool registration
- End-to-end agentic loop (mock LLM)
- Mode 2 escalation flow
"""

import pytest

from src.agent.causeway_agent import CausewayAgent, AgentResult
from src.agent.llm_client import LLMClient, ToolDefinition
from src.agent.orchestrator import AgentOrchestrator
from src.causal.service import CausalService
from src.pageindex.client import PageIndexClient
from src.pageindex.pageindex_tools import create_pageindex_tools
from src.protocol.mode_router import ModeRouter, RouteDecision, RouteReason
from src.models.enums import OperatingMode


class TestModeRouting:
    """Test that CausewayAgent correctly sets up tools based on routing."""

    @pytest.fixture
    def agent(self):
        """Create a CausewayAgent in mock mode (no API keys)."""
        return CausewayAgent()

    def test_route_mode1_query(self, agent):
        """Mode 1 pattern should produce a Mode 1 route decision."""
        decision = agent.router.route("Build a world model for pricing")
        assert decision.mode == OperatingMode.MODE_1

    def test_route_mode2_query(self, agent):
        """Decision question should produce a Mode 2 route decision."""
        decision = agent.router.route(
            "Should we increase prices to boost revenue?"
        )
        assert decision.mode == OperatingMode.MODE_2

    def test_mode1_orchestrator_has_mode1_tool(self, agent):
        """Mode 1 route should register run_world_model_construction."""
        decision = RouteDecision(
            mode=OperatingMode.MODE_1,
            confidence=1.0,
            reason=RouteReason.PATTERN_MATCH,
            extracted_domain="pricing",
            raw_query="Build model for pricing",
        )
        orchestrator = agent._build_orchestrator(decision)
        assert "run_world_model_construction" in orchestrator._tool_handlers

    def test_mode2_orchestrator_has_mode2_tool(self, agent):
        """Mode 2 route should register run_decision_support."""
        decision = RouteDecision(
            mode=OperatingMode.MODE_2,
            confidence=0.8,
            reason=RouteReason.PATTERN_MATCH,
            raw_query="Should we increase prices?",
        )
        orchestrator = agent._build_orchestrator(decision)
        assert "run_decision_support" in orchestrator._tool_handlers

    def test_mode1_orchestrator_does_not_have_mode2_tool(self, agent):
        """Mode 1 should NOT include run_decision_support."""
        decision = RouteDecision(
            mode=OperatingMode.MODE_1,
            confidence=1.0,
            reason=RouteReason.EXPLICIT_COMMAND,
            raw_query="/mode1 pricing",
        )
        orchestrator = agent._build_orchestrator(decision)
        assert "run_decision_support" not in orchestrator._tool_handlers


class TestPageIndexToolRegistration:
    """Test that PageIndex tools are correctly registered."""

    @pytest.fixture
    def agent(self):
        return CausewayAgent()

    def test_pageindex_tools_registered(self, agent):
        """All three PageIndex tools should be in the handler map."""
        decision = RouteDecision(
            mode=OperatingMode.MODE_2,
            confidence=0.8,
            reason=RouteReason.DEFAULT,
            raw_query="test query",
        )
        orchestrator = agent._build_orchestrator(decision)

        assert "list_sections" in orchestrator._tool_handlers
        assert "read_section" in orchestrator._tool_handlers
        assert "search_document" in orchestrator._tool_handlers

    def test_default_tools_also_present(self, agent):
        """Built-in orchestrator tools should still be available."""
        decision = RouteDecision(
            mode=OperatingMode.MODE_2,
            confidence=0.8,
            reason=RouteReason.DEFAULT,
            raw_query="test query",
        )
        orchestrator = agent._build_orchestrator(decision)

        # The orchestrator registers these by default
        assert "search_evidence" in orchestrator._tool_handlers
        assert "analyze_causal_path" in orchestrator._tool_handlers


class TestPageIndexToolProvider:
    """Test the PageIndex tool provider directly."""

    @pytest.fixture
    def tools(self):
        client = PageIndexClient()  # mock mode (no API key)
        return create_pageindex_tools(client)

    def test_returns_three_tools(self, tools):
        assert len(tools) == 3

    def test_tool_names(self, tools):
        names = [t[0].name for t in tools]
        assert "list_sections" in names
        assert "read_section" in names
        assert "search_document" in names

    def test_tool_definitions_have_parameters(self, tools):
        for defn, handler in tools:
            assert isinstance(defn, ToolDefinition)
            assert "properties" in defn.parameters
            assert callable(handler)

    @pytest.mark.asyncio
    async def test_list_sections_handler(self, tools):
        """Handler should call PageIndexClient.list_sections (mock)."""
        handler = tools[0][1]  # list_sections handler
        result = await handler(doc_id="test_doc")
        assert "sections" in result
        assert "count" in result

    @pytest.mark.asyncio
    async def test_search_document_handler(self, tools):
        """Handler should call PageIndexClient.search (mock)."""
        handler = tools[2][1]  # search_document handler
        result = await handler(doc_id="test_doc", query="pricing")
        assert "results" in result
        assert "count" in result


class TestCausewayAgentRun:
    """Test end-to-end CausewayAgent.run()."""

    @pytest.fixture
    def agent(self):
        return CausewayAgent()

    @pytest.mark.asyncio
    async def test_run_simple_query(self, agent):
        """Should return an AgentResult from a simple query."""
        result = await agent.run("What is 2+2?")

        assert isinstance(result, AgentResult)
        assert result.trace_id.startswith("orch_")
        assert result.query == "What is 2+2?"
        assert result.response is not None

    @pytest.mark.asyncio
    async def test_run_mode1_query(self, agent):
        """Mode 1 query should be routed correctly."""
        result = await agent.run(
            "/mode1 Build a world model for supply chain disruptions"
        )

        assert isinstance(result, AgentResult)
        assert result.routed_mode == OperatingMode.MODE_1.value

    @pytest.mark.asyncio
    async def test_run_mode2_query(self, agent):
        """Mode 2 query should be routed correctly."""
        result = await agent.run(
            "Should we increase prices to boost revenue?"
        )

        assert isinstance(result, AgentResult)
        assert result.routed_mode == OperatingMode.MODE_2.value

    @pytest.mark.asyncio
    async def test_run_includes_routing_metadata(self, agent):
        """Result should include routing confidence and reason."""
        result = await agent.run("/mode1 pricing")

        assert result.route_confidence == 1.0
        assert result.route_reason == RouteReason.EXPLICIT_COMMAND.value

    @pytest.mark.asyncio
    async def test_run_handles_error_gracefully(self, agent):
        """Agent should not raise — errors are captured in result."""
        # Even with mock mode, there should be no unhandled exceptions
        result = await agent.run("")
        assert isinstance(result, AgentResult)

    @pytest.mark.asyncio
    async def test_initialization_is_idempotent(self, agent):
        """Calling initialize() twice should be safe."""
        await agent.initialize()
        await agent.initialize()
        assert agent.is_initialized
