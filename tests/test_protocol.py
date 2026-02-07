"""
Tests for Module 3: Protocol Engine

Tests cover:
- State machine transitions
- Invalid transition handling
- Context management
- Mode router pattern matching
"""

import pytest
import asyncio
from datetime import datetime

from src.protocol.state_machine import (
    ProtocolState,
    ProtocolContext,
    ProtocolStateMachine,
    InvalidStateTransition,
    VALID_TRANSITIONS,
)
from src.protocol.mode_router import (
    ModeRouter,
    RouteDecision,
    RouteReason,
)
from src.models.enums import OperatingMode


class TestProtocolContext:
    """Test ProtocolContext dataclass."""
    
    def test_default_trace_id(self):
        """Context should have auto-generated trace_id."""
        ctx = ProtocolContext()
        assert ctx.trace_id.startswith("trace_")
        assert len(ctx.trace_id) > 6
    
    def test_duration_calculation(self):
        """duration_ms should calculate correctly."""
        ctx = ProtocolContext()
        ctx.mark_complete()
        assert ctx.duration_ms is not None
        assert ctx.duration_ms >= 0
    
    def test_mark_error(self):
        """mark_error should set error and timestamp."""
        ctx = ProtocolContext()
        ctx.mark_error("Something went wrong")
        assert ctx.error == "Something went wrong"
        assert ctx.ended_at is not None


class TestValidTransitions:
    """Test state transition rules."""
    
    def test_idle_can_transition_to_routing(self):
        """IDLE should transition to ROUTING."""
        assert ProtocolState.ROUTING in VALID_TRANSITIONS[ProtocolState.IDLE]
    
    def test_routing_can_go_to_mode1_or_mode2(self):
        """ROUTING should go to either mode."""
        valid = VALID_TRANSITIONS[ProtocolState.ROUTING]
        assert ProtocolState.WM_DISCOVERY_RUNNING in valid
        assert ProtocolState.DECISION_SUPPORT_RUNNING in valid
    
    def test_mode1_can_go_to_review(self):
        """WM_DISCOVERY_RUNNING should go to WM_REVIEW_PENDING."""
        valid = VALID_TRANSITIONS[ProtocolState.WM_DISCOVERY_RUNNING]
        assert ProtocolState.WM_REVIEW_PENDING in valid
    
    def test_review_can_approve_or_revise(self):
        """WM_REVIEW_PENDING should allow approve or revision."""
        valid = VALID_TRANSITIONS[ProtocolState.WM_REVIEW_PENDING]
        assert ProtocolState.WM_ACTIVE in valid
        assert ProtocolState.WM_DISCOVERY_RUNNING in valid
    
    def test_mode2_can_escalate_to_mode1(self):
        """Decision support can escalate to world model construction."""
        valid = VALID_TRANSITIONS[ProtocolState.DECISION_SUPPORT_RUNNING]
        assert ProtocolState.WM_DISCOVERY_RUNNING in valid


class TestProtocolStateMachine:
    """Test ProtocolStateMachine behavior."""
    
    @pytest.fixture
    def sm(self):
        """Create fresh state machine."""
        return ProtocolStateMachine()
    
    @pytest.mark.asyncio
    async def test_initial_state_is_idle(self, sm):
        """State machine starts in IDLE."""
        assert sm.state == ProtocolState.IDLE
        assert sm.is_idle
    
    @pytest.mark.asyncio
    async def test_start_transitions_to_routing(self, sm):
        """start() should transition to ROUTING."""
        ctx = await sm.start("Test query")
        assert sm.state == ProtocolState.ROUTING
        assert ctx.user_query == "Test query"
    
    @pytest.mark.asyncio
    async def test_set_mode1(self, sm):
        """set_mode(MODE_1) should transition correctly."""
        await sm.start("Build model for pricing")
        await sm.set_mode(OperatingMode.MODE_1)
        assert sm.state == ProtocolState.WM_DISCOVERY_RUNNING
        assert sm.context.mode == OperatingMode.MODE_1
    
    @pytest.mark.asyncio
    async def test_set_mode2(self, sm):
        """set_mode(MODE_2) should transition correctly."""
        await sm.start("Should we raise prices?")
        await sm.set_mode(OperatingMode.MODE_2)
        assert sm.state == ProtocolState.DECISION_SUPPORT_RUNNING
    
    @pytest.mark.asyncio
    async def test_mode1_full_cycle(self, sm):
        """Test full Mode 1 lifecycle."""
        await sm.start("Build model")
        await sm.set_mode(OperatingMode.MODE_1)
        
        # Discovery running
        assert sm.state == ProtocolState.WM_DISCOVERY_RUNNING
        
        # Complete discovery
        await sm.complete_discovery()
        assert sm.state == ProtocolState.WM_REVIEW_PENDING
        assert sm.is_waiting_review
        
        # Approve model
        await sm.approve_model()
        assert sm.state == ProtocolState.WM_ACTIVE
        
        # Finish
        await sm.finish(result={"model": "test"})
        assert sm.is_idle
        assert sm.context.result == {"model": "test"}
    
    @pytest.mark.asyncio
    async def test_mode1_revision_cycle(self, sm):
        """Test Mode 1 with revision request."""
        await sm.start("Build model")
        await sm.set_mode(OperatingMode.MODE_1)
        await sm.complete_discovery()
        
        # Request revision
        await sm.request_revision()
        assert sm.state == ProtocolState.WM_DISCOVERY_RUNNING
    
    @pytest.mark.asyncio
    async def test_mode2_full_cycle(self, sm):
        """Test full Mode 2 lifecycle."""
        await sm.start("Should we raise prices?")
        await sm.set_mode(OperatingMode.MODE_2)
        
        assert sm.state == ProtocolState.DECISION_SUPPORT_RUNNING
        
        await sm.complete_decision_support()
        assert sm.state == ProtocolState.RESPONSE_READY
        
        await sm.finish()
        assert sm.is_idle
    
    @pytest.mark.asyncio
    async def test_mode2_escalation(self, sm):
        """Test Mode 2 escalation to Mode 1."""
        await sm.start("Should we raise prices?")
        await sm.set_mode(OperatingMode.MODE_2)
        
        # Escalate to Mode 1
        await sm.escalate_to_mode1()
        assert sm.state == ProtocolState.WM_DISCOVERY_RUNNING
        assert sm.context.mode == OperatingMode.MODE_1
    
    @pytest.mark.asyncio
    async def test_invalid_transition_raises(self, sm):
        """Invalid transitions should raise InvalidStateTransition."""
        # Can't go directly from IDLE to WM_ACTIVE
        with pytest.raises(InvalidStateTransition) as exc_info:
            await sm.transition(ProtocolState.WM_ACTIVE)
        
        assert exc_info.value.from_state == ProtocolState.IDLE
        assert exc_info.value.to_state == ProtocolState.WM_ACTIVE
    
    @pytest.mark.asyncio
    async def test_cannot_start_when_not_idle(self, sm):
        """start() should fail if not IDLE."""
        await sm.start("First query")
        
        with pytest.raises(RuntimeError, match="Cannot start"):
            await sm.start("Second query")
    
    @pytest.mark.asyncio
    async def test_context_manager_success(self, sm):
        """run() context manager should handle success."""
        async with sm.run("Test query") as ctx:
            await sm.set_mode(OperatingMode.MODE_2)
            await sm.complete_decision_support()
            ctx.result = {"recommendation": "hold"}
        
        assert sm.is_idle
        assert ctx.result == {"recommendation": "hold"}
    
    @pytest.mark.asyncio
    async def test_context_manager_exception(self, sm):
        """run() context manager should handle exceptions."""
        with pytest.raises(ValueError):
            async with sm.run("Test query"):
                await sm.set_mode(OperatingMode.MODE_2)
                raise ValueError("Test error")
        
        # Should be back in IDLE after error
        assert sm.is_idle
        assert sm.context.error == "Test error"
    
    @pytest.mark.asyncio
    async def test_fail_method(self, sm):
        """fail() should transition to IDLE with error."""
        await sm.start("Test")
        await sm.set_mode(OperatingMode.MODE_2)
        await sm.fail("Something went wrong")
        
        assert sm.is_idle
        assert sm.context.error == "Something went wrong"
    
    @pytest.mark.asyncio
    async def test_reset(self, sm):
        """reset() should force IDLE state."""
        await sm.start("Test")
        await sm.set_mode(OperatingMode.MODE_2)
        
        await sm.reset()
        assert sm.is_idle
        assert sm.context is None
    
    @pytest.mark.asyncio
    async def test_state_history_tracking(self, sm):
        """State history should be tracked."""
        await sm.start("Test")
        await sm.set_mode(OperatingMode.MODE_2)
        await sm.complete_decision_support()
        await sm.finish()
        
        history = sm.get_state_history()
        assert len(history) >= 4
        states = [s for s, _ in history]
        assert "idle" in states
        assert "routing" in states


class TestModeRouter:
    """Test ModeRouter pattern matching."""
    
    @pytest.fixture
    def router(self):
        """Create fresh router."""
        return ModeRouter()
    
    def test_explicit_mode1_command(self, router):
        """Explicit /mode1 command should route to Mode 1."""
        decision = router.route("/mode1 pricing")
        assert decision.mode == OperatingMode.MODE_1
        assert decision.confidence == 1.0
        assert decision.reason == RouteReason.EXPLICIT_COMMAND
        assert decision.extracted_domain == "pricing"
    
    def test_explicit_worldmodel_command(self, router):
        """Explicit /worldmodel command should route to Mode 1."""
        decision = router.route("/worldmodel customer churn")
        assert decision.mode == OperatingMode.MODE_1
        assert decision.confidence == 1.0
    
    def test_explicit_mode2_command(self, router):
        """Explicit /mode2 command should route to Mode 2."""
        decision = router.route("/mode2 should we launch?")
        assert decision.mode == OperatingMode.MODE_2
        assert decision.confidence == 1.0
    
    def test_explicit_decide_command(self, router):
        """Explicit /decide command should route to Mode 2."""
        decision = router.route("/decide raise prices 10%")
        assert decision.mode == OperatingMode.MODE_2
        assert decision.confidence == 1.0
    
    def test_build_model_pattern(self, router):
        """'Build a model for X' should route to Mode 1."""
        decision = router.route("Build a model for pricing decisions")
        assert decision.mode == OperatingMode.MODE_1
        assert decision.reason == RouteReason.PATTERN_MATCH
        assert decision.extracted_domain is not None
    
    def test_create_causal_pattern(self, router):
        """'Create a causal graph for X' should route to Mode 1."""
        decision = router.route("Create a causal graph for customer retention")
        assert decision.mode == OperatingMode.MODE_1
    
    def test_analyze_relationships_pattern(self, router):
        """'Analyze causal relationships' should route to Mode 1."""
        decision = router.route("Analyze the causal relationships for marketing spend")
        assert decision.mode == OperatingMode.MODE_1
    
    def test_should_we_pattern(self, router):
        """'Should we X' questions route to Mode 2."""
        decision = router.route("Should we raise prices next quarter?")
        assert decision.mode == OperatingMode.MODE_2
        assert decision.reason == RouteReason.PATTERN_MATCH
    
    def test_what_would_happen_pattern(self, router):
        """'What would happen if' questions route to Mode 2."""
        decision = router.route("What would happen if we cut marketing spend?")
        assert decision.mode == OperatingMode.MODE_2
    
    def test_how_affect_pattern(self, router):
        """'How does X affect Y' questions route to Mode 2."""
        decision = router.route("How does pricing affect customer retention?")
        assert decision.mode == OperatingMode.MODE_2
    
    def test_impact_pattern(self, router):
        """'What is the impact of' questions route to Mode 2."""
        decision = router.route("What is the impact of the new feature?")
        assert decision.mode == OperatingMode.MODE_2
    
    def test_ambiguous_defaults_to_mode2(self, router):
        """Ambiguous queries should default to Mode 2."""
        decision = router.route("Hello, how are you?")
        assert decision.mode == OperatingMode.MODE_2
        assert decision.reason == RouteReason.DEFAULT
        assert decision.confidence == 0.5
    
    def test_explain_route(self, router):
        """explain_route should return human-readable string."""
        decision = router.route("/mode1 pricing")
        explanation = router.explain_route(decision)
        assert "World Model Construction" in explanation
        assert "explicit command" in explanation.lower()


class TestIntegration:
    """Integration tests for protocol engine."""
    
    @pytest.mark.asyncio
    async def test_router_with_state_machine(self):
        """Router decision should drive state machine."""
        router = ModeRouter()
        sm = ProtocolStateMachine()
        
        # Route the query
        query = "Should we raise prices 10%?"
        decision = router.route(query)
        
        # Use state machine
        async with sm.run(query) as ctx:
            await sm.set_mode(decision.mode)
            assert sm.state == ProtocolState.DECISION_SUPPORT_RUNNING
            
            await sm.complete_decision_support()
            ctx.result = {"recommendation": "Based on the model..."}
        
        assert sm.is_idle
        assert ctx.mode == OperatingMode.MODE_2
