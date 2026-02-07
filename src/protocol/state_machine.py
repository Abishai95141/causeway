"""
Protocol Engine State Machine

Manages the lifecycle of Mode 1 (World Model Construction) and 
Mode 2 (Decision Support) operations.

States:
- IDLE: No active operation
- ROUTING: Classifying incoming request
- WM_DISCOVERY_RUNNING: Mode 1 in progress
- WM_REVIEW_PENDING: Mode 1 awaiting human review
- WM_ACTIVE: Mode 1 complete, model active
- DECISION_SUPPORT_RUNNING: Mode 2 in progress
- RESPONSE_READY: Mode 2 complete, recommendation ready
"""

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, AsyncGenerator, Callable, Optional
from uuid import UUID, uuid4

from src.models.enums import OperatingMode, ProtocolState


# Valid state transitions
VALID_TRANSITIONS: dict[ProtocolState, set[ProtocolState]] = {
    ProtocolState.IDLE: {ProtocolState.ROUTING},
    ProtocolState.ROUTING: {
        ProtocolState.WM_DISCOVERY_RUNNING,
        ProtocolState.DECISION_SUPPORT_RUNNING,
        ProtocolState.IDLE,  # No action needed
        ProtocolState.ERROR,
    },
    ProtocolState.WM_DISCOVERY_RUNNING: {
        ProtocolState.WM_REVIEW_PENDING,
        ProtocolState.ERROR,
        ProtocolState.IDLE,  # Cancelled
    },
    ProtocolState.WM_REVIEW_PENDING: {
        ProtocolState.WM_ACTIVE,
        ProtocolState.WM_DISCOVERY_RUNNING,  # Revision requested
        ProtocolState.IDLE,  # Rejected
    },
    ProtocolState.WM_ACTIVE: {
        ProtocolState.IDLE,
    },
    ProtocolState.DECISION_SUPPORT_RUNNING: {
        ProtocolState.RESPONSE_READY,
        ProtocolState.WM_DISCOVERY_RUNNING,  # Escalation to Mode 1
        ProtocolState.ERROR,
        ProtocolState.IDLE,  # Cancelled
    },
    ProtocolState.RESPONSE_READY: {
        ProtocolState.IDLE,
    },
    ProtocolState.ERROR: {
        ProtocolState.IDLE,
    },
}


class InvalidStateTransition(Exception):
    """Raised when an invalid state transition is attempted."""
    
    def __init__(self, from_state: ProtocolState, to_state: ProtocolState):
        self.from_state = from_state
        self.to_state = to_state
        super().__init__(f"Invalid transition: {from_state.value} -> {to_state.value}")


@dataclass
class ProtocolContext:
    """
    Context for a protocol execution run.
    
    Carries all state needed for Mode 1/Mode 2 operations.
    """
    trace_id: str = field(default_factory=lambda: f"trace_{uuid4().hex[:12]}")
    session_id: Optional[str] = None
    user_query: str = ""
    mode: Optional[OperatingMode] = None
    
    # Timestamps
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    ended_at: Optional[datetime] = None
    
    # Results
    result: Optional[Any] = None
    error: Optional[str] = None
    
    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)
    
    @property
    def duration_ms(self) -> Optional[int]:
        """Calculate execution duration in milliseconds."""
        if self.ended_at:
            delta = self.ended_at - self.started_at
            return int(delta.total_seconds() * 1000)
        return None
    
    def mark_complete(self, result: Any = None) -> None:
        """Mark the context as complete."""
        self.ended_at = datetime.now(timezone.utc)
        self.result = result
    
    def mark_error(self, error: str) -> None:
        """Mark the context as failed."""
        self.ended_at = datetime.now(timezone.utc)
        self.error = error


# Type for state change callbacks
StateCallback = Callable[[ProtocolState, ProtocolState, ProtocolContext], None]


class ProtocolStateMachine:
    """
    State machine for protocol engine.
    
    Manages state transitions and provides context management
    for Mode 1/Mode 2 operations.
    """
    
    def __init__(
        self,
        timeout_seconds: float = 300.0,  # 5 minutes default
        on_state_change: Optional[StateCallback] = None,
    ):
        self._state = ProtocolState.IDLE
        self._context: Optional[ProtocolContext] = None
        self._timeout_seconds = timeout_seconds
        self._on_state_change = on_state_change
        self._lock = asyncio.Lock()
        self._state_history: list[tuple[ProtocolState, datetime]] = []
    
    @property
    def state(self) -> ProtocolState:
        """Current state."""
        return self._state
    
    @property
    def context(self) -> Optional[ProtocolContext]:
        """Current execution context."""
        return self._context
    
    @property
    def is_idle(self) -> bool:
        """Check if in IDLE state."""
        return self._state == ProtocolState.IDLE
    
    @property
    def is_running(self) -> bool:
        """Check if actively processing."""
        return self._state in {
            ProtocolState.ROUTING,
            ProtocolState.WM_DISCOVERY_RUNNING,
            ProtocolState.DECISION_SUPPORT_RUNNING,
        }
    
    @property
    def is_waiting_review(self) -> bool:
        """Check if waiting for human review."""
        return self._state == ProtocolState.WM_REVIEW_PENDING
    
    def _can_transition(self, to_state: ProtocolState) -> bool:
        """Check if transition is valid."""
        valid_targets = VALID_TRANSITIONS.get(self._state, set())
        return to_state in valid_targets
    
    async def transition(self, to_state: ProtocolState) -> None:
        """
        Transition to a new state.
        
        Raises InvalidStateTransition if not allowed.
        """
        async with self._lock:
            if not self._can_transition(to_state):
                raise InvalidStateTransition(self._state, to_state)
            
            from_state = self._state
            self._state = to_state
            self._state_history.append((to_state, datetime.now(timezone.utc)))
            
            # Invoke callback if registered
            if self._on_state_change and self._context:
                try:
                    self._on_state_change(from_state, to_state, self._context)
                except Exception:
                    pass  # Don't let callback errors affect state machine
    
    async def start(self, query: str, session_id: Optional[str] = None) -> ProtocolContext:
        """
        Start a new protocol run.
        
        Args:
            query: User's input query
            session_id: Optional session identifier
            
        Returns:
            New ProtocolContext
        """
        async with self._lock:
            if not self.is_idle:
                raise RuntimeError(f"Cannot start: current state is {self._state.value}")
            
            self._context = ProtocolContext(
                user_query=query,
                session_id=session_id,
            )
            self._state_history = [(ProtocolState.IDLE, self._context.started_at)]
        
        await self.transition(ProtocolState.ROUTING)
        return self._context
    
    async def set_mode(self, mode: OperatingMode) -> None:
        """
        Set the operating mode and transition accordingly.
        
        Args:
            mode: MODE_1 or MODE_2
        """
        if self._context:
            self._context.mode = mode
        
        if mode == OperatingMode.MODE_1:
            await self.transition(ProtocolState.WM_DISCOVERY_RUNNING)
        else:
            await self.transition(ProtocolState.DECISION_SUPPORT_RUNNING)
    
    async def complete_discovery(self) -> None:
        """Mark Mode 1 discovery as complete, pending review."""
        await self.transition(ProtocolState.WM_REVIEW_PENDING)
    
    async def approve_model(self) -> None:
        """Approve the world model after review."""
        await self.transition(ProtocolState.WM_ACTIVE)
    
    async def request_revision(self) -> None:
        """Request revision, back to discovery."""
        await self.transition(ProtocolState.WM_DISCOVERY_RUNNING)
    
    async def complete_decision_support(self) -> None:
        """Mark Mode 2 as complete."""
        await self.transition(ProtocolState.RESPONSE_READY)
    
    async def escalate_to_mode1(self) -> None:
        """Escalate from Mode 2 to Mode 1."""
        if self._context:
            self._context.mode = OperatingMode.MODE_1
        await self.transition(ProtocolState.WM_DISCOVERY_RUNNING)
    
    async def finish(self, result: Any = None) -> None:
        """
        Finish the protocol run and return to IDLE.
        
        Args:
            result: Optional result to store in context
        """
        if self._context:
            self._context.mark_complete(result)
        await self.transition(ProtocolState.IDLE)
    
    async def fail(self, error: str) -> None:
        """
        Mark the run as failed.
        
        Args:
            error: Error message
        """
        if self._context:
            self._context.mark_error(error)
        
        # Transition to ERROR state first (if valid), then IDLE
        if self._can_transition(ProtocolState.ERROR):
            await self.transition(ProtocolState.ERROR)
        await self.transition(ProtocolState.IDLE)
    
    async def reset(self) -> None:
        """Force reset to IDLE (for error recovery)."""
        async with self._lock:
            self._state = ProtocolState.IDLE
            self._context = None
            self._state_history = []
    
    @asynccontextmanager
    async def run(
        self,
        query: str,
        session_id: Optional[str] = None,
    ) -> AsyncGenerator[ProtocolContext, None]:
        """
        Context manager for a complete protocol run.
        
        Handles automatic cleanup on success or failure.
        
        Usage:
            async with state_machine.run("Should we raise prices?") as ctx:
                # Do processing...
                ctx.result = recommendation
        """
        ctx = await self.start(query, session_id)
        try:
            yield ctx
            if self._state != ProtocolState.IDLE:
                await self.finish(ctx.result)
        except asyncio.TimeoutError:
            await self.fail("Operation timed out")
            raise
        except Exception as e:
            await self.fail(str(e))
            raise
    
    def get_state_history(self) -> list[tuple[str, str]]:
        """Get state transition history."""
        return [(s.value, t.isoformat()) for s, t in self._state_history]
