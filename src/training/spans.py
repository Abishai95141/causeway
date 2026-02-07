"""
Span Collector

Minimal span collection for Agent Lightning training loop.
Captures execution traces without interfering with normal operation.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional
from uuid import uuid4
from enum import Enum


class SpanStatus(str, Enum):
    """Status of a collected span."""
    STARTED = "started"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Span:
    """A single execution span."""
    span_id: str
    trace_id: str
    name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    status: SpanStatus = SpanStatus.STARTED
    attributes: dict[str, Any] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)
    parent_id: Optional[str] = None
    
    @property
    def duration_ms(self) -> Optional[float]:
        """Get duration in milliseconds."""
        if self.end_time and self.start_time:
            return (self.end_time - self.start_time).total_seconds() * 1000
        return None


class SpanCollector:
    """
    Collects spans for training feedback.
    
    Non-blocking, minimal overhead implementation.
    Spans can be dumped to trajectory storage for offline training.
    """
    
    def __init__(self, enabled: bool = True, max_spans: int = 10000):
        self.enabled = enabled
        self.max_spans = max_spans
        self._spans: dict[str, Span] = {}
        self._current_trace: Optional[str] = None
        self._span_stack: list[str] = []
    
    def start_trace(self, name: str = "root") -> str:
        """Start a new trace."""
        trace_id = f"trace_{uuid4().hex[:12]}"
        self._current_trace = trace_id
        
        span = self.start_span(name, trace_id=trace_id)
        return trace_id
    
    def start_span(
        self,
        name: str,
        trace_id: Optional[str] = None,
        attributes: Optional[dict[str, Any]] = None,
    ) -> str:
        """Start a new span."""
        if not self.enabled:
            return ""
        
        span_id = f"span_{uuid4().hex[:12]}"
        parent_id = self._span_stack[-1] if self._span_stack else None
        
        span = Span(
            span_id=span_id,
            trace_id=trace_id or self._current_trace or "",
            name=name,
            start_time=datetime.now(timezone.utc),
            parent_id=parent_id,
            attributes=attributes or {},
        )
        
        self._spans[span_id] = span
        self._span_stack.append(span_id)
        
        # Trim old spans if needed
        if len(self._spans) > self.max_spans:
            oldest = sorted(self._spans.values(), key=lambda s: s.start_time)[:100]
            for s in oldest:
                del self._spans[s.span_id]
        
        return span_id
    
    def end_span(
        self,
        span_id: str,
        status: SpanStatus = SpanStatus.COMPLETED,
        attributes: Optional[dict[str, Any]] = None,
    ) -> None:
        """End a span."""
        if not self.enabled or span_id not in self._spans:
            return
        
        span = self._spans[span_id]
        span.end_time = datetime.now(timezone.utc)
        span.status = status
        
        if attributes:
            span.attributes.update(attributes)
        
        if self._span_stack and self._span_stack[-1] == span_id:
            self._span_stack.pop()
    
    def add_event(
        self,
        span_id: str,
        name: str,
        attributes: Optional[dict[str, Any]] = None,
    ) -> None:
        """Add an event to a span."""
        if not self.enabled or span_id not in self._spans:
            return
        
        self._spans[span_id].events.append({
            "name": name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "attributes": attributes or {},
        })
    
    def get_span(self, span_id: str) -> Optional[Span]:
        """Get a span by ID."""
        return self._spans.get(span_id)
    
    def get_trace_spans(self, trace_id: str) -> list[Span]:
        """Get all spans for a trace."""
        return [s for s in self._spans.values() if s.trace_id == trace_id]
    
    def export_trace(self, trace_id: str) -> list[dict[str, Any]]:
        """Export a trace as JSON-serializable data."""
        spans = self.get_trace_spans(trace_id)
        return [
            {
                "span_id": s.span_id,
                "trace_id": s.trace_id,
                "parent_id": s.parent_id,
                "name": s.name,
                "start_time": s.start_time.isoformat(),
                "end_time": s.end_time.isoformat() if s.end_time else None,
                "duration_ms": s.duration_ms,
                "status": s.status.value,
                "attributes": s.attributes,
                "events": s.events,
            }
            for s in spans
        ]
    
    def clear(self) -> None:
        """Clear all collected spans."""
        self._spans.clear()
        self._span_stack.clear()
        self._current_trace = None
