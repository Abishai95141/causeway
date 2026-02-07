"""Audit log entry model for full traceability."""

from datetime import datetime, timezone
from typing import Any, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from src.models.enums import OperatingMode


class AuditEntry(BaseModel):
    """
    Audit log entry for every Mode 1/Mode 2 run.
    
    Provides full traceability for debugging and compliance.
    Append-only - entries are never modified or deleted.
    """
    
    audit_id: UUID = Field(default_factory=uuid4)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    mode: OperatingMode = Field(..., description="Which mode was executed")
    trace_id: str = Field(
        ...,
        description="Trace ID linking to Agent Lightning spans"
    )
    input_query: str = Field(..., description="Original user input/request")
    input_metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional input context"
    )
    retrieval_bundle_ids: list[UUID] = Field(
        default_factory=list,
        description="EvidenceBundle IDs used in this run"
    )
    reasoning_steps: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Protocol execution trace"
    )
    output_type: str = Field(..., description="Type of output (WorldModelVersion, Recommendation, etc.)")
    output_data: Optional[dict[str, Any]] = Field(
        default=None,
        description="Summary of output (not full output to save space)"
    )
    output_id: Optional[str] = Field(
        default=None,
        description="ID of created artifact (version_id, recommendation_id)"
    )
    execution_time_ms: Optional[int] = Field(
        default=None,
        ge=0,
        description="Total execution time in milliseconds"
    )
    agent_version: str = Field(
        default="v0.1.0",
        description="Version of agent/prompt used"
    )
    error: Optional[str] = Field(
        default=None,
        description="Error message if run failed"
    )
    success: bool = Field(
        default=True,
        description="Whether run completed successfully"
    )
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "audit_id": "550e8400-e29b-41d4-a716-446655440000",
                "mode": "world_model_construction",
                "trace_id": "trace_abc123",
                "input_query": "Build world model for pricing decisions",
                "output_type": "WorldModelVersion",
                "output_id": "wm_pricing_20260206_143522",
                "execution_time_ms": 45000,
                "success": True,
            }
        }
    }
