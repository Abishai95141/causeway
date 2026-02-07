"""Decision query and recommendation models for Mode 2."""

from datetime import datetime, timezone
from typing import Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from src.models.enums import ConfidenceLevel
from src.models.causal import CausalPath


class DecisionQuery(BaseModel):
    """
    Structured representation of a user's decision question.
    
    Parsed from natural language input in Mode 2.
    """
    
    query_id: UUID = Field(default_factory=uuid4)
    text: str = Field(..., description="Original user question")
    objective: str = Field(..., description="What user wants to achieve")
    levers: list[str] = Field(
        default_factory=list,
        description="Variables the user can control"
    )
    constraints: list[str] = Field(
        default_factory=list,
        description="Known limitations/constraints"
    )
    context: dict[str, str] = Field(
        default_factory=dict,
        description="Additional context (timeframe, proposed changes, etc.)"
    )
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "text": "Should we raise prices 10% next quarter?",
                "objective": "Maximize revenue while maintaining market share",
                "levers": ["price"],
                "constraints": ["Competitor pricing", "Customer retention targets"],
                "context": {
                    "proposed_change": "10% increase",
                    "timeframe": "next quarter"
                }
            }
        }
    }


class DecisionRecommendation(BaseModel):
    """
    Decision recommendation synthesized from causal reasoning.
    
    Includes full reasoning trace and evidence citations.
    """
    
    recommendation: str = Field(..., description="Clear action statement")
    confidence: ConfidenceLevel = Field(..., description="Confidence classification")
    expected_outcome: str = Field(..., description="What we expect to happen")
    causal_paths: list[CausalPath] = Field(
        default_factory=list,
        description="Causal paths analyzed"
    )
    evidence_refs: list[UUID] = Field(
        default_factory=list,
        description="EvidenceBundle IDs supporting recommendation"
    )
    risks: list[str] = Field(
        default_factory=list,
        description="Identified risks"
    )
    unmeasured_factors: list[str] = Field(
        default_factory=list,
        description="Latent/unmeasured variables that could affect outcome"
    )
    suggested_actions: list[str] = Field(
        default_factory=list,
        description="Recommended next steps"
    )
    suggested_data_collection: Optional[list[str]] = Field(
        default=None,
        description="Data to collect for reducing uncertainty"
    )
    reasoning_trace: Optional[str] = Field(
        default=None,
        description="Detailed reasoning narrative"
    )
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "recommendation": "Delay price increase until Q2. Monitor competitor response.",
                "confidence": "medium",
                "expected_outcome": "Maintain market share while gathering intelligence",
                "risks": ["Revenue opportunity cost if delay too long"],
                "suggested_actions": ["Conduct customer price sensitivity survey"],
            }
        }
    }


class EscalationResponse(BaseModel):
    """
    Response when Mode 2 determines a Mode 1 update is needed.
    
    Triggered when:
    - Critical variable missing from model
    - New evidence contradicts model structure
    - Model age > staleness threshold
    """
    
    escalation_id: UUID = Field(default_factory=uuid4)
    message: str = Field(
        default="World model update recommended",
        description="User-facing message"
    )
    reason: str = Field(..., description="Why escalation is triggered")
    original_query: Optional[DecisionQuery] = Field(
        default=None,
        description="Original decision query"
    )
    suggested_mode1_scope: Optional[dict] = Field(
        default=None,
        description="Pre-populated scope for Mode 1 run"
    )
    conflicts_detected: list[dict] = Field(
        default_factory=list,
        description="Model conflicts that triggered escalation"
    )
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
