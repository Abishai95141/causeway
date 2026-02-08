"""Causal models for world model construction and DAG representation."""

from datetime import datetime, timezone
from typing import Any, Optional
from uuid import UUID

from pydantic import BaseModel, Field, field_validator

from src.models.enums import EvidenceStrength, MeasurementStatus, ModelStatus, VariableRole, VariableType


class VariableDefinition(BaseModel):
    """Definition of a causal variable in the world model."""
    
    variable_id: str = Field(..., min_length=1, description="Unique identifier (snake_case)")
    name: str = Field(..., min_length=1, description="Human-readable name")
    definition: str = Field(..., description="Clear definition of what this variable represents")
    type: VariableType = Field(..., description="Variable type")
    measurement_status: MeasurementStatus = Field(
        ...,
        description="Whether we can measure this variable"
    )
    data_source: Optional[str] = Field(
        default=None,
        description="Where data for this variable comes from"
    )
    unit: Optional[str] = Field(default=None, description="Unit of measurement if applicable")
    role: VariableRole = Field(
        default=VariableRole.UNKNOWN,
        description="Causal role in the DAG (treatment, outcome, confounder, etc.)"
    )
    
    @field_validator("variable_id")
    @classmethod
    def validate_snake_case(cls, v: str) -> str:
        """Ensure variable_id is valid snake_case identifier."""
        import re
        if not re.match(r"^[a-z][a-z0-9_]*$", v):
            raise ValueError("variable_id must be snake_case (lowercase, underscores)")
        return v


class EdgeMetadata(BaseModel):
    """Metadata for a causal edge between variables."""
    
    mechanism: str = Field(..., description="Description of the causal mechanism")
    evidence_strength: EvidenceStrength = Field(
        default=EvidenceStrength.HYPOTHESIS,
        description="Classification based on evidence support"
    )
    evidence_refs: list[UUID] = Field(
        default_factory=list,
        description="EvidenceBundle IDs supporting this edge"
    )
    contradicting_refs: list[UUID] = Field(
        default_factory=list,
        description="EvidenceBundle IDs contradicting this edge"
    )
    assumptions: list[str] = Field(
        default_factory=list,
        description="Assumptions underlying this causal relationship"
    )
    conditions: list[str] = Field(
        default_factory=list,
        description="Conditions under which this relationship holds (e.g., 'segment=price_sensitive')"
    )
    confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Confidence score (0-1)"
    )
    notes: Optional[str] = Field(default=None, description="Additional notes")
    
    def update_strength_from_evidence_count(self, 
                                             supporting: int, 
                                             contradicting: int = 0) -> None:
        """Update evidence_strength based on source counts."""
        if contradicting > 0:
            self.evidence_strength = EvidenceStrength.CONTESTED
        elif supporting >= 3:
            self.evidence_strength = EvidenceStrength.STRONG
        elif supporting == 2:
            self.evidence_strength = EvidenceStrength.MODERATE
        else:
            self.evidence_strength = EvidenceStrength.HYPOTHESIS


class CausalEdge(BaseModel):
    """A directed edge in the causal DAG."""
    
    from_var: str = Field(..., description="Source variable ID")
    to_var: str = Field(..., description="Target variable ID")
    metadata: EdgeMetadata = Field(..., description="Edge metadata")
    
    @property
    def edge_id(self) -> str:
        """Generate unique edge identifier."""
        return f"{self.from_var}->{self.to_var}"


class CausalPath(BaseModel):
    """A path through the causal DAG from lever to outcome."""
    
    path: list[str] = Field(..., min_length=2, description="Ordered list of variable IDs")
    edges: list[CausalEdge] = Field(..., description="Edges along the path")
    mechanism_chain: str = Field(..., description="Natural language description of path")
    strength: str = Field(..., description="Path strength classification")
    
    @property
    def length(self) -> int:
        """Number of edges in path."""
        return len(self.path) - 1
    
    @property
    def mediators(self) -> list[str]:
        """Variables between first and last (mediating variables)."""
        return self.path[1:-1] if len(self.path) > 2 else []


class WorldModelVersion(BaseModel):
    """
    A versioned causal world model for a specific decision domain.
    
    Contains the full DAG structure plus all variable definitions,
    edge metadata, and links to supporting evidence.
    """
    
    version_id: str = Field(..., description="Version ID: wm_{domain}_{timestamp}")
    domain: str = Field(..., min_length=1, description="Decision domain (e.g., 'pricing')")
    description: str = Field(..., description="Description of what this model covers")
    variables: dict[str, VariableDefinition] = Field(
        default_factory=dict,
        description="Variable definitions keyed by variable_id"
    )
    edges: list[CausalEdge] = Field(
        default_factory=list,
        description="All causal edges in the DAG"
    )
    dag_json: dict[str, Any] = Field(
        default_factory=dict,
        description="NetworkX JSON representation of the graph"
    )
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    created_by: str = Field(default="system", description="Creator identifier")
    approved_by: Optional[str] = Field(default=None, description="Approver for review gate")
    approved_at: Optional[datetime] = Field(default=None)
    status: ModelStatus = Field(
        default=ModelStatus.DRAFT,
        description="Current model status"
    )
    replaces_version: Optional[str] = Field(
        default=None,
        description="Previous version ID this replaces"
    )
    
    @field_validator("version_id")
    @classmethod
    def validate_version_format(cls, v: str) -> str:
        """Ensure version_id follows pattern: wm_{domain}_{timestamp}."""
        if not v.startswith("wm_"):
            raise ValueError("version_id must start with 'wm_'")
        return v
    
    @property
    def variable_count(self) -> int:
        """Number of variables in model."""
        return len(self.variables)
    
    @property
    def edge_count(self) -> int:
        """Number of edges in model."""
        return len(self.edges)
    
    def get_all_evidence_refs(self) -> set[UUID]:
        """Get all unique evidence bundle IDs referenced by edges."""
        refs: set[UUID] = set()
        for edge in self.edges:
            refs.update(edge.metadata.evidence_refs)
        return refs
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "version_id": "wm_pricing_20260206_143522",
                "domain": "pricing",
                "description": "Causal model for pricing decisions affecting revenue and retention",
                "status": "active",
            }
        }
    }
