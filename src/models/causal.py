"""Causal models for world model construction and DAG representation."""

from datetime import datetime, timezone
from typing import Any, Optional
from uuid import UUID

from pydantic import BaseModel, Field, field_validator

from src.models.enums import EdgeStatus, EvidenceStrength, MeasurementStatus, ModelStatus, VariableRole, VariableType


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
    edge_status: EdgeStatus = Field(
        default=EdgeStatus.DRAFT,
        description="Verification status of this edge (draft → grounded | rejected)"
    )
    rejection_reason: Optional[str] = Field(
        default=None,
        description="Why the verification judge rejected this edge (if edge_status=rejected)"
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


# ---------------------------------------------------------------------------
# World Model Patch (incremental update)
# ---------------------------------------------------------------------------

class EdgeUpdate(BaseModel):
    """Partial update for an existing edge's metadata."""
    from_var: str
    to_var: str
    mechanism: Optional[str] = None
    evidence_strength: Optional[EvidenceStrength] = None
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class WorldModelPatch(BaseModel):
    """Incremental patch to apply to an existing world model."""
    add_variables: list[VariableDefinition] = Field(default_factory=list)
    remove_variables: list[str] = Field(default_factory=list, description="variable_ids to remove")
    add_edges: list[CausalEdge] = Field(default_factory=list)
    remove_edges: list[dict[str, str]] = Field(
        default_factory=list,
        description="List of {from_var, to_var} dicts",
    )
    update_edges: list[EdgeUpdate] = Field(default_factory=list)


class WorldModelUpdateResult(BaseModel):
    """Result of applying a patch to a world model."""
    old_version_id: str
    new_version_id: str
    variables_added: int = 0
    variables_removed: int = 0
    edges_added: int = 0
    edges_removed: int = 0
    edges_updated: int = 0
    conflicts: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Cross-Model Bridge models
# ---------------------------------------------------------------------------

class ConceptMapping(BaseModel):
    """A mapping between semantically equivalent variables across models."""
    source_var: str
    target_var: str
    similarity_score: float = Field(ge=0.0, le=1.0)
    mapping_rationale: str = ""


class BridgeEdge(BaseModel):
    """A directed causal edge spanning two different domain models."""
    source_domain: str
    source_var: str
    target_domain: str
    target_var: str
    mechanism: str = ""
    strength: EvidenceStrength = EvidenceStrength.HYPOTHESIS
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    mapping_rationale: str = ""


class ModelBridge(BaseModel):
    """A bridge linking two world models via shared/causal concepts."""
    bridge_id: str
    source_version_id: str
    source_domain: str
    target_version_id: str
    target_domain: str
    bridge_edges: list[BridgeEdge] = Field(default_factory=list)
    shared_concepts: list[ConceptMapping] = Field(default_factory=list)
    status: ModelStatus = ModelStatus.DRAFT
    created_at: Optional[datetime] = None
