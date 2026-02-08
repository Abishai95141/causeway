"""
Module 1: Data Models & Schema Layer

Core Pydantic models for the Causeway decision support system.
All modules import from here - no circular dependencies allowed.
"""

from src.models.enums import (
    IngestionStatus,
    EvidenceStrength,
    ModelStatus,
    RetrievalMethod,
    VariableType,
    VariableRole,
    MeasurementStatus,
    ConfidenceLevel,
    OperatingMode,
)
from src.models.documents import DocumentRecord
from src.models.evidence import (
    SourceReference,
    LocationMetadata,
    RetrievalTrace,
    EvidenceBundle,
)
from src.models.causal import (
    VariableDefinition,
    EdgeMetadata,
    CausalEdge,
    CausalPath,
    WorldModelVersion,
)
from src.models.decision import (
    DecisionQuery,
    DecisionRecommendation,
    EscalationResponse,
)
from src.models.audit import AuditEntry

__all__ = [
    # Enums
    "IngestionStatus",
    "EvidenceStrength",
    "ModelStatus",
    "RetrievalMethod",
    "VariableType",
    "VariableRole",
    "MeasurementStatus",
    "ConfidenceLevel",
    "OperatingMode",
    # Documents
    "DocumentRecord",
    # Evidence
    "SourceReference",
    "LocationMetadata",
    "RetrievalTrace",
    "EvidenceBundle",
    # Causal
    "VariableDefinition",
    "EdgeMetadata",
    "CausalEdge",
    "CausalPath",
    "WorldModelVersion",
    # Decision
    "DecisionQuery",
    "DecisionRecommendation",
    "EscalationResponse",
    # Audit
    "AuditEntry",
]
