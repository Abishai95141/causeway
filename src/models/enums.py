"""Enumeration types for the Causeway system."""

from enum import Enum


class IngestionStatus(str, Enum):
    """Status of document ingestion into retrieval systems."""
    PENDING = "pending"
    INDEXING = "indexing"
    INDEXED = "indexed"
    FAILED = "failed"


class EvidenceStrength(str, Enum):
    """Classification of evidence support for causal edges."""
    STRONG = "strong"          # 3+ supporting sources
    MODERATE = "moderate"      # 2 supporting sources
    HYPOTHESIS = "hypothesis"  # 1 supporting source
    CONTESTED = "contested"    # Has contradicting evidence


class ModelStatus(str, Enum):
    """Status of a WorldModelVersion."""
    DRAFT = "draft"
    REVIEW = "review"
    ACTIVE = "active"
    DEPRECATED = "deprecated"


class EdgeStatus(str, Enum):
    """Verification status of a causal edge."""
    DRAFT = "draft"          # Initial extraction, not yet verified
    GROUNDED = "grounded"    # Passed verification judge
    REJECTED = "rejected"    # Failed verification (soft-pruned)


class RetrievalMethod(str, Enum):
    """Method used to retrieve evidence."""
    PAGEINDEX = "pageindex"
    HAYSTACK = "haystack"
    BOTH = "both"


class VariableType(str, Enum):
    """Type of causal variable."""
    CONTINUOUS = "continuous"
    DISCRETE = "discrete"
    BINARY = "binary"
    CATEGORICAL = "categorical"


class MeasurementStatus(str, Enum):
    """Whether a variable is directly measurable."""
    MEASURED = "measured"      # Have direct data
    OBSERVABLE = "observable"  # Can observe but no systematic data
    LATENT = "latent"          # Cannot directly observe


class VariableRole(str, Enum):
    """Causal role of a variable in a DAG."""
    TREATMENT = "treatment"        # Manipulable intervention variable
    OUTCOME = "outcome"            # Target outcome variable
    CONFOUNDER = "confounder"      # Common cause of treatment and outcome
    MEDIATOR = "mediator"          # On the causal path between treatment and outcome
    INSTRUMENTAL = "instrumental"  # Affects treatment but not outcome directly
    COVARIATE = "covariate"        # Other measured variable
    UNKNOWN = "unknown"            # Role not yet classified


class ConfidenceLevel(str, Enum):
    """Confidence level for recommendations."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class OperatingMode(str, Enum):
    """System operating mode."""
    MODE_1 = "world_model_construction"
    MODE_2 = "decision_support"


class ProtocolState(str, Enum):
    """State machine states for protocol engine."""
    IDLE = "idle"
    ROUTING = "routing"
    # Mode 1 states
    WM_DISCOVERY_RUNNING = "wm_discovery_running"
    WM_REVIEW_PENDING = "wm_review_pending"
    WM_ACTIVE = "wm_active"
    # Mode 2 states
    DECISION_SUPPORT_RUNNING = "decision_support_running"
    RESPONSE_READY = "response_ready"
    # Error state
    ERROR = "error"
