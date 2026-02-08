"""
Temporal Tracking for Causal World Models

Provides:
- Edge-level temporal metadata: when created, last validated, last updated
- Model-level staleness detection with configurable thresholds
- Confidence decay: edge confidence degrades over time without re-validation
- Freshness scoring: quantify how up-to-date a model is
- Staleness reports: identify which edges/variables need attention

Used in:
- Mode 2 Step 8: escalation check (world_model_age_days > staleness_threshold)
- Mode 1: track when evidence was last validated
- CausalService: automatic staleness flagging
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from src.models.enums import EvidenceStrength

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class EdgeTemporalMetadata:
    """Temporal tracking data for a single causal edge."""

    from_var: str
    to_var: str
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_validated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    validation_count: int = 0
    original_confidence: float = 0.5
    current_confidence: float = 0.5

    @property
    def edge_key(self) -> tuple[str, str]:
        return (self.from_var, self.to_var)

    @property
    def age_days(self) -> float:
        """Days since edge was created."""
        delta = datetime.now(timezone.utc) - self.created_at
        return delta.total_seconds() / 86400

    @property
    def days_since_validation(self) -> float:
        """Days since edge was last validated."""
        delta = datetime.now(timezone.utc) - self.last_validated_at
        return delta.total_seconds() / 86400

    def to_dict(self) -> dict[str, Any]:
        return {
            "edge": f"{self.from_var} → {self.to_var}",
            "created_at": self.created_at.isoformat(),
            "last_validated_at": self.last_validated_at.isoformat(),
            "last_updated_at": self.last_updated_at.isoformat(),
            "validation_count": self.validation_count,
            "original_confidence": self.original_confidence,
            "current_confidence": round(self.current_confidence, 4),
            "age_days": round(self.age_days, 1),
            "days_since_validation": round(self.days_since_validation, 1),
        }


@dataclass
class StalenessReport:
    """Report on model freshness/staleness."""

    domain: str
    model_age_days: float
    staleness_threshold_days: float
    is_stale: bool
    overall_freshness: float  # 0.0 (completely stale) to 1.0 (fully fresh)
    stale_edges: list[EdgeTemporalMetadata] = field(default_factory=list)
    fresh_edges: list[EdgeTemporalMetadata] = field(default_factory=list)
    edges_needing_validation: list[EdgeTemporalMetadata] = field(default_factory=list)
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def stale_edge_count(self) -> int:
        return len(self.stale_edges)

    @property
    def total_edges(self) -> int:
        return len(self.stale_edges) + len(self.fresh_edges)

    @property
    def stale_ratio(self) -> float:
        if self.total_edges == 0:
            return 0.0
        return self.stale_edge_count / self.total_edges

    def to_dict(self) -> dict[str, Any]:
        return {
            "domain": self.domain,
            "model_age_days": round(self.model_age_days, 1),
            "is_stale": self.is_stale,
            "overall_freshness": round(self.overall_freshness, 4),
            "staleness_threshold_days": self.staleness_threshold_days,
            "total_edges": self.total_edges,
            "stale_edge_count": self.stale_edge_count,
            "stale_ratio": round(self.stale_ratio, 3),
            "edges_needing_validation": len(self.edges_needing_validation),
        }


# ---------------------------------------------------------------------------
# Temporal Tracker
# ---------------------------------------------------------------------------


class TemporalTracker:
    """
    Tracks temporal metadata for all edges in a causal world model.

    Features:
    - Record edge creation, validation, and update timestamps
    - Compute confidence decay over time (exponential decay)
    - Generate staleness reports
    - Identify edges needing re-validation
    """

    def __init__(
        self,
        staleness_threshold_days: float = 30.0,
        decay_half_life_days: float = 60.0,
        validation_freshness_days: float = 14.0,
    ):
        """
        Args:
            staleness_threshold_days: Model age beyond which it's considered stale.
            decay_half_life_days: Days for confidence to decay to 50% of original.
            validation_freshness_days: Days after which an edge needs re-validation.
        """
        self.staleness_threshold_days = staleness_threshold_days
        self.decay_half_life_days = decay_half_life_days
        self.validation_freshness_days = validation_freshness_days

        self._edge_metadata: dict[tuple[str, str], EdgeTemporalMetadata] = {}
        self._model_created_at: datetime | None = None

    # ---- edge lifecycle ---- #

    def record_edge_created(
        self,
        from_var: str,
        to_var: str,
        confidence: float = 0.5,
        created_at: datetime | None = None,
    ) -> EdgeTemporalMetadata:
        """Record that an edge was just created."""
        now = created_at or datetime.now(timezone.utc)
        meta = EdgeTemporalMetadata(
            from_var=from_var,
            to_var=to_var,
            created_at=now,
            last_validated_at=now,
            last_updated_at=now,
            validation_count=0,
            original_confidence=confidence,
            current_confidence=confidence,
        )
        self._edge_metadata[(from_var, to_var)] = meta

        if self._model_created_at is None:
            self._model_created_at = now

        return meta

    def record_edge_validated(
        self,
        from_var: str,
        to_var: str,
        new_confidence: float | None = None,
        validated_at: datetime | None = None,
    ) -> EdgeTemporalMetadata | None:
        """Record that an edge was re-validated with fresh evidence."""
        key = (from_var, to_var)
        meta = self._edge_metadata.get(key)
        if meta is None:
            return None

        now = validated_at or datetime.now(timezone.utc)
        meta.last_validated_at = now
        meta.last_updated_at = now
        meta.validation_count += 1
        if new_confidence is not None:
            meta.original_confidence = new_confidence
            meta.current_confidence = new_confidence
        else:
            # Reset decay — validation restores original confidence
            meta.current_confidence = meta.original_confidence

        return meta

    def record_edge_updated(
        self,
        from_var: str,
        to_var: str,
        new_confidence: float | None = None,
        updated_at: datetime | None = None,
    ) -> EdgeTemporalMetadata | None:
        """Record a structural update to an edge (mechanism change, etc.)."""
        key = (from_var, to_var)
        meta = self._edge_metadata.get(key)
        if meta is None:
            return None

        now = updated_at or datetime.now(timezone.utc)
        meta.last_updated_at = now
        if new_confidence is not None:
            meta.original_confidence = new_confidence
            meta.current_confidence = new_confidence

        return meta

    def remove_edge(self, from_var: str, to_var: str) -> None:
        """Remove temporal tracking for an edge."""
        self._edge_metadata.pop((from_var, to_var), None)

    # ---- confidence decay ---- #

    def compute_decayed_confidence(
        self,
        from_var: str,
        to_var: str,
        reference_time: datetime | None = None,
    ) -> float:
        """
        Compute confidence after exponential time decay.

        Uses exponential decay: c(t) = c_0 * exp(-λ * t)
        where λ = ln(2) / half_life and t = days since last validation.
        """
        key = (from_var, to_var)
        meta = self._edge_metadata.get(key)
        if meta is None:
            return 0.0

        now = reference_time or datetime.now(timezone.utc)
        days_since = (now - meta.last_validated_at).total_seconds() / 86400

        if days_since <= 0:
            return meta.original_confidence

        decay_constant = math.log(2) / self.decay_half_life_days
        decayed = meta.original_confidence * math.exp(-decay_constant * days_since)
        return max(0.0, min(1.0, decayed))

    def apply_decay(
        self,
        reference_time: datetime | None = None,
    ) -> dict[tuple[str, str], float]:
        """
        Apply confidence decay to all tracked edges.

        Returns mapping of edge -> new decayed confidence.
        """
        now = reference_time or datetime.now(timezone.utc)
        updates: dict[tuple[str, str], float] = {}

        for key, meta in self._edge_metadata.items():
            decayed = self.compute_decayed_confidence(
                meta.from_var, meta.to_var, reference_time=now,
            )
            meta.current_confidence = decayed
            updates[key] = decayed

        return updates

    # ---- staleness analysis ---- #

    def check_staleness(
        self,
        domain: str = "",
        reference_time: datetime | None = None,
    ) -> StalenessReport:
        """
        Generate a staleness report for the model.

        Checks:
        1. Overall model age vs threshold
        2. Per-edge validation freshness
        3. Confidence decay levels
        """
        now = reference_time or datetime.now(timezone.utc)

        # Model age
        if self._model_created_at:
            model_age = (now - self._model_created_at).total_seconds() / 86400
        else:
            model_age = 0.0

        is_stale = model_age > self.staleness_threshold_days

        stale_edges: list[EdgeTemporalMetadata] = []
        fresh_edges: list[EdgeTemporalMetadata] = []
        needs_validation: list[EdgeTemporalMetadata] = []

        freshness_scores: list[float] = []

        for meta in self._edge_metadata.values():
            days_since_val = (now - meta.last_validated_at).total_seconds() / 86400

            # Apply decay to get current confidence
            decayed = self.compute_decayed_confidence(
                meta.from_var, meta.to_var, reference_time=now,
            )
            meta.current_confidence = decayed

            # Edge freshness = 1.0 when just validated, → 0.0 over time
            edge_freshness = max(
                0.0,
                1.0 - (days_since_val / (self.staleness_threshold_days * 2)),
            )
            freshness_scores.append(edge_freshness)

            if days_since_val > self.staleness_threshold_days:
                stale_edges.append(meta)
                is_stale = True
            else:
                fresh_edges.append(meta)

            if days_since_val > self.validation_freshness_days:
                needs_validation.append(meta)

        overall_freshness = (
            sum(freshness_scores) / len(freshness_scores)
            if freshness_scores
            else 1.0
        )

        return StalenessReport(
            domain=domain,
            model_age_days=model_age,
            staleness_threshold_days=self.staleness_threshold_days,
            is_stale=is_stale,
            overall_freshness=overall_freshness,
            stale_edges=stale_edges,
            fresh_edges=fresh_edges,
            edges_needing_validation=needs_validation,
        )

    # ---- queries ---- #

    def get_edge_metadata(
        self, from_var: str, to_var: str,
    ) -> EdgeTemporalMetadata | None:
        return self._edge_metadata.get((from_var, to_var))

    def get_all_metadata(self) -> list[EdgeTemporalMetadata]:
        return list(self._edge_metadata.values())

    @property
    def model_age_days(self) -> float:
        if self._model_created_at is None:
            return 0.0
        delta = datetime.now(timezone.utc) - self._model_created_at
        return delta.total_seconds() / 86400

    def set_model_created_at(self, dt: datetime) -> None:
        """Set the model creation timestamp (e.g. when loading from DB)."""
        self._model_created_at = dt

    def clear(self) -> None:
        """Clear all temporal metadata."""
        self._edge_metadata.clear()
        self._model_created_at = None
