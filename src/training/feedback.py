"""
Feedback Collection for Decision Outcomes

Provides:
- Outcome feedback tracking: record whether decisions led to good results
- Decision-edge linking: map decision outcomes back to causal edges
- Temporal reward signals: feed outcome data into training loop
- Edge-level success tracking: which edges reliably predict outcomes

Used in:
- Post-decision follow-up: user confirms outcome after N days
- Training loop: outcome feedback as delayed reward signal
- Model improvement: edges with poor track records get flagged
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional
from uuid import uuid4

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class OutcomeResult(str, Enum):
    """Result of a decision outcome."""

    POSITIVE = "positive"
    """Decision achieved the intended outcome."""

    NEUTRAL = "neutral"
    """No clear positive or negative result."""

    NEGATIVE = "negative"
    """Decision did not achieve the intended outcome."""

    UNEXPECTED = "unexpected"
    """Outcome was different from any prediction."""

    PENDING = "pending"
    """Outcome not yet known."""


class FeedbackSource(str, Enum):
    """Source of the feedback signal."""

    HUMAN = "human"
    """Manual feedback from a user."""

    METRIC = "metric"
    """Automated feedback from a measured KPI."""

    SYSTEM = "system"
    """System-generated (e.g. from follow-up retrieval)."""


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class OutcomeFeedback:
    """Feedback about the outcome of a decision."""

    feedback_id: str = field(default_factory=lambda: f"fb_{uuid4().hex[:12]}")
    decision_trace_id: str = ""
    """The Mode 2 trace_id of the decision this feedback refers to."""
    domain: str = ""
    result: OutcomeResult = OutcomeResult.PENDING
    description: str = ""
    """Free-form description of what happened."""
    edges_involved: list[tuple[str, str]] = field(default_factory=list)
    """Causal edges that were part of the reasoning."""
    predicted_outcome: str = ""
    actual_outcome: str = ""
    reward_delta: float = 0.0
    """Reward adjustment: +1.0 (perfect prediction) to -1.0 (opposite of predicted)."""
    source: FeedbackSource = FeedbackSource.HUMAN
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "feedback_id": self.feedback_id,
            "decision_trace_id": self.decision_trace_id,
            "domain": self.domain,
            "result": self.result.value,
            "description": self.description,
            "edges_involved": [f"{e[0]} → {e[1]}" for e in self.edges_involved],
            "predicted_outcome": self.predicted_outcome,
            "actual_outcome": self.actual_outcome,
            "reward_delta": self.reward_delta,
            "source": self.source.value,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class EdgeTrackRecord:
    """Track record of a single edge across decisions."""

    from_var: str
    to_var: str
    total_decisions: int = 0
    positive_outcomes: int = 0
    negative_outcomes: int = 0
    neutral_outcomes: int = 0
    avg_reward_delta: float = 0.0
    last_feedback_at: Optional[datetime] = None

    @property
    def success_rate(self) -> float:
        """Fraction of decisions with positive outcomes."""
        if self.total_decisions == 0:
            return 0.0
        return self.positive_outcomes / self.total_decisions

    @property
    def reliability_score(self) -> float:
        """
        Reliability score: biased toward positive outcomes.

        Returns 0.0-1.0 where higher means more reliable.
        Uses Wilson score lower bound for small-sample robustness.
        """
        n = self.total_decisions
        if n == 0:
            return 0.5  # No data → neutral prior

        p = self.positive_outcomes / n
        # Wilson score interval lower bound (95% confidence)
        z = 1.96
        denominator = 1 + z * z / n
        centre = p + z * z / (2 * n)
        spread = z * ((p * (1 - p) + z * z / (4 * n)) / n) ** 0.5
        return max(0.0, (centre - spread) / denominator)

    def to_dict(self) -> dict[str, Any]:
        return {
            "edge": f"{self.from_var} → {self.to_var}",
            "total_decisions": self.total_decisions,
            "positive": self.positive_outcomes,
            "negative": self.negative_outcomes,
            "neutral": self.neutral_outcomes,
            "success_rate": round(self.success_rate, 3),
            "reliability_score": round(self.reliability_score, 3),
            "avg_reward_delta": round(self.avg_reward_delta, 3),
        }


@dataclass
class FeedbackSummary:
    """Summary of all feedback for a domain."""

    domain: str
    total_feedback: int = 0
    positive_count: int = 0
    negative_count: int = 0
    avg_reward: float = 0.0
    low_reliability_edges: list[EdgeTrackRecord] = field(default_factory=list)
    high_reliability_edges: list[EdgeTrackRecord] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Feedback Collector
# ---------------------------------------------------------------------------


class FeedbackCollector:
    """
    Collects and analyses outcome feedback for decisions.

    Features:
    - Record outcome feedback per decision trace
    - Track per-edge success/failure rates
    - Identify unreliable edges for re-validation
    - Generate reward deltas for training loop
    """

    def __init__(self):
        self._feedback: dict[str, OutcomeFeedback] = {}
        self._edge_records: dict[tuple[str, str], EdgeTrackRecord] = {}

    # ---- feedback recording ---- #

    def record_feedback(self, feedback: OutcomeFeedback) -> str:
        """
        Record outcome feedback for a decision.

        Updates per-edge track records automatically.
        """
        self._feedback[feedback.feedback_id] = feedback

        # Update edge track records
        for edge_key in feedback.edges_involved:
            record = self._edge_records.setdefault(
                edge_key,
                EdgeTrackRecord(from_var=edge_key[0], to_var=edge_key[1]),
            )
            record.total_decisions += 1
            record.last_feedback_at = feedback.created_at

            if feedback.result == OutcomeResult.POSITIVE:
                record.positive_outcomes += 1
            elif feedback.result == OutcomeResult.NEGATIVE:
                record.negative_outcomes += 1
            else:
                record.neutral_outcomes += 1

            # Update rolling average reward
            n = record.total_decisions
            record.avg_reward_delta = (
                record.avg_reward_delta * (n - 1) + feedback.reward_delta
            ) / n

        logger.info(
            "Recorded feedback %s for decision %s: %s (reward_delta=%.2f)",
            feedback.feedback_id,
            feedback.decision_trace_id,
            feedback.result.value,
            feedback.reward_delta,
        )
        return feedback.feedback_id

    # ---- queries ---- #

    def get_feedback(self, feedback_id: str) -> OutcomeFeedback | None:
        return self._feedback.get(feedback_id)

    def get_feedback_for_decision(
        self, decision_trace_id: str,
    ) -> list[OutcomeFeedback]:
        """Get all feedback for a given decision trace."""
        return [
            fb for fb in self._feedback.values()
            if fb.decision_trace_id == decision_trace_id
        ]

    def get_edge_record(
        self, from_var: str, to_var: str,
    ) -> EdgeTrackRecord | None:
        return self._edge_records.get((from_var, to_var))

    def get_low_reliability_edges(
        self,
        threshold: float = 0.4,
        min_decisions: int = 2,
    ) -> list[EdgeTrackRecord]:
        """Get edges with reliability below threshold."""
        return [
            r for r in self._edge_records.values()
            if r.total_decisions >= min_decisions and r.reliability_score < threshold
        ]

    def get_high_reliability_edges(
        self,
        threshold: float = 0.7,
        min_decisions: int = 2,
    ) -> list[EdgeTrackRecord]:
        """Get edges with reliability above threshold."""
        return [
            r for r in self._edge_records.values()
            if r.total_decisions >= min_decisions and r.reliability_score >= threshold
        ]

    # ---- summaries ---- #

    def get_summary(self, domain: str = "") -> FeedbackSummary:
        """Get summary of all feedback, optionally filtered by domain."""
        feedbacks = (
            [f for f in self._feedback.values() if f.domain == domain]
            if domain
            else list(self._feedback.values())
        )

        positive = sum(1 for f in feedbacks if f.result == OutcomeResult.POSITIVE)
        negative = sum(1 for f in feedbacks if f.result == OutcomeResult.NEGATIVE)
        avg_reward = (
            sum(f.reward_delta for f in feedbacks) / len(feedbacks)
            if feedbacks
            else 0.0
        )

        return FeedbackSummary(
            domain=domain,
            total_feedback=len(feedbacks),
            positive_count=positive,
            negative_count=negative,
            avg_reward=avg_reward,
            low_reliability_edges=self.get_low_reliability_edges(),
            high_reliability_edges=self.get_high_reliability_edges(),
        )

    def compute_training_reward(
        self, decision_trace_id: str,
    ) -> float:
        """
        Compute a training reward signal from outcome feedback.

        Aggregates all feedback for a decision into a single reward.
        """
        feedbacks = self.get_feedback_for_decision(decision_trace_id)
        if not feedbacks:
            return 0.0

        # Weighted average: more recent feedback counts more
        total_weight = 0.0
        weighted_sum = 0.0
        for i, fb in enumerate(sorted(feedbacks, key=lambda f: f.created_at)):
            weight = 1.0 + i * 0.5  # Increasing weight for later feedback
            weighted_sum += fb.reward_delta * weight
            total_weight += weight

        return weighted_sum / total_weight if total_weight > 0 else 0.0

    # ---- lifecycle ---- #

    @property
    def feedback_count(self) -> int:
        return len(self._feedback)

    @property
    def tracked_edge_count(self) -> int:
        return len(self._edge_records)

    def clear(self) -> None:
        """Clear all feedback data."""
        self._feedback.clear()
        self._edge_records.clear()
