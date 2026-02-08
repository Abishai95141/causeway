"""Training package for Agent Lightning integration (stubs)."""

from src.training.spans import SpanCollector
from src.training.rewards import RewardFunction
from src.training.trajectories import TrajectoryStore
from src.training.feedback import (
    FeedbackCollector,
    OutcomeFeedback,
    EdgeTrackRecord,
    FeedbackSummary,
    OutcomeResult,
    FeedbackSource,
)

__all__ = [
    "SpanCollector",
    "RewardFunction",
    "TrajectoryStore",
    "FeedbackCollector",
    "OutcomeFeedback",
    "EdgeTrackRecord",
    "FeedbackSummary",
    "OutcomeResult",
    "FeedbackSource",
]
