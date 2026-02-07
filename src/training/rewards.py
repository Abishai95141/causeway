"""
Reward Function Interface

Interface for computing rewards from execution trajectories.
Used by Agent Lightning training loop for RL fine-tuning.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

from src.training.spans import Span


@dataclass
class RewardSignal:
    """Computed reward signal for a trajectory."""
    reward: float
    components: dict[str, float]
    explanation: str
    trajectory_id: str


class RewardFunction(ABC):
    """
    Abstract base class for reward computation.
    
    Implementations can provide domain-specific rewards
    based on causal reasoning quality, evidence usage, etc.
    """
    
    @abstractmethod
    def compute(
        self,
        trajectory_id: str,
        spans: list[Span],
        outcome: dict[str, Any],
    ) -> RewardSignal:
        """
        Compute reward for a trajectory.
        
        Args:
            trajectory_id: ID of the trajectory
            spans: Execution spans from the trajectory
            outcome: Final outcome/result of the trajectory
            
        Returns:
            RewardSignal with computed reward
        """
        pass


class DefaultRewardFunction(RewardFunction):
    """
    Default reward function implementation.
    
    Computes rewards based on:
    - Task completion (did it finish?)
    - Evidence usage (did it cite evidence?)
    - Causal reasoning (did it trace paths?)
    - Latency (was it fast?)
    """
    
    def __init__(
        self,
        completion_weight: float = 0.4,
        evidence_weight: float = 0.3,
        causal_weight: float = 0.2,
        latency_weight: float = 0.1,
        latency_threshold_ms: float = 30000,
    ):
        self.completion_weight = completion_weight
        self.evidence_weight = evidence_weight
        self.causal_weight = causal_weight
        self.latency_weight = latency_weight
        self.latency_threshold_ms = latency_threshold_ms
    
    def compute(
        self,
        trajectory_id: str,
        spans: list[Span],
        outcome: dict[str, Any],
    ) -> RewardSignal:
        """Compute reward based on multiple factors."""
        components = {}
        
        # Completion reward
        completed = outcome.get("success", False)
        components["completion"] = 1.0 if completed else 0.0
        
        # Evidence reward
        evidence_count = outcome.get("evidence_count", 0)
        components["evidence"] = min(1.0, evidence_count / 5.0)
        
        # Causal reasoning reward
        causal_paths = outcome.get("causal_paths", 0)
        components["causal"] = min(1.0, causal_paths / 3.0)
        
        # Latency reward
        total_duration = sum(
            (s.duration_ms or 0) for s in spans
        )
        if total_duration > 0:
            latency_factor = max(
                0.0,
                1.0 - (total_duration / self.latency_threshold_ms)
            )
        else:
            latency_factor = 1.0
        components["latency"] = latency_factor
        
        # Weighted sum
        reward = (
            self.completion_weight * components["completion"] +
            self.evidence_weight * components["evidence"] +
            self.causal_weight * components["causal"] +
            self.latency_weight * components["latency"]
        )
        
        # Generate explanation
        explanation = self._generate_explanation(components, reward)
        
        return RewardSignal(
            reward=reward,
            components=components,
            explanation=explanation,
            trajectory_id=trajectory_id,
        )
    
    def _generate_explanation(
        self,
        components: dict[str, float],
        reward: float,
    ) -> str:
        """Generate human-readable explanation."""
        parts = []
        
        if components["completion"] == 1.0:
            parts.append("Task completed successfully.")
        else:
            parts.append("Task did not complete.")
        
        if components["evidence"] > 0.5:
            parts.append(f"Good evidence usage ({components['evidence']:.1%}).")
        elif components["evidence"] > 0:
            parts.append(f"Limited evidence usage ({components['evidence']:.1%}).")
        
        if components["causal"] > 0.5:
            parts.append(f"Strong causal reasoning ({components['causal']:.1%}).")
        elif components["causal"] > 0:
            parts.append(f"Some causal reasoning ({components['causal']:.1%}).")
        
        parts.append(f"Overall reward: {reward:.3f}")
        
        return " ".join(parts)
