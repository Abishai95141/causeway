"""
Trajectory Storage

Stores execution trajectories for offline training.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional
from uuid import uuid4
import json


@dataclass
class Trajectory:
    """A complete execution trajectory for training."""
    trajectory_id: str
    trace_id: str
    mode: str  # "mode1" or "mode2"
    input_data: dict[str, Any]
    spans: list[dict[str, Any]]
    outcome: dict[str, Any]
    reward: Optional[float] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "trajectory_id": self.trajectory_id,
            "trace_id": self.trace_id,
            "mode": self.mode,
            "input_data": self.input_data,
            "spans": self.spans,
            "outcome": self.outcome,
            "reward": self.reward,
            "created_at": self.created_at.isoformat(),
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Trajectory":
        """Create from dictionary."""
        return cls(
            trajectory_id=data["trajectory_id"],
            trace_id=data["trace_id"],
            mode=data["mode"],
            input_data=data["input_data"],
            spans=data["spans"],
            outcome=data["outcome"],
            reward=data.get("reward"),
            created_at=datetime.fromisoformat(data["created_at"]),
        )


class TrajectoryStore:
    """
    In-memory trajectory storage.
    
    Can be extended to persist to SQLite/PostgreSQL for larger datasets.
    """
    
    def __init__(self, max_trajectories: int = 10000):
        self.max_trajectories = max_trajectories
        self._store: dict[str, Trajectory] = {}
    
    def save(self, trajectory: Trajectory) -> str:
        """Save a trajectory."""
        if not trajectory.trajectory_id:
            trajectory.trajectory_id = f"traj_{uuid4().hex[:12]}"
        
        self._store[trajectory.trajectory_id] = trajectory
        
        # Trim old trajectories
        if len(self._store) > self.max_trajectories:
            oldest = sorted(
                self._store.values(),
                key=lambda t: t.created_at
            )[:100]
            for t in oldest:
                del self._store[t.trajectory_id]
        
        return trajectory.trajectory_id
    
    def get(self, trajectory_id: str) -> Optional[Trajectory]:
        """Get a trajectory by ID."""
        return self._store.get(trajectory_id)
    
    def list_by_mode(self, mode: str, limit: int = 100) -> list[Trajectory]:
        """List trajectories by mode."""
        trajectories = [
            t for t in self._store.values()
            if t.mode == mode
        ]
        trajectories.sort(key=lambda t: t.created_at, reverse=True)
        return trajectories[:limit]
    
    def list_positive_examples(
        self,
        reward_threshold: float = 0.7,
        limit: int = 100,
    ) -> list[Trajectory]:
        """List high-reward trajectories for training."""
        trajectories = [
            t for t in self._store.values()
            if t.reward is not None and t.reward >= reward_threshold
        ]
        trajectories.sort(key=lambda t: t.reward or 0, reverse=True)
        return trajectories[:limit]
    
    def export_jsonl(self, filepath: str) -> int:
        """Export all trajectories to JSONL file."""
        count = 0
        with open(filepath, "w") as f:
            for trajectory in self._store.values():
                f.write(json.dumps(trajectory.to_dict()) + "\n")
                count += 1
        return count
    
    def import_jsonl(self, filepath: str) -> int:
        """Import trajectories from JSONL file."""
        count = 0
        with open(filepath, "r") as f:
            for line in f:
                data = json.loads(line.strip())
                trajectory = Trajectory.from_dict(data)
                self._store[trajectory.trajectory_id] = trajectory
                count += 1
        return count
    
    def count(self) -> int:
        """Get total trajectory count."""
        return len(self._store)
    
    def clear(self) -> None:
        """Clear all trajectories."""
        self._store.clear()
