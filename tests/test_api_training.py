"""
Tests for Modules 11-12: API and Training

Tests cover:
- API endpoints (health, metrics, document upload, mode1/2)
- Span collection
- Reward computation
- Trajectory storage
"""

import pytest
from uuid import uuid4
from datetime import datetime, timezone
from io import BytesIO

from fastapi.testclient import TestClient

from src.api.main import app
from src.training.spans import SpanCollector, SpanStatus
from src.training.rewards import DefaultRewardFunction, RewardSignal
from src.training.trajectories import TrajectoryStore, Trajectory


# ===== API Tests =====

class TestHealthEndpoint:
    """Test health check endpoint."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)
    
    def test_health_returns_200(self, client):
        """Should return 200 with health status."""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert data["version"] == "0.1.0"
    
    def test_metrics_returns_200(self, client):
        """Should return metrics."""
        response = client.get("/metrics")
        
        assert response.status_code == 200
        data = response.json()
        assert "uptime_seconds" in data
        assert "request_count" in data


class TestDocumentEndpoints:
    """Test document upload endpoints."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)
    
    def test_get_document_returns_placeholder(self, client):
        """Should return placeholder for document lookup."""
        response = client.get("/api/v1/documents/doc_test123")
        
        assert response.status_code == 200
        data = response.json()
        assert data["doc_id"] == "doc_test123"


class TestWorldModelEndpoints:
    """Test world model endpoints."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)
    
    def test_list_world_models_returns_list(self, client):
        """Should return list of world models."""
        response = client.get("/api/v1/world-models")
        
        assert response.status_code == 200
        assert isinstance(response.json(), list)
    
    def test_get_nonexistent_model_returns_404(self, client):
        """Should return 404 for nonexistent model."""
        response = client.get("/api/v1/world-models/nonexistent_domain")
        
        assert response.status_code == 404


# ===== Span Collector Tests =====

class TestSpanCollector:
    """Test span collection."""
    
    @pytest.fixture
    def collector(self):
        """Create span collector."""
        return SpanCollector()
    
    def test_start_trace(self, collector):
        """Should start a new trace."""
        trace_id = collector.start_trace("test")
        
        assert trace_id.startswith("trace_")
    
    def test_start_and_end_span(self, collector):
        """Should track span lifecycle."""
        trace_id = collector.start_trace()
        span_id = collector.start_span("test_span")
        
        collector.end_span(span_id)
        
        span = collector.get_span(span_id)
        assert span is not None
        assert span.status == SpanStatus.COMPLETED
        assert span.duration_ms is not None
    
    def test_span_hierarchy(self, collector):
        """Should track parent-child relationships."""
        collector.start_trace()
        parent_id = collector.start_span("parent")
        child_id = collector.start_span("child")
        
        child = collector.get_span(child_id)
        assert child.parent_id == parent_id
    
    def test_add_event(self, collector):
        """Should add events to spans."""
        collector.start_trace()
        span_id = collector.start_span("test")
        
        collector.add_event(span_id, "test_event", {"key": "value"})
        
        span = collector.get_span(span_id)
        assert len(span.events) == 1
        assert span.events[0]["name"] == "test_event"
    
    def test_export_trace(self, collector):
        """Should export trace as JSON."""
        trace_id = collector.start_trace()
        span_id = collector.start_span("test")
        collector.end_span(span_id)
        
        exported = collector.export_trace(trace_id)
        
        assert len(exported) >= 1
        # Find the test span (not root)
        span_ids = [s["span_id"] for s in exported]
        assert span_id in span_ids
    
    def test_disabled_collector(self):
        """Should not collect when disabled."""
        collector = SpanCollector(enabled=False)
        
        span_id = collector.start_span("test")
        
        assert span_id == ""
        assert collector.get_span(span_id) is None


# ===== Reward Function Tests =====

class TestDefaultRewardFunction:
    """Test reward computation."""
    
    @pytest.fixture
    def reward_fn(self):
        """Create reward function."""
        return DefaultRewardFunction()
    
    def test_compute_successful_trajectory(self, reward_fn):
        """Should compute high reward for successful trajectory."""
        collector = SpanCollector()
        collector.start_trace()
        span_id = collector.start_span("test")
        collector.end_span(span_id)
        
        spans = [collector.get_span(span_id)]
        outcome = {
            "success": True,
            "evidence_count": 5,
            "causal_paths": 3,
        }
        
        result = reward_fn.compute("traj_test", spans, outcome)
        
        assert isinstance(result, RewardSignal)
        assert result.reward > 0.5  # Should be high
        assert result.components["completion"] == 1.0
    
    def test_compute_failed_trajectory(self, reward_fn):
        """Should compute low reward for failed trajectory."""
        outcome = {"success": False}
        
        result = reward_fn.compute("traj_test", [], outcome)
        
        assert result.components["completion"] == 0.0
        assert result.reward < 0.5
    
    def test_explanation_generated(self, reward_fn):
        """Should generate explanation."""
        outcome = {"success": True, "evidence_count": 3}
        
        result = reward_fn.compute("traj_test", [], outcome)
        
        assert "Task completed" in result.explanation
        assert "reward" in result.explanation.lower()


# ===== Trajectory Store Tests =====

class TestTrajectoryStore:
    """Test trajectory storage."""
    
    @pytest.fixture
    def store(self):
        """Create trajectory store."""
        return TrajectoryStore()
    
    @pytest.fixture
    def sample_trajectory(self):
        """Create sample trajectory."""
        return Trajectory(
            trajectory_id="traj_test",
            trace_id="trace_test",
            mode="mode1",
            input_data={"domain": "pricing"},
            spans=[],
            outcome={"success": True},
            reward=0.8,
        )
    
    def test_save_and_get(self, store, sample_trajectory):
        """Should save and retrieve trajectory."""
        store.save(sample_trajectory)
        
        retrieved = store.get("traj_test")
        
        assert retrieved is not None
        assert retrieved.mode == "mode1"
    
    def test_list_by_mode(self, store, sample_trajectory):
        """Should list trajectories by mode."""
        store.save(sample_trajectory)
        
        mode1_trajs = store.list_by_mode("mode1")
        
        assert len(mode1_trajs) == 1
        assert mode1_trajs[0].trajectory_id == "traj_test"
    
    def test_list_positive_examples(self, store, sample_trajectory):
        """Should filter by reward threshold."""
        sample_trajectory.reward = 0.9
        store.save(sample_trajectory)
        
        low_reward = Trajectory(
            trajectory_id="traj_low",
            trace_id="trace_low",
            mode="mode1",
            input_data={},
            spans=[],
            outcome={},
            reward=0.3,
        )
        store.save(low_reward)
        
        positive = store.list_positive_examples(reward_threshold=0.7)
        
        assert len(positive) == 1
        assert positive[0].trajectory_id == "traj_test"
    
    def test_count(self, store, sample_trajectory):
        """Should count trajectories."""
        assert store.count() == 0
        
        store.save(sample_trajectory)
        
        assert store.count() == 1
    
    def test_clear(self, store, sample_trajectory):
        """Should clear all trajectories."""
        store.save(sample_trajectory)
        store.clear()
        
        assert store.count() == 0
    
    def test_trajectory_to_dict(self, sample_trajectory):
        """Should serialize to dict."""
        data = sample_trajectory.to_dict()
        
        assert data["trajectory_id"] == "traj_test"
        assert data["mode"] == "mode1"
    
    def test_trajectory_from_dict(self, sample_trajectory):
        """Should deserialize from dict."""
        data = sample_trajectory.to_dict()
        restored = Trajectory.from_dict(data)
        
        assert restored.trajectory_id == sample_trajectory.trajectory_id
        assert restored.mode == sample_trajectory.mode


# ===== Purge Endpoint Tests =====

class TestPurgeEndpoint:
    """Test admin purge-documents endpoint."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    def test_purge_rejects_without_confirm(self, client):
        """Should reject purge when confirm is false."""
        response = client.post(
            "/api/v1/admin/purge-documents",
            json={"confirm": False},
        )
        assert response.status_code == 400
        assert "not confirmed" in response.json()["detail"].lower()

    def test_purge_accepts_with_confirm(self, client):
        """Should accept and return PurgeResponse when confirm is true."""
        response = client.post(
            "/api/v1/admin/purge-documents",
            json={"confirm": True},
        )
        # May succeed or partially fail depending on infra, but should not 400
        assert response.status_code == 200
        data = response.json()
        assert "success" in data
        assert "documents_deleted" in data
        assert "vectors_deleted" in data
        assert "files_deleted" in data
        assert isinstance(data["errors"], list)
        assert isinstance(data["warnings"], list)

    def test_purge_response_has_pageindex_warning(self, client):
        """Purge response should warn that PageIndex was not cleared."""
        response = client.post(
            "/api/v1/admin/purge-documents",
            json={"confirm": True},
        )
        assert response.status_code == 200
        warnings = response.json().get("warnings", [])
        assert any("PageIndex" in w for w in warnings)
