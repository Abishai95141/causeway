"""
Phase 4 Tests: Temporal Feedback Loops

Tests:
1. TemporalTracker – edge lifecycle, staleness, confidence decay
2. FeedbackCollector – outcome recording, edge reliability, training rewards
3. CausalService integration – service-level temporal & feedback methods
4. Mode 2 integration – staleness escalation
"""

import math
import pytest
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from src.causal.temporal import (
    TemporalTracker,
    EdgeTemporalMetadata,
    StalenessReport,
)
from src.training.feedback import (
    FeedbackCollector,
    OutcomeFeedback,
    EdgeTrackRecord,
    FeedbackSummary,
    OutcomeResult,
    FeedbackSource,
)
from src.causal.service import CausalService
from src.models.enums import EvidenceStrength


# ===================================================================== #
#  Fixtures
# ===================================================================== #


@pytest.fixture
def tracker():
    """Fresh TemporalTracker with default settings."""
    return TemporalTracker()


@pytest.fixture
def tracker_fast_decay():
    """TemporalTracker with fast decay (7-day half-life)."""
    return TemporalTracker(
        staleness_threshold_days=10.0,
        decay_half_life_days=7.0,
        validation_freshness_days=5.0,
    )


@pytest.fixture
def collector():
    """Fresh FeedbackCollector."""
    return FeedbackCollector()


@pytest.fixture
def causal_service():
    """CausalService with a domain and some edges."""
    svc = CausalService()
    svc.create_world_model("test_domain")
    svc.add_variable("price", "Price", "Product price", domain="test_domain")
    svc.add_variable("demand", "Demand", "Customer demand", domain="test_domain")
    svc.add_variable("revenue", "Revenue", "Total revenue", domain="test_domain")
    svc.add_causal_link(
        "price", "demand", "Price inversely affects demand",
        domain="test_domain", strength=EvidenceStrength.MODERATE,
    )
    svc.add_causal_link(
        "demand", "revenue", "Higher demand drives revenue",
        domain="test_domain", strength=EvidenceStrength.STRONG,
    )
    return svc


# ===================================================================== #
#  1. TemporalTracker – EdgeTemporalMetadata
# ===================================================================== #


class TestEdgeTemporalMetadata:
    """Tests for EdgeTemporalMetadata dataclass."""

    def test_edge_key(self):
        meta = EdgeTemporalMetadata(from_var="A", to_var="B")
        assert meta.edge_key == ("A", "B")

    def test_age_days_is_non_negative(self):
        meta = EdgeTemporalMetadata(from_var="A", to_var="B")
        assert meta.age_days >= 0.0

    def test_days_since_validation(self):
        past = datetime.now(timezone.utc) - timedelta(days=5)
        meta = EdgeTemporalMetadata(
            from_var="A", to_var="B", last_validated_at=past,
        )
        assert meta.days_since_validation >= 4.9

    def test_to_dict_keys(self):
        meta = EdgeTemporalMetadata(from_var="A", to_var="B")
        d = meta.to_dict()
        expected_keys = {
            "edge", "created_at", "last_validated_at", "last_updated_at",
            "validation_count", "original_confidence", "current_confidence",
            "age_days", "days_since_validation",
        }
        assert expected_keys == set(d.keys())

    def test_to_dict_edge_format(self):
        meta = EdgeTemporalMetadata(from_var="price", to_var="demand")
        assert meta.to_dict()["edge"] == "price → demand"


# ===================================================================== #
#  2. TemporalTracker – Edge Lifecycle
# ===================================================================== #


class TestTemporalTrackerLifecycle:
    """Tests for edge creation, validation, and update tracking."""

    def test_record_edge_created(self, tracker):
        meta = tracker.record_edge_created("A", "B", confidence=0.8)
        assert meta.from_var == "A"
        assert meta.to_var == "B"
        assert meta.original_confidence == 0.8
        assert meta.current_confidence == 0.8
        assert meta.validation_count == 0

    def test_record_edge_created_sets_model_created_at(self, tracker):
        assert tracker._model_created_at is None
        tracker.record_edge_created("A", "B")
        assert tracker._model_created_at is not None

    def test_record_edge_created_custom_time(self, tracker):
        past = datetime(2024, 1, 1, tzinfo=timezone.utc)
        meta = tracker.record_edge_created("A", "B", created_at=past)
        assert meta.created_at == past

    def test_record_edge_validated(self, tracker):
        tracker.record_edge_created("A", "B", confidence=0.7)
        meta = tracker.record_edge_validated("A", "B", new_confidence=0.9)
        assert meta is not None
        assert meta.validation_count == 1
        assert meta.original_confidence == 0.9
        assert meta.current_confidence == 0.9

    def test_record_edge_validated_resets_decay(self, tracker):
        tracker.record_edge_created("A", "B", confidence=0.8)
        # Simulate decay
        meta = tracker.get_edge_metadata("A", "B")
        meta.current_confidence = 0.4
        # Validate without new confidence → restores original
        result = tracker.record_edge_validated("A", "B")
        assert result.current_confidence == result.original_confidence

    def test_record_edge_validated_nonexistent(self, tracker):
        result = tracker.record_edge_validated("X", "Y")
        assert result is None

    def test_record_edge_updated(self, tracker):
        tracker.record_edge_created("A", "B", confidence=0.5)
        meta = tracker.record_edge_updated("A", "B", new_confidence=0.6)
        assert meta is not None
        assert meta.original_confidence == 0.6
        assert meta.current_confidence == 0.6

    def test_record_edge_updated_nonexistent(self, tracker):
        result = tracker.record_edge_updated("X", "Y")
        assert result is None

    def test_remove_edge(self, tracker):
        tracker.record_edge_created("A", "B")
        tracker.remove_edge("A", "B")
        assert tracker.get_edge_metadata("A", "B") is None

    def test_remove_edge_nonexistent_no_error(self, tracker):
        tracker.remove_edge("X", "Y")  # should not raise

    def test_get_all_metadata(self, tracker):
        tracker.record_edge_created("A", "B")
        tracker.record_edge_created("C", "D")
        assert len(tracker.get_all_metadata()) == 2

    def test_clear(self, tracker):
        tracker.record_edge_created("A", "B")
        tracker.clear()
        assert len(tracker.get_all_metadata()) == 0
        assert tracker._model_created_at is None


# ===================================================================== #
#  3. TemporalTracker – Confidence Decay
# ===================================================================== #


class TestConfidenceDecay:
    """Tests for exponential confidence decay over time."""

    def test_no_decay_when_just_validated(self, tracker):
        tracker.record_edge_created("A", "B", confidence=0.8)
        decayed = tracker.compute_decayed_confidence("A", "B")
        assert decayed == pytest.approx(0.8, abs=0.01)

    def test_half_life_decay(self, tracker_fast_decay):
        """After one half-life, confidence should be ~50% of original."""
        past = datetime.now(timezone.utc) - timedelta(days=7)
        tracker_fast_decay.record_edge_created("A", "B", confidence=1.0, created_at=past)
        now = datetime.now(timezone.utc)
        decayed = tracker_fast_decay.compute_decayed_confidence("A", "B", reference_time=now)
        assert decayed == pytest.approx(0.5, abs=0.05)

    def test_double_half_life_decay(self, tracker_fast_decay):
        """After two half-lives, confidence should be ~25% of original."""
        past = datetime.now(timezone.utc) - timedelta(days=14)
        tracker_fast_decay.record_edge_created("A", "B", confidence=1.0, created_at=past)
        now = datetime.now(timezone.utc)
        decayed = tracker_fast_decay.compute_decayed_confidence("A", "B", reference_time=now)
        assert decayed == pytest.approx(0.25, abs=0.05)

    def test_decay_clamped_to_zero_one(self, tracker):
        """Confidence should never go negative."""
        very_old = datetime.now(timezone.utc) - timedelta(days=10000)
        tracker.record_edge_created("A", "B", confidence=1.0, created_at=very_old)
        decayed = tracker.compute_decayed_confidence("A", "B")
        assert 0.0 <= decayed <= 1.0

    def test_decay_nonexistent_edge(self, tracker):
        assert tracker.compute_decayed_confidence("X", "Y") == 0.0

    def test_apply_decay_updates_all_edges(self, tracker_fast_decay):
        past = datetime.now(timezone.utc) - timedelta(days=7)
        tracker_fast_decay.record_edge_created("A", "B", confidence=1.0, created_at=past)
        tracker_fast_decay.record_edge_created("C", "D", confidence=0.8, created_at=past)

        updates = tracker_fast_decay.apply_decay()
        assert len(updates) == 2
        # Both edges should have decayed
        assert updates[("A", "B")] == pytest.approx(0.5, abs=0.05)
        assert updates[("C", "D")] == pytest.approx(0.4, abs=0.05)

        # Metadata should be updated
        meta_ab = tracker_fast_decay.get_edge_metadata("A", "B")
        assert meta_ab.current_confidence == pytest.approx(0.5, abs=0.05)

    def test_validation_resets_decay_clock(self, tracker_fast_decay):
        """Re-validating an edge should restore confidence."""
        past = datetime.now(timezone.utc) - timedelta(days=7)
        tracker_fast_decay.record_edge_created("A", "B", confidence=0.8, created_at=past)

        # Decay
        tracker_fast_decay.apply_decay()
        meta = tracker_fast_decay.get_edge_metadata("A", "B")
        assert meta.current_confidence < 0.8

        # Re-validate
        tracker_fast_decay.record_edge_validated("A", "B")
        assert meta.current_confidence == meta.original_confidence

    def test_decay_formula_matches_exponential(self, tracker):
        """Verify the decay formula: c(t) = c_0 * exp(-λ * t)."""
        c0 = 0.9
        half_life = tracker.decay_half_life_days
        days = 30
        past = datetime.now(timezone.utc) - timedelta(days=days)
        tracker.record_edge_created("A", "B", confidence=c0, created_at=past)

        lam = math.log(2) / half_life
        expected = c0 * math.exp(-lam * days)
        actual = tracker.compute_decayed_confidence("A", "B")
        assert actual == pytest.approx(expected, abs=0.001)


# ===================================================================== #
#  4. TemporalTracker – Staleness Reports
# ===================================================================== #


class TestStalenessReport:
    """Tests for staleness report generation."""

    def test_fresh_model_not_stale(self, tracker):
        tracker.record_edge_created("A", "B")
        report = tracker.check_staleness(domain="test")
        assert not report.is_stale
        assert report.stale_edge_count == 0
        assert report.overall_freshness > 0.9

    def test_old_model_is_stale(self, tracker_fast_decay):
        past = datetime.now(timezone.utc) - timedelta(days=15)
        tracker_fast_decay.record_edge_created("A", "B", created_at=past)
        report = tracker_fast_decay.check_staleness(domain="test")
        assert report.is_stale
        assert report.model_age_days >= 14.9

    def test_stale_edges_detected(self, tracker_fast_decay):
        past = datetime.now(timezone.utc) - timedelta(days=15)
        tracker_fast_decay.record_edge_created("A", "B", created_at=past)
        # Fresh edge
        tracker_fast_decay.record_edge_created("C", "D")
        report = tracker_fast_decay.check_staleness(domain="test")
        assert report.stale_edge_count == 1
        assert len(report.fresh_edges) == 1

    def test_edges_needing_validation(self, tracker_fast_decay):
        # Edge validated 6 days ago (freshness threshold = 5)
        past = datetime.now(timezone.utc) - timedelta(days=6)
        tracker_fast_decay.record_edge_created("A", "B", created_at=past)
        report = tracker_fast_decay.check_staleness(domain="test")
        assert len(report.edges_needing_validation) == 1

    def test_empty_model_not_stale(self, tracker):
        report = tracker.check_staleness(domain="test")
        assert not report.is_stale
        assert report.overall_freshness == 1.0
        assert report.total_edges == 0

    def test_staleness_report_to_dict(self, tracker):
        tracker.record_edge_created("A", "B")
        report = tracker.check_staleness(domain="test")
        d = report.to_dict()
        assert d["domain"] == "test"
        assert "is_stale" in d
        assert "overall_freshness" in d
        assert "stale_ratio" in d

    def test_stale_ratio(self, tracker_fast_decay):
        past = datetime.now(timezone.utc) - timedelta(days=15)
        tracker_fast_decay.record_edge_created("A", "B", created_at=past)
        tracker_fast_decay.record_edge_created("C", "D")
        report = tracker_fast_decay.check_staleness(domain="test")
        assert report.stale_ratio == pytest.approx(0.5, abs=0.01)

    def test_overall_freshness_degrades_with_time(self, tracker_fast_decay):
        past = datetime.now(timezone.utc) - timedelta(days=15)
        tracker_fast_decay.record_edge_created("A", "B", created_at=past)
        tracker_fast_decay.record_edge_created("C", "D", created_at=past)
        report = tracker_fast_decay.check_staleness(domain="test")
        # All edges old → low freshness
        assert report.overall_freshness < 0.5

    def test_staleness_check_applies_decay(self, tracker_fast_decay):
        """Staleness check should update current_confidence via decay."""
        past = datetime.now(timezone.utc) - timedelta(days=7)
        tracker_fast_decay.record_edge_created("A", "B", confidence=1.0, created_at=past)
        tracker_fast_decay.check_staleness(domain="test")
        meta = tracker_fast_decay.get_edge_metadata("A", "B")
        assert meta.current_confidence < 1.0

    def test_model_age_days_property(self, tracker):
        past = datetime.now(timezone.utc) - timedelta(days=10)
        tracker.record_edge_created("A", "B", created_at=past)
        assert tracker.model_age_days >= 9.9

    def test_model_age_days_no_edges(self, tracker):
        assert tracker.model_age_days == 0.0

    def test_set_model_created_at(self, tracker):
        dt = datetime(2020, 1, 1, tzinfo=timezone.utc)
        tracker.set_model_created_at(dt)
        assert tracker.model_age_days > 365


# ===================================================================== #
#  5. FeedbackCollector – OutcomeFeedback
# ===================================================================== #


class TestOutcomeFeedback:
    """Tests for OutcomeFeedback dataclass."""

    def test_default_values(self):
        fb = OutcomeFeedback()
        assert fb.result == OutcomeResult.PENDING
        assert fb.source == FeedbackSource.HUMAN
        assert fb.reward_delta == 0.0
        assert fb.feedback_id.startswith("fb_")

    def test_to_dict(self):
        fb = OutcomeFeedback(
            decision_trace_id="m2_abc123",
            domain="pricing",
            result=OutcomeResult.POSITIVE,
            edges_involved=[("price", "demand")],
            reward_delta=0.8,
        )
        d = fb.to_dict()
        assert d["decision_trace_id"] == "m2_abc123"
        assert d["result"] == "positive"
        assert d["reward_delta"] == 0.8

    def test_outcome_result_values(self):
        assert OutcomeResult.POSITIVE.value == "positive"
        assert OutcomeResult.NEGATIVE.value == "negative"
        assert OutcomeResult.NEUTRAL.value == "neutral"
        assert OutcomeResult.UNEXPECTED.value == "unexpected"
        assert OutcomeResult.PENDING.value == "pending"

    def test_feedback_source_values(self):
        assert FeedbackSource.HUMAN.value == "human"
        assert FeedbackSource.METRIC.value == "metric"
        assert FeedbackSource.SYSTEM.value == "system"


# ===================================================================== #
#  6. FeedbackCollector – Recording
# ===================================================================== #


class TestFeedbackCollectorRecording:
    """Tests for feedback recording and retrieval."""

    def test_record_feedback(self, collector):
        fb = OutcomeFeedback(
            decision_trace_id="m2_abc",
            result=OutcomeResult.POSITIVE,
            edges_involved=[("A", "B")],
            reward_delta=0.5,
        )
        fid = collector.record_feedback(fb)
        assert fid == fb.feedback_id
        assert collector.feedback_count == 1

    def test_get_feedback(self, collector):
        fb = OutcomeFeedback(decision_trace_id="m2_abc")
        collector.record_feedback(fb)
        retrieved = collector.get_feedback(fb.feedback_id)
        assert retrieved is not None
        assert retrieved.decision_trace_id == "m2_abc"

    def test_get_feedback_nonexistent(self, collector):
        assert collector.get_feedback("nonexistent") is None

    def test_get_feedback_for_decision(self, collector):
        fb1 = OutcomeFeedback(decision_trace_id="m2_abc", result=OutcomeResult.POSITIVE)
        fb2 = OutcomeFeedback(decision_trace_id="m2_abc", result=OutcomeResult.NEUTRAL)
        fb3 = OutcomeFeedback(decision_trace_id="m2_xyz", result=OutcomeResult.NEGATIVE)
        collector.record_feedback(fb1)
        collector.record_feedback(fb2)
        collector.record_feedback(fb3)

        results = collector.get_feedback_for_decision("m2_abc")
        assert len(results) == 2
        assert all(fb.decision_trace_id == "m2_abc" for fb in results)

    def test_get_feedback_for_decision_empty(self, collector):
        assert collector.get_feedback_for_decision("nonexistent") == []

    def test_clear(self, collector):
        collector.record_feedback(OutcomeFeedback(
            edges_involved=[("A", "B")], result=OutcomeResult.POSITIVE,
        ))
        collector.clear()
        assert collector.feedback_count == 0
        assert collector.tracked_edge_count == 0


# ===================================================================== #
#  7. FeedbackCollector – Edge Track Records
# ===================================================================== #


class TestEdgeTrackRecord:
    """Tests for per-edge track records and reliability scoring."""

    def test_edge_record_created_on_feedback(self, collector):
        fb = OutcomeFeedback(
            edges_involved=[("A", "B")],
            result=OutcomeResult.POSITIVE,
            reward_delta=1.0,
        )
        collector.record_feedback(fb)
        record = collector.get_edge_record("A", "B")
        assert record is not None
        assert record.total_decisions == 1
        assert record.positive_outcomes == 1

    def test_multiple_feedback_updates_record(self, collector):
        for result in [OutcomeResult.POSITIVE, OutcomeResult.POSITIVE, OutcomeResult.NEGATIVE]:
            fb = OutcomeFeedback(
                edges_involved=[("A", "B")],
                result=result,
                reward_delta=0.5 if result == OutcomeResult.POSITIVE else -0.5,
            )
            collector.record_feedback(fb)

        record = collector.get_edge_record("A", "B")
        assert record.total_decisions == 3
        assert record.positive_outcomes == 2
        assert record.negative_outcomes == 1

    def test_success_rate(self, collector):
        for result in [OutcomeResult.POSITIVE, OutcomeResult.POSITIVE, OutcomeResult.NEGATIVE, OutcomeResult.NEUTRAL]:
            collector.record_feedback(OutcomeFeedback(
                edges_involved=[("A", "B")], result=result,
            ))
        record = collector.get_edge_record("A", "B")
        assert record.success_rate == pytest.approx(0.5, abs=0.01)

    def test_success_rate_no_data(self):
        record = EdgeTrackRecord(from_var="A", to_var="B")
        assert record.success_rate == 0.0

    def test_reliability_score_neutral_prior(self):
        record = EdgeTrackRecord(from_var="A", to_var="B")
        assert record.reliability_score == 0.5  # No data → neutral

    def test_reliability_score_with_data(self, collector):
        # 8 positive, 2 negative → high reliability
        for _ in range(8):
            collector.record_feedback(OutcomeFeedback(
                edges_involved=[("var_a", "var_b")], result=OutcomeResult.POSITIVE,
            ))
        for _ in range(2):
            collector.record_feedback(OutcomeFeedback(
                edges_involved=[("var_a", "var_b")], result=OutcomeResult.NEGATIVE,
            ))
        record = collector.get_edge_record("var_a", "var_b")
        # Wilson lower bound for 8/10 at 95% CI is ~0.49
        assert record.reliability_score > 0.4

    def test_reliability_score_all_negative(self, collector):
        for _ in range(5):
            collector.record_feedback(OutcomeFeedback(
                edges_involved=[("A", "B")], result=OutcomeResult.NEGATIVE,
            ))
        record = collector.get_edge_record("A", "B")
        assert record.reliability_score < 0.2

    def test_wilson_score_lower_bound_small_sample(self, collector):
        """With only 1 positive out of 1, Wilson score should be conservative."""
        collector.record_feedback(OutcomeFeedback(
            edges_involved=[("A", "B")], result=OutcomeResult.POSITIVE,
        ))
        record = collector.get_edge_record("A", "B")
        # Wilson lower bound for n=1, p=1.0 should be well below 1.0
        assert record.reliability_score < 0.8

    def test_avg_reward_delta(self, collector):
        collector.record_feedback(OutcomeFeedback(
            edges_involved=[("A", "B")], result=OutcomeResult.POSITIVE, reward_delta=1.0,
        ))
        collector.record_feedback(OutcomeFeedback(
            edges_involved=[("A", "B")], result=OutcomeResult.NEGATIVE, reward_delta=-0.5,
        ))
        record = collector.get_edge_record("A", "B")
        assert record.avg_reward_delta == pytest.approx(0.25, abs=0.01)

    def test_edge_record_to_dict(self, collector):
        collector.record_feedback(OutcomeFeedback(
            edges_involved=[("A", "B")], result=OutcomeResult.POSITIVE,
        ))
        record = collector.get_edge_record("A", "B")
        d = record.to_dict()
        assert d["edge"] == "A → B"
        assert "success_rate" in d
        assert "reliability_score" in d

    def test_multiple_edges_in_one_feedback(self, collector):
        fb = OutcomeFeedback(
            edges_involved=[("A", "B"), ("B", "C")],
            result=OutcomeResult.POSITIVE,
        )
        collector.record_feedback(fb)
        assert collector.tracked_edge_count == 2
        assert collector.get_edge_record("A", "B").total_decisions == 1
        assert collector.get_edge_record("B", "C").total_decisions == 1


# ===================================================================== #
#  8. FeedbackCollector – Low/High Reliability Edges
# ===================================================================== #


class TestReliabilityFiltering:
    """Tests for low/high reliability edge filtering."""

    def test_get_low_reliability_edges(self, collector):
        # All-negative edge
        for _ in range(5):
            collector.record_feedback(OutcomeFeedback(
                edges_involved=[("bad", "edge")], result=OutcomeResult.NEGATIVE,
            ))
        # All-positive edge
        for _ in range(5):
            collector.record_feedback(OutcomeFeedback(
                edges_involved=[("good", "edge")], result=OutcomeResult.POSITIVE,
            ))
        low = collector.get_low_reliability_edges(threshold=0.4, min_decisions=2)
        assert len(low) >= 1
        assert any(r.from_var == "bad" for r in low)

    def test_get_high_reliability_edges(self, collector):
        for _ in range(10):
            collector.record_feedback(OutcomeFeedback(
                edges_involved=[("good", "edge")], result=OutcomeResult.POSITIVE,
            ))
        high = collector.get_high_reliability_edges(threshold=0.5, min_decisions=2)
        assert len(high) >= 1
        assert any(r.from_var == "good" for r in high)

    def test_min_decisions_filter(self, collector):
        # Only 1 decision → should not appear in low reliability
        collector.record_feedback(OutcomeFeedback(
            edges_involved=[("A", "B")], result=OutcomeResult.NEGATIVE,
        ))
        low = collector.get_low_reliability_edges(threshold=0.4, min_decisions=2)
        assert len(low) == 0


# ===================================================================== #
#  9. FeedbackCollector – Summary & Training Rewards
# ===================================================================== #


class TestFeedbackSummaryAndRewards:
    """Tests for summary generation and training rewards."""

    def test_get_summary_empty(self, collector):
        summary = collector.get_summary()
        assert summary.total_feedback == 0
        assert summary.avg_reward == 0.0

    def test_get_summary_with_data(self, collector):
        collector.record_feedback(OutcomeFeedback(
            domain="pricing", result=OutcomeResult.POSITIVE, reward_delta=0.8,
        ))
        collector.record_feedback(OutcomeFeedback(
            domain="pricing", result=OutcomeResult.NEGATIVE, reward_delta=-0.5,
        ))
        summary = collector.get_summary(domain="pricing")
        assert summary.total_feedback == 2
        assert summary.positive_count == 1
        assert summary.negative_count == 1
        assert summary.avg_reward == pytest.approx(0.15, abs=0.01)

    def test_get_summary_filters_by_domain(self, collector):
        collector.record_feedback(OutcomeFeedback(domain="pricing"))
        collector.record_feedback(OutcomeFeedback(domain="marketing"))
        summary = collector.get_summary(domain="pricing")
        assert summary.total_feedback == 1

    def test_get_summary_no_domain_returns_all(self, collector):
        collector.record_feedback(OutcomeFeedback(domain="pricing"))
        collector.record_feedback(OutcomeFeedback(domain="marketing"))
        summary = collector.get_summary()
        assert summary.total_feedback == 2

    def test_compute_training_reward_no_feedback(self, collector):
        assert collector.compute_training_reward("nonexistent") == 0.0

    def test_compute_training_reward_single(self, collector):
        collector.record_feedback(OutcomeFeedback(
            decision_trace_id="m2_abc", reward_delta=0.7,
        ))
        reward = collector.compute_training_reward("m2_abc")
        assert reward == pytest.approx(0.7, abs=0.01)

    def test_compute_training_reward_weighted(self, collector):
        """Later feedback should have more weight."""
        from datetime import timedelta
        base = datetime.now(timezone.utc) - timedelta(hours=2)
        fb1 = OutcomeFeedback(
            decision_trace_id="m2_abc", reward_delta=0.5,
            created_at=base,
        )
        fb2 = OutcomeFeedback(
            decision_trace_id="m2_abc", reward_delta=1.0,
            created_at=base + timedelta(hours=1),
        )
        collector.record_feedback(fb1)
        collector.record_feedback(fb2)
        reward = collector.compute_training_reward("m2_abc")
        # Weight: fb1=1.0, fb2=1.5 → (0.5*1.0 + 1.0*1.5) / 2.5 = 0.8
        assert reward == pytest.approx(0.8, abs=0.01)


# ===================================================================== #
#  10. CausalService – Temporal Integration
# ===================================================================== #


class TestCausalServiceTemporal:
    """Tests for temporal tracking via CausalService."""

    def test_add_causal_link_creates_temporal_metadata(self, causal_service):
        tracker = causal_service._get_or_create_tracker("test_domain")
        meta = tracker.get_edge_metadata("price", "demand")
        assert meta is not None
        assert meta.from_var == "price"
        assert meta.to_var == "demand"

    def test_get_or_create_tracker_lazy(self, causal_service):
        tracker1 = causal_service._get_or_create_tracker("new_domain")
        tracker2 = causal_service._get_or_create_tracker("new_domain")
        assert tracker1 is tracker2

    def test_check_model_staleness_fresh(self, causal_service):
        report = causal_service.check_model_staleness(domain="test_domain")
        assert not report.is_stale
        assert report.domain == "test_domain"
        assert report.total_edges == 2

    def test_check_model_staleness_stale(self):
        svc = CausalService()
        svc.create_world_model("old")
        svc.add_variable("var_a", "A", "A", domain="old")
        svc.add_variable("var_b", "B", "B", domain="old")
        svc.add_causal_link("var_a", "var_b", "test", domain="old")
        # Manually backdate
        tracker = svc._get_or_create_tracker("old")
        past = datetime.now(timezone.utc) - timedelta(days=60)
        tracker.set_model_created_at(past)
        for meta in tracker.get_all_metadata():
            meta.created_at = past
            meta.last_validated_at = past

        report = svc.check_model_staleness(domain="old")
        assert report.is_stale

    def test_validate_edge(self, causal_service):
        meta = causal_service.validate_edge("price", "demand", domain="test_domain")
        assert meta is not None
        assert meta.validation_count == 1

    def test_validate_edge_with_new_confidence(self, causal_service):
        meta = causal_service.validate_edge(
            "price", "demand", new_confidence=0.95, domain="test_domain",
        )
        assert meta.original_confidence == 0.95
        assert meta.current_confidence == 0.95

    def test_validate_edge_nonexistent(self, causal_service):
        result = causal_service.validate_edge("X", "Y", domain="test_domain")
        assert result is None

    def test_apply_confidence_decay(self):
        svc = CausalService()
        svc.create_world_model("test")
        svc.add_variable("var_a", "A", "A", domain="test")
        svc.add_variable("var_b", "B", "B", domain="test")
        svc.add_causal_link("var_a", "var_b", "test", domain="test")
        # Backdate the edge
        tracker = svc._get_or_create_tracker("test")
        meta = tracker.get_edge_metadata("var_a", "var_b")
        meta.last_validated_at = datetime.now(timezone.utc) - timedelta(days=60)
        meta.original_confidence = 0.8

        updates = svc.apply_confidence_decay(domain="test")
        assert len(updates) == 1
        assert updates[("var_a", "var_b")] < 0.8


# ===================================================================== #
#  11. CausalService – Feedback Integration
# ===================================================================== #


class TestCausalServiceFeedback:
    """Tests for feedback collection via CausalService."""

    def test_record_decision_feedback(self, causal_service):
        fb = OutcomeFeedback(
            decision_trace_id="m2_test",
            domain="test_domain",
            result=OutcomeResult.POSITIVE,
            edges_involved=[("price", "demand")],
            reward_delta=0.8,
        )
        fid = causal_service.record_decision_feedback(fb)
        assert fid == fb.feedback_id

    def test_get_feedback_summary(self, causal_service):
        causal_service.record_decision_feedback(OutcomeFeedback(
            domain="test_domain", result=OutcomeResult.POSITIVE, reward_delta=0.5,
        ))
        causal_service.record_decision_feedback(OutcomeFeedback(
            domain="test_domain", result=OutcomeResult.NEGATIVE, reward_delta=-0.3,
        ))
        summary = causal_service.get_feedback_summary(domain="test_domain")
        assert summary.total_feedback == 2
        assert summary.positive_count == 1
        assert summary.negative_count == 1

    def test_get_training_reward(self, causal_service):
        causal_service.record_decision_feedback(OutcomeFeedback(
            decision_trace_id="m2_test", reward_delta=0.9,
        ))
        reward = causal_service.get_training_reward("m2_test")
        assert reward == pytest.approx(0.9, abs=0.01)

    def test_get_training_reward_no_feedback(self, causal_service):
        assert causal_service.get_training_reward("nonexistent") == 0.0


# ===================================================================== #
#  12. Mode 2 – Staleness Escalation
# ===================================================================== #


class TestMode2StalenessEscalation:
    """Tests for Mode 2 staleness-based escalation."""

    @pytest.fixture
    def mock_llm(self):
        llm = AsyncMock()
        response = MagicMock()
        response.content = '{"domain": "pricing", "intervention": "raise prices", "target_outcome": "revenue", "constraints": []}'
        llm.generate = AsyncMock(return_value=response)
        llm.initialize = AsyncMock()
        return llm

    @pytest.fixture
    def mock_retrieval(self):
        router = AsyncMock()
        router.initialize = AsyncMock()
        bundles = [
            MagicMock(content_hash="hash_1" + "x" * 50, content="Evidence about pricing"),
            MagicMock(content_hash="hash_2" + "x" * 50, content="Evidence about demand"),
            MagicMock(content_hash="hash_3" + "x" * 50, content="Evidence about revenue"),
        ]
        router.retrieve = AsyncMock(return_value=bundles)
        return router

    @pytest.mark.asyncio
    async def test_stale_model_escalates(self, mock_llm, mock_retrieval):
        """A stale model should trigger escalation to Mode 1."""
        from src.modes.mode2 import Mode2DecisionSupport, Mode2Stage

        svc = CausalService()
        svc.create_world_model("pricing")
        svc.add_variable("price", "Price", "Price", domain="pricing")
        svc.add_variable("demand", "Demand", "Demand", domain="pricing")
        svc.add_causal_link("price", "demand", "test", domain="pricing")

        # Backdate model to make it stale
        tracker = svc._get_or_create_tracker("pricing")
        past = datetime.now(timezone.utc) - timedelta(days=60)
        tracker.set_model_created_at(past)
        for meta in tracker.get_all_metadata():
            meta.created_at = past
            meta.last_validated_at = past

        mode2 = Mode2DecisionSupport(
            llm_client=mock_llm,
            retrieval_router=mock_retrieval,
            causal_service=svc,
        )
        result = await mode2.run("Should we raise prices?", domain_hint="pricing")

        assert result.escalate_to_mode1 is True
        assert result.stage == Mode2Stage.STALENESS_CHECK
        assert "stale" in result.escalation_reason.lower()
        assert result.model_staleness is not None
        assert result.confidence_decay_applied is True

    @pytest.mark.asyncio
    async def test_fresh_model_passes_staleness(self, mock_llm, mock_retrieval):
        """A fresh model should pass the staleness check."""
        from src.modes.mode2 import Mode2DecisionSupport, Mode2Stage

        svc = CausalService()
        svc.create_world_model("pricing")
        svc.add_variable("price", "Price", "Price", domain="pricing")
        svc.add_variable("demand", "Demand", "Demand", domain="pricing")
        svc.add_causal_link("price", "demand", "test", domain="pricing")

        mode2 = Mode2DecisionSupport(
            llm_client=mock_llm,
            retrieval_router=mock_retrieval,
            causal_service=svc,
        )

        # Patch conflict detection to not escalate
        with patch.object(svc, "detect_conflicts") as mock_detect:
            mock_report = MagicMock()
            mock_report.total = 0
            mock_report.critical_count = 0
            mock_report.has_critical = False
            mock_report.conflicts = []
            mock_detect.return_value = mock_report

            # Need LLM to return recommendation JSON for the final stage
            rec_response = MagicMock()
            rec_response.content = '{"recommendation": "Proceed", "confidence": "high", "reasoning": "OK", "actions": [], "risks": []}'
            mock_llm.generate = AsyncMock(side_effect=[
                # First call: query parsing
                MagicMock(content='{"domain": "pricing", "intervention": "raise prices", "target_outcome": "revenue", "constraints": []}'),
                # Second call: recommendation
                rec_response,
            ])

            result = await mode2.run("Should we raise prices?", domain_hint="pricing")

        assert result.stage != Mode2Stage.STALENESS_CHECK
        assert result.escalate_to_mode1 is False or result.stage == Mode2Stage.COMPLETE

    @pytest.mark.asyncio
    async def test_mode2_result_has_staleness_fields(self, mock_llm, mock_retrieval):
        """Mode2Result should include staleness info on success."""
        from src.modes.mode2 import Mode2DecisionSupport, Mode2Stage

        svc = CausalService()
        svc.create_world_model("pricing")
        svc.add_variable("price", "Price", "Price", domain="pricing")
        svc.add_variable("demand", "Demand", "Demand", domain="pricing")
        svc.add_causal_link("price", "demand", "test", domain="pricing")

        mode2 = Mode2DecisionSupport(
            llm_client=mock_llm,
            retrieval_router=mock_retrieval,
            causal_service=svc,
        )

        with patch.object(svc, "detect_conflicts") as mock_detect:
            mock_report = MagicMock()
            mock_report.total = 0
            mock_report.critical_count = 0
            mock_report.has_critical = False
            mock_report.conflicts = []
            mock_detect.return_value = mock_report

            rec_response = MagicMock()
            rec_response.content = '{"recommendation": "Go", "confidence": "medium", "reasoning": "OK", "actions": [], "risks": []}'
            mock_llm.generate = AsyncMock(side_effect=[
                MagicMock(content='{"domain": "pricing", "intervention": "raise prices", "target_outcome": "revenue", "constraints": []}'),
                rec_response,
            ])

            result = await mode2.run("Should we raise prices?", domain_hint="pricing")

        if result.stage == Mode2Stage.COMPLETE:
            assert result.model_staleness is not None
            assert result.confidence_decay_applied is True

    @pytest.mark.asyncio
    async def test_mode2_staleness_stage_in_enum(self):
        from src.modes.mode2 import Mode2Stage
        assert Mode2Stage.STALENESS_CHECK == "staleness_check"


# ===================================================================== #
#  13. Package Exports
# ===================================================================== #


class TestPhase4Exports:
    """Tests that Phase 4 types are properly exported."""

    def test_causal_temporal_exports(self):
        from src.causal import TemporalTracker, EdgeTemporalMetadata, StalenessReport
        assert TemporalTracker is not None
        assert EdgeTemporalMetadata is not None
        assert StalenessReport is not None

    def test_training_feedback_exports(self):
        from src.training import (
            FeedbackCollector, OutcomeFeedback, EdgeTrackRecord,
            FeedbackSummary, OutcomeResult, FeedbackSource,
        )
        assert FeedbackCollector is not None
        assert OutcomeFeedback is not None
        assert EdgeTrackRecord is not None
        assert FeedbackSummary is not None
        assert OutcomeResult is not None
        assert FeedbackSource is not None
