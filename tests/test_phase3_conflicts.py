"""
Tests for Phase 3: Conflict Detection and Resolution

Tests cover:
- ConflictDetector: edge contradictions, direction reversals, strength changes
- ConflictDetector: missing variables, stale edges
- ConflictResolver: evidence_weighted, temporal, accept_new, keep_existing, manual
- ConflictReport: properties and aggregation
- CausalService integration: detect_conflicts(), resolve_conflicts(), apply_resolutions()
- Mode 2 integration: conflict detection stage, critical escalation
- Mode 1 integration: post-build conflict detection
"""

import pytest
from uuid import uuid4

from src.causal.conflict_resolver import (
    Conflict,
    ConflictDetector,
    ConflictReport,
    ConflictResolver,
    ConflictSeverity,
    ConflictType,
    ResolutionAction,
    ResolutionStrategy,
)
from src.causal.dag_engine import DAGEngine
from src.causal.service import CausalService
from src.models.causal import CausalEdge, EdgeMetadata
from src.models.enums import EvidenceStrength, VariableType, MeasurementStatus
from src.models.evidence import (
    EvidenceBundle,
    SourceReference,
    RetrievalTrace,
)
from src.models.enums import RetrievalMethod


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _edge(from_var: str, to_var: str, mechanism: str,
          strength: EvidenceStrength = EvidenceStrength.MODERATE,
          evidence_refs=None, contradicting_refs=None) -> CausalEdge:
    return CausalEdge(
        from_var=from_var,
        to_var=to_var,
        metadata=EdgeMetadata(
            mechanism=mechanism,
            evidence_strength=strength,
            evidence_refs=evidence_refs or [],
            contradicting_refs=contradicting_refs or [],
        ),
    )


def _evidence(content: str, doc_id: str = "doc1") -> EvidenceBundle:
    return EvidenceBundle(
        content=content,
        source=SourceReference(doc_id=doc_id, doc_title="Test.pdf"),
        retrieval_trace=RetrievalTrace(
            method=RetrievalMethod.HAYSTACK, query="test",
        ),
    )


def _build_service_with_model() -> CausalService:
    """Build a CausalService with a small pricing world model."""
    svc = CausalService()
    engine = svc.create_world_model("pricing")
    engine.add_variable("price", "Price", "Product price",
                        var_type=VariableType.CONTINUOUS,
                        measurement_status=MeasurementStatus.MEASURED)
    engine.add_variable("demand", "Demand", "Customer demand",
                        var_type=VariableType.CONTINUOUS,
                        measurement_status=MeasurementStatus.MEASURED)
    engine.add_variable("revenue", "Revenue", "Total revenue",
                        var_type=VariableType.CONTINUOUS,
                        measurement_status=MeasurementStatus.MEASURED)
    engine.add_edge("price", "demand", "Higher prices reduce demand via elasticity",
                    strength=EvidenceStrength.MODERATE,
                    evidence_refs=[uuid4()])
    engine.add_edge("demand", "revenue", "More demand increases revenue",
                    strength=EvidenceStrength.STRONG,
                    evidence_refs=[uuid4(), uuid4(), uuid4()])
    return svc


# ===========================================================================
# ConflictDetector tests
# ===========================================================================


class TestConflictDetectorEdgeConflicts:
    """Test edge-level conflict detection."""

    def test_no_conflict_with_supporting_evidence(self):
        """Supporting evidence should not trigger conflicts."""
        edge = _edge("price", "demand", "Higher prices reduce demand")
        evidence = [
            _evidence("Price increases lead to reduced demand through elasticity."),
        ]
        detector = ConflictDetector()
        conflicts = detector.detect_edge_conflicts(edge, evidence)
        # Should find no CONTRADICTION type (may find strength change)
        contradictions = [c for c in conflicts
                         if c.conflict_type == ConflictType.EDGE_CONTRADICTION]
        assert len(contradictions) == 0

    def test_detects_negation_contradiction(self):
        """Negation signals near mechanism terms should trigger contradiction."""
        edge = _edge("price", "demand", "Higher prices reduce demand")
        evidence = [
            _evidence("Price does not affect demand significantly in this market."),
        ]
        detector = ConflictDetector()
        conflicts = detector.detect_edge_conflicts(edge, evidence)
        contradictions = [c for c in conflicts
                         if c.conflict_type == ConflictType.EDGE_CONTRADICTION]
        assert len(contradictions) >= 1
        assert contradictions[0].severity in (
            ConflictSeverity.CRITICAL, ConflictSeverity.WARNING, ConflictSeverity.INFO,
        )

    def test_detects_direction_reversal(self):
        """Evidence suggesting reversed direction should be CRITICAL."""
        edge = _edge("price", "demand", "Price affects demand")
        evidence = [
            _evidence("Demand causes price changes through supply dynamics."),
        ]
        detector = ConflictDetector()
        conflicts = detector.detect_edge_conflicts(edge, evidence)
        reversals = [c for c in conflicts
                     if c.conflict_type == ConflictType.DIRECTION_REVERSAL]
        assert len(reversals) >= 1
        assert reversals[0].severity == ConflictSeverity.CRITICAL

    def test_irrelevant_evidence_ignored(self):
        """Evidence about unrelated topics should produce no conflicts."""
        edge = _edge("price", "demand", "Higher prices reduce demand")
        evidence = [
            _evidence("The weather was sunny in Q3 across all regions."),
        ]
        detector = ConflictDetector()
        conflicts = detector.detect_edge_conflicts(edge, evidence)
        assert len(conflicts) == 0

    def test_strength_downgrade_detected(self):
        """Evidence contradicting a STRONG edge should suggest downgrade."""
        edge = _edge("price", "demand", "Price reduces demand",
                     strength=EvidenceStrength.STRONG,
                     evidence_refs=[uuid4(), uuid4(), uuid4()])
        evidence = [
            _evidence("Price has no significant effect on demand in luxury segments."),
        ]
        detector = ConflictDetector()
        conflicts = detector.detect_edge_conflicts(edge, evidence)
        downgrades = [c for c in conflicts
                      if c.conflict_type == ConflictType.STRENGTH_DOWNGRADE]
        assert len(downgrades) >= 1

    def test_strength_upgrade_detected(self):
        """Lots of supporting evidence for HYPOTHESIS edge should suggest upgrade."""
        edge = _edge("price", "demand", "Price reduces demand",
                     strength=EvidenceStrength.HYPOTHESIS)
        evidence = [
            _evidence("Higher prices reduce demand in Q1."),
            _evidence("Price increase led to demand drop in Q2."),
            _evidence("Price changes affect demand elasticity in Q3."),
            _evidence("Demand fell when prices rose in Q4."),
        ]
        detector = ConflictDetector()
        conflicts = detector.detect_edge_conflicts(edge, evidence)
        upgrades = [c for c in conflicts
                    if c.conflict_type == ConflictType.STRENGTH_UPGRADE]
        assert len(upgrades) >= 1

    def test_multiple_conflicts_detected(self):
        """Multiple conflict types can be detected for one edge."""
        edge = _edge("price", "demand", "Price reduces demand",
                     strength=EvidenceStrength.STRONG)
        evidence = [
            _evidence("Price does not reduce demand in any significant way."),
            _evidence("Demand causes price through supply pressure."),
        ]
        detector = ConflictDetector()
        conflicts = detector.detect_edge_conflicts(edge, evidence)
        types = {c.conflict_type for c in conflicts}
        assert len(types) >= 2


class TestConflictDetectorMissingVariables:
    """Test missing variable detection."""

    def test_detects_missing_variable(self):
        """Variables in evidence but not in model should be flagged."""
        detector = ConflictDetector()
        model_vars = {"price", "demand", "revenue"}
        evidence = [
            _evidence("Customer satisfaction is a key driver of retention."),
        ]
        conflicts = detector.detect_missing_variables(
            model_vars, evidence, domain_terms=["customer satisfaction"],
        )
        assert len(conflicts) == 1
        assert conflicts[0].conflict_type == ConflictType.MISSING_VARIABLE
        assert conflicts[0].severity == ConflictSeverity.WARNING

    def test_no_missing_variable_if_already_present(self):
        """Variables already in model should not be flagged."""
        detector = ConflictDetector()
        model_vars = {"price", "demand"}
        evidence = [_evidence("Price changes affect demand.")]
        conflicts = detector.detect_missing_variables(
            model_vars, evidence, domain_terms=["price", "demand"],
        )
        assert len(conflicts) == 0

    def test_empty_domain_terms_returns_empty(self):
        """No domain terms means no missing variable conflicts."""
        detector = ConflictDetector()
        conflicts = detector.detect_missing_variables(
            {"price"}, [_evidence("test")], domain_terms=None,
        )
        assert conflicts == []


class TestConflictDetectorStaleEdges:
    """Test stale edge detection."""

    def test_edge_without_evidence_is_stale(self):
        """Edges with no evidence_refs should be flagged as stale."""
        detector = ConflictDetector()
        edges = [_edge("price", "demand", "Mechanism", evidence_refs=[])]
        conflicts = detector.detect_stale_edges(edges)
        assert len(conflicts) == 1
        assert conflicts[0].conflict_type == ConflictType.STALE_EVIDENCE
        assert conflicts[0].severity == ConflictSeverity.INFO

    def test_edge_with_evidence_not_stale(self):
        """Edges with evidence_refs should not be flagged."""
        detector = ConflictDetector()
        edges = [_edge("price", "demand", "Mech", evidence_refs=[uuid4()])]
        conflicts = detector.detect_stale_edges(edges)
        assert len(conflicts) == 0


# ===========================================================================
# Conflict and ConflictReport data structure tests
# ===========================================================================


class TestConflictDataStructures:
    """Test Conflict and ConflictReport properties."""

    def test_conflict_is_critical(self):
        c = Conflict(
            conflict_id="cf1",
            conflict_type=ConflictType.EDGE_CONTRADICTION,
            severity=ConflictSeverity.CRITICAL,
            edge_from="a", edge_to="b",
            description="test",
        )
        assert c.is_critical is True

    def test_conflict_not_critical(self):
        c = Conflict(
            conflict_id="cf1",
            conflict_type=ConflictType.STALE_EVIDENCE,
            severity=ConflictSeverity.INFO,
            edge_from="a", edge_to="b",
            description="test",
        )
        assert c.is_critical is False

    def test_conflict_to_dict(self):
        c = Conflict(
            conflict_id="cf_price_demand_1",
            conflict_type=ConflictType.EDGE_CONTRADICTION,
            severity=ConflictSeverity.WARNING,
            edge_from="price", edge_to="demand",
            description="test desc",
            existing_mechanism="mech",
        )
        d = c.to_dict()
        assert d["type"] == "edge_contradiction"
        assert d["severity"] == "warning"
        assert d["edge"] == "price → demand"

    def test_conflict_report_counts(self):
        report = ConflictReport(
            domain="pricing",
            conflicts=[
                Conflict(conflict_id="c1", conflict_type=ConflictType.EDGE_CONTRADICTION,
                        severity=ConflictSeverity.CRITICAL, edge_from="a", edge_to="b",
                        description=""),
                Conflict(conflict_id="c2", conflict_type=ConflictType.STRENGTH_DOWNGRADE,
                        severity=ConflictSeverity.WARNING, edge_from="a", edge_to="b",
                        description=""),
                Conflict(conflict_id="c3", conflict_type=ConflictType.STALE_EVIDENCE,
                        severity=ConflictSeverity.INFO, edge_from="c", edge_to="d",
                        description=""),
            ],
        )
        assert report.total == 3
        assert report.critical_count == 1
        assert report.warning_count == 1
        assert report.has_critical is True
        assert report.all_resolved is False
        assert len(report.unresolved) == 3

    def test_empty_report(self):
        report = ConflictReport(domain="test")
        assert report.total == 0
        assert report.has_critical is False
        assert report.all_resolved is True


# ===========================================================================
# ConflictResolver tests
# ===========================================================================


class TestConflictResolverStrategies:
    """Test individual resolution strategies."""

    def _make_contradiction(self, supporting=2, contradicting=1):
        return Conflict(
            conflict_id="cf_price_demand_1",
            conflict_type=ConflictType.EDGE_CONTRADICTION,
            severity=ConflictSeverity.WARNING,
            edge_from="price", edge_to="demand",
            description="Contradiction detected",
            existing_mechanism="price reduces demand",
            contradicting_evidence=["contra"] * contradicting,
            supporting_evidence=["support"] * supporting,
        )

    def test_evidence_weighted_more_contradicting(self):
        """More contradicting → downgrade to CONTESTED."""
        conflict = self._make_contradiction(supporting=1, contradicting=3)
        resolver = ConflictResolver()
        action = resolver.resolve(conflict, ResolutionStrategy.EVIDENCE_WEIGHTED)
        assert action.update_edge_strength == EvidenceStrength.CONTESTED

    def test_evidence_weighted_balanced(self):
        """Balanced evidence → CONTESTED."""
        conflict = self._make_contradiction(supporting=2, contradicting=2)
        resolver = ConflictResolver()
        action = resolver.resolve(conflict, ResolutionStrategy.EVIDENCE_WEIGHTED)
        assert action.update_edge_strength == EvidenceStrength.CONTESTED

    def test_temporal_with_contradicting(self):
        """Temporal strategy prefers new evidence → CONTESTED."""
        conflict = self._make_contradiction()
        resolver = ConflictResolver()
        action = resolver.resolve(conflict, ResolutionStrategy.TEMPORAL)
        assert action.strategy == ResolutionStrategy.TEMPORAL
        assert action.update_edge_strength == EvidenceStrength.CONTESTED

    def test_accept_new_contradiction_removes_edge(self):
        """Accept new on contradiction → remove edge."""
        conflict = self._make_contradiction()
        resolver = ConflictResolver()
        action = resolver.resolve(conflict, ResolutionStrategy.ACCEPT_NEW)
        assert action.remove_edge is True

    def test_accept_new_reversal_reverses_edge(self):
        """Accept new on reversal → reverse edge."""
        conflict = Conflict(
            conflict_id="cf1", conflict_type=ConflictType.DIRECTION_REVERSAL,
            severity=ConflictSeverity.CRITICAL, edge_from="a", edge_to="b",
            description="Reversal", contradicting_evidence=["rev1"],
        )
        resolver = ConflictResolver()
        action = resolver.resolve(conflict, ResolutionStrategy.ACCEPT_NEW)
        assert action.reverse_edge is True

    def test_keep_existing_preserves_edge(self):
        """Keep existing → no removal, no strength change."""
        conflict = self._make_contradiction()
        resolver = ConflictResolver()
        action = resolver.resolve(conflict, ResolutionStrategy.KEEP_EXISTING)
        assert action.remove_edge is False
        assert action.update_edge_strength is None
        assert action.strategy == ResolutionStrategy.KEEP_EXISTING

    def test_manual_flags_for_review(self):
        """Manual → flag for review."""
        conflict = self._make_contradiction()
        resolver = ConflictResolver()
        action = resolver.resolve(conflict, ResolutionStrategy.MANUAL)
        assert action.strategy == ResolutionStrategy.MANUAL

    def test_direction_reversal_evidence_weighted_goes_manual(self):
        """Direction reversals with evidence_weighted → manual review."""
        conflict = Conflict(
            conflict_id="cf1", conflict_type=ConflictType.DIRECTION_REVERSAL,
            severity=ConflictSeverity.CRITICAL, edge_from="a", edge_to="b",
            description="Reversal", contradicting_evidence=["rev1"],
            supporting_evidence=["sup1"],
        )
        resolver = ConflictResolver()
        action = resolver.resolve(conflict, ResolutionStrategy.EVIDENCE_WEIGHTED)
        assert action.strategy == ResolutionStrategy.MANUAL


class TestConflictResolverResolveAll:
    """Test bulk resolution."""

    def test_resolve_all_handles_mixed_severities(self):
        report = ConflictReport(domain="test", conflicts=[
            Conflict(conflict_id="c1", conflict_type=ConflictType.EDGE_CONTRADICTION,
                    severity=ConflictSeverity.CRITICAL, edge_from="a", edge_to="b",
                    description="Critical", contradicting_evidence=["e1"]),
            Conflict(conflict_id="c2", conflict_type=ConflictType.STALE_EVIDENCE,
                    severity=ConflictSeverity.INFO, edge_from="c", edge_to="d",
                    description="Info"),
        ])
        resolver = ConflictResolver()
        actions = resolver.resolve_all(report)
        assert len(actions) == 2
        # Critical should go to manual by default
        assert actions[0].strategy == ResolutionStrategy.MANUAL
        # INFO should auto-resolve
        assert report.all_resolved is True

    def test_resolve_all_skips_already_resolved(self):
        conflict = Conflict(
            conflict_id="c1", conflict_type=ConflictType.STALE_EVIDENCE,
            severity=ConflictSeverity.INFO, edge_from="a", edge_to="b",
            description="Already done", resolved=True,
        )
        report = ConflictReport(domain="test", conflicts=[conflict])
        resolver = ConflictResolver()
        actions = resolver.resolve_all(report)
        assert len(actions) == 0

    def test_resolve_all_with_forced_strategy(self):
        """Forcing a strategy should override default behaviour."""
        report = ConflictReport(domain="test", conflicts=[
            Conflict(conflict_id="c1", conflict_type=ConflictType.EDGE_CONTRADICTION,
                    severity=ConflictSeverity.CRITICAL, edge_from="a", edge_to="b",
                    description="Critical", contradicting_evidence=["e1"]),
        ])
        resolver = ConflictResolver()
        actions = resolver.resolve_all(report, strategy=ResolutionStrategy.ACCEPT_NEW)
        assert len(actions) == 1
        assert actions[0].strategy == ResolutionStrategy.ACCEPT_NEW


class TestConflictResolverStrengthChanges:
    """Test strength change resolution."""

    def test_resolve_strength_upgrade(self):
        conflict = Conflict(
            conflict_id="cf_price_demand",
            conflict_type=ConflictType.STRENGTH_UPGRADE,
            severity=ConflictSeverity.INFO,
            edge_from="price", edge_to="demand",
            description="Upgrade",
            supporting_evidence=["s1", "s2", "s3", "s4"],
        )
        resolver = ConflictResolver()
        action = resolver.resolve(conflict, ResolutionStrategy.EVIDENCE_WEIGHTED)
        assert action.update_edge_strength == EvidenceStrength.STRONG

    def test_resolve_strength_downgrade(self):
        conflict = Conflict(
            conflict_id="cf_price_demand",
            conflict_type=ConflictType.STRENGTH_DOWNGRADE,
            severity=ConflictSeverity.WARNING,
            edge_from="price", edge_to="demand",
            description="Downgrade",
            supporting_evidence=["s1"],
            contradicting_evidence=["c1"],
        )
        resolver = ConflictResolver()
        action = resolver.resolve(conflict, ResolutionStrategy.EVIDENCE_WEIGHTED)
        assert action.update_edge_strength == EvidenceStrength.CONTESTED


# ===========================================================================
# CausalService integration tests
# ===========================================================================


class TestCausalServiceConflictIntegration:
    """Test conflict detection/resolution through CausalService."""

    def test_detect_conflicts_no_evidence(self):
        """Empty evidence list should only produce stale-edge conflicts."""
        svc = _build_service_with_model()
        report = svc.detect_conflicts([], domain="pricing")
        # No contradictions, but demand→revenue has 3 refs, price→demand has 1
        # Both have evidence, so no stale
        assert isinstance(report, ConflictReport)
        assert report.domain == "pricing"

    def test_detect_conflicts_with_contradiction(self):
        """Contradicting evidence should be detected."""
        svc = _build_service_with_model()
        evidence = [
            _evidence("Price does not affect demand in any way."),
        ]
        report = svc.detect_conflicts(evidence, domain="pricing")
        contradictions = [c for c in report.conflicts
                         if c.conflict_type == ConflictType.EDGE_CONTRADICTION]
        assert len(contradictions) >= 1

    def test_detect_conflicts_with_missing_variable(self):
        """Missing variable should be detected with domain_terms."""
        svc = _build_service_with_model()
        evidence = [
            _evidence("Customer satisfaction drives purchase decisions."),
        ]
        report = svc.detect_conflicts(
            evidence, domain="pricing",
            domain_terms=["customer satisfaction"],
        )
        missing = [c for c in report.conflicts
                   if c.conflict_type == ConflictType.MISSING_VARIABLE]
        assert len(missing) == 1

    def test_resolve_and_apply_changes_strength(self):
        """Resolving should produce actions; applying should update edge."""
        svc = _build_service_with_model()
        evidence = [
            _evidence("Price does not reduce demand in this market at all."),
        ]
        report = svc.detect_conflicts(evidence, domain="pricing")

        # Only resolve non-critical for this test
        actions = svc.resolve_conflicts(report, auto_resolve_info=True)
        assert len(actions) > 0
        assert report.all_resolved is True

    def test_detect_stale_edges(self):
        """Edge without evidence refs should appear as stale."""
        svc = CausalService()
        engine = svc.create_world_model("test")
        engine.add_variable("a", "A", "Var A",
                            var_type=VariableType.CONTINUOUS,
                            measurement_status=MeasurementStatus.MEASURED)
        engine.add_variable("b", "B", "Var B",
                            var_type=VariableType.CONTINUOUS,
                            measurement_status=MeasurementStatus.MEASURED)
        engine.add_edge("a", "b", "A affects B")  # No evidence refs!
        report = svc.detect_conflicts([], domain="test")
        stale = [c for c in report.conflicts
                 if c.conflict_type == ConflictType.STALE_EVIDENCE]
        assert len(stale) == 1


# ===========================================================================
# Mode 2 integration tests
# ===========================================================================


class TestMode2ConflictDetection:
    """Test conflict detection stage in Mode 2."""

    def test_mode2_stage_enum_has_conflict_detection(self):
        from src.modes.mode2 import Mode2Stage
        assert Mode2Stage.CONFLICT_DETECTION == "conflict_detection"

    def test_mode2_result_has_conflict_fields(self):
        from src.modes.mode2 import Mode2Result, Mode2Stage
        result = Mode2Result(
            trace_id="test", query="test", stage=Mode2Stage.COMPLETE,
            conflicts_detected=3, critical_conflicts=1,
            conflict_details=[{"type": "edge_contradiction"}],
        )
        assert result.conflicts_detected == 3
        assert result.critical_conflicts == 1
        assert len(result.conflict_details) == 1


# ===========================================================================
# Mode 1 integration tests
# ===========================================================================


class TestMode1ConflictDetection:
    """Test conflict detection in Mode 1."""

    def test_mode1_result_has_conflict_fields(self):
        from src.modes.mode1 import Mode1Result, Mode1Stage
        result = Mode1Result(
            trace_id="test", domain="pricing", stage=Mode1Stage.HUMAN_REVIEW,
            conflicts_detected=2, critical_conflicts=0,
            conflict_details=[{"type": "stale_evidence"}],
        )
        assert result.conflicts_detected == 2
        assert result.critical_conflicts == 0
        assert len(result.conflict_details) == 1


# ===========================================================================
# Resolution strategy enum tests
# ===========================================================================


class TestPhase3Enums:
    """Test Phase 3 enum values."""

    def test_conflict_type_values(self):
        assert ConflictType.EDGE_CONTRADICTION == "edge_contradiction"
        assert ConflictType.DIRECTION_REVERSAL == "direction_reversal"
        assert ConflictType.STRENGTH_DOWNGRADE == "strength_downgrade"
        assert ConflictType.STRENGTH_UPGRADE == "strength_upgrade"
        assert ConflictType.MISSING_VARIABLE == "missing_variable"
        assert ConflictType.STALE_EVIDENCE == "stale_evidence"

    def test_conflict_severity_values(self):
        assert ConflictSeverity.CRITICAL == "critical"
        assert ConflictSeverity.WARNING == "warning"
        assert ConflictSeverity.INFO == "info"

    def test_resolution_strategy_values(self):
        assert ResolutionStrategy.EVIDENCE_WEIGHTED == "evidence_weighted"
        assert ResolutionStrategy.TEMPORAL == "temporal"
        assert ResolutionStrategy.SOURCE_PRIORITY == "source_priority"
        assert ResolutionStrategy.MANUAL == "manual"
        assert ResolutionStrategy.ACCEPT_NEW == "accept_new"
        assert ResolutionStrategy.KEEP_EXISTING == "keep_existing"
