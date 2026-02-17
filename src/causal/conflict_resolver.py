"""
Conflict Detection and Resolution for Causal World Models

Provides:
- Conflict detection: finds contradictions between new evidence and existing DAG edges
- Conflict classification: edge_contradiction, direction_reversal, strength_change, missing_variable
- Severity assessment: critical, warning, info
- Resolution strategies: evidence_weighted, temporal, source_priority, manual
- Resolution application: auto-resolve or flag for human review

Used in:
- Mode 2 Step 4: Check for model conflicts before causal reasoning
- Mode 1 triangulation: detect contradictions during evidence linking
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional
from uuid import UUID

from src.models.causal import CausalEdge, EdgeMetadata
from src.models.enums import EvidenceStrength
from src.models.evidence import EvidenceBundle
from src.utils.text import truncate_at_sentence_boundary

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ConflictType(str, Enum):
    """Types of conflict detected in a world model."""

    EDGE_CONTRADICTION = "edge_contradiction"
    """New evidence directly contradicts an existing edge's claimed mechanism."""

    DIRECTION_REVERSAL = "direction_reversal"
    """Evidence suggests the causal direction is reversed (A→B should be B→A)."""

    STRENGTH_DOWNGRADE = "strength_downgrade"
    """New evidence weakens the claimed strength of a relationship."""

    STRENGTH_UPGRADE = "strength_upgrade"
    """New evidence strengthens a previously weak relationship."""

    MISSING_VARIABLE = "missing_variable"
    """Evidence introduces a variable not present in the model."""

    STALE_EVIDENCE = "stale_evidence"
    """Edge relies on evidence that may be outdated."""


class ConflictSeverity(str, Enum):
    """How critical a conflict is for decision-making."""

    CRITICAL = "critical"
    """Model cannot be trusted for decisions until resolved."""

    WARNING = "warning"
    """Model may produce unreliable results; proceed with caution."""

    INFO = "info"
    """Minor discrepancy; note for future model updates."""


class ResolutionStrategy(str, Enum):
    """Strategy used to resolve a conflict."""

    EVIDENCE_WEIGHTED = "evidence_weighted"
    """Resolve by counting/weighting supporting vs contradicting evidence."""

    TEMPORAL = "temporal"
    """Resolve by preferring more recent evidence."""

    SOURCE_PRIORITY = "source_priority"
    """Resolve by trusting higher-authority sources."""

    MANUAL = "manual"
    """Flag for human review; no automatic resolution."""

    ACCEPT_NEW = "accept_new"
    """Accept the new evidence and update the model."""

    KEEP_EXISTING = "keep_existing"
    """Keep existing model edge, note the contradiction."""


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class Conflict:
    """A detected conflict between evidence and the current world model."""

    conflict_id: str
    conflict_type: ConflictType
    severity: ConflictSeverity
    edge_from: str
    edge_to: str
    description: str
    existing_mechanism: str = ""
    contradicting_evidence: list[str] = field(default_factory=list)
    """Content snippets or bundle IDs of contradicting evidence."""
    supporting_evidence: list[str] = field(default_factory=list)
    """Content snippets or bundle IDs of supporting evidence."""
    suggested_resolution: Optional[str] = None
    resolved: bool = False
    resolution_strategy: Optional[ResolutionStrategy] = None
    resolution_detail: str = ""
    detected_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def is_critical(self) -> bool:
        return self.severity == ConflictSeverity.CRITICAL

    def to_dict(self) -> dict:
        """Serialize for JSON / audit logging."""
        return {
            "conflict_id": self.conflict_id,
            "type": self.conflict_type.value,
            "severity": self.severity.value,
            "edge": f"{self.edge_from} → {self.edge_to}",
            "description": self.description,
            "existing_mechanism": self.existing_mechanism,
            "contradicting_evidence_count": len(self.contradicting_evidence),
            "supporting_evidence_count": len(self.supporting_evidence),
            "resolved": self.resolved,
            "resolution_strategy": self.resolution_strategy.value if self.resolution_strategy else None,
            "resolution_detail": self.resolution_detail,
        }


@dataclass
class ConflictReport:
    """Summary of all detected conflicts for a domain/model."""

    domain: str
    conflicts: list[Conflict] = field(default_factory=list)
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def critical_count(self) -> int:
        return sum(1 for c in self.conflicts if c.is_critical)

    @property
    def warning_count(self) -> int:
        return sum(1 for c in self.conflicts if c.severity == ConflictSeverity.WARNING)

    @property
    def has_critical(self) -> bool:
        return self.critical_count > 0

    @property
    def total(self) -> int:
        return len(self.conflicts)

    @property
    def all_resolved(self) -> bool:
        return all(c.resolved for c in self.conflicts)

    @property
    def unresolved(self) -> list[Conflict]:
        return [c for c in self.conflicts if not c.resolved]


@dataclass
class ResolutionAction:
    """An action to take to resolve a conflict."""

    conflict_id: str
    strategy: ResolutionStrategy
    action: str
    """Human-readable description of what to do."""
    update_edge_strength: Optional[EvidenceStrength] = None
    add_contradicting_refs: list[UUID] = field(default_factory=list)
    remove_edge: bool = False
    reverse_edge: bool = False


# ---------------------------------------------------------------------------
# Conflict Detector
# ---------------------------------------------------------------------------

# Negation signals that indicate contradicting evidence.
_NEGATION_SIGNALS = frozenset({
    "not", "no", "doesn't", "does not", "doesn't",
    "contrary", "opposite", "despite", "however",
    "although", "nevertheless", "contradicts", "refutes",
    "disproves", "invalid", "incorrect", "fails to",
    "no significant", "no effect", "negligible",
    "reverses", "reversed", "inversely", "inverse",
})


class ConflictDetector:
    """
    Detects conflicts between fresh evidence and an existing DAG.

    Works entirely with in-memory data structures — no LLM or DB calls.
    """

    def __init__(
        self,
        negation_signals: frozenset[str] | None = None,
    ):
        self._negation_signals = negation_signals or _NEGATION_SIGNALS

    # ---- public API ---- #

    def detect_edge_conflicts(
        self,
        edge: CausalEdge,
        fresh_evidence: list[EvidenceBundle],
    ) -> list[Conflict]:
        """
        Detect conflicts between a single edge and fresh evidence.

        Checks:
        1. Negation signals in evidence text relative to the mechanism
        2. Evidence that mentions reversed direction
        3. Contradicting evidence count vs supporting count
        """
        conflicts: list[Conflict] = []
        mechanism_tokens = _tokenize(edge.metadata.mechanism)
        from_tok = edge.from_var.lower().replace("_", " ")
        to_tok = edge.to_var.lower().replace("_", " ")

        contradicting: list[str] = []
        supporting: list[str] = []
        reversal_evidence: list[str] = []

        for bundle in fresh_evidence:
            text_lower = bundle.content.lower()
            tokens = set(text_lower.split())

            # Is the evidence relevant to this edge at all?
            if not self._is_relevant(text_lower, from_tok, to_tok, mechanism_tokens):
                continue

            # Check for negation signals near mechanism terms
            has_negation = self._has_negation_near_mechanism(
                text_lower, mechanism_tokens,
            )

            # Check for reversal language (B causes A instead of A causes B)
            has_reversal = self._has_reversal_signal(text_lower, from_tok, to_tok)

            if has_reversal:
                reversal_evidence.append(truncate_at_sentence_boundary(bundle.content, max_chars=300))
            elif has_negation:
                contradicting.append(truncate_at_sentence_boundary(bundle.content, max_chars=300))
            else:
                supporting.append(truncate_at_sentence_boundary(bundle.content, max_chars=300))

        # Build conflicts
        seq = 0

        if reversal_evidence:
            seq += 1
            conflicts.append(Conflict(
                conflict_id=f"cf_{edge.from_var}_{edge.to_var}_{seq}",
                conflict_type=ConflictType.DIRECTION_REVERSAL,
                severity=ConflictSeverity.CRITICAL,
                edge_from=edge.from_var,
                edge_to=edge.to_var,
                description=(
                    f"Evidence suggests the causal direction between "
                    f"{edge.from_var} and {edge.to_var} may be reversed."
                ),
                existing_mechanism=edge.metadata.mechanism,
                contradicting_evidence=reversal_evidence,
                supporting_evidence=supporting,
                suggested_resolution="Review direction with domain expert.",
            ))

        if contradicting:
            severity = self._classify_contradiction_severity(
                supporting_count=len(supporting),
                contradicting_count=len(contradicting),
                current_strength=edge.metadata.evidence_strength,
            )
            seq += 1
            conflicts.append(Conflict(
                conflict_id=f"cf_{edge.from_var}_{edge.to_var}_{seq}",
                conflict_type=ConflictType.EDGE_CONTRADICTION,
                severity=severity,
                edge_from=edge.from_var,
                edge_to=edge.to_var,
                description=(
                    f"New evidence contradicts the claimed mechanism "
                    f"'{edge.metadata.mechanism}' for {edge.from_var} → {edge.to_var}."
                ),
                existing_mechanism=edge.metadata.mechanism,
                contradicting_evidence=contradicting,
                supporting_evidence=supporting,
                suggested_resolution=(
                    "Update edge strength to CONTESTED."
                    if severity != ConflictSeverity.CRITICAL
                    else "Escalate to Mode 1 for model rebuild."
                ),
            ))

        # Strength change detection
        strength_conflict = self._detect_strength_change(
            edge, len(supporting), len(contradicting),
        )
        if strength_conflict:
            conflicts.append(strength_conflict)

        return conflicts

    def detect_missing_variables(
        self,
        model_variables: set[str],
        fresh_evidence: list[EvidenceBundle],
        domain_terms: list[str] | None = None,
    ) -> list[Conflict]:
        """
        Detect potential causal variables mentioned in evidence but absent from model.

        Uses a simple heuristic: look for domain_terms in evidence not in model.
        """
        if not domain_terms:
            return []

        conflicts: list[Conflict] = []
        seq = 0
        model_lower = {v.lower().replace("_", " ") for v in model_variables}

        for term in domain_terms:
            term_lower = term.lower().strip()
            if term_lower in model_lower or term_lower.replace(" ", "_") in model_variables:
                continue

            # Check if any evidence mentions this term
            mentions = [
                truncate_at_sentence_boundary(b.content, max_chars=300) for b in fresh_evidence
                if term_lower in b.content.lower()
            ]
            if mentions:
                seq += 1
                conflicts.append(Conflict(
                    conflict_id=f"cf_missing_{term_lower.replace(' ', '_')}_{seq}",
                    conflict_type=ConflictType.MISSING_VARIABLE,
                    severity=ConflictSeverity.WARNING,
                    edge_from="",
                    edge_to="",
                    description=(
                        f"Variable '{term}' found in evidence but not in the world model."
                    ),
                    contradicting_evidence=mentions,
                    suggested_resolution=f"Consider adding '{term}' to the world model.",
                ))

        return conflicts

    def detect_stale_edges(
        self,
        edges: list[CausalEdge],
        max_evidence_age_days: int = 90,
    ) -> list[Conflict]:
        """
        Detect edges whose evidence may be outdated.

        Since EvidenceBundle UUIDs don't carry timestamps here,
        we flag edges with no evidence refs at all (never grounded).
        """
        conflicts: list[Conflict] = []
        seq = 0
        for edge in edges:
            if not edge.metadata.evidence_refs:
                seq += 1
                conflicts.append(Conflict(
                    conflict_id=f"cf_stale_{edge.from_var}_{edge.to_var}_{seq}",
                    conflict_type=ConflictType.STALE_EVIDENCE,
                    severity=ConflictSeverity.INFO,
                    edge_from=edge.from_var,
                    edge_to=edge.to_var,
                    description=(
                        f"Edge {edge.from_var} → {edge.to_var} has no linked evidence. "
                        f"Consider gathering supporting data."
                    ),
                    existing_mechanism=edge.metadata.mechanism,
                    suggested_resolution="Gather evidence via Mode 1.",
                ))
        return conflicts

    # ---- internal helpers ---- #

    def _is_relevant(
        self,
        text_lower: str,
        from_tok: str,
        to_tok: str,
        mechanism_tokens: set[str],
    ) -> bool:
        """Check if evidence text is relevant to the edge."""
        # Must mention at least one of the variables
        mentions_var = from_tok in text_lower or to_tok in text_lower
        if not mentions_var:
            return False
        # Must share some mechanism terms or mention both variables
        shares_mechanism = bool(mechanism_tokens & set(text_lower.split()))
        mentions_both = from_tok in text_lower and to_tok in text_lower
        return shares_mechanism or mentions_both

    def _has_negation_near_mechanism(
        self,
        text_lower: str,
        mechanism_tokens: set[str],
    ) -> bool:
        """Check if negation signals appear near mechanism terms."""
        words = text_lower.split()
        for i, word in enumerate(words):
            if word in self._negation_signals:
                # Check a window of 5 words around the negation
                window_start = max(0, i - 3)
                window_end = min(len(words), i + 4)
                window = set(words[window_start:window_end])
                if window & mechanism_tokens:
                    return True
        return False

    def _has_reversal_signal(
        self,
        text_lower: str,
        from_tok: str,
        to_tok: str,
    ) -> bool:
        """Check if evidence mentions reversed direction."""
        # Look for patterns like "to_var causes from_var" or "to_var → from_var"
        reversal_patterns = [
            f"{to_tok} causes {from_tok}",
            f"{to_tok} leads to {from_tok}",
            f"{to_tok} drives {from_tok}",
            f"{to_tok} affects {from_tok}",
            f"{to_tok} determines {from_tok}",
            f"{to_tok} → {from_tok}",
        ]
        return any(p in text_lower for p in reversal_patterns)

    def _classify_contradiction_severity(
        self,
        supporting_count: int,
        contradicting_count: int,
        current_strength: EvidenceStrength,
    ) -> ConflictSeverity:
        """Classify how severe a contradiction is."""
        if contradicting_count > supporting_count:
            return ConflictSeverity.CRITICAL
        if current_strength == EvidenceStrength.STRONG and contradicting_count > 0:
            return ConflictSeverity.WARNING
        if contradicting_count >= 2:
            return ConflictSeverity.WARNING
        return ConflictSeverity.INFO

    def _detect_strength_change(
        self,
        edge: CausalEdge,
        supporting_count: int,
        contradicting_count: int,
    ) -> Conflict | None:
        """Detect if evidence warrants a strength reclassification."""
        current = edge.metadata.evidence_strength

        if contradicting_count > 0 and current != EvidenceStrength.CONTESTED:
            # Should be CONTESTED but isn't
            new_str = EvidenceStrength.CONTESTED
        elif supporting_count >= 3 and current in (
            EvidenceStrength.HYPOTHESIS, EvidenceStrength.MODERATE,
        ):
            new_str = EvidenceStrength.STRONG
            conflict_type = ConflictType.STRENGTH_UPGRADE
            return Conflict(
                conflict_id=f"cf_strength_{edge.from_var}_{edge.to_var}",
                conflict_type=conflict_type,
                severity=ConflictSeverity.INFO,
                edge_from=edge.from_var,
                edge_to=edge.to_var,
                description=(
                    f"Edge {edge.from_var} → {edge.to_var} has {supporting_count} "
                    f"supporting pieces of evidence but is classified as '{current.value}'. "
                    f"Consider upgrading to '{new_str.value}'."
                ),
                existing_mechanism=edge.metadata.mechanism,
                suggested_resolution=f"Upgrade strength to {new_str.value}.",
            )
        else:
            return None

        # Strength downgrade (CONTESTED case)
        return Conflict(
            conflict_id=f"cf_strength_{edge.from_var}_{edge.to_var}",
            conflict_type=ConflictType.STRENGTH_DOWNGRADE,
            severity=ConflictSeverity.WARNING,
            edge_from=edge.from_var,
            edge_to=edge.to_var,
            description=(
                f"Edge {edge.from_var} → {edge.to_var} is classified as '{current.value}' "
                f"but has contradicting evidence. Should be '{new_str.value}'."
            ),
            existing_mechanism=edge.metadata.mechanism,
            suggested_resolution=f"Downgrade strength to {new_str.value}.",
        )


# ---------------------------------------------------------------------------
# Conflict Resolver
# ---------------------------------------------------------------------------


class ConflictResolver:
    """
    Resolves detected conflicts using configurable strategies.

    Works entirely with in-memory data structures.
    """

    def __init__(
        self,
        default_strategy: ResolutionStrategy = ResolutionStrategy.EVIDENCE_WEIGHTED,
    ):
        self._default_strategy = default_strategy

    def resolve(
        self,
        conflict: Conflict,
        strategy: ResolutionStrategy | None = None,
    ) -> ResolutionAction:
        """
        Resolve a single conflict and return the action to take.
        """
        strat = strategy or self._default_strategy

        if strat == ResolutionStrategy.EVIDENCE_WEIGHTED:
            return self._resolve_evidence_weighted(conflict)
        elif strat == ResolutionStrategy.TEMPORAL:
            return self._resolve_temporal(conflict)
        elif strat == ResolutionStrategy.ACCEPT_NEW:
            return self._resolve_accept_new(conflict)
        elif strat == ResolutionStrategy.KEEP_EXISTING:
            return self._resolve_keep_existing(conflict)
        else:
            return self._resolve_manual(conflict)

    def resolve_all(
        self,
        report: ConflictReport,
        strategy: ResolutionStrategy | None = None,
        auto_resolve_info: bool = True,
    ) -> list[ResolutionAction]:
        """
        Resolve all conflicts in a report.

        Args:
            report: ConflictReport with detected conflicts.
            strategy: Override strategy for all conflicts (else uses defaults).
            auto_resolve_info: If True, auto-resolve INFO-severity conflicts.

        Returns:
            List of ResolutionAction for each conflict.
        """
        actions: list[ResolutionAction] = []
        for conflict in report.conflicts:
            if conflict.resolved:
                continue

            # Critical conflicts always go to manual unless explicitly overridden
            if conflict.is_critical and strategy is None:
                action = self._resolve_manual(conflict)
            elif conflict.severity == ConflictSeverity.INFO and auto_resolve_info:
                action = self._resolve_evidence_weighted(conflict)
            else:
                action = self.resolve(conflict, strategy)

            conflict.resolved = True
            conflict.resolution_strategy = action.strategy
            conflict.resolution_detail = action.action
            actions.append(action)

        return actions

    # ---- strategy implementations ---- #

    def _resolve_evidence_weighted(self, conflict: Conflict) -> ResolutionAction:
        """Resolve by comparing supporting vs contradicting evidence counts."""
        supporting = len(conflict.supporting_evidence)
        contradicting = len(conflict.contradicting_evidence)

        if conflict.conflict_type == ConflictType.DIRECTION_REVERSAL:
            return ResolutionAction(
                conflict_id=conflict.conflict_id,
                strategy=ResolutionStrategy.MANUAL,
                action=(
                    f"Direction reversal detected for {conflict.edge_from} → {conflict.edge_to}. "
                    f"Requires manual review. ({contradicting} reversal, {supporting} supporting)"
                ),
                reverse_edge=contradicting > supporting,
            )

        if contradicting > supporting:
            new_strength = EvidenceStrength.CONTESTED
            return ResolutionAction(
                conflict_id=conflict.conflict_id,
                strategy=ResolutionStrategy.EVIDENCE_WEIGHTED,
                action=(
                    f"Contradicting evidence ({contradicting}) outweighs supporting ({supporting}). "
                    f"Downgrading edge to CONTESTED."
                ),
                update_edge_strength=new_strength,
            )

        if conflict.conflict_type in (
            ConflictType.STRENGTH_UPGRADE, ConflictType.STRENGTH_DOWNGRADE,
        ):
            # Determine appropriate strength from counts
            if contradicting > 0:
                new_strength = EvidenceStrength.CONTESTED
            elif supporting >= 3:
                new_strength = EvidenceStrength.STRONG
            elif supporting == 2:
                new_strength = EvidenceStrength.MODERATE
            else:
                new_strength = EvidenceStrength.HYPOTHESIS

            return ResolutionAction(
                conflict_id=conflict.conflict_id,
                strategy=ResolutionStrategy.EVIDENCE_WEIGHTED,
                action=(
                    f"Adjusting strength to {new_strength.value} based on evidence "
                    f"({supporting} supporting, {contradicting} contradicting)."
                ),
                update_edge_strength=new_strength,
            )

        # Default: note the contradiction, keep edge
        return ResolutionAction(
            conflict_id=conflict.conflict_id,
            strategy=ResolutionStrategy.EVIDENCE_WEIGHTED,
            action=(
                f"Evidence balanced ({supporting} supporting, {contradicting} contradicting). "
                f"Keeping edge but marking as CONTESTED."
            ),
            update_edge_strength=EvidenceStrength.CONTESTED,
        )

    def _resolve_temporal(self, conflict: Conflict) -> ResolutionAction:
        """Resolve by preferring the most recent evidence (new > old)."""
        # In this heuristic, fresh evidence always wins
        if conflict.contradicting_evidence:
            return ResolutionAction(
                conflict_id=conflict.conflict_id,
                strategy=ResolutionStrategy.TEMPORAL,
                action=(
                    f"Newer evidence contradicts edge {conflict.edge_from} → {conflict.edge_to}. "
                    f"Accepting new evidence."
                ),
                update_edge_strength=EvidenceStrength.CONTESTED,
            )
        return ResolutionAction(
            conflict_id=conflict.conflict_id,
            strategy=ResolutionStrategy.TEMPORAL,
            action="No temporal conflict; keeping existing edge.",
        )

    def _resolve_accept_new(self, conflict: Conflict) -> ResolutionAction:
        """Accept the new evidence unconditionally."""
        if conflict.conflict_type == ConflictType.DIRECTION_REVERSAL:
            return ResolutionAction(
                conflict_id=conflict.conflict_id,
                strategy=ResolutionStrategy.ACCEPT_NEW,
                action=f"Accepting reversal: {conflict.edge_to} → {conflict.edge_from}.",
                reverse_edge=True,
            )

        if conflict.conflict_type == ConflictType.EDGE_CONTRADICTION:
            return ResolutionAction(
                conflict_id=conflict.conflict_id,
                strategy=ResolutionStrategy.ACCEPT_NEW,
                action=f"Accepting contradiction. Removing edge {conflict.edge_from} → {conflict.edge_to}.",
                remove_edge=True,
            )

        return ResolutionAction(
            conflict_id=conflict.conflict_id,
            strategy=ResolutionStrategy.ACCEPT_NEW,
            action=f"Accepting new evidence for {conflict.edge_from} → {conflict.edge_to}.",
            update_edge_strength=EvidenceStrength.CONTESTED,
        )

    def _resolve_keep_existing(self, conflict: Conflict) -> ResolutionAction:
        """Keep the existing model, just note the contradiction."""
        return ResolutionAction(
            conflict_id=conflict.conflict_id,
            strategy=ResolutionStrategy.KEEP_EXISTING,
            action=(
                f"Keeping existing edge {conflict.edge_from} → {conflict.edge_to}. "
                f"Contradiction noted for audit."
            ),
        )

    def _resolve_manual(self, conflict: Conflict) -> ResolutionAction:
        """Flag for manual review."""
        return ResolutionAction(
            conflict_id=conflict.conflict_id,
            strategy=ResolutionStrategy.MANUAL,
            action=(
                f"Conflict requires manual review: {conflict.description}"
            ),
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tokenize(text: str) -> set[str]:
    """Simple tokenization for mechanism matching."""
    import re
    return {
        t for t in re.sub(r"[^a-z0-9\s]", "", text.lower()).split()
        if len(t) > 2
    }
