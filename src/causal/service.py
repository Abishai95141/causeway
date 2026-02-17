"""
Causal Service

High-level service for causal world model operations.
Combines DAG engine and path finder for complete causal analysis.

World models are cached in memory and persisted to PostgreSQL
via DatabaseService so state survives restarts.
"""

import logging
from typing import Any, Optional
from uuid import UUID

from src.causal.dag_engine import DAGEngine, DAGValidationResult, CycleDetectedError
from src.causal.path_finder import CausalPathFinder, CausalAnalysis
from src.causal.conflict_resolver import (
    ConflictDetector,
    ConflictResolver,
    ConflictReport,
    Conflict,
    ConflictSeverity,
    ResolutionAction,
    ResolutionStrategy,
)
from src.causal.temporal import TemporalTracker, StalenessReport, EdgeTemporalMetadata
from src.training.feedback import FeedbackCollector, OutcomeFeedback, FeedbackSummary
from src.models.causal import (
    WorldModelVersion, VariableDefinition, CausalEdge, EdgeMetadata,
    WorldModelPatch, WorldModelUpdateResult, ModelBridge, BridgeEdge, ConceptMapping,
)
from src.models.evidence import EvidenceBundle
from src.models.enums import EvidenceStrength, ModelStatus, VariableRole, VariableType, MeasurementStatus

logger = logging.getLogger(__name__)


class CausalService:
    """
    High-level causal intelligence service.
    
    Provides:
    - World model creation and management
    - Variable and edge operations
    - Causal reasoning (paths, confounders, mediators)
    - Evidence linking
    - PostgreSQL persistence (save / load)
    """
    
    def __init__(self):
        self._engines: dict[str, DAGEngine] = {}  # domain -> engine (in-memory cache)
        self._active_domain: Optional[str] = None
        self._conflict_detector = ConflictDetector()
        self._conflict_resolver = ConflictResolver()
        self._temporal_trackers: dict[str, TemporalTracker] = {}  # domain -> tracker
        self._feedback_collector = FeedbackCollector()
    
    def create_world_model(self, domain: str) -> DAGEngine:
        """
        Create a new world model for a domain.
        
        Args:
            domain: Domain description (e.g., "pricing", "customer retention")
            
        Returns:
            New DAGEngine instance
        """
        if domain in self._engines:
            raise ValueError(f"World model already exists for domain: {domain}")
        
        engine = DAGEngine()
        self._engines[domain] = engine
        self._active_domain = domain
        
        return engine
    
    def get_engine(self, domain: Optional[str] = None) -> DAGEngine:
        """Get DAGEngine for a domain."""
        target_domain = domain or self._active_domain
        if not target_domain or target_domain not in self._engines:
            raise ValueError(f"No world model for domain: {target_domain}")
        return self._engines[target_domain]
    
    def get_path_finder(self, domain: Optional[str] = None) -> CausalPathFinder:
        """Get CausalPathFinder for a domain."""
        engine = self.get_engine(domain)
        return CausalPathFinder(engine.graph)
    
    def add_variable(
        self,
        variable_id: str,
        name: str,
        definition: str,
        domain: Optional[str] = None,
        var_type: VariableType = VariableType.CONTINUOUS,
        measurement_status: MeasurementStatus = MeasurementStatus.MEASURED,
        unit: Optional[str] = None,
        role: VariableRole = VariableRole.UNKNOWN,
    ) -> VariableDefinition:
        """Add a variable to the active world model."""
        engine = self.get_engine(domain)
        return engine.add_variable(
            variable_id=variable_id,
            name=name,
            definition=definition,
            var_type=var_type,
            measurement_status=measurement_status,
            unit=unit,
            role=role,
        )
    
    def add_causal_link(
        self,
        from_var: str,
        to_var: str,
        mechanism: str,
        domain: Optional[str] = None,
        strength: EvidenceStrength = EvidenceStrength.HYPOTHESIS,
        evidence_refs: Optional[list[UUID]] = None,
        assumptions: Optional[list[str]] = None,
        conditions: Optional[list[str]] = None,
        contradicting_refs: Optional[list[UUID]] = None,
    ) -> CausalEdge:
        """Add a causal link to the active world model."""
        engine = self.get_engine(domain)
        edge = engine.add_edge(
            from_var, to_var, mechanism, strength, evidence_refs,
            assumptions=assumptions,
            conditions=conditions,
            contradicting_refs=contradicting_refs,
        )
        # Phase 4: record temporal metadata
        target_domain = domain or self._active_domain or "unknown"
        tracker = self._get_or_create_tracker(target_domain)
        tracker.record_edge_created(
            from_var, to_var, confidence=edge.metadata.confidence,
        )
        return edge
    
    def analyze_relationship(
        self,
        from_var: str,
        to_var: str,
        domain: Optional[str] = None,
    ) -> CausalAnalysis:
        """
        Analyze the causal relationship between two variables.
        
        Returns paths, confounders, mediators, and effect type.
        """
        finder = self.get_path_finder(domain)
        return finder.analyze(from_var, to_var)
    
    def trace_causal_path(
        self,
        source: str,
        target: str,
        domain: Optional[str] = None,
    ) -> list:
        """Find all causal paths from source to target."""
        finder = self.get_path_finder(domain)
        return finder.find_all_paths(source, target)
    
    def identify_confounders(
        self,
        from_var: str,
        to_var: str,
        domain: Optional[str] = None,
    ) -> list[str]:
        """Identify confounders that affect both cause and effect."""
        finder = self.get_path_finder(domain)
        result = finder.find_confounders(from_var, to_var)
        return result.confounders
    
    def get_variable_effects(
        self,
        variable: str,
        domain: Optional[str] = None,
    ) -> list[str]:
        """Get all variables that this variable affects (descendants)."""
        finder = self.get_path_finder(domain)
        return list(finder.get_descendants(variable))
    
    def get_variable_causes(
        self,
        variable: str,
        domain: Optional[str] = None,
    ) -> list[str]:
        """Get all variables that cause this variable (ancestors)."""
        finder = self.get_path_finder(domain)
        return list(finder.get_ancestors(variable))
    
    def link_evidence(
        self,
        from_var: str,
        to_var: str,
        evidence_id: UUID,
        domain: Optional[str] = None,
    ) -> None:
        """Link evidence to a causal edge."""
        engine = self.get_engine(domain)
        engine.add_evidence_to_edge(from_var, to_var, evidence_id)
    
    def update_edge_strength(
        self,
        from_var: str,
        to_var: str,
        strength: EvidenceStrength,
        domain: Optional[str] = None,
    ) -> None:
        """Update the strength classification of an edge."""
        engine = self.get_engine(domain)
        engine.update_edge_strength(from_var, to_var, strength)
    
    def validate_model(
        self,
        domain: Optional[str] = None,
    ) -> DAGValidationResult:
        """Validate the world model structure."""
        engine = self.get_engine(domain)
        return engine.validate()
    
    def export_world_model(
        self,
        domain: Optional[str] = None,
        version_id: Optional[str] = None,
    ) -> WorldModelVersion:
        """Export the world model as a WorldModelVersion."""
        engine = self.get_engine(domain)
        target_domain = domain or self._active_domain or "unknown"
        return engine.to_world_model(target_domain, f"World model for {target_domain}", version_id)
    
    def import_world_model(self, model: WorldModelVersion) -> DAGEngine:
        """Import a WorldModelVersion into the service."""
        engine = DAGEngine.from_world_model(model)
        self._engines[model.domain] = engine
        self._active_domain = model.domain
        return engine
    
    def list_domains(self) -> list[str]:
        """List all domains with world models."""
        return list(self._engines.keys())
    
    def delete_model(self, domain: str) -> None:
        """Delete a world model."""
        if domain in self._engines:
            del self._engines[domain]
            if self._active_domain == domain:
                self._active_domain = None
    
    def get_model_summary(self, domain: Optional[str] = None) -> dict[str, Any]:
        """Get a summary of the world model."""
        engine = self.get_engine(domain)
        validation = engine.validate()
        
        return {
            "domain": domain or self._active_domain,
            "node_count": engine.node_count,
            "edge_count": engine.edge_count,
            "is_valid": validation.is_valid,
            "warnings": validation.warnings,
            "variables": [v.variable_id for v in engine.variables],
            "edges": [f"{e.from_var} → {e.to_var}" for e in engine.edges],
        }

    # ------------------------------------------------------------------ #
    # Conflict detection & resolution  (Phase 3)
    # ------------------------------------------------------------------ #

    def detect_conflicts(
        self,
        fresh_evidence: list[EvidenceBundle],
        domain: Optional[str] = None,
        domain_terms: Optional[list[str]] = None,
    ) -> ConflictReport:
        """
        Detect conflicts between fresh evidence and the current world model.

        Checks every edge for contradictions, reversals, and strength changes.
        Also detects missing variables if domain_terms are supplied.

        Args:
            fresh_evidence: Newly retrieved evidence bundles.
            domain: Domain of the world model (uses active if omitted).
            domain_terms: Extra terms to check for missing variables.

        Returns:
            ConflictReport summarising all detected conflicts.
        """
        engine = self.get_engine(domain)
        target_domain = domain or self._active_domain or "unknown"
        all_conflicts: list[Conflict] = []

        # 1. Edge-level conflicts
        for edge in engine.edges:
            conflicts = self._conflict_detector.detect_edge_conflicts(
                edge, fresh_evidence,
            )
            all_conflicts.extend(conflicts)

        # 2. Missing-variable conflicts
        model_vars = {v.variable_id for v in engine.variables}
        missing = self._conflict_detector.detect_missing_variables(
            model_vars, fresh_evidence, domain_terms,
        )
        all_conflicts.extend(missing)

        # 3. Stale-evidence conflicts
        stale = self._conflict_detector.detect_stale_edges(engine.edges)
        all_conflicts.extend(stale)

        report = ConflictReport(domain=target_domain, conflicts=all_conflicts)
        logger.info(
            "Conflict detection complete for '%s': %d total (%d critical)",
            target_domain, report.total, report.critical_count,
        )
        return report

    def resolve_conflicts(
        self,
        report: ConflictReport,
        strategy: Optional[ResolutionStrategy] = None,
        auto_resolve_info: bool = True,
    ) -> list[ResolutionAction]:
        """
        Resolve all conflicts in a report.

        Critical conflicts default to MANUAL unless *strategy* is set.
        INFO-severity conflicts are auto-resolved unless disabled.

        Args:
            report: ConflictReport from detect_conflicts().
            strategy: Override resolution strategy (optional).
            auto_resolve_info: Auto-resolve INFO-level conflicts.

        Returns:
            List of ResolutionAction describing what to do.
        """
        actions = self._conflict_resolver.resolve_all(
            report, strategy=strategy, auto_resolve_info=auto_resolve_info,
        )
        logger.info(
            "Resolved %d / %d conflicts for '%s'",
            len(actions), report.total, report.domain,
        )
        return actions

    def apply_resolutions(
        self,
        actions: list[ResolutionAction],
        domain: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Apply resolution actions to the world model.

        Updates edge strengths, removes edges, or flags for manual review.

        Returns:
            Summary dict: {"applied": int, "manual": int, "errors": [str]}
        """
        engine = self.get_engine(domain)
        applied = 0
        manual = 0
        errors: list[str] = []

        for action in actions:
            try:
                if action.strategy == ResolutionStrategy.MANUAL:
                    manual += 1
                    continue

                if action.remove_edge:
                    engine.remove_edge(action.conflict_id.split("_")[1],
                                       action.conflict_id.split("_")[2])
                    applied += 1
                    continue

                if action.update_edge_strength:
                    # Extract edge endpoints from conflict_id (cf_from_to_seq)
                    parts = action.conflict_id.split("_")
                    if len(parts) >= 3:
                        from_var = parts[1]
                        to_var = parts[2]
                        try:
                            engine.update_edge_strength(
                                from_var, to_var, action.update_edge_strength,
                            )
                            applied += 1
                        except ValueError:
                            errors.append(f"Edge not found: {from_var} → {to_var}")
                    else:
                        errors.append(f"Cannot parse edge from conflict_id: {action.conflict_id}")
                else:
                    applied += 1  # No-op actions (keep existing)

            except Exception as exc:
                errors.append(f"Failed to apply {action.conflict_id}: {exc}")

        summary = {"applied": applied, "manual": manual, "errors": errors}
        logger.info("Applied resolutions: %s", summary)
        return summary

    # ------------------------------------------------------------------ #
    # Temporal tracking & feedback  (Phase 4)
    # ------------------------------------------------------------------ #

    def _get_or_create_tracker(self, domain: str) -> TemporalTracker:
        """Lazily initialise a TemporalTracker for *domain*."""
        if domain not in self._temporal_trackers:
            self._temporal_trackers[domain] = TemporalTracker()
        return self._temporal_trackers[domain]

    def check_model_staleness(
        self,
        domain: Optional[str] = None,
    ) -> StalenessReport:
        """
        Check how stale the world model is.

        Returns a StalenessReport with per-edge freshness, overall
        staleness flag, and edges needing re-validation.
        """
        target_domain = domain or self._active_domain or "unknown"
        tracker = self._get_or_create_tracker(target_domain)
        return tracker.check_staleness(domain=target_domain)

    def validate_edge(
        self,
        from_var: str,
        to_var: str,
        new_confidence: Optional[float] = None,
        domain: Optional[str] = None,
    ) -> Optional[EdgeTemporalMetadata]:
        """
        Mark an edge as re-validated (e.g. fresh evidence confirms it).

        Resets the confidence-decay clock for this edge.
        """
        target_domain = domain or self._active_domain or "unknown"
        tracker = self._get_or_create_tracker(target_domain)
        return tracker.record_edge_validated(from_var, to_var, new_confidence)

    def apply_confidence_decay(
        self,
        domain: Optional[str] = None,
    ) -> dict[tuple[str, str], float]:
        """
        Apply time-based confidence decay to all edges in a domain.

        Returns mapping of (from_var, to_var) → decayed confidence.
        """
        target_domain = domain or self._active_domain or "unknown"
        tracker = self._get_or_create_tracker(target_domain)
        return tracker.apply_decay()

    def record_decision_feedback(
        self,
        feedback: OutcomeFeedback,
    ) -> str:
        """
        Record outcome feedback for a past decision.

        Updates per-edge reliability scores automatically.

        Returns:
            The feedback_id.
        """
        return self._feedback_collector.record_feedback(feedback)

    def get_feedback_summary(
        self,
        domain: str = "",
    ) -> FeedbackSummary:
        """Get aggregated feedback summary, optionally filtered by domain."""
        return self._feedback_collector.get_summary(domain)

    def get_training_reward(
        self,
        decision_trace_id: str,
    ) -> float:
        """Compute training reward from outcome feedback for a decision."""
        return self._feedback_collector.compute_training_reward(decision_trace_id)

    # ------------------------------------------------------------------ #
    # World Model Update (incremental patch)
    # ------------------------------------------------------------------ #

    async def update_world_model(
        self,
        domain: str,
        patch: WorldModelPatch,
    ) -> WorldModelUpdateResult:
        """
        Apply an incremental patch to an existing world model and persist.

        1. Get the current engine (load from DB if needed).
        2. Snapshot the old version_id.
        3. Apply ``merge_patch`` to the in-memory DAG.
        4. Save new version to DB with ``replaces_version`` set.
        5. Return the update result.
        """
        engine = self._engines.get(domain)
        if engine is None:
            try:
                engine = await self.load_from_db(domain)
            except ValueError:
                raise ValueError(f"No world model found for domain '{domain}'")

        old_model = engine.to_world_model(domain, f"World model for {domain}")
        old_version_id = old_model.version_id

        result = engine.merge_patch(patch)

        # Persist the updated model
        new_version_id = await self.save_to_db(domain=domain)

        return WorldModelUpdateResult(
            old_version_id=old_version_id,
            new_version_id=new_version_id,
            variables_added=result["variables_added"],
            variables_removed=result["variables_removed"],
            edges_added=result["edges_added"],
            edges_removed=result["edges_removed"],
            edges_updated=result["edges_updated"],
            conflicts=result["conflicts"],
        )

    # ------------------------------------------------------------------ #
    # Cross-Model Bridging
    # ------------------------------------------------------------------ #

    async def build_bridge(
        self,
        source_domain: str,
        target_domain: str,
        llm_client: Any = None,
    ) -> ModelBridge:
        """
        Build a cross-model bridge between two domains.

        Delegates to ``BridgeEngine.build_bridge``.
        Persists the result to PostgreSQL.
        """
        from src.causal.bridge_engine import BridgeEngine

        source_engine = self._engines.get(source_domain)
        if source_engine is None:
            try:
                source_engine = await self.load_from_db(source_domain)
            except ValueError:
                raise ValueError(f"No world model for source domain '{source_domain}'")

        target_engine = self._engines.get(target_domain)
        if target_engine is None:
            try:
                target_engine = await self.load_from_db(target_domain)
            except ValueError:
                raise ValueError(f"No world model for target domain '{target_domain}'")

        source_model = source_engine.to_world_model(source_domain, f"World model for {source_domain}")
        target_model = target_engine.to_world_model(target_domain, f"World model for {target_domain}")

        bridge_eng = BridgeEngine()
        bridge = await bridge_eng.build_bridge(
            source_model, source_engine,
            target_model, target_engine,
            llm_client,
        )

        # Persist bridge to DB
        from src.storage.database import get_db_session, DatabaseService
        async with get_db_session() as session:
            db = DatabaseService(session)
            await db.create_model_bridge(
                source_version_id=bridge.source_version_id,
                target_version_id=bridge.target_version_id,
                bridge_edges=[e.model_dump(mode="json") for e in bridge.bridge_edges],
                shared_concepts=[c.model_dump(mode="json") for c in bridge.shared_concepts],
                status=bridge.status.value if bridge.status else "draft",
                description=f"Bridge: {source_domain} ↔ {target_domain}",
            )

        logger.info(
            "Bridge built: %s ↔ %s — %d edges, %d shared concepts",
            source_domain, target_domain,
            len(bridge.bridge_edges), len(bridge.shared_concepts),
        )
        return bridge

    # ------------------------------------------------------------------ #
    # PostgreSQL persistence
    # ------------------------------------------------------------------ #

    async def save_to_db(self, domain: Optional[str] = None, version_id: Optional[str] = None) -> str:
        """
        Persist the current in-memory world model to PostgreSQL.

        Returns:
            The version_id of the saved record.
        """
        from src.storage.database import get_db_session, DatabaseService

        target_domain = domain or self._active_domain
        if not target_domain:
            raise ValueError("No domain specified and no active domain set")

        model = self.export_world_model(domain=target_domain, version_id=version_id)

        # Serialise variables and edges to JSON-safe dicts
        variables_json = {
            vid: {
                "name": v.name,
                "definition": v.definition,
                "variable_type": v.type.value if v.type else None,
                "measurement_status": v.measurement_status.value if v.measurement_status else None,
                "unit": v.unit,
                "role": v.role.value if v.role else "unknown",
            }
            for vid, v in model.variables.items()
        }
        edges_json = [
            {
                "from_var": e.from_var,
                "to_var": e.to_var,
                "mechanism": e.metadata.mechanism,
                "strength": e.metadata.evidence_strength.value if e.metadata.evidence_strength else None,
                "evidence_refs": [str(r) for r in (e.metadata.evidence_refs or [])],
                "contradicting_refs": [str(r) for r in (e.metadata.contradicting_refs or [])],
                "assumptions": e.metadata.assumptions or [],
                "conditions": e.metadata.conditions or [],
                "confidence": e.metadata.confidence,
            }
            for e in model.edges
        ]

        async with get_db_session() as session:
            db = DatabaseService(session)
            existing = await db.get_world_model(model.version_id)
            engine = self.get_engine(target_domain)
            full_dag_json = engine.to_json()

            if existing:
                existing.variables = variables_json
                existing.edges = edges_json
                existing.dag_json = full_dag_json
                from datetime import datetime as _dt, timezone as _tz
                existing.updated_at = _dt.now(_tz.utc)
                await session.flush()
                saved_id = existing.version_id
            else:
                wm = await db.create_world_model(
                    version_id=model.version_id,
                    domain=target_domain,
                    description=f"World model for {target_domain}",
                    variables=variables_json,
                    edges=edges_json,
                    dag_json=full_dag_json,
                    status=model.status.value if model.status else "draft",
                    replaces_version=model.replaces_version or None,
                )
                saved_id = wm.version_id

        logger.info("World model saved to DB: domain=%s version=%s", target_domain, saved_id)
        return saved_id

    async def load_from_db(self, domain: str) -> DAGEngine:
        """
        Load a world model from PostgreSQL into in-memory cache.

        Returns:
            DAGEngine hydrated from the database row.
        """
        from src.storage.database import get_db_session, DatabaseService

        async with get_db_session() as session:
            db = DatabaseService(session)
            wm = await db.get_active_world_model(domain)

            if wm is None:
                # Try the most recent regardless of status
                from sqlalchemy import select
                from src.storage.database import WorldModelVersionDB
                result = await session.execute(
                    select(WorldModelVersionDB)
                    .where(WorldModelVersionDB.domain == domain)
                    .order_by(WorldModelVersionDB.created_at.desc())
                )
                wm = result.scalars().first()

            if wm is None:
                raise ValueError(f"No world model found in DB for domain: {domain}")

            # Reconstruct VariableDefinition and CausalEdge objects
            variables: dict[str, VariableDefinition] = {}
            for vid, vdata in (wm.variables or {}).items():
                variables[vid] = VariableDefinition(
                    variable_id=vid,
                    name=vdata.get("name", vid),
                    definition=vdata.get("definition", ""),
                    type=VariableType(vdata["variable_type"]) if vdata.get("variable_type") else VariableType.CONTINUOUS,
                    measurement_status=MeasurementStatus(vdata["measurement_status"]) if vdata.get("measurement_status") else MeasurementStatus.MEASURED,
                    unit=vdata.get("unit"),
                    role=VariableRole(vdata["role"]) if vdata.get("role") else VariableRole.UNKNOWN,
                )

            edges: list[CausalEdge] = []
            for edata in (wm.edges or []):
                edges.append(CausalEdge(
                    from_var=edata["from_var"],
                    to_var=edata["to_var"],
                    metadata=EdgeMetadata(
                        mechanism=edata.get("mechanism", ""),
                        evidence_strength=EvidenceStrength(edata["strength"]) if edata.get("strength") else EvidenceStrength.HYPOTHESIS,
                        evidence_refs=[UUID(r) for r in edata.get("evidence_refs", [])],
                        contradicting_refs=[UUID(r) for r in edata.get("contradicting_refs", [])],
                        assumptions=edata.get("assumptions", []),
                        conditions=edata.get("conditions", []),
                        confidence=edata.get("confidence", 0.5),
                    ),
                ))

            model = WorldModelVersion(
                version_id=wm.version_id,
                domain=domain,
                description=wm.description,
                variables=variables,
                edges=edges,
                status=ModelStatus(wm.status) if wm.status else ModelStatus.DRAFT,
            )

            engine = DAGEngine.from_world_model(model)
            self._engines[domain] = engine
            self._active_domain = domain
            logger.info("World model loaded from DB: domain=%s version=%s", domain, wm.version_id)
            return engine
