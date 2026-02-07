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
from src.models.causal import WorldModelVersion, VariableDefinition, CausalEdge, EdgeMetadata
from src.models.enums import EvidenceStrength, ModelStatus, VariableType, MeasurementStatus

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
        )
    
    def add_causal_link(
        self,
        from_var: str,
        to_var: str,
        mechanism: str,
        domain: Optional[str] = None,
        strength: EvidenceStrength = EvidenceStrength.HYPOTHESIS,
        evidence_refs: Optional[list[UUID]] = None,
    ) -> CausalEdge:
        """Add a causal link to the active world model."""
        engine = self.get_engine(domain)
        return engine.add_edge(from_var, to_var, mechanism, strength, evidence_refs)
    
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
            }
            for e in model.edges
        ]

        async with get_db_session() as session:
            db = DatabaseService(session)
            existing = await db.get_world_model(model.version_id)
            if existing:
                existing.variables = variables_json
                existing.edges = edges_json
                existing.dag_json = {"domain": target_domain}
                await session.flush()
                saved_id = existing.version_id
            else:
                wm = await db.create_world_model(
                    version_id=model.version_id,
                    domain=target_domain,
                    description=f"World model for {target_domain}",
                    variables=variables_json,
                    edges=edges_json,
                    dag_json={"domain": target_domain},
                    status=model.status.value if model.status else "draft",
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
