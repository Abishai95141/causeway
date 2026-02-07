"""
Mode 1: World Model Construction

Implements the full Mode 1 workflow:
1. Variable Discovery - Identify causal variables from evidence
2. Evidence Gathering - Deep search for supporting evidence
3. DAG Drafting - Build causal structure using LLM
4. Evidence Triangulation - Link evidence to edges
5. Human Review - Approval gate before activation
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional
from uuid import UUID, uuid4
from enum import Enum

from src.agent.orchestrator import AgentOrchestrator
from src.agent.llm_client import LLMClient
from src.causal.service import CausalService
from src.causal.dag_engine import DAGEngine
from src.retrieval.router import RetrievalRouter
from src.models.causal import WorldModelVersion, VariableDefinition, CausalEdge
from src.models.evidence import EvidenceBundle
from src.models.enums import (
    EvidenceStrength,
    ModelStatus,
    VariableType,
    MeasurementStatus,
)
from src.protocol.state_machine import ProtocolState


@dataclass
class AuditStep:
    """Simple audit step for internal tracking."""
    trace_id: str
    action: str
    data: dict[str, Any]
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class Mode1Stage(str, Enum):
    """Stages of Mode 1 execution."""
    VARIABLE_DISCOVERY = "variable_discovery"
    EVIDENCE_GATHERING = "evidence_gathering"
    DAG_DRAFTING = "dag_drafting"
    EVIDENCE_TRIANGULATION = "evidence_triangulation"
    HUMAN_REVIEW = "human_review"
    COMPLETE = "complete"


@dataclass
class VariableCandidate:
    """A candidate variable discovered from evidence."""
    name: str
    description: str
    var_type: VariableType
    measurement_status: MeasurementStatus
    evidence_sources: list[str] = field(default_factory=list)
    confidence: float = 0.5


@dataclass
class EdgeCandidate:
    """A candidate causal edge from LLM analysis."""
    from_var: str
    to_var: str
    mechanism: str
    strength: EvidenceStrength
    evidence_refs: list[str] = field(default_factory=list)


@dataclass
class Mode1Result:
    """Result of Mode 1 execution."""
    trace_id: str
    domain: str
    stage: Mode1Stage
    world_model: Optional[WorldModelVersion] = None
    variables_discovered: int = 0
    edges_created: int = 0
    evidence_linked: int = 0
    audit_entries: list["AuditStep"] = field(default_factory=list)
    error: Optional[str] = None
    requires_review: bool = False


class Mode1WorldModelConstruction:
    """
    Mode 1: World Model Construction
    
    Builds causal world models through:
    - LLM-guided variable discovery
    - Evidence retrieval and triangulation
    - DAG structure generation
    - Human approval workflow
    """
    
    VARIABLE_DISCOVERY_PROMPT = """You are analyzing documents to identify causal variables for a decision domain.

Domain: {domain}

Based on the evidence below, identify the key variables that could affect decisions in this domain.
For each variable, provide:
- variable_id: A snake_case identifier
- name: Human-readable name
- description: What this variable represents
- type: continuous, discrete, binary, or categorical
- measurement_status: measured, observable, or latent

Evidence:
{evidence}

Respond with a JSON array of variables:
```json
[
  {{"variable_id": "...", "name": "...", "description": "...", "type": "...", "measurement_status": "..."}}
]
```"""

    DAG_DRAFTING_PROMPT = """You are building a causal DAG for decision support.

Domain: {domain}

Variables in the model:
{variables}

Based on domain knowledge and the evidence, identify causal relationships between these variables.
For each edge, provide:
- from_var: Source variable ID (the cause)
- to_var: Target variable ID (the effect)
- mechanism: Explanation of how the cause affects the effect
- strength: hypothesis, moderate, or strong

Respond with a JSON array of edges:
```json
[
  {{"from_var": "...", "to_var": "...", "mechanism": "...", "strength": "..."}}
]
```"""

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        retrieval_router: Optional[RetrievalRouter] = None,
        causal_service: Optional[CausalService] = None,
    ):
        self.llm = llm_client or LLMClient()
        self.retrieval = retrieval_router or RetrievalRouter()
        self.causal = causal_service or CausalService()
        
        self._current_stage = Mode1Stage.VARIABLE_DISCOVERY
        self._evidence_cache: dict[str, EvidenceBundle] = {}
        self._variable_candidates: list[VariableCandidate] = []
        self._edge_candidates: list[EdgeCandidate] = []
    
    async def initialize(self) -> None:
        """Initialize all components."""
        await self.llm.initialize()
        await self.retrieval.initialize()
    
    async def run(
        self,
        domain: str,
        initial_query: str,
        max_variables: int = 20,
        max_edges: int = 50,
    ) -> Mode1Result:
        """
        Run full Mode 1 workflow.
        
        Args:
            domain: Decision domain (e.g., "pricing", "retention")
            initial_query: Starting query for evidence gathering
            max_variables: Maximum variables to discover
            max_edges: Maximum edges to create
            
        Returns:
            Mode1Result with world model and audit trail
        """
        trace_id = f"m1_{uuid4().hex[:12]}"
        audit_entries = []
        
        try:
            # Stage 1: Variable Discovery
            self._current_stage = Mode1Stage.VARIABLE_DISCOVERY
            audit_entries.append(self._create_audit(
                trace_id, "variable_discovery_start", {"domain": domain}
            ))
            
            variables = await self._discover_variables(domain, initial_query, max_variables)
            audit_entries.append(self._create_audit(
                trace_id, "variable_discovery_complete", 
                {"count": len(variables)}
            ))
            
            # Stage 2: Evidence Gathering
            self._current_stage = Mode1Stage.EVIDENCE_GATHERING
            evidence_map = await self._gather_evidence(variables)
            audit_entries.append(self._create_audit(
                trace_id, "evidence_gathering_complete",
                {"evidence_count": len(self._evidence_cache)}
            ))
            
            # Stage 3: DAG Drafting
            self._current_stage = Mode1Stage.DAG_DRAFTING
            edges = await self._draft_dag(domain, variables, max_edges)
            audit_entries.append(self._create_audit(
                trace_id, "dag_drafting_complete",
                {"edge_count": len(edges)}
            ))
            
            # Stage 4: Evidence Triangulation
            self._current_stage = Mode1Stage.EVIDENCE_TRIANGULATION
            await self._triangulate_evidence(edges)
            audit_entries.append(self._create_audit(
                trace_id, "evidence_triangulation_complete",
                {"linked_evidence": sum(len(e.evidence_refs) for e in edges)}
            ))
            
            # Build the world model
            engine = self.causal.create_world_model(domain)
            
            # Add variables
            for var in variables:
                engine.add_variable(
                    variable_id=var.name.lower().replace(" ", "_"),
                    name=var.name,
                    definition=var.description,
                    var_type=var.var_type,
                    measurement_status=var.measurement_status,
                )
            
            # Add edges
            for edge in edges:
                try:
                    engine.add_edge(
                        from_var=edge.from_var,
                        to_var=edge.to_var,
                        mechanism=edge.mechanism,
                        strength=edge.strength,
                    )
                except Exception:
                    # Skip invalid edges (cycles, missing nodes)
                    continue
            
            # Stage 5: Human Review
            self._current_stage = Mode1Stage.HUMAN_REVIEW
            world_model = engine.to_world_model(domain, f"World model for {domain}")
            world_model.status = ModelStatus.REVIEW
            
            return Mode1Result(
                trace_id=trace_id,
                domain=domain,
                stage=Mode1Stage.HUMAN_REVIEW,
                world_model=world_model,
                variables_discovered=len(variables),
                edges_created=engine.edge_count,
                evidence_linked=len(self._evidence_cache),
                audit_entries=audit_entries,
                requires_review=True,
            )
            
        except Exception as e:
            return Mode1Result(
                trace_id=trace_id,
                domain=domain,
                stage=self._current_stage,
                error=str(e),
                audit_entries=audit_entries,
            )
    
    async def _discover_variables(
        self,
        domain: str,
        query: str,
        max_variables: int,
    ) -> list[VariableCandidate]:
        """Discover causal variables from evidence."""
        # Retrieve initial evidence
        evidence = await self.retrieval.retrieve_simple(query, top_k=10)
        
        # Cache evidence
        for e in evidence:
            self._evidence_cache[e.content_hash[:12]] = e
        
        # Use LLM to identify variables
        evidence_text = "\n\n".join([
            f"[{e.content_hash[:8]}] {e.content[:500]}"
            for e in evidence[:5]
        ])
        
        prompt = self.VARIABLE_DISCOVERY_PROMPT.format(
            domain=domain,
            evidence=evidence_text,
        )
        
        response = await self.llm.generate(prompt)
        
        # Parse variables from response
        variables = self._parse_variables(response.content)
        
        self._variable_candidates = variables[:max_variables]
        return self._variable_candidates
    
    async def _gather_evidence(
        self,
        variables: list[VariableCandidate],
    ) -> dict[str, list[EvidenceBundle]]:
        """Gather additional evidence for each variable."""
        evidence_map: dict[str, list[EvidenceBundle]] = {}
        
        for var in variables:
            query = f"{var.name}: {var.description}"
            bundles = await self.retrieval.retrieve_simple(query, top_k=3)
            
            evidence_map[var.name] = bundles
            
            # Cache all evidence
            for e in bundles:
                self._evidence_cache[e.content_hash[:12]] = e
        
        return evidence_map
    
    async def _draft_dag(
        self,
        domain: str,
        variables: list[VariableCandidate],
        max_edges: int,
    ) -> list[EdgeCandidate]:
        """Draft causal DAG structure using LLM."""
        variables_text = "\n".join([
            f"- {v.name.lower().replace(' ', '_')}: {v.description}"
            for v in variables
        ])
        
        prompt = self.DAG_DRAFTING_PROMPT.format(
            domain=domain,
            variables=variables_text,
        )
        
        response = await self.llm.generate(prompt)
        
        # Parse edges from response
        edges = self._parse_edges(response.content)
        
        self._edge_candidates = edges[:max_edges]
        return self._edge_candidates
    
    async def _triangulate_evidence(
        self,
        edges: list[EdgeCandidate],
    ) -> None:
        """Link evidence to edge candidates."""
        for edge in edges:
            query = f"{edge.from_var} causes {edge.to_var}: {edge.mechanism}"
            bundles = await self.retrieval.retrieve_simple(query, top_k=2)
            
            for b in bundles:
                edge.evidence_refs.append(b.content_hash[:12])
                self._evidence_cache[b.content_hash[:12]] = b
    
    def _parse_variables(self, content: str) -> list[VariableCandidate]:
        """Parse variables from LLM response."""
        import json
        import re
        
        # Extract JSON from response
        json_match = re.search(r'\[[\s\S]*\]', content)
        if not json_match:
            return []
        
        try:
            data = json.loads(json_match.group())
            variables = []
            for item in data:
                var_type = item.get("type", "continuous").lower()
                measurement = item.get("measurement_status", "measured").lower()
                
                variables.append(VariableCandidate(
                    name=item.get("name", item.get("variable_id", "unknown")),
                    description=item.get("description", ""),
                    var_type=VariableType(var_type) if var_type in ["continuous", "discrete", "binary", "categorical"] else VariableType.CONTINUOUS,
                    measurement_status=MeasurementStatus(measurement) if measurement in ["measured", "observable", "latent"] else MeasurementStatus.MEASURED,
                ))
            return variables
        except json.JSONDecodeError:
            return []
    
    def _parse_edges(self, content: str) -> list[EdgeCandidate]:
        """Parse edges from LLM response."""
        import json
        import re
        
        # Extract JSON from response
        json_match = re.search(r'\[[\s\S]*\]', content)
        if not json_match:
            return []
        
        try:
            data = json.loads(json_match.group())
            edges = []
            for item in data:
                strength_str = item.get("strength", "hypothesis").lower()
                strength_map = {
                    "strong": EvidenceStrength.STRONG,
                    "moderate": EvidenceStrength.MODERATE,
                    "hypothesis": EvidenceStrength.HYPOTHESIS,
                }
                
                edges.append(EdgeCandidate(
                    from_var=item.get("from_var", ""),
                    to_var=item.get("to_var", ""),
                    mechanism=item.get("mechanism", ""),
                    strength=strength_map.get(strength_str, EvidenceStrength.HYPOTHESIS),
                ))
            return edges
        except json.JSONDecodeError:
            return []
    
    def _create_audit(
        self,
        trace_id: str,
        action: str,
        data: dict[str, Any],
    ) -> AuditStep:
        """Create an audit step."""
        return AuditStep(
            trace_id=trace_id,
            action=action,
            data=data,
        )
    
    async def approve_model(
        self,
        domain: str,
        approved_by: str,
    ) -> WorldModelVersion:
        """Approve and activate a world model."""
        engine = self.causal.get_engine(domain)
        model = engine.to_world_model(domain, f"World model for {domain}")
        
        model.status = ModelStatus.ACTIVE
        model.approved_by = approved_by
        model.approved_at = datetime.now(timezone.utc)
        
        self._current_stage = Mode1Stage.COMPLETE
        
        return model
    
    @property
    def current_stage(self) -> Mode1Stage:
        """Get current execution stage."""
        return self._current_stage
