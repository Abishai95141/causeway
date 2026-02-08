"""
Mode 1: World Model Construction

Implements the full Mode 1 workflow:
1. Variable Discovery - Identify causal variables from evidence
2. Evidence Gathering - Deep search for supporting evidence
3. DAG Drafting - Build causal structure using PyWhyLLM bridge + LLM
4. Evidence Triangulation - Link & validate evidence for each edge
5. Human Review - Approval gate before activation

Phase 1 improvements:
- PyWhyLLM bridge for structured causal reasoning
- Evidence-grounded edge creation (every edge must cite evidence)
- Evidence strength classification (strong/moderate/hypothesis/contested)
- Contradiction detection during triangulation
- Variable role classification (treatment, outcome, confounder, mediator)
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional
from uuid import UUID, uuid4
from enum import Enum
import logging

from src.agent.orchestrator import AgentOrchestrator
from src.agent.llm_client import LLMClient
from src.causal.service import CausalService
from src.causal.dag_engine import DAGEngine
from src.causal.pywhyllm_bridge import CausalGraphBridge, EdgeProposal, BridgeResult
from src.retrieval.router import RetrievalRouter, RetrievalRequest, RetrievalStrategy
from src.models.causal import WorldModelVersion, VariableDefinition, CausalEdge
from src.models.evidence import EvidenceBundle
from src.models.enums import (
    EvidenceStrength,
    ModelStatus,
    VariableRole,
    VariableType,
    MeasurementStatus,
)
from src.protocol.state_machine import ProtocolState

_logger = logging.getLogger(__name__)


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
    role: VariableRole = VariableRole.UNKNOWN


@dataclass
class EdgeCandidate:
    """A candidate causal edge with evidence grounding."""
    from_var: str
    to_var: str
    mechanism: str
    strength: EvidenceStrength
    evidence_refs: list[str] = field(default_factory=list)
    evidence_bundle_ids: list[UUID] = field(default_factory=list)
    contradicting_refs: list[str] = field(default_factory=list)
    contradicting_bundle_ids: list[UUID] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    conditions: list[str] = field(default_factory=list)
    confidence: float = 0.5


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
    conflicts_detected: int = 0
    critical_conflicts: int = 0
    conflict_details: list[dict] = field(default_factory=list)


class Mode1WorldModelConstruction:
    """
    Mode 1: World Model Construction
    
    Builds causal world models through:
    - LLM-guided variable discovery
    - Evidence retrieval and triangulation
    - PyWhyLLM-powered DAG structure generation with evidence grounding
    - Human approval workflow
    """
    
    VARIABLE_DISCOVERY_PROMPT = """You are analyzing documents to identify ALL causal variables for a decision domain.

Domain: {domain}

TASK: Read the evidence below and extract EVERY measurable variable mentioned or implied.
A causal model needs MANY variables to be useful — typically 8-15.

Think through these categories and extract variables from each:
1. INPUTS: resources, costs, investments, staffing, materials
2. PROCESSES: operations, methods, techniques, strategies
3. OUTPUTS: revenue, production, performance metrics
4. QUALITY: product quality, service quality, standards
5. DEMAND: customer traffic, market size, seasonal factors
6. SATISFACTION: customer satisfaction, retention, loyalty
7. COMPETITION: competitor actions, market position
8. ENVIRONMENT: location, regulations, economic conditions

For each variable provide:
- variable_id: snake_case identifier (e.g., "employee_training_hours")
- name: Human-readable name
- description: What this variable represents and how it could be measured
- type: continuous, discrete, binary, or categorical
- measurement_status: measured, observable, or latent

Evidence:
{evidence}

Return a JSON array with AT LEAST 8 variables. Include every factor mentioned in the evidence:
```json
[
  {{"variable_id": "example_var", "name": "Example Variable", "description": "...", "type": "continuous", "measurement_status": "measured"}}
]
```"""

    EVIDENCE_GROUNDED_DAG_PROMPT = """You are building a causal DAG for decision support.
IMPORTANT: Every proposed edge MUST be grounded in the evidence provided below.
Do NOT propose relationships based solely on general knowledge — cite specific evidence.

Domain: {domain}

Variables in the model (use these EXACT IDs as from_var / to_var):
{variables}

Available evidence (each prefixed with its hash ID):
{evidence}

TASK: Systematically consider ALL possible pairs of variables and identify
causal relationships supported by the evidence. You should find MULTIPLE edges —
a model with only 1 or 2 edges for 5+ variables is almost certainly incomplete.

For each causal edge you propose, you MUST:
1. Use the EXACT variable IDs listed above (the snake_case identifiers before the colon)
2. Reference which evidence hash IDs support the relationship
3. Describe the causal mechanism found in the evidence
4. Note any assumptions required for this causal claim
5. Rate strength based on evidence count: strong (3+ sources), moderate (2 sources), hypothesis (1 source)

Respond with a JSON array of edges:
```json
[
  {{
    "from_var": "exact_variable_id",
    "to_var": "exact_variable_id",
    "mechanism": "...",
    "strength": "hypothesis|moderate|strong",
    "evidence_ids": ["hash1", "hash2"],
    "assumptions": ["assumption 1", "assumption 2"]
  }}
]
```"""

    # Legacy prompt kept for fallback if bridge is unavailable
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
        causal_bridge: Optional[CausalGraphBridge] = None,
    ):
        self.llm = llm_client or LLMClient()
        self.retrieval = retrieval_router or RetrievalRouter()
        self.causal = causal_service or CausalService()

        # PyWhyLLM bridge — auto-creates from config if not injected
        if causal_bridge is not None:
            self.bridge = causal_bridge
        else:
            from src.config import get_settings
            settings = get_settings()
            self.bridge = CausalGraphBridge(api_key=settings.google_ai_api_key)
        
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
            supporting_total = sum(len(e.evidence_refs) for e in edges)
            contradicting_total = sum(len(e.contradicting_refs) for e in edges)
            contested_count = sum(
                1 for e in edges if e.strength == EvidenceStrength.CONTESTED
            )
            audit_entries.append(self._create_audit(
                trace_id, "evidence_triangulation_complete",
                {
                    "linked_evidence": supporting_total,
                    "contradicting_evidence": contradicting_total,
                    "contested_edges": contested_count,
                }
            ))
            
            # Build the world model
            engine = self.causal.create_world_model(domain)
            
            # Add variables
            import re as _re
            added_var_ids: set[str] = set()
            for var in variables:
                # Sanitize: lowercase, replace non-alphanum with _, collapse runs
                var_id = _re.sub(r'[^a-z0-9]+', '_', var.name.lower()).strip('_')
                if not var_id or var_id in added_var_ids:
                    _logger.warning("Variable skipped (empty or dup): name=%r id=%r", var.name, var_id)
                    continue
                try:
                    engine.add_variable(
                        variable_id=var_id,
                        name=var.name,
                        definition=var.description,
                        var_type=var.var_type,
                        measurement_status=var.measurement_status,
                        role=var.role,
                    )
                    added_var_ids.add(var_id)
                except Exception as _var_err:
                    _logger.warning("Variable %r skipped: %s", var.name, _var_err)
                    continue

            _logger.info("Added %d variables to engine: %s", len(added_var_ids), sorted(added_var_ids))

            # Build a fuzzy matcher: if the LLM returns a shortened variable ID,
            # try to find the closest match among added variables.
            def _resolve_var_id(raw_id: str) -> str:
                """Resolve a possibly-abbreviated variable ID to an actual one."""
                sanitized = _re.sub(r'[^a-z0-9]+', '_', raw_id.lower()).strip('_')
                if sanitized in added_var_ids:
                    return sanitized
                # Try suffix match: "customer_traffic" should match "daily_customer_traffic"
                suffix_matches = [v for v in added_var_ids if v.endswith(sanitized)]
                if len(suffix_matches) == 1:
                    return suffix_matches[0]
                # Try substring match: pick the shortest variable that contains the ID
                contains_matches = sorted(
                    [v for v in added_var_ids if sanitized in v],
                    key=len,
                )
                if len(contains_matches) == 1:
                    return contains_matches[0]
                # Try the reverse: does the raw ID contain any variable ID?
                contained_in = [v for v in added_var_ids if v in sanitized]
                if len(contained_in) == 1:
                    return contained_in[0]
                return sanitized  # fallback to exact sanitized form

            # Add edges with full evidence metadata
            edges_added = 0
            for edge in edges:
                try:
                    # Sanitize edge variable references and resolve via fuzzy matching
                    from_id = _resolve_var_id(edge.from_var)
                    to_id = _resolve_var_id(edge.to_var)
                    _logger.info("Attempting edge: %s → %s (raw: %s → %s)",
                                 from_id, to_id, edge.from_var, edge.to_var)
                    engine.add_edge(
                        from_var=from_id,
                        to_var=to_id,
                        mechanism=edge.mechanism,
                        strength=edge.strength,
                        evidence_refs=edge.evidence_bundle_ids or None,
                        confidence=edge.confidence,
                        assumptions=edge.assumptions or None,
                        conditions=edge.conditions or None,
                        contradicting_refs=edge.contradicting_bundle_ids or None,
                    )
                    edges_added += 1
                except Exception as _edge_err:
                    _logger.warning("Edge %s→%s skipped: %s", from_id, to_id, _edge_err)
                    continue

            _logger.info("Added %d/%d edges to engine", edges_added, len(edges))

            # Stage 4.5: Post-build conflict detection (Phase 3)
            all_evidence = list(self._evidence_cache.values())
            conflict_report = self.causal.detect_conflicts(
                fresh_evidence=all_evidence,
                domain=domain,
            )
            if conflict_report.total > 0:
                # Auto-resolve non-critical conflicts
                actions = self.causal.resolve_conflicts(conflict_report)
                self.causal.apply_resolutions(actions, domain=domain)
                audit_entries.append(self._create_audit(
                    trace_id, "conflict_detection_complete",
                    {
                        "total": conflict_report.total,
                        "critical": conflict_report.critical_count,
                        "resolved": len(actions),
                    }
                ))
            conflicts_detected = conflict_report.total
            critical_conflicts = conflict_report.critical_count
            conflict_details = [c.to_dict() for c in conflict_report.conflicts]

            # Stage 5: Human Review
            self._current_stage = Mode1Stage.HUMAN_REVIEW
            world_model = engine.to_world_model(domain, f"World model for {domain}")
            world_model.status = ModelStatus.REVIEW

            # Persist draft to PostgreSQL
            try:
                await self.causal.save_to_db(domain=domain, version_id=world_model.version_id)
            except Exception as exc:
                logging.getLogger(__name__).warning("DB save after run failed: %s", exc)

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
                conflicts_detected=conflicts_detected,
                critical_conflicts=critical_conflicts,
                conflict_details=conflict_details,
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
        """Discover causal variables from evidence using hybrid retrieval."""
        # Phase 2: use hybrid retrieval for better recall
        request = RetrievalRequest(
            query=query,
            strategy=RetrievalStrategy.HYBRID,
            max_results=10,
            use_reranking=True,
        )
        evidence = await self.retrieval.retrieve(request)
        
        # Cache evidence
        for e in evidence:
            self._evidence_cache[e.content_hash[:12]] = e
        
        # Use LLM to identify variables
        evidence_text = "\n\n".join([
            f"[{e.content_hash[:8]}] {e.content[:500]}"
            for e in evidence[:10]
        ])
        
        prompt = self.VARIABLE_DISCOVERY_PROMPT.format(
            domain=domain,
            evidence=evidence_text,
        )
        
        response = await self.llm.generate(prompt, temperature=0.3)
        _logger.info("Variable discovery: LLM returned %d chars", len(response.content))

        # Parse variables from response
        variables = self._parse_variables(response.content)

        # If too few variables found, retry with broader evidence and more explicit prompt
        if len(variables) < 7 and len(evidence) >= 3:
            _logger.info("Only %d variables found, retrying with broader evidence", len(variables))
            # Get additional evidence with a different query angle
            broader_query = f"factors costs revenue quality operations competition in {domain}"
            broader_request = RetrievalRequest(
                query=broader_query,
                strategy=RetrievalStrategy.HYBRID,
                max_results=10,
                use_reranking=True,
            )
            broader_evidence = await self.retrieval.retrieve(broader_request)
            # Merge evidence
            all_evidence = list(evidence)
            seen_hashes = {e.content_hash for e in evidence}
            for e in broader_evidence:
                if e.content_hash not in seen_hashes:
                    all_evidence.append(e)
                    seen_hashes.add(e.content_hash)
                    self._evidence_cache[e.content_hash[:12]] = e

            broader_text = "\n\n".join([
                f"[{e.content_hash[:8]}] {e.content[:500]}"
                for e in all_evidence[:15]
            ])
            existing_names = [v.name for v in variables]
            retry_prompt = (
                f"I already found these variables: {existing_names}\n"
                f"But there are MANY more causal factors in this evidence.\n\n"
                f"Evidence:\n{broader_text}\n\n"
                f"List ALL additional measurable variables from this evidence for "
                f"a causal model of '{domain}'. Look for EVERY factor mentioned — "
                f"costs, revenue, quality, satisfaction, traffic, location, marketing, "
                f"competition, pricing, training, staffing, operations, etc.\n\n"
                f"Return a JSON array of at least 6 NEW variables (not: {existing_names}):\n"
                f'[{{"variable_id": "...", "name": "...", "description": "...", '
                f'"type": "continuous|discrete|binary|categorical", '
                f'"measurement_status": "measured|observable|latent"}}]'
            )
            retry_response = await self.llm.generate(retry_prompt, temperature=0.3)
            retry_variables = self._parse_variables(retry_response.content)
            # Merge: keep new ones not already in the set
            existing_names = {v.name.lower() for v in variables}
            for rv in retry_variables:
                if rv.name.lower() not in existing_names:
                    variables.append(rv)
                    existing_names.add(rv.name.lower())
            _logger.info("After retry: %d variables", len(variables))

        _logger.info("Parsed %d variables: %s",
                     len(variables), [v.name for v in variables])

        self._variable_candidates = variables[:max_variables]
        return self._variable_candidates
    
    async def _gather_evidence(
        self,
        variables: list[VariableCandidate],
    ) -> dict[str, list[EvidenceBundle]]:
        """Gather additional evidence for each variable using hybrid retrieval."""
        evidence_map: dict[str, list[EvidenceBundle]] = {}
        
        for var in variables:
            query = f"{var.name}: {var.description}"
            # Phase 2: use hybrid retrieval with re-ranking
            request = RetrievalRequest(
                query=query,
                strategy=RetrievalStrategy.HYBRID,
                max_results=3,
                use_reranking=True,
            )
            bundles = await self.retrieval.retrieve(request)
            
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
        """
        Draft causal DAG structure using PyWhyLLM bridge + evidence-grounded LLM.

        Two-stage approach:
        1. PyWhyLLM bridge proposes edges via pairwise relationship analysis
        2. Evidence-grounded LLM prompt fills in any gaps with explicit evidence citations

        Every edge MUST reference at least one evidence bundle.
        """
        import re as _re

        var_ids = [
            _re.sub(r'[^a-z0-9]+', '_', v.name.lower()).strip('_')
            for v in variables
        ]
        # Remove empty / duplicate ids
        var_ids = list(dict.fromkeys(vid for vid in var_ids if vid))

        # Build evidence map keyed by variable ID
        evidence_by_var: dict[str, list[EvidenceBundle]] = {}
        for var in variables:
            var_id = _re.sub(r'[^a-z0-9]+', '_', var.name.lower()).strip('_')
            if not var_id:
                continue
            # Gather all evidence mentioning this variable
            matching = [
                eb for eb in self._evidence_cache.values()
                if var.name.lower() in eb.content.lower()
                or var_id in eb.content.lower()
            ]
            evidence_by_var[var_id] = matching

        # ── Stage A: PyWhyLLM bridge ──────────────────────────────────
        bridge_result: BridgeResult = self.bridge.build_graph_from_evidence(
            domain=domain,
            variables=var_ids,
            evidence_bundles=evidence_by_var,
        )
        _logger.info(
            "PyWhyLLM bridge proposed %d edges, %d confounder suggestions",
            len(bridge_result.edge_proposals),
            len(bridge_result.confounder_suggestions),
        )

        edges: list[EdgeCandidate] = []
        seen_pairs: set[tuple[str, str]] = set()

        for proposal in bridge_result.edge_proposals:
            if (proposal.from_var, proposal.to_var) in seen_pairs:
                continue
            seen_pairs.add((proposal.from_var, proposal.to_var))
            edges.append(EdgeCandidate(
                from_var=proposal.from_var,
                to_var=proposal.to_var,
                mechanism=proposal.mechanism,
                strength=proposal.strength,
                evidence_refs=proposal.supporting_evidence,
                contradicting_refs=proposal.contradicting_evidence,
                assumptions=proposal.assumptions,
                confidence=proposal.confidence,
            ))

        # ── Stage B: Evidence-grounded LLM prompt ─────────────────────
        # Supplement with LLM-proposed edges grounded in evidence text
        variables_text = "\n".join([
            f"- {_re.sub(r'[^a-z0-9]+', '_', v.name.lower()).strip('_')}: {v.description}"
            for v in variables
        ])

        evidence_text = "\n\n".join([
            f"[{eb.content_hash[:8]}] (Source: {eb.source.doc_title}) {eb.content[:400]}"
            for eb in list(self._evidence_cache.values())[:15]
        ])

        # Build a list of example pairs to guide the LLM
        import itertools as _itertools
        example_pairs = list(_itertools.combinations(var_ids[:8], 2))[:15]
        pairs_text = "\n".join(
            f"  - Does {a} cause or affect {b} (or vice versa)?"
            for a, b in example_pairs
        )

        prompt = self.EVIDENCE_GROUNDED_DAG_PROMPT.format(
            domain=domain,
            variables=variables_text,
            evidence=evidence_text,
        )
        # Append pair suggestions to help the LLM be thorough
        prompt += (
            f"\n\nHere are some variable pairs to consider "
            f"(check ALL of them for causal links):\n{pairs_text}\n\n"
            f"Return ALL edges you find — aim for at least "
            f"{max(3, len(var_ids) - 2)} edges for {len(var_ids)} variables."
        )

        response = await self.llm.generate(prompt)
        _logger.info("Evidence-grounded DAG LLM response (%d chars)", len(response.content))

        llm_edges = self._parse_edges(response.content)
        for e in llm_edges:
            pair = (e.from_var, e.to_var)
            if pair not in seen_pairs:
                seen_pairs.add(pair)
                edges.append(e)

        # Retry once if LLM returned too few edges relative to variable count.
        # Use a focused batch prompt listing uncovered variable pairs explicitly.
        min_expected = max(2, len(var_ids) // 2)
        if len(edges) < min_expected and len(var_ids) >= 4:
            _logger.info(
                "Only %d edges for %d variables (expected >= %d), retrying with batch pairwise prompt",
                len(edges), len(var_ids), min_expected,
            )
            existing_edges_text = "\n".join(
                f"  - {e.from_var} → {e.to_var}" for e in edges
            )
            # Identify uncovered pairs — variables not yet connected
            connected = set()
            for e in edges:
                connected.add(e.from_var)
                connected.add(e.to_var)
            uncovered = [v for v in var_ids if v not in connected]
            # Build pairs between connected and uncovered variables
            retry_pairs = []
            for u in uncovered[:6]:
                for c in list(connected)[:4]:
                    retry_pairs.append((c, u))
            if not retry_pairs:
                # Fallback: all pairs not already covered
                for a, b in _itertools.combinations(var_ids[:8], 2):
                    if (a, b) not in seen_pairs and (b, a) not in seen_pairs:
                        retry_pairs.append((a, b))
            retry_pairs = retry_pairs[:12]

            pairs_to_check = "\n".join(
                f"  - {a} → {b}?" for a, b in retry_pairs
            )
            retry_prompt = (
                f"You already found these edges:\n{existing_edges_text}\n\n"
                f"Domain: {domain}\n\n"
                f"Evidence:\n{evidence_text}\n\n"
                f"Check each of these specific pairs for causal relationships "
                f"supported by the evidence above:\n{pairs_to_check}\n\n"
                f"For each pair where the evidence supports a causal link, "
                f"add it to the result. Return ONLY a JSON array:\n"
                f'[{{"from_var": "...", "to_var": "...", "mechanism": "...", '
                f'"strength": "hypothesis|moderate|strong", "evidence_ids": [...], '
                f'"assumptions": [...]}}]'
            )
            retry_response = await self.llm.generate(retry_prompt)
            _logger.info("DAG retry response (%d chars)", len(retry_response.content))
            retry_edges = self._parse_edges(retry_response.content)
            for e in retry_edges:
                pair = (e.from_var, e.to_var)
                if pair not in seen_pairs:
                    seen_pairs.add(pair)
                    edges.append(e)
            _logger.info("After batch retry: %d total edges", len(edges))

        # If STILL too few edges, try individual pairwise prompts for high-priority pairs
        if len(edges) < min_expected and len(var_ids) >= 4:
            connected = set()
            for e in edges:
                connected.add(e.from_var)
                connected.add(e.to_var)
            uncovered = [v for v in var_ids if v not in connected]
            pairwise_to_try = []
            for u in uncovered[:5]:
                for c in list(connected)[:3]:
                    if (c, u) not in seen_pairs and (u, c) not in seen_pairs:
                        pairwise_to_try.append((c, u))
            # Also add some uncovered-uncovered pairs
            for i, u1 in enumerate(uncovered[:4]):
                for u2 in uncovered[i+1:i+3]:
                    if (u1, u2) not in seen_pairs and (u2, u1) not in seen_pairs:
                        pairwise_to_try.append((u1, u2))

            _logger.info("Trying %d individual pairwise prompts", len(pairwise_to_try[:8]))
            for a, b in pairwise_to_try[:8]:
                pair_prompt = (
                    f"Given this evidence:\n{evidence_text[:1500]}\n\n"
                    f"Is there a causal relationship between '{a}' and '{b}'? "
                    f"If yes, return a JSON object: "
                    f'{{"from_var": "cause_id", "to_var": "effect_id", '
                    f'"mechanism": "...", "strength": "hypothesis|moderate|strong"}}\n'
                    f"If no, return: {{}}"
                )
                try:
                    pair_resp = await self.llm.generate(pair_prompt, temperature=0.2)
                    pair_edges = self._parse_edges(pair_resp.content)
                    for e in pair_edges:
                        pair_key = (e.from_var, e.to_var)
                        if pair_key not in seen_pairs and e.from_var and e.to_var:
                            seen_pairs.add(pair_key)
                            edges.append(e)
                except Exception as _pair_err:
                    _logger.warning("Pairwise prompt %s→%s failed: %s", a, b, _pair_err)
            _logger.info("After pairwise prompts: %d total edges", len(edges))

        _logger.info("DAG draft: %d total edges (%d from bridge, %d from LLM)",
                     len(edges),
                     len(bridge_result.edge_proposals),
                     len(llm_edges))

        # Apply variable role classifications from bridge
        for vc in bridge_result.variable_classifications:
            for var_cand in variables:
                var_id = _re.sub(r'[^a-z0-9]+', '_', var_cand.name.lower()).strip('_')
                if var_id == vc.variable_id:
                    var_cand.role = vc.role
                    break

        self._edge_candidates = edges[:max_edges]
        return self._edge_candidates
    
    async def _triangulate_evidence(
        self,
        edges: list[EdgeCandidate],
    ) -> None:
        """
        Link evidence to edge candidates with support/contradiction search.

        For each edge, searches for:
        1. Supporting evidence: "X causes Y", "X affects Y"
        2. Contradicting evidence: "X does not cause Y", "no relationship between X and Y"

        Then classifies edge strength based on the balance of evidence.
        """
        for edge in edges:
            # ── Supporting evidence search ─────────────────────────────
            support_query = f"{edge.from_var} causes {edge.to_var}: {edge.mechanism}"
            support_bundles = await self.retrieval.retrieve_simple(support_query, max_results=3)

            for b in support_bundles:
                hash_prefix = b.content_hash[:12]
                if hash_prefix not in edge.evidence_refs:
                    edge.evidence_refs.append(hash_prefix)
                    edge.evidence_bundle_ids.append(b.bundle_id)
                self._evidence_cache[hash_prefix] = b

            # ── Contradicting evidence search ──────────────────────────
            contradict_query = (
                f"{edge.from_var} does not cause {edge.to_var} OR "
                f"no relationship between {edge.from_var} and {edge.to_var}"
            )
            contra_bundles = await self.retrieval.retrieve_simple(contradict_query, max_results=2)

            for b in contra_bundles:
                hash_prefix = b.content_hash[:12]
                # Only count as contradicting if the content actually
                # mentions the relationship negation (simple heuristic)
                content_lower = b.content.lower()
                negation_signals = ["not cause", "no effect", "no relationship",
                                     "does not affect", "no significant",
                                     "no evidence", "contrary"]
                is_contradicting = any(sig in content_lower for sig in negation_signals)

                if is_contradicting and hash_prefix not in edge.contradicting_refs:
                    edge.contradicting_refs.append(hash_prefix)
                    edge.contradicting_bundle_ids.append(b.bundle_id)
                elif hash_prefix not in edge.evidence_refs:
                    # If retrieved but not actually contradicting, count as support
                    edge.evidence_refs.append(hash_prefix)
                    edge.evidence_bundle_ids.append(b.bundle_id)

                self._evidence_cache[hash_prefix] = b

            # ── Classify edge strength ─────────────────────────────────
            strength, confidence = CausalGraphBridge.classify_edge_strength(
                supporting_count=len(edge.evidence_refs),
                contradicting_count=len(edge.contradicting_refs),
            )
            edge.strength = strength
            edge.confidence = confidence

            _logger.info(
                "Edge %s→%s: %d supporting, %d contradicting → %s (%.2f)",
                edge.from_var, edge.to_var,
                len(edge.evidence_refs), len(edge.contradicting_refs),
                strength.value, confidence,
            )
    
    @staticmethod
    def _extract_json_array(content: str) -> list[dict]:
        """Robustly extract a JSON array from LLM content."""
        import json
        import re

        # Strategy 1: greedy match inside a fenced code block
        m = re.search(r'```(?:json)?\s*(\[.*\])\s*```', content, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                pass

        # Strategy 2: find the outermost [ ... ] in the entire text
        start = content.find('[')
        end = content.rfind(']')
        if start != -1 and end > start:
            try:
                return json.loads(content[start:end + 1])
            except json.JSONDecodeError:
                pass

        # Strategy 3: LLM sometimes omits the outer [ ].
        # Wrap a fenced code block in [ ] and try again.
        m2 = re.search(r'```(?:json)?\s*(.*?)\s*```', content, re.DOTALL)
        body = m2.group(1).strip() if m2 else content.strip()
        if body and not body.startswith('['):
            try:
                return json.loads(f'[{body}]')
            except json.JSONDecodeError:
                pass

        # Strategy 4: find all top-level { ... } objects via bracket matching
        objects: list[dict] = []
        depth = 0
        obj_start = -1
        for i, ch in enumerate(content):
            if ch == '{':
                if depth == 0:
                    obj_start = i
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0 and obj_start != -1:
                    try:
                        obj = json.loads(content[obj_start:i + 1])
                        if isinstance(obj, dict):
                            objects.append(obj)
                    except json.JSONDecodeError:
                        pass
                    obj_start = -1
        if objects:
            return objects

        # Strategy 5: repair truncated JSON arrays — close open strings,
        # objects, and arrays so we can salvage complete items.
        if start != -1:
            fragment = content[start:]
            # Try closing progressively
            for repair in ['"}]', '"}]', '"}}]', '"]']:
                try:
                    parsed = json.loads(fragment + repair)
                    if isinstance(parsed, list):
                        # Only keep items that have at least from_var or source
                        return [
                            item for item in parsed
                            if isinstance(item, dict)
                            and (item.get("from_var") or item.get("source"))
                        ]
                except json.JSONDecodeError:
                    continue

        return []

    def _parse_variables(self, content: str) -> list[VariableCandidate]:
        """Parse variables from LLM response."""
        data = self._extract_json_array(content)
        if not data:
            return []

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
    
    def _parse_edges(self, content: str) -> list[EdgeCandidate]:
        """Parse edges from LLM response, including evidence IDs and assumptions."""
        data = self._extract_json_array(content)
        if not data:
            return []

        edges = []
        for item in data:
            # Accept both from_var/to_var and source/target naming conventions
            from_var = (
                item.get("from_var")
                or item.get("source")
                or item.get("cause")
                or ""
            ).strip()
            to_var = (
                item.get("to_var")
                or item.get("target")
                or item.get("effect")
                or ""
            ).strip()
            # Skip edges with empty or missing variable IDs
            if not from_var or not to_var:
                continue
            strength_str = item.get("strength", "hypothesis").lower()
            strength_map = {
                "strong": EvidenceStrength.STRONG,
                "moderate": EvidenceStrength.MODERATE,
                "hypothesis": EvidenceStrength.HYPOTHESIS,
            }

            # Parse evidence IDs if provided by evidence-grounded prompt
            evidence_ids = item.get("evidence_ids", [])
            assumptions = item.get("assumptions", [])

            edges.append(EdgeCandidate(
                from_var=from_var,
                to_var=to_var,
                mechanism=item.get("mechanism", ""),
                strength=strength_map.get(strength_str, EvidenceStrength.HYPOTHESIS),
                evidence_refs=evidence_ids if isinstance(evidence_ids, list) else [],
                assumptions=assumptions if isinstance(assumptions, list) else [],
            ))
        return edges
    
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
        """Approve and activate a world model, persisting to PostgreSQL."""
        engine = self.causal.get_engine(domain)
        model = engine.to_world_model(domain, f"World model for {domain}")
        
        model.status = ModelStatus.ACTIVE
        model.approved_by = approved_by
        model.approved_at = datetime.now(timezone.utc)
        
        # Persist to PostgreSQL
        try:
            await self.causal.save_to_db(domain=domain, version_id=model.version_id)
        except Exception as exc:
            logging.getLogger(__name__).warning("DB save on approve failed: %s", exc)
        
        self._current_stage = Mode1Stage.COMPLETE
        
        return model
    
    @property
    def current_stage(self) -> Mode1Stage:
        """Get current execution stage."""
        return self._current_stage
