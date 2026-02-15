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
import os
import textwrap

import langextract as lx

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
    
    # ── LangExtract prompt descriptions ────────────────────────────────────

    VARIABLE_EXTRACT_PROMPT = textwrap.dedent("""\
        Extract every measurable causal variable mentioned or implied in the
        evidence text.  A causal model typically needs 8-15 variables.
        Think across categories: inputs, processes, outputs, quality,
        demand, satisfaction, competition, and environment.
        Use exact phrases from the text for extraction_text.  Do not
        paraphrase or overlap entities.""")

    VARIABLE_EXTRACT_EXAMPLES = [
        lx.data.ExampleData(
            text=(
                "Higher staff training hours improved barista skill levels. "
                "Seasonal foot traffic drives daily customer volume. "
                "Specialty drink pricing is set above competitor averages."
            ),
            extractions=[
                lx.data.Extraction(
                    extraction_class="variable",
                    extraction_text="staff training hours",
                    attributes={
                        "type": "continuous",
                        "measurement_status": "measured",
                        "description": "Hours of barista training per quarter",
                    },
                ),
                lx.data.Extraction(
                    extraction_class="variable",
                    extraction_text="barista skill levels",
                    attributes={
                        "type": "continuous",
                        "measurement_status": "observable",
                        "description": "Assessed proficiency of barista staff",
                    },
                ),
                lx.data.Extraction(
                    extraction_class="variable",
                    extraction_text="Seasonal foot traffic",
                    attributes={
                        "type": "continuous",
                        "measurement_status": "measured",
                        "description": "Pedestrian volume influenced by season",
                    },
                ),
                lx.data.Extraction(
                    extraction_class="variable",
                    extraction_text="daily customer volume",
                    attributes={
                        "type": "continuous",
                        "measurement_status": "measured",
                        "description": "Number of customers visiting per day",
                    },
                ),
                lx.data.Extraction(
                    extraction_class="variable",
                    extraction_text="Specialty drink pricing",
                    attributes={
                        "type": "continuous",
                        "measurement_status": "measured",
                        "description": "Price point for specialty beverages",
                    },
                ),
                lx.data.Extraction(
                    extraction_class="variable",
                    extraction_text="competitor averages",
                    attributes={
                        "type": "continuous",
                        "measurement_status": "observable",
                        "description": "Average competitor pricing in the area",
                    },
                ),
            ],
        )
    ]

    EDGE_EXTRACT_PROMPT = textwrap.dedent("""\
        Extract every causal relationship between variables found in the
        evidence text.  For each relationship use exact sentences or phrases
        from the text as extraction_text — this is the proof that the
        relationship exists.
        Attributes must include from_var and to_var using snake_case IDs,
        the causal mechanism, and evidence strength.
        Consider ALL possible variable pairs; a useful model has many edges.""")

    EDGE_EXTRACT_EXAMPLES = [
        lx.data.ExampleData(
            text=(
                "Higher staff training hours improved barista skill levels. "
                "Better barista skills led to higher drink quality scores. "
                "Seasonal foot traffic drives daily customer volume."
            ),
            extractions=[
                lx.data.Extraction(
                    extraction_class="causal_edge",
                    extraction_text="Higher staff training hours improved barista skill levels",
                    attributes={
                        "from_var": "staff_training_hours",
                        "to_var": "barista_skill_levels",
                        "mechanism": "Training increases proficiency",
                        "strength": "strong",
                    },
                ),
                lx.data.Extraction(
                    extraction_class="causal_edge",
                    extraction_text="Better barista skills led to higher drink quality scores",
                    attributes={
                        "from_var": "barista_skill_levels",
                        "to_var": "drink_quality_scores",
                        "mechanism": "Skilled baristas produce better drinks",
                        "strength": "moderate",
                    },
                ),
                lx.data.Extraction(
                    extraction_class="causal_edge",
                    extraction_text="Seasonal foot traffic drives daily customer volume",
                    attributes={
                        "from_var": "seasonal_foot_traffic",
                        "to_var": "daily_customer_volume",
                        "mechanism": "More pedestrians means more walk-in customers",
                        "strength": "strong",
                    },
                ),
            ],
        )
    ]

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
        """Discover causal variables from evidence using LangExtract."""
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

        # Build plain-text evidence document for LangExtract
        evidence_text = "\n\n".join(
            e.content[:500] for e in evidence[:10]
        )

        # If too little text, broaden the search
        if len(evidence_text) < 200 and len(evidence) >= 1:
            broader_query = f"factors costs revenue quality operations competition in {domain}"
            broader_request = RetrievalRequest(
                query=broader_query,
                strategy=RetrievalStrategy.HYBRID,
                max_results=10,
                use_reranking=True,
            )
            broader_evidence = await self.retrieval.retrieve(broader_request)
            seen = {e.content_hash for e in evidence}
            for e in broader_evidence:
                if e.content_hash not in seen:
                    evidence.append(e)
                    seen.add(e.content_hash)
                    self._evidence_cache[e.content_hash[:12]] = e
            evidence_text = "\n\n".join(
                e.content[:500] for e in evidence[:15]
            )

        # ── LangExtract: structured variable extraction ───────────────
        prompt_desc = (
            f"Domain: {domain}. "
            + self.VARIABLE_EXTRACT_PROMPT
        )

        from src.config import get_settings
        api_key = get_settings().google_ai_api_key

        result = lx.extract(
            text_or_documents=evidence_text,
            prompt_description=prompt_desc,
            examples=self.VARIABLE_EXTRACT_EXAMPLES,
            model_id="gemini-2.5-flash",
            api_key=api_key or os.environ.get("LANGEXTRACT_API_KEY"),
            show_progress=False,
        )

        # Map extractions → VariableCandidate
        variables: list[VariableCandidate] = []
        seen_names: set[str] = set()
        for ext in (result.extractions or []):
            if ext.extraction_class != "variable":
                continue
            attrs = ext.attributes or {}
            name = ext.extraction_text.strip()
            if not name or name.lower() in seen_names:
                continue
            seen_names.add(name.lower())

            var_type_str = attrs.get("type", "continuous").lower()
            meas_str = attrs.get("measurement_status", "measured").lower()

            variables.append(VariableCandidate(
                name=name,
                description=attrs.get("description", ""),
                var_type=(
                    VariableType(var_type_str)
                    if var_type_str in ["continuous", "discrete", "binary", "categorical"]
                    else VariableType.CONTINUOUS
                ),
                measurement_status=(
                    MeasurementStatus(meas_str)
                    if meas_str in ["measured", "observable", "latent"]
                    else MeasurementStatus.MEASURED
                ),
                # Provenance: match extraction_text back to evidence cache
                evidence_sources=[
                    h for h, eb in self._evidence_cache.items()
                    if ext.extraction_text.lower() in eb.content.lower()
                ],
            ))

        _logger.info(
            "LangExtract discovered %d variables: %s",
            len(variables), [v.name for v in variables],
        )

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
        Draft causal DAG structure using PyWhyLLM bridge + LangExtract.

        Two-stage approach:
        1. PyWhyLLM bridge proposes edges via pairwise relationship analysis
        2. LangExtract extracts grounded causal edges from evidence text,
           with each extraction_text providing verifiable provenance

        Every edge's extraction_text is matched back to evidence bundles.
        """
        import re as _re

        var_ids = [
            _re.sub(r'[^a-z0-9]+', '_', v.name.lower()).strip('_')
            for v in variables
        ]
        var_ids = list(dict.fromkeys(vid for vid in var_ids if vid))

        # Build evidence map keyed by variable ID (for PyWhyLLM)
        evidence_by_var: dict[str, list[EvidenceBundle]] = {}
        for var in variables:
            var_id = _re.sub(r'[^a-z0-9]+', '_', var.name.lower()).strip('_')
            if not var_id:
                continue
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

        # ── Stage B: LangExtract — grounded causal edge extraction ────
        # Build variable context string so LangExtract knows the IDs
        var_context = ", ".join(
            f"{_re.sub(r'[^a-z0-9]+', '_', v.name.lower()).strip('_')} ({v.name})"
            for v in variables
        )
        evidence_text = "\n\n".join(
            eb.content[:500] for eb in list(self._evidence_cache.values())[:15]
        )
        full_text = (
            f"Domain: {domain}. "
            f"Variables: {var_context}.\n\n"
            f"Evidence:\n{evidence_text}"
        )

        # ── Dynamic prompt: constrain LangExtract to discovered vars ──
        allowed_vars_str = ", ".join(var_ids)
        dynamic_edge_prompt = textwrap.dedent(f"""\
            Extract every causal relationship between variables found in
            the evidence text.  For each relationship use exact sentences
            or phrases from the text as extraction_text — this is the
            proof that the relationship exists.

            CRITICAL CONSTRAINT: You MUST ONLY use variable IDs from
            this list: {allowed_vars_str}
            Do NOT invent new variable names.  The from_var and to_var
            attributes MUST exactly match one of the IDs listed above.

            Attributes must include from_var and to_var (using the exact
            IDs above), the causal mechanism, and evidence strength
            (strong / moderate / hypothesis).
            Consider ALL possible variable pairs; a useful model has
            many edges.""")

        # ── Dynamic examples: use actual variable IDs, not hardcoded ──
        dynamic_edge_examples: list[lx.data.ExampleData] = []
        if len(var_ids) >= 2:
            _ex_extractions = [
                lx.data.Extraction(
                    extraction_class="causal_edge",
                    extraction_text=(
                        f"Changes in {variables[0].name} influence "
                        f"{variables[1].name}"
                    ),
                    attributes={
                        "from_var": var_ids[0],
                        "to_var": var_ids[1],
                        "mechanism": "Direct causal influence",
                        "strength": "moderate",
                    },
                ),
            ]
            _ex_text_parts = [
                f"{variables[0].name} influences {variables[1].name}."
            ]
            if len(var_ids) >= 3:
                _ex_extractions.append(
                    lx.data.Extraction(
                        extraction_class="causal_edge",
                        extraction_text=(
                            f"{variables[1].name} drives changes in "
                            f"{variables[2].name}"
                        ),
                        attributes={
                            "from_var": var_ids[1],
                            "to_var": var_ids[2],
                            "mechanism": "Indirect influence",
                            "strength": "hypothesis",
                        },
                    )
                )
                _ex_text_parts.append(
                    f"{variables[1].name} drives {variables[2].name}."
                )
            dynamic_edge_examples = [
                lx.data.ExampleData(
                    text=" ".join(_ex_text_parts),
                    extractions=_ex_extractions,
                )
            ]
        else:
            # Fallback: use class-level examples when <2 variables
            dynamic_edge_examples = self.EDGE_EXTRACT_EXAMPLES

        from src.config import get_settings
        api_key = get_settings().google_ai_api_key

        lx_result = lx.extract(
            text_or_documents=full_text,
            prompt_description=dynamic_edge_prompt,
            examples=dynamic_edge_examples,
            model_id="gemini-2.5-flash",
            api_key=api_key or os.environ.get("LANGEXTRACT_API_KEY"),
            show_progress=False,
        )

        lx_edge_count = 0
        for ext in (lx_result.extractions or []):
            if ext.extraction_class != "causal_edge":
                continue
            attrs = ext.attributes or {}
            from_var = (attrs.get("from_var") or "").strip()
            to_var = (attrs.get("to_var") or "").strip()
            if not from_var or not to_var:
                continue
            pair = (from_var, to_var)
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)

            strength_str = (attrs.get("strength") or "hypothesis").lower()
            strength_map = {
                "strong": EvidenceStrength.STRONG,
                "moderate": EvidenceStrength.MODERATE,
                "hypothesis": EvidenceStrength.HYPOTHESIS,
            }

            # ── Provenance: match extraction_text to evidence cache ───
            grounded_refs: list[str] = []
            grounded_bundle_ids: list[UUID] = []
            quote = ext.extraction_text.lower()
            for h, eb in self._evidence_cache.items():
                if quote in eb.content.lower():
                    grounded_refs.append(h)
                    grounded_bundle_ids.append(eb.bundle_id)

            edges.append(EdgeCandidate(
                from_var=from_var,
                to_var=to_var,
                mechanism=attrs.get("mechanism", ""),
                strength=strength_map.get(strength_str, EvidenceStrength.HYPOTHESIS),
                evidence_refs=grounded_refs,
                evidence_bundle_ids=grounded_bundle_ids,
                assumptions=[],
                confidence=0.7 if grounded_refs else 0.4,
            ))
            lx_edge_count += 1

        _logger.info(
            "DAG draft: %d total edges (%d from bridge, %d from LangExtract)",
            len(edges), len(bridge_result.edge_proposals), lx_edge_count,
        )

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
    
    # ── Legacy regex extractors removed ─────────────────────────────────
    # _extract_json_array, _parse_variables, _parse_edges have been
    # replaced by LangExtract structured extraction (see _discover_variables
    # and _draft_dag).  No regex-based JSON parsing remains.
    
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
