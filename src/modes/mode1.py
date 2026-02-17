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
import asyncio
import logging

from src.extraction.service import (
    ExtractionService,
    CanonicalVariableList,
    SynthesizedMechanism,
)
from src.utils.text import truncate_evidence, canonicalize_var_id

from src.agent.orchestrator import AgentOrchestrator
from src.agent.llm_client import LLMClient
from src.causal.service import CausalService
from src.causal.dag_engine import DAGEngine
from src.causal.pywhyllm_bridge import CausalGraphBridge, EdgeProposal, BridgeResult
from src.retrieval.router import RetrievalRouter, RetrievalRequest, RetrievalStrategy
from src.models.causal import WorldModelVersion, VariableDefinition, CausalEdge
from src.models.evidence import EvidenceBundle
from src.models.enums import (
    EdgeStatus,
    EvidenceStrength,
    ModelStatus,
    VariableRole,
    VariableType,
    MeasurementStatus,
)
from src.protocol.state_machine import ProtocolState
from src.verification.loop import VerificationAgent, VerificationResult

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
    
    # ── LangExtract prompts/examples centralised in ExtractionService ───

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        retrieval_router: Optional[RetrievalRouter] = None,
        causal_service: Optional[CausalService] = None,
        causal_bridge: Optional[CausalGraphBridge] = None,
        extraction_service: Optional[ExtractionService] = None,
        verification_agent: Optional[VerificationAgent] = None,
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

        # Centralised extraction service
        self.extraction = extraction_service or ExtractionService()

        # Agentic verification loop
        if verification_agent is not None:
            self.verifier = verification_agent
        else:
            from src.training.spans import SpanCollector
            self.verifier = VerificationAgent(
                llm_client=self.llm,
                retrieval_router=self.retrieval,
                span_collector=SpanCollector(enabled=True),
            )
        
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
        doc_ids: list[str] | None = None,
    ) -> Mode1Result:
        """
        Run full Mode 1 workflow.
        
        Args:
            domain: Decision domain (e.g., "pricing", "retention")
            initial_query: Starting query for evidence gathering
            max_variables: Maximum variables to discover
            max_edges: Maximum edges to create
            doc_ids: Optional list of document IDs to restrict retrieval to.
                     When provided, only evidence from these documents is used.
            
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
            
            variables = await self._discover_variables(domain, initial_query, max_variables, doc_ids=doc_ids)
            audit_entries.append(self._create_audit(
                trace_id, "variable_discovery_complete", 
                {"count": len(variables)}
            ))

            # Stage 1.5: Variable Canonicalization — merge semantic duplicates
            raw_count = len(variables)
            variables = await self._canonicalize_variables(variables, domain)
            audit_entries.append(self._create_audit(
                trace_id, "variable_canonicalization_complete",
                {"raw_count": raw_count, "canonical_count": len(variables)}
            ))
            
            # Stage 2: Evidence Gathering
            self._current_stage = Mode1Stage.EVIDENCE_GATHERING
            evidence_map = await self._gather_evidence(variables, doc_ids=doc_ids)
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

            # Stage 3.5: Mechanism Synthesis — replace raw evidence
            # copy-paste with LLM-synthesized causal reasoning
            edges = await self._synthesize_mechanisms(edges)
            audit_entries.append(self._create_audit(
                trace_id, "mechanism_synthesis_complete",
                {"edges_synthesized": len(edges)}
            ))
            
            # Stage 4: Agentic Verification Loop
            # Replaces naive triangulation with Proposer-Retriever-Judge
            self._current_stage = Mode1Stage.EVIDENCE_TRIANGULATION

            edge_dicts = [
                {
                    "from_var": e.from_var,
                    "to_var": e.to_var,
                    "mechanism": e.mechanism,
                    "evidence_strength": e.strength.value if isinstance(e.strength, EvidenceStrength) else str(e.strength),
                }
                for e in edges
            ]
            verification_results = await self.verifier.verify_all_edges(
                edge_dicts,
                doc_ids=doc_ids,
                trace_id=trace_id,
            )

            grounded_count = sum(1 for vr in verification_results if vr.grounded)
            rejected_count = len(verification_results) - grounded_count
            audit_entries.append(self._create_audit(
                trace_id, "verification_loop_complete",
                {
                    "total_edges": len(verification_results),
                    "grounded": grounded_count,
                    "rejected": rejected_count,
                    "rejection_reasons": [
                        {"edge": f"{vr.from_var}->{vr.to_var}", "reason": vr.rejection_reason}
                        for vr in verification_results if not vr.grounded
                    ],
                }
            ))
            
            # Build the world model
            engine = self.causal.create_world_model(domain)
            
            # Add variables
            added_var_ids: set[str] = set()
            for var in variables:
                var_id = canonicalize_var_id(var.name)
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
                sanitized = canonicalize_var_id(raw_id)
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

            # Add ONLY grounded edges — rejected edges are soft-pruned
            edges_added = 0
            rejected_edges_audit: list[dict] = []
            for vr in verification_results:
                if not vr.grounded:
                    rejected_edges_audit.append({
                        "edge": f"{vr.from_var}->{vr.to_var}",
                        "reason": vr.rejection_reason,
                        "iterations": vr.iterations_used,
                    })
                    continue

                try:
                    from_id = _resolve_var_id(vr.from_var)
                    to_id = _resolve_var_id(vr.to_var)

                    # Build evidence refs from the supporting bundle
                    evidence_bundle_ids: list[UUID] = []
                    if vr.supporting_bundle:
                        evidence_bundle_ids.append(vr.supporting_bundle.bundle_id)
                        self._evidence_cache[vr.supporting_bundle.content_hash[:12]] = vr.supporting_bundle

                    # Map original EdgeCandidate strength
                    original_edge = next(
                        (e for e in edges if e.from_var == vr.from_var and e.to_var == vr.to_var),
                        None,
                    )
                    strength = original_edge.strength if original_edge else EvidenceStrength.HYPOTHESIS

                    _logger.info("Adding grounded edge: %s → %s (confidence=%.2f)",
                                 from_id, to_id, vr.confidence)
                    engine.add_edge(
                        from_var=from_id,
                        to_var=to_id,
                        mechanism=vr.mechanism,
                        strength=strength,
                        evidence_refs=evidence_bundle_ids or None,
                        confidence=vr.confidence,
                        assumptions=vr.assumptions or None,
                        conditions=vr.conditions or None,
                    )
                    edges_added += 1
                except Exception as _edge_err:
                    _logger.warning("Grounded edge %s→%s skipped: %s",
                                    vr.from_var, vr.to_var, _edge_err)
                    continue

            _logger.info("Added %d/%d grounded edges to engine (%d rejected)",
                         edges_added, len(verification_results), len(rejected_edges_audit))

            if rejected_edges_audit:
                audit_entries.append(self._create_audit(
                    trace_id, "rejected_edges_detail",
                    {"rejected": rejected_edges_audit},
                ))

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
        doc_ids: list[str] | None = None,
    ) -> list[VariableCandidate]:
        """Discover causal variables from evidence using LangExtract."""
        # Phase 2: use hybrid retrieval for better recall
        request = RetrievalRequest(
            query=query,
            strategy=RetrievalStrategy.HYBRID,
            max_results=10,
            use_reranking=True,
            doc_ids=doc_ids,
        )
        evidence = await self.retrieval.retrieve(request)

        # Cache evidence
        for e in evidence:
            self._evidence_cache[e.content_hash[:12]] = e

        # Build plain-text evidence document for LangExtract
        evidence_text = "\n\n".join(
            truncate_evidence(e.content, max_chars=800) for e in evidence[:10]
        )

        # If too little text, broaden the search
        if len(evidence_text) < 200 and len(evidence) >= 1:
            broader_query = f"factors costs revenue quality operations competition in {domain}"
            broader_request = RetrievalRequest(
                query=broader_query,
                strategy=RetrievalStrategy.HYBRID,
                max_results=10,
                use_reranking=True,
                doc_ids=doc_ids,
            )
            broader_evidence = await self.retrieval.retrieve(broader_request)
            seen = {e.content_hash for e in evidence}
            for e in broader_evidence:
                if e.content_hash not in seen:
                    evidence.append(e)
                    seen.add(e.content_hash)
                    self._evidence_cache[e.content_hash[:12]] = e
            evidence_text = "\n\n".join(
                truncate_evidence(e.content, max_chars=800) for e in evidence[:15]
            )

        # ── ExtractionService: structured variable extraction ─────────
        # Build evidence_map for citation validation
        ev_map: dict[str, str] = {
            e.content_hash[:12]: e.content
            for e in self._evidence_cache.values()
        }

        extracted = self.extraction.extract_variables(
            evidence_text=evidence_text,
            domain=domain,
            evidence_map=ev_map,
        )

        # Map ExtractedVariable → VariableCandidate
        variables: list[VariableCandidate] = []
        for ev in extracted:
            var_type_str = ev.var_type
            meas_str = ev.measurement_status

            variables.append(VariableCandidate(
                name=ev.name,
                description=ev.description,
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
                evidence_sources=ev.grounded_evidence_keys,
            ))

        _logger.info(
            "LangExtract discovered %d variables: %s",
            len(variables), [v.name for v in variables],
        )

        self._variable_candidates = variables[:max_variables]
        return self._variable_candidates

    # ── Variable Canonicalization ───────────────────────────────────────

    _CANONICALIZATION_SYSTEM_PROMPT = (
        "You are a senior data-modelling expert specialising in causal "
        "inference.  Your task is to consolidate a raw list of candidate "
        "causal variables by merging semantic duplicates.\n\n"
        "RULES:\n"
        "1. Two variables are duplicates if they refer to the SAME "
        "   real-world concept, even when worded differently "
        '   (e.g. "freshest herbs", "highest quality herbs", '
        '   "select herbs" → merge into "Herb Quality").\n'
        "2. Keep variables SEPARATE when they measure genuinely "
        "   different things, even if they are related "
        '   (e.g. "Revenue" and "Profit Margin" are distinct).\n'
        "3. For each merged group, choose the MOST general and precise "
        '   canonical_name (e.g. "Pricing" not "price drops").\n'
        "4. Preserve the var_type and measurement_status of the most "
        "   informative member of each group.\n"
        "5. merged_from MUST list every original variable name in the "
        "   group, even singleton groups.\n"
        "6. Return ONLY the deduplicated list — do NOT invent new "
        "   variables that were not in the input."
    )

    async def _canonicalize_variables(
        self,
        variables: list[VariableCandidate],
        domain: str,
    ) -> list[VariableCandidate]:
        """Merge semantically-duplicate variables using an LLM call.

        Takes the raw ``VariableCandidate`` list from ``_discover_variables``
        and asks the LLM to group synonymous concepts into single canonical
        entries.  Uses ``generate_structured_native`` with
        :class:`CanonicalVariableList` to guarantee valid JSON output.

        Falls back to returning *variables* unchanged if the LLM call
        fails for any reason (network, quota, parse error) so that the
        pipeline is never blocked.
        """
        if len(variables) <= 1:
            return variables

        # Build the prompt listing all candidates
        var_lines: list[str] = []
        for i, v in enumerate(variables, 1):
            var_lines.append(
                f"{i}. \"{v.name}\" — {v.description}  "
                f"[type={v.var_type.value}, status={v.measurement_status.value}]"
            )
        var_block = "\n".join(var_lines)

        prompt = (
            f"Domain: {domain}\n\n"
            f"## Raw candidate variables ({len(variables)} total)\n"
            f"{var_block}\n\n"
            "Merge any semantic duplicates and return the consolidated list."
        )

        try:
            result: CanonicalVariableList = (
                await self.llm.generate_structured_native(
                    prompt=prompt,
                    output_schema=CanonicalVariableList,
                    system_prompt=self._CANONICALIZATION_SYSTEM_PROMPT,
                )
            )
        except Exception as exc:
            _logger.warning(
                "Variable canonicalization LLM call failed — returning "
                "raw variables unchanged: %s",
                exc,
            )
            return variables

        # Build a lookup so we can carry over evidence_sources from the
        # original VariableCandidates that were merged.
        originals_by_lower: dict[str, VariableCandidate] = {
            v.name.lower(): v for v in variables
        }

        canonical: list[VariableCandidate] = []
        for cv in result.variables:
            # Collect evidence sources from every merged original
            merged_evidence: list[str] = []
            for src_name in cv.merged_from:
                orig = originals_by_lower.get(src_name.lower())
                if orig:
                    for key in orig.evidence_sources:
                        if key not in merged_evidence:
                            merged_evidence.append(key)

            canonical.append(VariableCandidate(
                name=cv.canonical_name,
                description=cv.description,
                var_type=(
                    VariableType(cv.var_type)
                    if cv.var_type in ["continuous", "discrete", "binary", "categorical"]
                    else VariableType.CONTINUOUS
                ),
                measurement_status=(
                    MeasurementStatus(cv.measurement_status)
                    if cv.measurement_status in ["measured", "observable", "latent"]
                    else MeasurementStatus.MEASURED
                ),
                evidence_sources=merged_evidence,
            ))

        merged_count = len(variables) - len(canonical)
        if merged_count > 0:
            _logger.info(
                "Canonicalization merged %d → %d variables (-%d)",
                len(variables), len(canonical), merged_count,
            )
            for cv in result.variables:
                if len(cv.merged_from) > 1:
                    _logger.info(
                        "  Merged group: %s → \"%s\"  (%s)",
                        cv.merged_from, cv.canonical_name, cv.merge_reasoning,
                    )
        else:
            _logger.info(
                "Canonicalization kept all %d variables (no merges needed)",
                len(canonical),
            )

        return canonical

    # ── Mechanism Synthesis ──────────────────────────────────────────────

    _MECHANISM_SYNTHESIS_SYSTEM_PROMPT = (
        "You are a senior causal inference expert.  Your task is to "
        "produce a concise, high-quality description of a causal "
        "mechanism based on retrieved evidence.\n\n"
        "RULES:\n"
        "1. `rationale_in_own_words` — Explain HOW the cause produces "
        "   the effect in 1–2 sentences.  Use YOUR OWN words.  Describe "
        "   the real-world causal pathway (e.g. biological process, "
        "   economic mechanism, operational chain).  NEVER copy-paste "
        "   text from the evidence — rephrase it completely.\n"
        "2. `exact_quote` — Copy the single most relevant sentence (or "
        "   clause) from the evidence VERBATIM.  Do not paraphrase, "
        "   truncate, or edit it.  If no suitable quote exists write "
        "   'N/A'.\n"
        "3. `source_citation` — Document title and section of the "
        "   exact_quote (e.g. 'Business_Plan.pdf, Section 3.2').  "
        "   Leave empty if exact_quote is 'N/A'.\n\n"
        "ANTI-PATTERN (FORBIDDEN):\n"
        "  rationale_in_own_words = 'The business plan states that "
        "  marketing spend drives customer acquisition through ...'\n"
        "  ↑ This is just rewording the evidence.  Instead explain the "
        "  actual mechanism: 'Marketing spend increases brand visibility "
        "  and reach, which expands the top of the acquisition funnel "
        "  and converts previously-unaware prospects into customers.'"
    )

    async def _synthesize_mechanisms(
        self,
        edges: list["EdgeCandidate"],
    ) -> list["EdgeCandidate"]:
        """Replace raw/placeholder mechanism strings with LLM-synthesized text.

        For each edge, gathers matching evidence from ``_evidence_cache``,
        calls ``generate_structured_native`` with :class:`SynthesizedMechanism`,
        and reformats the result into a clean mechanism string:

            "{rationale}. (Evidence: '{quote}' — {citation})"

        Falls back to the original mechanism on any per-edge failure so the
        pipeline is never blocked.
        """
        if not edges:
            return edges

        async def _synthesize_one(edge: "EdgeCandidate") -> None:
            """Synthesize mechanism for a single edge (mutates in place)."""
            # Gather up to 3 evidence snippets that mention both variables
            relevant_bundles: list[EvidenceBundle] = []
            from_lower = edge.from_var.replace("_", " ")
            to_lower = edge.to_var.replace("_", " ")
            for eb in self._evidence_cache.values():
                content_lower = eb.content.lower()
                if (
                    (from_lower in content_lower or edge.from_var in content_lower)
                    and (to_lower in content_lower or edge.to_var in content_lower)
                ):
                    relevant_bundles.append(eb)
                    if len(relevant_bundles) >= 3:
                        break

            # If we didn't find bundles mentioning both, broaden to any
            if not relevant_bundles:
                for eb in self._evidence_cache.values():
                    content_lower = eb.content.lower()
                    if from_lower in content_lower or to_lower in content_lower:
                        relevant_bundles.append(eb)
                        if len(relevant_bundles) >= 3:
                            break

            if not relevant_bundles:
                _logger.debug(
                    "No evidence for mechanism synthesis: %s→%s",
                    edge.from_var, edge.to_var,
                )
                return  # keep existing mechanism unchanged

            # Build evidence block for the prompt
            ev_parts: list[str] = []
            for i, eb in enumerate(relevant_bundles, 1):
                source = eb.source.doc_title or eb.source.doc_id
                loc_parts: list[str] = []
                if eb.location.section_name:
                    loc_parts.append(eb.location.section_name)
                if eb.location.page_number:
                    loc_parts.append(f"p.{eb.location.page_number}")
                loc = ", ".join(loc_parts) if loc_parts else "unknown section"
                snippet = truncate_evidence(eb.content, max_chars=600)
                ev_parts.append(
                    f"### Chunk {i} [{source} — {loc}]\n{snippet}\n"
                )
            evidence_block = "\n".join(ev_parts)

            prompt = (
                f"## Proposed causal edge\n"
                f"- **Cause:** {edge.from_var}\n"
                f"- **Effect:** {edge.to_var}\n"
                f"- **Draft mechanism (may be low-quality):** "
                f"{edge.mechanism}\n\n"
                f"## Retrieved evidence\n{evidence_block}\n\n"
                f"Synthesize the causal mechanism."
            )

            try:
                result: SynthesizedMechanism = (
                    await self.llm.generate_structured_native(
                        prompt=prompt,
                        output_schema=SynthesizedMechanism,
                        system_prompt=self._MECHANISM_SYNTHESIS_SYSTEM_PROMPT,
                    )
                )
            except Exception as exc:
                _logger.warning(
                    "Mechanism synthesis failed for %s→%s — keeping "
                    "original: %s",
                    edge.from_var, edge.to_var, exc,
                )
                return  # keep existing mechanism

            # Format the final mechanism string
            rationale = result.rationale_in_own_words.strip()
            quote = result.exact_quote.strip()
            citation = result.source_citation.strip()

            if quote and quote.upper() != "N/A":
                cite_suffix = f" — {citation}" if citation else ""
                edge.mechanism = (
                    f"{rationale} (Evidence: '{quote}'{cite_suffix})"
                )
            else:
                edge.mechanism = rationale

        # Run all synthesis calls concurrently (LLMClient semaphore
        # gates the actual API concurrency).
        await asyncio.gather(
            *[_synthesize_one(e) for e in edges],
            return_exceptions=True,   # don't let one failure abort all
        )

        _logger.info(
            "Mechanism synthesis complete for %d edges", len(edges),
        )
        return edges
    
    async def _gather_evidence(
        self,
        variables: list[VariableCandidate],
        doc_ids: list[str] | None = None,
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
                doc_ids=doc_ids,
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
        var_ids = [
            canonicalize_var_id(v.name)
            for v in variables
        ]
        var_ids = list(dict.fromkeys(vid for vid in var_ids if vid))

        # Build evidence map keyed by variable ID (for PyWhyLLM)
        evidence_by_var: dict[str, list[EvidenceBundle]] = {}
        for var in variables:
            var_id = canonicalize_var_id(var.name)
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

        # ── Stage B: ExtractionService — grounded causal edge extraction ─
        evidence_text = "\n\n".join(
            truncate_evidence(eb.content, max_chars=800) for eb in list(self._evidence_cache.values())[:15]
        )

        var_names = [v.name for v in variables]

        # Build evidence_map for citation validation
        ev_map: dict[str, str] = {
            h: eb.content for h, eb in self._evidence_cache.items()
        }

        extracted_edges = self.extraction.extract_edges(
            evidence_text=evidence_text,
            domain=domain,
            variable_ids=var_ids,
            variable_names=var_names,
            evidence_map=ev_map,
        )

        lx_edge_count = 0
        strength_map = {
            "strong": EvidenceStrength.STRONG,
            "moderate": EvidenceStrength.MODERATE,
            "hypothesis": EvidenceStrength.HYPOTHESIS,
        }
        for ee in extracted_edges:
            pair = (ee.from_var, ee.to_var)
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)

            strength_str = ee.strength
            grounded_refs = ee.grounded_evidence_keys

            # Look up bundle IDs for grounded refs
            grounded_bundle_ids: list[UUID] = []
            for h in grounded_refs:
                eb = self._evidence_cache.get(h)
                if eb:
                    grounded_bundle_ids.append(eb.bundle_id)

            edges.append(EdgeCandidate(
                from_var=ee.from_var,
                to_var=ee.to_var,
                mechanism=ee.mechanism,
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
                var_id = canonicalize_var_id(var_cand.name)
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
