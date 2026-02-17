"""
Mode 2: Decision Support

Implements the full Mode 2 workflow:
1. Query Parsing - Understand the decision question
2. Model Retrieval - Find relevant world models
3. Evidence Refresh - Update with recent evidence
4. Causal Reasoning - Trace paths and identify confounders
5. Recommendation Synthesis - Generate actionable recommendations
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any, Optional
from uuid import UUID, uuid4
from enum import Enum
import textwrap

from src.extraction.service import ExtractionService

from src.agent.llm_client import LLMClient
from src.causal.service import CausalService
from src.causal.path_finder import CausalAnalysis
from src.retrieval.router import RetrievalRouter, RetrievalRequest, RetrievalStrategy
from src.models.causal import WorldModelVersion
from src.models.evidence import EvidenceBundle
from src.models.decision import DecisionQuery, DecisionRecommendation
from src.models.enums import ConfidenceLevel, EvidenceStrength


@dataclass
class AuditStep:
    """Simple audit step for internal tracking."""
    trace_id: str
    action: str
    data: dict[str, Any]
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class Mode2Stage(str, Enum):
    """Stages of Mode 2 execution."""
    QUERY_PARSING = "query_parsing"
    MODEL_RETRIEVAL = "model_retrieval"
    STALENESS_CHECK = "staleness_check"
    EVIDENCE_REFRESH = "evidence_refresh"
    CONFLICT_DETECTION = "conflict_detection"
    CAUSAL_REASONING = "causal_reasoning"
    RECOMMENDATION_SYNTHESIS = "recommendation_synthesis"
    COMPLETE = "complete"


@dataclass
class ParsedQuery:
    """Result of query parsing."""
    domain: str
    intervention: str  # What action is being considered
    target_outcome: str  # What outcome is being optimized
    constraints: list[str] = field(default_factory=list)


@dataclass
class CausalInsight:
    """Insight from causal analysis."""
    path_description: str
    confounders: list[str]
    mediators: list[str]
    direct_effect: bool
    strength: str


@dataclass
class Mode2Result:
    """Result of Mode 2 execution."""
    trace_id: str
    query: str
    stage: Mode2Stage
    recommendation: Optional[DecisionRecommendation] = None
    model_used: Optional[str] = None
    evidence_count: int = 0
    causal_insights: list[CausalInsight] = field(default_factory=list)
    audit_entries: list["AuditStep"] = field(default_factory=list)
    error: Optional[str] = None
    escalate_to_mode1: bool = False
    escalation_reason: Optional[str] = None
    conflicts_detected: int = 0
    critical_conflicts: int = 0
    conflict_details: list[dict] = field(default_factory=list)
    model_staleness: Optional[dict] = None
    confidence_decay_applied: bool = False


class Mode2DecisionSupport:
    """
    Mode 2: Decision Support
    
    Provides causal reasoning-backed recommendations:
    - Query understanding
    - World model retrieval
    - Evidence-based reasoning
    - Confidence-scored recommendations
    """
    
    # ── LangExtract prompts/examples centralised in ExtractionService ──

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        retrieval_router: Optional[RetrievalRouter] = None,
        causal_service: Optional[CausalService] = None,
        evidence_freshness_days: int = 30,
        extraction_service: Optional[ExtractionService] = None,
    ):
        self.llm = llm_client or LLMClient()
        self.retrieval = retrieval_router or RetrievalRouter()
        self.causal = causal_service or CausalService()
        self.evidence_freshness_days = evidence_freshness_days

        # Centralised extraction service
        self.extraction = extraction_service or ExtractionService()
        
        self._current_stage = Mode2Stage.QUERY_PARSING
        self._evidence_cache: dict[str, EvidenceBundle] = {}
    
    async def initialize(self) -> None:
        """Initialize all components."""
        await self.llm.initialize()
        await self.retrieval.initialize()
    
    async def run(
        self,
        query: str,
        domain_hint: Optional[str] = None,
    ) -> Mode2Result:
        """
        Run full Mode 2 workflow.
        
        Args:
            query: Decision query from user
            domain_hint: Optional domain hint
            
        Returns:
            Mode2Result with recommendation and evidence
        """
        trace_id = f"m2_{uuid4().hex[:12]}"
        audit_entries = []
        
        try:
            # Stage 1: Query Parsing
            self._current_stage = Mode2Stage.QUERY_PARSING
            available_domains = self.causal.list_domains()
            parsed = await self._parse_query(query, available_domains=available_domains or None)
            domain = domain_hint or parsed.domain

            # Fuzzy-resolve domain against what actually exists
            if domain not in available_domains and available_domains:
                resolved = self._resolve_domain(domain, available_domains)
                if resolved:
                    domain = resolved

            audit_entries.append(self._create_audit(
                trace_id, "query_parsed",
                {"domain": domain, "intervention": parsed.intervention}
            ))
            
            # Stage 2: Model Retrieval
            self._current_stage = Mode2Stage.MODEL_RETRIEVAL
            
            # Check if we have a world model for this domain
            if domain not in available_domains:
                # Escalate to Mode 1
                return Mode2Result(
                    trace_id=trace_id,
                    query=query,
                    stage=Mode2Stage.MODEL_RETRIEVAL,
                    audit_entries=audit_entries,
                    escalate_to_mode1=True,
                    escalation_reason=f"No world model found for domain: {domain}. Consider running Mode 1 first.",
                )
            
            audit_entries.append(self._create_audit(
                trace_id, "model_retrieved", {"domain": domain}
            ))
            
            # Stage 2.5: Staleness Check (Phase 4)
            self._current_stage = Mode2Stage.STALENESS_CHECK
            staleness_report = self.causal.check_model_staleness(domain=domain)
            model_staleness = staleness_report.to_dict()

            # Apply confidence decay before reasoning
            self.causal.apply_confidence_decay(domain=domain)
            confidence_decay_applied = True

            if staleness_report.is_stale:
                return Mode2Result(
                    trace_id=trace_id,
                    query=query,
                    stage=Mode2Stage.STALENESS_CHECK,
                    audit_entries=audit_entries,
                    escalate_to_mode1=True,
                    escalation_reason=(
                        f"World model is stale (age: {staleness_report.model_age_days:.0f} days, "
                        f"threshold: {staleness_report.staleness_threshold_days:.0f} days, "
                        f"freshness: {staleness_report.overall_freshness:.1%}). "
                        f"Rebuild recommended via Mode 1."
                    ),
                    model_staleness=model_staleness,
                    confidence_decay_applied=confidence_decay_applied,
                )

            audit_entries.append(self._create_audit(
                trace_id, "staleness_check_passed",
                model_staleness,
            ))
            
            # Stage 3: Evidence Refresh
            self._current_stage = Mode2Stage.EVIDENCE_REFRESH
            evidence = await self._refresh_evidence(parsed)
            
            if len(evidence) < 2:
                return Mode2Result(
                    trace_id=trace_id,
                    query=query,
                    stage=Mode2Stage.EVIDENCE_REFRESH,
                    audit_entries=audit_entries,
                    escalate_to_mode1=True,
                    escalation_reason="Insufficient evidence found. Consider gathering more documents via Mode 1.",
                )
            
            audit_entries.append(self._create_audit(
                trace_id, "evidence_refreshed",
                {"count": len(evidence)}
            ))
            
            # Stage 3.5: Conflict Detection (Phase 3)
            self._current_stage = Mode2Stage.CONFLICT_DETECTION
            conflict_report = self.causal.detect_conflicts(
                fresh_evidence=evidence,
                domain=domain,
            )

            conflicts_detected = conflict_report.total
            critical_conflicts = conflict_report.critical_count
            conflict_details = [c.to_dict() for c in conflict_report.conflicts]

            if conflict_report.has_critical:
                # Auto-resolve non-critical; critical → manual
                self.causal.resolve_conflicts(conflict_report)
                return Mode2Result(
                    trace_id=trace_id,
                    query=query,
                    stage=Mode2Stage.CONFLICT_DETECTION,
                    audit_entries=audit_entries,
                    escalate_to_mode1=True,
                    escalation_reason=(
                        f"Critical conflicts detected ({critical_conflicts} critical, "
                        f"{conflicts_detected} total). World model rebuild recommended."
                    ),
                    conflicts_detected=conflicts_detected,
                    critical_conflicts=critical_conflicts,
                    conflict_details=conflict_details,
                )

            # Resolve non-critical conflicts automatically
            if conflict_report.total > 0:
                actions = self.causal.resolve_conflicts(conflict_report)
                self.causal.apply_resolutions(actions, domain=domain)

            audit_entries.append(self._create_audit(
                trace_id, "conflict_detection_complete",
                {
                    "total": conflicts_detected,
                    "critical": critical_conflicts,
                }
            ))

            # Stage 4: Causal Reasoning
            self._current_stage = Mode2Stage.CAUSAL_REASONING
            insights = await self._perform_causal_reasoning(
                domain, parsed.intervention, parsed.target_outcome
            )
            
            audit_entries.append(self._create_audit(
                trace_id, "causal_reasoning_complete",
                {"insights_count": len(insights)}
            ))
            
            # Stage 5: Recommendation Synthesis
            self._current_stage = Mode2Stage.RECOMMENDATION_SYNTHESIS
            recommendation = await self._synthesize_recommendation(
                query, parsed, insights, evidence
            )
            
            self._current_stage = Mode2Stage.COMPLETE
            
            audit_entries.append(self._create_audit(
                trace_id, "recommendation_complete",
                {"confidence": recommendation.confidence.value}
            ))
            
            return Mode2Result(
                trace_id=trace_id,
                query=query,
                stage=Mode2Stage.COMPLETE,
                recommendation=recommendation,
                model_used=domain,
                evidence_count=len(evidence),
                causal_insights=insights,
                audit_entries=audit_entries,
                conflicts_detected=conflicts_detected,
                critical_conflicts=critical_conflicts,
                conflict_details=conflict_details,
                model_staleness=model_staleness,
                confidence_decay_applied=confidence_decay_applied,
            )
            
        except Exception as e:
            return Mode2Result(
                trace_id=trace_id,
                query=query,
                stage=self._current_stage,
                error=str(e),
                audit_entries=audit_entries,
            )
    
    async def _parse_query(
        self,
        query: str,
        available_domains: list[str] | None = None,
    ) -> ParsedQuery:
        """Parse the decision query using LangExtract.

        If *available_domains* is provided the prompt is augmented so the
        LLM picks a domain from the known list instead of hallucinating.
        """
        extracted = self.extraction.parse_query(
            query=query,
            available_domains=available_domains,
        )

        return ParsedQuery(
            domain=extracted.domain,
            intervention=extracted.intervention,
            target_outcome=extracted.target_outcome,
            constraints=extracted.constraints,
        )
    
    async def _refresh_evidence(
        self,
        parsed: ParsedQuery,
    ) -> list[EvidenceBundle]:
        """Refresh evidence using hybrid retrieval (Phase 2)."""
        # Phase 2: use hybrid retrieval for better recall
        query = f"{parsed.intervention} effect on {parsed.target_outcome}"
        request = RetrievalRequest(
            query=query,
            strategy=RetrievalStrategy.HYBRID,
            max_results=5,
            use_reranking=True,
        )
        evidence = await self.retrieval.retrieve(request)
        
        # Cache evidence
        for e in evidence:
            self._evidence_cache[e.content_hash[:12]] = e
        
        return evidence
    
    async def _perform_causal_reasoning(
        self,
        domain: str,
        intervention: str,
        target_outcome: str,
    ) -> list[CausalInsight]:
        """Perform causal reasoning on the world model."""
        insights = []
        
        try:
            # Try to find matching variables in the model
            summary = self.causal.get_model_summary(domain)
            variables = summary.get("variables", [])
            
            # Find best matching intervention and outcome variables
            intervention_var = self._find_matching_variable(intervention, variables)
            outcome_var = self._find_matching_variable(target_outcome, variables)
            
            if intervention_var and outcome_var:
                analysis = self.causal.analyze_relationship(
                    intervention_var, outcome_var, domain
                )
                
                insights.append(CausalInsight(
                    path_description=f"Analysis of {intervention_var} → {outcome_var}",
                    confounders=analysis.confounders,
                    mediators=analysis.mediators,
                    direct_effect=analysis.direct_effect,
                    strength="strong" if analysis.total_paths > 0 else "none",
                ))
        except ValueError:
            # Model or variables not found
            pass
        
        return insights
    
    async def _synthesize_recommendation(
        self,
        query: str,
        parsed: ParsedQuery,
        insights: list[CausalInsight],
        evidence: list[EvidenceBundle],
    ) -> DecisionRecommendation:
        """Synthesize recommendation using ExtractionService with grounded evidence."""
        # Build analysis context text
        paths_text = "No causal paths found in model"
        confounders_text = "None identified"
        mediators_text = "None identified"

        if insights:
            insight = insights[0]
            paths_text = insight.path_description
            if insight.confounders:
                confounders_text = ", ".join(insight.confounders)
            if insight.mediators:
                mediators_text = ", ".join(insight.mediators)

        evidence_text = "\n\n".join(
            e.content[:400] for e in evidence[:5]
        )

        full_text = (
            f"Query: {query}\n"
            f"Intervention: {parsed.intervention}\n"
            f"Target Outcome: {parsed.target_outcome}\n"
            f"Causal Paths: {paths_text}\n"
            f"Confounders: {confounders_text}\n"
            f"Mediators: {mediators_text}\n\n"
            f"Evidence:\n{evidence_text}"
        )

        # Build evidence map for citation validation
        evidence_map = {
            eb.content_hash[:12]: eb.content[:200]
            for eb in evidence
        }

        extracted = self.extraction.synthesize_recommendation(
            context_text=full_text,
            evidence_map=evidence_map,
        )

        confidence_map = {
            "high": ConfidenceLevel.HIGH,
            "medium": ConfidenceLevel.MEDIUM,
            "low": ConfidenceLevel.LOW,
        }

        # Map grounded evidence keys back to UUIDs
        grounded_ids: list[UUID] = []
        for key in extracted.grounded_evidence_keys:
            for eb in evidence:
                if eb.content_hash[:12] == key:
                    grounded_ids.append(eb.bundle_id)
                    break

        return DecisionRecommendation(
            recommendation=extracted.recommendation,
            confidence=confidence_map.get(
                extracted.confidence, ConfidenceLevel.MEDIUM
            ),
            expected_outcome=extracted.reasoning,
            suggested_actions=extracted.actions,
            risks=extracted.risks,
            evidence_refs=grounded_ids,
        )

    # ── Legacy regex extractors removed ─────────────────────────────
    # _extract_parsed_query and _extract_recommendation have been
    # replaced by LangExtract structured extraction (see _parse_query
    # and _synthesize_recommendation).  No regex-based JSON parsing remains.

    @staticmethod
    def _resolve_domain(parsed_domain: str, available_domains: list[str]) -> Optional[str]:
        """Fuzzy-match *parsed_domain* to the closest entry in *available_domains*.

        Tries, in order:
        1. Case-insensitive exact match
        2. Substring / overlap (e.g. 'egg production' ∩ 'hobby_farm')
        3. Word-overlap between the parsed domain and domain variable names
        4. If exactly one domain exists, return it as the default
        """
        target = parsed_domain.lower().replace("_", " ")

        # 1. Case-insensitive exact
        for d in available_domains:
            if d.lower() == target:
                return d

        # 2. Substring containment either way
        for d in available_domains:
            d_lower = d.lower().replace("_", " ")
            if d_lower in target or target in d_lower:
                return d

        # 3. Any shared word (tokens of 3+ chars)
        target_words = {w for w in target.split() if len(w) >= 3}
        for d in available_domains:
            d_words = {w for w in d.lower().replace("_", " ").split() if len(w) >= 3}
            if target_words & d_words:
                return d

        # 4. Sole domain fallback
        if len(available_domains) == 1:
            return available_domains[0]

        return None

    def _find_matching_variable(
        self,
        term: str,
        variables: list[str],
    ) -> Optional[str]:
        """Find a variable that best matches the given term."""
        term_lower = term.lower()
        
        # Exact match
        for var in variables:
            if var.lower() == term_lower:
                return var
        
        # Partial match
        for var in variables:
            if term_lower in var.lower() or var.lower() in term_lower:
                return var
        
        return None
    
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
    
    @property
    def current_stage(self) -> Mode2Stage:
        """Get current execution stage."""
        return self._current_stage
