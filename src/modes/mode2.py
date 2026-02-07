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

from src.agent.llm_client import LLMClient
from src.causal.service import CausalService
from src.causal.path_finder import CausalAnalysis
from src.retrieval.router import RetrievalRouter
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
    EVIDENCE_REFRESH = "evidence_refresh"
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


class Mode2DecisionSupport:
    """
    Mode 2: Decision Support
    
    Provides causal reasoning-backed recommendations:
    - Query understanding
    - World model retrieval
    - Evidence-based reasoning
    - Confidence-scored recommendations
    """
    
    QUERY_PARSING_PROMPT = """Analyze this decision query and extract:
- domain: The decision domain (e.g., pricing, marketing, operations)
- intervention: The action being considered
- target_outcome: The outcome to optimize
- constraints: Any constraints mentioned

Query: {query}

Respond with JSON:
```json
{{"domain": "...", "intervention": "...", "target_outcome": "...", "constraints": ["..."]}}
```"""

    RECOMMENDATION_PROMPT = """Based on the causal analysis and evidence, provide a recommendation.

Query: {query}

Causal Analysis:
- Intervention: {intervention}
- Target Outcome: {outcome}
- Causal Paths: {paths}
- Confounders to consider: {confounders}
- Mediating variables: {mediators}

Supporting Evidence:
{evidence}

Provide your recommendation as JSON:
```json
{{
  "recommendation": "Your main recommendation",
  "confidence": "high/medium/low",
  "reasoning": "Why this recommendation",
  "actions": ["Specific action 1", "Specific action 2"],
  "risks": ["Risk 1", "Risk 2"],
  "evidence_refs": ["evidence_id_1", "evidence_id_2"]
}}
```"""

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        retrieval_router: Optional[RetrievalRouter] = None,
        causal_service: Optional[CausalService] = None,
        evidence_freshness_days: int = 30,
    ):
        self.llm = llm_client or LLMClient()
        self.retrieval = retrieval_router or RetrievalRouter()
        self.causal = causal_service or CausalService()
        self.evidence_freshness_days = evidence_freshness_days
        
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
            parsed = await self._parse_query(query)
            domain = domain_hint or parsed.domain
            
            audit_entries.append(self._create_audit(
                trace_id, "query_parsed",
                {"domain": domain, "intervention": parsed.intervention}
            ))
            
            # Stage 2: Model Retrieval
            self._current_stage = Mode2Stage.MODEL_RETRIEVAL
            
            # Check if we have a world model for this domain
            if domain not in self.causal.list_domains():
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
            )
            
        except Exception as e:
            return Mode2Result(
                trace_id=trace_id,
                query=query,
                stage=self._current_stage,
                error=str(e),
                audit_entries=audit_entries,
            )
    
    async def _parse_query(self, query: str) -> ParsedQuery:
        """Parse the decision query to extract components."""
        prompt = self.QUERY_PARSING_PROMPT.format(query=query)
        response = await self.llm.generate(prompt)
        
        parsed = self._extract_parsed_query(response.content)
        return parsed
    
    async def _refresh_evidence(
        self,
        parsed: ParsedQuery,
    ) -> list[EvidenceBundle]:
        """Refresh evidence for the query."""
        # Search for evidence about the intervention and outcome
        query = f"{parsed.intervention} effect on {parsed.target_outcome}"
        evidence = await self.retrieval.retrieve_simple(query, max_results=5)
        
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
        """Synthesize final recommendation using LLM."""
        # Format causal insights
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
        
        # Format evidence
        evidence_text = "\n".join([
            f"[{e.content_hash[:8]}] {e.content[:300]}"
            for e in evidence[:3]
        ])
        
        prompt = self.RECOMMENDATION_PROMPT.format(
            query=query,
            intervention=parsed.intervention,
            outcome=parsed.target_outcome,
            paths=paths_text,
            confounders=confounders_text,
            mediators=mediators_text,
            evidence=evidence_text,
        )
        
        response = await self.llm.generate(prompt)
        recommendation = self._extract_recommendation(response.content, evidence)
        
        return recommendation
    
    def _extract_parsed_query(self, content: str) -> ParsedQuery:
        """Extract parsed query from LLM response."""
        import json
        import re
        
        # Extract JSON from response
        json_match = re.search(r'\{[\s\S]*\}', content)
        if json_match:
            try:
                data = json.loads(json_match.group())
                return ParsedQuery(
                    domain=data.get("domain", "general"),
                    intervention=data.get("intervention", "unknown action"),
                    target_outcome=data.get("target_outcome", "unknown outcome"),
                    constraints=data.get("constraints", []),
                )
            except json.JSONDecodeError:
                pass
        
        # Default parsing
        return ParsedQuery(
            domain="general",
            intervention="the proposed action",
            target_outcome="business outcomes",
        )
    
    def _extract_recommendation(
        self,
        content: str,
        evidence: list[EvidenceBundle],
    ) -> DecisionRecommendation:
        """Extract recommendation from LLM response."""
        import json
        import re
        
        # Extract JSON from response
        json_match = re.search(r'\{[\s\S]*\}', content)
        if json_match:
            try:
                data = json.loads(json_match.group())
                
                confidence_map = {
                    "high": ConfidenceLevel.HIGH,
                    "medium": ConfidenceLevel.MEDIUM,
                    "low": ConfidenceLevel.LOW,
                }
                
                return DecisionRecommendation(
                    recommendation=data.get("recommendation", "Unable to provide recommendation"),
                    confidence=confidence_map.get(
                        data.get("confidence", "medium").lower(),
                        ConfidenceLevel.MEDIUM
                    ),
                    expected_outcome=data.get("expected_outcome", data.get("reasoning", "")),
                    suggested_actions=data.get("actions", []),
                    risks=data.get("risks", []),
                    evidence_refs=[],  # UUIDs from actual evidence bundles
                )
            except json.JSONDecodeError:
                pass
        
        return DecisionRecommendation(
            recommendation="Unable to parse recommendation from analysis",
            confidence=ConfidenceLevel.LOW,
            expected_outcome="Unknown",
            suggested_actions=[],
            risks=["Analysis may be incomplete"],
            evidence_refs=[],
        )
    
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
