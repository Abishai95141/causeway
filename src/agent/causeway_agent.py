"""
CausewayAgent — Unified Agentic Entrypoint

Single entrypoint where the LLM drives execution via tool calls,
replacing the fixed Mode 1 / Mode 2 pipelines.

Flow:
  1.  ModeRouter classifies the query → RouteDecision
  2.  Agent builds a mode-specific system prompt + tool set
  3.  AgentOrchestrator runs the LLM-driven loop (multi-step
      tool calling with span instrumentation)
  4.  Returns AgentResult with response, tool trace, and metadata

Phase 1 design (the "lever" approach):
  Mode 1 and Mode 2 are wrapped as single callable tools
  (run_world_model_construction, run_decision_support).
  The LLM can navigate documents via PageIndex tools before
  deciding which mode to invoke.

Phase 2 will decompose Mode 2 into finer tools (e.g.
check_staleness, trace_causal_path, detect_conflicts).
"""

from dataclasses import dataclass, field
from typing import Any, Optional

from src.agent.llm_client import LLMClient, ToolDefinition
from src.agent.orchestrator import AgentOrchestrator, OrchestratorResult
from src.causal.service import CausalService
from src.modes.mode1 import Mode1WorldModelConstruction
from src.modes.mode2 import Mode2DecisionSupport
from src.pageindex.client import PageIndexClient
from src.pageindex.pageindex_tools import create_pageindex_tools
from src.protocol.mode_router import ModeRouter, RouteDecision
from src.models.enums import OperatingMode


# ── System prompts ──────────────────────────────────────────────────

_SYSTEM_PROMPT_MODE1 = """\
You are Causeway, a causal-intelligence assistant in World Model \
Construction mode.

Your job:
  1. Navigate the uploaded documents using list_sections / read_section \
     to understand the domain.
  2. Once you have enough context, call run_world_model_construction \
     to build a causal DAG from the evidence.
  3. Summarise the result for the user.

Always ground claims in document evidence.  If evidence is insufficient, \
say so rather than guessing.\
"""

_SYSTEM_PROMPT_MODE2 = """\
You are Causeway, a causal-intelligence assistant in Decision Support \
mode.

Your job:
  1. Use search_evidence and PageIndex tools (list_sections, \
     read_section, search_document) to gather relevant context.
  2. BEFORE calling run_decision_support, you MUST call \
     list_available_domains to discover which world-model domains \
     already exist.  Use one of the returned domain strings as the \
     domain_hint parameter — do NOT guess or invent a domain name.
  3. Call run_decision_support with the user's question and the \
     exact domain_hint obtained in step 2.
  4. Present the recommendation, confidence level, and key causal \
     insights to the user.

Always cite evidence.  Flag when a world model is missing or stale.\
"""


# ── Result dataclass ────────────────────────────────────────────────

@dataclass
class AgentResult:
    """Result of a CausewayAgent run."""
    trace_id: str
    query: str
    routed_mode: str
    route_confidence: float
    route_reason: str
    response: str
    tool_calls: list[Any] = field(default_factory=list)
    total_tokens: int = 0
    latency_ms: int = 0
    escalate_to_mode1: bool = False
    escalation_reason: Optional[str] = None
    error: Optional[str] = None


# ── Agent ───────────────────────────────────────────────────────────

class CausewayAgent:
    """
    Unified agentic entrypoint for Causeway.

    Replaces the static Mode 1 / Mode 2 dispatch in the ``/query`` API
    route with an LLM-driven tool-calling loop.

    The orchestrator's ContextManager is shared across all tool calls
    within a single query, so insights from PageIndex reads are visible
    when the LLM later invokes run_decision_support.
    """

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        retrieval_router: Optional[Any] = None,
        causal_service: Optional[CausalService] = None,
        pageindex_client: Optional[PageIndexClient] = None,
        span_collector: Optional[Any] = None,
        mode_router: Optional[ModeRouter] = None,
        mode1: Optional[Mode1WorldModelConstruction] = None,
        mode2: Optional[Mode2DecisionSupport] = None,
        max_tool_calls: int = 10,
    ):
        self.llm = llm_client or LLMClient()
        # RetrievalRouter / SpanCollector may hang if external
        # services (Qdrant, Redis) are unreachable, so accept None
        # and only create them lazily when passed from the API layer.
        self.retrieval = retrieval_router
        self.causal = causal_service or CausalService()
        self.pageindex = pageindex_client or PageIndexClient()
        self.spans = span_collector
        self.router = mode_router or ModeRouter()

        # Mode implementations (Phase 1: wrapped as single tools)
        self._mode1 = mode1 or Mode1WorldModelConstruction()
        self._mode2 = mode2 or Mode2DecisionSupport()

        self._max_tool_calls = max_tool_calls
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize all sub-components."""
        if self._initialized:
            return
        await self.llm.initialize()
        if self.retrieval is not None:
            await self.retrieval.initialize()
        await self._mode1.initialize()
        await self._mode2.initialize()
        self._initialized = True

    # ── Public API ──────────────────────────────────────────────────

    async def run(
        self,
        query: str,
        session_id: Optional[str] = None,
    ) -> AgentResult:
        """
        Run the agentic loop for a user query.

        Steps:
          1. Route the query via ModeRouter.
          2. Build an AgentOrchestrator with mode-specific tools.
          3. Execute the LLM-driven tool loop.
          4. Package results into AgentResult.

        The orchestrator's ContextManager persists across tool calls,
        so PageIndex reads flow naturally into subsequent mode calls.
        """
        await self.initialize()

        # Step 1 — Route
        decision: RouteDecision = self.router.route(query)

        # Step 2 — Build orchestrator with appropriate tool set
        orchestrator = self._build_orchestrator(decision)

        # Step 3 — Choose system prompt
        system_prompt = (
            _SYSTEM_PROMPT_MODE1
            if decision.mode == OperatingMode.MODE_1
            else _SYSTEM_PROMPT_MODE2
        )

        # Step 4 — Run the LLM-driven loop
        orch_result: OrchestratorResult = await orchestrator.run(
            query=query,
            system_prompt=system_prompt,
        )

        # Step 5 — Package
        return AgentResult(
            trace_id=orch_result.trace_id,
            query=query,
            routed_mode=decision.mode.value,
            route_confidence=decision.confidence,
            route_reason=decision.reason.value,
            response=orch_result.response,
            tool_calls=orch_result.tool_calls,
            total_tokens=orch_result.total_tokens,
            latency_ms=orch_result.latency_ms,
            error=orch_result.error,
        )

    # ── Private helpers ─────────────────────────────────────────────

    def _build_orchestrator(
        self,
        decision: RouteDecision,
    ) -> AgentOrchestrator:
        """
        Create a fresh AgentOrchestrator with mode-appropriate tools.

        Common tools (always registered):
          - search_evidence  (built-in to orchestrator)
          - list_sections, read_section, search_document (PageIndex)

        Mode-specific tools:
          - Mode 1: run_world_model_construction
          - Mode 2: run_decision_support
        """
        orchestrator = AgentOrchestrator(
            llm_client=self.llm,
            causal_service=self.causal,
            max_tool_calls=self._max_tool_calls,
        )

        # Register PageIndex navigation tools
        for tool_def, handler in create_pageindex_tools(self.pageindex):
            orchestrator.register_tool(tool_def, handler)

        # Register list_available_domains (useful in all modes)
        domains_def = ToolDefinition(
            name="list_available_domains",
            description=(
                "List all world-model domain names that currently exist "
                "in the causal service.  Call this BEFORE "
                "run_decision_support so you can pass an exact "
                "domain_hint instead of guessing."
            ),
            parameters={
                "type": "object",
                "properties": {},
                "required": [],
            },
        )

        async def _handle_list_domains(**_kwargs: Any) -> dict[str, Any]:
            domains = self.causal.list_domains()
            return {"available_domains": domains}

        orchestrator.register_tool(domains_def, _handle_list_domains)

        # Register mode-specific tools
        if decision.mode == OperatingMode.MODE_1:
            self._register_mode1_tool(orchestrator, decision)
        else:
            self._register_mode2_tool(orchestrator, decision)

        return orchestrator

    def _register_mode1_tool(
        self,
        orchestrator: AgentOrchestrator,
        decision: RouteDecision,
    ) -> None:
        """Register Mode 1 as a callable tool."""
        mode1_def = ToolDefinition(
            name="run_world_model_construction",
            description=(
                "Build a causal world model (DAG) for a given domain. "
                "Uses evidence from indexed documents to discover "
                "variables and causal relationships."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "domain": {
                        "type": "string",
                        "description": (
                            "Decision domain (e.g. 'pricing', 'supply_chain')"
                        ),
                    },
                    "initial_query": {
                        "type": "string",
                        "description": "Starting query for evidence retrieval",
                    },
                    "max_variables": {
                        "type": "integer",
                        "description": "Maximum variables to discover (default 20)",
                    },
                },
                "required": ["domain", "initial_query"],
            },
        )

        async def _handle_mode1(
            domain: str,
            initial_query: str,
            max_variables: int = 20,
        ) -> dict[str, Any]:
            result = await self._mode1.run(
                domain=domain,
                initial_query=initial_query,
                max_variables=max_variables,
            )
            return {
                "trace_id": result.trace_id,
                "domain": result.domain,
                "stage": result.stage.value,
                "variables_discovered": result.variables_discovered,
                "edges_created": result.edges_created,
                "evidence_linked": result.evidence_linked,
                "requires_review": result.requires_review,
                "error": result.error,
            }

        orchestrator.register_tool(mode1_def, _handle_mode1)

    def _register_mode2_tool(
        self,
        orchestrator: AgentOrchestrator,
        decision: RouteDecision,
    ) -> None:
        """Register Mode 2 as a callable tool."""
        mode2_def = ToolDefinition(
            name="run_decision_support",
            description=(
                "Run causal decision support analysis. Parses the query, "
                "retrieves the relevant world model, refreshes evidence, "
                "performs causal reasoning, and synthesises a recommendation."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Decision question to analyse",
                    },
                    "domain_hint": {
                        "type": "string",
                        "description": (
                            "Optional domain hint to select the right "
                            "world model (e.g. 'pricing')"
                        ),
                    },
                },
                "required": ["query"],
            },
        )

        async def _handle_mode2(
            query: str,
            domain_hint: Optional[str] = None,
        ) -> dict[str, Any]:
            # Use domain from routing if not explicitly provided
            effective_hint = domain_hint or decision.extracted_domain
            result = await self._mode2.run(
                query=query,
                domain_hint=effective_hint,
            )

            rec_text = None
            conf_text = None
            if result.recommendation:
                rec_text = result.recommendation.recommendation
                conf_text = result.recommendation.confidence.value

            return {
                "trace_id": result.trace_id,
                "query": result.query,
                "stage": result.stage.value,
                "recommendation": rec_text,
                "confidence": conf_text,
                "model_used": result.model_used,
                "evidence_count": result.evidence_count,
                "escalate_to_mode1": result.escalate_to_mode1,
                "escalation_reason": result.escalation_reason,
                "error": result.error,
            }

        orchestrator.register_tool(mode2_def, _handle_mode2)

    # ── Introspection ───────────────────────────────────────────────

    @property
    def is_initialized(self) -> bool:
        """Check if the agent has been initialized."""
        return self._initialized
