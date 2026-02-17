"""
Agent Orchestrator

High-level orchestrator for LLM-driven operations:
- Tool dispatch
- Multi-step reasoning
- Error handling
- Result formatting
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
import logging
from typing import Any, Callable, Optional
from uuid import uuid4

from src.agent.llm_client import LLMClient, LLMResponse, ToolDefinition
from src.agent.context_manager import ContextManager, MessageRole
from src.retrieval.router import RetrievalRouter, RetrievalRequest
from src.causal.service import CausalService
from src.training.spans import SpanCollector, SpanStatus
from src.utils.text import truncate_evidence, truncate_for_context_tracking


@dataclass
class ToolResult:
    """Result of a tool execution."""
    tool_name: str
    success: bool
    result: Any
    error: Optional[str] = None
    execution_time_ms: int = 0


@dataclass
class OrchestratorResult:
    """Result of an orchestrator run."""
    trace_id: str
    query: str
    response: str
    tool_calls: list[ToolResult] = field(default_factory=list)
    total_tokens: int = 0
    latency_ms: int = 0
    error: Optional[str] = None


class AgentOrchestrator:
    """
    Orchestrates LLM interactions with tools.
    
    Features:
    - Multi-step tool calling
    - Evidence retrieval integration
    - Causal reasoning integration
    - Error handling and recovery
    """

    _log = logging.getLogger("src.agent.orchestrator")
    
    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        retrieval_router: Optional[RetrievalRouter] = None,
        causal_service: Optional[CausalService] = None,
        span_collector: Optional[SpanCollector] = None,
        max_tool_calls: int = 5,
    ):
        self.llm = llm_client or LLMClient()
        self.retrieval = retrieval_router or RetrievalRouter()
        self.causal = causal_service or CausalService()
        self.spans = span_collector or SpanCollector(enabled=True)
        self.max_tool_calls = max_tool_calls

        # Lazy import to avoid circular: orchestrator → verification → judge → llm_client → agent.__init__ → orchestrator
        from src.verification.loop import VerificationAgent
        self.verifier = VerificationAgent(
            llm_client=self.llm,
            retrieval_router=self.retrieval,
            span_collector=self.spans,
        )
        
        self._context = ContextManager()
        self._tools = self._register_default_tools()
        self._tool_handlers: dict[str, Callable] = {}
        
        self._setup_tool_handlers()
    
    async def initialize(self) -> None:
        """Initialize all components."""
        await self.llm.initialize()
        await self.retrieval.initialize()
    
    async def run(
        self,
        query: str,
        system_prompt: Optional[str] = None,
    ) -> OrchestratorResult:
        """
        Run the orchestrator with a query.
        Instruments every LLM call and tool execution as spans.
        """
        trace_id = f"orch_{uuid4().hex[:12]}"
        start_time = datetime.now(timezone.utc)
        
        # Start a trace for this run
        span_trace_id = self.spans.start_trace(name="orchestrator.run")
        root_span = self.spans._span_stack[-1] if self.spans._span_stack else ""
        self.spans.add_event(root_span, "query_received", {"query": query[:200]})
        
        # Set up context
        self._context.clear()
        if system_prompt:
            self._context.set_system_prompt(system_prompt)
        self._context.add_user_message(query)
        
        tool_results = []
        total_tokens = 0
        current_response = ""
        
        try:
            for iteration in range(self.max_tool_calls + 1):
                # Span: LLM call
                llm_span = self.spans.start_span(
                    "llm.generate",
                    trace_id=span_trace_id,
                    attributes={"iteration": iteration},
                )

                response = await self.llm.generate_with_tools(
                    prompt=self._context.build_prompt(),
                    tools=self._tools,
                    system_prompt=system_prompt,
                )

                self.spans.end_span(llm_span, SpanStatus.COMPLETED, {
                    "tokens": response.total_tokens,
                    "has_tool_calls": bool(response.tool_calls),
                })

                total_tokens += response.total_tokens
                current_response = response.content
                
                # Check for tool calls
                if response.tool_calls and iteration < self.max_tool_calls:
                    for tool_call in response.tool_calls:
                        tc_name = tool_call.get("tool", "")
                        tc_args = tool_call.get("arguments", {})
                        self._log.info(
                            "Tool call [iter=%d]: %s  args=%s",
                            iteration, tc_name, str(tc_args)[:300],
                        )
                        result = await self._execute_tool(
                            tc_name,
                            tc_args,
                            trace_id=span_trace_id,
                        )
                        self._log.info(
                            "Tool result [%s]: success=%s  result=%s",
                            result.tool_name, result.success,
                            str(result.result)[:300],
                        )
                        tool_results.append(result)
                        
                        # Add tool result to context
                        self._context.add_assistant_message(
                            f"Calling tool: {result.tool_name}"
                        )
                        self._context.add_tool_result(
                            result.tool_name,
                            str(result.result) if result.success else f"Error: {result.error}",
                        )
                else:
                    # No more tool calls, we have the final response
                    break
            
            latency = int((datetime.now(timezone.utc) - start_time).total_seconds() * 1000)

            self.spans.end_span(root_span, SpanStatus.COMPLETED, {
                "total_tokens": total_tokens,
                "tool_calls": len(tool_results),
                "latency_ms": latency,
            })

            return OrchestratorResult(
                trace_id=trace_id,
                query=query,
                response=current_response,
                tool_calls=tool_results,
                total_tokens=total_tokens,
                latency_ms=latency,
            )
            
        except Exception as e:
            latency = int((datetime.now(timezone.utc) - start_time).total_seconds() * 1000)
            self.spans.end_span(root_span, SpanStatus.FAILED, {"error": str(e)})
            return OrchestratorResult(
                trace_id=trace_id,
                query=query,
                response="",
                tool_calls=tool_results,
                total_tokens=total_tokens,
                latency_ms=latency,
                error=str(e),
            )
    
    async def _execute_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        trace_id: Optional[str] = None,
    ) -> ToolResult:
        """Execute a tool by name, recording a span."""
        start_time = datetime.now(timezone.utc)

        # Create a span for this tool execution
        tool_span = self.spans.start_span(
            f"tool.{tool_name}",
            trace_id=trace_id,
            attributes={"tool": tool_name, "arguments": str(arguments)[:200]},
        )

        handler = self._tool_handlers.get(tool_name)
        if not handler:
            self.spans.end_span(tool_span, SpanStatus.FAILED, {"error": "unknown_tool"})
            return ToolResult(
                tool_name=tool_name,
                success=False,
                result=None,
                error=f"Unknown tool: {tool_name}",
            )
        
        try:
            result = await handler(**arguments)
            execution_time = int((datetime.now(timezone.utc) - start_time).total_seconds() * 1000)
            self.spans.end_span(tool_span, SpanStatus.COMPLETED, {
                "execution_time_ms": execution_time,
            })
            return ToolResult(
                tool_name=tool_name,
                success=True,
                result=result,
                execution_time_ms=execution_time,
            )
        except Exception as e:
            execution_time = int((datetime.now(timezone.utc) - start_time).total_seconds() * 1000)
            self.spans.end_span(tool_span, SpanStatus.FAILED, {"error": str(e)})
            return ToolResult(
                tool_name=tool_name,
                success=False,
                result=None,
                error=str(e),
                execution_time_ms=execution_time,
            )
    
    def _register_default_tools(self) -> list[ToolDefinition]:
        """Register default available tools."""
        return [
            ToolDefinition(
                name="search_evidence",
                description="Search for relevant evidence from documents",
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query",
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "Maximum results (default 5)",
                        },
                    },
                    "required": ["query"],
                },
            ),
            ToolDefinition(
                name="analyze_causal_path",
                description="Analyze causal relationship between two variables",
                parameters={
                    "type": "object",
                    "properties": {
                        "cause": {
                            "type": "string",
                            "description": "The cause variable",
                        },
                        "effect": {
                            "type": "string",
                            "description": "The effect variable",
                        },
                        "domain": {
                            "type": "string",
                            "description": "Domain of the world model",
                        },
                    },
                    "required": ["cause", "effect"],
                },
            ),
            ToolDefinition(
                name="get_model_summary",
                description="Get summary of a causal world model",
                parameters={
                    "type": "object",
                    "properties": {
                        "domain": {
                            "type": "string",
                            "description": "Domain of the world model",
                        },
                    },
                    "required": ["domain"],
                },
            ),
            ToolDefinition(
                name="find_confounders",
                description="Find confounding variables between cause and effect",
                parameters={
                    "type": "object",
                    "properties": {
                        "cause": {
                            "type": "string",
                            "description": "The cause variable",
                        },
                        "effect": {
                            "type": "string",
                            "description": "The effect variable",
                        },
                    },
                    "required": ["cause", "effect"],
                },
            ),
            ToolDefinition(
                name="verify_edge",
                description=(
                    "Verify whether a proposed causal edge is grounded in evidence. "
                    "Runs a multi-turn retrieve-judge loop that checks if the "
                    "evidence corpus explicitly supports the causal claim."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "cause": {
                            "type": "string",
                            "description": "The cause variable ID or name",
                        },
                        "effect": {
                            "type": "string",
                            "description": "The effect variable ID or name",
                        },
                        "mechanism": {
                            "type": "string",
                            "description": "Description of the causal mechanism",
                        },
                    },
                    "required": ["cause", "effect"],
                },
            ),
        ]
    
    def _setup_tool_handlers(self) -> None:
        """Set up tool execution handlers."""
        self._tool_handlers = {
            "search_evidence": self._handle_search_evidence,
            "analyze_causal_path": self._handle_analyze_causal_path,
            "get_model_summary": self._handle_get_model_summary,
            "find_confounders": self._handle_find_confounders,
            "verify_edge": self._handle_verify_edge,
        }
    
    async def _handle_search_evidence(
        self,
        query: str,
        max_results: int = 5,
    ) -> dict[str, Any]:
        """Handle evidence search tool call."""
        bundles = await self.retrieval.retrieve_simple(query, max_results)
        
        results = []
        for bundle in bundles:
            results.append({
                "content": truncate_evidence(bundle.content, max_chars=800),
                "source": bundle.source.doc_title,
                "location": {
                    "page": bundle.location.page_number,
                    "section": bundle.location.section_name,
                },
            })
            
            # Track evidence in context
            self._context.add_evidence(
                bundle.content_hash[:8],
                truncate_for_context_tracking(bundle.content, max_chars=200),
            )
        
        return {"results": results, "count": len(results)}
    
    async def _handle_analyze_causal_path(
        self,
        cause: str,
        effect: str,
        domain: Optional[str] = None,
    ) -> dict[str, Any]:
        """Handle causal path analysis tool call."""
        try:
            analysis = self.causal.analyze_relationship(cause, effect, domain)
            return {
                "cause": analysis.cause,
                "effect": analysis.effect,
                "direct_effect": analysis.direct_effect,
                "total_paths": analysis.total_paths,
                "confounders": analysis.confounders,
                "mediators": analysis.mediators,
                "paths": [{"path": p.path, "length": p.length} for p in analysis.paths[:5]],
            }
        except ValueError as e:
            return {"error": str(e)}
    
    async def _handle_get_model_summary(
        self,
        domain: str,
    ) -> dict[str, Any]:
        """Handle model summary tool call."""
        try:
            return self.causal.get_model_summary(domain)
        except ValueError as e:
            return {"error": str(e)}
    
    async def _handle_find_confounders(
        self,
        cause: str,
        effect: str,
    ) -> dict[str, Any]:
        """Handle confounder finding tool call."""
        try:
            confounders = self.causal.identify_confounders(cause, effect)
            return {
                "cause": cause,
                "effect": effect,
                "confounders": confounders,
                "count": len(confounders),
            }
        except ValueError as e:
            return {"error": str(e)}

    async def _handle_verify_edge(
        self,
        cause: str,
        effect: str,
        mechanism: str = "",
    ) -> dict[str, Any]:
        """Handle verify_edge tool call — runs the agentic verification loop."""
        try:
            result = await self.verifier.verify_edge(
                from_var=cause,
                to_var=effect,
                mechanism=mechanism,
            )
            response: dict[str, Any] = {
                "cause": cause,
                "effect": effect,
                "grounded": result.grounded,
                "edge_status": result.edge_status.value,
                "confidence": result.confidence,
                "iterations_used": result.iterations_used,
            }
            if result.supporting_quote:
                response["supporting_quote"] = result.supporting_quote
            if result.rejection_reason:
                response["rejection_reason"] = result.rejection_reason
            if result.alternative_explanations:
                response["alternative_explanations"] = result.alternative_explanations
            if result.assumptions:
                response["assumptions"] = result.assumptions
            return response
        except Exception as e:
            return {"error": str(e)}
    
    def register_tool(
        self,
        definition: ToolDefinition,
        handler: Callable,
    ) -> None:
        """Register a custom tool."""
        self._tools.append(definition)
        self._tool_handlers[definition.name] = handler
    
    @property
    def context(self) -> ContextManager:
        """Get the context manager."""
        return self._context
