"""
Verification Loop (VerificationAgent)

Multi-turn agentic loop that orchestrates per-edge verification:
    retrieve → judge → (refine query?) → re-retrieve → re-judge

Key safety mechanisms:
    - ``LoopState.attempted_queries`` dedup set prevents infinite loops
      when the judge suggests the same (or equivalent) query twice.
    - Hard ``max_iterations`` cap from ``VerificationConfig``.
    - All iterations are traced via ``SpanCollector`` for auditability.
    - Concurrent execution via ``asyncio.gather`` with the shared
      ``LLMClient`` semaphore controlling API rate.
"""

from __future__ import annotations

import asyncio
import logging
import re as _re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional
from uuid import UUID

from src.agent.llm_client import LLMClient, LLMModel
from src.config import VerificationConfig, get_verification_config
from src.models.causal import EdgeMetadata
from src.models.enums import EdgeStatus, EvidenceStrength
from src.models.evidence import EvidenceBundle
from src.retrieval.router import RetrievalRouter
from src.training.spans import SpanCollector, SpanStatus
from src.verification.grounding_retriever import GroundingRetriever, GroundingResult
from src.verification.judge import (
    AdversarialVerdict,
    SupportType,
    VerificationJudge,
    VerificationVerdict,
)

_logger = logging.getLogger(__name__)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Loop state tracker
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass
class LoopState:
    """Mutable state for a single edge's verification loop."""

    from_var: str
    to_var: str
    mechanism: str

    iteration: int = 0
    max_iterations: int = 3

    attempted_queries: set[str] = field(default_factory=set)
    verdicts: list[VerificationVerdict] = field(default_factory=list)
    adversarial_verdict: Optional[AdversarialVerdict] = None

    grounded: bool = False
    final_quote: Optional[str] = None
    final_confidence: float = 0.0
    rejection_reason: Optional[str] = None
    supporting_bundle: Optional[EvidenceBundle] = None

    @property
    def edge_label(self) -> str:
        return f"{self.from_var}→{self.to_var}"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Per-edge verification result
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass
class VerificationResult:
    """Outcome of the full verification loop for one edge."""

    from_var: str
    to_var: str
    mechanism: str

    grounded: bool
    edge_status: EdgeStatus
    confidence: float = 0.0
    supporting_quote: Optional[str] = None
    rejection_reason: Optional[str] = None
    supporting_bundle: Optional[EvidenceBundle] = None

    # Adversarial results (populated only for strong edges)
    alternative_explanations: list[str] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    conditions: list[str] = field(default_factory=list)

    iterations_used: int = 0
    verdicts: list[VerificationVerdict] = field(default_factory=list)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Query normalisation (for dedup)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_STOPWORDS = frozenset(
    "a an the of in on for to and or is are was were does do "
    "not no how what why which that this".split()
)


def _normalise_query(query: str) -> str:
    """Lowercase, strip stopwords, collapse whitespace for dedup."""
    tokens = _re.sub(r"[^a-z0-9\s]", "", query.lower()).split()
    return " ".join(t for t in tokens if t not in _STOPWORDS)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Verification Agent
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class VerificationAgent:
    """Orchestrates the Proposer-Retriever-Judge loop.

    Parameters
    ----------
    llm_client:
        Shared ``LLMClient`` (its internal semaphore gates concurrency).
    retrieval_router:
        Shared ``RetrievalRouter`` for evidence retrieval.
    span_collector:
        For tracing every iteration.
    config:
        ``VerificationConfig`` controlling thresholds, iterations, model.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        retrieval_router: RetrievalRouter,
        span_collector: Optional[SpanCollector] = None,
        config: Optional[VerificationConfig] = None,
    ) -> None:
        self.config = config or get_verification_config()
        self.llm = llm_client
        self.spans = span_collector or SpanCollector(enabled=True)

        judge_model = LLMModel(self.config.judge_model)
        self.judge = VerificationJudge(llm_client, judge_model=judge_model)
        self.retriever = GroundingRetriever(
            retrieval_router, top_k=self.config.retrieval_top_k,
        )

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    async def verify_edge(
        self,
        from_var: str,
        to_var: str,
        mechanism: str,
        evidence_strength: EvidenceStrength = EvidenceStrength.HYPOTHESIS,
        *,
        doc_ids: list[str] | None = None,
        trace_id: str | None = None,
    ) -> VerificationResult:
        """Run the full verification loop for a single edge.

        Steps:
            1. Retrieve evidence (hypothesis-aware or hybrid)
            2. Judge the evidence
            3. If judge suggests a refinement query → dedup-check → re-retrieve
            4. Repeat up to ``max_iterations``
            5. (Optional) Adversarial pass for strong edges

        Returns
        -------
        VerificationResult
            Full outcome including grounding status, quote, rejection reason.
        """
        state = LoopState(
            from_var=from_var,
            to_var=to_var,
            mechanism=mechanism,
            max_iterations=self.config.max_judge_iterations,
        )

        # Open a span for this edge
        edge_span = self.spans.start_span(
            f"verify_edge.{state.edge_label}",
            trace_id=trace_id,
            attributes={
                "from_var": from_var,
                "to_var": to_var,
                "mechanism": mechanism[:200],
                "max_iterations": state.max_iterations,
            },
        )

        try:
            await self._run_loop(state, doc_ids=doc_ids, trace_id=trace_id)

            # ── Adversarial pass for strong grounded edges ─────────────
            if (
                state.grounded
                and self.config.enable_adversarial_pass
                and evidence_strength in (EvidenceStrength.STRONG, EvidenceStrength.MODERATE)
            ):
                adv_span = self.spans.start_span(
                    f"adversarial.{state.edge_label}",
                    trace_id=trace_id,
                )
                try:
                    adv = await self.judge.evaluate_adversarial(
                        from_var=from_var,
                        to_var=to_var,
                        mechanism=mechanism,
                        supporting_quote=state.final_quote or "",
                    )
                    state.adversarial_verdict = adv

                    # If adversarial judge says spurious, downgrade
                    if not adv.still_grounded:
                        _logger.warning(
                            "Adversarial pass rejected %s: %s",
                            state.edge_label,
                            adv.alternative_explanations[:2],
                        )
                        state.grounded = False
                        state.rejection_reason = (
                            f"Adversarial rejection: {'; '.join(adv.alternative_explanations[:3])}"
                        )
                    self.spans.end_span(adv_span, SpanStatus.COMPLETED, {
                        "still_grounded": adv.still_grounded,
                        "alternatives": len(adv.alternative_explanations),
                    })
                except Exception as exc:
                    _logger.warning("Adversarial pass failed for %s: %s", state.edge_label, exc)
                    self.spans.end_span(adv_span, SpanStatus.FAILED, {"error": str(exc)})

            self.spans.end_span(edge_span, SpanStatus.COMPLETED, {
                "grounded": state.grounded,
                "iterations": state.iteration,
                "confidence": state.final_confidence,
            })

        except Exception as exc:
            _logger.error("Verification loop crashed for %s: %s", state.edge_label, exc)
            state.grounded = False
            state.rejection_reason = f"verification_error: {exc}"
            self.spans.end_span(edge_span, SpanStatus.FAILED, {"error": str(exc)})

        return self._build_result(state)

    async def verify_all_edges(
        self,
        edges: list[dict],
        *,
        doc_ids: list[str] | None = None,
        trace_id: str | None = None,
    ) -> list[VerificationResult]:
        """Verify a batch of edges concurrently.

        Parameters
        ----------
        edges:
            List of dicts with keys ``from_var``, ``to_var``,
            ``mechanism``, and optionally ``evidence_strength``.
        doc_ids:
            Document filter for retrieval scoping.
        trace_id:
            Parent trace ID for span hierarchy.

        Returns
        -------
        list[VerificationResult]
            One result per input edge, in the same order.
        """
        batch_span = self.spans.start_span(
            "verify_all_edges",
            trace_id=trace_id,
            attributes={"edge_count": len(edges)},
        )

        async def _verify_one(edge_dict: dict) -> VerificationResult:
            strength_str = edge_dict.get("evidence_strength", "hypothesis")
            try:
                strength = EvidenceStrength(strength_str)
            except ValueError:
                strength = EvidenceStrength.HYPOTHESIS

            return await self.verify_edge(
                from_var=edge_dict["from_var"],
                to_var=edge_dict["to_var"],
                mechanism=edge_dict.get("mechanism", ""),
                evidence_strength=strength,
                doc_ids=doc_ids,
                trace_id=trace_id,
            )

        results = await asyncio.gather(
            *[_verify_one(e) for e in edges],
            return_exceptions=True,
        )

        # Convert exceptions to rejection results
        final: list[VerificationResult] = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                _logger.error(
                    "Edge verification failed with exception: %s", result,
                )
                edge = edges[i]
                final.append(VerificationResult(
                    from_var=edge["from_var"],
                    to_var=edge["to_var"],
                    mechanism=edge.get("mechanism", ""),
                    grounded=False,
                    edge_status=EdgeStatus.REJECTED,
                    rejection_reason=f"exception: {result}",
                ))
            else:
                final.append(result)

        grounded_count = sum(1 for r in final if r.grounded)
        self.spans.end_span(batch_span, SpanStatus.COMPLETED, {
            "total": len(final),
            "grounded": grounded_count,
            "rejected": len(final) - grounded_count,
        })

        _logger.info(
            "Verification batch complete: %d/%d edges grounded",
            grounded_count, len(final),
        )
        return final

    # ------------------------------------------------------------------ #
    # Core loop
    # ------------------------------------------------------------------ #

    async def _run_loop(
        self,
        state: LoopState,
        *,
        doc_ids: list[str] | None = None,
        trace_id: str | None = None,
    ) -> None:
        """Execute the retrieve → judge → refine loop."""
        custom_query: str | None = None

        while state.iteration < state.max_iterations:
            state.iteration += 1
            iter_span = self.spans.start_span(
                f"verify_iter.{state.edge_label}.{state.iteration}",
                trace_id=trace_id,
                attributes={
                    "iteration": state.iteration,
                    "custom_query": custom_query,
                    "attempted_queries": list(state.attempted_queries),
                },
            )

            # ── Retrieve ───────────────────────────────────────────────
            if custom_query:
                grounding = await self.retriever.retrieve_with_query(
                    custom_query, doc_ids=doc_ids,
                )
            else:
                grounding = await self.retriever.ground_edge(
                    from_var=state.from_var,
                    to_var=state.to_var,
                    mechanism=state.mechanism,
                    doc_ids=doc_ids,
                )

            # Track the query used
            norm_query = _normalise_query(grounding.query_used)
            state.attempted_queries.add(norm_query)

            all_chunks = grounding.support_chunks + grounding.refute_chunks
            if not all_chunks:
                _logger.warning(
                    "No evidence retrieved for %s (iter %d)",
                    state.edge_label, state.iteration,
                )
                state.rejection_reason = "no_evidence_retrieved"
                self.spans.end_span(iter_span, SpanStatus.COMPLETED, {
                    "chunks_retrieved": 0,
                    "verdict": "no_evidence",
                })
                continue

            # ── Judge ──────────────────────────────────────────────────
            verdict = await self.judge.evaluate(
                from_var=state.from_var,
                to_var=state.to_var,
                mechanism=state.mechanism,
                evidence_chunks=all_chunks,
            )
            state.verdicts.append(verdict)

            self.spans.end_span(iter_span, SpanStatus.COMPLETED, {
                "chunks_retrieved": len(all_chunks),
                "is_grounded": verdict.is_grounded,
                "support_type": verdict.support_type.value,
                "confidence": verdict.confidence,
                "has_refinement": verdict.suggested_refinement_query is not None,
            })

            # ── Accept? ────────────────────────────────────────────────
            if verdict.is_grounded and verdict.confidence >= self.config.grounding_confidence_threshold:
                state.grounded = True
                state.final_quote = verdict.supporting_quote
                state.final_confidence = verdict.confidence

                # Pick the best supporting chunk for provenance
                if grounding.support_chunks:
                    state.supporting_bundle = grounding.support_chunks[0]

                _logger.info(
                    "Edge %s grounded on iteration %d (confidence=%.2f)",
                    state.edge_label, state.iteration, verdict.confidence,
                )
                return

            # ── Refine? ────────────────────────────────────────────────
            refinement = verdict.suggested_refinement_query
            if refinement:
                norm_refinement = _normalise_query(refinement)
                if norm_refinement in state.attempted_queries:
                    _logger.warning(
                        "Judge suggested duplicate query for %s: '%s' — "
                        "breaking loop to prevent infinite cycle",
                        state.edge_label, refinement,
                    )
                    state.rejection_reason = (
                        f"exhausted_refinements: judge repeated query '{refinement}'"
                    )
                    return
                custom_query = refinement
            else:
                # Judge rejected without suggesting refinement — done
                state.rejection_reason = (
                    verdict.rejection_reason or "judge_rejected_no_refinement"
                )
                return

        # Exhausted all iterations
        if not state.grounded:
            state.rejection_reason = (
                state.rejection_reason
                or f"max_iterations_reached ({state.max_iterations})"
            )

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _build_result(state: LoopState) -> VerificationResult:
        """Convert final ``LoopState`` into a ``VerificationResult``."""
        adv = state.adversarial_verdict
        return VerificationResult(
            from_var=state.from_var,
            to_var=state.to_var,
            mechanism=state.mechanism,
            grounded=state.grounded,
            edge_status=EdgeStatus.GROUNDED if state.grounded else EdgeStatus.REJECTED,
            confidence=state.final_confidence,
            supporting_quote=state.final_quote,
            rejection_reason=state.rejection_reason,
            supporting_bundle=state.supporting_bundle,
            alternative_explanations=adv.alternative_explanations if adv else [],
            assumptions=adv.assumptions_required if adv else [],
            conditions=adv.conditions if adv else [],
            iterations_used=state.iteration,
            verdicts=state.verdicts,
        )
