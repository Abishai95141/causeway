"""
Grounding Retriever

Wraps the existing retrieval stack to build targeted support and
refutation queries from a drafted edge's mechanism description.

Returns top-K evidence chunks per direction for the verification
judge to evaluate.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from src.models.evidence import EvidenceBundle
from src.retrieval.router import RetrievalRouter, RetrievalRequest, RetrievalStrategy
from src.haystack_svc.service import EvidenceAssessment

_logger = logging.getLogger(__name__)


@dataclass
class GroundingResult:
    """Evidence chunks retrieved for a single edge."""

    support_chunks: list[EvidenceBundle] = field(default_factory=list)
    refute_chunks: list[EvidenceBundle] = field(default_factory=list)
    query_used: str = ""


class GroundingRetriever:
    """Retrieves targeted evidence for verifying a causal edge.

    Delegates to :class:`RetrievalRouter` using hypothesis-aware
    retrieval when available, falling back to plain hybrid search.

    Parameters
    ----------
    retrieval_router:
        Pre-initialised ``RetrievalRouter`` instance.
    top_k:
        Maximum evidence chunks per direction (support / refute).
    """

    def __init__(
        self,
        retrieval_router: RetrievalRouter,
        top_k: int = 5,
    ) -> None:
        self.router = retrieval_router
        self.top_k = top_k

    async def ground_edge(
        self,
        from_var: str,
        to_var: str,
        mechanism: str,
        *,
        custom_query: str | None = None,
        doc_ids: list[str] | None = None,
    ) -> GroundingResult:
        """Retrieve evidence for a single drafted edge.

        Parameters
        ----------
        from_var:
            Source variable name / ID.
        to_var:
            Target variable name / ID.
        mechanism:
            Description of the causal mechanism linking the variables.
        custom_query:
            If provided, overrides the auto-generated search query
            (used when the judge suggests a refinement).
        doc_ids:
            Optional document filter for scoped retrieval.

        Returns
        -------
        GroundingResult
            Support and refutation chunks plus the query that was used.
        """
        query = custom_query or self._build_query(from_var, to_var, mechanism)

        try:
            assessment: EvidenceAssessment = await self.router.retrieve_for_hypothesis(
                cause=from_var,
                effect=to_var,
                mechanism=mechanism if not custom_query else custom_query,
                top_k=self.top_k,
            )
            return GroundingResult(
                support_chunks=assessment.supporting,
                refute_chunks=assessment.contradicting,
                query_used=query,
            )
        except Exception as exc:
            _logger.warning(
                "Hypothesis-aware retrieval failed for %s→%s, "
                "falling back to plain hybrid: %s",
                from_var, to_var, exc,
            )
            return await self._fallback_hybrid(query, doc_ids=doc_ids)

    async def retrieve_with_query(
        self,
        query: str,
        *,
        doc_ids: list[str] | None = None,
    ) -> GroundingResult:
        """Retrieve evidence using an explicit query string.

        Used when the judge suggests a ``refinement_query``.
        """
        return await self._fallback_hybrid(query, doc_ids=doc_ids)

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _build_query(from_var: str, to_var: str, mechanism: str) -> str:
        """Build a natural-language search query from edge metadata."""
        readable_from = from_var.replace("_", " ")
        readable_to = to_var.replace("_", " ")
        base = f"{readable_from} causes {readable_to}"
        if mechanism:
            base += f" through {mechanism}"
        return base

    async def _fallback_hybrid(
        self,
        query: str,
        *,
        doc_ids: list[str] | None = None,
    ) -> GroundingResult:
        """Plain hybrid retrieval without hypothesis awareness."""
        request = RetrievalRequest(
            query=query,
            strategy=RetrievalStrategy.HYBRID,
            max_results=self.top_k,
            use_reranking=True,
            doc_ids=doc_ids,
        )
        bundles = await self.router.retrieve(request)
        return GroundingResult(
            support_chunks=bundles,
            refute_chunks=[],
            query_used=query,
        )
