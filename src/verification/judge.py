"""
Verification Judge

LLM-based judge that evaluates whether retrieved evidence actually
supports a proposed causal edge.  Uses Gemini's native structured
output (``response_schema``) to guarantee valid JSON verdicts.

Two judge modes:
    1. **Grounding judge** — "Does this evidence support the claim?"
    2. **Adversarial judge** — "Assume the claim is spurious; find
       alternative explanations."
"""

from __future__ import annotations

import logging
import textwrap
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

from src.agent.llm_client import LLMClient, LLMModel
from src.models.evidence import EvidenceBundle

_logger = logging.getLogger(__name__)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Verdict schemas (used as Gemini response_schema)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class SupportType(str, Enum):
    """How the evidence relates to the causal claim."""

    DIRECT_CAUSAL = "direct_causal"
    CORRELATION_ONLY = "correlation_only"
    IRRELEVANT = "irrelevant"


class VerificationVerdict(BaseModel):
    """Structured output from the grounding judge."""

    is_grounded: bool = Field(
        ...,
        description="True if the evidence explicitly supports a causal link",
    )
    support_type: SupportType = Field(
        ...,
        description="Classification of the relationship the evidence demonstrates",
    )
    supporting_quote: Optional[str] = Field(
        default=None,
        description="Exact verbatim quote from the evidence that supports the claim",
    )
    rejection_reason: Optional[str] = Field(
        default=None,
        description="Why the evidence does not support the causal claim",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Judge's confidence in this verdict (0-1)",
    )
    suggested_refinement_query: Optional[str] = Field(
        default=None,
        description=(
            "A better search query to find supporting evidence, "
            "or null if the judge does not believe better evidence exists"
        ),
    )


class AdversarialVerdict(BaseModel):
    """Structured output from the adversarial (devil's advocate) judge."""

    still_grounded: bool = Field(
        ...,
        description="True if the causal claim survives adversarial scrutiny",
    )
    alternative_explanations: list[str] = Field(
        default_factory=list,
        description="Plausible non-causal explanations for the observed evidence",
    )
    assumptions_required: list[str] = Field(
        default_factory=list,
        description="Assumptions that must hold for the causal claim to be valid",
    )
    conditions: list[str] = Field(
        default_factory=list,
        description="Boundary conditions under which the relationship holds",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Judge's confidence that the claim is NOT spurious",
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Prompt templates
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_GROUNDING_SYSTEM = textwrap.dedent("""\
    You are a strict verifier specializing in causal inference.
    Your task is to determine whether provided evidence TEXT supports
    a proposed causal relationship.

    Rules:
    - Accept DIRECT causal evidence (A causes B, A leads to B, A
      drives B, A results in B) — the evidence must explicitly mention
      or clearly imply a causal mechanism.
    - Also accept well-established domain knowledge stated in business
      plans, strategy documents, or industry analyses (e.g. "marketing
      increases customer acquisition" is a widely accepted causal claim
      even without a controlled experiment).
    - If the text only shows two variables co-occurring without any
      implied mechanism, classify as "correlation_only" and set
      is_grounded=false.
    - If the text is about unrelated topics, classify as "irrelevant"
      and set is_grounded=false.
    - When is_grounded=true, extract the EXACT verbatim quote (no
      paraphrasing) as supporting_quote.
    - When is_grounded=false and you believe better evidence might exist
      in the document corpus, suggest a refined search query in
      suggested_refinement_query.  If you do not believe better evidence
      exists, leave it null.
    - Be fair: accept claims when the evidence clearly supports a causal
      relationship, even if the evidence comes from a business plan
      rather than a scientific study.""")


_GROUNDING_TEMPLATE = textwrap.dedent("""\
    ## Proposed Causal Edge
    - **Cause variable:** {from_var}
    - **Effect variable:** {to_var}
    - **Proposed mechanism:** {mechanism}

    ## Retrieved Evidence Chunks
    {evidence_block}

    ## Task
    Does the evidence above explicitly support the claim that
    **{from_var}** causes **{to_var}** through the mechanism described?
    Evaluate carefully and respond with a structured verdict.""")


_ADVERSARIAL_SYSTEM = textwrap.dedent("""\
    You are a devil's advocate reviewer for causal claims.
    Assume the proposed relationship might be spurious and look for
    reasons why the evidence might be misleading.

    Consider:
    - Confounding variables that could explain the association
    - Reverse causation (B causes A instead)
    - Selection bias in the evidence
    - Measurement issues
    - Temporal ordering problems

    IMPORTANT NUANCE — Evidence types have different standards:
    - **Academic / empirical studies**: require rigorous causal evidence.
    - **Business plans, projections, strategy documents**: these express
      domain-expert causal beliefs grounded in industry knowledge.  A
      business plan stating "marketing drives customer acquisition" is a
      valid causal claim based on established business logic, even if it
      is forward-looking.  Do NOT reject claims merely because they come
      from a planning or strategy document.
    - **Mission statements and aspirational text**: these are weaker but
      still signal believed causal relationships.

    Be thorough but fair — if after scrutiny the claim genuinely holds
    as a reasonable causal belief supported by the evidence context, set
    still_grounded=true.""")


_ADVERSARIAL_TEMPLATE = textwrap.dedent("""\
    ## Proposed Causal Edge
    - **Cause variable:** {from_var}
    - **Effect variable:** {to_var}
    - **Proposed mechanism:** {mechanism}
    - **Supporting quote:** {supporting_quote}

    ## Task
    Assume this causal relationship is spurious.  What alternative
    explanations could account for the observed evidence?  What
    assumptions must hold for the causal claim to be valid?""")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Judge
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class VerificationJudge:
    """LLM-based judge for causal edge verification.

    Parameters
    ----------
    llm_client:
        Pre-initialised ``LLMClient`` (shares the global semaphore).
    judge_model:
        Which Gemini model to use for judging.  Defaults to
        ``gemini-2.5-pro`` for stronger reasoning.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        judge_model: LLMModel = LLMModel.GEMINI_PRO,
    ) -> None:
        self.llm = llm_client
        self.judge_model = judge_model

    async def evaluate(
        self,
        from_var: str,
        to_var: str,
        mechanism: str,
        evidence_chunks: list[EvidenceBundle],
    ) -> VerificationVerdict:
        """Run the grounding judge on an edge + evidence.

        Returns a ``VerificationVerdict`` with the judge's assessment.
        """
        evidence_block = self._format_evidence(evidence_chunks)

        prompt = _GROUNDING_TEMPLATE.format(
            from_var=from_var,
            to_var=to_var,
            mechanism=mechanism,
            evidence_block=evidence_block,
        )

        verdict = await self.llm.generate_structured_native(
            prompt=prompt,
            output_schema=VerificationVerdict,
            system_prompt=_GROUNDING_SYSTEM,
            model_override=self.judge_model,
        )

        _logger.info(
            "Judge verdict for %s→%s: grounded=%s  type=%s  confidence=%.2f",
            from_var, to_var, verdict.is_grounded,
            verdict.support_type.value, verdict.confidence,
        )
        return verdict

    async def evaluate_adversarial(
        self,
        from_var: str,
        to_var: str,
        mechanism: str,
        supporting_quote: str,
    ) -> AdversarialVerdict:
        """Run the adversarial (devil's advocate) judge.

        Only called for edges that passed the grounding judge with
        ``evidence_strength=strong``.
        """
        prompt = _ADVERSARIAL_TEMPLATE.format(
            from_var=from_var,
            to_var=to_var,
            mechanism=mechanism,
            supporting_quote=supporting_quote or "(no quote extracted)",
        )

        verdict = await self.llm.generate_structured_native(
            prompt=prompt,
            output_schema=AdversarialVerdict,
            system_prompt=_ADVERSARIAL_SYSTEM,
            model_override=self.judge_model,
        )

        _logger.info(
            "Adversarial verdict for %s→%s: still_grounded=%s  "
            "alternatives=%d  confidence=%.2f",
            from_var, to_var, verdict.still_grounded,
            len(verdict.alternative_explanations), verdict.confidence,
        )
        return verdict

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _format_evidence(chunks: list[EvidenceBundle]) -> str:
        """Format evidence chunks into a numbered block for the prompt."""
        if not chunks:
            return "(no evidence retrieved)"

        parts: list[str] = []
        for i, chunk in enumerate(chunks, 1):
            source = chunk.source.doc_title or chunk.source.doc_id
            loc_parts: list[str] = []
            if chunk.location.page_number:
                loc_parts.append(f"p.{chunk.location.page_number}")
            if chunk.location.section_name:
                loc_parts.append(chunk.location.section_name)
            loc = ", ".join(loc_parts) if loc_parts else "unknown location"

            parts.append(
                f"### Chunk {i} [{source} — {loc}]\n{chunk.content}\n"
            )

        return "\n".join(parts)
