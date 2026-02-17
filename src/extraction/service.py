"""
Extraction Service — centralised LangExtract wrapper.

Consolidates all structured-extraction calls behind a single service
so Mode 1 and Mode 2 share prompts, dedup logic, CoT separation,
and citation-validation code.

Data-quality fixes baked in:
    ► Variable dedup   – prompt instructs LLM to merge synonyms;
                         post-processing collapses near-duplicates.
    ► CoT separation   – every extraction schema includes a dedicated
                         ``chain_of_thought`` field so reasoning never
                         leaks into ``mechanism`` / ``description``.
    ► Citation check   – ``_validate_citation`` confirms that the
                         extraction_text actually appears (fuzzy) in
                         the source evidence before we trust the ref.
"""

from __future__ import annotations

import logging
import os
import textwrap
from dataclasses import dataclass, field
from typing import Any, Optional

import langextract as lx
from pydantic import BaseModel, Field

from src.config import get_settings
from src.utils.text import canonicalize_var_id

_logger = logging.getLogger(__name__)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Result dataclasses (service-level; mode layers map to their own types)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass
class ExtractedVariable:
    """A single variable returned by ``extract_variables``."""
    name: str
    description: str
    var_type: str           # continuous | discrete | binary | categorical
    measurement_status: str  # measured | observable | latent
    chain_of_thought: str = ""
    grounded_evidence_keys: list[str] = field(default_factory=list)


@dataclass
class ExtractedEdge:
    """A single causal edge returned by ``extract_edges``."""
    from_var: str
    to_var: str
    mechanism: str
    strength: str           # strong | moderate | hypothesis
    chain_of_thought: str = ""
    extraction_text: str = ""
    grounded_evidence_keys: list[str] = field(default_factory=list)


@dataclass
class ExtractedQuery:
    """Parsed decision-query components from ``parse_query``."""
    domain: str
    intervention: str
    target_outcome: str
    constraints: list[str] = field(default_factory=list)


@dataclass
class ExtractedRecommendation:
    """Recommendation claim from ``synthesize_recommendation``."""
    recommendation: str
    confidence: str         # high | medium | low
    reasoning: str = ""
    actions: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    extraction_text: str = ""
    grounded_evidence_keys: list[str] = field(default_factory=list)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Pydantic schemas for LLM canonicalization (structured output)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class CanonicalVariable(BaseModel):
    """One consolidated variable produced by the canonicalization LLM."""

    canonical_name: str = Field(
        ...,
        description=(
            "The single best name for this concept — short, precise, "
            "and general enough to cover all merged source names."
        ),
    )
    description: str = Field(
        ...,
        description="One-sentence factual definition of this variable.",
    )
    var_type: str = Field(
        ...,
        description="One of: continuous, discrete, binary, categorical.",
    )
    measurement_status: str = Field(
        ...,
        description="One of: measured, observable, latent.",
    )
    merged_from: list[str] = Field(
        default_factory=list,
        description=(
            "Original variable names that were merged into this one. "
            "Include ALL source names, even if there is only one."
        ),
    )
    merge_reasoning: str = Field(
        default="",
        description="Brief explanation of why these names are the same concept.",
    )


class CanonicalVariableList(BaseModel):
    """Top-level response schema for the variable canonicalization call."""

    variables: list[CanonicalVariable] = Field(
        ...,
        description=(
            "The deduplicated list of causal variables.  Semantic "
            "duplicates must be merged into a single entry."
        ),
    )


class SynthesizedMechanism(BaseModel):
    """Structured output for LLM-synthesized causal mechanism text.

    Forces the LLM to separate its own reasoning from the verbatim
    evidence, eliminating the copy-paste anti-pattern where raw
    evidence snippets were dumped directly into the mechanism field.
    """

    rationale_in_own_words: str = Field(
        ...,
        description=(
            "Explain HOW the cause variable produces the effect in 1–2 "
            "sentences using YOUR OWN words.  Describe the causal "
            "pathway or mechanism — not what the document says, but "
            "what is actually happening in the real world.  NEVER "
            "copy-paste text from the evidence."
        ),
    )
    exact_quote: str = Field(
        ...,
        description=(
            "The single most relevant sentence (or clause) from the "
            "evidence that best supports this causal link.  Copy it "
            "EXACTLY as written — do not paraphrase, truncate, or "
            "edit.  If no suitable quote exists, write 'N/A'."
        ),
    )
    source_citation: str = Field(
        default="",
        description=(
            "Document title and section from which the exact_quote "
            "was taken (e.g. 'Business_Plan.pdf, Section 3.2').  "
            "Leave empty if the quote is 'N/A'."
        ),
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Prompts — single source of truth
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_VARIABLE_PROMPT = textwrap.dedent("""\
    Extract every measurable causal variable mentioned or implied in the
    evidence text.  A causal model typically atleast needs 8-15 variables.
    Think across categories: inputs, processes, outputs, quality,
    demand, satisfaction, competition, and environment.
    Use exact phrases from the text for extraction_text.  Do not
    paraphrase or overlap entities.

    DEDUP RULE: If two phrases refer to the same real-world concept
    (e.g. "local residents" vs "local area population"), merge them
    into ONE canonical variable and pick the most precise phrase.

    You MUST populate the ``chain_of_thought`` attribute with your
    internal reasoning about why this is a distinct, measurable
    variable.  Do NOT put reasoning into the ``description`` field —
    description should be a short factual definition only.""")

_VARIABLE_EXAMPLES = [
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
                    "chain_of_thought": (
                        "Mentioned as a direct input — 'training hours' — "
                        "and it is measurable in hours."
                    ),
                },
            ),
            lx.data.Extraction(
                extraction_class="variable",
                extraction_text="barista skill levels",
                attributes={
                    "type": "continuous",
                    "measurement_status": "observable",
                    "description": "Assessed proficiency of barista staff",
                    "chain_of_thought": (
                        "Outcome of training; distinct from training hours "
                        "because it measures proficiency, not duration."
                    ),
                },
            ),
            lx.data.Extraction(
                extraction_class="variable",
                extraction_text="Seasonal foot traffic",
                attributes={
                    "type": "continuous",
                    "measurement_status": "measured",
                    "description": "Pedestrian volume influenced by season",
                    "chain_of_thought": (
                        "External demand driver; measurable via footfall "
                        "counters."
                    ),
                },
            ),
            lx.data.Extraction(
                extraction_class="variable",
                extraction_text="daily customer volume",
                attributes={
                    "type": "continuous",
                    "measurement_status": "measured",
                    "description": "Number of customers visiting per day",
                    "chain_of_thought": (
                        "Downstream from foot traffic; measured at POS."
                    ),
                },
            ),
            lx.data.Extraction(
                extraction_class="variable",
                extraction_text="Specialty drink pricing",
                attributes={
                    "type": "continuous",
                    "measurement_status": "measured",
                    "description": "Price point for specialty beverages",
                    "chain_of_thought": (
                        "Controllable lever; distinct from competitor pricing."
                    ),
                },
            ),
            lx.data.Extraction(
                extraction_class="variable",
                extraction_text="competitor averages",
                attributes={
                    "type": "continuous",
                    "measurement_status": "observable",
                    "description": "Average competitor pricing in the area",
                    "chain_of_thought": (
                        "External benchmark; observable but not under our "
                        "control."
                    ),
                },
            ),
        ],
    )
]

_EDGE_PROMPT_TEMPLATE = textwrap.dedent("""\
    Extract every causal relationship between variables found in
    the evidence text.  For each relationship use exact sentences
    or phrases from the text as extraction_text — this is the
    proof that the relationship exists.

    CRITICAL CONSTRAINT: You MUST ONLY use variable IDs from
    this list: {allowed_vars}
    Do NOT invent new variable names.  The from_var and to_var
    attributes MUST exactly match one of the IDs listed above.

    Attributes must include from_var and to_var (using the exact
    IDs above), the causal mechanism, and evidence strength
    (strong / moderate / hypothesis).

    You MUST populate ``chain_of_thought`` with your reasoning
    about why this causal link exists and how strong the evidence
    is.  Do NOT put reasoning into ``mechanism`` — mechanism
    should be a short factual description of the causal pathway.

    Consider ALL possible variable pairs; a useful model has
    many edges.""")

_QUERY_PROMPT = textwrap.dedent("""\
    Extract the decision query components from the text: the business
    domain, the proposed intervention (action being considered), the
    target outcome to optimise, and any constraints mentioned.
    Use exact phrases from the text.""")

_QUERY_EXAMPLES = [
    lx.data.ExampleData(
        text="Should we increase prices to boost revenue while staying within budget?",
        extractions=[
            lx.data.Extraction(
                extraction_class="decision_query",
                extraction_text="increase prices to boost revenue",
                attributes={
                    "domain": "pricing",
                    "intervention": "increase prices",
                    "target_outcome": "revenue",
                    "constraints": "staying within budget",
                },
            ),
        ],
    ),
]

_RECOMMENDATION_PROMPT = textwrap.dedent("""\
    Extract recommendation claims from the causal analysis and evidence.
    Each claim should be grounded in exact text from the evidence.

    You MUST populate ``chain_of_thought`` with your reasoning about
    why this recommendation follows from the causal analysis.  Do NOT
    put reasoning into ``reasoning`` — that field should be a concise
    one-sentence summary of the causal mechanism, not internal thought.

    Attributes must include the recommendation, confidence level,
    reasoning, suggested actions, and risks.""")

_RECOMMENDATION_EXAMPLES = [
    lx.data.ExampleData(
        text=(
            "Analysis shows price increases reduce demand through elasticity. "
            "Revenue impact is positive when demand drop is below 15%. "
            "Risk: competitor under-cutting may negate gains."
        ),
        extractions=[
            lx.data.Extraction(
                extraction_class="recommendation_claim",
                extraction_text="price increases reduce demand through elasticity",
                attributes={
                    "recommendation": "Implement moderate price increase of 5-10%",
                    "confidence": "medium",
                    "reasoning": "Elasticity effect is present but manageable",
                    "actions": "Raise prices gradually, monitor demand weekly",
                    "risks": "Competitor under-cutting may negate gains",
                    "chain_of_thought": (
                        "The evidence states demand only drops significantly "
                        "above 15% — a 5-10% hike stays under that threshold, "
                        "so net revenue should rise."
                    ),
                },
            ),
        ],
    ),
]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Service
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class ExtractionService:
    """Centralised LangExtract wrapper used by Mode 1 and Mode 2.

    Parameters
    ----------
    model_id : str
        LLM model identifier for LangExtract.
    api_key : str | None
        Google AI API key.  Falls back to
        ``get_settings().google_ai_api_key`` then to the
        ``LANGEXTRACT_API_KEY`` environment variable.
    """

    def __init__(
        self,
        model_id: str = "gemini-2.5-flash",
        api_key: str | None = None,
    ) -> None:
        self.model_id = model_id
        self._api_key = (
            api_key
            or get_settings().google_ai_api_key
            or os.environ.get("LANGEXTRACT_API_KEY")
        )

    # ── helpers ────────────────────────────────────────────────────────

    def _call_lx(
        self,
        text: str,
        prompt: str,
        examples: list[lx.data.ExampleData],
    ) -> lx.data.ExtractionResult:
        """Single choke-point for every ``lx.extract`` call."""
        return lx.extract(
            text_or_documents=text,
            prompt_description=prompt,
            examples=examples,
            model_id=self.model_id,
            api_key=self._api_key,
            show_progress=False,
        )

    @staticmethod
    def _validate_citation(
        extraction_text: str,
        evidence_map: dict[str, str],
        *,
        min_overlap_ratio: float = 0.4,
    ) -> list[str]:
        """Return evidence keys whose content actually contains the quote.

        Uses token-overlap ratio so minor paraphrases still match, but
        completely unrelated chunks are rejected.

        Parameters
        ----------
        extraction_text : str
            The ``extraction_text`` returned by LangExtract.
        evidence_map : dict[str, str]
            ``{key: plain_text_content}`` — the evidence chunks we want
            to validate against.
        min_overlap_ratio : float
            Minimum Jaccard-like overlap to accept a match.
        """
        if not extraction_text:
            return []

        quote_tokens = set(extraction_text.lower().split())
        if not quote_tokens:
            return []

        matched_keys: list[str] = []
        for key, content in evidence_map.items():
            content_tokens = set(content.lower().split())
            if not content_tokens:
                continue
            overlap = len(quote_tokens & content_tokens)
            ratio = overlap / len(quote_tokens)
            if ratio >= min_overlap_ratio:
                matched_keys.append(key)

        return matched_keys

    @staticmethod
    def _dedup_variables(
        variables: list[ExtractedVariable],
    ) -> list[ExtractedVariable]:
        """Collapse near-duplicate variables by normalised name.

        Keeps the *first* occurrence (which tends to have the best
        description) and merges evidence keys from duplicates.
        """
        seen: dict[str, ExtractedVariable] = {}
        for var in variables:
            canon = canonicalize_var_id(var.name)
            if canon in seen:
                # Merge grounded evidence keys
                existing = seen[canon]
                for k in var.grounded_evidence_keys:
                    if k not in existing.grounded_evidence_keys:
                        existing.grounded_evidence_keys.append(k)
                continue
            seen[canon] = var
        return list(seen.values())

    # ── public API ─────────────────────────────────────────────────────

    def extract_variables(
        self,
        evidence_text: str,
        domain: str,
        evidence_map: dict[str, str] | None = None,
    ) -> list[ExtractedVariable]:
        """Extract causal variables from evidence text.

        Parameters
        ----------
        evidence_text : str
            Concatenated evidence passages.
        domain : str
            Business domain (injected into the prompt).
        evidence_map : dict[str, str] | None
            ``{key: content}`` used for citation validation.  If
            ``None``, citation validation is skipped.

        Returns
        -------
        list[ExtractedVariable]
            Deduplicated, citation-validated variable list.
        """
        prompt = f"Domain: {domain}. " + _VARIABLE_PROMPT
        result = self._call_lx(evidence_text, prompt, _VARIABLE_EXAMPLES)

        variables: list[ExtractedVariable] = []
        seen_names: set[str] = set()

        for ext in result.extractions or []:
            if ext.extraction_class != "variable":
                continue
            attrs = ext.attributes or {}
            name = ext.extraction_text.strip()
            if not name or name.lower() in seen_names:
                continue
            seen_names.add(name.lower())

            # Citation validation
            grounded: list[str] = []
            if evidence_map:
                grounded = self._validate_citation(
                    ext.extraction_text, evidence_map
                )

            variables.append(ExtractedVariable(
                name=name,
                description=attrs.get("description", ""),
                var_type=attrs.get("type", "continuous").lower(),
                measurement_status=attrs.get(
                    "measurement_status", "measured"
                ).lower(),
                chain_of_thought=attrs.get("chain_of_thought", ""),
                grounded_evidence_keys=grounded,
            ))

        # Post-processing dedup
        variables = self._dedup_variables(variables)

        _logger.info(
            "ExtractionService.extract_variables → %d variables: %s",
            len(variables),
            [v.name for v in variables],
        )
        return variables

    def extract_edges(
        self,
        evidence_text: str,
        domain: str,
        variable_ids: list[str],
        variable_names: list[str],
        evidence_map: dict[str, str] | None = None,
        *,
        dynamic_examples: list[lx.data.ExampleData] | None = None,
    ) -> list[ExtractedEdge]:
        """Extract causal edges from evidence, constrained to known vars.

        Parameters
        ----------
        evidence_text : str
            Concatenated evidence passages.
        domain : str
            Business domain.
        variable_ids : list[str]
            Snake-case IDs the LLM is allowed to use.
        variable_names : list[str]
            Human-readable names (same order as *variable_ids*).
        evidence_map : dict[str, str] | None
            Deprecated — citation validation is now handled by the
            verification loop.  Accepted for backward compatibility
            but ignored.
        dynamic_examples : list | None
            Caller-generated few-shot examples using the actual variable
            IDs.  Falls back to a generic example if ``None``.

        Returns
        -------
        list[ExtractedEdge]
            Edge list with mechanism + strength (no evidence grounding).
        """
        allowed_vars_str = ", ".join(variable_ids)
        prompt = _EDGE_PROMPT_TEMPLATE.format(allowed_vars=allowed_vars_str)

        # Prepend domain + variable context
        var_context = ", ".join(
            f"{vid} ({vname})"
            for vid, vname in zip(variable_ids, variable_names)
        )
        full_text = (
            f"Domain: {domain}. "
            f"Variables: {var_context}.\n\n"
            f"Evidence:\n{evidence_text}"
        )

        examples = dynamic_examples or self._build_edge_examples(
            variable_ids, variable_names
        )

        result = self._call_lx(full_text, prompt, examples)

        edges: list[ExtractedEdge] = []
        seen_pairs: set[tuple[str, str]] = set()
        allowed_set = set(variable_ids)

        for ext in result.extractions or []:
            if ext.extraction_class != "causal_edge":
                continue
            attrs = ext.attributes or {}
            from_var = (attrs.get("from_var") or "").strip()
            to_var = (attrs.get("to_var") or "").strip()
            if not from_var or not to_var:
                continue
            # Reject hallucinated variable IDs
            if from_var not in allowed_set or to_var not in allowed_set:
                _logger.debug(
                    "Rejected edge %s→%s: var ID not in allowed set",
                    from_var, to_var,
                )
                continue
            pair = (from_var, to_var)
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)

            # Note: citation validation removed — edges start as
            # "draft" and are grounded by the verification loop.

            edges.append(ExtractedEdge(
                from_var=from_var,
                to_var=to_var,
                mechanism=attrs.get("mechanism", ""),
                strength=(attrs.get("strength") or "hypothesis").lower(),
                chain_of_thought=attrs.get("chain_of_thought", ""),
                extraction_text=ext.extraction_text,
                grounded_evidence_keys=[],
            ))

        _logger.info(
            "ExtractionService.extract_edges → %d edges", len(edges),
        )
        return edges

    def parse_query(
        self,
        query: str,
        available_domains: list[str] | None = None,
    ) -> ExtractedQuery:
        """Parse a decision query into structured components.

        Parameters
        ----------
        query : str
            Raw user query.
        available_domains : list[str] | None
            If provided, the prompt constrains the LLM to pick from
            this list.

        Returns
        -------
        ExtractedQuery
        """
        prompt = _QUERY_PROMPT
        if available_domains:
            prompt += (
                f"\n\nCRITICAL CONSTRAINT: The 'domain' attribute MUST be "
                f"one of these available domains: "
                f"{', '.join(available_domains)}. "
                f"Pick the single best match for the user's query."
            )

        result = self._call_lx(query, prompt, _QUERY_EXAMPLES)

        for ext in result.extractions or []:
            if ext.extraction_class == "decision_query":
                attrs = ext.attributes or {}
                constraints_raw = attrs.get("constraints", "")
                return ExtractedQuery(
                    domain=attrs.get("domain", "general"),
                    intervention=attrs.get("intervention", "unknown action"),
                    target_outcome=attrs.get(
                        "target_outcome", "unknown outcome"
                    ),
                    constraints=(
                        [c.strip() for c in constraints_raw.split(",")]
                        if isinstance(constraints_raw, str) and constraints_raw
                        else constraints_raw
                        if isinstance(constraints_raw, list)
                        else []
                    ),
                )

        # Fallback
        return ExtractedQuery(
            domain="general",
            intervention="the proposed action",
            target_outcome="business outcomes",
        )

    def synthesize_recommendation(
        self,
        context_text: str,
        evidence_map: dict[str, str] | None = None,
    ) -> ExtractedRecommendation:
        """Extract a recommendation claim from causal analysis text.

        Parameters
        ----------
        context_text : str
            Full analysis context (query + paths + evidence).
        evidence_map : dict[str, str] | None
            For citation validation.

        Returns
        -------
        ExtractedRecommendation
        """
        result = self._call_lx(
            context_text, _RECOMMENDATION_PROMPT, _RECOMMENDATION_EXAMPLES,
        )

        for ext in result.extractions or []:
            if ext.extraction_class != "recommendation_claim":
                continue
            attrs = ext.attributes or {}

            # Citation validation
            grounded: list[str] = []
            if evidence_map:
                grounded = self._validate_citation(
                    ext.extraction_text, evidence_map
                )

            actions_raw = attrs.get("actions", "")
            risks_raw = attrs.get("risks", "")

            return ExtractedRecommendation(
                recommendation=attrs.get(
                    "recommendation",
                    "Unable to provide recommendation",
                ),
                confidence=attrs.get("confidence", "medium").lower(),
                reasoning=attrs.get("reasoning", ""),
                actions=(
                    [a.strip() for a in actions_raw.split(",")]
                    if isinstance(actions_raw, str) and actions_raw
                    else actions_raw
                    if isinstance(actions_raw, list)
                    else []
                ),
                risks=(
                    [r.strip() for r in risks_raw.split(",")]
                    if isinstance(risks_raw, str) and risks_raw
                    else risks_raw
                    if isinstance(risks_raw, list)
                    else []
                ),
                extraction_text=ext.extraction_text,
                grounded_evidence_keys=grounded,
            )

        # Fallback
        return ExtractedRecommendation(
            recommendation="Unable to synthesize recommendation from analysis",
            confidence="low",
            reasoning="",
            actions=[],
            risks=["Analysis may be incomplete"],
        )

    # ── internal example builders ──────────────────────────────────────

    @staticmethod
    def _build_edge_examples(
        variable_ids: list[str],
        variable_names: list[str],
    ) -> list[lx.data.ExampleData]:
        """Build dynamic few-shot examples using actual variable IDs.

        Falls back to a minimal generic example if fewer than 2 variables.
        """
        if len(variable_ids) < 2:
            return [
                lx.data.ExampleData(
                    text="A influences B.",
                    extractions=[
                        lx.data.Extraction(
                            extraction_class="causal_edge",
                            extraction_text="A influences B",
                            attributes={
                                "from_var": "a",
                                "to_var": "b",
                                "mechanism": "Direct influence",
                                "strength": "moderate",
                                "chain_of_thought": (
                                    "The text states A influences B directly."
                                ),
                            },
                        ),
                    ],
                )
            ]

        extractions = [
            lx.data.Extraction(
                extraction_class="causal_edge",
                extraction_text=(
                    f"Changes in {variable_names[0]} influence "
                    f"{variable_names[1]}"
                ),
                attributes={
                    "from_var": variable_ids[0],
                    "to_var": variable_ids[1],
                    "mechanism": "Direct causal influence",
                    "strength": "moderate",
                    "chain_of_thought": (
                        "The evidence explicitly links these two variables."
                    ),
                },
            ),
        ]
        text_parts = [
            f"{variable_names[0]} influences {variable_names[1]}."
        ]

        if len(variable_ids) >= 3:
            extractions.append(
                lx.data.Extraction(
                    extraction_class="causal_edge",
                    extraction_text=(
                        f"{variable_names[1]} drives changes in "
                        f"{variable_names[2]}"
                    ),
                    attributes={
                        "from_var": variable_ids[1],
                        "to_var": variable_ids[2],
                        "mechanism": "Indirect influence",
                        "strength": "hypothesis",
                        "chain_of_thought": (
                            "Plausible chain from the second to third "
                            "variable."
                        ),
                    },
                )
            )
            text_parts.append(
                f"{variable_names[1]} drives {variable_names[2]}."
            )

        return [
            lx.data.ExampleData(
                text=" ".join(text_parts),
                extractions=extractions,
            )
        ]
