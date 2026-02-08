"""
PyWhyLLM Bridge

Integrates PyWhyLLM's causal reasoning capabilities with the Causeway system.
Uses Google Gemini via its OpenAI-compatible endpoint so PyWhyLLM's guidance-based
suggesters work out of the box.

Provides:
- Evidence-grounded causal graph construction
- Variable role classification (treatment, outcome, confounder, mediator, instrumental)
- Confounder suggestion
- Assumption tracking for every edge
- Fallback mode when no API key is available (for testing)
"""

from __future__ import annotations

import itertools
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Optional
from uuid import UUID

from src.models.causal import CausalEdge, EdgeMetadata, VariableDefinition
from src.models.enums import (
    EvidenceStrength,
    MeasurementStatus,
    VariableRole,
    VariableType,
)
from src.models.evidence import EvidenceBundle

_logger = logging.getLogger(__name__)

# Gemini OpenAI-compatible endpoint
GEMINI_OPENAI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
GEMINI_OPENAI_MODEL = "gemini-2.5-flash"


# ---------------------------------------------------------------------------
# Data classes for bridge results
# ---------------------------------------------------------------------------

@dataclass
class EdgeProposal:
    """A proposed causal edge with full evidence grounding."""
    from_var: str
    to_var: str
    mechanism: str
    strength: EvidenceStrength
    supporting_evidence: list[str] = field(default_factory=list)   # bundle content_hash prefixes
    contradicting_evidence: list[str] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    conditions: list[str] = field(default_factory=list)
    confidence: float = 0.5
    pywhyllm_description: str = ""  # Raw description from PyWhyLLM


@dataclass
class VariableClassification:
    """Classification of a variable's causal role."""
    variable_id: str
    role: VariableRole
    reason: str = ""


@dataclass
class ConfounderSuggestion:
    """A suggested confounder between treatment and outcome."""
    name: str
    description: str = ""
    affects_treatment: bool = True
    affects_outcome: bool = True


@dataclass
class BridgeResult:
    """Complete result from the PyWhyLLM bridge."""
    edge_proposals: list[EdgeProposal] = field(default_factory=list)
    variable_classifications: list[VariableClassification] = field(default_factory=list)
    confounder_suggestions: list[ConfounderSuggestion] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Bridge implementation
# ---------------------------------------------------------------------------

class CausalGraphBridge:
    """
    Bridge between PyWhyLLM and the Causeway causal intelligence layer.

    When an API key is available, uses PyWhyLLM's ``SimpleModelSuggester``
    backed by Gemini through Google's OpenAI-compatible endpoint.

    When no key is available (tests / CI), falls back to a deterministic
    mock that mirrors the same data structures.
    """

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        """
        Args:
            api_key: Google AI API key (also used for OpenAI-compat endpoint).
                     If *None*, mock mode is activated automatically.
            model:   Gemini model name for the OpenAI-compat endpoint.
                     Defaults to ``gemini-2.5-flash``.
        """
        self._api_key = api_key
        self._model = model or GEMINI_OPENAI_MODEL
        self._mock_mode = api_key is None
        self._suggester = None

        if not self._mock_mode:
            self._init_pywhyllm()

    # -- initialisation helpers ------------------------------------------------

    def _init_pywhyllm(self) -> None:
        """Initialise PyWhyLLM ``SimpleModelSuggester`` with Gemini backend."""
        try:
            import guidance.models
            from pywhyllm import SimpleModelSuggester

            # Create a guidance OpenAI model pointed at Gemini's compat endpoint
            llm = guidance.models.OpenAI(
                self._model,
                api_key=self._api_key,
                base_url=GEMINI_OPENAI_BASE_URL,
                echo=False,
            )

            # Gemini's OpenAI-compat endpoint does NOT recognise the logprobs /
            # top_logprobs fields AT ALL (even when False).  guidance always
            # passes them through ``OpenAIClientWrapper.streaming_chat_completions``
            # → openai.Client.chat.completions.create(), so we monkey-patch
            # the wrapper's method to strip those keys before they hit the wire.
            _orig_scc = llm._interpreter.client.streaming_chat_completions

            def _patched_scc(model, messages, logprobs=False, **kwargs):
                kwargs.pop("top_logprobs", None)
                # Call the original but with logprobs stripped from kwargs
                # The original passes logprobs to openai client, so we need to
                # directly call the openai client without logprobs
                return llm._interpreter.client.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    stream=True,
                    stream_options={"include_usage": True},
                    **kwargs,
                )

            llm._interpreter.client.streaming_chat_completions = _patched_scc

            # SimpleModelSuggester expects ``self.llm`` to be set —
            # we construct an empty instance and patch the attribute.
            suggester = SimpleModelSuggester.__new__(SimpleModelSuggester)
            suggester.llm = llm
            self._suggester = suggester

            _logger.info(
                "PyWhyLLM bridge initialised (model=%s, endpoint=%s)",
                self._model,
                GEMINI_OPENAI_BASE_URL,
            )
        except Exception as exc:
            _logger.warning("PyWhyLLM init failed, falling back to mock: %s", exc)
            self._mock_mode = True

    @property
    def is_mock_mode(self) -> bool:
        return self._mock_mode

    # ------------------------------------------------------------------
    # 1. build_graph_from_evidence
    # ------------------------------------------------------------------

    def build_graph_from_evidence(
        self,
        domain: str,
        variables: list[str],
        evidence_bundles: dict[str, list[EvidenceBundle]],
    ) -> BridgeResult:
        """
        Build a causal graph from evidence using PyWhyLLM.

        For each pair of variables, asks PyWhyLLM to determine whether a
        causal relationship exists, then grounds the edge in retrieved
        evidence bundles.

        Args:
            domain:           Decision domain (e.g. "pricing").
            variables:        List of variable IDs already discovered.
            evidence_bundles: Mapping ``variable_id -> [EvidenceBundle, ...]``
                              collected in Stage 2.

        Returns:
            ``BridgeResult`` with edge proposals, classifications, etc.
        """
        if self._mock_mode:
            return self._mock_build_graph(domain, variables, evidence_bundles)

        result = BridgeResult()

        # -- Step 1: pairwise relationship suggestion via PyWhyLLM ----------
        try:
            relationships: dict[tuple[str, str], str] = (
                self._suggester.suggest_relationships(variables)
            )
        except Exception as exc:
            _logger.warning("PyWhyLLM suggest_relationships failed: %s", exc)
            result.warnings.append(f"PyWhyLLM relationship suggestion failed: {exc}")
            relationships = {}

        # -- Step 2: convert to EdgeProposals with evidence grounding -------
        for (from_var, to_var), description in relationships.items():
            proposal = self._ground_edge_in_evidence(
                from_var, to_var, description, evidence_bundles,
            )
            result.edge_proposals.append(proposal)

        # -- Step 3: suggest confounders ------------------------------------
        if len(variables) >= 2:
            try:
                confounders = self._suggester.suggest_confounders(
                    variables, variables[0], variables[-1],
                )
                for name in confounders:
                    result.confounder_suggestions.append(
                        ConfounderSuggestion(name=name)
                    )
            except Exception as exc:
                _logger.warning("Confounder suggestion failed: %s", exc)

        # -- Step 4: classify variable roles --------------------------------
        result.variable_classifications = self._classify_roles(
            variables, result.edge_proposals,
        )

        return result

    # ------------------------------------------------------------------
    # 2. classify_variable_roles
    # ------------------------------------------------------------------

    def classify_variable_roles(
        self,
        variables: list[str],
        edge_proposals: list[EdgeProposal],
    ) -> list[VariableClassification]:
        """Determine variable roles from graph structure."""
        return self._classify_roles(variables, edge_proposals)

    # ------------------------------------------------------------------
    # 3. suggest_missing_confounders
    # ------------------------------------------------------------------

    def suggest_missing_confounders(
        self,
        variables: list[str],
        treatment: str,
        outcome: str,
    ) -> list[ConfounderSuggestion]:
        """
        Identify potential unmeasured confounders between treatment and outcome.
        """
        if self._mock_mode:
            return [
                ConfounderSuggestion(
                    name="unmeasured_confounder",
                    description=f"Potential unmeasured confounder between {treatment} and {outcome}",
                )
            ]

        try:
            names = self._suggester.suggest_confounders(variables, treatment, outcome)
            return [ConfounderSuggestion(name=n) for n in names]
        except Exception as exc:
            _logger.warning("suggest_confounders failed: %s", exc)
            return []

    # ------------------------------------------------------------------
    # 4. classify_edge_strength_from_evidence
    # ------------------------------------------------------------------

    @staticmethod
    def classify_edge_strength(
        supporting_count: int,
        contradicting_count: int = 0,
    ) -> tuple[EvidenceStrength, float]:
        """
        Classify edge strength and compute confidence from evidence counts.

        Returns:
            ``(EvidenceStrength, confidence_float)``
        """
        if contradicting_count > 0:
            total = supporting_count + contradicting_count
            confidence = max(0.1, (supporting_count - contradicting_count) / total)
            return EvidenceStrength.CONTESTED, round(confidence, 2)
        elif supporting_count >= 3:
            return EvidenceStrength.STRONG, 0.9
        elif supporting_count == 2:
            return EvidenceStrength.MODERATE, 0.7
        elif supporting_count == 1:
            return EvidenceStrength.HYPOTHESIS, 0.5
        else:
            return EvidenceStrength.HYPOTHESIS, 0.3

    # ------------------------------------------------------------------
    # 5. extract_mechanism_from_evidence
    # ------------------------------------------------------------------

    @staticmethod
    def extract_mechanism_from_evidence(
        from_var: str,
        to_var: str,
        evidence_bundles: list[EvidenceBundle],
    ) -> str:
        """
        Build a mechanism description that cites specific documents.

        Example output:
            "Price increases reduce demand through elasticity effects
            (Q3_Postmortem.pdf, Section 4.2; Pricing_Policy_v7.docx, Section 2.1)"
        """
        if not evidence_bundles:
            return f"{from_var} affects {to_var} (no supporting evidence)"

        # Build citation list
        citations: list[str] = []
        for eb in evidence_bundles[:5]:  # cap at 5 citations
            cite_parts = [eb.source.doc_title]
            if eb.location.section_name:
                cite_parts.append(f"Section: {eb.location.section_name}")
            if eb.location.page_number:
                cite_parts.append(f"p.{eb.location.page_number}")
            citations.append(", ".join(cite_parts))

        citation_str = "; ".join(citations)

        # Use first evidence snippet to describe mechanism
        snippet = evidence_bundles[0].content[:200].strip()
        return f"{snippet}... ({citation_str})"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ground_edge_in_evidence(
        self,
        from_var: str,
        to_var: str,
        pywhyllm_description: str,
        evidence_bundles: dict[str, list[EvidenceBundle]],
    ) -> EdgeProposal:
        """Ground a PyWhyLLM edge proposal in retrieved evidence."""
        supporting: list[str] = []
        supporting_bundles: list[EvidenceBundle] = []

        # Gather evidence from both endpoint variables
        for var_key in (from_var, to_var):
            for eb in evidence_bundles.get(var_key, []):
                # Simple heuristic: does the evidence mention both variables?
                content_lower = eb.content.lower()
                from_mentioned = from_var.replace("_", " ") in content_lower or from_var in content_lower
                to_mentioned = to_var.replace("_", " ") in content_lower or to_var in content_lower
                if from_mentioned or to_mentioned:
                    hash_prefix = eb.content_hash[:12]
                    if hash_prefix not in supporting:
                        supporting.append(hash_prefix)
                        supporting_bundles.append(eb)

        strength, confidence = self.classify_edge_strength(len(supporting))

        mechanism = self.extract_mechanism_from_evidence(
            from_var, to_var, supporting_bundles,
        ) if supporting_bundles else pywhyllm_description

        # Default causal assumptions
        assumptions = [
            f"No unmeasured confounders between {from_var} and {to_var}",
            f"Temporal ordering: {from_var} precedes {to_var}",
            "SUTVA: stable unit treatment value assumption",
        ]

        return EdgeProposal(
            from_var=from_var,
            to_var=to_var,
            mechanism=mechanism,
            strength=strength,
            supporting_evidence=supporting,
            assumptions=assumptions,
            confidence=confidence,
            pywhyllm_description=pywhyllm_description,
        )

    @staticmethod
    def _classify_roles(
        variables: list[str],
        edge_proposals: list[EdgeProposal],
    ) -> list[VariableClassification]:
        """Infer variable roles from DAG structure."""
        # Count in-degree and out-degree
        in_degree: dict[str, int] = {v: 0 for v in variables}
        out_degree: dict[str, int] = {v: 0 for v in variables}
        for ep in edge_proposals:
            out_degree[ep.from_var] = out_degree.get(ep.from_var, 0) + 1
            in_degree[ep.to_var] = in_degree.get(ep.to_var, 0) + 1

        classifications: list[VariableClassification] = []
        for v in variables:
            i_deg = in_degree.get(v, 0)
            o_deg = out_degree.get(v, 0)

            if i_deg == 0 and o_deg > 0:
                role = VariableRole.TREATMENT
                reason = f"Root cause with {o_deg} outgoing edge(s), no incoming"
            elif i_deg > 0 and o_deg == 0:
                role = VariableRole.OUTCOME
                reason = f"Terminal node with {i_deg} incoming edge(s), no outgoing"
            elif i_deg > 0 and o_deg > 0:
                role = VariableRole.MEDIATOR
                reason = f"On causal path ({i_deg} in, {o_deg} out)"
            else:
                role = VariableRole.COVARIATE
                reason = "Isolated node — no edges"

            classifications.append(VariableClassification(
                variable_id=v, role=role, reason=reason,
            ))

        return classifications

    # ------------------------------------------------------------------
    # Mock mode (for testing without LLM)
    # ------------------------------------------------------------------

    def _mock_build_graph(
        self,
        domain: str,
        variables: list[str],
        evidence_bundles: dict[str, list[EvidenceBundle]],
    ) -> BridgeResult:
        """Deterministic mock that creates edges between consecutive variables."""
        result = BridgeResult()
        result.warnings.append("Running in mock mode — edges are heuristic only")

        # Create edges between consecutive variables (simple chain)
        for i in range(len(variables) - 1):
            from_var = variables[i]
            to_var = variables[i + 1]

            proposal = self._ground_edge_in_evidence(
                from_var, to_var,
                f"{from_var} causes {to_var} (mock)",
                evidence_bundles,
            )
            result.edge_proposals.append(proposal)

        # Classify roles based on chain structure
        result.variable_classifications = self._classify_roles(
            variables, result.edge_proposals,
        )

        # Add a mock confounder suggestion
        if len(variables) >= 2:
            result.confounder_suggestions.append(
                ConfounderSuggestion(
                    name="unmeasured_confounder",
                    description=f"Potential confounder in {domain} domain",
                )
            )

        return result
