"""
Bridge Engine — Cross-Model Federated Graphing.

Discovers concept mappings and causal edges between two separate
domain world models.  Uses domain-prefixed variable names to
guarantee namespace isolation when checking acyclicity across the
combined (federated) graph.

Prefix convention:
    ``finance::demand``   vs   ``labour::demand``
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import networkx as nx

from src.models.causal import (
    BridgeEdge,
    ConceptMapping,
    ModelBridge,
    VariableDefinition,
    WorldModelVersion,
)
from src.models.enums import EvidenceStrength, ModelStatus
from src.causal.dag_engine import DAGEngine

_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _prefixed(domain: str, var_id: str) -> str:
    """Return ``domain::var_id``."""
    return f"{domain}::{var_id}"


def _build_federated_graph(
    source_engine: DAGEngine,
    source_domain: str,
    target_engine: DAGEngine,
    target_domain: str,
) -> nx.DiGraph:
    """Combine two DAGs into a single domain-prefixed DiGraph."""
    fed = nx.DiGraph()
    for eng, dom in [(source_engine, source_domain), (target_engine, target_domain)]:
        for node in eng._graph.nodes:
            fed.add_node(_prefixed(dom, node))
        for u, v in eng._graph.edges:
            fed.add_edge(_prefixed(dom, u), _prefixed(dom, v))
    return fed


# ---------------------------------------------------------------------------
# BridgeEngine
# ---------------------------------------------------------------------------

class BridgeEngine:
    """Discovers and validates cross-model bridges.

    Public API
    ----------
    discover_concept_mappings(source, target, llm_client=None)
        Find semantically equivalent variables across two models.

    propose_bridge_edges(source, target, mappings, llm_client=None)
        Propose directed causal edges between the two models.

    validate_acyclicity(source_engine, source_domain, target_engine,
                        target_domain, bridge_edges)
        Check that adding bridge edges keeps the federated graph acyclic.
    """

    # ------------------------------------------------------------------
    # Concept mapping
    # ------------------------------------------------------------------

    async def discover_concept_mappings(
        self,
        source: WorldModelVersion,
        target: WorldModelVersion,
        llm_client: Any | None = None,
    ) -> list[ConceptMapping]:
        """Identify semantically equivalent variables between two models.

        If *llm_client* is provided the LLM is asked to score similarity;
        otherwise a simple name / keyword heuristic is used.
        """
        source_vars = source.variables or {}
        target_vars = target.variables or {}

        if llm_client is not None:
            return await self._llm_concept_mapping(
                source.domain, source_vars,
                target.domain, target_vars,
                llm_client,
            )

        # -- Heuristic fallback (keyword overlap) --
        return self._heuristic_concept_mapping(source_vars, target_vars)

    @staticmethod
    def _heuristic_concept_mapping(
        source_vars: dict[str, Any],
        target_vars: dict[str, Any],
    ) -> list[ConceptMapping]:
        """Token-overlap heuristic for concept similarity."""
        mappings: list[ConceptMapping] = []

        def _tokens(text: str) -> set[str]:
            return {w.lower() for w in text.replace("_", " ").split() if len(w) > 2}

        for s_id, s_def in source_vars.items():
            s_name = s_def.get("name", s_id) if isinstance(s_def, dict) else getattr(s_def, "name", s_id)
            s_defn = s_def.get("definition", "") if isinstance(s_def, dict) else getattr(s_def, "definition", "")
            s_tok = _tokens(s_name) | _tokens(s_defn)

            for t_id, t_def in target_vars.items():
                t_name = t_def.get("name", t_id) if isinstance(t_def, dict) else getattr(t_def, "name", t_id)
                t_defn = t_def.get("definition", "") if isinstance(t_def, dict) else getattr(t_def, "definition", "")
                t_tok = _tokens(t_name) | _tokens(t_defn)

                if not s_tok or not t_tok:
                    continue
                overlap = len(s_tok & t_tok) / max(len(s_tok | t_tok), 1)
                if overlap >= 0.35:
                    mappings.append(ConceptMapping(
                        source_var=s_id,
                        target_var=t_id,
                        similarity_score=round(overlap, 3),
                        mapping_rationale=f"Token overlap ({overlap:.0%}) between '{s_name}' and '{t_name}'",
                    ))

        return sorted(mappings, key=lambda m: m.similarity_score, reverse=True)

    async def _llm_concept_mapping(
        self,
        source_domain: str,
        source_vars: dict[str, Any],
        target_domain: str,
        target_vars: dict[str, Any],
        llm_client: Any,
    ) -> list[ConceptMapping]:
        """Use LLM to determine concept equivalences."""
        import json

        def _var_summary(vars_dict: dict) -> str:
            lines = []
            for vid, vdef in vars_dict.items():
                name = vdef.get("name", vid) if isinstance(vdef, dict) else getattr(vdef, "name", vid)
                defn = vdef.get("definition", "") if isinstance(vdef, dict) else getattr(vdef, "definition", "")
                lines.append(f"  - {vid}: {name} — {defn}")
            return "\n".join(lines)

        prompt = (
            f"You are a causal-modelling expert.  Two world models exist:\n\n"
            f"**Model A** (domain: {source_domain}):\n{_var_summary(source_vars)}\n\n"
            f"**Model B** (domain: {target_domain}):\n{_var_summary(target_vars)}\n\n"
            "Identify pairs of variables that refer to the **same real-world concept** "
            "(or are close enough that one could substitute for the other in cross-model "
            "reasoning).  Return **only** a JSON array of objects, each with:\n"
            '  {"source_var": "...", "target_var": "...", "similarity_score": 0.0-1.0, '
            '"mapping_rationale": "..."}\n'
            "If there are no matches, return an empty array []."
        )

        try:
            response = await llm_client.generate(prompt=prompt, temperature=0.3, max_tokens=2048)
            text = response.content.strip()
            # Strip markdown fences
            if text.startswith("```"):
                text = text.split("\n", 1)[-1].rsplit("```", 1)[0]
            raw = json.loads(text)
            return [ConceptMapping(**item) for item in raw]
        except Exception as exc:
            _logger.warning("LLM concept mapping failed, falling back to heuristic: %s", exc)
            return self._heuristic_concept_mapping(source_vars, target_vars)

    # ------------------------------------------------------------------
    # Bridge edge proposal
    # ------------------------------------------------------------------

    async def propose_bridge_edges(
        self,
        source: WorldModelVersion,
        target: WorldModelVersion,
        concept_mappings: list[ConceptMapping],
        llm_client: Any | None = None,
    ) -> list[BridgeEdge]:
        """Propose directed causal edges spanning the two models.

        For each shared concept the LLM (or heuristic) decides whether a
        causal link exists *across* the domain boundary and in which
        direction it flows.
        """
        import json

        source_vars = source.variables or {}
        target_vars = target.variables or {}

        if llm_client is None or not concept_mappings:
            # Without an LLM, just create hypothesis edges from shared concepts
            edges: list[BridgeEdge] = []
            for m in concept_mappings:
                edges.append(BridgeEdge(
                    source_domain=source.domain,
                    source_var=m.source_var,
                    target_domain=target.domain,
                    target_var=m.target_var,
                    mechanism=f"Shared concept: {m.mapping_rationale}",
                    strength=EvidenceStrength.HYPOTHESIS,
                    confidence=m.similarity_score * 0.7,
                    mapping_rationale=m.mapping_rationale,
                ))
            return edges

        # -- LLM edge proposal --
        def _var_context(domain: str, vars_dict: dict) -> str:
            lines = []
            for vid, vdef in vars_dict.items():
                name = vdef.get("name", vid) if isinstance(vdef, dict) else getattr(vdef, "name", vid)
                defn = vdef.get("definition", "") if isinstance(vdef, dict) else getattr(vdef, "definition", "")
                lines.append(f"  {domain}::{vid} — {name}: {defn}")
            return "\n".join(lines)

        mapping_text = "\n".join(
            f"  {m.source_var} <-> {m.target_var} (score={m.similarity_score})"
            for m in concept_mappings
        )

        prompt = (
            "You are a causal-modelling expert building a **cross-domain bridge** "
            "between two world models.\n\n"
            f"**Model A** ({source.domain}):\n{_var_context(source.domain, source_vars)}\n\n"
            f"**Model B** ({target.domain}):\n{_var_context(target.domain, target_vars)}\n\n"
            f"Shared concepts:\n{mapping_text}\n\n"
            "For each shared concept pair, decide:\n"
            "1. Is there a **directed causal link** across the two models?\n"
            "2. If yes, which direction? (A→B or B→A)\n"
            "3. What is the mechanism?\n\n"
            "Return a JSON array of objects:\n"
            '  {"source_domain": "...", "source_var": "...", "target_domain": "...", '
            '"target_var": "...", "mechanism": "...", "strength": "hypothesis|moderate|strong", '
            '"confidence": 0.0-1.0, "mapping_rationale": "..."}\n'
            "Only include pairs where a real causal link exists."
        )

        try:
            response = await llm_client.generate(prompt=prompt, temperature=0.3, max_tokens=4096)
            text = response.content.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[-1].rsplit("```", 1)[0]
            raw = json.loads(text)
            edges = []
            for item in raw:
                # Normalise strength string to enum
                strength_str = item.get("strength", "hypothesis").upper()
                try:
                    strength = EvidenceStrength(strength_str)
                except ValueError:
                    strength = EvidenceStrength.HYPOTHESIS
                edges.append(BridgeEdge(
                    source_domain=item.get("source_domain", source.domain),
                    source_var=item["source_var"],
                    target_domain=item.get("target_domain", target.domain),
                    target_var=item["target_var"],
                    mechanism=item.get("mechanism", ""),
                    strength=strength,
                    confidence=float(item.get("confidence", 0.5)),
                    mapping_rationale=item.get("mapping_rationale", ""),
                ))
            return edges
        except Exception as exc:
            _logger.warning("LLM bridge-edge proposal failed: %s", exc)
            return []

    # ------------------------------------------------------------------
    # Acyclicity validation (domain-prefixed)
    # ------------------------------------------------------------------

    def validate_acyclicity(
        self,
        source_engine: DAGEngine,
        source_domain: str,
        target_engine: DAGEngine,
        target_domain: str,
        bridge_edges: list[BridgeEdge],
    ) -> tuple[bool, list[str]]:
        """Check that adding *bridge_edges* keeps the federated graph acyclic.

        Returns:
            (is_acyclic, list_of_problem_edges)
        """
        fed = _build_federated_graph(
            source_engine, source_domain, target_engine, target_domain
        )

        bad_edges: list[str] = []
        for be in bridge_edges:
            u = _prefixed(be.source_domain, be.source_var)
            v = _prefixed(be.target_domain, be.target_var)
            fed.add_edge(u, v)
            if not nx.is_directed_acyclic_graph(fed):
                bad_edges.append(f"{u} → {v}")
                fed.remove_edge(u, v)  # rollback offending edge

        is_acyclic = len(bad_edges) == 0
        return is_acyclic, bad_edges

    # ------------------------------------------------------------------
    # Full bridge-building pipeline
    # ------------------------------------------------------------------

    async def build_bridge(
        self,
        source: WorldModelVersion,
        source_engine: DAGEngine,
        target: WorldModelVersion,
        target_engine: DAGEngine,
        llm_client: Any | None = None,
    ) -> ModelBridge:
        """End-to-end bridge creation.

        1. Discover concept mappings
        2. Propose bridge edges
        3. Validate acyclicity (drop violating edges)
        4. Return assembled ``ModelBridge``
        """
        from datetime import datetime, timezone
        from uuid import uuid4

        # Step 1 — Concept mappings
        mappings = await self.discover_concept_mappings(source, target, llm_client)
        _logger.info(
            "Discovered %d concept mappings between %s and %s",
            len(mappings), source.domain, target.domain,
        )

        # Step 2 — Propose edges
        proposed = await self.propose_bridge_edges(source, target, mappings, llm_client)
        _logger.info("Proposed %d bridge edges", len(proposed))

        # Step 3 — Acyclicity filter
        is_ok, bad = self.validate_acyclicity(
            source_engine, source.domain, target_engine, target.domain, proposed
        )
        if bad:
            _logger.warning("Dropped %d cycle-causing bridge edges: %s", len(bad), bad)
            bad_set = set(bad)
            proposed = [
                e for e in proposed
                if f"{_prefixed(e.source_domain, e.source_var)} → "
                   f"{_prefixed(e.target_domain, e.target_var)}" not in bad_set
            ]

        # Step 4 — Assemble
        bridge = ModelBridge(
            bridge_id=str(uuid4()),
            source_version_id=source.version_id,
            source_domain=source.domain,
            target_version_id=target.version_id,
            target_domain=target.domain,
            bridge_edges=proposed,
            shared_concepts=mappings,
            status=ModelStatus.DRAFT,
            created_at=datetime.now(timezone.utc),
        )
        return bridge
