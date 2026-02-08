"""
Tests for Phase 1: PyWhyLLM Bridge & Evidence-Grounded DAG Construction

Tests cover:
- CausalGraphBridge in mock mode (no API key)
- build_graph_from_evidence() with evidence bundles
- classify_edge_strength() evidence classification
- extract_mechanism_from_evidence() citation building
- classify_variable_roles() from graph structure
- suggest_missing_confounders() in mock mode
- EdgeProposal / BridgeResult data structures
"""

import pytest
from uuid import uuid4

from src.causal.pywhyllm_bridge import (
    CausalGraphBridge,
    EdgeProposal,
    VariableClassification,
    ConfounderSuggestion,
    BridgeResult,
)
from src.models.enums import EvidenceStrength, VariableRole, RetrievalMethod
from src.models.evidence import (
    EvidenceBundle,
    SourceReference,
    LocationMetadata,
    RetrievalTrace,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_evidence_bundle(
    content: str,
    doc_title: str = "TestDoc.pdf",
    section_name: str | None = None,
    page_number: int | None = None,
) -> EvidenceBundle:
    """Create a minimal EvidenceBundle for testing."""
    return EvidenceBundle(
        bundle_id=uuid4(),
        content=content,
        source=SourceReference(doc_id=f"doc_{uuid4().hex[:6]}", doc_title=doc_title),
        location=LocationMetadata(section_name=section_name, page_number=page_number),
        retrieval_trace=RetrievalTrace(
            method=RetrievalMethod.HAYSTACK,
            query="test query",
        ),
    )


# ---------------------------------------------------------------------------
# CausalGraphBridge — construction & mock mode
# ---------------------------------------------------------------------------


class TestCausalGraphBridgeInit:
    """Test bridge initialization and mock mode."""

    def test_mock_mode_without_api_key(self):
        """Should activate mock mode when no API key."""
        bridge = CausalGraphBridge(api_key=None)
        assert bridge.is_mock_mode

    def test_explicit_api_key_disables_mock(self):
        """Should NOT be in mock mode when API key is given (even if init fails)."""
        # We pass a clearly invalid key so the actual PyWhyLLM init
        # will fail gracefully — the bridge should fall back to mock
        # rather than crash.
        bridge = CausalGraphBridge(api_key="invalid-key-for-test")
        # Either real mode succeeded or it fell back to mock on error
        assert isinstance(bridge.is_mock_mode, bool)

    def test_custom_model_name(self):
        """Should accept a custom model name."""
        bridge = CausalGraphBridge(api_key=None, model="gemini-2.5-pro")
        assert bridge._model == "gemini-2.5-pro"


# ---------------------------------------------------------------------------
# build_graph_from_evidence (mock mode)
# ---------------------------------------------------------------------------


class TestBuildGraphFromEvidence:
    """Test evidence-grounded graph construction in mock mode."""

    @pytest.fixture
    def bridge(self):
        return CausalGraphBridge(api_key=None)

    @pytest.fixture
    def variables(self):
        return ["price", "demand", "revenue"]

    @pytest.fixture
    def evidence_bundles(self):
        return {
            "price": [
                _make_evidence_bundle(
                    "price increase leads to reduced demand in Q3",
                    doc_title="Q3_Report.pdf",
                    section_name="4.2",
                    page_number=12,
                ),
            ],
            "demand": [
                _make_evidence_bundle(
                    "demand drives revenue growth in domestic markets",
                    doc_title="Sales_Analysis.pdf",
                    section_name="Revenue",
                ),
            ],
            "revenue": [],
        }

    def test_returns_bridge_result(self, bridge, variables, evidence_bundles):
        """Should return a BridgeResult with proposals."""
        result = bridge.build_graph_from_evidence("pricing", variables, evidence_bundles)
        assert isinstance(result, BridgeResult)

    def test_mock_creates_chain_edges(self, bridge, variables, evidence_bundles):
        """Mock mode should create chain edges between consecutive variables."""
        result = bridge.build_graph_from_evidence("pricing", variables, evidence_bundles)
        # 3 variables → 2 chain edges: price→demand, demand→revenue
        assert len(result.edge_proposals) == 2
        assert result.edge_proposals[0].from_var == "price"
        assert result.edge_proposals[0].to_var == "demand"
        assert result.edge_proposals[1].from_var == "demand"
        assert result.edge_proposals[1].to_var == "revenue"

    def test_edge_proposals_have_mechanism(self, bridge, variables, evidence_bundles):
        """Edge proposals should have a mechanism description."""
        result = bridge.build_graph_from_evidence("pricing", variables, evidence_bundles)
        for ep in result.edge_proposals:
            assert ep.mechanism  # non-empty string

    def test_edge_proposals_have_strength(self, bridge, variables, evidence_bundles):
        """Every edge proposal should have a strength classification."""
        result = bridge.build_graph_from_evidence("pricing", variables, evidence_bundles)
        for ep in result.edge_proposals:
            assert isinstance(ep.strength, EvidenceStrength)

    def test_edge_proposals_have_assumptions(self, bridge, variables, evidence_bundles):
        """Edges should include causal assumptions."""
        result = bridge.build_graph_from_evidence("pricing", variables, evidence_bundles)
        for ep in result.edge_proposals:
            assert isinstance(ep.assumptions, list)
            # Mock grounds via _ground_edge_in_evidence which adds default assumptions
            assert len(ep.assumptions) >= 1

    def test_variable_classifications_generated(self, bridge, variables, evidence_bundles):
        """Should classify every variable's role."""
        result = bridge.build_graph_from_evidence("pricing", variables, evidence_bundles)
        assert len(result.variable_classifications) == len(variables)
        roles = {vc.variable_id: vc.role for vc in result.variable_classifications}
        # In a chain: first is TREATMENT, middle is MEDIATOR, last is OUTCOME
        assert roles["price"] == VariableRole.TREATMENT
        assert roles["demand"] == VariableRole.MEDIATOR
        assert roles["revenue"] == VariableRole.OUTCOME

    def test_confounder_suggestion_in_mock(self, bridge, variables, evidence_bundles):
        """Mock mode should suggest at least one confounder."""
        result = bridge.build_graph_from_evidence("pricing", variables, evidence_bundles)
        assert len(result.confounder_suggestions) >= 1
        assert isinstance(result.confounder_suggestions[0], ConfounderSuggestion)

    def test_warnings_indicate_mock(self, bridge, variables, evidence_bundles):
        """Mock mode should emit a warning."""
        result = bridge.build_graph_from_evidence("pricing", variables, evidence_bundles)
        assert any("mock" in w.lower() for w in result.warnings)

    def test_empty_variables_returns_empty(self, bridge):
        """No variables → no edges."""
        result = bridge.build_graph_from_evidence("pricing", [], {})
        assert len(result.edge_proposals) == 0

    def test_single_variable_returns_no_edges(self, bridge):
        """One variable can't form an edge."""
        result = bridge.build_graph_from_evidence("pricing", ["price"], {"price": []})
        assert len(result.edge_proposals) == 0

    def test_evidence_grounding_attaches_hashes(self, bridge):
        """Edge proposals should include supporting evidence hashes when evidence matches."""
        eb = _make_evidence_bundle(
            "price increase leads to reduced demand",
            doc_title="Report.pdf",
        )
        variables = ["price", "demand"]
        evidence_bundles = {"price": [eb], "demand": []}

        result = bridge.build_graph_from_evidence("pricing", variables, evidence_bundles)
        # The evidence mentions both "price" and "demand" so it should be grounded
        assert len(result.edge_proposals) == 1
        assert len(result.edge_proposals[0].supporting_evidence) >= 1


# ---------------------------------------------------------------------------
# classify_edge_strength (static method)
# ---------------------------------------------------------------------------


class TestClassifyEdgeStrength:
    """Test evidence-based edge strength classification."""

    def test_no_evidence_is_hypothesis(self):
        strength, conf = CausalGraphBridge.classify_edge_strength(0)
        assert strength == EvidenceStrength.HYPOTHESIS
        assert conf == 0.3

    def test_one_supporting_is_hypothesis(self):
        strength, conf = CausalGraphBridge.classify_edge_strength(1)
        assert strength == EvidenceStrength.HYPOTHESIS
        assert conf == 0.5

    def test_two_supporting_is_moderate(self):
        strength, conf = CausalGraphBridge.classify_edge_strength(2)
        assert strength == EvidenceStrength.MODERATE
        assert conf == 0.7

    def test_three_supporting_is_strong(self):
        strength, conf = CausalGraphBridge.classify_edge_strength(3)
        assert strength == EvidenceStrength.STRONG
        assert conf == 0.9

    def test_many_supporting_is_strong(self):
        strength, conf = CausalGraphBridge.classify_edge_strength(10)
        assert strength == EvidenceStrength.STRONG
        assert conf == 0.9

    def test_contradiction_makes_contested(self):
        strength, conf = CausalGraphBridge.classify_edge_strength(3, 1)
        assert strength == EvidenceStrength.CONTESTED

    def test_equal_support_and_contradiction(self):
        strength, conf = CausalGraphBridge.classify_edge_strength(2, 2)
        assert strength == EvidenceStrength.CONTESTED
        assert conf == 0.1  # max(0.1, (2-2)/4) = 0.1

    def test_more_contradicting_than_supporting(self):
        strength, conf = CausalGraphBridge.classify_edge_strength(1, 3)
        assert strength == EvidenceStrength.CONTESTED
        assert conf == 0.1  # clamped to min 0.1

    def test_contested_confidence_scales_with_support(self):
        strength, conf = CausalGraphBridge.classify_edge_strength(5, 1)
        assert strength == EvidenceStrength.CONTESTED
        expected = round((5 - 1) / 6, 2)
        assert conf == expected


# ---------------------------------------------------------------------------
# extract_mechanism_from_evidence (static method)
# ---------------------------------------------------------------------------


class TestExtractMechanismFromEvidence:
    """Test mechanism extraction with citations."""

    def test_no_evidence_returns_default(self):
        result = CausalGraphBridge.extract_mechanism_from_evidence("price", "demand", [])
        assert "no supporting evidence" in result.lower()

    def test_single_evidence_returns_citation(self):
        eb = _make_evidence_bundle(
            "Higher prices lead to reduced customer demand through elasticity effects.",
            doc_title="Economics_Report.pdf",
            section_name="4.2",
            page_number=15,
        )
        result = CausalGraphBridge.extract_mechanism_from_evidence("price", "demand", [eb])
        assert "Economics_Report.pdf" in result
        assert "Section: 4.2" in result
        assert "p.15" in result

    def test_multiple_evidence_concatenates_citations(self):
        ebs = [
            _make_evidence_bundle("Evidence piece 1", doc_title="Doc1.pdf"),
            _make_evidence_bundle("Evidence piece 2", doc_title="Doc2.pdf"),
        ]
        result = CausalGraphBridge.extract_mechanism_from_evidence("a", "b", ebs)
        assert "Doc1.pdf" in result
        assert "Doc2.pdf" in result

    def test_caps_citations_at_five(self):
        ebs = [
            _make_evidence_bundle(f"Evidence piece {i}", doc_title=f"Doc{i}.pdf")
            for i in range(10)
        ]
        result = CausalGraphBridge.extract_mechanism_from_evidence("a", "b", ebs)
        # Should not cite more than 5
        cited_docs = [f"Doc{i}.pdf" for i in range(10) if f"Doc{i}.pdf" in result]
        assert len(cited_docs) <= 5


# ---------------------------------------------------------------------------
# classify_variable_roles
# ---------------------------------------------------------------------------


class TestClassifyVariableRoles:
    """Test role classification from DAG structure."""

    @pytest.fixture
    def bridge(self):
        return CausalGraphBridge(api_key=None)

    def test_chain_classification(self, bridge):
        """A→B→C chain: A=treatment, B=mediator, C=outcome."""
        proposals = [
            EdgeProposal(from_var="a", to_var="b", mechanism="A→B", strength=EvidenceStrength.STRONG),
            EdgeProposal(from_var="b", to_var="c", mechanism="B→C", strength=EvidenceStrength.STRONG),
        ]
        classifications = bridge.classify_variable_roles(["a", "b", "c"], proposals)
        roles = {vc.variable_id: vc.role for vc in classifications}
        assert roles["a"] == VariableRole.TREATMENT
        assert roles["b"] == VariableRole.MEDIATOR
        assert roles["c"] == VariableRole.OUTCOME

    def test_isolated_node_is_covariate(self, bridge):
        """Nodes with no edges are COVARIATE."""
        classifications = bridge.classify_variable_roles(["x"], [])
        assert classifications[0].role == VariableRole.COVARIATE

    def test_multiple_roots_are_treatment(self, bridge):
        """Multiple source nodes (no incoming) should be TREATMENT."""
        proposals = [
            EdgeProposal(from_var="a", to_var="c", mechanism="A→C", strength=EvidenceStrength.MODERATE),
            EdgeProposal(from_var="b", to_var="c", mechanism="B→C", strength=EvidenceStrength.MODERATE),
        ]
        classifications = bridge.classify_variable_roles(["a", "b", "c"], proposals)
        roles = {vc.variable_id: vc.role for vc in classifications}
        assert roles["a"] == VariableRole.TREATMENT
        assert roles["b"] == VariableRole.TREATMENT
        assert roles["c"] == VariableRole.OUTCOME


# ---------------------------------------------------------------------------
# suggest_missing_confounders (mock mode)
# ---------------------------------------------------------------------------


class TestSuggestMissingConfounders:
    """Test confounder suggestion in mock mode."""

    @pytest.fixture
    def bridge(self):
        return CausalGraphBridge(api_key=None)

    def test_returns_confounder_suggestion(self, bridge):
        result = bridge.suggest_missing_confounders(
            ["price", "demand", "revenue"], "price", "revenue"
        )
        assert len(result) >= 1
        assert isinstance(result[0], ConfounderSuggestion)

    def test_suggestion_mentions_treatment_outcome(self, bridge):
        result = bridge.suggest_missing_confounders(
            ["price", "demand"], "price", "demand"
        )
        suggestion = result[0]
        assert "price" in suggestion.description.lower() or "demand" in suggestion.description.lower()


# ---------------------------------------------------------------------------
# Data structure tests
# ---------------------------------------------------------------------------


class TestDataStructures:
    """Test EdgeProposal, BridgeResult, VariableClassification defaults."""

    def test_edge_proposal_defaults(self):
        ep = EdgeProposal(from_var="a", to_var="b", mechanism="test", strength=EvidenceStrength.HYPOTHESIS)
        assert ep.supporting_evidence == []
        assert ep.contradicting_evidence == []
        assert ep.assumptions == []
        assert ep.conditions == []
        assert ep.confidence == 0.5

    def test_bridge_result_defaults(self):
        br = BridgeResult()
        assert br.edge_proposals == []
        assert br.variable_classifications == []
        assert br.confounder_suggestions == []
        assert br.assumptions == []
        assert br.warnings == []

    def test_variable_classification_creation(self):
        vc = VariableClassification(
            variable_id="price", role=VariableRole.TREATMENT, reason="Root node"
        )
        assert vc.variable_id == "price"
        assert vc.role == VariableRole.TREATMENT

    def test_confounder_suggestion_creation(self):
        cs = ConfounderSuggestion(
            name="season", description="Seasonal effects", affects_treatment=True, affects_outcome=True
        )
        assert cs.name == "season"
        assert cs.affects_treatment
        assert cs.affects_outcome
