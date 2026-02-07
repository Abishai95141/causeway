"""
DAG Engine

NetworkX-based directed acyclic graph engine for causal world models.
Provides:
- Node/edge management with schema validation
- Acyclicity enforcement
- Graph serialization/deserialization
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional
from uuid import UUID

import networkx as nx

from src.models.causal import (
    VariableDefinition,
    CausalEdge,
    EdgeMetadata,
    WorldModelVersion,
)
from src.models.enums import EvidenceStrength, MeasurementStatus, ModelStatus, VariableType


class CycleDetectedError(Exception):
    """Raised when adding an edge would create a cycle."""
    
    def __init__(self, cycle: list[str]):
        self.cycle = cycle
        super().__init__(f"Adding edge would create cycle: {' → '.join(cycle)}")


class NodeNotFoundError(Exception):
    """Raised when referencing a non-existent node."""
    
    def __init__(self, node_id: str):
        self.node_id = node_id
        super().__init__(f"Node not found: {node_id}")


@dataclass
class DAGValidationResult:
    """Result of DAG validation."""
    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    node_count: int = 0
    edge_count: int = 0


class DAGEngine:
    """
    NetworkX-based DAG engine for causal world models.
    
    Features:
    - Node/edge CRUD with validation
    - Automatic acyclicity enforcement
    - Serialization to/from WorldModelVersion
    """
    
    def __init__(self):
        self._graph = nx.DiGraph()
        self._variables: dict[str, VariableDefinition] = {}
        self._edges: dict[tuple[str, str], CausalEdge] = {}
        self._domain: Optional[str] = None
        self._version_id: Optional[str] = None
    
    @property
    def graph(self) -> nx.DiGraph:
        """Get the underlying NetworkX graph."""
        return self._graph
    
    @property
    def node_count(self) -> int:
        """Number of nodes in the graph."""
        return self._graph.number_of_nodes()
    
    @property
    def edge_count(self) -> int:
        """Number of edges in the graph."""
        return self._graph.number_of_edges()
    
    @property
    def variables(self) -> list[VariableDefinition]:
        """Get all variables (nodes)."""
        return list(self._variables.values())
    
    @property
    def edges(self) -> list[CausalEdge]:
        """Get all causal edges."""
        return list(self._edges.values())
    
    def add_variable(
        self,
        variable_id: str,
        name: str,
        definition: str,
        var_type: VariableType = VariableType.CONTINUOUS,
        measurement_status: MeasurementStatus = MeasurementStatus.MEASURED,
        unit: Optional[str] = None,
        data_source: Optional[str] = None,
    ) -> VariableDefinition:
        """
        Add a variable (node) to the DAG.
        
        Args:
            variable_id: Unique snake_case identifier
            name: Human-readable name
            definition: Clear definition of the variable
            var_type: Variable type (continuous, categorical, etc.)
            measurement_status: Whether it can be measured
            unit: Measurement unit (optional)
            data_source: Where data comes from (optional)
            
        Returns:
            Created VariableDefinition
        """
        if variable_id in self._variables:
            raise ValueError(f"Variable already exists: {variable_id}")
        
        variable = VariableDefinition(
            variable_id=variable_id,
            name=name,
            definition=definition,
            type=var_type,
            measurement_status=measurement_status,
            unit=unit,
            data_source=data_source,
        )
        
        self._variables[variable_id] = variable
        self._graph.add_node(variable_id, **variable.model_dump())
        
        return variable
    
    def add_edge(
        self,
        from_var: str,
        to_var: str,
        mechanism: str,
        strength: EvidenceStrength = EvidenceStrength.HYPOTHESIS,
        evidence_refs: Optional[list[UUID]] = None,
        confidence: float = 0.5,
    ) -> CausalEdge:
        """
        Add a causal edge to the DAG.
        
        Args:
            from_var: Source variable ID
            to_var: Target variable ID
            mechanism: Explanation of causal mechanism
            strength: Evidence strength classification
            evidence_refs: List of evidence bundle UUIDs
            confidence: Confidence score (0-1)
            
        Returns:
            Created CausalEdge
            
        Raises:
            NodeNotFoundError: If from_var or to_var not in graph
            CycleDetectedError: If edge would create a cycle
        """
        # Validate nodes exist
        if from_var not in self._variables:
            raise NodeNotFoundError(from_var)
        if to_var not in self._variables:
            raise NodeNotFoundError(to_var)
        
        # Check for existing edge
        edge_key = (from_var, to_var)
        if edge_key in self._edges:
            raise ValueError(f"Edge already exists: {from_var} → {to_var}")
        
        # Check for cycle before adding
        if self._would_create_cycle(from_var, to_var):
            # Find the cycle for error message
            self._graph.add_edge(from_var, to_var)
            cycles = list(nx.simple_cycles(self._graph))
            self._graph.remove_edge(from_var, to_var)
            raise CycleDetectedError(cycles[0] if cycles else [from_var, to_var, from_var])
        
        # Create edge
        edge = CausalEdge(
            from_var=from_var,
            to_var=to_var,
            metadata=EdgeMetadata(
                mechanism=mechanism,
                evidence_strength=strength,
                evidence_refs=evidence_refs or [],
                confidence=confidence,
            ),
        )
        
        self._edges[edge_key] = edge
        self._graph.add_edge(
            from_var, to_var,
            mechanism=mechanism,
            strength=strength.value,
            evidence_refs=evidence_refs or [],
        )
        
        return edge
    
    def remove_variable(self, variable_id: str) -> None:
        """Remove a variable and all connected edges."""
        if variable_id not in self._variables:
            raise NodeNotFoundError(variable_id)
        
        # Remove from variable dict
        del self._variables[variable_id]
        
        # Remove edges involving this variable
        edges_to_remove = [
            k for k in self._edges.keys()
            if k[0] == variable_id or k[1] == variable_id
        ]
        for key in edges_to_remove:
            del self._edges[key]
        
        # Remove from graph
        self._graph.remove_node(variable_id)
    
    def remove_edge(self, from_var: str, to_var: str) -> None:
        """Remove a causal edge."""
        edge_key = (from_var, to_var)
        if edge_key not in self._edges:
            raise ValueError(f"Edge not found: {from_var} → {to_var}")
        
        del self._edges[edge_key]
        self._graph.remove_edge(from_var, to_var)
    
    def update_edge_strength(
        self,
        from_var: str,
        to_var: str,
        strength: EvidenceStrength,
    ) -> None:
        """Update edge strength classification."""
        edge_key = (from_var, to_var)
        if edge_key not in self._edges:
            raise ValueError(f"Edge not found: {from_var} → {to_var}")
        
        edge = self._edges[edge_key]
        edge.metadata.evidence_strength = strength
        self._graph[from_var][to_var]["strength"] = strength.value
    
    def add_evidence_to_edge(
        self,
        from_var: str,
        to_var: str,
        evidence_id: UUID,
    ) -> None:
        """Add evidence reference to an edge."""
        edge_key = (from_var, to_var)
        if edge_key not in self._edges:
            raise ValueError(f"Edge not found: {from_var} → {to_var}")
        
        edge = self._edges[edge_key]
        if evidence_id not in edge.metadata.evidence_refs:
            edge.metadata.evidence_refs.append(evidence_id)
            self._graph[from_var][to_var]["evidence_refs"].append(evidence_id)
    
    def validate(self) -> DAGValidationResult:
        """Validate the DAG structure."""
        errors = []
        warnings = []
        
        # Check for cycles (should never happen due to guards)
        if not nx.is_directed_acyclic_graph(self._graph):
            errors.append("Graph contains cycles")
        
        # Check for isolated nodes
        isolated = list(nx.isolates(self._graph))
        if isolated:
            warnings.append(f"Isolated nodes without edges: {isolated}")
        
        # Check for nodes without evidence
        for edge_key, edge in self._edges.items():
            if not edge.metadata.evidence_refs:
                warnings.append(f"Edge {edge_key[0]} → {edge_key[1]} has no evidence")
        
        return DAGValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            node_count=self.node_count,
            edge_count=self.edge_count,
        )
    
    def _would_create_cycle(self, source: str, target: str) -> bool:
        """Check if adding edge would create a cycle."""
        # If target can reach source, adding source→target creates cycle
        try:
            return nx.has_path(self._graph, target, source)
        except nx.NetworkXError:
            return False
    
    def to_world_model(
        self,
        domain: str,
        description: str = "",
        version_id: Optional[str] = None,
    ) -> WorldModelVersion:
        """
        Serialize DAG to WorldModelVersion.
        
        Args:
            domain: Domain description
            description: What this model covers
            version_id: Optional version ID
            
        Returns:
            WorldModelVersion with DAG data
        """
        validation = self.validate()
        
        # Generate version_id if not provided
        if not version_id:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            version_id = f"wm_{domain}_{timestamp}"
        
        return WorldModelVersion(
            version_id=version_id,
            domain=domain,
            description=description or f"World model for {domain}",
            variables=self._variables.copy(),
            edges=list(self._edges.values()),
            dag_json=self.to_json(),
            status=ModelStatus.DRAFT,
        )
    
    def to_json(self) -> dict[str, Any]:
        """Serialize DAG to JSON-compatible dict."""
        return {
            "nodes": [v.model_dump(mode='json') for v in self._variables.values()],
            "edges": [
                {
                    "from_var": e.from_var,
                    "to_var": e.to_var,
                    "metadata": e.metadata.model_dump(mode='json'),
                }
                for e in self._edges.values()
            ],
        }
    
    @classmethod
    def from_world_model(cls, model: WorldModelVersion) -> "DAGEngine":
        """
        Create DAGEngine from WorldModelVersion.
        
        Args:
            model: WorldModelVersion with DAG data
            
        Returns:
            Populated DAGEngine
        """
        engine = cls()
        engine._domain = model.domain
        engine._version_id = model.version_id
        
        # Add variables (from dict)
        for var_id, var in model.variables.items():
            engine._variables[var_id] = var
            engine._graph.add_node(var_id, **var.model_dump())
        
        # Add edges
        for edge in model.edges:
            edge_key = (edge.from_var, edge.to_var)
            engine._edges[edge_key] = edge
            engine._graph.add_edge(
                edge.from_var, edge.to_var,
                mechanism=edge.metadata.mechanism,
                strength=edge.metadata.evidence_strength.value,
                evidence_refs=edge.metadata.evidence_refs,
            )
        
        return engine
    
    def clear(self) -> None:
        """Clear all nodes and edges."""
        self._graph.clear()
        self._variables.clear()
        self._edges.clear()
