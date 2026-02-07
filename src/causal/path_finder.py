"""
Causal Path Finder

Provides causal reasoning operations on DAGs:
- Path finding between variables
- Confounder detection
- Mediator detection
- Causal effect estimation scaffolding
"""

from dataclasses import dataclass, field
from typing import Optional

import networkx as nx


@dataclass
class SimplePath:
    """A simple path through the DAG (internal representation)."""
    path: list[str]
    length: int


@dataclass
class ConfounderResult:
    """Result of confounder analysis."""
    cause: str
    effect: str
    confounders: list[str]
    explanation: str


@dataclass
class MediatorResult:
    """Result of mediator analysis."""
    cause: str
    effect: str
    mediators: list[str]
    direct_path_exists: bool


@dataclass
class CausalAnalysis:
    """Complete causal analysis between two variables."""
    cause: str
    effect: str
    paths: list[SimplePath]
    confounders: list[str]
    mediators: list[str]
    direct_effect: bool
    total_paths: int
    min_path_length: Optional[int]
    max_path_length: Optional[int]


class CausalPathFinder:
    """
    Causal path finding and analysis.
    
    Provides methods for:
    - Finding all causal paths
    - Detecting confounders (common causes)
    - Detecting mediators (intermediate variables)
    - Analyzing causal relationships
    """
    
    def __init__(self, graph: nx.DiGraph):
        """
        Initialize with a NetworkX DiGraph.
        
        Args:
            graph: The causal DAG
        """
        self._graph = graph
    
    def find_all_paths(
        self,
        source: str,
        target: str,
        max_length: Optional[int] = None,
    ) -> list[SimplePath]:
        """
        Find all directed paths from source to target.
        
        Args:
            source: Starting variable
            target: Ending variable
            max_length: Maximum path length (optional)
            
        Returns:
            List of SimplePath objects
        """
        if source not in self._graph or target not in self._graph:
            return []
        
        try:
            if max_length:
                all_paths = list(nx.all_simple_paths(
                    self._graph, source, target, cutoff=max_length
                ))
            else:
                all_paths = list(nx.all_simple_paths(
                    self._graph, source, target
                ))
        except nx.NetworkXNoPath:
            return []
        
        return [
            SimplePath(
                path=path,
                length=len(path) - 1,
            )
            for path in all_paths
        ]
    
    def find_shortest_path(
        self,
        source: str,
        target: str,
    ) -> Optional[SimplePath]:
        """
        Find the shortest directed path.
        
        Args:
            source: Starting variable
            target: Ending variable
            
        Returns:
            SimplePath or None if no path exists
        """
        try:
            path = nx.shortest_path(self._graph, source, target)
            return SimplePath(path=path, length=len(path) - 1)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None
    
    def has_path(self, source: str, target: str) -> bool:
        """Check if a directed path exists."""
        try:
            return nx.has_path(self._graph, source, target)
        except nx.NodeNotFound:
            return False
    
    def find_confounders(self, cause: str, effect: str) -> ConfounderResult:
        """
        Find confounders (common causes) for a cause-effect pair.
        
        A confounder is a variable that:
        - Has a directed path TO the cause
        - Has a directed path TO the effect
        - Is not on any directed path FROM cause TO effect
        
        Args:
            cause: The cause variable
            effect: The effect variable
            
        Returns:
            ConfounderResult with list of confounders
        """
        confounders = []
        
        # Get all ancestors of both cause and effect
        cause_ancestors = nx.ancestors(self._graph, cause) if cause in self._graph else set()
        effect_ancestors = nx.ancestors(self._graph, effect) if effect in self._graph else set()
        
        # Common ancestors are potential confounders
        common_ancestors = cause_ancestors & effect_ancestors
        
        # Get nodes on paths from cause to effect (these are mediators, not confounders)
        on_path = set()
        paths = self.find_all_paths(cause, effect)
        for p in paths:
            on_path.update(p.path[1:-1])  # Exclude cause and effect themselves
        
        # Confounders are common ancestors not on the path
        confounders = [n for n in common_ancestors if n not in on_path]
        
        explanation = ""
        if confounders:
            explanation = (
                f"Found {len(confounders)} confounder(s) affecting both "
                f"'{cause}' and '{effect}': {', '.join(confounders)}. "
                "These should be controlled for in causal analysis."
            )
        else:
            explanation = f"No confounders found between '{cause}' and '{effect}'."
        
        return ConfounderResult(
            cause=cause,
            effect=effect,
            confounders=confounders,
            explanation=explanation,
        )
    
    def find_mediators(self, cause: str, effect: str) -> MediatorResult:
        """
        Find mediators (intermediate variables) between cause and effect.
        
        A mediator is a variable on the directed path from cause to effect.
        
        Args:
            cause: The cause variable
            effect: The effect variable
            
        Returns:
            MediatorResult with list of mediators
        """
        mediators = set()
        direct_path_exists = False
        
        paths = self.find_all_paths(cause, effect)
        
        for path in paths:
            if path.length == 1:
                direct_path_exists = True
            else:
                # All intermediate nodes are mediators
                mediators.update(path.path[1:-1])
        
        return MediatorResult(
            cause=cause,
            effect=effect,
            mediators=list(mediators),
            direct_path_exists=direct_path_exists,
        )
    
    def analyze(self, cause: str, effect: str) -> CausalAnalysis:
        """
        Perform complete causal analysis between two variables.
        
        Args:
            cause: The cause variable
            effect: The effect variable
            
        Returns:
            CausalAnalysis with paths, confounders, mediators
        """
        paths = self.find_all_paths(cause, effect)
        confounder_result = self.find_confounders(cause, effect)
        mediator_result = self.find_mediators(cause, effect)
        
        path_lengths = [p.length for p in paths]
        
        return CausalAnalysis(
            cause=cause,
            effect=effect,
            paths=paths,
            confounders=confounder_result.confounders,
            mediators=mediator_result.mediators,
            direct_effect=mediator_result.direct_path_exists,
            total_paths=len(paths),
            min_path_length=min(path_lengths) if path_lengths else None,
            max_path_length=max(path_lengths) if path_lengths else None,
        )
    
    def get_descendants(self, node: str) -> set[str]:
        """Get all descendants of a node."""
        if node not in self._graph:
            return set()
        return nx.descendants(self._graph, node)
    
    def get_ancestors(self, node: str) -> set[str]:
        """Get all ancestors of a node."""
        if node not in self._graph:
            return set()
        return nx.ancestors(self._graph, node)
    
    def get_parents(self, node: str) -> list[str]:
        """Get immediate parents (direct causes) of a node."""
        if node not in self._graph:
            return []
        return list(self._graph.predecessors(node))
    
    def get_children(self, node: str) -> list[str]:
        """Get immediate children (direct effects) of a node."""
        if node not in self._graph:
            return []
        return list(self._graph.successors(node))
    
    def topological_sort(self) -> list[str]:
        """Get nodes in topological order (causes before effects)."""
        try:
            return list(nx.topological_sort(self._graph))
        except nx.NetworkXUnfeasible:
            return []  # Graph has cycles (shouldn't happen in proper DAG)
