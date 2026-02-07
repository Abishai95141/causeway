"""
Mode Router

Classifies incoming user requests to determine which operating mode 
should handle them:
- Mode 1: World Model Construction
- Mode 2: Decision Support

Uses pattern matching and optional LLM classification.
"""

import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from src.models.enums import OperatingMode


class RouteReason(str, Enum):
    """Reason for routing decision."""
    EXPLICIT_COMMAND = "explicit_command"
    PATTERN_MATCH = "pattern_match"
    LLM_CLASSIFICATION = "llm_classification"
    DEFAULT = "default"


@dataclass
class RouteDecision:
    """Result of mode routing."""
    mode: OperatingMode
    confidence: float  # 0.0 - 1.0
    reason: RouteReason
    extracted_domain: Optional[str] = None
    extracted_objective: Optional[str] = None
    raw_query: str = ""


# Pattern matching rules for Mode 1
MODE1_PATTERNS = [
    # Explicit commands
    (r"(?i)build\s+(?:a\s+)?(?:world\s+)?model\s+for\s+(.+)", "domain"),
    (r"(?i)create\s+(?:a\s+)?causal\s+(?:model|graph|dag)\s+for\s+(.+)", "domain"),
    (r"(?i)construct\s+(?:a\s+)?world\s+model\s+(?:for|about)\s+(.+)", "domain"),
    (r"(?i)analyze\s+(?:the\s+)?causal\s+(?:relationships|structure)\s+(?:for|in|of)\s+(.+)", "domain"),
    (r"(?i)map\s+(?:out\s+)?(?:the\s+)?(?:causal\s+)?relationships\s+(?:for|in)\s+(.+)", "domain"),
    # Update existing model
    (r"(?i)update\s+(?:the\s+)?(?:world\s+)?model\s+(?:for|with)\s+(.+)", "domain"),
    (r"(?i)add\s+(?:new\s+)?variables?\s+to\s+(?:the\s+)?model", None),
]

# Pattern matching rules for Mode 2
MODE2_PATTERNS = [
    # Decision questions
    (r"(?i)should\s+(?:we|i)\s+(.+)\??", "objective"),
    (r"(?i)what\s+(?:would|will)\s+happen\s+if\s+(.+)\??", "objective"),
    (r"(?i)how\s+(?:would|will|does)\s+(.+)\s+affect\s+(.+)\??", None),
    (r"(?i)what\s+is\s+the\s+(?:impact|effect)\s+of\s+(.+)\??", "objective"),
    (r"(?i)recommend\s+(?:a\s+)?(?:course\s+of\s+)?action\s+for\s+(.+)", "objective"),
    (r"(?i)analyze\s+(?:the\s+)?decision\s+(?:to|about)\s+(.+)", "objective"),
    # Impact analysis
    (r"(?i)if\s+we\s+(.+),\s+(?:what|how).+\??", "objective"),
    (r"(?i)(?:predict|forecast)\s+(?:the\s+)?(?:outcome|impact)\s+of\s+(.+)", "objective"),
]


class ModeRouter:
    """
    Routes incoming requests to the appropriate operating mode.
    
    Uses a combination of:
    1. Explicit command detection
    2. Pattern matching
    3. (Optional) LLM-based classification for ambiguous cases
    """
    
    def __init__(self, use_llm_fallback: bool = False):
        """
        Initialize the router.
        
        Args:
            use_llm_fallback: Whether to use LLM for ambiguous cases
        """
        self.use_llm_fallback = use_llm_fallback
    
    def route(self, query: str) -> RouteDecision:
        """
        Determine the appropriate mode for a query.
        
        Args:
            query: User's input query
            
        Returns:
            RouteDecision with mode, confidence, and reason
        """
        query = query.strip()
        
        # Check for explicit command prefixes
        decision = self._check_explicit_commands(query)
        if decision:
            return decision
        
        # Check Mode 1 patterns first (more specific)
        decision = self._check_patterns(query, MODE1_PATTERNS, OperatingMode.MODE_1)
        if decision and decision.confidence > 0.7:
            return decision
        
        # Check Mode 2 patterns
        decision = self._check_patterns(query, MODE2_PATTERNS, OperatingMode.MODE_2)
        if decision and decision.confidence > 0.5:
            return decision
        
        # Default to Mode 2 (decision support) for ambiguous queries
        return RouteDecision(
            mode=OperatingMode.MODE_2,
            confidence=0.5,
            reason=RouteReason.DEFAULT,
            raw_query=query,
        )
    
    def _check_explicit_commands(self, query: str) -> Optional[RouteDecision]:
        """Check for explicit mode command prefixes."""
        lower = query.lower()
        
        # Mode 1 explicit commands
        if lower.startswith("/mode1") or lower.startswith("/worldmodel"):
            remaining = query.split(maxsplit=1)
            domain = remaining[1] if len(remaining) > 1 else None
            return RouteDecision(
                mode=OperatingMode.MODE_1,
                confidence=1.0,
                reason=RouteReason.EXPLICIT_COMMAND,
                extracted_domain=domain,
                raw_query=query,
            )
        
        # Mode 2 explicit commands
        if lower.startswith("/mode2") or lower.startswith("/decide"):
            remaining = query.split(maxsplit=1)
            objective = remaining[1] if len(remaining) > 1 else None
            return RouteDecision(
                mode=OperatingMode.MODE_2,
                confidence=1.0,
                reason=RouteReason.EXPLICIT_COMMAND,
                extracted_objective=objective,
                raw_query=query,
            )
        
        return None
    
    def _check_patterns(
        self,
        query: str,
        patterns: list[tuple[str, Optional[str]]],
        mode: OperatingMode,
    ) -> Optional[RouteDecision]:
        """Check query against pattern list."""
        for pattern, capture_type in patterns:
            match = re.search(pattern, query)
            if match:
                extracted = match.group(1) if match.groups() else None
                return RouteDecision(
                    mode=mode,
                    confidence=0.8,
                    reason=RouteReason.PATTERN_MATCH,
                    extracted_domain=extracted if capture_type == "domain" else None,
                    extracted_objective=extracted if capture_type == "objective" else None,
                    raw_query=query,
                )
        return None
    
    def classify_with_llm(self, query: str) -> RouteDecision:
        """
        Use LLM to classify ambiguous queries.
        
        Note: This is a placeholder - actual implementation would call the LLM.
        """
        # For now, return default Mode 2 decision
        return RouteDecision(
            mode=OperatingMode.MODE_2,
            confidence=0.6,
            reason=RouteReason.LLM_CLASSIFICATION,
            raw_query=query,
        )
    
    def explain_route(self, decision: RouteDecision) -> str:
        """Generate human-readable explanation of routing decision."""
        mode_name = "World Model Construction" if decision.mode == OperatingMode.MODE_1 else "Decision Support"
        
        explanations = {
            RouteReason.EXPLICIT_COMMAND: f"Routing to {mode_name} (explicit command)",
            RouteReason.PATTERN_MATCH: f"Routing to {mode_name} (pattern match, {decision.confidence:.0%} confidence)",
            RouteReason.LLM_CLASSIFICATION: f"Routing to {mode_name} (LLM classification, {decision.confidence:.0%} confidence)",
            RouteReason.DEFAULT: f"Routing to {mode_name} (default)",
        }
        return explanations.get(decision.reason, f"Routing to {mode_name}")
