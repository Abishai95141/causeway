"""
Agentic Verification Loop

Proposer-Retriever-Judge pipeline that grounds every drafted causal
edge in retrieved evidence before it enters the final DAG.

Components:
    grounding_retriever — targeted evidence retrieval per edge
    judge               — LLM-based verification verdicts
    loop                — multi-turn orchestration with dedup & pruning
"""

from src.verification.judge import (
    AdversarialVerdict,
    SupportType,
    VerificationJudge,
    VerificationVerdict,
)
from src.verification.grounding_retriever import GroundingRetriever
from src.verification.loop import VerificationAgent, VerificationResult

__all__ = [
    "AdversarialVerdict",
    "GroundingRetriever",
    "SupportType",
    "VerificationAgent",
    "VerificationJudge",
    "VerificationResult",
    "VerificationVerdict",
]
