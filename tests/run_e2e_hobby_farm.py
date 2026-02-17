#!/usr/bin/env python3
"""
End-to-End Integration Test — Hobby Farm + Verification Loop

Runs the FULL causeway pipeline against the real Hobby Farm PDF:
  1. Document ingestion (PDF → Qdrant via Haystack)
  2. Mode 1: Variable Discovery → Evidence Gathering → DAG Drafting
             → Agentic Verification Loop → DAG Assembly
  3. Mode 2: Decision Support query on the built world model

This script uses the REAL Gemini API and real infrastructure.
It produces a detailed report at the end.

Usage:
    python tests/run_e2e_hobby_farm.py
"""

from __future__ import annotations

import asyncio
import json
import logging
import pathlib
import sys
import time
from datetime import datetime, timezone
from uuid import uuid4

# ── Setup logging ──────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
_logger = logging.getLogger("e2e_test")

# Reduce noise from verbose libraries
for noisy in ("httpx", "httpcore", "urllib3", "qdrant_client",
               "haystack", "filelock", "huggingface_hub"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

# ── Paths ──────────────────────────────────────────────────────────────
REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
PDF_PATH = REPO_ROOT / "The Profitable Hobby Farm - 2010 - Aubrey - Sample Business Documents.pdf"
sys.path.insert(0, str(REPO_ROOT))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def banner(msg: str) -> None:
    sep = "═" * 70
    _logger.info("\n%s\n  %s\n%s", sep, msg, sep)


def elapsed(start: float) -> str:
    return f"{time.time() - start:.1f}s"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Stage 1: Document Ingestion
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

async def ingest_pdf(router) -> str:
    """Ingest the hobby farm PDF and return the doc_id."""
    banner("STAGE 1: Document Ingestion")
    t0 = time.time()

    doc_id = f"e2e_full_{uuid4().hex[:8]}"
    pdf_bytes = PDF_PATH.read_bytes()
    _logger.info("PDF size: %d bytes", len(pdf_bytes))

    result = await router.ingest_document(
        doc_id=doc_id,
        filename="hobby_farm.pdf",
        content=pdf_bytes,
        content_type="application/pdf",
    )

    chunk_count = len(result.get("haystack_chunk_ids", []))
    _logger.info("Ingested as doc_id=%s  chunks=%d  [%s]", doc_id, chunk_count, elapsed(t0))
    return doc_id


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Stage 2: Mode 1 — World Model Construction
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

async def run_mode1(router, doc_id: str) -> dict:
    """Run full Mode 1 pipeline and return a summary dict."""
    banner("STAGE 2: Mode 1 — World Model Construction + Verification")
    t0 = time.time()

    from src.agent.llm_client import LLMClient
    from src.causal.service import CausalService
    from src.modes.mode1 import Mode1WorldModelConstruction

    llm = LLMClient()
    await llm.initialize()
    _logger.info("LLM client initialised  (mock=%s)", llm.is_mock_mode)

    svc = CausalService()

    mode1 = Mode1WorldModelConstruction(
        llm_client=llm,
        retrieval_router=router,
        causal_service=svc,
    )
    await mode1.initialize()

    _logger.info("Starting Mode 1 run with doc_ids=[%s]", doc_id)
    result = await mode1.run(
        domain="hobby_farm_e2e",
        initial_query="herb business profitability, revenue, costs, marketing, and growth factors",
        max_variables=12,
        max_edges=20,
        doc_ids=[doc_id],
    )

    # ── Report ─────────────────────────────────────────────────────────
    _logger.info("Mode 1 complete in %s", elapsed(t0))
    _logger.info("  trace_id:      %s", result.trace_id)
    _logger.info("  stage:         %s", result.stage.value)
    _logger.info("  variables:     %d", result.variables_discovered)
    _logger.info("  edges created: %d", result.edges_created)
    _logger.info("  evidence:      %d", result.evidence_linked)
    _logger.info("  conflicts:     %d (critical: %d)", result.conflicts_detected, result.critical_conflicts)

    if result.error:
        _logger.error("  ERROR: %s", result.error)

    # Dump audit trail
    for entry in result.audit_entries:
        _logger.info("  audit: %-35s %s", entry.action, json.dumps(entry.data, default=str)[:200])

    summary = {
        "trace_id": result.trace_id,
        "stage": result.stage.value,
        "variables_discovered": result.variables_discovered,
        "edges_created": result.edges_created,
        "evidence_linked": result.evidence_linked,
        "error": result.error,
        "elapsed": elapsed(t0),
        "conflicts": result.conflicts_detected,
        "causal_service": svc,
        "domain": "hobby_farm_e2e",
    }

    # ── Sanity checks ──────────────────────────────────────────────────
    if result.error:
        _logger.error("Mode 1 FAILED: %s", result.error)
    else:
        if result.variables_discovered == 0:
            _logger.warning("⚠  Zero variables discovered — LLM may have failed")
        if result.edges_created == 0:
            _logger.warning("⚠  Zero edges in DAG — judge may have pruned everything")
        elif result.edges_created < 3:
            _logger.warning("⚠  Only %d edges — judge may be too strict", result.edges_created)
        else:
            _logger.info("✓  Mode 1 produced a viable world model")

    return summary


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Stage 3: Mode 2 — Decision Support
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

async def run_mode2(router, svc, domain: str) -> dict:
    """Run Mode 2 decision support query."""
    banner("STAGE 3: Mode 2 — Decision Support Query")
    t0 = time.time()

    from src.agent.llm_client import LLMClient
    from src.modes.mode2 import Mode2DecisionSupport

    llm = LLMClient()
    await llm.initialize()

    mode2 = Mode2DecisionSupport(
        llm_client=llm,
        retrieval_router=router,
        causal_service=svc,
    )

    query = "Should we increase our marketing budget for farmers markets to grow revenue?"

    _logger.info("Mode 2 query: %s", query)
    result = await mode2.run(
        query=query,
        domain_hint=domain,
    )

    _logger.info("Mode 2 complete in %s", elapsed(t0))
    _logger.info("  stage:             %s", result.stage.value)
    _logger.info("  model_used:        %s", result.model_used)
    _logger.info("  evidence_count:    %d", result.evidence_count)
    _logger.info("  escalate_to_mode1: %s", result.escalate_to_mode1)

    if result.recommendation:
        rec = result.recommendation
        _logger.info("  recommendation:    %s", rec.recommendation[:200] if rec.recommendation else "N/A")
        _logger.info("  confidence:        %s", rec.confidence)
        _logger.info("  actions:           %s", rec.actions[:3] if rec.actions else [])
        _logger.info("  risks:             %s", rec.risks[:3] if rec.risks else [])
    elif result.escalation_reason:
        _logger.warning("  escalation:        %s", result.escalation_reason)

    if result.error:
        _logger.error("  MODE 2 ERROR:      %s", result.error)

    return {
        "stage": result.stage.value,
        "model_used": result.model_used,
        "evidence_count": result.evidence_count,
        "has_recommendation": result.recommendation is not None,
        "escalated": result.escalate_to_mode1,
        "error": result.error,
        "elapsed": elapsed(t0),
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Main
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

async def main() -> int:
    overall_start = time.time()
    banner("CAUSEWAY E2E TEST — Hobby Farm Business Document")

    # ── Pre-flight checks ──────────────────────────────────────────────
    if not PDF_PATH.exists():
        _logger.error("PDF not found at: %s", PDF_PATH)
        return 1

    # Check Qdrant
    try:
        import urllib.request
        urllib.request.urlopen("http://localhost:6333/readyz", timeout=2)
        _logger.info("Qdrant: OK")
    except Exception:
        _logger.error("Qdrant not available at localhost:6333")
        return 1

    # Check API key
    from src.config import get_settings
    settings = get_settings()
    if not settings.google_ai_api_key:
        _logger.error("No google_ai_api_key configured — set it in .env")
        return 1
    _logger.info("API key: configured (%s...)", settings.google_ai_api_key[:8])

    # ── Initialise retrieval ───────────────────────────────────────────
    from src.retrieval.router import RetrievalRouter
    router = RetrievalRouter()
    await router.initialize()
    _logger.info("RetrievalRouter initialised")

    # ── Run pipeline ───────────────────────────────────────────────────
    results = {}

    try:
        doc_id = await ingest_pdf(router)
        results["ingestion"] = {"doc_id": doc_id, "status": "ok"}
    except Exception as exc:
        _logger.error("Ingestion FAILED: %s", exc, exc_info=True)
        results["ingestion"] = {"status": "failed", "error": str(exc)}
        return 1

    try:
        m1 = await run_mode1(router, doc_id)
        results["mode1"] = m1
    except Exception as exc:
        _logger.error("Mode 1 FAILED: %s", exc, exc_info=True)
        results["mode1"] = {"status": "failed", "error": str(exc)}
        return 1

    # Only run Mode 2 if Mode 1 produced a model with edges
    if m1.get("edges_created", 0) > 0 and not m1.get("error"):
        try:
            m2 = await run_mode2(router, m1["causal_service"], m1["domain"])
            results["mode2"] = m2
        except Exception as exc:
            _logger.error("Mode 2 FAILED: %s", exc, exc_info=True)
            results["mode2"] = {"status": "failed", "error": str(exc)}
    else:
        _logger.warning("Skipping Mode 2 — Mode 1 produced no viable model")
        results["mode2"] = {"status": "skipped", "reason": "no edges from mode1"}

    # ── Final report ───────────────────────────────────────────────────
    banner("FINAL REPORT")
    total = elapsed(overall_start)
    _logger.info("Total elapsed: %s", total)

    # Filter out non-serialisable objects
    report = {}
    for k, v in results.items():
        if isinstance(v, dict):
            report[k] = {kk: vv for kk, vv in v.items() if not hasattr(vv, '__dict__') or isinstance(vv, str)}
        else:
            report[k] = v
    _logger.info("Results:\n%s", json.dumps(report, indent=2, default=str))

    # ── Pass / Fail ────────────────────────────────────────────────────
    m1_ok = (m1.get("edges_created", 0) > 0 and not m1.get("error"))
    m2_ok = results.get("mode2", {}).get("has_recommendation", False) or results.get("mode2", {}).get("status") == "skipped"

    if m1_ok:
        _logger.info("━━━ PASS: Full pipeline completed successfully ━━━")
        return 0
    else:
        _logger.error("━━━ FAIL: Pipeline did not produce a viable world model ━━━")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
