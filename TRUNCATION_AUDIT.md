# Causeway Evidence Truncation Audit

**Date:** 2026-02-16  
**Auditor:** AI Systems Architect  
**Severity:** CRITICAL — LLM causal reasoning is being poisoned by mid-word evidence truncation  

---

## Executive Summary

The LLM generates causal graphs with mechanism text cut off mid-word (e.g., `"...initiatives in tec..."`). This audit traced the full evidence lifecycle from retrieval to LLM context and identified **13 destructive truncation sites** across 7 source files, plus a blind context-window trimming policy that silently discards evidence.

All 13 sites used raw Python string slicing (`content[:N]`) which cuts at arbitrary character boundaries with zero awareness of word or sentence structure. The fix replaces every instance with a new `truncate_at_sentence_boundary()` utility.

---

## 1. Data Flow Trace

```
 ┌─────────────────────────────┐
 │  Haystack / PageIndex       │  Evidence retrieved as EvidenceBundle.content
 │  (No truncation — CLEAN)    │  Content is intact at this layer.
 └────────────┬────────────────┘
              │
 ┌────────────▼────────────────┐
 │  RetrievalRouter            │  Routes to Haystack, PageIndex, or Hybrid.
 │  (No truncation — CLEAN)    │  Passes bundles through unchanged.
 └────────────┬────────────────┘
              │
              ├──────────────────────────────────────────────┐
              │                                              │
 ┌────────────▼────────────────┐          ┌──────────────────▼──────────────┐
 │  AgentOrchestrator          │          │  Mode 1 / Mode 2 Pipelines     │
 │  _handle_search_evidence()  │          │  _discover_variables()         │
 │  ⚠ content[:500]            │          │  ⚠ e.content[:500]             │
 │  ⚠ content[:100] + "..."    │          │  _draft_dag()                  │
 │                              │          │  ⚠ eb.content[:500]            │
 │  PageIndex tools             │          │  _synthesize_recommendation()  │
 │  ⚠ section.content[:500]    │          │  ⚠ e.content[:400]             │
 └────────────┬────────────────┘          └──────────┬─────────────────────┘
              │                                      │
              │                            ┌─────────▼─────────────────┐
              │                            │  PyWhyLLM Bridge          │
              │                            │  extract_mechanism_from_  │
              │                            │  evidence()               │
              │                            │  ⚠ content[:200] ← ROOT  │
              │                            │  CAUSE of "...in tec..."  │
              │                            └─────────┬─────────────────┘
              │                                      │
 ┌────────────▼──────────────────────────────────────▼──┐
 │  ContextManager                                       │
 │  _trim_context(): blind FIFO message deletion         │
 │  ⚠ Silently discards oldest messages, no summary     │
 └───────────────────────────────────────────────────────┘
              │
 ┌────────────▼────────────────┐
 │  LLM (Gemini)               │  Receives truncated, incomplete evidence
 │  Generates broken mechanism │  → "...initiatives in tec..."
 └─────────────────────────────┘
```

---

## 2. Root Cause Identification — All Truncation Sites

### CRITICAL severity (directly cause broken mechanism text)

| # | File | Line | Code | Impact |
|---|------|------|------|--------|
| 1 | `src/causal/pywhyllm_bridge.py` | 521 | `evidence_bundles[0].content[:200]` | **PRIMARY ROOT CAUSE.** Mechanism snippet is hard-sliced at 200 chars. This is the exact line that produces `"...initiatives in tec..."`. The 200-char budget is far too small for meaningful causal explanations. |
| 2 | `src/agent/orchestrator.py` | 378 | `bundle.content[:500]` | Evidence returned to LLM from `search_evidence` tool is cut at 500 chars — mid-word. |
| 3 | `src/agent/orchestrator.py` | 389 | `bundle.content[:100] + "..."` | Evidence tracking summary is only 100 chars — context manager's evidence index is useless. |
| 4 | `src/pageindex/pageindex_tools.py` | 31 | `section.content[:500]` | PageIndex section content cut at 500 chars before LLM sees it. |

### HIGH severity (degrade LLM input quality for variable/edge extraction)

| # | File | Line | Code | Impact |
|---|------|------|------|--------|
| 5 | `src/modes/mode1.py` | 450 | `e.content[:500]` | Variable discovery evidence truncated. LLM misses variables described later in chunks. |
| 6 | `src/modes/mode1.py` | 471 | `e.content[:500]` | Broader search evidence also truncated. Same problem. |
| 7 | `src/modes/mode1.py` | 614 | `eb.content[:500]` | DAG drafting evidence for LangExtract — edges missed because evidence is incomplete. |
| 8 | `src/modes/mode2.py` | 423 | `e.content[:400]` | Recommendation synthesis evidence is even shorter (400 chars). |

### MODERATE severity (affect conflict detection & API)

| # | File | Line | Code | Impact |
|---|------|------|------|--------|
| 9  | `src/api/routes.py` | 289 | `b.content[:1000]` | API search results truncated — downstream consumers may re-process truncated text. |
| 10 | `src/causal/conflict_resolver.py` | 257 | `bundle.content[:200]` | Reversal evidence too short to be useful in conflict reports. |
| 11 | `src/causal/conflict_resolver.py` | 259 | `bundle.content[:200]` | Contradicting evidence also 200 chars. |
| 12 | `src/causal/conflict_resolver.py` | 261 | `bundle.content[:200]` | Supporting evidence also 200 chars. |
| 13 | `src/causal/conflict_resolver.py` | 345 | `b.content[:200]` | Missing-variable evidence mentions sliced at 200 chars. |

### HIGH severity (context window management)

| # | File | Method | Impact |
|---|------|--------|--------|
| 14 | `src/agent/context_manager.py` | `_trim_context()` | When token budget exceeds 80%, oldest messages are **silently deleted** via `_messages.pop(0)`. No summarization. Evidence gathered in early tool calls is permanently lost before the LLM can use it for DAG construction. |

---

## 3. Why This Matters for Causal Reasoning

The Causeway system follows a multi-stage pipeline:

1. **Retrieve** evidence chunks (intact)
2. **Feed** evidence to LLM for variable discovery, edge extraction, mechanism description
3. **Store** mechanism text in the causal DAG permanently

When step 2 receives `"Company's strategic initiatives in tec"` instead of `"Company's strategic initiatives in technology modernization drive operational efficiency"`, the LLM:
- Cannot understand the causal mechanism
- Generates vague or hallucinated mechanisms
- The DAG stores garbage mechanism descriptions permanently
- Mode 2 decision support reasons over broken mechanisms → bad recommendations

---

## 4. Fixes Applied

### 4.1 New Utility Module: `src/utils/text.py`

Created `truncate_at_sentence_boundary()` with the following behavior:
- If text fits within `max_chars`, returns it unchanged
- Otherwise, finds the **last sentence boundary** (`.`, `!`, `?`) before the limit
- Falls back to the **last word boundary** (space) if no sentence ending exists
- Appends a configurable suffix (default `" [...]"`) so consumers know it was shortened
- Never returns empty — at minimum returns the first word + suffix

Two convenience wrappers:
- `truncate_evidence(content, max_chars=800)` — for evidence passed to LLM prompts
- `truncate_for_context_tracking(content, max_chars=200)` — for the context manager's lightweight evidence index

### 4.2 Files Modified

| File | Change |
|------|--------|
| `src/agent/orchestrator.py` | `content[:500]` → `truncate_evidence(content, 800)` ; `content[:100]+"..."` → `truncate_for_context_tracking(content, 200)` |
| `src/pageindex/pageindex_tools.py` | `section.content[:500]` → `truncate_evidence(section.content, 800)` |
| `src/causal/pywhyllm_bridge.py` | `content[:200]` → `truncate_at_sentence_boundary(content, 400, suffix="")` — doubled budget + sentence-safe |
| `src/modes/mode1.py` | All three `e.content[:500]` → `truncate_evidence(e.content, 800)` |
| `src/modes/mode2.py` | `e.content[:400]` → `truncate_evidence(e.content, 800)` |
| `src/api/routes.py` | `b.content[:1000]` → `truncate_evidence(b.content, 1200)` |
| `src/causal/conflict_resolver.py` | All four `content[:200]` → `truncate_at_sentence_boundary(content, 300)` |
| `src/agent/context_manager.py` | `_trim_context()` now compresses tool messages at sentence boundaries before deleting; only removes messages as last resort |

### 4.3 Budget Adjustments

Evidence char limits were also increased to provide more context to the LLM:

| Location | Before | After | Rationale |
|----------|--------|-------|-----------|
| Orchestrator tool results | 500 | 800 | A single Haystack chunk is ~3 sentences (~400-600 chars). 800 comfortably fits 1-2 chunks. |
| Mechanism extraction | 200 | 400 | Mechanisms need enough text to describe the causal pathway. 200 was cutting mid-sentence constantly. |
| Mode 1/2 evidence text | 400-500 | 800 | Matches orchestrator budget. Consistent across the pipeline. |
| Conflict resolver | 200 | 300 | Conflict descriptions benefit from slightly more context. |
| API search results | 1000 | 1200 | Generous for API consumers; now sentence-safe. |

---

## 5. Verification

After all fixes:
- `grep -rn '\.content\[:[0-9]' src/` → **0 matches** (all raw slicing eliminated)
- Pylance/mypy: **0 errors** across all 9 modified files
- The `truncate_at_sentence_boundary()` function guarantees no mid-word cuts

---

## 6. Recommendations for Future Work

1. **Token-aware truncation**: Replace character-based limits with `tiktoken` token counting for precise LLM context budget management.
2. **Chunk size tuning**: The Haystack pipeline splits at 3 sentences per chunk (`split_length=3`). If average chunk size grows, the 800-char budget may need adjustment.
3. **Streaming summarization**: For very long multi-tool conversations, implement an LLM-based summarization step in `_trim_context()` rather than the current sentence-boundary compression.
4. **Evidence importance scoring**: When trimming context, prioritize evidence by relevance score rather than FIFO age.
5. **Test coverage**: Add unit tests for `truncate_at_sentence_boundary()` with edge cases (empty strings, strings with no punctuation, very long single sentences, unicode).
