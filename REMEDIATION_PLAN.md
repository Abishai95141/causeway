# Causeway — Remediation & Implementation Plan

**Date:** 2026-02-07  
**Scope:** Five critical issues in the Causeway Agentic Decision Support System  
**Author:** Architecture Review

---

## Executive Summary

After a full audit of the codebase, all five reported issues have been root-caused. They range from a simple ID-format mismatch (Issue 1) to a missing architectural layer (Issue 4). This document provides the exact root cause, the affected files, and a step-by-step technical specification for each fix.

---

## Issue 1 — Ingestion Critical Failure: "File not found" after upload

### Root Cause

There is a **Document ID format mismatch** between the upload and indexing endpoints.

| Step | ID Used | Example |
|---|---|---|
| **Upload** (`POST /uploads`) | `doc_id = f"doc_{internal_id.hex[:12]}"` | `doc_abc123def456` |
| **ObjectStore.upload_bytes** | Receives `internal_id` (a raw `UUID`) | Object stored at `uploads/<full-UUID>.pdf` |
| **Index** (`POST /index/{doc_id}`) | Receives the `doc_` prefixed string | Searches `uploads/doc_abc123def456` |

The upload endpoint passes the raw `UUID` (`internal_id`) to `ObjectStore.upload_bytes`, which generates the S3 object key as `uploads/{uuid}.ext`. But the API returns `doc_id = "doc_{hex[:12]}"` to the user. When the user later calls `/index/{doc_id}`, the route does `store.list_files(prefix=f"uploads/{doc_id}")` — searching for `uploads/doc_abc123def456` — which will **never** match the actual key `uploads/a1b2c3d4-...`.

There is also a secondary bug: the `storage_uri` construction logic in the `index_document` route contains a variable-scoping issue (`object_name` may not be defined in the `else` branch), and the `if not filename` / `else` path logic is fragile.

### Affected Files

- `src/api/routes.py` — `upload_document()` (lines ~175–215) and `index_document()` (lines ~235–295)
- `src/storage/object_store.py` — `_generate_object_name()` (line 77)

### Fix Specification

**Step 1.1: Normalize the ID flow.** Use a single, consistent ID throughout the system. The simplest approach: use the `doc_` prefixed ID as the canonical ID everywhere, including as the key passed to `ObjectStore`.

```
# In upload_document():
doc_id = f"doc_{internal_id.hex[:12]}"

# Pass doc_id (string) to ObjectStore, NOT internal_id (UUID)
storage_uri = store.upload_bytes(
    doc_id=doc_id,          # <-- was: internal_id (UUID)
    filename=filename,
    content=content,
    content_type=...
)
```

**Step 1.2: Update `ObjectStore` type signature.** Change `upload_file` and `upload_bytes` to accept `doc_id: str` instead of `doc_id: UUID`. Update `_generate_object_name` accordingly:

```python
def _generate_object_name(self, doc_id: str, filename: str) -> str:
    ext = filename.rsplit(".", 1)[-1] if "." in filename else ""
    return f"uploads/{doc_id}.{ext}" if ext else f"uploads/{doc_id}"
```

**Step 1.3: Introduce a document metadata registry.** The current codebase has no persistent mapping of `doc_id → storage_uri, filename, content_type`. This means `index_document` has to guess the storage URI. Add an in-memory registry (to be replaced with PostgreSQL later):

```python
# In routes.py — module-level
_document_registry: dict[str, DocumentResponse] = {}

# In upload_document(): store the metadata
_document_registry[doc_id] = DocumentResponse(...)

# In index_document(): look up directly
if doc_id not in _document_registry:
    raise HTTPException(404, "Document not registered")
meta = _document_registry[doc_id]
content = store.download_file(meta.storage_uri)
```

**Step 1.4: Simplify `index_document` route.** Remove the brittle `list_files` + `object_name` scoping logic and replace with a direct lookup from the registry + `download_file(storage_uri)`.

---

## Issue 2 — PDF Processing Gap: Binary bytes passed to text embedder

### Root Cause

In `src/retrieval/router.py`, the `ingest_document()` method handles all content types identically:

```python
# Line ~246 in router.py
text_content = content.decode("utf-8", errors="ignore")
```

For PDF files, `content` is raw binary (the PDF byte-stream). Calling `.decode("utf-8", errors="ignore")` on a PDF produces garbage (stripped binary characters), not the document's text. This garbage is then passed to the Haystack embedder, which either fails or produces useless embeddings.

The system has **no text extraction layer** for binary formats (PDF, XLSX).

### Affected Files

- `src/retrieval/router.py` — `ingest_document()` (lines ~228–261)
- `src/haystack_svc/pipeline.py` — `add_document()` is text-only by design
- `pyproject.toml` — `pypdf` is listed as a dependency but never imported anywhere

### Fix Specification

**Step 2.1: Create a dedicated document extraction module** at `src/extraction/` with a `DocumentExtractor` class.

```
src/extraction/
    __init__.py
    extractor.py
```

**Step 2.2: Implement `DocumentExtractor`.**

```python
# src/extraction/extractor.py

import io
from pypdf import PdfReader

class DocumentExtractor:
    """Extracts text from binary document formats."""

    def extract(self, content: bytes, content_type: str, filename: str = "") -> str:
        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

        if content_type == "application/pdf" or ext == "pdf":
            return self._extract_pdf(content)
        elif content_type == "text/markdown" or ext == "md":
            return content.decode("utf-8", errors="replace")
        elif ext == "xlsx":
            return self._extract_xlsx(content)
        else:
            # text/plain fallback
            return content.decode("utf-8", errors="replace")

    def _extract_pdf(self, content: bytes) -> str:
        reader = PdfReader(io.BytesIO(content))
        pages = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            pages.append(f"[Page {i+1}]\n{text}")
        return "\n\n".join(pages)

    def _extract_xlsx(self, content: bytes) -> str:
        from openpyxl import load_workbook
        wb = load_workbook(io.BytesIO(content), read_only=True, data_only=True)
        sheets = []
        for ws in wb.worksheets:
            rows = []
            for row in ws.iter_rows(values_only=True):
                rows.append("\t".join(str(c) if c is not None else "" for c in row))
            sheets.append(f"[Sheet: {ws.title}]\n" + "\n".join(rows))
        return "\n\n".join(sheets)
```

**Step 2.3: Integrate into `RetrievalRouter.ingest_document`.**

Replace the raw `.decode()` call:

```python
async def ingest_document(self, doc_id, filename, content, content_type):
    from src.extraction.extractor import DocumentExtractor
    extractor = DocumentExtractor()
    text_content = extractor.extract(content, content_type, filename)

    pi_doc_id = await self.register_document_pageindex(...)
    hs_chunk_ids = await self.index_document_haystack(
        doc_id=doc_id, content=text_content, filename=filename,
    )
    return {"pageindex_doc_id": pi_doc_id, "haystack_chunk_ids": hs_chunk_ids}
```

**Step 2.4 (Future enhancement):** For production-quality PDF extraction (scanned documents, OCR), integrate Haystack's own converter pipeline using `PyPDFToDocument` + `FileTypeRouter`:

```python
from haystack.components.converters import PyPDFToDocument, TextFileToDocument, MarkdownToDocument
from haystack.components.routers import FileTypeRouter
```

This would replace the custom extractor with Haystack's first-party pipeline but requires writing the file to disk (or using `ByteStream`), which is a larger refactor.

---

## Issue 3 — Dependency Deviation: Regex-based DAG vs. PyWhy-LLM

### Root Cause

The design spec calls for `pywhyllm` for causal graph generation, but the current `Mode1WorldModelConstruction._draft_dag()` method uses a free-text LLM prompt and regex/JSON parsing to extract edges:

```python
# mode1.py, _draft_dag()
response = await self.llm.generate(prompt)
edges = self._parse_edges(response.content)    # regex + json.loads
```

This approach is:
1. **Fragile** — JSON extraction from free-text LLM output fails on formatting variations.
2. **Unvalidated** — No expert voting, no confounder detection, no domain expertise suggestions.
3. **Deviates from spec** — `pywhyllm` provides structured, multi-expert DAG construction with pairwise relationship voting.

### Affected Files

- `src/modes/mode1.py` — `_draft_dag()`, `_parse_edges()`, and `DAG_DRAFTING_PROMPT`
- `pyproject.toml` — `pywhyllm` is commented out in dependencies

### Fix Specification

**Step 3.1: Uncomment and install `pywhyllm`.**

```toml
# pyproject.toml
"pywhyllm>=0.1.0",
```

**Step 3.2: Create a `PyWhyLLM` integration service.**

```
src/causal/
    pywhyllm_bridge.py   # <-- new file
```

```python
# src/causal/pywhyllm_bridge.py

from pywhyllm import ModelSuggester, RelationshipStrategy, ValidationSuggester

class CausalGraphBuilder:
    """Bridges PyWhy-LLM into the Causeway DAG construction pipeline."""

    def __init__(self, llm_model: str = "gemini-2.0-flash"):
        self.modeler = ModelSuggester(llm_model)
        self.validator = ValidationSuggester(llm_model)

    def suggest_dag(
        self,
        treatment: str,
        outcome: str,
        all_factors: list[str],
    ) -> dict[tuple[str, str], int]:
        """
        Use pywhyllm to generate a voted DAG.

        Returns:
            Dict mapping (cause, effect) edge tuples to vote counts.
        """
        # 1. Get domain expertise suggestions
        domain_expertises = self.modeler.suggest_domain_expertises(all_factors)

        # 2. Suggest full pairwise DAG with expert voting
        suggested_dag = self.modeler.suggest_relationships(
            treatment, outcome, all_factors, domain_expertises,
            RelationshipStrategy.Pairwise,
        )
        return suggested_dag

    def validate_dag(
        self,
        all_factors: list[str],
        dag: dict[tuple[str, str], int],
    ) -> list[str]:
        """Critique an existing DAG and return issues."""
        domain_expertises = self.modeler.suggest_domain_expertises(all_factors)
        return self.validator.critique_graph(
            all_factors, dag, domain_expertises,
            RelationshipStrategy.Pairwise,
        )

    def suggest_confounders(
        self,
        treatment: str,
        outcome: str,
        all_factors: list[str],
    ) -> list[str]:
        """Identify confounders between treatment and outcome."""
        domain_expertises = self.modeler.suggest_domain_expertises(all_factors)
        _, confounders = self.modeler.suggest_confounders(
            treatment, outcome, all_factors, domain_expertises,
        )
        return confounders
```

**Step 3.3: Refactor `Mode1._draft_dag()` to use the bridge.**

```python
async def _draft_dag(self, domain, variables, max_edges):
    from src.causal.pywhyllm_bridge import CausalGraphBuilder

    builder = CausalGraphBuilder()
    factor_names = [v.name.lower().replace(" ", "_") for v in variables]

    # Pick the first and last variables as treatment/outcome heuristic,
    # or use domain-specific logic
    treatment = factor_names[0] if factor_names else "unknown"
    outcome = factor_names[-1] if factor_names else "unknown"

    voted_dag = builder.suggest_dag(treatment, outcome, factor_names)

    edges = []
    for (src, tgt), votes in voted_dag.items():
        if votes >= 2:  # Minimum 2 expert votes
            strength = EvidenceStrength.STRONG if votes >= 3 else EvidenceStrength.MODERATE
            edges.append(EdgeCandidate(
                from_var=src, to_var=tgt,
                mechanism=f"Suggested by {votes} domain experts via PyWhy-LLM",
                strength=strength,
            ))

    self._edge_candidates = edges[:max_edges]
    return self._edge_candidates
```

**Step 3.4: Keep the old regex parser as a fallback.** Rename `_draft_dag` → `_draft_dag_llm_fallback` for use when `pywhyllm` is unavailable or the factor list is too small for pairwise analysis:

```python
async def _draft_dag(self, domain, variables, max_edges):
    try:
        return await self._draft_dag_pywhyllm(domain, variables, max_edges)
    except Exception:
        return await self._draft_dag_llm_fallback(domain, variables, max_edges)
```

---

## Issue 4 — Missing Training Loop: AgentOrchestrator ↔ SpanCollector disconnect

### Root Cause

The `src/training/` module is fully implemented with three components:
- `SpanCollector` — records execution spans (start/end/events)
- `TrajectoryStore` — persists full execution trajectories
- `DefaultRewardFunction` — computes reward signals

However, `AgentOrchestrator` has **zero references** to any of these. No spans are started, no trajectories are saved, and no rewards are computed. The orchestrator's `run()` loop and `_execute_tool()` method operate without any instrumentation.

### Affected Files

- `src/agent/orchestrator.py` — needs span instrumentation
- `src/training/spans.py` — already implemented, unused
- `src/training/trajectories.py` — already implemented, unused
- `src/training/rewards.py` — already implemented, unused

### Fix Specification

**Step 4.1: Inject training dependencies into `AgentOrchestrator.__init__`.**

```python
from src.training.spans import SpanCollector
from src.training.trajectories import TrajectoryStore, Trajectory
from src.training.rewards import DefaultRewardFunction, RewardFunction

class AgentOrchestrator:
    def __init__(
        self,
        llm_client=None,
        retrieval_router=None,
        causal_service=None,
        max_tool_calls=5,
        span_collector: Optional[SpanCollector] = None,
        trajectory_store: Optional[TrajectoryStore] = None,
        reward_function: Optional[RewardFunction] = None,
        enable_training: bool = True,
    ):
        # ... existing init ...
        self.span_collector = span_collector or SpanCollector(enabled=enable_training)
        self.trajectory_store = trajectory_store or TrajectoryStore()
        self.reward_function = reward_function or DefaultRewardFunction()
```

**Step 4.2: Instrument `run()` with span collection.**

```python
async def run(self, query, system_prompt=None):
    trace_id = f"orch_{uuid4().hex[:12]}"

    # === START TRACE ===
    self.span_collector.start_trace("orchestrator_run")
    run_span = self.span_collector.start_span(
        "run", attributes={"query": query[:200], "trace_id": trace_id}
    )

    # ... existing loop ...
    for iteration in range(self.max_tool_calls + 1):
        llm_span = self.span_collector.start_span(
            "llm_generate", attributes={"iteration": iteration}
        )
        response = await self.llm.generate_with_tools(...)
        self.span_collector.end_span(llm_span, attributes={
            "tokens": response.total_tokens,
            "has_tool_calls": bool(response.tool_calls),
        })

        if response.tool_calls:
            for tool_call in response.tool_calls:
                tool_span = self.span_collector.start_span(
                    f"tool_{tool_call.get('tool', 'unknown')}",
                    attributes={"arguments": str(tool_call.get('arguments', {}))[:200]},
                )
                result = await self._execute_tool(...)
                self.span_collector.end_span(tool_span, attributes={
                    "success": result.success,
                    "execution_time_ms": result.execution_time_ms,
                })

    # === END TRACE ===
    self.span_collector.end_span(run_span)

    # === SAVE TRAJECTORY ===
    trace_spans = self.span_collector.export_trace(
        self.span_collector._current_trace
    )
    trajectory = Trajectory(
        trajectory_id=f"traj_{uuid4().hex[:12]}",
        trace_id=trace_id,
        mode="orchestrator",
        input_data={"query": query, "system_prompt": system_prompt},
        spans=trace_spans,
        outcome={
            "success": result_obj.error is None,
            "evidence_count": len(tool_results),
            "total_tokens": total_tokens,
        },
    )

    # Compute reward
    reward_signal = self.reward_function.compute(
        trajectory.trajectory_id,
        [self.span_collector.get_span(s["span_id"])
         for s in trace_spans
         if self.span_collector.get_span(s["span_id"])],
        trajectory.outcome,
    )
    trajectory.reward = reward_signal.reward

    self.trajectory_store.save(trajectory)

    return result_obj
```

**Step 4.3: Expose training data via API.** Add a route in `routes.py`:

```python
@router.get("/training/trajectories")
async def list_trajectories(mode: str = "orchestrator", limit: int = 50):
    """List recent execution trajectories for training review."""
    ...
```

**Step 4.4: Wire Mode 1 and Mode 2 to also record trajectories.** Both `Mode1WorldModelConstruction.run()` and `Mode2DecisionSupport.run()` should accept an optional `SpanCollector` and record their stage transitions as spans.

---

## Issue 5 — Runtime API Errors: `unexpected keyword argument` in RetrievalRouter

### Root Cause

The `AgentOrchestrator._handle_search_evidence()` method calls:

```python
bundles = await self.retrieval.retrieve_simple(query, max_results)
```

Inspecting `RetrievalRouter.retrieve_simple()`:

```python
async def retrieve_simple(self, query: str, max_results: int = 5):
```

This signature matches — **two positional args**. So this specific call is fine.

However, the error surfaces when the **LLM tool-calling machinery** invokes `_handle_search_evidence` via `_execute_tool`:

```python
result = await handler(**arguments)
```

The LLM returns tool arguments as a dict like `{"query": "...", "max_results": 5}`. This works because `_handle_search_evidence` accepts `query` and `max_results`.

**The real mismatch** is in `_handle_search_evidence` itself when it accesses the returned `EvidenceBundle` fields:

```python
"source": bundle.source.doc_title,     # 'source' is a SourceReference
"location": {
    "page": bundle.location.page_number,
    "section": bundle.location.section_name,
},
```

But the primary keyword mismatch error is most likely from **Mode 1 / Mode 2 calling `retrieve_simple` with `top_k` instead of `max_results`**:

```python
# mode1.py line ~282
evidence = await self.retrieval.retrieve_simple(query, top_k=10)
                                                       ^^^^^^
# mode2.py line ~289
evidence = await self.retrieval.retrieve_simple(query, top_k=5)
                                                       ^^^^^
```

But `retrieve_simple` signature is:
```python
async def retrieve_simple(self, query: str, max_results: int = 5):
```

**`top_k` is not a valid keyword argument.** This produces `TypeError: retrieve_simple() got an unexpected keyword argument 'top_k'`.

### Affected Files

- `src/modes/mode1.py` — `_discover_variables()`, `_gather_evidence()`, `_triangulate_evidence()` — all use `top_k=`
- `src/modes/mode2.py` — `_refresh_evidence()` — uses `top_k=`
- `src/retrieval/router.py` — `retrieve_simple()` expects `max_results=`

### Fix Specification

**Step 5.1: Rename all `top_k=` calls to `max_results=`** in mode1.py and mode2.py.

```python
# mode1.py _discover_variables:
evidence = await self.retrieval.retrieve_simple(query, max_results=10)

# mode1.py _gather_evidence:
bundles = await self.retrieval.retrieve_simple(query, max_results=3)

# mode1.py _triangulate_evidence:
bundles = await self.retrieval.retrieve_simple(query, max_results=2)

# mode2.py _refresh_evidence:
evidence = await self.retrieval.retrieve_simple(query, max_results=5)
```

**Step 5.2 (Alternative — backward-compatible):** Add `top_k` as an alias in `retrieve_simple`:

```python
async def retrieve_simple(
    self,
    query: str,
    max_results: int = 5,
    top_k: Optional[int] = None,  # backward-compatible alias
) -> list[EvidenceBundle]:
    effective_limit = top_k if top_k is not None else max_results
    request = RetrievalRequest(query=query, max_results=effective_limit, ...)
    return await self.retrieve(request)
```

**Step 5.1 is preferred** (fix the callers) — aliasing hides the real inconsistency.

---

## Implementation Roadmap

| Phase | Issue | Effort | Priority | Dependency |
|---|---|---|---|---|
| **Phase 1** | Issue 5 — Parameter mismatch (`top_k` → `max_results`) | 15 min | P0 (blocks all runtime) | None |
| **Phase 2** | Issue 1 — Ingestion ID mismatch + registry | 1 hr | P0 (blocks ingestion) | None |
| **Phase 3** | Issue 2 — PDF text extraction layer | 2 hr | P0 (blocks PDF ingestion) | Phase 2 |
| **Phase 4** | Issue 4 — Training loop integration | 3 hr | P1 (feature gap) | None |
| **Phase 5** | Issue 3 — PyWhy-LLM integration | 4 hr | P1 (spec compliance) | None |

### Phase 1 — Quick Fix (15 min)
1. `mode1.py`: Replace all `top_k=N` with `max_results=N` (3 call sites)
2. `mode2.py`: Replace `top_k=5` with `max_results=5` (1 call site)
3. Run tests to confirm no more `TypeError`

### Phase 2 — Ingestion Fix (1 hr)
1. Change `ObjectStore.upload_file` / `upload_bytes` to accept `doc_id: str`
2. Normalize `upload_document` route to pass `doc_id` string consistently
3. Add `_document_registry` dict for metadata lookup
4. Rewrite `index_document` route to use registry lookup
5. Verify end-to-end: upload → index → retrieve

### Phase 3 — PDF Extraction (2 hr)
1. Create `src/extraction/extractor.py` with `DocumentExtractor`
2. Implement `_extract_pdf` (using `pypdf`), `_extract_xlsx` (using `openpyxl`)
3. Integrate into `RetrievalRouter.ingest_document`
4. Add unit tests for each format
5. Test with real PDF upload end-to-end

### Phase 4 — Training Loop (3 hr)
1. Add `SpanCollector`, `TrajectoryStore`, `DefaultRewardFunction` to `AgentOrchestrator.__init__`
2. Instrument `run()` and `_execute_tool()` with span start/end calls
3. Save `Trajectory` at end of each run
4. Compute reward via `DefaultRewardFunction`
5. Add `/training/trajectories` API endpoint
6. Optionally instrument Mode 1 and Mode 2 stage transitions

### Phase 5 — PyWhy-LLM (4 hr)
1. Uncomment `pywhyllm` in `pyproject.toml`, install
2. Create `src/causal/pywhyllm_bridge.py`
3. Refactor `Mode1._draft_dag` to use `CausalGraphBuilder`
4. Keep old regex parser as `_draft_dag_llm_fallback`
5. Add `validate_dag` step between DAG drafting and evidence triangulation
6. Test with variable lists of varying sizes

---

## Appendix A — Library API Reference (from audit)

### PyWhy-LLM Key APIs
```
ModelSuggester(model)
  .suggest_domain_expertises(factors) → List[str]
  .suggest_relationships(treatment, outcome, factors, experts, strategy) → Dict[Tuple, int]
  .suggest_confounders(treatment, outcome, factors, experts) → (Dict, List)

ValidationSuggester(model)
  .critique_graph(factors, dag, experts, strategy) → List[str]

RelationshipStrategy.Pairwise  # Recommended for full DAG
```

### Haystack 2.x Indexing Pipeline (for future Phase 3 enhancement)
```
FileTypeRouter → PyPDFToDocument / TextFileToDocument / MarkdownToDocument
    → DocumentJoiner → DocumentCleaner → DocumentSplitter
    → SentenceTransformersDocumentEmbedder → DocumentWriter(QdrantDocumentStore)
```

### PageIndex
```
# Local (open-source)
from pageindex import page_index
tree = page_index("report.pdf", model="gpt-4o-2024-11-20", ...)

# Cloud API
from pageindex import PageIndexClient
client = PageIndexClient(api_key=...)
doc_id = client.submit_document("report.pdf")["doc_id"]
tree = client.get_tree(doc_id)
```
