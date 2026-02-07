## **OVERALL SYSTEM INSTRUCTION FILE**

### **0\) Role and Non-Negotiables**

You are implementing the system described in these reference specs (treat as source of truth):

* `systemdoc.md`  
* `lowlevel-systemdoc.md`

Non-negotiables:

1. **No hallucination**: if something is not specified, implement the smallest sane default and record it in `DECISIONS.md`.  
2. **Sequential build**: implement modules **1 → 12** in order. Do not start a module until the previous module’s tests pass.  
3. **Strict contracts**: every module exposes explicit interfaces (Pydantic models \+ service protocols). No implicit coupling.  
4. **Prototype scope**: ONLY support **local user uploads** as data sources:  
   * PDFs, TXT, MD, XLSX (Excel)  
   * No live DB connectors or CRM APIs in this prototype (but design with a pluggable connector interface).  
5. **Auditability first**: every retrieval-backed claim must be attachable to an `EvidenceBundle` persisted in Postgres.

---

### **1\) What the End Result Must Do (Expected Whole-System Outcome)**

At the end, a developer can:

1. Open a simple input surface (minimal web UI or API-first UI) and **upload**: PDF/TXT/MD/XLSX.  
2. The system persists uploads to object storage, registers them, and indexes them so they are queryable by:  
   * **Haystack hybrid retrieval** (broad recall)  
   * **PageIndex navigation** (precise, structured citations)  
3. Run **Mode 1 (World Model Construction)** to produce a versioned `WorldModelVersion`:  
   * variable discovery  
   * evidence gathering (Haystack \+ PageIndex)  
   * causal drafting into DAG form (PyWhyLLM/NetworkX)  
   * evidence triangulation \+ edge strength labeling  
   * human review step → approve → “active”  
4. Run **Mode 2 (Decision Support)**:  
   * fetch best world model version  
   * freshness check  
   * evidence refresh (Haystack \+ PageIndex)  
   * causal reasoning \+ recommendation with cited evidence refs  
5. Every run writes:  
   * `audit_log` entry  
   * EvidenceBundles stored and linked to world model edges  
   * a trace ID usable for debugging and (later) Lightning training

---

### **2\) Mandatory Agent Log System (Context for the Agent)**

Maintain these files at repo root under `agent_state/` and update them **after every meaningful change** (new module, new endpoint, new schema, new dependency, bug fix):

* `agent_state/BUILD_LOG.md` (append-only journal)  
* `agent_state/STATUS.json` (current state snapshot)  
* `agent_state/DECISIONS.md` (design choices not explicitly mandated)

**2.1 BUILD\_LOG.md format (append-only)**  
Each entry must include:

* Timestamp  
* Module number \+ name  
* What changed (files touched)  
* Why (link to requirement)  
* Tests run \+ results  
* Current blockers / next step

**2.2 STATUS.json minimum fields**

* `current_module`: int  
* `completed_modules`: list\[int\]  
* `services_running`: {service\_name: bool}  
* `db_migrations_applied`: bool  
* `last_test_run`: {command: str, passed: bool, timestamp: str}  
* `known_gaps`: \[str\]

**2.3 DECISIONS.md**  
Whenever you choose an approach not explicitly defined in the specs, record:

* Decision  
* Alternatives considered  
* Why chosen  
* Impact / follow-up

---

### **3\) Cross-Module Integration Contracts (Must Be Implemented)**

These contracts ensure “uploads → retrieval → evidence → modes” is fully integrated.

#### **3.1 Document Intake & Registry (Prototype-only requirement)**

Implement a first-class “Document Registry” that maps uploaded files to both retrieval systems.

**Core entity: `DocumentRecord` (Pydantic \+ DB)**  
Minimum fields:

* `doc_id` (UUID) — internal canonical ID  
* `filename`, `content_type`, `size_bytes`  
* `sha256` (dedupe \+ provenance)  
* `storage_uri` (MinIO/S3 path)  
* `ingestion_status` (pending/indexed/failed)  
* `pageindex_doc_id` (nullable)  
* `haystack_corpus_tag` or `haystack_doc_ids` (as needed)  
* `created_at`

Rules:

* Every upload creates a DocumentRecord.  
* Every retrieval hit must be traceable back to `doc_id`.

#### **3.2 EvidenceBundle Normalization (Single Evidence Type Everywhere)**

All retrieval outputs (Haystack chunks, PageIndex sections/pages) must be converted into a single canonical object: `EvidenceBundle`, and persisted.

`EvidenceBundle` must include:

* content text  
* hash  
* `source_doc_id` (internal `doc_id`)  
* source title/filename  
* location fields (section/page when available)  
* retrieval method (`haystack` | `pageindex` | `both`)  
* retrieval query \+ timestamp  
* any external IDs needed for debugging (e.g., PageIndex citation pointer)

#### **3.3 Retrieval Router Output Contract**

`RetrievalRouter.retrieve(query, intent, constraints) -> RetrievalResult` where:

* `RetrievalResult.evidence: list[EvidenceBundle]`  
* `RetrievalResult.deduped: bool`  
* `RetrievalResult.stats: {method_counts, latency_ms, unique_docs}`

Router decision logic MUST support:

* Haystack only  
* PageIndex only  
* Both (merge \+ dedupe)

#### **3.4 Protocol Engine → Agent Runtime Contract**

Protocol Engine owns:

* mode selection  
* state machine (IDLE → RUNNING → REVIEW → ACTIVE)  
* task definitions and step ordering

Agent Runtime owns:

* LLM calls  
* tool calls via Retrieval Router / PageIndex / Haystack  
* strict JSON parsing into Pydantic outputs  
* retries \+ error normalization

---

### **4\) Prototype Upload Surface (Must Exist)**

Provide ONE simple way to submit files:

* Preferred: FastAPI endpoint \+ minimal HTML upload page  
* Accept: API-only if it includes a clear curl example and returns usable JSON

Minimum endpoints:

* `POST /uploads` (multipart upload: pdf/txt/md/xlsx; returns `doc_id`)  
* `GET /documents/{doc_id}` (returns registry record)  
* `POST /index/{doc_id}` or auto-index on upload (returns ingestion status)  
* `GET /health`

---

### **5\) Module-by-Module Deliverables \+ Testing Criteria (You Must Implement Tests)**

Implement exactly the modules below, in order, with the following “definition of done”.

---

## **Module 1 — Data Models & Schema Layer**

Deliverables:

* All Pydantic models for: DocumentRecord, EvidenceBundle, WorldModelVersion, causal edges/metadata, DecisionQuery/Recommendation, AuditEntry.  
* Enums \+ validation rules.

Tests (must pass):

* `pytest` validating:  
  * model serialization/deserialization  
  * enum validation  
  * evidence bundle required fields \+ hash formatting

Done when:

* Models are stable and imported by other modules without circular deps.

---

## **Module 2 — Storage Layer (Postgres \+ Redis \+ Object Storage)**

Deliverables:

* Postgres persistence for:  
  * documents registry  
  * evidence bundles  
  * world model versions  
  * audit log  
* Redis cache wrappers (session cache, retrieval cache)  
* MinIO/S3 client wrappers for upload/download

Tests:

* Integration tests using test containers or docker-compose:  
  * create/read/update DocumentRecord  
  * store/retrieve EvidenceBundle  
  * write/read AuditEntry  
  * upload/download object storage roundtrip

Done when:

* CRUD works with no business logic, and migrations are repeatable.

---

## **Module 3 — Protocol Engine (State Machine)**

Deliverables:

* Mode router (Mode 1 vs Mode 2\)  
* State machine transitions  
* Task queue abstraction (even if in-memory for prototype)

Tests:

* Unit tests for:  
  * valid/invalid state transitions  
  * idempotency on retries  
  * task ordering correctness

Done when:

* Mode 1/2 tasks can be scheduled without any LLM calls.

---

## **Module 4 — PageIndex Integration (MCP Client)**

Deliverables:

* MCP client with JSON-RPC formatting  
* Auth/token management abstraction (mockable)  
* Connection pooling  
* Tool interface:  
  * `pageindex_register_document(doc_uri|bytes) -> pageindex_doc_id`  
  * `pageindex_query(pageindex_doc_id, query) -> structured citations + text`

Prototype requirement:

* Upload flow MUST populate `DocumentRecord.pageindex_doc_id` when PageIndex indexing is enabled.

Tests:

* Contract tests with mocked server:  
  * JSON-RPC request shape  
  * timeout/retry behavior  
  * connection pool concurrency safety  
* If a real PageIndex server is available locally, add an optional live smoke test.

Done when:

* Given a `doc_id`, system can query PageIndex and produce EvidenceBundles with citation fields.

---

## **Module 5 — Haystack Pipeline Service**

Deliverables:

* Indexing pipeline for uploaded docs:  
  * PDF/TXT/MD ingestion  
  * XLSX ingestion (convert sheets to text/markdown blocks with metadata)  
* Query pipeline:  
  * hybrid retrieval (BM25 \+ vector)  
  * reranking hook (even if noop initially)  
* Must attach metadata enabling EvidenceBundle creation:  
  * source doc\_id  
  * chunk boundaries / page when available

Tests:

* Local corpus smoke test:  
  * upload 3–5 docs (mix formats)  
  * verify indexing completes  
  * query returns relevant hits  
* Determinism test:  
  * same query on same corpus yields stable top-k ordering within tolerance

Done when:

* Upload → index → query works on local documents end-to-end.

---

## **Module 6 — Retrieval Router**

Deliverables:

* Query classification \+ routing:  
  * “needs exact section citation” → prefer PageIndex  
  * “broad discovery” → prefer Haystack  
  * support “both”  
* Aggregation \+ deduplication (hash-based, source-aware)

Tests:

* Unit tests with synthetic retrieval outputs:  
  * dedupe correctness  
  * stable merge ordering  
  * method selection rules

Done when:

* Single `retrieve()` call returns canonical EvidenceBundles regardless of backend.

---

## **Module 7 — Causal Intelligence Core**

Deliverables:

* DAG engine (NetworkX)  
* PyWhyLLM integration for LLM→DAG conversion (behind interface)  
* Edge strength classification \+ confounder detection scaffolding  
* Optional DoWhy validator hooks (off by default)

Tests:

* DAG invariants:  
  * acyclic enforcement  
  * node/edge schema validation  
* Synthetic graph tests:  
  * edge strength labeling rules  
  * confounder detection on simple patterns

Done when:

* Given structured inputs, you can produce a valid DAG JSON.

---

## **Module 8 — Agent Runtime (LLM Orchestrator)**

Deliverables:

* Gemini API wrapper  
* Context manager (sliding window)  
* Tool calling interface to Retrieval Router  
* Strict output parsing into Pydantic  
* Retry \+ backoff \+ “invalid JSON repair” strategy

Tests:

* Mock LLM tests:  
  * tool call dispatch  
  * JSON parse failure handling  
  * retries capped correctly

Done when:

* Agent can run a scripted workflow using mocked retrieval and produce valid model outputs.

---

## **Module 9 — Mode 1 Implementation (World Model Construction)**

Deliverables:

* Variable discovery workflow using Retrieval Router  
* Deep evidence gathering \+ EvidenceBundle persistence  
* DAG drafting (Agent Runtime → Causal Core)  
* Evidence triangulation \+ edge strength  
* Human review step:  
  * “approve” transition produces WorldModelVersion status=active

Tests:

* End-to-end Mode 1 test on sample uploads:  
  * produces WorldModelVersion  
  * links EvidenceBundles  
  * writes audit log entry

Done when:

* A user can create a reviewed world model from uploaded docs only.

---

## **Module 10 — Mode 2 Implementation (Decision Support)**

Deliverables:

* Decision query parsing  
* Model retrieval \+ freshness checks  
* Evidence refresh logic  
* Causal reasoning application \+ recommendation synthesis  
* Escalation triggers (e.g., insufficient evidence → suggest Mode 1 update)

Tests:

* End-to-end Mode 2 test:  
  * uses existing active WorldModelVersion  
  * retrieves fresh evidence  
  * returns DecisionRecommendation with evidence\_refs  
  * writes audit log

Done when:

* A user gets a cited recommendation based on their uploaded knowledge base.

---

## **Module 11 — Agent Lightning Training Loop (Optional for Prototype Functionality)**

Deliverables:

* Stubs that can be enabled later:  
  * span collection hooks  
  * reward function interface  
  * storage for trajectories

Tests:

* Smoke test that span capture writes a record without breaking Mode 1/2.

Done when:

* System runs fully with this module disabled.

---

## **Module 12 — Observability & API Layer**

Deliverables:

* FastAPI endpoints wrapping:  
  * uploads \+ document registry  
  * indexing triggers \+ status  
  * Mode 1 run \+ Mode 2 run  
  * world model listing \+ detail fetch  
* OpenTelemetry hooks (minimal)  
* Metrics endpoint (Prometheus scrape) minimal counters:  
  * uploads\_total  
  * retrieval\_calls\_total by method  
  * mode\_runs\_total by mode  
  * errors\_total

Tests:

* API smoke test:  
  * upload docs  
  * index docs  
  * run Mode 1  
  * run Mode 2  
  * verify `/health` and metrics endpoint respond

Done when:

* A developer can operate the prototype entirely via API (and optionally a minimal web upload page).

---

### **6\) End-to-End Acceptance Checklist (System “Done” Gate)**

You are not done until all are true:

1. Upload PDF/TXT/MD/XLSX returns `doc_id`, persists to object storage, registry reflects it.  
2. Indexing completes; Haystack queries return evidence from uploaded docs.  
3. PageIndex doc registration works for PDFs/MD/TXT (or is explicitly disabled with a recorded decision \+ router fallback).  
4. Retrieval Router returns canonical EvidenceBundles with correct provenance fields.  
5. Mode 1 creates an “active” WorldModelVersion linked to EvidenceBundles.  
6. Mode 2 returns a recommendation with evidence\_refs and logs audit entries.  
7. Logs (`BUILD_LOG.md`, `STATUS.json`, `DECISIONS.md`) accurately reflect current system state.

---

### **7\) If You Discover Gaps in the Provided Specs**

If any spec detail is missing or conflicts:

1. Prefer `lowlevel-systemdoc.md` for implementation details.  
2. Prefer `systemdoc.md` for product behavior and intent.  
3. Implement the smallest compatible default.  
4. Record the choice in `agent_state/DECISIONS.md`.  
5. Add/adjust tests to lock the behavior.

