# Causeway

> Agentic decision-support system that builds causal world models from documents and answers "what-if" questions with traceable, evidence-grounded reasoning.

Organizations make high-stakes decisions using scattered documents — policy handbooks, strategy decks, compliance guides. Causeway ingests those documents, automatically discovers causal relationships between variables (Mode 1), and then uses the resulting world model to answer decision questions like *"Should we increase prices by 15%?"* with structured recommendations backed by causal paths and cited evidence (Mode 2).

Every recommendation traces back to retrieved evidence, through a causal graph, with human-in-the-loop review at every stage.

---

## Table of Contents

- [Tech Stack](#tech-stack)
- [Level 1 — System Context](#level-1--system-context)
- [Level 2 — Architecture Overview](#level-2--architecture-overview)
- [Level 3 — Data Flow](#level-3--data-flow)
- [Level 4 — Component Breakdown: Causal Engine](#level-4--component-breakdown-causal-engine)
- [Key Design Decisions](#key-design-decisions)
- [Repository Layout](#repository-layout)
- [Quick Start](#quick-start)
- [API Endpoints](#api-endpoints)
- [Testing](#testing)

---

## Tech Stack

| Technology | Role | Why |
|---|---|---|
| **Python 3.12** | Backend language | Async-first ecosystem, rich ML/NLP library support |
| **FastAPI** | API framework | Native async, auto-generated OpenAPI docs, Pydantic validation |
| **React 18 + TypeScript** | Frontend | Component-driven UI with type safety; React Flow for graph visualization |
| **Vite** | Frontend build | Sub-second HMR, native ESM, fast production builds |
| **Tailwind CSS + shadcn/ui** | Styling | Utility-first CSS with accessible Radix UI primitives |
| **PostgreSQL 15** | Primary database | JSONB for flexible causal model storage, ACID for audit logs |
| **Redis 7** | Cache layer | Sub-ms retrieval result caching, session state, TTL-based expiry |
| **MinIO** | Object storage | S3-compatible binary document storage (PDF, XLSX) without cloud lock-in |
| **Qdrant** | Vector database | Purpose-built for dense vector search; 384-dim MiniLM embeddings |
| **Haystack 2.x** | RAG framework | Modular retrieval pipelines with BM25 + vector hybrid fusion |
| **sentence-transformers** | Embeddings | `all-MiniLM-L6-v2` — fast 384-dim embeddings for semantic search and variable matching |
| **NetworkX** | Graph engine | Mature DAG primitives — cycle detection, path finding, topological analysis |
| **PyWhyLLM** | Causal discovery | LLM-driven pairwise causal relationship classification |
| **Google Gemini** | LLM backbone | Structured output, long context window, used for extraction + reasoning + verification |
| **LangExtract** | Structured extraction | Schema-driven LLM extraction of variables, edges, queries, recommendations |
| **Docker Compose** | Infrastructure | Single-command local stack for Postgres, Redis, MinIO, Qdrant |

---

## Level 1 — System Context

How Causeway relates to its users and external dependencies.

```mermaid
C4Context
    title System Context — Causeway

    Person(analyst, "Decision Analyst", "Asks causal what-if questions")
    Person(builder, "Domain Expert", "Uploads documents, reviews world models")

    System(causeway, "Causeway", "Agentic decision-support system<br/>Builds causal world models from documents<br/>Answers what-if questions with evidence")

    System_Ext(gemini, "Google Gemini API", "LLM for extraction, reasoning, verification")
    System_Ext(pageindex, "PageIndex MCP", "Optional structural document navigation")

    SystemDb_Ext(postgres, "PostgreSQL", "Documents, world models, audit logs")
    SystemDb_Ext(qdrant, "Qdrant", "Vector embeddings for semantic search")
    SystemDb_Ext(redis, "Redis", "Cache and session state")
    SystemDb_Ext(minio, "MinIO (S3)", "Binary document storage")

    Rel(analyst, causeway, "Queries decisions", "HTTP / React UI")
    Rel(builder, causeway, "Uploads docs, reviews models", "HTTP / React UI")
    Rel(causeway, gemini, "Extraction, reasoning, verification", "HTTPS")
    Rel(causeway, pageindex, "Document navigation", "HTTPS")
    Rel(causeway, postgres, "Reads/writes", "TCP :5432")
    Rel(causeway, qdrant, "Embeds/searches", "HTTP :6333")
    Rel(causeway, redis, "Caches", "TCP :6379")
    Rel(causeway, minio, "Stores files", "HTTP :9000")
```
- Vite
- TanStack Query
- React Router
- Tailwind + shadcn/ui
- React Flow + dagre

---

## Repository Layout

```text
---

## Level 2 — Architecture Overview

The internal module structure and how the layers connect.

```mermaid
graph TB
    subgraph Frontend["Frontend — React + Vite (:8080)"]
        UI["React 18 + TypeScript<br/>React Flow · dagre · Recharts<br/>Tailwind + shadcn/ui"]
    end

    subgraph API["API Layer — FastAPI (:8000)"]
        Routes["routes.py<br/>20 REST endpoints"]
        Middleware["CORS · Metrics · Lifespan"]
    end

    subgraph Agent["Agent Layer"]
        CA["CausewayAgent<br/>Unified entrypoint"]
        Orch["AgentOrchestrator<br/>LLM tool-calling loop"]
        LLM["LLMClient<br/>Gemini wrapper + mock mode"]
    end

    subgraph Protocol["Protocol Layer"]
        SM["StateMachine<br/>IDLE → ROUTING → RUNNING → REVIEW"]
        MR["ModeRouter<br/>Pattern matching + confidence"]
    end

    subgraph Modes["Mode Pipelines"]
        M1["Mode 1<br/>World Model Construction<br/>6 stages"]
        M2["Mode 2<br/>Decision Support<br/>7 stages"]
    end

    subgraph Causal["Causal Layer"]
        CS["CausalService<br/>Orchestrator"]
        DAG["DAGEngine<br/>NetworkX DiGraph"]
        PF["PathFinder<br/>Paths · confounders · mediators"]
        PW["PyWhyLLM Bridge<br/>Pairwise causal discovery"]
        CR["ConflictResolver<br/>Contradiction detection"]
        TT["TemporalTracker<br/>Confidence decay"]
    end

    subgraph Extraction["Extraction Layer"]
        ES["ExtractionService<br/>LangExtract + Gemini"]
        DE["DocumentExtractor<br/>PDF · XLSX · TXT"]
    end

    subgraph Verification["Verification Layer"]
        VA["VerificationAgent<br/>Proposer-Retriever-Judge loop"]
        VJ["VerificationJudge<br/>Grounding + adversarial"]
        GR["GroundingRetriever<br/>Targeted evidence"]
    end

    subgraph Retrieval["Retrieval Layer"]
        RR["RetrievalRouter<br/>Strategy selection + dedup"]
        HP["HaystackPipeline<br/>BM25 + vector hybrid"]
        PI["PageIndex Client<br/>Structural navigation"]
    end

    subgraph Storage["Storage Layer"]
        DB["DatabaseService<br/>SQLAlchemy async · 6 tables"]
        OS["ObjectStore<br/>MinIO S3 client"]
        RC["RedisCache<br/>TTL cache + sessions"]
    end

    subgraph Training["Training Layer"]
        SC["SpanCollector"]
        TS["TrajectoryStore"]
        RF["RewardFunction"]
    end

    UI -->|"HTTP /api/v1/*"| Routes
    Routes --> CA
    Routes --> M1
    Routes --> M2
    CA --> MR
    CA --> Orch
    Orch --> LLM
    MR --> SM
    M1 --> ES
    M1 --> PW
    M1 --> VA
    M1 --> CS
    M2 --> ES
    M2 --> RR
    M2 --> CS
    CS --> DAG
    CS --> PF
    CS --> CR
    CS --> TT
    VA --> GR
    VA --> VJ
    GR --> RR
    RR --> HP
    RR --> PI
    HP --> Qdrant[(Qdrant :6333)]
    PI -->|"HTTPS"| PageIdx[("PageIndex MCP")]
    ES -->|"HTTPS"| Gemini[("Gemini API")]
    LLM -->|"HTTPS"| Gemini
    VJ -->|"HTTPS"| Gemini
    PW -->|"HTTPS"| Gemini
    DB --> PG[(PostgreSQL :5432)]
    OS --> MinIO[(MinIO :9000)]
    RC --> Redis[(Redis :6379)]
    M1 --> DB
    M2 --> DB
    VA --> SC
```

---

## Level 3 — Data Flow

### Document Ingestion

```mermaid
sequenceDiagram
    participant User
    participant API as FastAPI
    participant OS as MinIO
    participant DB as PostgreSQL
    participant EX as DocumentExtractor
    participant HP as HaystackPipeline
    participant QD as Qdrant
    participant PI as PageIndex

    User->>API: POST /uploads (file)
    API->>OS: upload_file(bytes)
    OS-->>API: storage_uri
    API->>DB: create_document(metadata)
    API-->>User: 200 {doc_id}

    User->>API: POST /index/{doc_id}
    API->>OS: download_file(doc_id)
    OS-->>API: raw bytes
    API->>EX: extract(bytes) → plain text
    API->>HP: add_document(text, doc_id)
    HP->>HP: split (3 sentences/chunk)
    HP->>HP: embed (MiniLM-L6-v2, 384-dim)
    HP->>QD: store vectors + metadata
    API->>PI: register_document (optional)
    API->>DB: update_status("indexed")
    API-->>User: 200 {indexed}
```

### Mode 1 — World Model Construction

```mermaid
sequenceDiagram
    participant User
    participant M1 as Mode 1 Pipeline
    participant RR as RetrievalRouter
    participant ES as ExtractionService
    participant PW as PyWhyLLM Bridge
    participant VA as VerificationAgent
    participant CS as CausalService
    participant DB as PostgreSQL
    participant LLM as Gemini API

    User->>M1: POST /mode1/run {domain, query}

    rect rgb(240, 248, 255)
        Note over M1,LLM: Stage 1 — Variable Discovery
        M1->>RR: retrieve(query, hybrid)
        RR-->>M1: evidence[]
        M1->>ES: extract_variables(evidence)
        ES->>LLM: structured extraction
        LLM-->>ES: variable candidates
    end

    rect rgb(245, 245, 220)
        Note over M1,LLM: Stage 2 — Canonicalization
        M1->>ES: deduplicate(variables)
        ES->>LLM: merge semantic duplicates
        LLM-->>M1: canonical variables
    end

    rect rgb(240, 248, 255)
        Note over M1,RR: Stage 3 — Evidence Gathering
        loop For each variable
            M1->>RR: retrieve(variable, hybrid)
        end
    end

    rect rgb(245, 245, 220)
        Note over M1,LLM: Stage 4 — DAG Drafting
        M1->>PW: build_graph(variables, evidence)
        loop O(n²) pairwise
            PW->>LLM: "Does A cause B?"
        end
        PW-->>M1: edge candidates
    end

    rect rgb(240, 248, 255)
        Note over M1,LLM: Stage 5 — Mechanism Synthesis
        M1->>ES: synthesize_mechanisms(edges)
        ES->>LLM: rationale + quote + citation
    end

    rect rgb(245, 245, 220)
        Note over M1,VA: Stage 6 — Verification Loop
        loop For each edge
            VA->>RR: ground_edge(from, to)
            VA->>LLM: judge(evidence, edge)
            alt Refinement needed
                VA->>RR: refined query
                VA->>LLM: re-judge
            end
        end
    end

    M1->>CS: build DAG (NetworkX)
    CS->>CS: validate (acyclicity)
    M1->>DB: save_world_model(REVIEW)
    M1-->>User: Mode1Result {model, audit_log}
```

### Mode 2 — Decision Support

```mermaid
sequenceDiagram
    participant User
    participant M2 as Mode 2 Pipeline
    participant ES as ExtractionService
    participant CS as CausalService
    participant RR as RetrievalRouter
    participant CR as ConflictResolver
    participant TT as TemporalTracker
    participant LLM as Gemini API

    User->>M2: POST /mode2/run {query, domain_hint}

    rect rgb(240, 248, 255)
        Note over M2,LLM: Stage 1 — Query Parsing
        M2->>ES: parse_query(query)
        ES->>LLM: extract intervention, outcome, constraints
        LLM-->>M2: ParsedQuery
    end

    rect rgb(245, 245, 220)
        Note over M2,CS: Stage 2 — Model Retrieval
        M2->>CS: load_model(domain)
        alt No model exists
            M2-->>User: Escalate to Mode 1
        end
    end

    rect rgb(240, 248, 255)
        Note over M2,TT: Stage 3 — Staleness Check
        M2->>TT: check_staleness(model)
        TT-->>M2: freshness score + decay
    end

    rect rgb(245, 245, 220)
        Note over M2,RR: Stage 4 — Evidence Refresh
        M2->>RR: retrieve_hybrid(intervention + outcome)
        RR-->>M2: fresh evidence[]
    end

    rect rgb(240, 248, 255)
        Note over M2,CR: Stage 5 — Conflict Detection
        M2->>CR: detect_conflicts(evidence, model)
        CR-->>M2: ConflictReport
    end

    rect rgb(245, 245, 220)
        Note over M2,CS: Stage 6 — Causal Reasoning
        M2->>CS: find_paths(intervention_var, outcome_var)
        CS-->>M2: paths, confounders, mediators
    end

    rect rgb(240, 248, 255)
        Note over M2,LLM: Stage 7 — Recommendation
        M2->>ES: synthesize(query, paths, evidence)
        ES->>LLM: generate grounded recommendation
        LLM-->>M2: DecisionRecommendation
    end

    M2-->>User: Mode2Result {recommendation, evidence, audit}
```

---

## Level 4 — Component Breakdown: Causal Engine

The most critical module — responsible for building, validating, and querying causal world models.

```mermaid
graph TB
    subgraph CausalLayer["src/causal/"]

        subgraph Service["CausalService — Orchestrator"]
            S_create["create_world_model(domain)"]
            S_import["import_world_model(model)"]
            S_export["export_world_model(domain)"]
            S_analyze["analyze_relationship(A, B)"]
            S_conflicts["detect_conflicts(evidence)"]
            S_staleness["check_model_staleness()"]
        end

        subgraph Engine["DAGEngine — NetworkX Graph"]
            E_addvar["add_variable(id, name, def)"]
            E_addedge["add_edge(from, to, mechanism)<br/>⚠ CycleDetectedError"]
            E_validate["validate()<br/>acyclicity · isolation · evidence"]
            E_serial["to_world_model() ↔ from_world_model()<br/>JSON serialization"]
        end

        subgraph PathFinder["CausalPathFinder — Query Engine"]
            PF_paths["find_all_paths(A, B)"]
            PF_conf["find_confounders(A, B)<br/>Common ancestors"]
            PF_med["find_mediators(A, B)<br/>Intermediate nodes"]
            PF_anc["get_ancestors / get_descendants"]
        end

        subgraph Bridge["PyWhyLLM Bridge — Causal Discovery"]
            B_build["build_graph_from_evidence()"]
            B_pair["O(n²) pairwise LLM calls"]
            B_pre["_prefilter_pairs()<br/>Evidence-density pruning"]
            B_norm["_normalize_answer()<br/>yes/no/uncertain parser"]
        end

        subgraph Conflict["ConflictResolver"]
            C_detect["detect_edge_conflicts()"]
            C_types["Types: contradiction · reversal<br/>strength_change · missing_var"]
            C_resolve["resolve_all(strategy)<br/>evidence-weighted · temporal<br/>source-priority · manual"]
        end

        subgraph Temporal["TemporalTracker"]
            T_stale["check_staleness()"]
            T_decay["apply_confidence_decay()<br/>Exponential: f(t) = e^(-λt)"]
            T_fresh["freshness_report()<br/>Per-edge age tracking"]
        end
    end

    Service --> Engine
    Service --> PathFinder
    Service --> Conflict
    Service --> Temporal
    PathFinder -.->|"reads"| Engine
    Bridge -->|"edges"| Engine
```

---

## Key Design Decisions

### Two-Mode Architecture

The system separates **model construction** (Mode 1) from **decision support** (Mode 2). Mode 1 is expensive — it runs O(n²) pairwise LLM calls and multi-turn verification. Mode 2 is fast — it loads an existing model and traces causal paths. This separation means the heavy construction work happens once, and decision queries can be answered quickly against the persisted model.

### Human-in-the-Loop by Default

Every Mode 1 run produces a world model in `REVIEW` status, not `ACTIVE`. A domain expert must inspect the discovered variables, edges, and mechanisms before the model serves live queries. This prevents the system from confidently answering questions based on a hallucinated causal graph.

### Proposer-Retriever-Judge Verification

Rather than trusting the LLM's initial edge proposals, every edge passes through a verification loop: retrieve supporting evidence → judge whether it's grounded → refine the query if the judge suggests it → re-judge. An adversarial pass then checks for confounding and reverse causation. This multi-turn structure catches edges that sound plausible but lack evidence.

### Semantic Variable Matching

When a user asks *"increase prices by 15%"*, the system must find the relevant variable in a model that might call it `financial_measures` or `value`. The matching uses a 4-tier strategy: exact match → substring containment → stemmed token overlap → sentence-transformer embedding cosine similarity. This allows natural-language queries to connect to formal model variables without requiring exact terminology.

### Escalation Over Hallucination

When the causal model has no relevant variable for a query, the system escalates to Mode 1 ("build a model first") rather than fabricating a recommendation from unrelated evidence. Every escalation includes the specific reason and what the user should do next.

### Hybrid Retrieval (BM25 + Vector)

Document retrieval uses reciprocal rank fusion of BM25 (keyword) and dense vector (semantic) results. This catches both exact-match terminology and semantic paraphrases. The system uses `all-MiniLM-L6-v2` (384-dim) for fast embedding with acceptable quality.

### Append-Only Audit Trail

Every pipeline stage writes to an append-only audit log with trace IDs. This means every recommendation can be traced back through: which causal paths were found → what evidence was retrieved → which edges were verified → which documents were indexed. The audit log is queryable by trace ID, mode, and timestamp.

---

## Repository Layout

```
src/
├── api/                # FastAPI app, routes, middleware
├── agent/              # CausewayAgent, Orchestrator, LLMClient
├── protocol/           # State machine, mode router
├── modes/              # Mode 1 (construction), Mode 2 (decisions)
├── causal/             # DAG engine, path finder, PyWhyLLM bridge,
│                       # conflict resolver, temporal tracker
├── extraction/         # LangExtract service, document extractor
├── verification/       # Proposer-Retriever-Judge loop
├── retrieval/          # Retrieval router (Haystack + PageIndex)
├── haystack_svc/       # Qdrant vector pipeline
├── pageindex/          # PageIndex MCP client
├── storage/            # PostgreSQL, MinIO, Redis
├── training/           # Spans, trajectories, rewards (Agent Lightning)
├── models/             # Pydantic domain models
└── utils/              # Text truncation, telemetry
frontend/ui/            # React + Vite + Tailwind + shadcn/ui
tests/                  # 500 unit/integration tests
docker-compose.yml      # Postgres, Redis, MinIO, Qdrant
```

---

## Quick Start

### Prerequisites

- Python 3.11+, Node.js 18+, Docker

### 1. Start infrastructure

```bash
docker compose up -d
```

### 2. Start backend

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env   # edit GOOGLE_AI_API_KEY
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

### 3. Start frontend

```bash
cd frontend/ui
npm install
npm run dev             # → http://localhost:8080
```

### Environment variables

```bash
DATABASE_URL=postgresql+asyncpg://causeway:causeway_dev@localhost:5432/causeway
REDIS_URL=redis://localhost:6379/0
MINIO_ENDPOINT=localhost:9000
MINIO_ACCESS_KEY=causeway
MINIO_SECRET_KEY=causeway_dev_key
MINIO_BUCKET=causeway-docs
QDRANT_HOST=localhost
QDRANT_PORT=6333
GOOGLE_AI_API_KEY=your_key_here
```

---

## API Endpoints

Base prefix: `/api/v1` — Interactive docs at `http://localhost:8000/docs`

| Group | Method | Path | Purpose |
|---|---|---|---|
| System | GET | `/health` | Health check |
| | GET | `/metrics` | Uptime, request/error counts |
| Documents | POST | `/uploads` | Upload document |
| | GET | `/documents` | List all documents |
| | GET | `/documents/{id}` | Document metadata |
| | POST | `/index/{id}` | Index document for search |
| Search | POST | `/search` | Semantic evidence search |
| Mode 1 | POST | `/mode1/run` | Start world model construction |
| | GET | `/mode1/status` | Construction progress |
| | POST | `/mode1/approve` | Approve draft model |
| Mode 2 | POST | `/mode2/run` | Decision support query |
| World Models | GET | `/world-models` | List all models |
| | GET | `/world-models/{domain}` | Model summary |
| | GET | `/world-models/{domain}/detail` | Full model with edges |
| | PATCH | `/world-models/{domain}` | Update model |
| Bridges | GET | `/world-models/bridges` | List cross-domain bridges |
| | POST | `/world-models/bridge` | Create bridge |
| Query | POST | `/query` | Unified agentic query |
| Protocol | GET | `/protocol/status` | State machine status |
| Admin | POST | `/admin/purge-documents` | Delete all documents |

---

## Testing

```bash
source .venv/bin/activate
pytest                    # 500 passed, 17 skipped
```

Frontend:

```bash
cd frontend/ui
npm run test
npm run lint
```

---

## License

MIT
