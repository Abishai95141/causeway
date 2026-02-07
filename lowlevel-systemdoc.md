# Agentic Decision Support System: Comprehensive Implementation Plan

**Version:** 1.0  
**Date:** February 6, 2026  
**Purpose:** Low-level architectural specification and implementation roadmap

---

## Table of Contents

1. System Architecture Overview
2. Component Deep Dive
3. LLM Flow Architecture
4. Agent Orchestration
5. Data Models & Schemas
6. Integration Patterns
7. Mode 1: World Model Construction - Detailed Flow
8. Mode 2: Decision Support - Detailed Flow
9. Retrieval Strategy Implementation
10. Causal Intelligence Layer
11. Agent Lightning Training Loop
12. Storage Layer Architecture
13. API Contracts & Interfaces
14. Deployment Architecture
15. Error Handling & Resilience
16. Monitoring & Observability

---

## 1. System Architecture Overview

### 1.1 High-Level Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                         User Interface Layer                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐  │
│  │   Web UI     │  │   REST API   │  │   Admin Dashboard        │  │
│  └──────────────┘  └──────────────┘  └──────────────────────────┘  │
└─────────────────────────────────┬───────────────────────────────────┘
                                  │
┌─────────────────────────────────▼───────────────────────────────────┐
│                      Orchestration Layer                             │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │              Protocol Engine (State Machine)                   │  │
│  │  - Mode Router                                                │  │
│  │  - State Manager (IDLE → RUNNING → REVIEW → ACTIVE)          │  │
│  │  - Task Queue Manager                                         │  │
│  └───────────────────────────────────────────────────────────────┘  │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │           Agent Runtime (Primary LLM Orchestrator)            │  │
│  │  - LLM: Gemini 2.0 Flash (via Google AI API)                 │  │
│  │  - Context Manager (sliding window, 1M token context)        │  │
│  │  - Tool Calling Interface                                     │  │
│  │  - Structured Output Parser                                   │  │
│  └───────────────────────────────────────────────────────────────┘  │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │       Agent Lightning Integration (Training Layer)            │  │
│  │  - Lightning Server (Training Coordinator)                    │  │
│  │  - Lightning Client (Span Collection)                         │  │
│  │  - LightningStore (Training Data Repository)                  │  │
│  └───────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────┬───────────────────────────────────┘
                                  │
        ┌─────────────────────────┼─────────────────────────┐
        │                         │                         │
┌───────▼────────┐    ┌───────────▼──────────┐    ┌────────▼──────────┐
│  Retrieval     │    │  Causal Intelligence │    │  Storage & Audit  │
│  Layer         │    │  Layer               │    │                   │
│                │    │                      │    │                   │
│ ┌────────────┐ │    │ ┌──────────────────┐│    │ ┌───────────────┐ │
│ │PageIndex   │ │    │ │  PyWhyLLM        ││    │ │PostgreSQL DB  │ │
│ │MCP Server  │ │    │ │  Integration     ││    │ │- World Models │ │
│ └────────────┘ │    │ └──────────────────┘│    │ │- Evidence     │ │
│                │    │                      │    │ │- Audit Log    │ │
│ ┌────────────┐ │    │ ┌──────────────────┐│    │ └───────────────┘ │
│ │Haystack    │ │    │ │  NetworkX DAG    ││    │                   │
│ │Pipeline    │ │    │ │  Engine          ││    │ ┌───────────────┐ │
│ │Service     │ │    │ └──────────────────┘│    │ │Redis Cache    │ │
│ └────────────┘ │    │                      │    │ │- Session      │ │
│                │    │ ┌──────────────────┐│    │ │- Results      │ │
│ ┌────────────┐ │    │ │  (Optional)      ││    │ └───────────────┘ │
│ │Retrieval   │ │    │ │  DoWhy/SCD       ││    │                   │
│ │Router      │ │    │ │  Validators      ││    │ ┌───────────────┐ │
│ └────────────┘ │    │ └──────────────────┘│    │ │S3/MinIO       │ │
└────────────────┘    └──────────────────────┘    │ │- Documents    │ │
        │                                          │ │- Artifacts    │ │
        │                                          │ └───────────────┘ │
        │                                          └───────────────────┘
        │
┌───────▼────────────────────────────────────────────────────┐
│              External Data Sources                          │
│  ┌──────────────┐  ┌──────────────┐  ┌─────────────────┐  │
│  │  Document    │  │  Structured  │  │  Event Logs/    │  │
│  │  Repository  │  │  Databases   │  │  APIs           │  │
│  │  (S3/MinIO)  │  │  (SQL/NoSQL) │  │                 │  │
│  └──────────────┘  └──────────────┘  └─────────────────┘  │
└────────────────────────────────────────────────────────────┘
```

### 1.2 Technology Stack

**Primary Components:**
- **Orchestration:** Python 3.11+ with FastAPI
- **Agent Runtime LLM:** Gemini 2.0 Flash (Free API tier)
- **Retrieval:**
  - PageIndex MCP Server (HTTP transport via OAuth)
  - Haystack 2.x with Qdrant vector store
- **Causal Layer:** PyWhyLLM + NetworkX + (optional) DoWhy
- **Training:** Agent Lightning (Microsoft) + veRL backend
- **Storage:** PostgreSQL 15+ (primary), Redis 7+ (cache), MinIO (objects)
- **Monitoring:** OpenTelemetry + Prometheus + Grafana

### 1.3 Communication Patterns

```
User Request → FastAPI → Protocol Engine → Agent Runtime → Tools
                                              ↓
                                    ┌─────────┴─────────┐
                                    ↓                   ↓
                            Retrieval Layer      Causal Layer
                                    ↓                   ↓
                            Storage Layer ←─────────────┘
                                    ↓
                            Lightning Server (async, background)
```

---

## 2. Component Deep Dive

### 2.1 PageIndex MCP Server Integration

**Purpose:** Structured document navigation with tree-based reasoning

**Architecture:**
```
┌─────────────────────────────────────────────────┐
│         PageIndex MCP Server (Remote)           │
│  URL: https://chat.pageindex.ai/mcp             │
│  Transport: Streamable HTTP (SSE for streaming) │
│  Auth: OAuth 2.0 (automatic token refresh)      │
└─────────────────────────────────────────────────┘
                    ↑
                    │ HTTP POST + SSE
                    │
┌───────────────────┴─────────────────────────────┐
│         MCP Client (in Agent Runtime)           │
│  - JSON-RPC 2.0 message formatter               │
│  - OAuth token manager                          │
│  - Connection pool (max 10 concurrent)          │
│  - Request timeout: 30s                         │
└─────────────────────────────────────────────────┘
```

**Key Features:**
- Tree-based document indexing (sections, headings, pages)
- LLM navigates via "what sections exist?" then "read section X"
- No chunking, no vectors, full page-level provenance
- Returns precise citations: section 3.2.1 of doc.pdf, page 12

**When to Use:**
- Policy documents, postmortems, technical specs, contracts
- Need exact citation for audit trail
- Long-form documents where context hierarchy matters

### 2.2 Haystack Pipeline Service

**Purpose:** Hybrid retrieval across large document corpora

**Architecture:**
```
┌──────────────────────────────────────────────────────┐
│           Haystack Pipeline Service                  │
│  ┌────────────────────────────────────────────────┐  │
│  │         Indexing Pipeline (Async Worker)       │  │
│  │                                                │  │
│  │  MarkdownToDocument → DocumentSplitter        │  │
│  │       ↓                      ↓                 │  │
│  │  SentenceTransformersDocumentEmbedder         │  │
│  │       ↓                                        │  │
│  │  DocumentWriter → Qdrant Vector Store         │  │
│  └────────────────────────────────────────────────┘  │
│                                                       │
│  ┌────────────────────────────────────────────────┐  │
│  │         Query Pipeline (Synchronous)           │  │
│  │                                                │  │
│  │  SentenceTransformersTextEmbedder             │  │
│  │       ↓                                        │  │
│  │  HybridRetriever (BM25 + Vector)              │  │
│  │       ├─ InMemoryBM25Retriever                │  │
│  │       └─ QdrantEmbeddingRetriever             │  │
│  │       ↓                                        │  │
│  │  DiversityRanker (reduce redundancy)          │  │
│  │       ↓                                        │  │
│  │  LostInTheMiddleRanker (optimize context)     │  │
│  │       ↓                                        │  │
│  │  Results (top_k=10)                           │  │
│  └────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────┘
```

**Key Features:**
- Hybrid retrieval: BM25 (keyword) + Vector (semantic)
- Document processing with chunking (5 sentences per chunk, 1 overlap)
- Embedding: sentence-transformers/all-MiniLM-L6-v2 (384 dim)
- Reranking for diversity and context optimization
- Qdrant vector store for efficient similarity search

**When to Use:**
- Large document corpora (100+ docs)
- Recall across many sources matters more than exact provenance
- Fast semantic search for variable discovery
- Initial filtering before deep PageIndex navigation

### 2.3 Retrieval Router

**Purpose:** Intelligent routing between PageIndex and Haystack

**Decision Logic:**

1. **PageIndex Only** - When:
   - Explicit document reference (e.g., "in the pricing policy doc")
   - Document is structured/long-form (policy, contract, spec)
   - Query has structural markers (section 3.2, page 5, chapter 2)
   - Need precise citations for audit

2. **Haystack Only** - When:
   - Broad discovery needed across many documents
   - Semantic similarity matters more than structure
   - Fast search required
   - Don't know which specific document contains info

3. **Hybrid (Haystack → PageIndex)** - When:
   - Need both breadth (find relevant docs) and depth (precise extraction)
   - Complex query requiring cross-document patterns plus detailed evidence
   - Building world models (discover variables, then extract definitions)

**Output Contract (Unified):**
All retrieval methods return `EvidenceBundle` with:
- `content`: Extracted text
- `source`: {doc_id, doc_title, url/path}
- `location`: {section_name, page_number, paragraph_id}
- `retrieval_trace`: How this was found (method, query, scores)
- `timestamp`: When retrieved
- `hash`: Content fingerprint (SHA256)

---

## 3. LLM Flow Architecture

### 3.1 Primary Agent LLM: Gemini 2.0 Flash

**Why Gemini 2.0 Flash?**
- Free API tier (60 requests/minute)
- 1M token context window
- Native function calling / tool use
- Strong reasoning capabilities
- Low latency (~2s for complex reasoning)
- JSON structured output support

**Configuration:**
```python
import google.generativeai as genai

genai.configure(api_key=os.environ["GOOGLE_AI_API_KEY"])

model = genai.GenerativeModel(
    model_name="gemini-2.0-flash-exp",
    generation_config={
        "temperature": 0.1,          # Low for consistency
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 8192,
        "response_mime_type": "application/json",  # Force JSON output
    },
    safety_settings={
        "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE",
        "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE",
        "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE",
        "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE",
    }
)
```

### 3.2 Multi-Turn Tool Calling Flow

**Architecture:**
```
User Query
    ↓
┌─────────────────────────────────────┐
│  Turn 1: Agent analyzes query       │
│  Returns: tool_call                 │
│  {                                  │
│    "name": "haystack_retrieve",     │
│    "args": {"query": "...", ...}    │
│  }                                  │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  Tool Executor runs function        │
│  Returns: result + metadata         │
│  Logs span to Agent Lightning       │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  Turn 2: Agent receives result      │
│  Decides: call more tools OR        │
│           generate final answer     │
└─────────────────────────────────────┘
    ↓
[Repeat until final answer or max turns]
    ↓
Final Response
```

**Key Points:**
- Agent can call multiple tools in sequence
- Each tool call captured as span for training
- Max 10 turns to prevent infinite loops
- Context includes full conversation + tool results
- Structured JSON output for parsing

### 3.3 Context Window Management Strategy

**Challenge:** 1M token context, but need to prioritize relevance

**Solution: Hierarchical Context Budget**

Priority levels:
1. **System Prompt** (Always include) - ~5K tokens
2. **World Model Slice** (High priority) - ~50K tokens
3. **Recent Evidence** (Last 10 bundles) - ~100K tokens
4. **Conversation History** (Sliding window) - Remaining budget

**Implementation:**
- Track token count per section
- When approaching limit (500K target, 1M max):
  - Keep all high-priority sections
  - Trim oldest conversation history first
  - Summarize very old evidence if needed
- Maintain audit trail of what was included/excluded

---

## 4. Agent Orchestration

### 4.1 Protocol Engine State Machine

**States:**

```
IDLE
  ↓ (user request)
ROUTING (classify Mode 1 vs Mode 2)
  ↓
  ├─ Mode 1 ────────────────────────┐
  │  WM_DISCOVERY_RUNNING           │
  │    ↓                            │
  │  WM_REVIEW_PENDING              │
  │    ↓ (approved)                 │
  │  WM_ACTIVE                      │
  │    ↓                            │
  └────────────────────────────────┘
  │
  └─ Mode 2 ────────────────────────┐
     DECISION_SUPPORT_RUNNING       │
       ↓                            │
     RESPONSE_READY                 │
       ↓                            │
     ──────────────────────────────┘
  │
  ↓
IDLE (ready for next request)
```

**State Transitions:**
- Each transition logged with timestamp, reason, metadata
- State persisted to DB for recovery
- Timeouts: WM_DISCOVERY (30 min), DECISION_SUPPORT (5 min)
- Human approval required for WM_REVIEW_PENDING → WM_ACTIVE

### 4.2 Multi-Agent Coordination

**Pattern: Coordinator-Specialist**

For complex tasks (Mode 1), use multiple specialized agents:

**Agents:**
1. **Retrieval Specialist**
   - Tools: PageIndex, Haystack
   - Task: Find variables and evidence
   - Prompt: Focus on comprehensive search

2. **Causal Analyst**
   - Tools: PyWhyLLM, DAG tools
   - Task: Propose causal relationships
   - Prompt: Focus on mechanisms and assumptions

3. **Evidence Validator**
   - Tools: Validation tools
   - Task: Check evidence quality
   - Prompt: Focus on contradictions and gaps

**Coordinator:**
- Orchestrates specialists in sequence
- Aggregates their outputs
- Makes final synthesis decisions
- Single point of audit logging

---

## 5. Data Models & Schemas

### 5.1 Core Pydantic Models

```python
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Literal
from datetime import datetime
from uuid import UUID, uuid4

# Evidence Models
class SourceReference(BaseModel):
    doc_id: str
    doc_title: str
    url: Optional[str] = None
    version: Optional[str] = None

class LocationMetadata(BaseModel):
    section_name: Optional[str] = None
    section_number: Optional[str] = None
    page_number: Optional[int] = None
    paragraph_id: Optional[str] = None
    chunk_id: Optional[str] = None

class RetrievalTrace(BaseModel):
    method: Literal["pageindex", "haystack_hybrid", "haystack_vector"]
    timestamp: datetime
    query: str
    scores: Optional[Dict[str, float]] = None
    retrieval_path: Optional[List[str]] = None

class EvidenceBundle(BaseModel):
    bundle_id: UUID = Field(default_factory=uuid4)
    content: str
    source: SourceReference
    location: LocationMetadata
    retrieval_trace: RetrievalTrace
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    hash: str
    used_in_models: List[str] = Field(default_factory=list)

# Causal Models
class VariableDefinition(BaseModel):
    variable_id: str
    name: str
    definition: str
    type: Literal["continuous", "discrete", "binary", "categorical"]
    measurement_status: Literal["measured", "observable", "latent"]
    data_source: Optional[str] = None

class EdgeMetadata(BaseModel):
    mechanism: str
    evidence_strength: Literal["strong", "moderate", "hypothesis", "contested"]
    evidence_refs: List[UUID]
    assumptions: List[str]
    confidence: float = Field(ge=0, le=1)
    notes: Optional[str] = None

class CausalEdge(BaseModel):
    from_var: str
    to_var: str
    metadata: EdgeMetadata

class WorldModelVersion(BaseModel):
    version_id: str  # "wm_{domain}_{timestamp}"
    domain: str
    description: str
    variables: Dict[str, VariableDefinition]
    edges: List[CausalEdge]
    dag_json: Dict  # NetworkX JSON
    created_at: datetime = Field(default_factory=datetime.utcnow)
    created_by: str
    approved_by: Optional[str] = None
    status: Literal["draft", "review", "active", "deprecated"]

# Decision Support Models
class DecisionQuery(BaseModel):
    query_id: UUID = Field(default_factory=uuid4)
    text: str
    objective: str
    levers: List[str]
    constraints: List[str]
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class CausalPath(BaseModel):
    path: List[str]
    edges: List[CausalEdge]
    mechanism_chain: str
    strength: Literal["strong", "moderate", "weak"]

class DecisionRecommendation(BaseModel):
    recommendation: str
    confidence: Literal["high", "medium", "low"]
    expected_outcome: str
    causal_paths: List[CausalPath]
    evidence_refs: List[UUID]
    risks: List[str]
    unmeasured_factors: List[str]
```

### 5.2 PostgreSQL Database Schema

```sql
-- Evidence Storage
CREATE TABLE evidence_bundles (
    bundle_id UUID PRIMARY KEY,
    content TEXT NOT NULL,
    content_hash VARCHAR(64) NOT NULL,
    source_doc_id VARCHAR(255),
    source_doc_title TEXT,
    source_url TEXT,
    section_name TEXT,
    page_number INTEGER,
    retrieval_method VARCHAR(50) NOT NULL,
    retrieval_query TEXT NOT NULL,
    retrieval_timestamp TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    INDEX idx_content_hash (content_hash),
    INDEX idx_source_doc (source_doc_id)
);

-- World Models
CREATE TABLE world_model_versions (
    version_id VARCHAR(100) PRIMARY KEY,
    domain VARCHAR(100) NOT NULL,
    description TEXT,
    variables JSONB NOT NULL,
    edges JSONB NOT NULL,
    dag_json JSONB NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    created_by VARCHAR(100) NOT NULL,
    approved_by VARCHAR(100),
    status VARCHAR(20) NOT NULL,
    INDEX idx_domain (domain),
    INDEX idx_status (status)
);

-- Junction: Evidence ↔ World Models
CREATE TABLE wm_evidence_links (
    version_id VARCHAR(100) NOT NULL,
    bundle_id UUID NOT NULL,
    edge_id VARCHAR(255),
    PRIMARY KEY (version_id, bundle_id),
    FOREIGN KEY (version_id) REFERENCES world_model_versions(version_id),
    FOREIGN KEY (bundle_id) REFERENCES evidence_bundles(bundle_id)
);

-- Audit Log
CREATE TABLE audit_log (
    audit_id UUID PRIMARY KEY,
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    mode VARCHAR(50) NOT NULL,
    input_query TEXT NOT NULL,
    trace_id VARCHAR(100) NOT NULL,
    output_type VARCHAR(50),
    output_data JSONB,
    execution_time_ms INTEGER,
    INDEX idx_timestamp (timestamp DESC),
    INDEX idx_trace_id (trace_id)
);
```

---

## 6. Mode 1: World Model Construction - Complete Flow

### 6.1 Detailed Step-by-Step Process

**STEP 1: Scope Definition**

Input: User request (natural language)
Output: Structured `Scope` object

Process:
1. LLM extracts: domain, time_horizon, target_outcomes, constraints
2. User confirms/refines scope interactively
3. Scope persisted to session

LLM Prompt:
```
User Request: {request}

Extract decision-making scope. Return JSON:
{
  "domain": "pricing|product|operations|...",
  "time_horizon": "next quarter|6 months|...",
  "target_outcomes": ["revenue", "churn", ...],
  "known_constraints": ["budget limit", ...]
}
```

**STEP 2: Variable Discovery (Haystack)**

Input: Scope
Output: List of candidate variables with definitions

Process:
1. Generate 4-6 search queries from scope
   - "{domain} metrics"
   - "factors affecting {outcomes}"
   - "{domain} KPIs"
2. Execute Haystack retrieval (top_k=20 per query)
3. LLM extracts variable mentions from documents
4. Deduplicate by fuzzy name matching
5. Return ~20-30 candidate variables

LLM Prompt:
```
Documents: [retrieved_content]

Extract all variables/metrics/factors mentioned that relate to:
Domain: {scope.domain}
Outcomes: {scope.target_outcomes}

For each variable return JSON:
{
  "name": "...",
  "definition": "brief description from context",
  "type": "continuous|discrete|binary",
  "measurement_status": "measured|observable|latent"
}
```

**STEP 3: Deep Evidence Gathering**

Input: Candidate variables
Output: Dict mapping each variable to List[EvidenceBundle]

Process:
1. For each candidate variable:
   a. Check for structured docs (policies, specs) → PageIndex
   b. Broad search via Haystack
   c. Combine results
2. Store all EvidenceBundles to database
3. Return variable → evidence mapping

PageIndex Query Example:
```
query_document(
    doc_id="pricing_policy_v7",
    query="definition and relationships of price elasticity"
)
```

Haystack Query Example:
```
retrieve(
    query="price elasticity definition effects on revenue",
    top_k=5
)
```

**STEP 4: Causal Structure Drafting (PyWhyLLM)**

Input: Scope, candidates, evidence
Output: Proposed DAG with edges + assumptions

Process:
1. Initialize PyWhyLLM ModelSuggester
2. Prepare domain expertise strings from evidence
3. Call suggester.suggest_model()
4. Convert to NetworkX DAG
5. Validate acyclicity
6. For each edge, find supporting evidence
7. Assign initial confidence (0.5) and strength (hypothesis)

PyWhyLLM Usage:
```python
from pywhyllm.suggesters.model_suggester import ModelSuggester

suggester = ModelSuggester('gpt-4')

suggested_dag = suggester.suggest_model(
    all_factors=[var.name for var in candidates],
    domain_expertises=evidence_summaries,
    treatment=scope.target_outcomes[0],
    outcome=scope.target_outcomes[-1],
    RelationshipStrategy.Pairwise
)
```

Output Structure:
- DAG as NetworkX DiGraph
- Each edge has: from_var, to_var, mechanism description
- Variables mapped to VariableDefinition objects

**STEP 5: Evidence Triangulation**

Input: Proposed DAG, evidence mapping
Output: Updated DAG with evidence strength classifications

Process:
1. For each edge in DAG:
   a. Gather evidence for both variables
   b. LLM evaluates: Does evidence support this edge?
   c. Count: supporting docs, contradicting docs
   d. Classify strength: strong (3+ sources), moderate (2), hypothesis (1), contested (contradictions)
   e. Update edge metadata
2. Flag edges with low confidence for review

LLM Evaluation Prompt (per edge):
```
Causal Claim: {from_var} → {to_var}
Mechanism: {mechanism}

Evidence for {from_var}: [...]
Evidence for {to_var}: [...]

Rate evidence support. Return JSON:
{
  "strength": "strong|moderate|hypothesis|contested",
  "supporting_count": int,
  "contradicting_count": int,
  "confidence": 0-1,
  "reasoning": "..."
}
```

**STEP 6: Package for Human Review**

Input: Validated DAG
Output: ReviewPackage with visualizations + reports

Contents:
1. DAG Visualization (Graphviz PNG)
   - Nodes colored by measurement_status
   - Edges colored by evidence_strength
   - Edge labels show confidence scores

2. Variable Definitions Table (markdown)
   - Name, definition, type, measurement status

3. Edge Strength Report (table)
   - Each edge with mechanism, strength, evidence count

4. Uncertainty Flags (list)
   - Edges marked "hypothesis" or "contested"
   - Suggested next steps (gather more data, validate assumptions)

**STEP 7: Human Approval Gate**

Process:
1. Present ReviewPackage to user via UI
2. User can:
   - Approve → transition to WM_ACTIVE, commit to DB
   - Reject → provide feedback, restart from Step 4
   - Request changes → edit variables/edges, re-validate
3. On approval:
   - Assign version_id (wm_{domain}_{timestamp})
   - Set status = 'active'
   - Deprecate previous version (if exists)
   - Log audit entry

**STEP 8: Commit to Database**

Atomic transaction:
1. Insert WorldModelVersion row
2. Insert all EvidenceBundle rows (if not exist)
3. Insert wm_evidence_links junction records
4. Update previous version status to 'deprecated'
5. Create audit_log entry
6. Notify Agent Lightning of completion (for training)

---

## 7. Mode 2: Decision Support - Complete Flow

### 7.1 Detailed Step-by-Step Process

**STEP 1: Parse Decision Query**

Input: User question (natural language)
Output: Structured DecisionQuery

LLM Prompt:
```
User Question: {request}

Extract decision structure. Return JSON:
{
  "objective": "What user wants to achieve",
  "levers": ["Variables user can control"],
  "constraints": ["Known limitations"]
}
```

Example:
- Input: "Should we raise prices 10% next quarter?"
- Output: 
  ```json
  {
    "objective": "Maximize revenue while maintaining market share",
    "levers": ["price"],
    "constraints": ["Competitor pricing", "Customer retention"]
  }
  ```

**STEP 2: Load World Model Slice**

Input: DecisionQuery
Output: WorldModelSlice (subgraph containing relevant paths)

Process:
1. Identify domain from query (LLM classification)
2. Load active world model for domain from DB
3. Reconstruct NetworkX DAG from JSON
4. Match query terms to variable IDs (fuzzy matching)
5. Extract subgraph:
   - Find all paths: lever_vars → objective_vars
   - Include all nodes in these paths
   - Include confounders (common ancestors)
6. Return subgraph + variable definitions + edges

**STEP 3: Fresh Evidence Retrieval**

Input: DecisionQuery
Output: List[EvidenceBundle] from last 30 days

Process:
1. Generate search queries from decision query
2. Add date filter: >= (today - 30 days)
3. Execute Haystack retrieval
4. Check for new documents not in world model
5. Flag if significant new evidence found

Purpose:
- Detect market changes (e.g., competitor price drop)
- Find recent data (e.g., last month's churn rates)
- Validate world model is still current

**STEP 4: Check for Model Conflicts**

Input: WorldModelSlice, fresh_evidence
Output: List[Conflict] or empty

Process:
For each edge in world model:
1. LLM evaluates: Does new evidence contradict this edge?
2. If yes, create Conflict object with:
   - edge details
   - contradicting evidence references
   - explanation

Conflict Example:
```json
{
  "type": "edge_contradiction",
  "edge": "price → revenue (positive)",
  "contradicting_evidence": ["bundle_123"],
  "explanation": "Recent competitor price cut caused revenue drop despite our price stability"
}
```

Decision Logic:
- If critical conflicts found → Escalate to Mode 1 (rebuild model)
- If minor conflicts → Flag in response, continue with caution
- If no conflicts → Proceed normally

**STEP 5: Causal Reasoning**

Input: DecisionQuery, WorldModelSlice, fresh_evidence
Output: CausalReasoning with paths + analysis

Process:
1. **Find Causal Paths**:
   - Use NetworkX: all_simple_paths(lever, objective, cutoff=5)
   - For each path, extract edges
   - Calculate path strength (product of edge confidences)
   - Classify: strong (>0.7), moderate (0.4-0.7), weak (<0.4)

2. **Identify Mediators**:
   - Nodes in paths that aren't levers or objectives
   - Example: price → churn → revenue (churn is mediator)

3. **Identify Confounders**:
   - Common ancestors of lever and objective
   - Example: market_conditions → price AND revenue

4. **Flag Unmeasured Factors**:
   - Latent variables in causal paths
   - Impact: "We assume X affects Y but can't measure X"

5. **Estimate Net Effect**:
   - Combine path effects (LLM reasoning)
   - Account for mediators/confounders
   - Provide qualitative estimate

LLM Reasoning Prompt:
```
Causal Paths:
1. price → revenue (direct, strength=0.8)
2. price → churn → revenue (indirect, strength=0.6)

Confounders: market_conditions
Mediators: churn
Unmeasured: competitor_actions (latent)

Question: If we raise price 10%, what's the net effect on revenue?

Consider:
- Direct effect: Higher price per unit (+)
- Indirect effect: Higher churn reduces volume (-)
- Confounder: Market conditions may amplify/dampen effects
- Unmeasured: Competitor response unknown

Return JSON:
{
  "net_effect": "positive|negative|uncertain",
  "confidence": "high|medium|low",
  "explanation": "..."
}
```

**STEP 6: Synthesize Decision Recommendation**

Input: CausalReasoning
Output: DecisionRecommendation

LLM Synthesis Prompt:
```
Causal Analysis: {reasoning}

Task: Generate actionable recommendation.

Return JSON:
{
  "recommendation": "Clear action statement",
  "confidence": "high|medium|low",
  "expected_outcome": "What we expect to happen",
  "risks": ["risk1", "risk2"],
  "suggested_actions": ["action1", "action2"],
  "suggested_data_collection": ["data1", ...] or null
}
```

Example Output:
```json
{
  "recommendation": "Delay price increase until Q2. Monitor competitor response to market changes.",
  "confidence": "medium",
  "expected_outcome": "Maintain market share while gathering intelligence on competitive dynamics",
  "risks": [
    "Revenue opportunity cost if delay too long",
    "Competitors may raise prices first, making our increase easier"
  ],
  "suggested_actions": [
    "Conduct customer price sensitivity survey",
    "Set up competitor price monitoring dashboard"
  ],
  "suggested_data_collection": [
    "Weekly competitor pricing data",
    "Customer churn reasons (exit interviews)"
  ]
}
```

**STEP 7: Package Response**

Create DecisionResponse with:
- Recommendation
- Full reasoning trace (causal paths shown)
- Evidence links (clickable references to source docs)
- Uncertainty quantification
- Audit ID for full trace

**STEP 8: Escalation Logic**

Check if Mode 1 update needed:
```python
if (
    len(conflicts) > 0 and any(c.is_critical for c in conflicts)
    or world_model_age_days > staleness_threshold
    or critical_variables_missing_from_model
):
    return EscalationResponse(
        message="World model update recommended",
        reason="...",
        suggested_mode1_scope=build_scope_from_conflicts(conflicts)
    )
```

---

## 8. Causal Intelligence Layer

### 8.1 PyWhyLLM Integration

**Purpose:** Bridge LLM reasoning to formal causal graphs

**Architecture:**
```
LLM Causal Reasoning (narratives, mechanisms)
              ↓
    PyWhyLLM ModelSuggester
              ↓
    NetworkX DiGraph (structured DAG)
              ↓
    (Optional) DoWhy validation
```

**PyWhyLLM Workflow:**

```python
from pywhyllm.suggesters.model_suggester import ModelSuggester
from pywhyllm import RelationshipStrategy
import networkx as nx

# Step 1: Prepare inputs
all_factors = ["price", "demand", "revenue", "competition", "churn"]
domain_expertise = [
    "Price increases typically reduce demand",
    "High competition limits pricing power",
    "Churn reduces revenue through customer loss"
]
treatment = "price"
outcome = "revenue"

# Step 2: Initialize suggester (uses GPT-4 by default)
suggester = ModelSuggester(model='gpt-4')

# Step 3: Generate DAG
suggested_model = suggester.suggest_model(
    all_factors,
    domain_expertise,
    treatment,
    outcome,
    RelationshipStrategy.Pairwise  # Consider all pairs
)

# Step 4: Convert to NetworkX
G = nx.DiGraph()
for edge in suggested_model['edges']:
    G.add_edge(
        edge['from'],
        edge['to'],
        mechanism=edge.get('mechanism', ''),
        assumptions=edge.get('assumptions', [])
    )

# Step 5: Validate structure
assert nx.is_directed_acyclic_graph(G), "Graph contains cycles!"

# Step 6: (Optional) DoWhy validation
if ENABLE_DOWHY:
    from dowhy import CausalModel
    
    causal_model = CausalModel(
        data=historical_data,  # If available
        treatment=treatment,
        outcome=outcome,
        graph=G
    )
    
    # Check identification (backdoor criterion, etc.)
    identified_estimand = causal_model.identify_effect()
```

**Key Features:**
- Generates DAGs from natural language expertise
- Suggests confounders and mediators
- Creates variable definitions
- Outputs NetworkX-compatible graphs
- Optional statistical validation with DoWhy

**When DoWhy Validation is Useful:**
- Historical data exists for key variables
- Want to check if causal effect is statistically identifiable
- Need to estimate effect sizes (not just directions)
- Validating specific causal claims with data

**When to Skip DoWhy:**
- No historical data available
- Variables are latent/unmeasured
- Rapid model construction needed
- Qualitative reasoning sufficient

### 8.2 DAG Operations

**Common DAG Queries:**

```python
import networkx as nx

# 1. Find all paths from lever to outcome
def find_causal_paths(G, lever, outcome, max_length=5):
    return list(nx.all_simple_paths(G, lever, outcome, cutoff=max_length))

# 2. Identify mediators in a path
def get_mediators(path):
    return path[1:-1]  # Exclude first (lever) and last (outcome)

# 3. Find confounders (common ancestors)
def find_confounders(G, lever, outcome):
    lever_ancestors = nx.ancestors(G, lever)
    outcome_ancestors = nx.ancestors(G, outcome)
    return lever_ancestors & outcome_ancestors

# 4. Check if path is blocked by conditioning
def is_path_blocked(G, path, conditioned_on):
    # Implement d-separation check
    # (Simplified - full implementation in DoWhy)
    pass

# 5. Get all variables affecting outcome
def get_all_causes(G, outcome):
    return nx.ancestors(G, outcome)

# 6. Topological sort (causal ordering)
def get_causal_order(G):
    return list(nx.topological_sort(G))
```

### 8.3 Evidence Linking

**Challenge:** Connect each edge to supporting evidence

**Solution:**

```python
def link_evidence_to_edges(
    dag: nx.DiGraph,
    evidence_by_variable: Dict[str, List[EvidenceBundle]]
) -> List[CausalEdge]:
    """
    For each edge in DAG, find supporting evidence
    """
    edges_with_evidence = []
    
    for (from_var, to_var, edge_data) in dag.edges(data=True):
        # Get evidence mentioning both variables
        from_evidence = evidence_by_variable.get(from_var, [])
        to_evidence = evidence_by_variable.get(to_var, [])
        
        # Find evidence bundles that mention relationship
        supporting_bundles = []
        
        for bundle in from_evidence + to_evidence:
            # LLM check: Does this evidence mention the relationship?
            if mentions_relationship(bundle.content, from_var, to_var):
                supporting_bundles.append(bundle)
        
        edges_with_evidence.append(CausalEdge(
            from_var=from_var,
            to_var=to_var,
            metadata=EdgeMetadata(
                mechanism=edge_data.get('mechanism', ''),
                evidence_strength="hypothesis",  # Updated in triangulation
                evidence_refs=[b.bundle_id for b in supporting_bundles],
                assumptions=edge_data.get('assumptions', []),
                confidence=0.5
            )
        ))
    
    return edges_with_evidence

def mentions_relationship(text: str, var1: str, var2: str) -> bool:
    """
    LLM-based check if text mentions relationship between variables
    """
    prompt = f"""
    Text: {text}
    
    Question: Does this text mention or imply a relationship between 
    "{var1}" and "{var2}"?
    
    Return JSON: {{"mentions_relationship": true/false}}
    """
    
    response = llm.generate(prompt)
    return json.loads(response)["mentions_relationship"]
```

---

## 9. Agent Lightning Training Loop

### 9.1 Architecture

**Purpose:** Continuous improvement of agent behavior through RL/optimization

**Components:**

```
┌─────────────────────────────────────────┐
│     Lightning Server (Coordinator)      │
│  - Receives spans from agent runs       │
│  - Stores in LightningStore             │
│  - Triggers training jobs periodically  │
│  - Deploys improved prompts/policies    │
└─────────────────────────────────────────┘
              ↑               ↓
              │ spans     improved prompts
              │               │
┌─────────────┴───────────────▼───────────┐
│        Agent Runtime (Production)       │
│  - Executes tasks                       │
│  - Logs spans to Lightning              │
│  - Uses current prompt version          │
└─────────────────────────────────────────┘
```

### 9.2 Span Structure

**What is a Span?**
- A recorded segment of agent execution
- Contains: inputs, outputs, tool calls, reasoning, timestamps
- Hierarchical: parent spans contain child spans
- Enables full execution trace reconstruction

**Span Schema:**
```python
{
    "span_id": "uuid",
    "parent_span_id": "uuid | null",
    "trace_id": "uuid",  # Groups related spans
    "span_type": "task | tool_call | reasoning | validation",
    "start_time": "timestamp",
    "end_time": "timestamp",
    "inputs": {
        "query": "...",
        "context": {...}
    },
    "outputs": {
        "result": "...",
        "success": true/false
    },
    "metadata": {
        "tool_name": "...",
        "token_count": 123,
        "model_version": "..."
    }
}
```

**Example Span Hierarchy:**
```
task_execution (root span)
├── tool_call: haystack_retrieve
│   ├── query: "pricing factors"
│   └── result: [10 documents]
├── tool_call: pageindex_query
│   ├── query: "discount policy"
│   └── result: [policy section]
├── reasoning: draft_dag
│   ├── inputs: [variables, evidence]
│   └── output: proposed_dag
└── validation: triangulate_evidence
    ├── inputs: proposed_dag
    └── output: validated_dag
```

### 9.3 Reward Function

**Multi-Objective Reward:**

```python
def compute_reward(span_tree, ground_truth=None):
    """
    Compute reward for a complete task execution
    
    Components:
    1. Evidence Quality (+)
    2. Provenance Complete (+)
    3. Uncertainty Honest (+)
    4. Decision Clarity (+)
    5. Protocol Adherence (+)
    6. Outcome Feedback (delayed, +/-)
    """
    rewards = {}
    
    # R1: Evidence Quality (0-1)
    # Check: Every claim linked to evidence?
    claims = extract_claims(span_tree.outputs)
    evidence_linked = sum(
        1 for claim in claims 
        if has_evidence_link(claim, span_tree)
    )
    rewards['evidence_quality'] = evidence_linked / len(claims) if claims else 0
    
    # R2: Provenance Complete (0-1)
    # Check: Every evidence has source + location?
    evidence_bundles = extract_evidence(span_tree)
    complete_provenance = sum(
        1 for eb in evidence_bundles
        if eb.source and eb.location
    )
    rewards['provenance'] = complete_provenance / len(evidence_bundles) if evidence_bundles else 0
    
    # R3: Uncertainty Honest (0-1)
    # Check: Did agent flag gaps/uncertainties?
    has_uncertainty_section = 'uncertainty' in span_tree.outputs or 'risks' in span_tree.outputs
    flagged_gaps = extract_uncertainty_flags(span_tree.outputs)
    rewards['uncertainty'] = 1.0 if (has_uncertainty_section and flagged_gaps) else 0.5
    
    # R4: Decision Clarity (0-1)
    # Check: Clear recommendation + reasoning?
    has_recommendation = 'recommendation' in span_tree.outputs
    has_reasoning = 'reasoning' in span_tree.outputs or 'causal_paths' in span_tree.outputs
    rewards['clarity'] = (0.5 if has_recommendation else 0) + (0.5 if has_reasoning else 0)
    
    # R5: Protocol Adherence (0-1)
    # Check: All required steps executed?
    required_steps = get_required_protocol_steps(span_tree.metadata['mode'])
    executed_steps = [s.span_type for s in span_tree.children]
    adherence = len(set(required_steps) & set(executed_steps)) / len(required_steps)
    rewards['protocol'] = adherence
    
    # R6: Outcome Feedback (delayed, -1 to +1)
    # This comes from human feedback or downstream results
    if ground_truth:
        rewards['outcome'] = ground_truth.get('outcome_reward', 0)
    
    # Weighted combination
    weights = {
        'evidence_quality': 0.25,
        'provenance': 0.20,
        'uncertainty': 0.15,
        'clarity': 0.15,
        'protocol': 0.15,
        'outcome': 0.10  # Lower weight since delayed/sparse
    }
    
    total_reward = sum(rewards[k] * weights[k] for k in rewards)
    
    return total_reward, rewards  # Return total and breakdown
```

### 9.4 Training Process

**Workflow:**

```
1. Collection Phase (1 week)
   - Agent executes 100+ tasks
   - All spans logged to LightningStore
   - Human feedback collected for subset

2. Reward Computation
   - Batch process all completed tasks
   - Compute rewards using reward function
   - Store rewards in training dataset

3. Optimization (Agent Lightning)
   - Load dataset: (span_tree, reward) pairs
   - Run optimizer:
     * APO (Automatic Prompt Optimization) - for prompt tuning
     * PPO (Proximal Policy Optimization) - for tool selection
     * SFT (Supervised Fine-Tuning) - for specific behaviors
   - Generate improved prompts/policies

4. Evaluation
   - Test improved agent on held-out tasks
   - Compare: avg reward, task success rate
   - Human review of sample outputs

5. Deployment (if improved)
   - Deploy new prompt version
   - Monitor performance
   - Rollback if regression detected

6. Repeat weekly
```

**Agent Lightning Code:**

```python
from lightning import TrainingStore, Optimizer
from lightning.optimizers import APOOptimizer

# Initialize store
store = TrainingStore(connection_string="postgresql://...")

# Collect training data
training_data = []
for task_id in completed_tasks:
    span_tree = store.get_span_tree(task_id)
    reward, _ = compute_reward(span_tree)
    training_data.append((span_tree, reward))

# Initialize optimizer
optimizer = APOOptimizer(
    model="gemini-2.0-flash-exp",
    optimization_target="system_prompt",
    metric="reward",
    improvement_threshold=0.05  # Deploy if 5% improvement
)

# Run optimization
improved_prompt = optimizer.optimize(
    training_data=training_data,
    iterations=10,
    batch_size=20
)

# Evaluate
eval_rewards = []
for task in eval_tasks:
    result = agent.execute_with_prompt(task, improved_prompt)
    reward, _ = compute_reward(result.span_tree)
    eval_rewards.append(reward)

avg_reward_new = np.mean(eval_rewards)
avg_reward_old = np.mean(baseline_rewards)

if avg_reward_new > avg_reward_old * 1.05:
    print("Deploying improved prompt")
    deploy_prompt(improved_prompt, version="v2.1")
else:
    print("No significant improvement, keeping current prompt")
```

---

## 10. Deployment Architecture

### 10.1 Service Topology

```
┌─────────────────────────────────────────────────────┐
│                   Load Balancer                      │
│                  (nginx / AWS ALB)                   │
└────────────────┬────────────────────────────────────┘
                 │
    ┌────────────┼────────────┐
    │            │            │
┌───▼───┐   ┌───▼───┐   ┌───▼───┐
│FastAPI│   │FastAPI│   │FastAPI│  (3+ instances)
│App 1  │   │App 2  │   │App 3  │
└───┬───┘   └───┬───┘   └───┬───┘
    │           │           │
    └───────────┼───────────┘
                │
    ┌───────────┼──────────────────────────┐
    │           │                          │
┌───▼────┐ ┌───▼──────┐ ┌────▼────────┐ ┌▼─────────┐
│Postgres│ │Redis     │ │Haystack     │ │PageIndex │
│Primary │ │Cluster   │ │Service      │ │MCP       │
│+Replica│ │(3 nodes) │ │(2 instances)│ │(remote)  │
└────────┘ └──────────┘ └─────────────┘ └──────────┘
```

### 10.2 Container Configuration (Docker Compose)

```yaml
version: '3.8'

services:
  # FastAPI Application
  app:
    build: ./app
    image: decision-support-app:latest
    replicas: 3
    environment:
      - DATABASE_URL=postgresql://user:pass@postgres:5432/decision_db
      - REDIS_URL=redis://redis:6379/0
      - GOOGLE_AI_API_KEY=${GOOGLE_AI_API_KEY}
      - HAYSTACK_URL=http://haystack:8000
      - PAGEINDEX_API_KEY=${PAGEINDEX_API_KEY}
    depends_on:
      - postgres
      - redis
      - haystack
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  # PostgreSQL
  postgres:
    image: postgres:15
    environment:
      - POSTGRES_DB=decision_db
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=pass
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  # Redis
  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data
    ports:
      - "6379:6379"

  # Haystack Service
  haystack:
    build: ./haystack_service
    image: haystack-service:latest
    replicas: 2
    environment:
      - QDRANT_URL=http://qdrant:6333
    depends_on:
      - qdrant
    ports:
      - "8001:8000"

  # Qdrant Vector DB
  qdrant:
    image: qdrant/qdrant:latest
    volumes:
      - qdrant_data:/qdrant/storage
    ports:
      - "6333:6333"

  # Celery Worker (for async tasks)
  celery_worker:
    build: ./app
    command: celery -A app.tasks worker --loglevel=info
    environment:
      - DATABASE_URL=postgresql://user:pass@postgres:5432/decision_db
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      - postgres
      - redis

volumes:
  postgres_data:
  redis_data:
  qdrant_data:
```

### 10.3 Scaling Considerations

**Horizontal Scaling:**
- FastAPI: Scale to N instances behind load balancer
- Haystack: 2-4 instances with shared Qdrant backend
- Celery Workers: Scale based on async task queue depth

**Vertical Scaling:**
- PostgreSQL: Increase RAM for larger world models
- Redis: Increase memory for more cached results
- Qdrant: GPU instance for faster vector search

**Caching Strategy:**
- L1: Redis (1 hour TTL for retrieval results)
- L2: CDN (24 hour TTL for static world model visualizations)
- Invalidation: On world model update, clear related cache keys

---

## 11. Error Handling & Resilience

### 11.1 Error Categories

**1. LLM Errors**
- Rate limit exceeded
- Invalid JSON output
- Timeout (> 60s)
- Safety filter triggered

**2. Retrieval Errors**
- PageIndex: Document not found, MCP connection failed
- Haystack: Qdrant down, embedding model error
- Network timeout

**3. Data Errors**
- DAG contains cycle
- Missing required variables
- Evidence hash mismatch
- Database constraint violation

**4. Business Logic Errors**
- No world model for domain
- Contradictory evidence detected
- Confidence below threshold

### 11.2 Retry & Fallback Strategy

```python
from tenacity import retry, stop_after_attempt, wait_exponential

# LLM calls with retry
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry_error_callback=lambda retry_state: log_and_fallback(retry_state)
)
async def call_llm_with_retry(prompt):
    try:
        return await llm.generate(prompt)
    except RateLimitError:
        await asyncio.sleep(60)  # Wait 1 min
        raise
    except InvalidJSONError as e:
        # Try to fix JSON
        fixed = attempt_json_fix(e.raw_output)
        if fixed:
            return fixed
        raise

# Retrieval with fallback
async def retrieve_with_fallback(query):
    try:
        # Primary: PageIndex
        return await pageindex.query(query)
    except PageIndexError:
        logger.warning("PageIndex failed, falling back to Haystack")
        # Fallback: Haystack
        return await haystack.retrieve(query)
    except Exception:
        # Last resort: return empty with error flag
        return EvidenceBundle(error="All retrieval methods failed")

# Database operations with transaction rollback
async def save_world_model_safe(wm):
    async with db.transaction():
        try:
            await db.insert_world_model(wm)
            await db.insert_evidence_links(wm.evidence_links)
            await db.deprecate_old_version(wm.domain)
        except Exception as e:
            logger.error(f"Failed to save world model: {e}")
            await db.rollback()
            raise
```

---

## 12. Monitoring & Observability

### 12.1 Metrics to Track

**System Health:**
- Request rate (requests/minute)
- Response latency (p50, p95, p99)
- Error rate by type
- Cache hit rate
- Database connection pool utilization

**Agent Performance:**
- Average reward score (from Agent Lightning)
- Tool call success rate
- LLM token usage
- Context window utilization
- Multi-turn conversation depth

**Business Metrics:**
- World models created/updated per week
- Decision support queries per day
- Human approval rate (Mode 1)
- Evidence bundles retrieved per query
- Average evidence quality score

**Cost Tracking:**
- LLM API costs (tokens * price)
- Vector DB query costs
- Storage costs (DB + object storage)

### 12.2 Observability Stack

**OpenTelemetry Integration:**

```python
from opentelemetry import trace, metrics
from opentelemetry.exporter.prometheus import PrometheusMetricReader
from opentelemetry.sdk.metrics import MeterProvider

# Initialize tracer
tracer = trace.get_tracer(__name__)

# Trace agent execution
@tracer.start_as_current_span("agent.execute_task")
async def execute_task(query):
    span = trace.get_current_span()
    span.set_attribute("query.length", len(query))
    span.set_attribute("mode", determine_mode(query))
    
    try:
        result = await agent.run(query)
        span.set_attribute("result.success", True)
        span.set_attribute("result.tool_calls", len(result.tool_calls))
        return result
    except Exception as e:
        span.set_attribute("error", str(e))
        span.set_status(trace.Status(trace.StatusCode.ERROR))
        raise

# Metrics
meter = metrics.get_meter(__name__)

# Counters
requests_counter = meter.create_counter(
    "agent.requests.total",
    description="Total agent requests"
)

# Histograms
latency_histogram = meter.create_histogram(
    "agent.latency.seconds",
    description="Agent request latency"
)

# Usage
requests_counter.add(1, {"mode": "decision_support"})
latency_histogram.record(2.5, {"mode": "decision_support"})
```

**Grafana Dashboard Panels:**

1. **Request Volume**
   - Time series: requests/min by mode
   - Alert: > 100 req/min (rate limit risk)

2. **Latency Distribution**
   - Histogram: p50, p95, p99 latencies
   - Alert: p95 > 10s

3. **Error Rate**
   - Time series: errors/min by type
   - Alert: error rate > 5%

4. **LLM Performance**
   - Gauge: avg tokens per request
   - Time series: token cost per hour
   - Alert: daily cost > $100

5. **World Model Status**
   - Table: active models by domain
   - Time series: model updates per week
   - Gauge: average model age (days)

6. **Evidence Quality**
   - Gauge: avg evidence bundles per query
   - Histogram: evidence strength distribution
   - Time series: provenance completeness %

### 12.3 Alerting Rules

```yaml
# Prometheus alert rules
groups:
  - name: agent_alerts
    interval: 30s
    rules:
      - alert: HighErrorRate
        expr: rate(agent_errors_total[5m]) > 0.05
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "Agent error rate above 5%"
          
      - alert: SlowResponses
        expr: histogram_quantile(0.95, agent_latency_seconds) > 10
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "95th percentile latency > 10s"
          
      - alert: LLMRateLimitApproaching
        expr: rate(llm_requests_total[1m]) > 50
        for: 1m
        labels:
          severity: warning
        annotations:
          summary: "Approaching LLM rate limit (60/min)"
          
      - alert: StaleWorldModel
        expr: (time() - world_model_last_updated_timestamp) > 2592000  # 30 days
        labels:
          severity: warning
        annotations:
          summary: "World model not updated in 30 days"
```

---

## 13. End-to-End Example Scenarios

### 13.1 Scenario A: Initial World Model Creation (Mode 1)

**User Request:**
"Build a world model for our pricing decisions"

**System Execution:**

**Step 1: Scope Definition**
```
Agent → User: "To build your pricing world model, I need to understand:
- Time horizon? (next quarter, year, etc.)
- Key outcomes to optimize? (revenue, market share, profitability)
- Known constraints? (competitive pressure, cost structure, regulations)"

User: "Next 6 months, focus on revenue and customer retention, constraints are competitor pricing and cost inflation"

Agent extracts:
{
  "domain": "pricing",
  "time_horizon": "6 months",
  "target_outcomes": ["revenue", "customer_retention"],
  "known_constraints": ["competitor_pricing", "cost_inflation"]
}
```

**Step 2: Variable Discovery**
```
Haystack retrieval queries:
1. "pricing metrics KPIs"
2. "factors affecting revenue customer retention"
3. "pricing decision variables"
4. "price elasticity churn drivers"

Retrieved 47 documents, extracted 28 candidate variables:
- price
- demand
- customer_churn
- competitor_prices
- price_elasticity
- acquisition_cost
- customer_lifetime_value
- market_share
- cost_per_unit
- discount_rate
... (18 more)

Deduplicated to 23 unique variables
```

**Step 3: Deep Evidence Gathering**
```
For each variable, retrieve evidence:

"price" evidence:
- PageIndex: pricing_policy_v7.pdf, Section 2.1 "Price Setting Framework"
- PageIndex: Q3_pricing_postmortem.pdf, Section 4 "Price Changes Impact"
- Haystack: 5 documents from strategy memos

"customer_churn" evidence:
- PageIndex: customer_success_report_2024.pdf, Section 3 "Churn Analysis"
- Haystack: 8 documents from support tickets, surveys

Total: 156 evidence bundles collected
```

**Step 4: Causal Structure Drafting**
```
PyWhyLLM ModelSuggester proposed DAG with 31 edges:

Key paths:
price → revenue (direct effect)
price → customer_churn → revenue (indirect, negative)
competitor_prices → price (constraint)
competitor_prices → market_share → revenue
cost_inflation → price (cost-push)
price_elasticity (moderator) affects price→demand strength

Confounders identified:
- market_conditions affects both price and demand
- brand_strength affects price tolerance and churn

Assumptions generated:
- "Price elasticity is approximately -1.5 (from Q3 analysis)"
- "Competitor response time is 2-4 weeks"
- "Churn impact on revenue is delayed by 1 quarter"
```

**Step 5: Evidence Triangulation**
```
Edge evaluation results:

price → revenue:
- Evidence strength: STRONG (6 sources: pricing policy, 3 postmortems, 2 data analyses)
- Confidence: 0.85

price → customer_churn:
- Evidence strength: MODERATE (3 sources: customer success report, survey data, anecdotal)
- Confidence: 0.65

competitor_prices → price:
- Evidence strength: STRONG (5 sources: competitive intel reports, historical data)
- Confidence: 0.80

price_elasticity affects price→demand:
- Evidence strength: HYPOTHESIS (1 source: market research study from 2023)
- Confidence: 0.45
- Flag: "Needs validation with recent data"

... (27 more edges evaluated)

Summary:
- 12 edges: STRONG
- 11 edges: MODERATE
- 7 edges: HYPOTHESIS
- 1 edge: CONTESTED (discount_rate → customer_acquisition had contradictory evidence)
```

**Step 6: Package for Review**
```
Review Package Generated:
1. DAG Visualization (PNG): shows all variables, edges colored by strength
2. Variable Definitions Table: 23 variables with definitions
3. Edge Report: 31 edges with mechanisms, evidence counts, confidence scores
4. Uncertainty Flags:
   - "price_elasticity value needs recent validation"
   - "competitor_response timing assumption not well-supported"
   - "brand_strength measurement unclear (latent variable)"
```

**Step 7: Human Review**
```
User reviews visualization and report

User feedback: "Looks good, but we're missing 'seasonal_demand' - we see major Q4 spikes"

Agent: "Adding variable 'seasonal_demand'..."
- Re-runs evidence gathering for seasonal_demand
- Updates DAG: seasonal_demand → demand → revenue
- Re-validates with new structure

User: "Approved!"
```

**Step 8: Commit**
```
WorldModelVersion created:
- version_id: "wm_pricing_20260206_143522"
- status: "active"
- 24 variables, 32 edges
- 156 evidence bundles linked
- Audit entry logged with full trace

Previous version "wm_pricing_20250115_092314" marked deprecated

System: "World model active! Ready for decision support queries."
```

---

### 13.2 Scenario B: Decision Support Query (Mode 2)

**User Request:**
"Should we raise prices 8% next quarter?"

**System Execution:**

**Step 1: Parse Query**
```
Parsed DecisionQuery:
{
  "objective": "Determine optimal pricing strategy for next quarter",
  "levers": ["price"],
  "constraints": ["competitive pressure", "customer retention targets"],
  "context": {
    "proposed_change": "8% increase",
    "timeframe": "next quarter"
  }
}
```

**Step 2: Load World Model**
```
Loaded: wm_pricing_20260206_143522 (created today, fresh)

Extracted subgraph containing paths from "price" to "revenue" and "customer_retention":

Relevant paths:
1. price → revenue (direct)
2. price → customer_churn → customer_retention
3. price → customer_churn → revenue
4. price → demand → revenue

Relevant confounders:
- competitor_prices
- market_conditions

Relevant moderators:
- price_elasticity

Subgraph: 8 variables, 6 edges
```

**Step 3: Fresh Evidence Retrieval**
```
Haystack search (last 30 days):
- "pricing strategy"
- "competitor pricing recent"
- "customer churn Q1 2026"
- "market conditions"

Found 12 new documents:

KEY FINDING:
- competitor_price_alert_20260204.pdf: "Major competitor announced 5% price CUT starting March 1"

This is NEW information not in world model!
```

**Step 4: Conflict Detection**
```
Checking for conflicts...

CONFLICT DETECTED:
- World model assumes: competitor_prices → price (positive relationship, we follow competitors)
- New evidence: competitor cutting prices 5% while we consider +8% increase
- This creates major divergence risk

Classification: CRITICAL CONFLICT
Recommendation: Escalate to Mode 1 OR proceed with caution
```

**Step 5: Causal Reasoning (proceeding with caution)**
```
Causal paths analyzed:

Path 1: price → revenue
- Direction: Positive (higher price = higher revenue per unit)
- Strength: 0.85 (strong evidence)
- BUT: Assumes demand stays constant

Path 2: price → customer_churn → revenue
- Direction: Negative (higher price → more churn → less revenue)
- Strength: 0.65 (moderate evidence)
- Magnitude: ~2% churn increase per 5% price increase (from historical data)

Path 3: price → demand → revenue
- Direction: Negative (elastic demand)
- Strength: 0.45 (hypothesis - needs validation)
- Elasticity: -1.5 assumed

Confounders:
- competitor_prices: Now moving opposite direction! (conflict)
- market_conditions: Current state unclear from recent evidence

Net effect estimate:
+8% price with competitor at -5% (13% gap) likely causes:
- Demand drop: ~12% (using elasticity -1.5)
- Churn increase: ~3.2% (from historical pattern)
- Revenue: 0.88 * 1.08 = 0.95 (5% revenue DECREASE)

Confidence: LOW (due to untested competitive divergence)
```

**Step 6: Synthesize Decision**
```
DecisionRecommendation:
{
  "recommendation": "DO NOT raise prices 8% next quarter. Recommend holding prices steady and monitoring competitor strategy.",
  "confidence": "medium",
  "expected_outcome": "Maintain current revenue and market share while gathering competitive intelligence",
  "risks": [
    "Revenue opportunity cost if competitor price cut is temporary",
    "Margin pressure if costs continue rising without price adjustment",
    "Market may perceive our pricing as 'following' rather than 'leading'"
  ],
  "suggested_actions": [
    "Conduct rapid customer price sensitivity survey",
    "Set up weekly competitor pricing monitoring",
    "Model alternative scenarios: 0%, +3%, +5% increases",
    "Prepare contingency: if competitor reverts, implement 5% increase in Q2"
  ],
  "suggested_data_collection": [
    "Daily competitor pricing feeds",
    "Weekly churn rates by pricing tier",
    "Customer feedback on value perception"
  ]
}
```

**Step 7: Package Response**
```
DecisionResponse delivered to user:

Summary: "Based on causal analysis, an 8% price increase is HIGH RISK given competitor's recent 5% cut. Recommend holding prices steady."

Reasoning trace shown:
- Path diagrams with effect estimates
- Evidence links (click to see source documents)
- Uncertainty factors highlighted
- Alternative scenarios proposed

Audit logged: audit_id = xxx, trace_id = yyy
```

**Step 8: Escalation Triggered**
```
System recommendation: "Update world model to include competitive_divergence_risk factor"

User can click "Update World Model" to initiate Mode 1 with pre-populated scope
```

---


## 15. Key Implementation Notes

### 15.1 LLM Prompt Engineering Tips

**System Prompt Structure:**
```
You are a Decision Support Agent specializing in causal reasoning.

Your task: [Mode 1 or Mode 2 description]

Protocol:
1. [Step 1]
2. [Step 2]
...

Rules:
- ALWAYS link claims to evidence
- ALWAYS flag uncertainties
- NEVER make unsupported causal claims
- Use JSON for structured outputs

Available tools: [list]
```

**Structured Output Enforcement:**
- Use `response_mime_type: "application/json"` in Gemini config
- Provide JSON schema in prompt
- Validate and retry if malformed

**Few-Shot Examples:**
- Include 2-3 examples of good tool usage
- Show proper evidence linking format
- Demonstrate uncertainty flagging

### 15.2 Data Management Best Practices

**Evidence Deduplication:**
- Use SHA256 content hash
- Store hash in DB with index
- Before inserting, check if hash exists
- If exists, return existing bundle_id

**World Model Versioning:**
- Never delete old versions (audit trail)
- Use semantic versioning for major/minor updates
- Maintain pointer to "active" version per domain
- Allow rollback to previous version

**Audit Log Retention:**
- Keep all audit logs indefinitely (regulatory compliance)
- Partition by month for query performance
- Compress old partitions (> 6 months)
- Separate archive storage for > 2 years

### 15.3 Testing Strategy

**Unit Tests:**
- Data model validation (Pydantic)
- DAG operations (NetworkX)
- Evidence linking logic
- Reward function calculation

**Integration Tests:**
- End-to-end Mode 1 flow
- End-to-end Mode 2 flow
- Retrieval router with mocked backends
- State machine transitions

**LLM Evaluation:**
- Create test dataset of 50 queries
- Run agent on all queries
- Human eval: precision, recall, reasoning quality
- Compare versions before deploying

---

## Conclusion

This implementation plan provides comprehensive low-level details for building the Agentic Decision Support System. Key takeaways:

**Architecture Decisions:**
- Gemini 2.0 Flash for cost-effective LLM reasoning
- Dual retrieval (PageIndex + Haystack) for precision + recall
- PyWhyLLM bridges LLM narratives to formal causal graphs
- Agent Lightning enables continuous improvement

**Critical Success Factors:**
1. **Evidence Provenance:** Every claim traceable to source
2. **Uncertainty Quantification:** Honest about what we don't know
3. **Human-in-Loop:** Approval gates for high-stakes decisions
4. **Continuous Learning:** Training loop improves over time

**Next Steps:**
1. Set up development environment (Docker Compose)
2. Implement Phase 1 (foundation)
3. Start with simple Mode 2 queries
4. Gradually add Mode 1 complexity
5. Deploy Agent Lightning after sufficient data

The system is designed to be practical, auditable, and continuously improving - suitable for real business decision-making.