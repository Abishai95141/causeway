# Agentic Decision Support System: World Model Architecture

**Purpose**: Business decision support through maintained causal world models with full audit provenance.

**Core Principle**: LLMs can discover causal relationships through rigorous retrieval + reasoning protocols, without requiring statistical validation for every claim.

---

## Component Analysis

### PageIndex
**What it is**: Document navigation system using structural reasoning instead of vector embeddings.
- Repository: https://github.com/VectifyAI/PageIndex
- MCP Server: https://github.com/VectifyAI/pageindex-mcp

**How it works**:
- Builds tree-based index of document structure (sections, headings, pages)
- LLM navigates by asking "what sections exist?" then "read section X"
- No chunking, no vectors, full page-level provenance

**When to use in this system**:
- Policy documents, postmortems, technical specs, contracts
- When you need exact citation (section 3.2.1 of Q4-2024-Pricing-Policy.pdf)
- Long-form documents where context hierarchy matters

**Integration point**: Evidence retrieval for world model construction where audit trail must show "claim X comes from document Y, section Z, page N"

---

### Haystack
**What it is**: RAG pipeline framework with hybrid retrieval + tool-calling capabilities.
- Documentation: https://haystack.deepset.ai/
- Repository: https://github.com/deepset-ai/haystack
- RAG Tutorial: https://haystack.deepset.ai/tutorials/27_first_rag_pipeline

**How it works**:
- Pipeline builder for retrieval-augmented generation
- Supports vector DBs (Qdrant, Pinecone, Weaviate), keyword search, hybrid
- Built-in document processing, embedding, reranking
- Tool-calling interfaces for agentic workflows

**When to use in this system**:
- Large document corpora (100+ docs)
- When recall across many sources matters more than exact provenance
- Fast semantic search for variable discovery
- Initial filtering before deep PageIndex navigation

**Integration point**: Broad retrieval for candidate node discovery, cross-document pattern finding, quick context gathering

---

### PyWhyLLM
**What it is**: Experimental library integrating LLMs into causal analysis workflows.
- Repository: https://github.com/py-why/pywhyllm
- Integrates with: DoWhy, causal-learn (PyWhy ecosystem)

**How it works**:
- LLM-assisted DAG construction from business context
- Variable naming, confounder suggestion, assumption drafting
- Bridges natural language descriptions to causal graph structures
- Works with existing PyWhy tools (DoWhy for identification/estimation)

**When to use in this system**:
- Translating business narratives into causal hypotheses
- Drafting initial world model structures
- Suggesting confounders/mediators based on domain knowledge
- Generating assumptions that humans then validate

**Integration point**: World model construction phase - LLM proposes causal structures, PyWhyLLM formats them into proper causal graphs with metadata

---

### Agent Lightning
**What it is**: Framework for training LLM agents through optimization loops.
- Documentation: https://microsoft.github.io/agent-lightning/
- Repository: https://github.com/microsoft/agent-lightning
- Training Guide: https://microsoft.github.io/agent-lightning/latest/how-to/train-first-agent/

**How it works**:
- Agent executes rollout (task attempt)
- System captures detailed spans (execution trace)
- Reward signal generated from outcome
- Optimizer improves agent behavior (RL, APO, SFT)
- Birds-eye view: https://microsoft.github.io/agent-lightning/latest/deep-dive/birds-eye-view/

**When to use in this system**:
- Training the agent to follow decision protocols correctly
- Optimizing retrieval strategies over time
- Learning which evidence patterns lead to good decisions
- Capturing full execution traces for audit

**Integration point**: Training loop for protocol adherence, automatic audit trail generation through span capture, continuous improvement of decision quality

---

## System Architecture

### Core Components

```
┌─────────────────────────────────────────────────────────────┐
│                     Orchestration Layer                      │
│  ┌────────────┐  ┌──────────────┐  ┌───────────────────┐   │
│  │  Protocol  │  │ Agent Runtime│  │ Agent Lightning   │   │
│  │  Engine    │  │  (ex-gemini)    │  │  Trainer          │   │
│  └────────────┘  └──────────────┘  └───────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
┌───────▼─────┐    ┌────────▼────────┐   ┌─────▼──────────┐
│  Retrieval  │    │   Causal        │   │   Storage &    │
│  Layer      │    │   Intelligence  │   │   Audit        │
│             │    │   Layer         │   │                │
│ PageIndex   │    │ PyWhyLLM        │   │ World Model DB │
│ Haystack    │    │ (Optional:      │   │ Evidence Store │
│ Router      │    │  DoWhy/SCD)     │   │ Audit Ledger   │
└─────────────┘    └─────────────────┘   └────────────────┘
        │                                          │
        └─────────────┬────────────────────────────┘
                      │
              ┌───────▼────────┐
              │  Data Sources  │
              │  - Structured  │
              │  - Documents   │
              │  - Event Logs  │
              └────────────────┘
```

---

## Two Operating Modes

### Mode 1: World Model Construction

**Purpose**: Build/update causal world model through evidence-based discovery.

**Protocol Flow**:

1. **Scope Definition**
   - Decision domain (pricing, product, operations)
   - Time horizon
   - Target outcomes
   - Known constraints

2. **Variable Discovery** (Haystack + PageIndex)
   - Haystack: broad search across all documents for candidate variables
   - Extract: mentioned metrics, KPIs, factors, mechanisms
   - Output: `CandidateVariableList` with document references

3. **Deep Evidence Gathering** (PageIndex)
   - For each candidate variable: navigate to source documents
   - Extract: definitions, relationships, historical context
   - Output: `EvidenceBundle` per variable with page-level citations

4. **Causal Structure Drafting** (PyWhyLLM)
   - LLM proposes: nodes, edges, mechanisms, confounders
   - PyWhyLLM formats: proper DAG structure with metadata
   - Includes: variable definitions, assumed relationships, uncertainty markers
   - Output: `ProposedWorldModel` (DAG + assumptions + evidence links)

5. **Optional Validation** (if data available)
   - If structured data exists: run correlation checks, temporal precedence
   - If experiments exist: validate predicted vs actual effects
   - NOT required - absence doesn't block model acceptance

6. **Evidence Triangulation**
   - Cross-check: document claims vs data patterns vs domain knowledge
   - Flag: contradictions, gaps, weak support
   - Classify edges: {strong evidence, moderate, hypothesis, contested}

7. **Model Packaging**
   - Freeze: graph structure + node definitions + edge classifications
   - Link: every edge to evidence bundles
   - Version: `WM_v{timestamp}`
   - Output: `WorldModelVersion` ready for approval

8. **Human Review Gate**
   - Present: visual graph + evidence summaries + uncertainty flags
   - Approve/Reject/Request Changes
   - On approval: commit to World Model DB

**State Transitions**: `IDLE → DISCOVERY_RUNNING → REVIEW_PENDING → MODEL_ACTIVE`

---

### Mode 2: Decision Support

**Purpose**: Answer decision questions using current world model + fresh evidence.

**Protocol Flow**:

1. **Query Intake**
   - User question about decision
   - Extract: objective, levers (controllable variables), constraints
   - Output: `DecisionQuery` structure

2. **Model Retrieval**
   - Load: relevant world model slice (subgraph)
   - Check: model freshness, coverage of query variables
   - Output: `WorldModelSlice` + metadata

3. **Evidence Refresh** (Haystack + PageIndex)
   - Haystack: quick search for recent relevant documents (last 30 days)
   - PageIndex: navigate to specific updated sections in key docs
   - Check: does new evidence contradict current model?
   - Output: `FreshEvidenceBundle`

4. **Causal Reasoning**
   - Apply world model: trace paths from levers to objectives
   - Identify: mediators, confounders, effect modifiers
   - Account for: uncertainty in edges, unmeasured factors
   - If model gaps detected: surface them explicitly

5. **Decision Synthesis**
   - Recommendation: based on causal paths + evidence
   - Confidence: classified by evidence strength
   - Risks: what could go wrong, unmeasured confounders
   - Next steps: if uncertainty high, recommend data gathering or model update

6. **Escalation Logic**
   ```
   IF: critical variable missing from model
       OR: new evidence contradicts model structure
       OR: model age > staleness_threshold
   THEN: recommend Mode 1 (world model update)
   ELSE: provide decision with uncertainty bounds
   ```

7. **Response Packaging**
   - Decision recommendation + reasoning trace
   - Evidence citations (with document + section links)
   - Uncertainty quantification
   - Audit trail reference

**State Transitions**: `QUERY_RECEIVED → RETRIEVING_CONTEXT → REASONING → RESPONSE_READY`

---

## Integration Architecture

### 1. Retrieval Router

**Purpose**: Choose retrieval strategy based on query and document type.

**Decision Logic**:
```
IF: query targets specific document (policy, contract, spec)
    AND: document is structured/long-form
THEN: use PageIndex for precise navigation

IF: query needs broad recall across many documents  
    OR: semantic similarity matters more than structure
THEN: use Haystack hybrid retrieval

IF: query needs both breadth and depth
THEN: Haystack for candidate filtering → PageIndex for deep navigation
```

**Output Contract** (unified across both):
```
EvidenceBundle {
  content: extracted text
  source: {doc_id, doc_title, url/path}
  location: {section_name, page_number, paragraph_id}
  retrieval_trace: how this was found
  timestamp: when retrieved
  hash: content fingerprint
}
```

---

### 2. Causal Intelligence Pipeline

**PyWhyLLM Integration**:
- Input: business context + candidate variables + evidence bundles
- Process: LLM drafts causal graph with PyWhyLLM structure
- Output: formal DAG with node metadata, edge types, assumptions

**Optional Enhancement** (not required):
- If DoWhy installed: can run identification checks (backdoor criterion, etc.)
- If causal-learn installed: can suggest edges based on data patterns
- If neither: pure LLM reasoning with evidence is acceptable

**Key Point**: Statistical validation is optional enhancement, not requirement.

---

### 3. Agent Lightning Training Loop

**Span Capture** (automatic audit):
- Every protocol step generates a span
- Span includes: tool calls, retrievals, reasoning steps, outputs
- Spans linked to form full execution trace

**Reward Function** (multi-objective):
- **Evidence Quality** (+): every causal claim links to evidence bundle
- **Provenance Complete** (+): all evidence has source + location
- **Uncertainty Honest** (+): gaps/weaknesses explicitly surfaced
- **Decision Clarity** (+): clear recommendation with reasoning
- **Protocol Adherence** (+): all required steps executed
- **Outcome Feedback** (delayed): did decision lead to expected result?

**Training Process**:
1. Agent executes decision support rollouts
2. System captures spans + computes rewards
3. Periodically: run optimization (APO for prompt tuning, RL for tool selection)
4. Improved agent deployed to production

**Benefit**: System learns better retrieval strategies, evidence synthesis, and uncertainty calibration over time.

---

## Storage Architecture

### World Model Database
```
WorldModelVersion {
  version_id: "wm_2024-12-15_v7"
  domain: "pricing_decisions"
  graph: {nodes, edges, structure}
  nodes: {
    variable_id: {name, definition, type, measurement_status}
  }
  edges: {
    from -> to: {
      mechanism: description
      evidence_strength: {strong|moderate|hypothesis}
      evidence_refs: [evidence_bundle_ids]
      assumptions: [list]
    }
  }
  metadata: {created, approved_by, replaces_version}
}
```

### Evidence Store
```
EvidenceBundle {
  bundle_id: uuid
  content: text
  source: document reference
  location: precise citation
  retrieval_trace: how found
  used_in: [world_model_versions]
  hash: content fingerprint
}
```

### Audit Ledger (append-only)
```
AuditEntry {
  timestamp
  mode: "world_model_construction" | "decision_support"
  trace_id: links to Agent Lightning span
  inputs: query/scope
  retrievals: [evidence_bundle_ids]
  reasoning_steps: protocol execution trace
  outputs: model_version or decision_response
  agent_version: which model/prompt was used
}
```

---

## End-to-End Flows

### Flow A: Initial World Model Creation

```
1. User: "Build world model for pricing decisions"
   
2. Protocol Engine: initiate Mode 1
   
3. Scope Definition
   → Agent: "What pricing outcomes matter?"
   → User: "Revenue, market share, churn"
   
4. Variable Discovery
   → Haystack: search all docs for "pricing", "revenue", "churn", "market"
   → Returns: 50 candidate documents
   → Agent: extracts 25 candidate variables
   
5. Deep Evidence
   → PageIndex: navigate to pricing policy doc, section 3.1
   → PageIndex: navigate to Q3 postmortem, "churn analysis" section
   → Builds evidence bundles with precise citations
   
6. Causal Drafting (PyWhyLLM)
   → Agent: proposes DAG
     price → revenue
     price → churn → revenue
     market_conditions → price (confounder)
   → PyWhyLLM: formats as proper causal graph
   
7. Triangulation
   → Check: do documents support these edges?
   → Edge "price → churn": strong (3 postmortems, pricing doc)
   → Edge "market_conditions → price": hypothesis (mentioned but unclear)
   
8. Package & Review
   → Visual graph shown to user
   → Evidence links visible
   → User approves
   
9. Commit
   → WorldModelVersion created
   → Audit entry logged
   → Agent Lightning captures full span
```

### Flow B: Decision Support Query

```
1. User: "Should we raise prices 10% next quarter?"
   
2. Protocol Engine: initiate Mode 2
   
3. Model Retrieval
   → Load: pricing world model v7
   → Check: last updated 2 weeks ago (fresh enough)
   
4. Evidence Refresh
   → Haystack: search last 30 days for "pricing", "market", "competition"
   → Finds: competitor price drop announcement (3 days ago)
   → PageIndex: navigate to new market analysis doc
   
5. Causal Reasoning
   → Trace: price ↑ → revenue (depends on elasticity)
   → Trace: price ↑ → churn ↑ → revenue ↓
   → New evidence: competitor prices dropped (confounding factor)
   → Net effect uncertain given market dynamics
   
6. Synthesis
   → Recommendation: "Delay price increase pending competitive response analysis"
   → Confidence: Low (new market factor not in model)
   → Risks: churn spike if competitors stay low
   → Next step: update world model with competitive pricing dynamics
   
7. Response
   → User gets: recommendation + reasoning + evidence links
   → Audit entry logged with full trace
   
8. Escalation Triggered
   → System: "Recommend world model update to include competitor pricing"
```

---

## Practical Deployment Considerations

### Retrieval Service Configuration
- **PageIndex**: Deploy as MCP server (https://github.com/VectifyAI/pageindex-mcp)
- **Haystack**: Deploy as API service with document processor pipeline
- **Router**: Simple decision logic in orchestration layer

### Agent Configuration
- **Base Model**: gemini free api (or equivalent reasoning-capable LLM)
- **Agent Lightning**: Training store + reward grader + optimizer
- **Protocol**: Defined as system prompts + tool schemas

### Data Integration
- **Structured**: Read-only query service with snapshot capability
- **Documents**: File system or object storage indexed by both systems
- **Real-time**: Event stream connectors (optional)

### Human-in-Loop Points
- World model approval (required)
- High-stakes decision review (optional)
- Contradiction resolution (as needed)

---

## Key Design Decisions

**Why allow LLM causal discovery?**
- Business causality often documented but not measured
- LLM + retrieval can surface mechanisms from postmortems, policies, expert docs
- Statistical tests aren't always possible (small n, unmeasured variables)
- Explicit evidence linking + uncertainty labeling maintains rigor

**Why two modes?**
- World model construction is slow, deliberate, reviewed
- Decision support must be fast, using cached knowledge
- Clear separation prevents "hallucinated models" while enabling responsive decisions

**Why Agent Lightning?**
- Automatic audit through span capture
- Continuous improvement of retrieval and reasoning
- Systematic protocol adherence training
- Avoids prompt drift and regression

**Why both PageIndex and Haystack?**
- Different retrieval paradigms for different needs
- PageIndex: precision + provenance for audit
- Haystack: recall + speed for discovery
- Router chooses based on task requirements

**Why PyWhyLLM?**
- Bridges business language to causal formalism
- Leverages PyWhy ecosystem if statistical validation desired
- Maintains structured causal graphs (not just text descriptions)
- Optional: system works without it, just less structured

---

**End of System Design**