# Causeway Codebase Audit

Welcome to the **Causeway** codebase! Causeway is an Agentic Decision Support System built around causal world models. It extracts variables and causal edges from documents, constructs Directed Acyclic Graphs (DAGs), and uses them to answer decision-oriented questions with traceable reasoning.

This audit breaks down the entire repository so you can understand what each section does, how the pieces fit together, and where to look when you want to make changes or study the system.

---

## 1. High-Level Architecture
At a high level, the system consists of three main parts:
- **Backend API (`src/`)**: A Python/FastAPI backend that handles all the heavy lifting—document ingestion, causal graph construction, LLM interactions, and data storage.
- **Frontend UI (`frontend/ui/`)**: A modern React application that provides the user interface for exploring causal models, uploading documents, and interacting with the decision support features.
- **Infrastructure (`docker-compose.yml`)**: Containerized services like PostgreSQL (relational data), Redis (caching/queues), MinIO (object storage for documents), and Qdrant (vector database for semantic search).

---

## 2. The Backend (`src/`)

The core intelligence of Causeway lives in `src/`. It is structured into multiple specialized modules:

### `src/api/` (API Layer)
This is the entry point for the backend.
- **`main.py`**: Initializes the FastAPI application, sets up CORS, and handles lifespan events (like DB initialization).
- **`routes.py`**: Defines all the REST API endpoints (e.g., `/api/v1/documents`, `/api/v1/mode1/run`) that the frontend calls.

### `src/causal/` (Causal Intelligence Engine)
This is the heart of the system where causal graphs are managed.
- **`dag_engine.py`**: The `DAGEngine` uses `NetworkX` to build logic maps (DAGs). It prevents loops (cycles) and tracks variables, causal edges, and evidence metadata.
- **Other files**: Code here likely deals with resolving conflicting evidence (`conflict_resolver.py`), finding paths through the graph (`path_finder.py`), and communicating hypothesis testing.

### `src/extraction/` (LLM Information Extraction)
Responsible for pulling structured data out of unstructured text.
- **`service.py`**: The `ExtractionService` is a centralized wrapper around `langextract` and Gemini-2.5-flash. It uses detailed prompts to extract **variables**, **causal edges**, and **recommendations**. It contains logic to deduplicate variables and ensure the LLM correctly cites exact quotes as "evidence".

### `src/haystack_svc/` (Hybrid Search & Retrieval)
Handles searching through the ingested documents to find evidence.
- **`service.py` (`HaystackService`)**: Handles semantic search using a Vector DB (Qdrant) and keyword search (BM25). It includes advanced features like "Hypothesis-aware retrieval", which searches the corpus for both supporting and refuting evidence for a given causal claim.

### `src/protocol/` (Orchestration & Routing)
Controls the overall flow of the application.
- **`mode_router.py`**: Looks at user queries and decides if the system should be in **Mode 1** (building a world model from documents) or **Mode 2** (answering a decision question using an existing model).
- **`state_machine.py`**: Likely manages the steps and states required to complete a complex task (like the multi-step process of reading a document, extracting data, and updating a graph).

### `src/agent/` & `src/models/` and other directories
- **`agent/`**: Contains the logic for the autonomous agents orchestrating tasks and interacting with LLMs.
- **`models/`**: Pydantic data models used across the system (e.g., schemas for API responses, database records, and causal edges).
- **`storage/`**: Interfaces with Postgres (metadata/graph state), MinIO (raw PDFs), and Qdrant (embeddings).

---

## 3. The Frontend (`frontend/ui/`)

Located in `frontend/ui/`, the frontend is a single-page application built with modern web tools:
- **Core Tech**: React 18, TypeScript, and Vite (for fast building).
- **Styling & UI**: Tailwind CSS combined with `shadcn/ui` (accessible, customizable UI components like buttons, dialogs, and forms) and Radix UI primitives.
- **Data Fetching**: `@tanstack/react-query` handles caching and communicating with the FastAPI backend.
- **Graph Visualization**: Uses `@xyflow/react` (React Flow) and `dagre` to visually render the causal nodes and edges so users can inspect the World Models interactively.

---

## 4. Infrastructure & Root Files

### `docker-compose.yml`
Defines the local development environment so you can spin up all dependencies with `docker compose up -d`. It includes:
- **`postgres`**: Stores graph metadata and application state.
- **`redis`**: Likely used for caching or task queues.
- **`minio`**: An S3-compatible object store where uploaded PDFs are saved.
- **`qdrant`**: A high-performance vector database used by `haystack_svc` to perform semantic searches over document chunks.

### `prototype/`
Contains a Streamlit application (`app.py`). This was likely an earlier prototype of the system used to validate the LLM pipelines and backend logic before the React frontend was built.

### `tests/`
A comprehensive Python test suite using `pytest`. It includes unit tests for the causal engine (`test_causal_agent.py`), the extraction services, the retrieval services (`test_retrieval.py`), and end-to-end integration tests (`test_e2e_hobby_farm.py`). 

### `migrations/`
Intended for database schema migrations (likely via `alembic`), though currently empty.

### Documentation Files (`.md`)
The root contains several rich markdown files:
- **`README.md`**: The primary entry point explaining how to set up, run, and deploy the system.
- **`ROOT_CAUSE_ANALYSIS.md` & `TRUNCATION_AUDIT.md`**: Deep-dive technical documents likely written to document specific debugging sessions or system audits.
- **`systemdoc.md` & `lowlevel-systemdoc.md`**: Detailed architectural specifications of the system flow.

---

## 5. Summary Workflows

To tie it all together, here is how the system typically operates:

1. **Ingestion**: A user uploads a PDF via the React UI. The FastAPI backend saves the file to MinIO, chunks the text, creates vector embeddings, and stores them in Qdrant via `haystack_svc`.
2. **Mode 1 (Model Construction)**: The user asks to build a model. The system pulls relevant chunks, sends them to `extraction/service.py` to ask the LLM to identify variables and edges, and iteratively adds them to a NetworkX graph via `causal/dag_engine.py`.
3. **Mode 2 (Decision Support)**: The user asks "What happens if we increase Staff Training?". `protocol/mode_router.py` detects a Mode 2 question. The system queries the existing DAG to find causal paths, validates them against evidence in Qdrant, and uses the LLM to synthesize a recommendation.

---
*Created by your AI Assistant to help you understand the Causeway framework.*
