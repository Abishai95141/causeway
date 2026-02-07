# Causeway

Agentic decision support system that builds and maintains causal world models from documents with full audit provenance.

## What It Does

Causeway discovers causal relationships from business documents (policies, postmortems, specs) and maintains them as queryable directed acyclic graphs. When asked "what happens if we change X?", it traces causal paths and provides evidence-backed answers.

## Architecture

```
Documents → Retrieval Layer → Causal Analysis → World Model → Decision Support
                ↓                    ↓               ↓
           PageIndex/Haystack    NetworkX/PyWhyLLM   SQLAlchemy + Redis
```

### Core Components

- **PageIndex**: Structural document navigation for precise citation
- **Haystack**: Semantic retrieval across large document corpora  
- **Causal Engine**: DAG construction and path finding with NetworkX
- **Agent Orchestrator**: Multi-mode reasoning protocols
- **Storage**: PostgreSQL + Redis + MinIO

### Modes

**Mode 1 - Retrieval Augmented DAG Construction**  
Build world models from scratch using document evidence

**Mode 2 - World Model Query & Update**  
Answer questions using existing models, update when evidence changes

## Setup

```bash
# Clone
git clone https://github.com/Abishai95141/causeway.git
cd causeway

# Install
pip install -e .

# Configure
cp .env.example .env
# Edit .env with your API keys and database URLs

# Run services
docker-compose up -d

# Start API
uvicorn src.api.main:app --reload
```

## Requirements

- Python ≥3.11
- PostgreSQL
- Redis
- MinIO (or S3-compatible storage)
- Qdrant (vector database)

## API Endpoints

### Documents
- `POST /documents/upload` - Upload document
- `GET /documents/{id}` - Retrieve document
- `POST /documents/{id}/index` - Index with PageIndex/Haystack

### World Models
- `POST /world-models` - Create new model
- `GET /world-models/{id}` - Get model state
- `POST /world-models/{id}/nodes` - Add causal node
- `POST /world-models/{id}/edges` - Add causal edge

### Queries
- `POST /queries` - Ask causal question
- `GET /queries/{id}` - Get query result with evidence chain

### Training
- `POST /training/trajectories` - Submit agent trajectory
- `POST /training/spans/start` - Start span
- `POST /training/spans/end` - End span with metadata

## Project Structure

```
src/
├── agent/          # LLM orchestration and context management
├── api/            # FastAPI routes and main app
├── causal/         # DAG engine and path finding
├── extraction/     # Document text extraction
├── haystack_svc/   # Haystack RAG pipelines
├── models/         # SQLAlchemy ORM models
├── modes/          # Mode 1 & Mode 2 implementations
├── pageindex/      # PageIndex client wrapper
├── protocol/       # Mode routing and state machine
├── retrieval/      # Retrieval strategy router
├── storage/        # Database, cache, object store
└── training/       # Trajectory logging and rewards

tests/              # Unit and integration tests
prototype/          # Streamlit UI prototype
migrations/         # Alembic database migrations
```

## Development

```bash
# Run tests
pytest

# Type checking
mypy src/

# Format
black src/ tests/
isort src/ tests/

# Run with hot reload
uvicorn src.api.main:app --reload --port 8000
```

## Environment Variables

Key variables (see `.env.example`):

```bash
DATABASE_URL=postgresql+asyncpg://...
REDIS_URL=redis://localhost:6379
MINIO_ENDPOINT=localhost:9000
QDRANT_URL=http://localhost:6333
GEMINI_API_KEY=your_key
```

## Documentation

- [System Documentation](systemdoc.md) - Full architecture overview
- [Remediation Plan](REMEDIATION_PLAN.md) - Current refactoring status
- [Build Log](agent_state/BUILD_LOG.md) - Development history

## License

MIT
