"""
API Routes

Implements all REST endpoints:
- Document upload and management
- Indexing triggers
- Mode 1/2 execution
- World model retrieval

All document metadata is persisted in PostgreSQL via DatabaseService.
"""

import hashlib
import logging
from datetime import datetime, timezone
from typing import Any, Optional
from uuid import UUID, uuid4

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from pydantic import BaseModel, Field
from sqlalchemy import String

from src.config import get_settings
from src.storage.object_store import ObjectStore
from src.storage.database import get_db_session, DatabaseService
from src.retrieval.router import RetrievalRouter
from src.modes.mode1 import Mode1WorldModelConstruction, Mode1Stage
from src.modes.mode2 import Mode2DecisionSupport, Mode2Stage
from src.causal.service import CausalService
from src.protocol.state_machine import ProtocolStateMachine
from src.protocol.mode_router import ModeRouter
from src.agent.causeway_agent import CausewayAgent
from src.models.enums import IngestionStatus, ModelStatus, OperatingMode
from src.utils.text import truncate_evidence

logger = logging.getLogger(__name__)

router = APIRouter()

# Global service instances (initialized lazily)
_object_store: Optional[ObjectStore] = None
_retrieval_router: Optional[RetrievalRouter] = None
_causal_service: Optional[CausalService] = None
_mode1: Optional[Mode1WorldModelConstruction] = None
_mode2: Optional[Mode2DecisionSupport] = None
_protocol_sm: Optional[ProtocolStateMachine] = None
_mode_router: Optional[ModeRouter] = None
_causeway_agent: Optional[CausewayAgent] = None


def get_protocol_sm() -> ProtocolStateMachine:
    """Get or create the protocol state machine singleton."""
    global _protocol_sm
    if _protocol_sm is None:
        _protocol_sm = ProtocolStateMachine()
    return _protocol_sm


def get_mode_router() -> ModeRouter:
    """Get or create the mode router."""
    global _mode_router
    if _mode_router is None:
        _mode_router = ModeRouter()
    return _mode_router


async def get_causeway_agent() -> CausewayAgent:
    """Get or create the CausewayAgent singleton (agentic entrypoint)."""
    global _causeway_agent
    if _causeway_agent is None:
        _causeway_agent = CausewayAgent(
            causal_service=get_causal_service(),
            retrieval_router=await get_retrieval_router(),
            mode_router=get_mode_router(),
        )
        await _causeway_agent.initialize()
    return _causeway_agent


async def get_object_store() -> ObjectStore:
    """Get or create object store client."""
    global _object_store
    if _object_store is None:
        _object_store = ObjectStore()
    return _object_store


async def get_retrieval_router() -> RetrievalRouter:
    """Get or create retrieval router."""
    global _retrieval_router
    if _retrieval_router is None:
        _retrieval_router = RetrievalRouter()
        await _retrieval_router.initialize()
    return _retrieval_router


def get_causal_service() -> CausalService:
    """Get or create causal service."""
    global _causal_service
    if _causal_service is None:
        _causal_service = CausalService()
    return _causal_service


async def get_mode1() -> Mode1WorldModelConstruction:
    """Get or create Mode 1 instance."""
    global _mode1
    if _mode1 is None:
        _mode1 = Mode1WorldModelConstruction(
            causal_service=get_causal_service(),
        )
        await _mode1.initialize()
    return _mode1


async def get_mode2() -> Mode2DecisionSupport:
    """Get or create Mode 2 instance."""
    global _mode2
    if _mode2 is None:
        _mode2 = Mode2DecisionSupport(
            causal_service=get_causal_service(),
        )
        await _mode2.initialize()
    return _mode2


# ===== Request/Response Models =====

class DocumentResponse(BaseModel):
    """Response for document operations."""
    doc_id: str
    filename: str
    content_hash: str
    storage_uri: str
    status: str
    created_at: str


class IndexRequest(BaseModel):
    """Request to index a document."""
    doc_id: str
    filename: Optional[str] = None


class IndexResponse(BaseModel):
    """Response for indexing operations."""
    doc_id: str
    status: str
    message: str


class Mode1Request(BaseModel):
    """Request to run Mode 1."""
    domain: str = Field(..., description="Decision domain (e.g., 'pricing')")
    initial_query: str = Field(..., description="Starting query for evidence")
    max_variables: int = Field(default=20, ge=1, le=100)
    max_edges: int = Field(default=50, ge=1, le=200)
    doc_ids: Optional[list[str]] = Field(default=None, description="Restrict evidence to these document IDs")


class Mode1Response(BaseModel):
    """Response from Mode 1 execution."""
    trace_id: str
    domain: str
    stage: str
    variables_discovered: int
    edges_created: int
    evidence_linked: int
    requires_review: bool
    error: Optional[str] = None


class Mode2Request(BaseModel):
    """Request to run Mode 2."""
    query: str = Field(..., description="Decision question")
    domain_hint: Optional[str] = Field(default=None, description="Optional domain hint")


class Mode2Response(BaseModel):
    """Response from Mode 2 execution."""
    trace_id: str
    query: str
    stage: str
    recommendation: Optional[str] = None
    confidence: Optional[str] = None
    model_used: Optional[str] = None
    evidence_count: int = 0
    escalate_to_mode1: bool = False
    escalation_reason: Optional[str] = None
    error: Optional[str] = None


class WorldModelSummary(BaseModel):
    """Summary of a world model."""
    domain: str
    version_id: Optional[str] = None
    node_count: int
    edge_count: int
    status: str
    variables: list[str]


class ApprovalRequest(BaseModel):
    """Request to approve a world model."""
    domain: str
    approved_by: str


# ── Rich detail models for human review ──────────────────────────

class VariableDetail(BaseModel):
    """Full variable info for human review."""
    variable_id: str
    name: str
    definition: str
    var_type: Optional[str] = None
    role: Optional[str] = None

class EdgeDetail(BaseModel):
    """Full edge info for human review."""
    from_var: str
    to_var: str
    mechanism: str
    strength: Optional[str] = None
    confidence: Optional[float] = None

class WorldModelDetail(BaseModel):
    """Rich model detail including variable/edge definitions."""
    domain: str
    version_id: Optional[str] = None
    node_count: int
    edge_count: int
    status: str
    variables: list[VariableDetail]
    edges: list[EdgeDetail]


# ===== Phase 6 Response Models (declared early for route ordering) =====

class BridgeSummary(BaseModel):
    """Summary of a bridge for listing."""
    bridge_id: str
    source_version_id: str
    target_version_id: str
    edge_count: int
    concept_count: int
    status: str
    created_at: Optional[str] = None


# ===== Search Endpoint =====

class SearchRequest(BaseModel):
    """Request for direct evidence search."""
    query: str = Field(..., description="Search query")
    max_results: int = Field(default=5, ge=1, le=50)
    doc_id: Optional[str] = Field(default=None, description="Filter to a specific document")


class SearchResult(BaseModel):
    """A single search result."""
    content: str
    doc_id: str
    doc_title: str
    score: float = 0.0
    section: Optional[str] = None
    page: Optional[int] = None


class SearchResponse(BaseModel):
    """Response from search endpoint."""
    query: str
    total_results: int
    results: list[SearchResult]


@router.post("/search", response_model=SearchResponse)
async def search_evidence(request: SearchRequest):
    """
    Direct semantic search over indexed documents.
    No LLM required — just retrieval.
    """
    retrieval = await get_retrieval_router()

    if request.doc_id:
        from src.retrieval.router import RetrievalRequest, RetrievalStrategy
        bundles = await retrieval.retrieve(
            RetrievalRequest(
                query=request.query,
                doc_ids=[request.doc_id],
                strategy=RetrievalStrategy.HAYSTACK_ONLY,
                max_results=request.max_results,
            )
        )
    else:
        bundles = await retrieval.retrieve_simple(
            query=request.query,
            max_results=request.max_results,
        )

    results = [
        SearchResult(
            content=truncate_evidence(b.content, max_chars=1200),
            doc_id=b.source.doc_id,
            doc_title=b.source.doc_title,
            score=(
                b.retrieval_trace.scores.get("vector")
                or b.retrieval_trace.scores.get("fused")
                or b.retrieval_trace.scores.get("similarity")
                or max(b.retrieval_trace.scores.values(), default=0.0)
            ) if b.retrieval_trace.scores else b.retrieval_trace.confidence,
            section=b.location.section_name,
            page=b.location.page_number,
        )
        for b in bundles
    ]

    return SearchResponse(
        query=request.query,
        total_results=len(results),
        results=results,
    )


# ===== Document Endpoints =====

@router.post("/uploads", response_model=DocumentResponse)
async def upload_document(
    file: UploadFile = File(...),
    description: str = Form(default=""),
):
    """
    Upload a document for processing.
    
    Supported formats: PDF, TXT, MD, XLSX
    Persists metadata to PostgreSQL and binary to MinIO.
    """
    # Validate file type
    allowed_types = {".pdf", ".txt", ".md", ".xlsx", ".csv"}
    filename = file.filename or "unknown"
    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    
    if ext not in allowed_types:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File type not supported. Allowed: {allowed_types}",
        )
    
    # Read and hash content
    content = await file.read()
    content_hash = hashlib.sha256(content).hexdigest()
    
    # Generate document ID (string-based for consistency)
    internal_id = uuid4()
    doc_id = f"doc_{internal_id.hex[:12]}"
    content_type = file.content_type or "application/octet-stream"
    
    # Upload binary to object store
    store = await get_object_store()
    storage_uri = store.upload_bytes(
        doc_id=doc_id,
        filename=filename,
        content=content,
        content_type=content_type,
    )
    
    # Persist metadata to PostgreSQL
    try:
        async with get_db_session() as session:
            db = DatabaseService(session)
            await db.create_document(
                doc_id=internal_id,
                filename=filename,
                content_type=content_type,
                size_bytes=len(content),
                sha256=content_hash,
                storage_uri=storage_uri,
                ingestion_status=IngestionStatus.PENDING.value,
            )
    except Exception as exc:
        logger.warning("DB persist failed (non-fatal): %s", exc)
    
    return DocumentResponse(
        doc_id=doc_id,
        filename=filename,
        content_hash=content_hash,
        storage_uri=storage_uri,
        status=IngestionStatus.PENDING.value,
        created_at=datetime.now(timezone.utc).isoformat(),
    )


@router.get("/documents/{doc_id}", response_model=DocumentResponse)
async def get_document(doc_id: str):
    """Get document metadata by ID from PostgreSQL."""
    # Extract the UUID portion from doc_id (e.g. "doc_abc123def456" → UUID)
    try:
        async with get_db_session() as session:
            db = DatabaseService(session)
            # Try lookup by the hex prefix embedded in doc_id
            hex_part = doc_id.replace("doc_", "") if doc_id.startswith("doc_") else doc_id
            # List all documents and match by storage_uri or doc_id pattern
            from sqlalchemy import select
            from src.storage.database import DocumentRecordDB
            result = await session.execute(
                select(DocumentRecordDB).where(
                    DocumentRecordDB.storage_uri.contains(doc_id)
                )
            )
            record = result.scalar_one_or_none()

            if record is None:
                # Fallback: try by sha256 or broad match
                result = await session.execute(
                    select(DocumentRecordDB).where(
                        DocumentRecordDB.doc_id.cast(String).startswith(hex_part[:12])
                    )
                )
                record = result.scalar_one_or_none()

            if record is None:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Document not found: {doc_id}",
                )

            return DocumentResponse(
                doc_id=doc_id,
                filename=record.filename,
                content_hash=record.sha256,
                storage_uri=record.storage_uri,
                status=record.ingestion_status,
                created_at=record.created_at.isoformat(),
            )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("DB lookup failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {str(exc)}",
        )


@router.post("/index/{doc_id}", response_model=IndexResponse)
async def index_document(doc_id: str, request: Optional[IndexRequest] = None):
    """Trigger indexing for a document. Updates status in PostgreSQL."""
    retrieval = await get_retrieval_router()
    store = await get_object_store()
    
    # Resolve filename — prefer the request, then DB record, then fallback
    filename = request.filename if request else None

    files = store.list_files(prefix=f"uploads/{doc_id}")
    if not files:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document file not found for ID: {doc_id} in uploads/",
        )
    object_name = files[0]

    # Look up the original filename from PostgreSQL if not supplied
    if not filename:
        try:
            async with get_db_session() as session:
                from sqlalchemy import select
                from src.storage.database import DocumentRecordDB
                row = (await session.execute(
                    select(DocumentRecordDB).where(
                        DocumentRecordDB.storage_uri.contains(doc_id)
                    )
                )).scalar_one_or_none()
                if row and row.filename:
                    filename = row.filename
        except Exception as exc:
            logger.warning("DB filename lookup failed (non-fatal): %s", exc)

    # Ultimate fallback to object name
    if not filename:
        filename = object_name.split("/")[-1]

    # Download binary from MinIO
    storage_uri = f"s3://{store.bucket}/{object_name}"
    try:
        content = store.download_file(storage_uri)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to read document content: {str(e)}",
        )

    # Determine content type from extension
    ext = filename.split(".")[-1].lower() if "." in filename else "txt"
    content_type = {
        "pdf": "application/pdf",
        "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "csv": "text/csv",
        "md": "text/markdown",
    }.get(ext, "text/plain")

    try:
        result = await retrieval.ingest_document(
            doc_id=doc_id,
            filename=filename,
            content=content,
            content_type=content_type,
        )

        chunk_ids = result.get("haystack_chunk_ids", [])

        # Update status in PostgreSQL
        try:
            async with get_db_session() as session:
                db = DatabaseService(session)
                from sqlalchemy import select
                from src.storage.database import DocumentRecordDB

                row = (await session.execute(
                    select(DocumentRecordDB).where(
                        DocumentRecordDB.storage_uri.contains(doc_id)
                    )
                )).scalar_one_or_none()

                if row:
                    row.ingestion_status = IngestionStatus.INDEXED.value
                    row.haystack_doc_ids = chunk_ids
                    row.updated_at = datetime.now(timezone.utc)
                    await session.flush()
        except Exception as exc:
            logger.warning("DB status update failed (non-fatal): %s", exc)

        return IndexResponse(
            doc_id=doc_id,
            status=IngestionStatus.INDEXING.value,
            message=f"Document indexed successfully. Chunks: {len(chunk_ids)}",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Indexing failed: {str(e)}",
        )


# ===== Document list =====

class DocumentListItem(BaseModel):
    """Summary of a document for listing."""
    doc_id: str
    filename: str
    status: str

@router.get("/documents", response_model=list[DocumentListItem])
async def list_documents():
    """List all known documents with their ingestion status."""
    try:
        async with get_db_session() as session:
            from sqlalchemy import select
            from src.storage.database import DocumentRecordDB
            rows = (await session.execute(
                select(DocumentRecordDB).order_by(DocumentRecordDB.created_at.desc())
            )).scalars().all()
            return [
                DocumentListItem(
                    doc_id=f"doc_{row.doc_id.hex[:12]}",
                    filename=row.filename,
                    status=row.ingestion_status,
                )
                for row in rows
            ]
    except Exception as exc:
        logger.warning("list_documents DB error: %s", exc)
        return []


# ===== Mode 1 Endpoints =====

@router.post("/mode1/run", response_model=Mode1Response)
async def run_mode1(request: Mode1Request):
    """
    Run Mode 1: World Model Construction
    
    Builds a causal world model from evidence.
    """
    mode1 = await get_mode1()
    
    result = await mode1.run(
        domain=request.domain,
        initial_query=request.initial_query,
        max_variables=request.max_variables,
        max_edges=request.max_edges,
        doc_ids=request.doc_ids,
    )
    
    return Mode1Response(
        trace_id=result.trace_id,
        domain=result.domain,
        stage=result.stage.value,
        variables_discovered=result.variables_discovered,
        edges_created=result.edges_created,
        evidence_linked=result.evidence_linked,
        requires_review=result.requires_review,
        error=result.error,
    )


@router.get("/mode1/status")
async def get_mode1_status():
    """Return the current live stage of Mode 1 (useful for polling)."""
    if _mode1 is None:
        return {"stage": "idle", "detail": "Mode 1 not initialised"}
    return {"stage": _mode1.current_stage.value}


@router.post("/mode1/approve", response_model=WorldModelSummary)
async def approve_world_model(request: ApprovalRequest):
    """Approve a world model for activation."""
    mode1 = await get_mode1()
    
    try:
        model = await mode1.approve_model(request.domain, request.approved_by)
        
        return WorldModelSummary(
            domain=model.domain,
            version_id=model.version_id,
            node_count=len(model.variables),
            edge_count=len(model.edges),
            status=model.status.value,
            variables=list(model.variables.keys()),
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )


# ===== Mode 2 Endpoints =====

@router.post("/mode2/run", response_model=Mode2Response)
async def run_mode2(request: Mode2Request):
    """
    Run Mode 2: Decision Support
    
    Provides causal reasoning-backed recommendations.
    """
    mode2 = await get_mode2()
    
    result = await mode2.run(
        query=request.query,
        domain_hint=request.domain_hint,
    )
    
    recommendation_text = None
    confidence_text = None
    
    if result.recommendation:
        recommendation_text = result.recommendation.recommendation
        confidence_text = result.recommendation.confidence.value
    
    return Mode2Response(
        trace_id=result.trace_id,
        query=result.query,
        stage=result.stage.value,
        recommendation=recommendation_text,
        confidence=confidence_text,
        model_used=result.model_used,
        evidence_count=result.evidence_count,
        escalate_to_mode1=result.escalate_to_mode1,
        escalation_reason=result.escalation_reason,
        error=result.error,
    )


# ===== World Model Endpoints =====

@router.get("/world-models", response_model=list[WorldModelSummary])
async def list_world_models():
    """List all world models (from in-memory + DB)."""
    causal = get_causal_service()

    summaries = []

    # In-memory models (actively loaded)
    for domain in causal.list_domains():
        summary = causal.get_model_summary(domain)
        summaries.append(WorldModelSummary(
            domain=domain,
            node_count=summary["node_count"],
            edge_count=summary["edge_count"],
            status=ModelStatus.ACTIVE.value if summary["is_valid"] else ModelStatus.DRAFT.value,
            variables=summary["variables"],
        ))

    # Supplement with DB-persisted models not yet loaded
    try:
        async with get_db_session() as session:
            from sqlalchemy import select
            from src.storage.database import WorldModelVersionDB
            result = await session.execute(select(WorldModelVersionDB))
            for wm in result.scalars().all():
                if wm.domain not in causal.list_domains():
                    variables = list(wm.variables.keys()) if isinstance(wm.variables, dict) else []
                    edges = wm.edges if isinstance(wm.edges, list) else []
                    summaries.append(WorldModelSummary(
                        domain=wm.domain,
                        version_id=wm.version_id,
                        node_count=len(variables),
                        edge_count=len(edges),
                        status=wm.status,
                        variables=variables,
                    ))
    except Exception as exc:
        logger.warning("DB world-model list failed (showing in-memory only): %s", exc)

    return summaries


@router.get("/world-models/bridges", response_model=list[BridgeSummary])
async def list_bridges():
    """List all cross-model bridges."""
    from src.storage.database import get_db_session, DatabaseService

    try:
        async with get_db_session() as session:
            db = DatabaseService(session)
            bridges = await db.list_all_bridges()
            return [
                BridgeSummary(
                    bridge_id=str(b.bridge_id),
                    source_version_id=b.source_version_id,
                    target_version_id=b.target_version_id,
                    edge_count=len(b.bridge_edges or []),
                    concept_count=len(b.shared_concepts or []),
                    status=b.status,
                    created_at=b.created_at.isoformat() if b.created_at else None,
                )
                for b in bridges
            ]
    except Exception as exc:
        logger.error("list_bridges failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/world-models/bridges/{bridge_id}")
async def get_bridge(bridge_id: str):
    """Get details of a specific bridge."""
    from src.storage.database import get_db_session, DatabaseService
    from uuid import UUID as _UUID

    try:
        async with get_db_session() as session:
            db = DatabaseService(session)
            bridge = await db.get_model_bridge(_UUID(bridge_id))
            if not bridge:
                raise HTTPException(status_code=404, detail=f"Bridge {bridge_id} not found")
            return {
                "bridge_id": str(bridge.bridge_id),
                "source_version_id": bridge.source_version_id,
                "target_version_id": bridge.target_version_id,
                "bridge_edges": bridge.bridge_edges,
                "shared_concepts": bridge.shared_concepts,
                "status": bridge.status,
                "created_at": bridge.created_at.isoformat() if bridge.created_at else None,
                "description": bridge.description,
            }
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("get_bridge failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/world-models/{domain}", response_model=WorldModelSummary)
async def get_world_model(domain: str):
    """Get a specific world model by domain (in-memory or DB)."""
    causal = get_causal_service()

    # Check in-memory first
    if domain in causal.list_domains():
        summary = causal.get_model_summary(domain)
        return WorldModelSummary(
            domain=domain,
            node_count=summary["node_count"],
            edge_count=summary["edge_count"],
            status=ModelStatus.ACTIVE.value if summary["is_valid"] else ModelStatus.DRAFT.value,
            variables=summary["variables"],
        )

    # Fallback: try PostgreSQL
    try:
        async with get_db_session() as session:
            db = DatabaseService(session)
            wm = await db.get_active_world_model(domain)
            if wm is None:
                # Try any status
                from sqlalchemy import select
                from src.storage.database import WorldModelVersionDB
                result = await session.execute(
                    select(WorldModelVersionDB).where(
                        WorldModelVersionDB.domain == domain
                    ).order_by(WorldModelVersionDB.created_at.desc())
                )
                wm = result.scalars().first()

            if wm:
                variables = list(wm.variables.keys()) if isinstance(wm.variables, dict) else []
                edges = wm.edges if isinstance(wm.edges, list) else []
                return WorldModelSummary(
                    domain=domain,
                    version_id=wm.version_id,
                    node_count=len(variables),
                    edge_count=len(edges),
                    status=wm.status,
                    variables=variables,
                )
    except Exception as exc:
        logger.warning("DB world-model lookup failed: %s", exc)

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"World model not found: {domain}",
    )


@router.get("/world-models/{domain}/detail", response_model=WorldModelDetail)
async def get_world_model_detail(domain: str):
    """Return full variable definitions and edge details for human review."""
    causal = get_causal_service()

    # Try in-memory first
    if domain in causal.list_domains():
        engine = causal.get_engine(domain)
        return WorldModelDetail(
            domain=domain,
            node_count=engine.node_count,
            edge_count=engine.edge_count,
            status="draft",
            variables=[
                VariableDetail(
                    variable_id=v.variable_id,
                    name=v.name,
                    definition=v.definition,
                    var_type=v.type.value if v.type else None,
                    role=v.role.value if v.role else None,
                )
                for v in engine.variables
            ],
            edges=[
                EdgeDetail(
                    from_var=e.from_var,
                    to_var=e.to_var,
                    mechanism=e.metadata.mechanism if e.metadata else "",
                    strength=e.metadata.evidence_strength.value if e.metadata and e.metadata.evidence_strength else None,
                    confidence=e.metadata.confidence if e.metadata else None,
                )
                for e in engine.edges
            ],
        )

    # Fallback: try PostgreSQL
    try:
        async with get_db_session() as session:
            from sqlalchemy import select
            from src.storage.database import WorldModelVersionDB
            result = await session.execute(
                select(WorldModelVersionDB)
                .where(WorldModelVersionDB.domain == domain)
                .order_by(WorldModelVersionDB.created_at.desc())
            )
            wm = result.scalars().first()
            if wm:
                var_details = []
                for vid, vdata in (wm.variables or {}).items():
                    var_details.append(VariableDetail(
                        variable_id=vid,
                        name=vdata.get("name", vid),
                        definition=vdata.get("definition", ""),
                        var_type=vdata.get("variable_type"),
                        role=vdata.get("role"),
                    ))
                edge_details = []
                for edata in (wm.edges or []):
                    edge_details.append(EdgeDetail(
                        from_var=edata["from_var"],
                        to_var=edata["to_var"],
                        mechanism=edata.get("mechanism", ""),
                        strength=edata.get("strength"),
                        confidence=edata.get("confidence"),
                    ))
                return WorldModelDetail(
                    domain=domain,
                    version_id=wm.version_id,
                    node_count=len(var_details),
                    edge_count=len(edge_details),
                    status=wm.status or "draft",
                    variables=var_details,
                    edges=edge_details,
                )
    except Exception as exc:
        logger.warning("DB world-model detail lookup failed: %s", exc)

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"World model not found: {domain}",
    )


# ===== Unified Query Endpoint (Protocol-aware) =====

class QueryRequest(BaseModel):
    """Unified query request — the protocol routes to Mode 1 or Mode 2."""
    query: str = Field(..., description="Natural-language query or command")
    session_id: Optional[str] = None


class QueryResponse(BaseModel):
    """Unified query response."""
    trace_id: str
    routed_mode: str
    confidence: float
    route_reason: str
    result: dict[str, Any] = {}
    error: Optional[str] = None


@router.post("/query", response_model=QueryResponse)
async def unified_query(request: QueryRequest):
    """
    Unified entry point — delegates to CausewayAgent which runs an
    LLM-driven tool-calling loop instead of a fixed pipeline.

    The agent uses ModeRouter internally to classify the query, then
    registers mode-specific tools (and PageIndex navigation tools)
    before handing control to the LLM.
    """
    agent = await get_causeway_agent()

    try:
        result = await agent.run(
            query=request.query,
            session_id=request.session_id,
        )

        return QueryResponse(
            trace_id=result.trace_id,
            routed_mode=result.routed_mode,
            confidence=result.route_confidence,
            route_reason=result.route_reason,
            result={
                "response": result.response,
                "tool_calls": len(result.tool_calls),
                "total_tokens": result.total_tokens,
                "escalate_to_mode1": result.escalate_to_mode1,
                "escalation_reason": result.escalation_reason,
            },
            error=result.error,
        )

    except Exception as exc:
        logger.error("Unified query failed: %s", exc)
        return QueryResponse(
            trace_id="error",
            routed_mode="unknown",
            confidence=0.0,
            route_reason="error",
            error=str(exc),
        )



# ===== Protocol Status =====

@router.get("/protocol/status")
async def protocol_status():
    """Get current protocol state machine status."""
    sm = get_protocol_sm()
    return {
        "state": sm.state.value,
        "is_idle": sm.is_idle,
        "is_running": sm.is_running,
        "is_waiting_review": sm.is_waiting_review,
        "history": sm.get_state_history(),
    }


# ===== Admin: Purge All Document Data =====

class PurgeRequest(BaseModel):
    """Request to purge all document data."""
    confirm: bool = Field(
        ...,
        description="Must be True to execute. Safety guard against accidental calls.",
    )


class PurgeResponse(BaseModel):
    """Response from purge operation."""
    success: bool
    documents_deleted: int = 0
    vectors_deleted: int = 0
    files_deleted: int = 0
    errors: list[str] = []
    warnings: list[str] = []


@router.post("/admin/purge-documents", response_model=PurgeResponse)
async def purge_all_documents(request: PurgeRequest):
    """
    Purge **all** document data across every local store.

    Clears:
    - Qdrant / Haystack vector chunks
    - MinIO uploaded files
    - PostgreSQL document-record rows

    Does NOT touch: world models, evidence bundles, audit log, PageIndex.
    Requires ``confirm=true`` in the request body.
    """
    if not request.confirm:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Purge not confirmed. Send confirm=true to proceed.",
        )

    errors: list[str] = []
    warnings: list[str] = []
    documents_deleted = 0
    vectors_deleted = 0
    files_deleted = 0

    # ── 1. Fetch all document records from PostgreSQL ────────────
    doc_rows: list = []
    try:
        async with get_db_session() as session:
            from sqlalchemy import select
            from src.storage.database import DocumentRecordDB

            result = await session.execute(select(DocumentRecordDB))
            doc_rows = list(result.scalars().all())
    except Exception as exc:
        errors.append(f"DB fetch failed: {exc}")
        logger.error("purge: DB fetch failed: %s", exc)

    # ── 2. Drop + recreate the entire Qdrant collection ─────────
    try:
        retrieval = await get_retrieval_router()
        vectors_deleted = await retrieval.haystack.delete_all_documents()
    except Exception as exc:
        errors.append(f"Vector store wipe failed: {exc}")
        logger.error("purge: vector store wipe failed: %s", exc)

    # ── 3. Delete files from MinIO ──────────────────────────────
    try:
        store = await get_object_store()
        for row in doc_rows:
            if row.storage_uri:
                try:
                    store.delete_file(row.storage_uri)
                    files_deleted += 1
                except Exception as exc:
                    errors.append(f"MinIO delete failed for {row.storage_uri}: {exc}")
    except Exception as exc:
        errors.append(f"Object store init failed: {exc}")
        logger.error("purge: object store init failed: %s", exc)

    # ── 4. Delete document rows from PostgreSQL ─────────────────
    try:
        async with get_db_session() as session:
            from sqlalchemy import delete as sa_delete
            from src.storage.database import DocumentRecordDB

            result = await session.execute(sa_delete(DocumentRecordDB))
            documents_deleted = result.rowcount  # type: ignore[assignment]
    except Exception as exc:
        errors.append(f"DB delete failed: {exc}")
        logger.error("purge: DB row delete failed: %s", exc)

    # ── 5. Warnings ─────────────────────────────────────────────
    warnings.append("PageIndex entries were NOT cleared (out of scope).")

    logger.info(
        "purge complete: docs=%d, vectors=%d, files=%d, errors=%d",
        documents_deleted, vectors_deleted, files_deleted, len(errors),
    )

    return PurgeResponse(
        success=len(errors) == 0,
        documents_deleted=documents_deleted,
        vectors_deleted=vectors_deleted,
        files_deleted=files_deleted,
        errors=errors,
        warnings=warnings,
    )


# =====================================================================
# World Model Update (PATCH) — Phase 6
# =====================================================================

class PatchWorldModelRequest(BaseModel):
    """Request body for PATCH /world-models/{domain}."""
    add_variables: list[dict[str, Any]] = Field(default_factory=list)
    remove_variables: list[str] = Field(default_factory=list)
    add_edges: list[dict[str, Any]] = Field(default_factory=list)
    remove_edges: list[dict[str, str]] = Field(default_factory=list)
    update_edges: list[dict[str, Any]] = Field(default_factory=list)


class PatchWorldModelResponse(BaseModel):
    """Response body for PATCH /world-models/{domain}."""
    old_version_id: str
    new_version_id: str
    variables_added: int = 0
    variables_removed: int = 0
    edges_added: int = 0
    edges_removed: int = 0
    edges_updated: int = 0
    conflicts: list[str] = Field(default_factory=list)


@router.patch("/world-models/{domain}", response_model=PatchWorldModelResponse)
async def patch_world_model(domain: str, request: PatchWorldModelRequest):
    """
    Apply an incremental patch to an existing world model.

    Supports adding/removing variables, adding/removing edges,
    and updating edge metadata.  Edges that would create cycles
    are automatically skipped and reported in ``conflicts``.
    """
    from src.models.causal import (
        WorldModelPatch, VariableDefinition, CausalEdge, EdgeMetadata, EdgeUpdate,
    )
    from src.models.enums import (
        EvidenceStrength, MeasurementStatus, VariableType, VariableRole,
    )

    causal = get_causal_service()

    # Build VariableDefinition objects
    add_vars = []
    for v in request.add_variables:
        add_vars.append(VariableDefinition(
            variable_id=v["variable_id"],
            name=v.get("name", v["variable_id"]),
            definition=v.get("definition", ""),
            type=VariableType(v["type"].lower()) if v.get("type") else VariableType.CONTINUOUS,
            measurement_status=MeasurementStatus(v["measurement_status"].lower()) if v.get("measurement_status") else MeasurementStatus.LATENT,
            role=VariableRole(v["role"].lower()) if v.get("role") else VariableRole.UNKNOWN,
        ))

    # Build CausalEdge objects
    add_edges = []
    for e in request.add_edges:
        add_edges.append(CausalEdge(
            from_var=e["from_var"],
            to_var=e["to_var"],
            metadata=EdgeMetadata(
                mechanism=e.get("mechanism", ""),
                evidence_strength=EvidenceStrength(e["strength"].lower()) if e.get("strength") else EvidenceStrength.HYPOTHESIS,
                confidence=e.get("confidence", 0.5),
            ),
        ))

    # Build EdgeUpdate objects
    update_edges = []
    for eu in request.update_edges:
        update_edges.append(EdgeUpdate(
            from_var=eu["from_var"],
            to_var=eu["to_var"],
            mechanism=eu.get("mechanism"),
            evidence_strength=EvidenceStrength(eu["evidence_strength"].lower()) if eu.get("evidence_strength") else None,
            confidence=eu.get("confidence"),
        ))

    patch = WorldModelPatch(
        add_variables=add_vars,
        remove_variables=request.remove_variables,
        add_edges=add_edges,
        remove_edges=request.remove_edges,
        update_edges=update_edges,
    )

    try:
        result = await causal.update_world_model(domain, patch)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        logger.error("patch_world_model failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))

    return PatchWorldModelResponse(
        old_version_id=result.old_version_id,
        new_version_id=result.new_version_id,
        variables_added=result.variables_added,
        variables_removed=result.variables_removed,
        edges_added=result.edges_added,
        edges_removed=result.edges_removed,
        edges_updated=result.edges_updated,
        conflicts=result.conflicts,
    )


# =====================================================================
# Cross-Model Bridges — Phase 6
# =====================================================================

class BuildBridgeRequest(BaseModel):
    """Request body for POST /world-models/bridge."""
    source_domain: str
    target_domain: str
    use_llm: bool = Field(default=True, description="Use LLM for concept mapping (else heuristic)")


class BridgeEdgeResponse(BaseModel):
    """A single bridge edge in a response."""
    source_domain: str
    source_var: str
    target_domain: str
    target_var: str
    mechanism: str = ""
    strength: str = "HYPOTHESIS"
    confidence: float = 0.5


class ConceptMappingResponse(BaseModel):
    """A concept mapping in a response."""
    source_var: str
    target_var: str
    similarity_score: float
    mapping_rationale: str = ""


class BuildBridgeResponse(BaseModel):
    """Response from bridge building."""
    bridge_id: str
    source_domain: str
    target_domain: str
    bridge_edges: list[BridgeEdgeResponse]
    shared_concepts: list[ConceptMappingResponse]
    status: str


@router.post("/world-models/bridge", response_model=BuildBridgeResponse)
async def build_bridge(request: BuildBridgeRequest):
    """
    Build a cross-model bridge between two domain world models.

    Discovers shared concepts, proposes directed bridge edges,
    and validates acyclicity across the federated graph.
    """
    causal = get_causal_service()

    llm_client = None
    if request.use_llm:
        try:
            from src.agent.llm_client import LLMClient
            llm_client = LLMClient()
            await llm_client.initialize()
        except Exception as exc:
            logger.warning("LLM init failed for bridge, using heuristic: %s", exc)
            llm_client = None

    try:
        bridge = await causal.build_bridge(
            source_domain=request.source_domain,
            target_domain=request.target_domain,
            llm_client=llm_client,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        logger.error("build_bridge failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))

    return BuildBridgeResponse(
        bridge_id=bridge.bridge_id,
        source_domain=bridge.source_domain,
        target_domain=bridge.target_domain,
        bridge_edges=[
            BridgeEdgeResponse(
                source_domain=e.source_domain,
                source_var=e.source_var,
                target_domain=e.target_domain,
                target_var=e.target_var,
                mechanism=e.mechanism,
                strength=e.strength.value if hasattr(e.strength, "value") else str(e.strength),
                confidence=e.confidence,
            )
            for e in bridge.bridge_edges
        ],
        shared_concepts=[
            ConceptMappingResponse(
                source_var=c.source_var,
                target_var=c.target_var,
                similarity_score=c.similarity_score,
                mapping_rationale=c.mapping_rationale,
            )
            for c in bridge.shared_concepts
        ],
        status=bridge.status.value if hasattr(bridge.status, "value") else str(bridge.status),
    )
