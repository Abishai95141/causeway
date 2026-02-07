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
from src.models.enums import IngestionStatus, ModelStatus, OperatingMode

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
            content=b.content[:1000],
            doc_id=b.source.doc_id,
            doc_title=b.source.doc_title,
            score=b.retrieval_trace.scores.get("vector", 0.0) if b.retrieval_trace.scores else 0.0,
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
    Unified entry point that auto-routes to Mode 1 or Mode 2
    using the ProtocolStateMachine + ModeRouter.
    """
    sm = get_protocol_sm()
    mr = get_mode_router()

    # Classify the query
    decision = mr.route(request.query)

    try:
        async with sm.run(request.query, session_id=request.session_id) as ctx:
            await sm.set_mode(decision.mode)

            if decision.mode == OperatingMode.MODE_1:
                mode1 = await get_mode1()
                domain = decision.extracted_domain or "general"
                m1_result = await mode1.run(
                    domain=domain,
                    initial_query=request.query,
                )
                await sm.complete_discovery()
                ctx.result = {
                    "trace_id": m1_result.trace_id,
                    "domain": m1_result.domain,
                    "stage": m1_result.stage.value,
                    "variables_discovered": m1_result.variables_discovered,
                    "edges_created": m1_result.edges_created,
                    "requires_review": m1_result.requires_review,
                    "error": m1_result.error,
                }
            else:
                mode2 = await get_mode2()
                m2_result = await mode2.run(
                    query=request.query,
                    domain_hint=decision.extracted_domain,
                )
                await sm.complete_decision_support()

                rec_text = None
                conf_text = None
                if m2_result.recommendation:
                    rec_text = m2_result.recommendation.recommendation
                    conf_text = m2_result.recommendation.confidence.value

                ctx.result = {
                    "trace_id": m2_result.trace_id,
                    "query": m2_result.query,
                    "stage": m2_result.stage.value,
                    "recommendation": rec_text,
                    "confidence": conf_text,
                    "evidence_count": m2_result.evidence_count,
                    "escalate_to_mode1": m2_result.escalate_to_mode1,
                    "error": m2_result.error,
                }

        return QueryResponse(
            trace_id=ctx.trace_id,
            routed_mode=decision.mode.value,
            confidence=decision.confidence,
            route_reason=decision.reason.value,
            result=ctx.result or {},
        )

    except Exception as exc:
        logger.error("Unified query failed: %s", exc)
        return QueryResponse(
            trace_id="error",
            routed_mode=decision.mode.value,
            confidence=decision.confidence,
            route_reason=decision.reason.value,
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
