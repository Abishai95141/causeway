"""
API Routes

Implements all REST endpoints:
- Document upload and management
- Indexing triggers
- Mode 1/2 execution
- World model retrieval
"""

import hashlib
from datetime import datetime, timezone
from typing import Any, Optional
from uuid import UUID, uuid4

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from pydantic import BaseModel, Field

from src.config import get_settings
from src.storage.object_store import ObjectStore
from src.retrieval.router import RetrievalRouter
from src.modes.mode1 import Mode1WorldModelConstruction, Mode1Stage
from src.modes.mode2 import Mode2DecisionSupport, Mode2Stage
from src.causal.service import CausalService
from src.models.enums import IngestionStatus, ModelStatus


router = APIRouter()

# Global service instances (initialized lazily)
_object_store: Optional[ObjectStore] = None
_retrieval_router: Optional[RetrievalRouter] = None
_causal_service: Optional[CausalService] = None
_mode1: Optional[Mode1WorldModelConstruction] = None
_mode2: Optional[Mode2DecisionSupport] = None


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


# ===== Document Endpoints =====

@router.post("/uploads", response_model=DocumentResponse)
async def upload_document(
    file: UploadFile = File(...),
    description: str = Form(default=""),
):
    """
    Upload a document for processing.
    
    Supported formats: PDF, TXT, MD, XLSX
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
    
    # Generate document ID
    internal_id = uuid4()
    doc_id = f"doc_{internal_id.hex[:12]}"
    
    # Upload to object store
    store = await get_object_store()
    # storage_uri is returned by upload_bytes, not passed to it
    storage_uri = store.upload_bytes(
        doc_id=internal_id,
        filename=filename,
        content=content,
        content_type=file.content_type or "application/octet-stream"
    )
    
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
    """Get document metadata by ID."""
    # In production, would query database
    # For now, return mock response
    return DocumentResponse(
        doc_id=doc_id,
        filename="placeholder.txt",
        content_hash="placeholder",
        storage_uri=f"documents/{doc_id}/placeholder.txt",
        status=IngestionStatus.PENDING.value,
        created_at=datetime.now(timezone.utc).isoformat(),
    )


@router.post("/index/{doc_id}", response_model=IndexResponse)
async def index_document(doc_id: str, request: Optional[IndexRequest] = None):
    """Trigger indexing for a document."""
    router = await get_retrieval_router()
    store = await get_object_store()
    
    # 1. content: Fetch document from ObjectStore
    # Using list_files to find the file if filename is not provided
    filename = request.filename if request else None
    
    if not filename:
        # Try to find file with this doc_id prefix
        files = store.list_files(prefix=f"uploads/{doc_id}")
        if not files:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document file not found for ID: {doc_id} in uploads/"
            )
        # Use simple logic: pick the first one matching
        # Note: ObjectStore stores as uploads/{doc_id}.ext or uploads/{doc_id}
        # The list_files returns 'uploads/<doc_id>.ext'
        object_name = files[0]
        filename = object_name.split("/")[-1]
    
    # Generate storage URI using internal method logic or manual construction
    # ObjectStore uses manual construction in upload_bytes, but we need URI for download
    # Since we found it via list_files, we can use the object name
    storage_uri = f"s3://{store.bucket}/{object_name}" if not filename else None
    
    # If we constructed logic above correctly, object_name is available from list_files
    # But if filename WAS provided, we need to reconstruct object_name
    if not locals().get("object_name"):
        # Reconstruct object_name logic from ObjectStore._generate_object_name
        # This is brittle, ideally ObjectStore exposes public method
        # But we can try List again to be safe if performance allows, or just use list_files always
        files = store.list_files(prefix=f"uploads/{doc_id}")
        if not files:
             raise HTTPException(status_code=404, detail="File not found")
        object_name = files[0]
        storage_uri = f"s3://{store.bucket}/{object_name}"

    try:
        content = store.download_file(storage_uri)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to read document content: {str(e)}"
        )

    # 2. Extract text & 3. Index via router
    try:
        # Determine content type from filename extension
        ext = filename.split(".")[-1].lower() if "." in filename else "txt"
        content_type = "text/plain"
        if ext == "pdf":
            content_type = "application/pdf"
        elif ext == "md":
            content_type = "text/markdown"
            
        result = await router.ingest_document(
            doc_id=doc_id,
            filename=filename,
            content=content,
            content_type=content_type
        )
        
        return IndexResponse(
            doc_id=doc_id,
            status=IngestionStatus.INDEXING.value,
            message=f"Document indexed successfully. Chunks: {len(result.get('haystack_chunk_ids', []))}",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Indexing failed: {str(e)}"
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
    """List all world models."""
    causal = get_causal_service()
    
    summaries = []
    for domain in causal.list_domains():
        summary = causal.get_model_summary(domain)
        summaries.append(WorldModelSummary(
            domain=domain,
            node_count=summary["node_count"],
            edge_count=summary["edge_count"],
            status=ModelStatus.ACTIVE.value if summary["is_valid"] else ModelStatus.DRAFT.value,
            variables=summary["variables"],
        ))
    
    return summaries


@router.get("/world-models/{domain}", response_model=WorldModelSummary)
async def get_world_model(domain: str):
    """Get a specific world model by domain."""
    causal = get_causal_service()
    
    if domain not in causal.list_domains():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"World model not found: {domain}",
        )
    
    summary = causal.get_model_summary(domain)
    return WorldModelSummary(
        domain=domain,
        node_count=summary["node_count"],
        edge_count=summary["edge_count"],
        status=ModelStatus.ACTIVE.value if summary["is_valid"] else ModelStatus.DRAFT.value,
        variables=summary["variables"],
    )
