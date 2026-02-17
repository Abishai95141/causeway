"""
Database layer using SQLAlchemy 2.0 with async support.

Provides:
- SQLAlchemy ORM models for all entities
- Async session management
- CRUD operations via DatabaseService
"""

from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, AsyncGenerator, Optional
from uuid import UUID, uuid4

from sqlalchemy import (
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID as PGUUID
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from src.config import get_settings


# Base class for all ORM models
class Base(DeclarativeBase):
    """Base class for SQLAlchemy ORM models."""
    pass


# -----------------------------------------------------------------------------
# ORM Models
# -----------------------------------------------------------------------------

class DocumentRecordDB(Base):
    """Document registry table."""
    
    __tablename__ = "documents"
    
    doc_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    filename: Mapped[str] = mapped_column(String(512), nullable=False)
    content_type: Mapped[str] = mapped_column(String(128), nullable=False)
    size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    sha256: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    storage_uri: Mapped[str] = mapped_column(String(1024), nullable=False)
    ingestion_status: Mapped[str] = mapped_column(String(32), nullable=False, default="pending")
    pageindex_doc_id: Mapped[Optional[str]] = mapped_column(String(256), nullable=True)
    haystack_doc_ids: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), 
        nullable=False, 
        default=lambda: datetime.now(timezone.utc)
    )
    updated_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    
    __table_args__ = (
        Index("idx_documents_sha256", "sha256"),
        Index("idx_documents_status", "ingestion_status"),
    )


class EvidenceBundleDB(Base):
    """Evidence bundles table."""
    
    __tablename__ = "evidence_bundles"
    
    bundle_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    
    # Source reference (flattened)
    source_doc_id: Mapped[str] = mapped_column(String(256), nullable=False)
    source_doc_title: Mapped[str] = mapped_column(String(512), nullable=False)
    source_url: Mapped[Optional[str]] = mapped_column(String(1024), nullable=True)
    
    # Location metadata
    section_name: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    section_number: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    page_number: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    
    # Retrieval trace
    retrieval_method: Mapped[str] = mapped_column(String(32), nullable=False)
    retrieval_query: Mapped[str] = mapped_column(Text, nullable=False)
    retrieval_timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    retrieval_scores: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc)
    )
    
    __table_args__ = (
        Index("idx_evidence_content_hash", "content_hash"),
        Index("idx_evidence_source_doc", "source_doc_id"),
    )


class WorldModelVersionDB(Base):
    """World model versions table."""
    
    __tablename__ = "world_model_versions"
    
    version_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    domain: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    variables: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    edges: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    dag_json: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc)
    )
    created_by: Mapped[str] = mapped_column(String(128), nullable=False, default="system")
    approved_by: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    approved_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="draft", index=True)
    replaces_version: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    updated_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    
    # Relationship to evidence links
    evidence_links: Mapped[list["WMEvidenceLinkDB"]] = relationship(
        "WMEvidenceLinkDB",
        back_populates="world_model",
        cascade="all, delete-orphan"
    )
    
    __table_args__ = (
        Index("idx_wm_domain", "domain"),
        Index("idx_wm_status", "status"),
    )


class WMEvidenceLinkDB(Base):
    """Junction table linking world model versions to evidence bundles."""
    
    __tablename__ = "wm_evidence_links"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    version_id: Mapped[str] = mapped_column(
        String(128),
        ForeignKey("world_model_versions.version_id", ondelete="CASCADE"),
        nullable=False
    )
    bundle_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("evidence_bundles.bundle_id", ondelete="CASCADE"),
        nullable=False
    )
    edge_id: Mapped[Optional[str]] = mapped_column(String(256), nullable=True)
    
    # Relationships
    world_model: Mapped["WorldModelVersionDB"] = relationship(
        "WorldModelVersionDB",
        back_populates="evidence_links"
    )
    
    __table_args__ = (
        UniqueConstraint("version_id", "bundle_id", name="uq_wm_evidence"),
        Index("idx_wmel_version", "version_id"),
        Index("idx_wmel_bundle", "bundle_id"),
    )


class ModelBridgeDB(Base):
    """Cross-model bridge linking two world models."""

    __tablename__ = "model_bridges"

    bridge_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    source_version_id: Mapped[str] = mapped_column(
        String(128),
        ForeignKey("world_model_versions.version_id", ondelete="CASCADE"),
        nullable=False,
    )
    target_version_id: Mapped[str] = mapped_column(
        String(128),
        ForeignKey("world_model_versions.version_id", ondelete="CASCADE"),
        nullable=False,
    )
    bridge_edges: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    shared_concepts: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="draft")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    created_by: Mapped[str] = mapped_column(String(128), nullable=False, default="system")
    description: Mapped[str] = mapped_column(Text, nullable=False, default="")
    metadata_json: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)

    __table_args__ = (
        UniqueConstraint("source_version_id", "target_version_id", name="uq_bridge_pair"),
        Index("idx_bridge_source", "source_version_id"),
        Index("idx_bridge_target", "target_version_id"),
    )


class AuditLogDB(Base):
    """Audit log table (append-only)."""
    
    __tablename__ = "audit_log"
    
    audit_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        index=True
    )
    mode: Mapped[str] = mapped_column(String(64), nullable=False)
    trace_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    input_query: Mapped[str] = mapped_column(Text, nullable=False)
    input_metadata: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    retrieval_bundle_ids: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    reasoning_steps: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    output_type: Mapped[str] = mapped_column(String(64), nullable=False)
    output_data: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    output_id: Mapped[Optional[str]] = mapped_column(String(256), nullable=True)
    execution_time_ms: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    agent_version: Mapped[str] = mapped_column(String(32), nullable=False, default="v0.1.0")
    error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    success: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    
    __table_args__ = (
        Index("idx_audit_timestamp", "timestamp"),
        Index("idx_audit_trace", "trace_id"),
        Index("idx_audit_mode", "mode"),
    )


# -----------------------------------------------------------------------------
# Database Engine & Session
# -----------------------------------------------------------------------------

_engine = None
_async_session_factory = None


def get_engine():
    """Get or create the async engine."""
    global _engine
    if _engine is None:
        settings = get_settings()
        _engine = create_async_engine(
            settings.database_url,
            pool_size=settings.db_pool_size,
            max_overflow=settings.db_max_overflow,
            echo=settings.debug,
        )
    return _engine


def get_session_factory():
    """Get or create the session factory."""
    global _async_session_factory
    if _async_session_factory is None:
        _async_session_factory = async_sessionmaker(
            bind=get_engine(),
            class_=AsyncSession,
            expire_on_commit=False,
        )
    return _async_session_factory


@asynccontextmanager
async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """Get an async database session."""
    factory = get_session_factory()
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def init_db() -> None:
    """Initialize database - create all tables."""
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def drop_db() -> None:
    """Drop all tables (for testing)."""
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


# -----------------------------------------------------------------------------
# Database Service (CRUD operations)
# -----------------------------------------------------------------------------

class DatabaseService:
    """Service class for database CRUD operations."""
    
    def __init__(self, session: AsyncSession):
        self.session = session
    
    # ----- Documents -----
    
    async def create_document(self, **kwargs) -> DocumentRecordDB:
        """Create a new document record."""
        doc = DocumentRecordDB(**kwargs)
        self.session.add(doc)
        await self.session.flush()
        return doc
    
    async def get_document(self, doc_id: UUID) -> Optional[DocumentRecordDB]:
        """Get document by ID."""
        return await self.session.get(DocumentRecordDB, doc_id)
    
    async def get_document_by_hash(self, sha256: str) -> Optional[DocumentRecordDB]:
        """Get document by SHA256 hash."""
        from sqlalchemy import select
        result = await self.session.execute(
            select(DocumentRecordDB).where(DocumentRecordDB.sha256 == sha256)
        )
        return result.scalar_one_or_none()
    
    async def update_document_status(
        self, 
        doc_id: UUID, 
        status: str,
        pageindex_doc_id: Optional[str] = None,
        haystack_doc_ids: Optional[list[str]] = None,
    ) -> Optional[DocumentRecordDB]:
        """Update document ingestion status."""
        doc = await self.get_document(doc_id)
        if doc:
            doc.ingestion_status = status
            doc.updated_at = datetime.now(timezone.utc)
            if pageindex_doc_id:
                doc.pageindex_doc_id = pageindex_doc_id
            if haystack_doc_ids:
                doc.haystack_doc_ids = haystack_doc_ids
            await self.session.flush()
        return doc
    
    # ----- Evidence Bundles -----
    
    async def create_evidence_bundle(self, **kwargs) -> EvidenceBundleDB:
        """Create a new evidence bundle."""
        bundle = EvidenceBundleDB(**kwargs)
        self.session.add(bundle)
        await self.session.flush()
        return bundle
    
    async def get_evidence_bundle(self, bundle_id: UUID) -> Optional[EvidenceBundleDB]:
        """Get evidence bundle by ID."""
        return await self.session.get(EvidenceBundleDB, bundle_id)
    
    async def get_evidence_by_hash(self, content_hash: str) -> Optional[EvidenceBundleDB]:
        """Get evidence bundle by content hash (for deduplication)."""
        from sqlalchemy import select
        result = await self.session.execute(
            select(EvidenceBundleDB).where(EvidenceBundleDB.content_hash == content_hash)
        )
        return result.scalar_one_or_none()
    
    # ----- World Models -----
    
    async def create_world_model(self, **kwargs) -> WorldModelVersionDB:
        """Create a new world model version."""
        wm = WorldModelVersionDB(**kwargs)
        self.session.add(wm)
        await self.session.flush()
        return wm
    
    async def get_world_model(self, version_id: str) -> Optional[WorldModelVersionDB]:
        """Get world model by version ID."""
        return await self.session.get(WorldModelVersionDB, version_id)
    
    async def get_active_world_model(self, domain: str) -> Optional[WorldModelVersionDB]:
        """Get the active world model for a domain."""
        from sqlalchemy import select
        result = await self.session.execute(
            select(WorldModelVersionDB).where(
                WorldModelVersionDB.domain == domain,
                WorldModelVersionDB.status == "active"
            )
        )
        return result.scalar_one_or_none()
    
    async def update_world_model_status(
        self,
        version_id: str,
        status: str,
        approved_by: Optional[str] = None,
    ) -> Optional[WorldModelVersionDB]:
        """Update world model status."""
        wm = await self.get_world_model(version_id)
        if wm:
            wm.status = status
            if approved_by:
                wm.approved_by = approved_by
                wm.approved_at = datetime.now(timezone.utc)
            await self.session.flush()
        return wm
    
    async def deprecate_world_model(self, version_id: str) -> Optional[WorldModelVersionDB]:
        """Mark a world model as deprecated."""
        return await self.update_world_model_status(version_id, "deprecated")
    
    async def link_evidence_to_world_model(
        self,
        version_id: str,
        bundle_id: UUID,
        edge_id: Optional[str] = None,
    ) -> WMEvidenceLinkDB:
        """Create a link between world model and evidence."""
        link = WMEvidenceLinkDB(
            version_id=version_id,
            bundle_id=bundle_id,
            edge_id=edge_id,
        )
        self.session.add(link)
        await self.session.flush()
        return link
    
    # ----- Model Bridges -----

    async def create_model_bridge(self, **kwargs) -> ModelBridgeDB:
        """Create a new cross-model bridge."""
        bridge = ModelBridgeDB(**kwargs)
        self.session.add(bridge)
        await self.session.flush()
        return bridge

    async def get_model_bridge(self, bridge_id: UUID) -> Optional[ModelBridgeDB]:
        """Get bridge by ID."""
        return await self.session.get(ModelBridgeDB, bridge_id)

    async def get_bridges_for_model(self, version_id: str) -> list[ModelBridgeDB]:
        """Get all bridges involving a world model (as source or target)."""
        from sqlalchemy import select, or_
        result = await self.session.execute(
            select(ModelBridgeDB).where(
                or_(
                    ModelBridgeDB.source_version_id == version_id,
                    ModelBridgeDB.target_version_id == version_id,
                )
            )
        )
        return list(result.scalars().all())

    async def list_all_bridges(self) -> list[ModelBridgeDB]:
        """List all cross-model bridges."""
        from sqlalchemy import select
        result = await self.session.execute(
            select(ModelBridgeDB).order_by(ModelBridgeDB.created_at.desc())
        )
        return list(result.scalars().all())

    async def update_model_bridge(
        self, bridge_id: UUID, **kwargs
    ) -> Optional[ModelBridgeDB]:
        """Update bridge fields."""
        bridge = await self.get_model_bridge(bridge_id)
        if bridge:
            for k, v in kwargs.items():
                if hasattr(bridge, k):
                    setattr(bridge, k, v)
            await self.session.flush()
        return bridge

    # ----- Audit Log -----
    
    async def create_audit_entry(self, **kwargs) -> AuditLogDB:
        """Create a new audit log entry."""
        entry = AuditLogDB(**kwargs)
        self.session.add(entry)
        await self.session.flush()
        return entry
    
    async def get_audit_entry(self, audit_id: UUID) -> Optional[AuditLogDB]:
        """Get audit entry by ID."""
        return await self.session.get(AuditLogDB, audit_id)
    
    async def get_audit_by_trace(self, trace_id: str) -> list[AuditLogDB]:
        """Get all audit entries for a trace ID."""
        from sqlalchemy import select
        result = await self.session.execute(
            select(AuditLogDB).where(AuditLogDB.trace_id == trace_id)
        )
        return list(result.scalars().all())
