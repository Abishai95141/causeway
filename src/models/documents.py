"""Document registry model for uploaded files."""

from datetime import datetime, timezone
from typing import Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator

from src.models.enums import IngestionStatus


class DocumentRecord(BaseModel):
    """
    Document registry entry mapping uploaded files to retrieval systems.
    
    Every upload creates a DocumentRecord. Every retrieval hit must be
    traceable back to doc_id.
    """
    
    doc_id: UUID = Field(default_factory=uuid4, description="Internal canonical ID")
    filename: str = Field(..., min_length=1, description="Original filename")
    content_type: str = Field(..., description="MIME type (application/pdf, text/plain, etc.)")
    size_bytes: int = Field(..., ge=0, description="File size in bytes")
    sha256: str = Field(..., min_length=64, max_length=64, description="SHA256 hash for dedupe/provenance")
    storage_uri: str = Field(..., description="MinIO/S3 path where file is stored")
    ingestion_status: IngestionStatus = Field(
        default=IngestionStatus.PENDING,
        description="Current ingestion status"
    )
    pageindex_doc_id: Optional[str] = Field(
        default=None,
        description="PageIndex document ID after registration"
    )
    haystack_doc_ids: list[str] = Field(
        default_factory=list,
        description="Haystack document IDs (may be multiple chunks)"
    )
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: Optional[datetime] = Field(default=None)
    
    @field_validator("sha256")
    @classmethod
    def validate_sha256_format(cls, v: str) -> str:
        """Ensure SHA256 is lowercase hex."""
        if not all(c in "0123456789abcdef" for c in v.lower()):
            raise ValueError("SHA256 must be hexadecimal")
        return v.lower()
    
    @field_validator("content_type")
    @classmethod
    def validate_supported_types(cls, v: str) -> str:
        """Validate supported content types for prototype."""
        supported = {
            "application/pdf",
            "text/plain",
            "text/markdown",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        }
        if v not in supported:
            raise ValueError(f"Unsupported content type: {v}. Supported: {supported}")
        return v
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "doc_id": "550e8400-e29b-41d4-a716-446655440000",
                "filename": "pricing_policy_v7.pdf",
                "content_type": "application/pdf",
                "size_bytes": 245678,
                "sha256": "a" * 64,
                "storage_uri": "s3://causeway-docs/uploads/550e8400.pdf",
                "ingestion_status": "indexed",
                "pageindex_doc_id": "pi_12345",
                "haystack_doc_ids": ["hs_chunk_1", "hs_chunk_2"],
            }
        }
    }
