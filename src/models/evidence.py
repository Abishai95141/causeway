"""Evidence models for retrieval results and provenance tracking."""

import hashlib
from datetime import datetime, timezone
from typing import Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, computed_field, field_validator

from src.models.enums import RetrievalMethod


class SourceReference(BaseModel):
    """Reference to the source document of evidence."""
    
    doc_id: str = Field(..., description="Internal document ID (from DocumentRecord)")
    doc_title: str = Field(..., description="Human-readable document title/filename")
    url: Optional[str] = Field(default=None, description="URL if externally accessible")
    version: Optional[str] = Field(default=None, description="Document version if tracked")


class LocationMetadata(BaseModel):
    """Precise location within a document for citations."""
    
    section_name: Optional[str] = Field(default=None, description="Section heading/title")
    section_number: Optional[str] = Field(default=None, description="Section number (e.g., '3.2.1')")
    page_number: Optional[int] = Field(default=None, ge=1, description="1-indexed page number")
    paragraph_id: Optional[str] = Field(default=None, description="Paragraph identifier")
    chunk_id: Optional[str] = Field(default=None, description="Haystack chunk ID if applicable")
    start_char: Optional[int] = Field(default=None, ge=0, description="Start character offset")
    end_char: Optional[int] = Field(default=None, ge=0, description="End character offset")


class RetrievalTrace(BaseModel):
    """Provenance information for how evidence was retrieved."""
    
    method: RetrievalMethod = Field(..., description="Retrieval method used")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="When retrieved")
    query: str = Field(..., description="Query string used for retrieval")
    scores: Optional[dict[str, float]] = Field(
        default=None,
        description="Relevance scores (e.g., {'bm25': 0.8, 'vector': 0.9})"
    )
    retrieval_path: Optional[list[str]] = Field(
        default=None,
        description="PageIndex navigation path if applicable"
    )


class EvidenceBundle(BaseModel):
    """
    Canonical evidence object used throughout the system.
    
    All retrieval outputs (Haystack chunks, PageIndex sections) must be 
    converted into this single canonical format and persisted.
    """
    
    bundle_id: UUID = Field(default_factory=uuid4)
    content: str = Field(..., min_length=1, description="Extracted text content")
    source: SourceReference = Field(..., description="Source document reference")
    location: LocationMetadata = Field(
        default_factory=LocationMetadata,
        description="Location within source document"
    )
    retrieval_trace: RetrievalTrace = Field(..., description="How this evidence was found")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    used_in_models: list[str] = Field(
        default_factory=list,
        description="World model version IDs that reference this evidence"
    )
    
    @computed_field
    @property
    def content_hash(self) -> str:
        """SHA256 hash of content for deduplication."""
        return hashlib.sha256(self.content.encode("utf-8")).hexdigest()
    
    @field_validator("content")
    @classmethod
    def strip_whitespace(cls, v: str) -> str:
        """Normalize whitespace in content."""
        return " ".join(v.split())
    
    def matches_hash(self, other_hash: str) -> bool:
        """Check if content matches another hash for deduplication."""
        return self.content_hash == other_hash.lower()
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "bundle_id": "550e8400-e29b-41d4-a716-446655440000",
                "content": "Price increases typically reduce demand by approximately 1.5x...",
                "source": {
                    "doc_id": "doc_12345",
                    "doc_title": "Q3 Pricing Postmortem",
                    "url": None,
                },
                "location": {
                    "section_name": "Churn Analysis",
                    "page_number": 5,
                },
                "retrieval_trace": {
                    "method": "pageindex",
                    "query": "price elasticity effects",
                    "timestamp": "2026-02-06T12:00:00Z",
                },
            }
        }
    }
