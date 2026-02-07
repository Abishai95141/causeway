"""
PageIndex Service

High-level service for PageIndex operations, converting results
to canonical EvidenceBundle format.
"""

from datetime import datetime, timezone
from typing import Optional
from uuid import UUID

from src.config import get_settings
from src.models.evidence import (
    EvidenceBundle,
    SourceReference,
    LocationMetadata,
    RetrievalTrace,
)
from src.models.enums import RetrievalMethod
from src.pageindex.client import PageIndexClient, PageIndexSection


class PageIndexService:
    """
    High-level PageIndex service.
    
    Provides:
    - Document registration with tracking
    - Evidence retrieval with full provenance
    - Navigation path tracking
    """
    
    def __init__(self, client: Optional[PageIndexClient] = None):
        self.client = client or PageIndexClient()
        self._navigation_paths: dict[str, list[str]] = {}  # Track nav paths per query
    
    async def close(self) -> None:
        """Close the client."""
        await self.client.close()
    
    async def register_document(
        self,
        doc_id: str,
        filename: str,
        content: bytes,
        content_type: str,
    ) -> str:
        """
        Register a document and return its PageIndex ID.
        
        Args:
            doc_id: Internal document ID
            filename: Original filename
            content: Document content
            content_type: MIME type
            
        Returns:
            PageIndex document ID for future queries
        """
        result = await self.client.register_document(
            doc_id=doc_id,
            filename=filename,
            content=content,
            content_type=content_type,
        )
        return result.doc_id
    
    async def retrieve_evidence(
        self,
        doc_id: str,
        doc_title: str,
        query: str,
        max_results: int = 5,
    ) -> list[EvidenceBundle]:
        """
        Retrieve evidence from a document using PageIndex navigation.
        
        Args:
            doc_id: PageIndex document ID
            doc_title: Human-readable document title
            query: Search query
            max_results: Maximum evidence bundles to return
            
        Returns:
            List of EvidenceBundle with full provenance
        """
        # Search for relevant sections
        sections = await self.client.search(doc_id, query, max_results)
        
        bundles = []
        for section in sections:
            bundle = self._section_to_evidence(
                section=section,
                doc_id=doc_id,
                doc_title=doc_title,
                query=query,
            )
            bundles.append(bundle)
        
        return bundles
    
    async def navigate_and_retrieve(
        self,
        doc_id: str,
        doc_title: str,
        query: str,
        navigation_path: Optional[list[str]] = None,
    ) -> list[EvidenceBundle]:
        """
        Navigate document structure and retrieve evidence.
        
        This simulates the LLM navigation pattern where sections
        are explored based on structure rather than just search.
        
        Args:
            doc_id: PageIndex document ID
            doc_title: Document title
            query: What we're looking for
            navigation_path: Optional list of section names to follow
            
        Returns:
            Evidence bundles from navigation
        """
        bundles = []
        
        # Get document structure
        sections = await self.client.list_sections(doc_id)
        
        if navigation_path:
            # Follow specific path
            for section_name in navigation_path:
                matching = [s for s in sections if section_name.lower() in s.section_name.lower()]
                for section in matching:
                    full_section = await self.client.read_section(doc_id, section.section_id)
                    bundle = self._section_to_evidence(
                        section=full_section,
                        doc_id=doc_id,
                        doc_title=doc_title,
                        query=query,
                        nav_path=navigation_path,
                    )
                    bundles.append(bundle)
        else:
            # Auto-navigate based on query keywords
            query_words = set(query.lower().split())
            for section in sections:
                section_words = set(section.section_name.lower().split())
                if query_words & section_words:  # Any word overlap
                    full_section = await self.client.read_section(doc_id, section.section_id)
                    bundle = self._section_to_evidence(
                        section=full_section,
                        doc_id=doc_id,
                        doc_title=doc_title,
                        query=query,
                    )
                    bundles.append(bundle)
        
        return bundles[:5]  # Limit results
    
    async def read_page_as_evidence(
        self,
        doc_id: str,
        doc_title: str,
        page_number: int,
        query: str,
    ) -> EvidenceBundle:
        """
        Read a specific page and return as evidence.
        
        Args:
            doc_id: PageIndex document ID
            doc_title: Document title
            page_number: Page to read
            query: Query context
            
        Returns:
            EvidenceBundle with page content
        """
        content = await self.client.read_page(doc_id, page_number)
        
        return EvidenceBundle(
            content=content,
            source=SourceReference(
                doc_id=doc_id,
                doc_title=doc_title,
            ),
            location=LocationMetadata(
                page_number=page_number,
            ),
            retrieval_trace=RetrievalTrace(
                method=RetrievalMethod.PAGEINDEX,
                query=query,
                timestamp=datetime.now(timezone.utc),
            ),
        )
    
    def _section_to_evidence(
        self,
        section: PageIndexSection,
        doc_id: str,
        doc_title: str,
        query: str,
        nav_path: Optional[list[str]] = None,
    ) -> EvidenceBundle:
        """Convert PageIndex section to canonical EvidenceBundle."""
        return EvidenceBundle(
            content=section.content,
            source=SourceReference(
                doc_id=doc_id,
                doc_title=doc_title,
            ),
            location=LocationMetadata(
                section_name=section.section_name,
                section_number=section.section_number,
                page_number=section.page_number,
            ),
            retrieval_trace=RetrievalTrace(
                method=RetrievalMethod.PAGEINDEX,
                query=query,
                timestamp=datetime.now(timezone.utc),
                retrieval_path=nav_path or section.parent_sections,
            ),
        )
    
    @property
    def is_mock_mode(self) -> bool:
        """Check if running in mock mode."""
        return self.client.is_mock_mode
