"""
PageIndex MCP Client

HTTP client for interacting with PageIndex MCP server.
PageIndex provides structural document navigation using tree-based indexing
instead of vector embeddings.

Reference: https://github.com/VectifyAI/pageindex-mcp
"""

import httpx
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional
from uuid import uuid4

from src.config import get_settings


@dataclass
class PageIndexDocument:
    """Registered document in PageIndex."""
    doc_id: str
    filename: str
    page_count: int
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class PageIndexSection:
    """Section found via PageIndex navigation."""
    section_id: str
    section_name: str
    section_number: Optional[str]
    page_number: int
    content: str
    parent_sections: list[str] = field(default_factory=list)


@dataclass  
class PageIndexError(Exception):
    """Error from PageIndex API."""
    message: str
    status_code: Optional[int] = None


class PageIndexClient:
    """
    HTTP client for PageIndex MCP server.
    
    Provides:
    - Document registration
    - Structure navigation (list sections, read pages)
    - Full provenance tracking for evidence bundles
    
    Note: This is a fallback/mock implementation. Production would use
    actual MCP server at https://chat.pageindex.ai/mcp
    """
    
    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: float = 30.0,
    ):
        settings = get_settings()
        self.base_url = base_url or settings.pageindex_url
        self.api_key = api_key or settings.pageindex_api_key
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None
        
        # For mock mode when no API key available
        self._mock_mode = self.api_key is None
        self._mock_docs: dict[str, PageIndexDocument] = {}
        self._mock_content: dict[str, str] = {}
    
    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            headers = {}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                headers=headers,
                timeout=self.timeout,
            )
        return self._client
    
    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None
    
    async def register_document(
        self,
        doc_id: str,
        filename: str,
        content: bytes,
        content_type: str,
    ) -> PageIndexDocument:
        """
        Register a document with PageIndex for indexing.
        
        Args:
            doc_id: Internal document ID
            filename: Original filename
            content: Document content as bytes
            content_type: MIME type
            
        Returns:
            PageIndexDocument with assigned pageindex_doc_id
        """
        if self._mock_mode:
            return await self._mock_register(doc_id, filename, content)
        
        client = await self._get_client()
        
        try:
            response = await client.post(
                "/documents",
                files={"file": (filename, content, content_type)},
                data={"external_id": doc_id},
            )
            response.raise_for_status()
            data = response.json()
            
            return PageIndexDocument(
                doc_id=data.get("id", doc_id),
                filename=filename,
                page_count=data.get("page_count", 0),
            )
        except httpx.HTTPError as e:
            raise PageIndexError(str(e), getattr(e, "status_code", None))
    
    async def list_sections(
        self,
        doc_id: str,
    ) -> list[PageIndexSection]:
        """
        List all sections/headings in a document.
        
        Args:
            doc_id: PageIndex document ID
            
        Returns:
            List of sections with hierarchy
        """
        if self._mock_mode:
            return await self._mock_list_sections(doc_id)
        
        client = await self._get_client()
        
        try:
            response = await client.get(f"/documents/{doc_id}/sections")
            response.raise_for_status()
            data = response.json()
            
            return [
                PageIndexSection(
                    section_id=s["id"],
                    section_name=s["name"],
                    section_number=s.get("number"),
                    page_number=s["page"],
                    content=s.get("content", ""),
                    parent_sections=s.get("parents", []),
                )
                for s in data.get("sections", [])
            ]
        except httpx.HTTPError as e:
            raise PageIndexError(str(e), getattr(e, "status_code", None))
    
    async def read_section(
        self,
        doc_id: str,
        section_id: str,
    ) -> PageIndexSection:
        """
        Read the full content of a specific section.
        
        Args:
            doc_id: PageIndex document ID
            section_id: Section identifier
            
        Returns:
            Section with full content
        """
        if self._mock_mode:
            return await self._mock_read_section(doc_id, section_id)
        
        client = await self._get_client()
        
        try:
            response = await client.get(
                f"/documents/{doc_id}/sections/{section_id}"
            )
            response.raise_for_status()
            data = response.json()
            
            return PageIndexSection(
                section_id=data["id"],
                section_name=data["name"],
                section_number=data.get("number"),
                page_number=data["page"],
                content=data["content"],
                parent_sections=data.get("parents", []),
            )
        except httpx.HTTPError as e:
            raise PageIndexError(str(e), getattr(e, "status_code", None))
    
    async def read_page(
        self,
        doc_id: str,
        page_number: int,
    ) -> str:
        """
        Read the full content of a specific page.
        
        Args:
            doc_id: PageIndex document ID
            page_number: 1-indexed page number
            
        Returns:
            Page content as text
        """
        if self._mock_mode:
            return await self._mock_read_page(doc_id, page_number)
        
        client = await self._get_client()
        
        try:
            response = await client.get(
                f"/documents/{doc_id}/pages/{page_number}"
            )
            response.raise_for_status()
            data = response.json()
            return data.get("content", "")
        except httpx.HTTPError as e:
            raise PageIndexError(str(e), getattr(e, "status_code", None))
    
    async def search(
        self,
        doc_id: str,
        query: str,
        max_results: int = 5,
    ) -> list[PageIndexSection]:
        """
        Search within a document for relevant sections.
        
        Args:
            doc_id: PageIndex document ID
            query: Search query
            max_results: Maximum sections to return
            
        Returns:
            Matching sections ranked by relevance
        """
        if self._mock_mode:
            return await self._mock_search(doc_id, query, max_results)
        
        client = await self._get_client()
        
        try:
            response = await client.post(
                f"/documents/{doc_id}/search",
                json={"query": query, "limit": max_results},
            )
            response.raise_for_status()
            data = response.json()
            
            return [
                PageIndexSection(
                    section_id=s["id"],
                    section_name=s["name"],
                    section_number=s.get("number"),
                    page_number=s["page"],
                    content=s.get("content", ""),
                    parent_sections=s.get("parents", []),
                )
                for s in data.get("results", [])
            ]
        except httpx.HTTPError as e:
            raise PageIndexError(str(e), getattr(e, "status_code", None))
    
    # ----- Mock implementations for testing without API -----
    
    async def _mock_register(
        self,
        doc_id: str,
        filename: str,
        content: bytes,
    ) -> PageIndexDocument:
        """Mock document registration."""
        pi_doc_id = f"pi_{uuid4().hex[:8]}"
        doc = PageIndexDocument(
            doc_id=pi_doc_id,
            filename=filename,
            page_count=max(1, len(content) // 3000),  # Estimate pages
        )
        self._mock_docs[pi_doc_id] = doc
        self._mock_content[pi_doc_id] = content.decode("utf-8", errors="ignore")
        return doc
    
    async def _mock_list_sections(self, doc_id: str) -> list[PageIndexSection]:
        """Mock section listing."""
        content = self._mock_content.get(doc_id, "")
        sections = []
        
        # Simple mock: create sections from content
        lines = content.split("\n")
        section_num = 1
        for i, line in enumerate(lines):
            if line.strip().startswith("#") or (line.isupper() and len(line) > 5):
                sections.append(PageIndexSection(
                    section_id=f"sec_{section_num}",
                    section_name=line.strip().lstrip("#").strip(),
                    section_number=str(section_num),
                    page_number=max(1, i // 50 + 1),
                    content=line,
                ))
                section_num += 1
        
        return sections[:10]  # Limit to 10 sections
    
    async def _mock_read_section(
        self,
        doc_id: str,
        section_id: str,
    ) -> PageIndexSection:
        """Mock section reading."""
        return PageIndexSection(
            section_id=section_id,
            section_name="Mock Section",
            section_number="1",
            page_number=1,
            content=f"Mock content for section {section_id}",
        )
    
    async def _mock_read_page(self, doc_id: str, page_number: int) -> str:
        """Mock page reading."""
        content = self._mock_content.get(doc_id, "")
        # Return a portion of content
        start = (page_number - 1) * 3000
        end = page_number * 3000
        return content[start:end] if content else f"Mock page {page_number} content"
    
    async def _mock_search(
        self,
        doc_id: str,
        query: str,
        max_results: int,
    ) -> list[PageIndexSection]:
        """Mock search."""
        # Return mock sections that "match" the query
        return [
            PageIndexSection(
                section_id=f"result_{i}",
                section_name=f"Result for: {query}",
                section_number=str(i),
                page_number=i,
                content=f"Content matching '{query}' in section {i}",
            )
            for i in range(1, min(max_results + 1, 4))
        ]
    
    @property
    def is_mock_mode(self) -> bool:
        """Check if running in mock mode."""
        return self._mock_mode
