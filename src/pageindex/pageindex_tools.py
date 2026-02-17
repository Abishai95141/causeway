"""
PageIndex Tool Provider

Converts PageIndexClient methods into (ToolDefinition, handler) pairs
that can be registered with AgentOrchestrator.register_tool().

This keeps PageIndex coupling out of the agent and orchestrator layers.
Tools exposed:
  - list_sections: list all sections/headings in a document
  - read_section:  read a specific section's content
  - search_document: semantic search within a document

Phase 1 note: these three tools give the LLM structural document
navigation *before* it decides to invoke Mode 1 or Mode 2. Phase 2
will add finer-grained Mode 2 sub-step tools.
"""

from typing import Any, Callable

from src.agent.llm_client import ToolDefinition
from src.pageindex.client import PageIndexClient
from src.utils.text import truncate_evidence


def _section_to_dict(section: Any) -> dict[str, Any]:
    """Serialize a PageIndexSection to a JSON-safe dict."""
    return {
        "section_id": section.section_id,
        "section_name": section.section_name,
        "section_number": section.section_number,
        "page_number": section.page_number,
        "content": truncate_evidence(section.content, max_chars=800),
        "parent_sections": section.parent_sections,
    }


def create_pageindex_tools(
    client: PageIndexClient,
) -> list[tuple[ToolDefinition, Callable]]:
    """
    Build tool definitions + async handlers from a PageIndexClient.

    Returns:
        List of (ToolDefinition, handler) tuples ready for
        ``orchestrator.register_tool(definition, handler)``.
    """

    # ── list_sections ───────────────────────────────────────────────
    list_sections_def = ToolDefinition(
        name="list_sections",
        description=(
            "List all sections/headings in a document registered with "
            "PageIndex.  Returns section IDs, names, and page numbers "
            "so you can decide which section to read in detail."
        ),
        parameters={
            "type": "object",
            "properties": {
                "doc_id": {
                    "type": "string",
                    "description": "PageIndex document ID",
                },
            },
            "required": ["doc_id"],
        },
    )

    async def _handle_list_sections(doc_id: str) -> dict[str, Any]:
        sections = await client.list_sections(doc_id)
        return {
            "doc_id": doc_id,
            "sections": [_section_to_dict(s) for s in sections],
            "count": len(sections),
        }

    # ── read_section ────────────────────────────────────────────────
    read_section_def = ToolDefinition(
        name="read_section",
        description=(
            "Read the full content of a specific section in a document. "
            "Use after list_sections to drill into a section of interest."
        ),
        parameters={
            "type": "object",
            "properties": {
                "doc_id": {
                    "type": "string",
                    "description": "PageIndex document ID",
                },
                "section_id": {
                    "type": "string",
                    "description": "Section identifier from list_sections",
                },
            },
            "required": ["doc_id", "section_id"],
        },
    )

    async def _handle_read_section(
        doc_id: str, section_id: str
    ) -> dict[str, Any]:
        section = await client.read_section(doc_id, section_id)
        return _section_to_dict(section)

    # ── search_document ─────────────────────────────────────────────
    search_document_def = ToolDefinition(
        name="search_document",
        description=(
            "Search within a specific document for sections matching a "
            "query.  Returns ranked results with content snippets."
        ),
        parameters={
            "type": "object",
            "properties": {
                "doc_id": {
                    "type": "string",
                    "description": "PageIndex document ID",
                },
                "query": {
                    "type": "string",
                    "description": "Search query",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum sections to return (default 5)",
                },
            },
            "required": ["doc_id", "query"],
        },
    )

    async def _handle_search_document(
        doc_id: str, query: str, max_results: int = 5
    ) -> dict[str, Any]:
        results = await client.search(doc_id, query, max_results)
        return {
            "doc_id": doc_id,
            "query": query,
            "results": [_section_to_dict(s) for s in results],
            "count": len(results),
        }

    return [
        (list_sections_def, _handle_list_sections),
        (read_section_def, _handle_read_section),
        (search_document_def, _handle_search_document),
    ]
