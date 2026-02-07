"""
LLM Client

Wrapper for Google Gemini API with:
- Retry with exponential backoff
- Structured output parsing
- Tool calling interface
- Mock mode for testing
"""

import asyncio
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional, Type, TypeVar
from enum import Enum

from pydantic import BaseModel

from src.config import get_settings


T = TypeVar("T", bound=BaseModel)


class LLMModel(str, Enum):
    """Available LLM models."""
    GEMINI_FLASH = "gemini-3-flash-preview"
    GEMINI_PRO = "gemini-1.5-pro"


@dataclass
class LLMResponse:
    """Response from LLM."""
    content: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    finish_reason: str = "stop"
    latency_ms: int = 0


@dataclass
class ToolDefinition:
    """Definition of a tool for LLM."""
    name: str
    description: str
    parameters: dict[str, Any]


class LLMClient:
    """
    Google Gemini API client.
    
    Features:
    - Async API calls with retry
    - Structured JSON output parsing
    - Tool/function calling
    - Mock mode for testing
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: LLMModel = LLMModel.GEMINI_FLASH,
        max_retries: int = 3,
        timeout: float = 60.0,
    ):
        settings = get_settings()
        self.api_key = api_key or settings.google_ai_api_key
        self.model = model
        self.max_retries = max_retries
        self.timeout = timeout
        self._client = None
        self._mock_mode = self.api_key is None
        
        # Mock responses for testing
        self._mock_responses: list[str] = []
        self._mock_index = 0
    
    async def initialize(self) -> None:
        """Initialize the Gemini client."""
        if self._mock_mode:
            return
        
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            self._client = genai.GenerativeModel(self.model.value)
        except ImportError:
            self._mock_mode = True
    
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        """
        Generate text completion.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system instructions
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum tokens to generate
            
        Returns:
            LLMResponse with generated content
        """
        start_time = datetime.now(timezone.utc)
        
        if self._mock_mode:
            return self._generate_mock_response(prompt)
        
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"
        
        for attempt in range(self.max_retries):
            try:
                response = await asyncio.to_thread(
                    self._client.generate_content,
                    full_prompt,
                    generation_config={
                        "temperature": temperature,
                        "max_output_tokens": max_tokens,
                    },
                )
                
                latency = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
                
                return LLMResponse(
                    content=response.text,
                    model=self.model.value,
                    prompt_tokens=response.usage_metadata.prompt_token_count if hasattr(response, 'usage_metadata') else 0,
                    completion_tokens=response.usage_metadata.candidates_token_count if hasattr(response, 'usage_metadata') else 0,
                    total_tokens=response.usage_metadata.total_token_count if hasattr(response, 'usage_metadata') else 0,
                    latency_ms=int(latency),
                )
                
            except Exception as e:
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                else:
                    raise
    
    async def generate_structured(
        self,
        prompt: str,
        output_schema: Type[T],
        system_prompt: Optional[str] = None,
    ) -> T:
        """
        Generate structured output matching a Pydantic model.
        
        Args:
            prompt: User prompt
            output_schema: Pydantic model class for output
            system_prompt: Optional system instructions
            
        Returns:
            Instance of output_schema
        """
        schema_json = json.dumps(output_schema.model_json_schema(), indent=2)
        
        structured_prompt = f"""{prompt}

You MUST respond with valid JSON that matches this schema:
```json
{schema_json}
```

Respond ONLY with the JSON, no other text."""

        response = await self.generate(
            prompt=structured_prompt,
            system_prompt=system_prompt,
            temperature=0.3,  # Lower temperature for structured output
        )
        
        # Parse JSON from response
        json_data = self._extract_json(response.content)
        return output_schema.model_validate(json_data)
    
    async def generate_with_tools(
        self,
        prompt: str,
        tools: list[ToolDefinition],
        system_prompt: Optional[str] = None,
    ) -> LLMResponse:
        """
        Generate with function calling capabilities.
        
        Args:
            prompt: User prompt
            tools: List of tool definitions
            system_prompt: Optional system instructions
            
        Returns:
            LLMResponse with possible tool_calls
        """
        # Format tools for prompt
        tools_descriptions = []
        for tool in tools:
            tools_descriptions.append(
                f"- {tool.name}: {tool.description}\n"
                f"  Parameters: {json.dumps(tool.parameters)}"
            )
        
        tools_prompt = f"""{prompt}

Available tools:
{chr(10).join(tools_descriptions)}

If you need to call a tool, respond with a JSON object like:
{{"tool": "tool_name", "arguments": {{...}}}}

If you don't need to call a tool, just respond normally."""

        response = await self.generate(
            prompt=tools_prompt,
            system_prompt=system_prompt,
        )
        
        # Try to parse tool calls from response
        tool_calls = self._extract_tool_calls(response.content)
        response.tool_calls = tool_calls
        
        return response
    
    def _extract_json(self, text: str) -> dict[str, Any]:
        """Extract JSON from text that may contain markdown or other content."""
        # Try to find JSON in code blocks
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)```', text)
        if json_match:
            text = json_match.group(1)
        
        # Try to parse the text as JSON
        text = text.strip()
        
        # Find the first { and last } for object
        start = text.find('{')
        end = text.rfind('}')
        
        if start != -1 and end != -1:
            text = text[start:end + 1]
        
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            # Attempt repair for common issues
            text = self._repair_json(text)
            return json.loads(text)
    
    def _repair_json(self, text: str) -> str:
        """Attempt to repair malformed JSON."""
        # Remove trailing commas
        text = re.sub(r',\s*}', '}', text)
        text = re.sub(r',\s*]', ']', text)
        
        # Ensure property names are quoted
        text = re.sub(r'(\{|\,)\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', text)
        
        return text
    
    def _extract_tool_calls(self, text: str) -> list[dict[str, Any]]:
        """Extract tool calls from response."""
        try:
            data = self._extract_json(text)
            if "tool" in data and "arguments" in data:
                return [data]
        except (json.JSONDecodeError, ValueError):
            pass
        return []
    
    def _generate_mock_response(self, prompt: str) -> LLMResponse:
        """Generate a mock response for testing."""
        if self._mock_responses:
            content = self._mock_responses[self._mock_index % len(self._mock_responses)]
            self._mock_index += 1
        else:
            # Default mock response
            content = '{"status": "ok", "message": "Mock response"}'
        
        return LLMResponse(
            content=content,
            model="mock",
            prompt_tokens=len(prompt.split()),
            completion_tokens=len(content.split()),
            total_tokens=len(prompt.split()) + len(content.split()),
            latency_ms=10,
        )
    
    def set_mock_responses(self, responses: list[str]) -> None:
        """Set mock responses for testing."""
        self._mock_responses = responses
        self._mock_index = 0
    
    @property
    def is_mock_mode(self) -> bool:
        """Check if running in mock mode."""
        return self._mock_mode
