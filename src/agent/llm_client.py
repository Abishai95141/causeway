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
import logging
import random
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional, Type, TypeVar
from enum import Enum

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
    before_sleep_log,
    RetryError,
)

logger = logging.getLogger(__name__)

from pydantic import BaseModel

from src.config import get_settings


T = TypeVar("T", bound=BaseModel)


class LLMModel(str, Enum):
    """Available LLM models."""
    GEMINI_FLASH = "gemini-2.5-flash"
    GEMINI_PRO = "gemini-2.5-pro"


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
        max_retries: int = 5,
        timeout: float = 60.0,
        semaphore_limit: int = 8,
    ):
        settings = get_settings()
        self.api_key = api_key or settings.google_ai_api_key
        self.model = model
        self.max_retries = max_retries
        self.timeout = timeout
        self._client = None
        self._mock_mode = self.api_key is None
        self._semaphore = asyncio.Semaphore(semaphore_limit)
        
        # Mock responses for testing
        self._mock_responses: list[str] = []
        self._mock_index = 0
    
    async def initialize(self) -> None:
        """Initialize the Gemini client."""
        if self._mock_mode:
            return
        
        try:
            from google import genai
            self._genai_client = genai.Client(api_key=self.api_key)
            self._client = self._genai_client
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
        
        from google.genai import types as genai_types

        config = genai_types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )

        for attempt in range(self.max_retries):
            try:
                async with self._semaphore:
                    response = await asyncio.to_thread(
                        self._client.models.generate_content,
                        model=self.model.value,
                        contents=full_prompt,
                        config=config,
                    )
                
                latency = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
                
                usage = response.usage_metadata
                return LLMResponse(
                    content=response.text,
                    model=self.model.value,
                    prompt_tokens=usage.prompt_token_count if usage else 0,
                    completion_tokens=usage.candidates_token_count if usage else 0,
                    total_tokens=usage.total_token_count if usage else 0,
                    latency_ms=int(latency),
                )
                
            except Exception as e:
                if self._is_daily_quota_error(e):
                    logger.error("Gemini daily quota exhausted (free tier: 20 req/day). Cannot retry.")
                    raise RuntimeError(
                        "Gemini API daily quota exhausted (free tier: 20 requests/day). "
                        "Wait until tomorrow or upgrade to a paid plan at https://ai.google.dev/pricing"
                    ) from e
                if attempt < self.max_retries - 1:
                    delay = self._get_retry_delay(e, attempt)
                    logger.warning(
                        "LLM call failed (attempt %d/%d): %s — retrying in %.0fs",
                        attempt + 1, self.max_retries, str(e)[:120], delay,
                    )
                    await asyncio.sleep(delay)
                else:
                    raise

    @staticmethod
    def _is_daily_quota_error(error: Exception) -> bool:
        """Check if this is an unretryable daily quota exhaustion."""
        err_str = str(error)
        return 'FreeTier' in err_str or 'PerDay' in err_str or 'free_tier' in err_str

    @staticmethod
    def _get_retry_delay(error: Exception, attempt: int) -> float:
        """Extract retry delay from rate-limit errors, or use exponential backoff with jitter."""
        import re as _re
        err_str = str(error)
        if '429' in err_str or 'RESOURCE_EXHAUSTED' in err_str:
            # Try to parse the suggested retry delay
            match = _re.search(r'retryDelay.*?(\d+)', err_str)
            if match:
                base = float(match.group(1)) + 2  # Add buffer
            else:
                base = 30.0  # Default 30s for rate limits
            # Add jitter to prevent thundering herd
            return base * (0.5 + random.random())
        base = 2 ** attempt
        return base * (0.5 + random.random())  # Jittered exponential backoff

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

    async def generate_structured_native(
        self,
        prompt: str,
        output_schema: Type[T],
        system_prompt: Optional[str] = None,
        model_override: Optional[LLMModel] = None,
    ) -> T:
        """
        Generate structured output using Gemini's native response_schema.

        Unlike ``generate_structured`` (which injects JSON schema into
        the prompt and regex-parses the response), this method uses
        ``response_mime_type="application/json"`` and ``response_schema``
        so the API guarantees valid JSON conforming to the schema.

        Falls back to the prompt-injection method in mock mode.

        Args:
            prompt:         User prompt
            output_schema:  Pydantic model class for the response
            system_prompt:  Optional system instructions
            model_override: Use a different model (e.g. gemini-2.5-pro for judge)

        Returns:
            Validated instance of *output_schema*
        """
        if self._mock_mode:
            return await self.generate_structured(
                prompt=prompt,
                output_schema=output_schema,
                system_prompt=system_prompt,
            )

        from google.genai import types as genai_types

        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"

        # Convert Pydantic JSON Schema → Gemini-compatible schema subset
        json_schema = output_schema.model_json_schema()
        gemini_schema = self._jsonschema_to_gemini(json_schema)

        config = genai_types.GenerateContentConfig(
            temperature=0.2,
            max_output_tokens=4096,
            response_mime_type="application/json",
            response_schema=gemini_schema,
        )

        target_model = (model_override or self.model).value

        for attempt in range(self.max_retries):
            try:
                async with self._semaphore:
                    response = await asyncio.to_thread(
                        self._client.models.generate_content,
                        model=target_model,
                        contents=full_prompt,
                        config=config,
                    )

                json_data = json.loads(response.text)
                return output_schema.model_validate(json_data)

            except Exception as e:
                if self._is_daily_quota_error(e):
                    raise RuntimeError(
                        "Gemini API daily quota exhausted (free tier: 20 requests/day). "
                        "Wait until tomorrow or upgrade to a paid plan at https://ai.google.dev/pricing"
                    ) from e
                if attempt < self.max_retries - 1:
                    delay = self._get_retry_delay(e, attempt)
                    logger.warning(
                        "Structured-native call failed (attempt %d/%d): %s — retrying in %.0fs",
                        attempt + 1, self.max_retries, str(e)[:120], delay,
                    )
                    await asyncio.sleep(delay)
                else:
                    # Final fallback: try prompt-injection method
                    logger.warning(
                        "Native structured output failed after %d attempts, "
                        "falling back to prompt-injection method",
                        self.max_retries,
                    )
                    return await self.generate_structured(
                        prompt=prompt,
                        output_schema=output_schema,
                        system_prompt=system_prompt,
                    )

    async def generate_with_tools(
        self,
        prompt: str,
        tools: list[ToolDefinition],
        system_prompt: Optional[str] = None,
    ) -> LLMResponse:
        """
        Generate with native Gemini function calling.

        Converts ToolDefinition list into google.generativeai
        FunctionDeclaration objects so Gemini returns structured
        tool_calls natively (no regex parsing needed).

        Falls back to prompt-injected approach in mock mode.
        """
        start_time = datetime.now(timezone.utc)

        if self._mock_mode:
            response = self._generate_mock_response(prompt)
            response.tool_calls = self._extract_tool_calls(response.content)
            return response

        from google.genai import types as genai_types

        # Build native FunctionDeclaration objects
        fn_declarations = []
        for tool in tools:
            fn_declarations.append(
                genai_types.FunctionDeclaration(
                    name=tool.name,
                    description=tool.description,
                    parameters=self._jsonschema_to_gemini(tool.parameters),
                )
            )

        tool_config = genai_types.Tool(function_declarations=fn_declarations)

        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"

        config = genai_types.GenerateContentConfig(
            temperature=0.7,
            max_output_tokens=4096,
            tools=[tool_config],
        )

        for attempt in range(self.max_retries):
            try:
                async with self._semaphore:
                    response = await asyncio.to_thread(
                        self._client.models.generate_content,
                        model=self.model.value,
                        contents=full_prompt,
                        config=config,
                    )

                latency = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

                # Extract tool calls from the native response
                tool_calls: list[dict[str, Any]] = []
                text_content = ""

                for part in (response.candidates[0].content.parts if response.candidates else []):
                    if part.function_call and part.function_call.name:
                        fc = part.function_call
                        tool_calls.append({
                            "tool": fc.name,
                            "arguments": dict(fc.args) if fc.args else {},
                        })
                    elif part.text:
                        text_content += part.text

                usage = response.usage_metadata
                return LLMResponse(
                    content=text_content,
                    model=self.model.value,
                    prompt_tokens=usage.prompt_token_count if usage else 0,
                    completion_tokens=usage.candidates_token_count if usage else 0,
                    total_tokens=usage.total_token_count if usage else 0,
                    tool_calls=tool_calls,
                    latency_ms=int(latency),
                )

            except Exception as e:
                if self._is_daily_quota_error(e):
                    logger.error("Gemini daily quota exhausted (free tier: 20 req/day). Cannot retry.")
                    raise RuntimeError(
                        "Gemini API daily quota exhausted (free tier: 20 requests/day). "
                        "Wait until tomorrow or upgrade to a paid plan at https://ai.google.dev/pricing"
                    ) from e
                if attempt < self.max_retries - 1:
                    delay = self._get_retry_delay(e, attempt)
                    logger.warning(
                        "LLM tool call failed (attempt %d/%d): %s — retrying in %.0fs",
                        attempt + 1, self.max_retries, str(e)[:120], delay,
                    )
                    await asyncio.sleep(delay)
                else:
                    raise

    @staticmethod
    def _jsonschema_to_gemini(schema: dict[str, Any]) -> dict[str, Any]:
        """
        Convert a JSON Schema dict to the subset accepted by
        ``google.generativeai.protos.Schema``.

        Gemini expects ``type`` (as enum string like "STRING") and
        ``properties`` but does *not* accept ``additionalProperties``,
        ``$defs``, ``$ref``, ``anyOf``, ``title``, etc.

        This converter handles:
        - ``$defs`` / ``$ref`` resolution (Pydantic v2 puts enums and
          nested models in ``$defs`` and uses ``$ref`` pointers)
        - ``anyOf`` for ``Optional[T]`` (Pydantic v2 emits
          ``anyOf: [{type: T}, {type: null}]``)
        - Nested objects / arrays / enums
        """
        TYPE_MAP = {
            "string": "STRING",
            "integer": "INTEGER",
            "number": "NUMBER",
            "boolean": "BOOLEAN",
            "array": "ARRAY",
            "object": "OBJECT",
        }

        # Top-level $defs dict for resolving $ref pointers
        defs: dict[str, Any] = schema.get("$defs", {})

        def _resolve_ref(node: dict[str, Any]) -> dict[str, Any]:
            """Resolve ``$ref`` pointers like ``#/$defs/SupportType``."""
            ref = node.get("$ref", "")
            if ref.startswith("#/$defs/"):
                def_name = ref[len("#/$defs/"):]
                resolved = defs.get(def_name, {})
                # Merge any sibling keys (e.g. 'description') from the
                # referencing node into the resolved definition.
                merged = {**resolved}
                for k, v in node.items():
                    if k != "$ref":
                        merged[k] = v
                return merged
            return node

        def _convert(node: dict[str, Any]) -> dict[str, Any]:
            # Resolve $ref first
            node = _resolve_ref(node)

            # Handle anyOf (Pydantic v2 Optional[T] pattern)
            if "anyOf" in node:
                non_null = [
                    branch for branch in node["anyOf"]
                    if branch.get("type") != "null"
                ]
                if non_null:
                    # Use the first non-null branch as the actual type
                    base = _convert(non_null[0])
                else:
                    base = {"type": "STRING"}
                # Carry over description from the parent node
                if "description" in node and "description" not in base:
                    base["description"] = node["description"]
                base["nullable"] = True
                return base

            result: dict[str, Any] = {}
            json_type = node.get("type", "string")
            result["type"] = TYPE_MAP.get(json_type, "STRING")

            if "description" in node:
                result["description"] = node["description"]

            if json_type == "object" and "properties" in node:
                result["properties"] = {
                    k: _convert(v) for k, v in node["properties"].items()
                }
                if "required" in node:
                    result["required"] = node["required"]

            if json_type == "array" and "items" in node:
                result["items"] = _convert(node["items"])

            if "enum" in node:
                result["enum"] = node["enum"]

            return result

        return _convert(schema)
    
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
