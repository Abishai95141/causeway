"""
Context Manager

Manages conversation context with:
- Sliding window for token budget
- Message history
- Evidence tracking
- Summarization
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional
from enum import Enum

from src.utils.text import truncate_at_sentence_boundary


class MessageRole(str, Enum):
    """Message roles in conversation."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass
class Message:
    """A message in the conversation."""
    role: MessageRole
    content: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)
    token_estimate: int = 0


@dataclass
class ContextStats:
    """Statistics about the current context."""
    total_messages: int
    total_tokens: int
    user_messages: int
    assistant_messages: int
    tool_messages: int
    oldest_message_age_seconds: int


class ContextManager:
    """
    Manages conversation context with sliding window.
    
    Features:
    - Token budget enforcement
    - Message history management
    - Evidence context tracking
    - Automatic summarization trigger
    """
    
    def __init__(
        self,
        max_tokens: int = 30000,
        reserve_tokens: int = 4000,
        summarize_threshold: float = 0.8,
    ):
        """
        Initialize context manager.
        
        Args:
            max_tokens: Maximum context window size
            reserve_tokens: Tokens to reserve for response
            summarize_threshold: When to trigger summarization (0-1)
        """
        self.max_tokens = max_tokens
        self.reserve_tokens = reserve_tokens
        self.summarize_threshold = summarize_threshold
        
        self._messages: list[Message] = []
        self._system_prompt: Optional[str] = None
        self._system_tokens: int = 0
        self._evidence_context: dict[str, str] = {}  # evidence_id -> summary
    
    @property
    def available_tokens(self) -> int:
        """Tokens available for new content."""
        used = self._system_tokens + sum(m.token_estimate for m in self._messages)
        return max(0, self.max_tokens - self.reserve_tokens - used)
    
    @property
    def used_tokens(self) -> int:
        """Tokens currently used."""
        return self._system_tokens + sum(m.token_estimate for m in self._messages)
    
    @property
    def messages(self) -> list[Message]:
        """Get all messages."""
        return self._messages.copy()
    
    def set_system_prompt(self, prompt: str) -> None:
        """Set the system prompt."""
        self._system_prompt = prompt
        self._system_tokens = self._estimate_tokens(prompt)
    
    def add_message(
        self,
        role: MessageRole,
        content: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> Message:
        """
        Add a message to the context.
        
        Args:
            role: Message role
            content: Message content
            metadata: Optional metadata
            
        Returns:
            Created Message
        """
        token_estimate = self._estimate_tokens(content)
        
        message = Message(
            role=role,
            content=content,
            metadata=metadata or {},
            token_estimate=token_estimate,
        )
        
        self._messages.append(message)
        
        # Check if we need to trim
        if self.used_tokens > self.max_tokens * self.summarize_threshold:
            self._trim_context()
        
        return message
    
    def add_user_message(self, content: str) -> Message:
        """Add a user message."""
        return self.add_message(MessageRole.USER, content)
    
    def add_assistant_message(self, content: str) -> Message:
        """Add an assistant message."""
        return self.add_message(MessageRole.ASSISTANT, content)
    
    def add_tool_result(self, tool_name: str, result: str) -> Message:
        """Add a tool result message."""
        return self.add_message(
            MessageRole.TOOL,
            result,
            metadata={"tool_name": tool_name},
        )
    
    def add_evidence(self, evidence_id: str, summary: str) -> None:
        """Add evidence to context tracking."""
        self._evidence_context[evidence_id] = summary
    
    def get_evidence_context(self) -> str:
        """Get formatted evidence context."""
        if not self._evidence_context:
            return ""
        
        lines = ["Relevant Evidence:"]
        for eid, summary in self._evidence_context.items():
            lines.append(f"- [{eid}] {summary}")
        return "\n".join(lines)
    
    def build_prompt(self, include_system: bool = True) -> str:
        """
        Build the full prompt from context.
        
        Args:
            include_system: Whether to include system prompt
            
        Returns:
            Formatted prompt string
        """
        parts = []
        
        if include_system and self._system_prompt:
            parts.append(f"System: {self._system_prompt}")
        
        # Add evidence context if any
        evidence_ctx = self.get_evidence_context()
        if evidence_ctx:
            parts.append(evidence_ctx)
        
        # Add messages
        for msg in self._messages:
            if msg.role == MessageRole.USER:
                parts.append(f"User: {msg.content}")
            elif msg.role == MessageRole.ASSISTANT:
                parts.append(f"Assistant: {msg.content}")
            elif msg.role == MessageRole.TOOL:
                tool_name = msg.metadata.get("tool_name", "tool")
                parts.append(f"Tool ({tool_name}): {msg.content}")
        
        return "\n\n".join(parts)
    
    def get_last_message(self, role: Optional[MessageRole] = None) -> Optional[Message]:
        """Get the last message, optionally filtered by role."""
        for msg in reversed(self._messages):
            if role is None or msg.role == role:
                return msg
        return None
    
    def get_stats(self) -> ContextStats:
        """Get context statistics."""
        now = datetime.now(timezone.utc)
        
        roles = {role: 0 for role in MessageRole}
        for msg in self._messages:
            roles[msg.role] += 1
        
        oldest_age = 0
        if self._messages:
            oldest_age = int((now - self._messages[0].timestamp).total_seconds())
        
        return ContextStats(
            total_messages=len(self._messages),
            total_tokens=self.used_tokens,
            user_messages=roles[MessageRole.USER],
            assistant_messages=roles[MessageRole.ASSISTANT],
            tool_messages=roles[MessageRole.TOOL],
            oldest_message_age_seconds=oldest_age,
        )
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text."""
        # Rough estimate: ~4 characters per token
        return len(text) // 4
    
    def _trim_context(self) -> None:
        """Trim older messages to fit within budget.

        Instead of blindly deleting messages (which destroys evidence
        the LLM may need later), we:
        1. Preserve the system prompt and last user+assistant exchange.
        2. Compress older TOOL messages by summarising their content
           at sentence boundaries rather than dropping them entirely.
        3. Only fully remove messages as a last resort.
        """
        target_tokens = int(self.max_tokens * 0.6)  # Trim to 60% capacity

        # Phase 1: Compress older tool messages (keep last 2 messages intact)
        safe_tail = 2  # always preserve the most recent exchange
        i = 0
        while self.used_tokens > target_tokens and i < len(self._messages) - safe_tail:
            msg = self._messages[i]
            if msg.role == MessageRole.TOOL and len(msg.content) > 300:
                # Summarise rather than destroy
                compressed = truncate_at_sentence_boundary(
                    msg.content, max_chars=300, suffix=" [trimmed]",
                )
                saved_tokens = msg.token_estimate - self._estimate_tokens(compressed)
                if saved_tokens > 0:
                    msg.content = compressed
                    msg.token_estimate = self._estimate_tokens(compressed)
            i += 1

        # Phase 2: If still over budget, remove oldest non-essential messages
        while self.used_tokens > target_tokens and len(self._messages) > safe_tail:
            removed = self._messages.pop(0)
    
    def clear(self) -> None:
        """Clear all messages (keeps system prompt)."""
        self._messages.clear()
        self._evidence_context.clear()
    
    def reset(self) -> None:
        """Reset everything including system prompt."""
        self._messages.clear()
        self._evidence_context.clear()
        self._system_prompt = None
        self._system_tokens = 0
