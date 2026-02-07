"""
Redis cache wrapper for session and retrieval result caching.

Provides:
- Key-value caching with TTL
- JSON serialization for complex objects
- Session cache for user contexts
- Retrieval result cache for deduplication
"""

import json
from datetime import timedelta
from typing import Any, Optional, TypeVar, Generic
from uuid import UUID

import redis.asyncio as redis
from pydantic import BaseModel

from src.config import get_settings

T = TypeVar("T", bound=BaseModel)


class UUIDEncoder(json.JSONEncoder):
    """JSON encoder that handles UUID objects."""
    
    def default(self, obj):
        if isinstance(obj, UUID):
            return str(obj)
        return super().default(obj)


class RedisCache:
    """
    Async Redis cache wrapper.
    
    Features:
    - TTL-based expiration (default 1 hour)
    - JSON serialization for Pydantic models
    - Namespace prefixes for key organization
    """
    
    def __init__(
        self,
        redis_url: Optional[str] = None,
        default_ttl: Optional[int] = None,
        prefix: str = "causeway",
    ):
        settings = get_settings()
        self.redis_url = redis_url or settings.redis_url
        self.default_ttl = default_ttl or settings.redis_ttl_seconds
        self.prefix = prefix
        self._client: Optional[redis.Redis] = None
    
    async def connect(self) -> None:
        """Establish Redis connection."""
        if self._client is None:
            self._client = redis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True,
            )
    
    async def disconnect(self) -> None:
        """Close Redis connection."""
        if self._client:
            await self._client.close()
            self._client = None
    
    async def _ensure_connected(self) -> redis.Redis:
        """Ensure we have a connection and return client."""
        if self._client is None:
            await self.connect()
        return self._client  # type: ignore
    
    def _make_key(self, namespace: str, key: str) -> str:
        """Create prefixed cache key."""
        return f"{self.prefix}:{namespace}:{key}"
    
    # ----- Basic Operations -----
    
    async def get(self, namespace: str, key: str) -> Optional[str]:
        """Get a string value from cache."""
        client = await self._ensure_connected()
        full_key = self._make_key(namespace, key)
        return await client.get(full_key)
    
    async def set(
        self,
        namespace: str,
        key: str,
        value: str,
        ttl: Optional[int] = None,
    ) -> bool:
        """Set a string value in cache."""
        client = await self._ensure_connected()
        full_key = self._make_key(namespace, key)
        ttl_seconds = ttl or self.default_ttl
        return await client.setex(full_key, timedelta(seconds=ttl_seconds), value)
    
    async def delete(self, namespace: str, key: str) -> bool:
        """Delete a key from cache."""
        client = await self._ensure_connected()
        full_key = self._make_key(namespace, key)
        result = await client.delete(full_key)
        return result > 0
    
    async def exists(self, namespace: str, key: str) -> bool:
        """Check if key exists in cache."""
        client = await self._ensure_connected()
        full_key = self._make_key(namespace, key)
        return await client.exists(full_key) > 0
    
    # ----- JSON Operations -----
    
    async def get_json(self, namespace: str, key: str) -> Optional[dict]:
        """Get a JSON value from cache."""
        value = await self.get(namespace, key)
        if value:
            return json.loads(value)
        return None
    
    async def set_json(
        self,
        namespace: str,
        key: str,
        value: dict | list,
        ttl: Optional[int] = None,
    ) -> bool:
        """Set a JSON value in cache."""
        json_str = json.dumps(value, cls=UUIDEncoder)
        return await self.set(namespace, key, json_str, ttl)
    
    # ----- Pydantic Model Operations -----
    
    async def get_model(
        self,
        namespace: str,
        key: str,
        model_class: type[T],
    ) -> Optional[T]:
        """Get a Pydantic model from cache."""
        value = await self.get(namespace, key)
        if value:
            return model_class.model_validate_json(value)
        return None
    
    async def set_model(
        self,
        namespace: str,
        key: str,
        model: BaseModel,
        ttl: Optional[int] = None,
    ) -> bool:
        """Set a Pydantic model in cache."""
        json_str = model.model_dump_json()
        return await self.set(namespace, key, json_str, ttl)
    
    # ----- Specialized Caches -----
    
    async def cache_retrieval_result(
        self,
        query: str,
        method: str,
        result: list[dict],
        ttl: Optional[int] = None,
    ) -> bool:
        """Cache a retrieval result."""
        import hashlib
        query_hash = hashlib.sha256(f"{query}:{method}".encode()).hexdigest()[:16]
        return await self.set_json("retrieval", query_hash, result, ttl)
    
    async def get_cached_retrieval(
        self,
        query: str,
        method: str,
    ) -> Optional[list[dict]]:
        """Get cached retrieval result."""
        import hashlib
        query_hash = hashlib.sha256(f"{query}:{method}".encode()).hexdigest()[:16]
        return await self.get_json("retrieval", query_hash)
    
    async def set_session(
        self,
        session_id: str,
        data: dict,
        ttl: int = 3600,  # 1 hour default for sessions
    ) -> bool:
        """Store session data."""
        return await self.set_json("session", session_id, data, ttl)
    
    async def get_session(self, session_id: str) -> Optional[dict]:
        """Get session data."""
        return await self.get_json("session", session_id)
    
    async def delete_session(self, session_id: str) -> bool:
        """Delete session data."""
        return await self.delete("session", session_id)
    
    # ----- Health Check -----
    
    async def ping(self) -> bool:
        """Check if Redis is reachable."""
        try:
            client = await self._ensure_connected()
            return await client.ping()
        except Exception:
            return False
