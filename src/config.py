"""Configuration settings using pydantic-settings."""

from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
    
    # Database
    database_url: str = "postgresql+asyncpg://causeway:causeway_dev@localhost:5432/causeway"
    db_pool_size: int = 5
    db_max_overflow: int = 10
    
    # Redis
    redis_url: str = "redis://localhost:6379/0"
    redis_ttl_seconds: int = 3600  # 1 hour default
    
    # MinIO / S3
    minio_endpoint: str = "localhost:9000"
    minio_access_key: str = "causeway"
    minio_secret_key: str = "causeway_dev_key"
    minio_bucket: str = "causeway-docs"
    minio_secure: bool = False
    
    # Qdrant
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    
    # Google AI (Gemini)
    google_ai_api_key: Optional[str] = None
    
    # PageIndex (optional)
    pageindex_api_key: Optional[str] = None
    pageindex_url: str = "https://chat.pageindex.ai/mcp"
    
    # Application
    debug: bool = True
    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
