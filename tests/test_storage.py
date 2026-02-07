"""
Tests for Module 2: Storage Layer

Tests cover:
- Database CRUD operations (mocked connection)
- Redis cache operations (mocked Redis)
- Object store operations (mocked MinIO)
- SQLAlchemy model definitions

Note: Full integration tests require running docker-compose services.
Unit tests here mock external dependencies.
"""

import pytest
from datetime import datetime, timezone
from uuid import uuid4
from unittest.mock import AsyncMock, MagicMock, patch

# Test SQLAlchemy model definitions (doesn't require connection)
from src.storage.database import (
    Base,
    DocumentRecordDB,
    EvidenceBundleDB,
    WorldModelVersionDB,
    WMEvidenceLinkDB,
    AuditLogDB,
    DatabaseService,
)
from src.storage.redis_cache import RedisCache, UUIDEncoder
from src.storage.object_store import ObjectStore


class TestSQLAlchemyModels:
    """Test SQLAlchemy ORM model definitions."""
    
    def test_document_record_db_columns(self):
        """DocumentRecordDB should have all required columns."""
        columns = {c.name for c in DocumentRecordDB.__table__.columns}
        required = {
            "doc_id", "filename", "content_type", "size_bytes",
            "sha256", "storage_uri", "ingestion_status",
            "pageindex_doc_id", "haystack_doc_ids",
            "created_at", "updated_at"
        }
        assert required.issubset(columns)
    
    def test_evidence_bundle_db_columns(self):
        """EvidenceBundleDB should have all required columns."""
        columns = {c.name for c in EvidenceBundleDB.__table__.columns}
        required = {
            "bundle_id", "content", "content_hash",
            "source_doc_id", "source_doc_title",
            "retrieval_method", "retrieval_query", "retrieval_timestamp",
            "created_at"
        }
        assert required.issubset(columns)
    
    def test_world_model_version_db_columns(self):
        """WorldModelVersionDB should have all required columns."""
        columns = {c.name for c in WorldModelVersionDB.__table__.columns}
        required = {
            "version_id", "domain", "description",
            "variables", "edges", "dag_json",
            "created_at", "created_by", "approved_by", "status"
        }
        assert required.issubset(columns)
    
    def test_audit_log_db_columns(self):
        """AuditLogDB should have all required columns."""
        columns = {c.name for c in AuditLogDB.__table__.columns}
        required = {
            "audit_id", "timestamp", "mode", "trace_id",
            "input_query", "output_type", "success"
        }
        assert required.issubset(columns)
    
    def test_base_metadata_tables(self):
        """All tables should be registered in Base metadata."""
        table_names = set(Base.metadata.tables.keys())
        expected = {
            "documents", "evidence_bundles",
            "world_model_versions", "wm_evidence_links", "audit_log"
        }
        assert expected.issubset(table_names)


class TestUUIDEncoder:
    """Test custom JSON encoder for UUIDs."""
    
    def test_encodes_uuid(self):
        """UUIDEncoder should handle UUID objects."""
        import json
        test_uuid = uuid4()
        data = {"id": test_uuid}
        result = json.dumps(data, cls=UUIDEncoder)
        assert str(test_uuid) in result
    
    def test_handles_regular_types(self):
        """UUIDEncoder should handle regular types."""
        import json
        data = {"name": "test", "count": 42}
        result = json.dumps(data, cls=UUIDEncoder)
        assert "test" in result
        assert "42" in result


class TestRedisCacheUnit:
    """Unit tests for RedisCache with mocked Redis."""
    
    def test_make_key(self):
        """Key generation should include prefix and namespace."""
        cache = RedisCache(prefix="test")
        key = cache._make_key("session", "user123")
        assert key == "test:session:user123"
    
    @pytest.mark.asyncio
    async def test_set_and_get(self):
        """set and get should work with mocked Redis."""
        cache = RedisCache()
        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(return_value="test_value")
        mock_redis.setex = AsyncMock(return_value=True)
        cache._client = mock_redis
        
        # Test set
        result = await cache.set("ns", "key", "value")
        assert result is True
        mock_redis.setex.assert_called_once()
        
        # Test get
        value = await cache.get("ns", "key")
        assert value == "test_value"
    
    @pytest.mark.asyncio
    async def test_json_operations(self):
        """JSON operations should serialize/deserialize."""
        cache = RedisCache()
        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(return_value='{"foo": "bar"}')
        mock_redis.setex = AsyncMock(return_value=True)
        cache._client = mock_redis
        
        # Test set_json
        await cache.set_json("ns", "key", {"foo": "bar"})
        mock_redis.setex.assert_called_once()
        
        # Test get_json
        result = await cache.get_json("ns", "key")
        assert result == {"foo": "bar"}
    
    @pytest.mark.asyncio
    async def test_ping(self):
        """ping should return True when Redis is available."""
        cache = RedisCache()
        mock_redis = AsyncMock()
        mock_redis.ping = AsyncMock(return_value=True)
        cache._client = mock_redis
        
        result = await cache.ping()
        assert result is True
    
    @pytest.mark.asyncio
    async def test_ping_failure(self):
        """ping should return False on connection error."""
        cache = RedisCache()
        mock_redis = AsyncMock()
        mock_redis.ping = AsyncMock(side_effect=Exception("Connection failed"))
        cache._client = mock_redis
        
        result = await cache.ping()
        assert result is False


class TestObjectStoreUnit:
    """Unit tests for ObjectStore with mocked MinIO."""
    
    def test_generate_object_name(self):
        """Object names should include doc_id and extension."""
        store = ObjectStore()
        doc_id = uuid4()
        
        name = store._generate_object_name(doc_id, "document.pdf")
        assert str(doc_id) in name
        assert name.endswith(".pdf")
        assert name.startswith("uploads/")
    
    def test_generate_storage_uri(self):
        """Storage URIs should be s3:// format."""
        store = ObjectStore(bucket="test-bucket")
        uri = store._generate_storage_uri("uploads/file.pdf")
        assert uri == "s3://test-bucket/uploads/file.pdf"
    
    def test_health_check_success(self):
        """health_check should return True when MinIO is available."""
        store = ObjectStore()
        mock_client = MagicMock()
        mock_client.list_buckets = MagicMock(return_value=[])
        store._client = mock_client
        
        result = store.health_check()
        assert result is True
    
    def test_health_check_failure(self):
        """health_check should return False on error."""
        store = ObjectStore()
        mock_client = MagicMock()
        mock_client.list_buckets = MagicMock(side_effect=Exception("Connection failed"))
        store._client = mock_client
        
        result = store.health_check()
        assert result is False
    
    def test_upload_bytes(self):
        """upload_bytes should call put_object correctly."""
        store = ObjectStore(bucket="test-bucket")
        mock_client = MagicMock()
        mock_client.bucket_exists = MagicMock(return_value=True)
        mock_client.put_object = MagicMock()
        store._client = mock_client
        
        doc_id = uuid4()
        uri = store.upload_bytes(
            doc_id=doc_id,
            filename="test.pdf",
            content=b"test content",
            content_type="application/pdf"
        )
        
        assert uri.startswith("s3://test-bucket/")
        mock_client.put_object.assert_called_once()
    
    def test_file_exists_true(self):
        """file_exists should return True when file is found."""
        store = ObjectStore()
        mock_client = MagicMock()
        mock_client.stat_object = MagicMock(return_value=MagicMock())
        store._client = mock_client
        
        result = store.file_exists("s3://bucket/uploads/file.pdf")
        assert result is True
    
    def test_file_exists_false(self):
        """file_exists should return False on error."""
        from minio.error import S3Error
        store = ObjectStore()
        mock_client = MagicMock()
        mock_client.stat_object = MagicMock(
            side_effect=S3Error("NoSuchKey", "Not found", "", "", "", "")
        )
        store._client = mock_client
        
        result = store.file_exists("s3://bucket/uploads/file.pdf")
        assert result is False
    
    def test_invalid_uri_raises_error(self):
        """Invalid URIs should raise ValueError."""
        store = ObjectStore()
        
        with pytest.raises(ValueError, match="Invalid storage URI"):
            store.download_file("http://invalid/uri")


class TestDatabaseServiceMocked:
    """Unit tests for DatabaseService with mocked session."""
    
    @pytest.mark.asyncio
    async def test_create_document(self):
        """create_document should add to session."""
        mock_session = AsyncMock()
        mock_session.add = MagicMock()
        mock_session.flush = AsyncMock()
        
        service = DatabaseService(mock_session)
        doc = await service.create_document(
            filename="test.pdf",
            content_type="application/pdf",
            size_bytes=1024,
            sha256="a" * 64,
            storage_uri="s3://bucket/test.pdf"
        )
        
        assert doc.filename == "test.pdf"
        mock_session.add.assert_called_once()
        mock_session.flush.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_document(self):
        """get_document should query by ID."""
        mock_session = AsyncMock()
        mock_doc = DocumentRecordDB(
            filename="test.pdf",
            content_type="application/pdf",
            size_bytes=1024,
            sha256="a" * 64,
            storage_uri="s3://bucket/test.pdf"
        )
        mock_session.get = AsyncMock(return_value=mock_doc)
        
        service = DatabaseService(mock_session)
        doc = await service.get_document(uuid4())
        
        assert doc is not None
        assert doc.filename == "test.pdf"
    
    @pytest.mark.asyncio
    async def test_create_audit_entry(self):
        """create_audit_entry should add audit log."""
        mock_session = AsyncMock()
        mock_session.add = MagicMock()
        mock_session.flush = AsyncMock()
        
        service = DatabaseService(mock_session)
        entry = await service.create_audit_entry(
            mode="world_model_construction",
            trace_id="trace_123",
            input_query="Build model",
            output_type="WorldModelVersion"
        )
        
        assert entry.trace_id == "trace_123"
        mock_session.add.assert_called_once()


# Note: Integration tests requiring actual services would be in a separate file
# and run with: pytest tests/test_storage_integration.py --docker
