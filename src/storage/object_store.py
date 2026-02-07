"""
MinIO/S3-compatible object storage client.

Provides:
- File upload/download for documents
- Bucket management
- Storage URI generation
"""

import io
from typing import BinaryIO, Optional
from uuid import UUID

from minio import Minio
from minio.error import S3Error

from src.config import get_settings


class ObjectStore:
    """
    MinIO/S3 object storage client.
    
    Features:
    - Upload documents with auto-generated URIs
    - Download documents by doc_id
    - Bucket auto-creation
    """
    
    def __init__(
        self,
        endpoint: Optional[str] = None,
        access_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        bucket: Optional[str] = None,
        secure: Optional[bool] = None,
    ):
        settings = get_settings()
        self.endpoint = endpoint or settings.minio_endpoint
        self.access_key = access_key or settings.minio_access_key
        self.secret_key = secret_key or settings.minio_secret_key
        self.bucket = bucket or settings.minio_bucket
        self.secure = secure if secure is not None else settings.minio_secure
        
        self._client: Optional[Minio] = None
    
    def _get_client(self) -> Minio:
        """Get or create MinIO client."""
        if self._client is None:
            self._client = Minio(
                self.endpoint,
                access_key=self.access_key,
                secret_key=self.secret_key,
                secure=self.secure,
            )
        return self._client
    
    def ensure_bucket(self) -> bool:
        """Ensure the bucket exists, create if not."""
        client = self._get_client()
        try:
            if not client.bucket_exists(self.bucket):
                client.make_bucket(self.bucket)
                return True
            return True
        except S3Error as e:
            raise RuntimeError(f"Failed to create bucket: {e}") from e
    
    def _generate_object_name(self, doc_id: str, filename: str) -> str:
        """Generate object name from doc_id and filename."""
        # Use doc_id prefix for organization, preserve original extension
        ext = filename.rsplit(".", 1)[-1] if "." in filename else ""
        return f"uploads/{doc_id}.{ext}" if ext else f"uploads/{doc_id}"
    
    def _generate_storage_uri(self, object_name: str) -> str:
        """Generate storage URI for an object."""
        return f"s3://{self.bucket}/{object_name}"
    
    def upload_file(
        self,
        doc_id: str,
        filename: str,
        data: BinaryIO,
        content_type: str,
        size: int,
    ) -> str:
        """
        Upload a file to object storage.
        
        Args:
            doc_id: Document ID for organization
            filename: Original filename
            data: File data as binary stream
            content_type: MIME type
            size: File size in bytes
            
        Returns:
            Storage URI (s3://bucket/path)
        """
        client = self._get_client()
        self.ensure_bucket()
        
        object_name = self._generate_object_name(doc_id, filename)
        
        try:
            client.put_object(
                bucket_name=self.bucket,
                object_name=object_name,
                data=data,
                length=size,
                content_type=content_type,
            )
            return self._generate_storage_uri(object_name)
        except S3Error as e:
            raise RuntimeError(f"Failed to upload file: {e}") from e
    
    def upload_bytes(
        self,
        doc_id: str,
        filename: str,
        content: bytes,
        content_type: str,
    ) -> str:
        """
        Upload bytes to object storage.
        
        Args:
            doc_id: Document ID
            filename: Original filename
            content: File content as bytes
            content_type: MIME type
            
        Returns:
            Storage URI
        """
        data = io.BytesIO(content)
        return self.upload_file(
            doc_id=doc_id,
            filename=filename,
            data=data,
            content_type=content_type,
            size=len(content),
        )
    
    def download_file(self, storage_uri: str) -> bytes:
        """
        Download a file from object storage.
        
        Args:
            storage_uri: Storage URI (s3://bucket/path)
            
        Returns:
            File content as bytes
        """
        client = self._get_client()
        
        # Parse URI
        if not storage_uri.startswith("s3://"):
            raise ValueError(f"Invalid storage URI: {storage_uri}")
        
        path = storage_uri[5:]  # Remove "s3://"
        parts = path.split("/", 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid storage URI format: {storage_uri}")
        
        bucket_name, object_name = parts
        
        response = None
        try:
            response = client.get_object(bucket_name, object_name)
            return response.read()
        except S3Error as e:
            raise RuntimeError(f"Failed to download file: {e}") from e
        finally:
            if response is not None:
                response.close()
                response.release_conn()
    
    def download_to_file(self, storage_uri: str, file_path: str) -> None:
        """
        Download a file to a local path.
        
        Args:
            storage_uri: Storage URI
            file_path: Local file path to save to
        """
        content = self.download_file(storage_uri)
        with open(file_path, "wb") as f:
            f.write(content)
    
    def delete_file(self, storage_uri: str) -> bool:
        """
        Delete a file from object storage.
        
        Args:
            storage_uri: Storage URI
            
        Returns:
            True if deleted successfully
        """
        client = self._get_client()
        
        if not storage_uri.startswith("s3://"):
            raise ValueError(f"Invalid storage URI: {storage_uri}")
        
        path = storage_uri[5:]
        parts = path.split("/", 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid storage URI format: {storage_uri}")
        
        bucket_name, object_name = parts
        
        try:
            client.remove_object(bucket_name, object_name)
            return True
        except S3Error as e:
            raise RuntimeError(f"Failed to delete file: {e}") from e
    
    def file_exists(self, storage_uri: str) -> bool:
        """
        Check if a file exists in object storage.
        
        Args:
            storage_uri: Storage URI
            
        Returns:
            True if file exists
        """
        client = self._get_client()
        
        if not storage_uri.startswith("s3://"):
            return False
        
        path = storage_uri[5:]
        parts = path.split("/", 1)
        if len(parts) != 2:
            return False
        
        bucket_name, object_name = parts
        
        try:
            client.stat_object(bucket_name, object_name)
            return True
        except S3Error:
            return False
    
    def get_file_info(self, storage_uri: str) -> Optional[dict]:
        """
        Get file metadata from object storage.
        
        Args:
            storage_uri: Storage URI
            
        Returns:
            Dict with size, content_type, last_modified, or None if not found
        """
        client = self._get_client()
        
        if not storage_uri.startswith("s3://"):
            return None
        
        path = storage_uri[5:]
        parts = path.split("/", 1)
        if len(parts) != 2:
            return None
        
        bucket_name, object_name = parts
        
        try:
            stat = client.stat_object(bucket_name, object_name)
            return {
                "size": stat.size,
                "content_type": stat.content_type,
                "last_modified": stat.last_modified,
                "etag": stat.etag,
            }
        except S3Error:
            return None
    
    def health_check(self) -> bool:
        """Check if MinIO is accessible."""
        try:
            client = self._get_client()
            client.list_buckets()
            return True
        except Exception:
            return False

    def list_files(self, prefix: str = "") -> list[str]:
        """
        List files in the bucket with optional prefix.
        
        Args:
            prefix: Filter prefix (e.g. "uploads/doc_123")
            
        Returns:
            List of object names
        """
        client = self._get_client()
        self.ensure_bucket()
        
        try:
            objects = client.list_objects(self.bucket, prefix=prefix, recursive=True)
            return [obj.object_name for obj in objects]
        except S3Error:
            return []
