"""Optimized MinIO service supporting async ops, connection pooling, progress tracking, chunked uploads, and resumable sessions."""
import asyncio
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional, Tuple

from minio import Minio
from minio.error import S3Error

from app.config import settings
from app.utils.logger import get_logger
from .async_minio_client import AsyncMinioClient, UploadProgress

logger: logging.Logger = get_logger(__name__)


class MinioService:
    """Optimized MinIO service wrapper."""

    def __init__(self):
        """Initialize MinIO service dependencies."""
        logger.info(f"Initializing optimized MinIO service")

        # Initialize sync MinIO client (used when async operations are unavailable)
        self.sync_client = Minio(
            settings.minio_endpoint,
            access_key=settings.minio_access_key,
            secret_key=settings.minio_secret_key,
            secure=settings.minio_secure
        )

        self.bucket_name = settings.minio_bucket_name
        self.chunk_size = 100 * 1024 * 1024  # Default chunk size: 100 MB
        self.max_workers = 20  # Default concurrency level
        self.max_retries = 3  # Default retry attempts
        self.retry_delay = 1.0  # Base retry delay
        self.connection_pool_size = 100  # HTTP connection pool size

        # Ensure primary bucket exists
        self._ensure_bucket_exists()

        # Track upload sessions locally
        self.upload_sessions = {}

        logger.info("Optimized MinIO service initialized successfully")

    def _ensure_bucket_exists(self) -> None:
        """Create the target bucket if it does not already exist."""
        try:
            if not self.sync_client.bucket_exists(self.bucket_name):
                logger.info(f"Creating MinIO bucket: {self.bucket_name}")
                self.sync_client.make_bucket(self.bucket_name)
                logger.info(f"Bucket '{self.bucket_name}' created successfully")
            else:
                logger.info(f"Bucket '{self.bucket_name}' already exists")
        except S3Error as e:
            logger.error(f"Failed to create bucket '{self.bucket_name}': {e}")
            raise Exception(f"Failed to create bucket: {e}")

    async def upload_file_async(
        self,
        file_path: str,
        object_name: str,
        content_type: str = "application/octet-stream",
        progress_callback: Optional[Callable[[UploadProgress], None]] = None,
        bucket_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Upload a file asynchronously with progress tracking and chunked support.

        Args:
            file_path: Local file path.
            object_name: Object name in MinIO.
            content_type: MIME type of the file.
            progress_callback: Optional progress callback.
            bucket_name: Optional bucket override.

        Returns:
            dict summarizing upload result.
        """
        bucket = bucket_name or self.bucket_name

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        file_size = os.path.getsize(file_path)
        logger.info(f"Async upload started: {object_name} ({file_size / (1024 ** 3):.2f} GB)")

        async with AsyncMinioClient(
            endpoint=settings.minio_endpoint,
            access_key=settings.minio_access_key,
            secret_key=settings.minio_secret_key,
            secure=settings.minio_secure,
            chunk_size=self.chunk_size,
            max_workers=self.max_workers,
            max_retries=self.max_retries,
            retry_delay=self.retry_delay,
            connection_pool_size=self.connection_pool_size
        ) as client:
            # Ensure target bucket exists
            await client.ensure_bucket_exists(bucket)

            # Perform upload
            result = await client.upload_file_with_progress(
                file_path=file_path,
                bucket_name=bucket,
                object_name=object_name,
                content_type=content_type,
                progress_callback=progress_callback
            )

            if result["success"]:
                logger.info(
                    f"Async upload completed: {object_name} - "
                    f"{result['uploaded_bytes'] / (1024 ** 3):.2f} GB in "
                    f"{result['upload_time']:.2f}s "
                    f"({result['speed_mbps']:.2f} MB/s)"
                )
            else:
                logger.error(f"Async upload failed: {object_name} - {result['error']}")

            return result

    async def upload_files_parallel_async(
        self,
        files: List[Tuple[str, str, str]],  # (file_path, object_name, content_type)
        bucket_name: Optional[str] = None,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None
    ) -> Dict[str, Any]:
        """
        Upload multiple files concurrently using the async client.

        Args:
            files: List of (file_path, object_name, content_type) tuples.
            bucket_name: Optional bucket override.
            progress_callback: Callback invoked with progress info.

        Returns:
            dict summarizing batch upload results.
        """
        bucket = bucket_name or self.bucket_name

        if not files:
            return {
                "total": 0,
                "successful": 0,
                "failed": 0,
                "successful_files": [],
                "failed_files": []
            }

        # Verify each file exists before uploading
        valid_files = []
        for file_path, object_name, content_type in files:
            if os.path.exists(file_path):
                valid_files.append((file_path, object_name, content_type))
            else:
                logger.warning(f"File not found: {file_path}")

        if not valid_files:
            raise ValueError("No valid files found")

        logger.info(f"Starting async parallel upload of {len(valid_files)} image files")

        async with AsyncMinioClient(
            endpoint=settings.minio_endpoint,
            access_key=settings.minio_access_key,
            secret_key=settings.minio_secret_key,
            secure=settings.minio_secure,
            chunk_size=self.chunk_size,
            max_workers=self.max_workers,
            max_retries=self.max_retries,
            retry_delay=self.retry_delay,
            connection_pool_size=self.connection_pool_size
        ) as client:
            # Ensure target bucket exists
            await client.ensure_bucket_exists(bucket)

            # Upload files concurrently
            result = await client.upload_files_parallel(
                files=valid_files,
                bucket_name=bucket,
                progress_callback=progress_callback
            )

            logger.info(
                f"Async parallel upload completed: "
                f"{result['successful']}/{result['total']} successful"
            )

            return result

    def upload_file_with_retry(
        self,
        file_path: str,
        object_name: str,
        content_type: str = "application/octet-stream",
        max_retries: int = None,
        bucket_name: Optional[str] = None,
        enable_async: bool = True
    ) -> Dict[str, Any]:
        """
        Upload a file with retry logic (sync implementation for compatibility).

        Args:
            file_path: Local path to the file.
            object_name: Target MinIO object name.
            content_type: MIME type for the object.
            max_retries: Override for retry attempts.
            bucket_name: Optional bucket override.
            enable_async: Use async path automatically for large files.

        Returns:
            dict summarizing upload result.
        """
        bucket = bucket_name or self.bucket_name
        max_retries = max_retries or self.max_retries

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        file_size = os.path.getsize(file_path)

        # Automatically move large files to async pipeline
        if enable_async and file_size > 50 * 1024 * 1024:  # 50MB+
            # Create event loop if current thread lacks one
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            def progress_callback(progress: UploadProgress):
                logger.info(
                    f"Upload progress: {progress.object_name} - "
                    f"{progress.percentage:.1f}% ({progress.speed_mbps:.2f} MB/s)"
                )

            result = loop.run_until_complete(
                self.upload_file_async(
                    file_path=file_path,
                    object_name=object_name,
                    content_type=content_type,
                    progress_callback=progress_callback,
                    bucket_name=bucket
                )
            )
            return result

        # Upload smaller files synchronously
        return self._sync_upload_with_retry(
            file_path, object_name, content_type, max_retries, bucket
        )

    def _sync_upload_with_retry(
        self,
        file_path: str,
        object_name: str,
        content_type: str,
        max_retries: int,
        bucket_name: str
    ) -> Dict[str, Any]:
        """Synchronous upload with retry support."""
        file_size = os.path.getsize(file_path)

        for attempt in range(max_retries + 1):
            try:
                # Perform upload
                self.sync_client.fput_object(
                    bucket_name,
                    object_name,
                    file_path,
                    content_type=content_type
                )

                url = f"http://{settings.minio_endpoint}/{bucket_name}/{object_name}"

                logger.info(f"File uploaded successfully: {object_name}")
                return {
                    "success": True,
                    "object_name": object_name,
                    "bucket_name": bucket_name,
                    "file_path": file_path,
                    "url": url,
                    "file_size": file_size,
                    "upload_time": None,  # Sync version does not measure duration
                    "speed_mbps": None,
                    "attempt": attempt + 1
                }

            except Exception as e:
                logger.warning(
                    f"Upload attempt {attempt + 1} failed for {object_name}: {e}"
                )

                if attempt < max_retries:
                    time.sleep(self.retry_delay * (2 ** attempt))  # exponential backoff
                else:
                    logger.error(f"Upload failed after {max_retries + 1} attempts")
                    return {
                        "success": False,
                        "object_name": object_name,
                        "bucket_name": bucket_name,
                        "file_path": file_path,
                        "error": str(e),
                        "file_size": file_size,
                        "attempt": attempt + 1
                    }

    def upload_files_with_progress(
        self,
        files: List[Tuple[str, str]],
        max_workers: int = None,
        max_retries: int = None,
        content_type: str = "application/octet-stream",
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        bucket_name: Optional[str] = None,
        enable_async: bool = True
    ) -> Dict[str, Any]:
        """
        Batch upload files with progress tracking.

        Args:
            files: List of (file_path, object_name) tuples.
            max_workers: Concurrency level for sync uploads.
            max_retries: Retry attempts for sync path.
            content_type: Default content type applied to sync uploads.
            progress_callback: Optional callback for progress metrics.
            bucket_name: Optional bucket override.
            enable_async: Whether to leverage async path for large files.

        Returns:
            dict summarizing upload outcomes.
        """
        bucket = bucket_name or self.bucket_name
        max_workers = max_workers or self.max_workers
        max_retries = max_retries or self.max_retries

        if not files:
            return {
                "total": 0,
                "successful": 0,
                "failed": 0,
                "results": [],
                "success_list": [],
                "failed_list": [],
                "retry_info": {"total_retries": 0, "retry_attempts": []}
            }

        # Dispatch small files to sync path, large files to async path
        small_files = []
        large_files = []

        for file_path, object_name in files:
            if os.path.exists(file_path):
                file_size = os.path.getsize(file_path)
                if file_size > 50 * 1024 * 1024:  # Larger than 50 MB
                    large_files.append((file_path, object_name, content_type))
                else:
                    small_files.append((file_path, object_name, content_type))
            else:
                logger.warning(f"File not found: {file_path}")

        results = []
        success_list = []
        failed_list = []

        # Handle small files via synchronous concurrent uploads
        if small_files:
            logger.info(f"Processing {len(small_files)} small files with sync upload")

            def upload_single_small_file(args):
                file_path, object_name, content_type = args
                return self._sync_upload_with_retry(
                    file_path, object_name, content_type, max_retries, bucket
                )

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(upload_single_small_file, file_info): file_info
                    for file_info in small_files
                }

                for future in as_completed(futures):
                    result = future.result()
                    results.append(result)

                    if result["success"]:
                        success_list.append(result["object_name"])
                    else:
                        failed_list.append(result)

                    if progress_callback:
                        progress_callback(result)

        # Handle large files via async path
        if large_files:
            logger.info(f"Processing {len(large_files)} large files with async upload")

            # Fire async uploads for large files
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

                def large_file_progress_callback(progress_info):
                    if progress_callback:
                        progress_callback(progress_info)

                async_result = loop.run_until_complete(
                    self.upload_files_parallel_async(
                        files=large_files,
                        bucket_name=bucket,
                        progress_callback=large_file_progress_callback
                    )
                )

                # Merge async results into summary
                results.extend(async_result["successful_files"])
                results.extend(async_result["failed_files"])
                success_list.extend([r["object_name"] for r in async_result["successful_files"]])
                failed_list.extend(async_result["failed_files"])

                if progress_callback:
                    progress_callback(async_result)

            finally:
                loop.close()

        summary = {
            "total": len(files),
            "successful": len(success_list),
            "failed": len(failed_list),
            "results": results,
            "success_list": success_list,
            "failed_list": failed_list,
            "small_files_count": len(small_files),
            "large_files_count": len(large_files),
            "retry_info": {"total_retries": 0, "retry_attempts": []}  # Retries handled internally
        }

        logger.info(
            f"Batch upload completed: {summary['successful']}/{summary['total']} successful"
        )

        return summary

    async def start_multipart_upload(
        self,
        file_path: str,
        object_name: str,
        bucket_name: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> str:
        """
        Start a multipart upload session (for resumable uploads).

        Args:
            file_path: Local path to the file.
            object_name: Target object name.
            bucket_name: Optional bucket override.
            session_id: Optional predefined session id.

        Returns:
            session id for tracking/resume.
        """
        bucket = bucket_name or self.bucket_name

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        if not session_id:
            session_id = f"{bucket}/{object_name}/{int(time.time() * 1000)}"

        file_size = os.path.getsize(file_path)

        logger.info(f"Starting multipart upload session: {session_id}")

        async with AsyncMinioClient(
            endpoint=settings.minio_endpoint,
            access_key=settings.minio_access_key,
            secret_key=settings.minio_secret_key,
            secure=settings.minio_secure,
            chunk_size=self.chunk_size,
            max_workers=self.max_workers,
            max_retries=self.max_retries,
            retry_delay=self.retry_delay,
            connection_pool_size=self.connection_pool_size
        ) as client:
            await client.ensure_bucket_exists(bucket)

            # Initialize upload (placeholder, currently delegates to async uploader)
            result = await client.upload_file_with_progress(
                file_path=file_path,
                bucket_name=bucket,
                object_name=object_name,
                progress_callback=None  # hide progress, initialization only
            )

            if result["success"]:
                # Persist session metadata in memory
                self.upload_sessions[session_id] = {
                    "bucket_name": bucket,
                    "object_name": object_name,
                    "file_path": file_path,
                    "status": "initializing",
                    "created_at": time.time(),
                    "result": result
                }

                # Persist session metadata to disk for recovery
                session_file = f"/tmp/upload_session_{session_id.replace('/', '_')}.json"
                client.save_upload_session(session_id, session_file)

                logger.info(f"Multipart upload session started: {session_id}")
                return session_id
            else:
                raise Exception(f"Failed to start multipart upload: {result['error']}")

    async def resume_upload_session(
        self,
        session_id: str,
        progress_callback: Optional[Callable[[UploadProgress], None]] = None
    ) -> Dict[str, Any]:
        """
        Resume a previously started upload session.

        Args:
            session_id: Upload session identifier.
            progress_callback: Optional progress callback.

        Returns:
            dict summarizing resumed upload result.
        """
        if session_id not in self.upload_sessions:
            # Attempt to load session metadata from disk
            session_file = f"/tmp/upload_session_{session_id.replace('/', '_')}.json"

            async with AsyncMinioClient(
                endpoint=settings.minio_endpoint,
                access_key=settings.minio_access_key,
                secret_key=settings.minio_secret_key,
                secure=settings.minio_secure,
                chunk_size=self.chunk_size,
                max_workers=self.max_workers,
                max_retries=self.max_retries,
                retry_delay=self.retry_delay,
                connection_pool_size=self.connection_pool_size
            ) as client:
                loaded = client.load_upload_session(session_id, session_file)
                if not loaded:
                    raise ValueError(f"Upload session {session_id} not found")

        session = self.upload_sessions[session_id]

        logger.info(f"Resuming upload session: {session_id}")

        async with AsyncMinioClient(
            endpoint=settings.minio_endpoint,
            access_key=settings.minio_access_key,
            secret_key=settings.minio_secret_key,
            secure=settings.minio_secure,
            chunk_size=self.chunk_size,
            max_workers=self.max_workers,
            max_retries=self.max_retries,
            retry_delay=self.retry_delay,
            connection_pool_size=self.connection_pool_size
        ) as client:
            result = await client.resume_upload(session_id)

            if result["success"]:
                # Update in-memory session state
                session["status"] = "completed"
                session["completed_at"] = time.time()
                session["result"] = result

                # Remove persisted session file
                session_file = f"/tmp/upload_session_{session_id.replace('/', '_')}.json"
                try:
                    os.remove(session_file)
                except:
                    pass

                logger.info(f"Upload session completed: {session_id}")
            else:
                session["status"] = "failed"
                session["error"] = result.get("error")
                logger.error(f"Upload session failed: {session_id}")

            return result

    def get_upload_session_status(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Return metadata for a stored upload session."""
        session = self.upload_sessions.get(session_id)
        if session:
            return {
                "session_id": session_id,
                "status": session.get("status", "unknown"),
                "created_at": session.get("created_at"),
                "completed_at": session.get("completed_at"),
                "bucket_name": session.get("bucket_name"),
                "object_name": session.get("object_name"),
                "file_path": session.get("file_path")
            }
        return None

    def cleanup_completed_sessions(self, max_age_hours: int = 24) -> int:
        """Remove completed/failed sessions older than the given age."""
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600

        sessions_to_remove = []

        for session_id, session in self.upload_sessions.items():
            if session.get("status") in ["completed", "failed"]:
                if current_time - session.get("created_at", current_time) > max_age_seconds:
                    sessions_to_remove.append(session_id)

        for session_id in sessions_to_remove:
            del self.upload_sessions[session_id]
            # Remove persisted session file
            session_file = f"/tmp/upload_session_{session_id.replace('/', '_')}.json"
            try:
                os.remove(session_file)
            except:
                pass

        logger.info(f"Cleaned up {len(sessions_to_remove)} old sessions")
        return len(sessions_to_remove)

    # Backwards-compatible synchronous helpers
    def upload_file(self, file_path: str, object_name: str, content_type: str = "application/octet-stream") -> str:
        """Synchronous wrapper maintained for backwards compatibility."""
        result = self.upload_file_with_retry(file_path, object_name, content_type)

        if not result["success"]:
            raise Exception(f"Upload failed: {result.get('error')}")

        return f"http://{settings.minio_endpoint}/{self.bucket_name}/{object_name}"

    def get_file_url(self, object_name: str) -> str:
        """Generate a presigned URL (sync version)."""
        try:
            url = self.sync_client.presigned_get_object(self.bucket_name, object_name)
            return url
        except Exception as e:
            logger.error(f"Failed to generate URL for '{object_name}': {e}")
            raise Exception(f"Failed to generate URL: {e}")

    def delete_file(self, object_name: str) -> bool:
        """Delete an object (sync helper)."""
        try:
            self.sync_client.remove_object(self.bucket_name, object_name)
            logger.info(f"File deleted successfully: {object_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete file '{object_name}': {e}")
            return False

    def file_exists(self, object_name: str) -> bool:
        """Check whether an object exists (sync helper)."""
        try:
            self.sync_client.stat_object(self.bucket_name, object_name)
            return True
        except S3Error:
            return False

    def upload_files(
        self,
        file_list: List[Tuple[str, str]],
        max_workers: int = 10,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        content_type: str = "application/octet-stream",
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None
    ) -> Dict[str, Any]:
        """
        Backwards-compatible batch upload helper that auto-selects best strategy.

        Args:
            file_list: List of (file_path, object_name) tuples.
            max_workers: Worker count for sync uploads.
            max_retries: Retry attempts for sync path.
            retry_delay: Delay used between retries.
            content_type: Default MIME type for sync uploads.
            progress_callback: Optional callback for tracking progress.

        Returns:
            dict summarizing upload result.
        """
        # Delegate to optimized implementation
        return self.upload_files_with_progress(
            files=file_list,
            max_workers=max_workers,
            max_retries=max_retries,
            content_type=content_type,
            progress_callback=progress_callback,
            enable_async=True  # use async optimization when helpful
        )

    def get_files_urls(self, object_names: List[str], max_workers: int = 20) -> Dict[str, Any]:
        """
        Retrieve presigned URLs in bulk (sync helper).

        Args:
            object_names: Names of objects to fetch.
            max_workers: Worker count for the thread pool.

        Returns:
            dict of URL results.
        """
        if not object_names:
            return {
                "total": 0,
                "successful": 0,
                "failed": 0,
                "results": [],
                "urls": {},
                "failed_list": []
            }

        results = []
        urls = {}
        failed_list = []

        def get_single_url(object_name):
            try:
                url = self.sync_client.presigned_get_object(self.bucket_name, object_name)
                return {
                    "success": True,
                    "object_name": object_name,
                    "url": url,
                    "error": None
                }
            except Exception as e:
                return {
                    "success": False,
                    "object_name": object_name,
                    "url": None,
                    "error": str(e)
                }

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(get_single_url, object_name): object_name
                for object_name in object_names
            }

            for future in as_completed(futures):
                result = future.result()
                results.append(result)

                if result["success"]:
                    urls[result["object_name"]] = result["url"]
                else:
                    failed_list.append({
                        "object_name": result["object_name"],
                        "error": result["error"]
                    })

        summary = {
            "total": len(object_names),
            "successful": len(urls),
            "failed": len(failed_list),
            "results": results,
            "urls": urls,
            "failed_list": failed_list
        }

        return summary

    # Performance helpers
    def get_performance_stats(self) -> Dict[str, Any]:
        """Return current performance-related settings."""
        return {
            "chunk_size": self.chunk_size,
            "max_workers": self.max_workers,
            "max_retries": self.max_retries,
            "retry_delay": self.retry_delay,
            "connection_pool_size": self.connection_pool_size,
            "active_sessions": len(self.upload_sessions),
            "bucket_name": self.bucket_name,
            "endpoint": settings.minio_endpoint
        }

    def update_performance_config(
        self,
        chunk_size: int = None,
        max_workers: int = None,
        max_retries: int = None,
        retry_delay: float = None,
        connection_pool_size: int = None
    ):
        """Update runtime performance configuration."""
        if chunk_size:
            self.chunk_size = chunk_size
        if max_workers:
            self.max_workers = max_workers
        if max_retries:
            self.max_retries = max_retries
        if retry_delay:
            self.retry_delay = retry_delay
        if connection_pool_size:
            self.connection_pool_size = connection_pool_size

        logger.info("Performance configuration updated")


# Global optimized MinIO service instance
minio_service = MinioService()
