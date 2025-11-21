"""Asynchronous MinIO client implementation with chunked uploads and progress tracking."""
import asyncio
import io
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from urllib.parse import urlparse

import aiofiles
import aiohttp
from minio import Minio
from minio.datatypes import Object
from minio.error import S3Error

from app.config import settings
from app.utils.logger import get_logger

logger: logging.Logger = get_logger(__name__)

@dataclass
class UploadProgress:
    """Progress metadata reported during uploads."""
    object_name: str
    uploaded_bytes: int
    total_bytes: int
    percentage: float
    speed_mbps: float
    eta_seconds: float

@dataclass
class ChunkInfo:
    """Metadata about individual chunks in a multipart upload."""
    chunk_id: int
    offset: int
    size: int
    etag: Optional[str] = None
    uploaded: bool = False
    retry_count: int = 0

class AsyncMinioClient:
    """Async MinIO client offering chunked uploads, resumable sessions, and progress updates."""

    def __init__(
        self,
        endpoint: str,
        access_key: str,
        secret_key: str,
        secure: bool = False,
        chunk_size: int = 100 * 1024 * 1024,  # 100MB chunk size
        max_workers: int = 10,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        connection_pool_size: int = 100
    ):
        """
        Initialize asynchronous MinIO client wrapper.

        Args:
            endpoint: MinIO server endpoint.
            access_key: Access key credential.
            secret_key: Secret key credential.
            secure: Whether to use TLS/HTTPS.
            chunk_size: Chunk size in bytes (default 100MB).
            max_workers: Max concurrent chunk uploads.
            max_retries: Retry attempts per chunk.
            retry_delay: Base delay for retry backoff.
            connection_pool_size: aiohttp connection pool size.
        """
        self.endpoint = endpoint
        self.access_key = access_key
        self.secret_key = secret_key
        self.secure = secure
        self.chunk_size = chunk_size
        self.max_workers = max_workers
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.connection_pool_size = connection_pool_size

        # Create underlying MinIO client
        self.client = Minio(
            endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=secure
        )

        # aiohttp session placeholder (initialized in __aenter__)
        self.session = None

        # Upload session store (used for resumable uploads)
        self.upload_sessions = {}

    async def __aenter__(self):
        """Async context manager entrypoint."""
        # Create aiohttp session
        connector = aiohttp.TCPConnector(
            limit=self.connection_pool_size,
            limit_per_host=50,
            keepalive_timeout=30
        )

        timeout = aiohttp.ClientTimeout(total=300, connect=60)

        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout
        )

        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit cleanup."""
        if self.session:
            await self.session.close()

    async def ensure_bucket_exists(self, bucket_name: str) -> bool:
        """Ensure specified bucket exists, creating it if necessary."""
        try:
            if not self.client.bucket_exists(bucket_name):
                logger.info(f"Creating bucket: {bucket_name}")
                self.client.make_bucket(bucket_name)
                logger.info(f"Bucket '{bucket_name}' created successfully")
            return True
        except S3Error as e:
            logger.error(f"Failed to create bucket '{bucket_name}': {e}")
            return False

    async def upload_file_with_progress(
        self,
        file_path: str,
        bucket_name: str,
        object_name: str,
        content_type: str = "application/octet-stream",
        progress_callback: Optional[Callable[[UploadProgress], None]] = None
    ) -> Dict[str, Any]:
        """
        Upload a file while emitting progress callbacks.

        Args:
            file_path: Local source path.
            bucket_name: Target bucket name.
            object_name: Target object name.
            content_type: MIME type applied to object.
            progress_callback: Optional progress callback.

        Returns:
            dict describing upload outcome.
        """
        try:
            file_size = os.path.getsize(file_path)

            # Choose small-file vs multipart strategy
            if file_size <= self.chunk_size:
                return await self._upload_small_file(
                    file_path, bucket_name, object_name, content_type, progress_callback
                )
            else:
                return await self._upload_large_file(
                    file_path, bucket_name, object_name, content_type, progress_callback
                )

        except Exception as e:
            logger.error(f"Upload failed for {object_name}: {e}")
            return {
                "success": False,
                "object_name": object_name,
                "error": str(e),
                "uploaded_bytes": 0,
                "total_bytes": file_size if 'file_size' in locals() else 0
            }

    async def _upload_small_file(
        self,
        file_path: str,
        bucket_name: str,
        object_name: str,
        content_type: str,
        progress_callback: Optional[Callable[[UploadProgress], None]]
    ) -> Dict[str, Any]:
        """Upload small files in a single request."""
        start_time = time.time()
        uploaded_bytes = 0

        try:
            # Determine file size
            file_size = os.path.getsize(file_path)

            async with aiofiles.open(file_path, 'rb') as file:
                # Read entire file contents
                file_data = await file.read()
                uploaded_bytes = len(file_data)

                # Wrap bytes in file-like object (MinIO put_object expects file-like)
                file_like = io.BytesIO(file_data)

                # Run blocking put_object in executor to avoid blocking event loop
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    None,
                    lambda: self.client.put_object(
                        bucket_name,
                        object_name,
                        file_like,
                        len(file_data),
                        content_type=content_type
                    )
                )

            upload_time = time.time() - start_time
            speed_mbps = (uploaded_bytes / (1024 * 1024)) / upload_time if upload_time > 0 else 0

            # Emit progress callback
            if progress_callback:
                progress = UploadProgress(
                    object_name=object_name,
                    uploaded_bytes=uploaded_bytes,
                    total_bytes=file_size,
                    percentage=100.0,
                    speed_mbps=speed_mbps,
                    eta_seconds=0
                )
                progress_callback(progress)

            return {
                "success": True,
                "object_name": object_name,
                "bucket_name": bucket_name,
                "uploaded_bytes": uploaded_bytes,
                "total_bytes": file_size,
                "upload_time": upload_time,
                "speed_mbps": speed_mbps,
                "etag": None  # requires separate stat_object call to fetch
            }

        except Exception as e:
            logger.error(f"Small file upload failed: {e}")
            return {
                "success": False,
                "object_name": object_name,
                "bucket_name": bucket_name,
                "error": str(e),
                "uploaded_bytes": uploaded_bytes,
                "total_bytes": file_size if 'file_size' in locals() else 0
            }

    async def _upload_large_file(
        self,
        file_path: str,
        bucket_name: str,
        object_name: str,
        content_type: str,
        progress_callback: Optional[Callable[[UploadProgress], None]]
    ) -> Dict[str, Any]:
        """Upload large files using multipart/chunked strategy."""
        start_time = time.time()
        file_size = os.path.getsize(file_path)

        # Build chunk metadata list
        chunks = self._create_chunks(file_path, file_size)

        # Initialize session tracking
        session_id = f"{bucket_name}/{object_name}/{int(time.time())}"
        self.upload_sessions[session_id] = {
            "bucket_name": bucket_name,
            "object_name": object_name,
            "file_path": file_path,
            "content_type": content_type,
            "chunks": chunks,
            "uploaded_chunks": 0,
            "uploaded_bytes": 0,
            "total_bytes": file_size,
            "start_time": start_time
        }

        try:
            # Initialize multipart upload id
            upload_id = self.client.client.create_multipart_upload(
                bucket_name, object_name
            ).upload_id

            self.upload_sessions[session_id]["upload_id"] = upload_id

            # Upload chunks concurrently
            await self._upload_chunks_parallel(
                session_id, progress_callback
            )

            # Complete multipart upload
            etag = await self._complete_multipart_upload(session_id)

            upload_time = time.time() - start_time
            speed_mbps = (file_size / (1024 * 1024)) / upload_time if upload_time > 0 else 0

            # Clean up session tracking
            del self.upload_sessions[session_id]

            return {
                "success": True,
                "object_name": object_name,
                "bucket_name": bucket_name,
                "uploaded_bytes": file_size,
                "total_bytes": file_size,
                "upload_time": upload_time,
                "speed_mbps": speed_mbps,
                "etag": etag,
                "chunks_count": len(chunks)
            }

        except Exception as e:
            logger.error(f"Large file upload failed: {e}")

            # Attempt to abort partial upload if necessary
            if "upload_id" in self.upload_sessions.get(session_id, {}):
                try:
                    self.client.client.abort_multipart_upload(
                        bucket_name, object_name,
                        self.upload_sessions[session_id]["upload_id"]
                    )
                except Exception as cleanup_error:
                    logger.warning(f"Failed to cleanup multipart upload: {cleanup_error}")

            # Remove session tracking
            if session_id in self.upload_sessions:
                del self.upload_sessions[session_id]

            return {
                "success": False,
                "object_name": object_name,
                "bucket_name": bucket_name,
                "error": str(e),
                "uploaded_bytes": self.upload_sessions.get(session_id, {}).get("uploaded_bytes", 0),
                "total_bytes": file_size,
                "chunks_count": len(chunks)
            }

    def _create_chunks(self, file_path: str, file_size: int) -> List[ChunkInfo]:
        """Generate chunk metadata for multipart upload."""
        chunks = []
        chunk_id = 1

        for offset in range(0, file_size, self.chunk_size):
            size = min(self.chunk_size, file_size - offset)
            chunks.append(ChunkInfo(
                chunk_id=chunk_id,
                offset=offset,
                size=size
            ))
            chunk_id += 1

        return chunks

    async def _upload_chunks_parallel(
        self,
        session_id: str,
        progress_callback: Optional[Callable[[UploadProgress], None]]
    ):
        """Upload all chunks concurrently using bounded semaphore."""
        session = self.upload_sessions[session_id]
        chunks = session["chunks"]

        # Bound concurrency with semaphore
        semaphore = asyncio.Semaphore(self.max_workers)

        async def upload_chunk_with_semaphore(chunk: ChunkInfo):
            async with semaphore:
                return await self._upload_single_chunk(session_id, chunk, progress_callback)

        # Create upload tasks
        tasks = [
            upload_chunk_with_semaphore(chunk)
            for chunk in chunks
        ]

        # Await completion of all chunk uploads
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Identify failed chunks
        failed_chunks = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                failed_chunks.append(chunks[i])
                logger.error(f"Chunk {chunks[i].chunk_id} failed: {result}")

        # Retry failed chunks individually
        await self._retry_failed_chunks(session_id, failed_chunks, progress_callback)

    async def _upload_single_chunk(
        self,
        session_id: str,
        chunk: ChunkInfo,
        progress_callback: Optional[Callable[[UploadProgress], None]]
    ) -> bool:
        """Upload a single chunk with retries and progress updates."""
        session = self.upload_sessions[session_id]
        bucket_name = session["bucket_name"]
        object_name = session["object_name"]
        file_path = session["file_path"]
        upload_id = session["upload_id"]

        for attempt in range(self.max_retries + 1):
            try:
                # Read chunk bytes
                async with aiofiles.open(file_path, 'rb') as file:
                    await file.seek(chunk.offset)
                    chunk_data = await file.read(chunk.size)

                # Upload chunk part
                result = self.client.client.put_object_part(
                    bucket_name,
                    object_name,
                    upload_id,
                    chunk.chunk_id,
                    chunk_data,
                    len(chunk_data)
                )

                # Update chunk metadata
                chunk.etag = result.etag
                chunk.uploaded = True
                chunk.retry_count = attempt

                # Update session counters
                session["uploaded_chunks"] += 1
                session["uploaded_bytes"] += chunk.size

                # Emit progress callback
                if progress_callback:
                    progress = self._calculate_progress(session)
                    progress_callback(progress)

                logger.debug(f"Chunk {chunk.chunk_id} uploaded successfully")
                return True

            except Exception as e:
                logger.warning(
                    f"Chunk {chunk.chunk_id} upload attempt {attempt + 1} failed: {e}"
                )
                if attempt < self.max_retries:
                    await asyncio.sleep(self.retry_delay * (2 ** attempt))  # exponential backoff
                else:
                    logger.error(f"Chunk {chunk.chunk_id} failed after {self.max_retries} retries")
                    return False

        return False

    async def _retry_failed_chunks(
        self,
        session_id: str,
        failed_chunks: List[ChunkInfo],
        progress_callback: Optional[Callable[[UploadProgress], None]]
    ):
        """Retry uploading chunks that previously failed."""
        if not failed_chunks:
            return

        logger.info(f"Retrying {len(failed_chunks)} failed chunks for session {session_id}")

        for chunk in failed_chunks:
            success = await self._upload_single_chunk(session_id, chunk, progress_callback)
            if not success:
                raise Exception(f"Failed to upload chunk {chunk.chunk_id} after all retries")

    def _calculate_progress(self, session: Dict) -> UploadProgress:
        """Calculate overall progress for the given session."""
        uploaded_bytes = session["uploaded_bytes"]
        total_bytes = session["total_bytes"]
        start_time = session["start_time"]

        percentage = (uploaded_bytes / total_bytes) * 100 if total_bytes > 0 else 0

        # Calculate speed
        elapsed_time = time.time() - start_time
        speed_mbps = (uploaded_bytes / (1024 * 1024)) / elapsed_time if elapsed_time > 0 else 0

        # Estimate remaining time
        if speed_mbps > 0:
            remaining_bytes = total_bytes - uploaded_bytes
            eta_seconds = remaining_bytes / (speed_mbps * 1024 * 1024)
        else:
            eta_seconds = 0

        return UploadProgress(
            object_name=session["object_name"],
            uploaded_bytes=uploaded_bytes,
            total_bytes=total_bytes,
            percentage=percentage,
            speed_mbps=speed_mbps,
            eta_seconds=eta_seconds
        )

    async def _complete_multipart_upload(self, session_id: str) -> str:
        """Finalize multipart upload by committing all parts."""
        session = self.upload_sessions[session_id]
        bucket_name = session["bucket_name"]
        object_name = session["object_name"]
        chunks = session["chunks"]
        upload_id = session["upload_id"]

        # Collect uploaded chunk metadata
        uploaded_chunks = [
            (chunk.chunk_id, chunk.etag)
            for chunk in chunks if chunk.uploaded and chunk.etag
        ]

        # Sort parts by chunk id
        uploaded_chunks.sort(key=lambda x: x[0])

        if len(uploaded_chunks) != len(chunks):
            raise Exception(f"Not all chunks uploaded: {len(uploaded_chunks)}/{len(chunks)}")

        # Complete multipart upload
        result = self.client.client.complete_multipart_upload(
            bucket_name,
            object_name,
            upload_id,
            uploaded_chunks
        )

        return result.etag

    async def upload_files_parallel(
        self,
        files: List[Tuple[str, str, str]],  # (file_path, object_name, content_type)
        bucket_name: str,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None
    ) -> Dict[str, Any]:
        """Upload multiple files in parallel with per-file progress callbacks."""

        async def upload_single_file(args) -> Dict[str, Any]:
            file_path, object_name, content_type = args
            return await self.upload_file_with_progress(
                file_path, bucket_name, object_name, content_type, progress_callback
            )

        # Limit concurrency via semaphore
        semaphore = asyncio.Semaphore(self.max_workers)

        async def upload_with_semaphore(args):
            async with semaphore:
                return await upload_single_file(args)

        # Build upload tasks
        tasks = [upload_with_semaphore(file_info) for file_info in files]

        # Execute uploads concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Aggregate success/failure results
        successful = []
        failed = []

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                failed.append({
                    "file_path": files[i][0],
                    "object_name": files[i][1],
                    "error": str(result)
                })
            elif result.get("success"):
                successful.append(result)
            else:
                failed.append(result)

        return {
            "total": len(files),
            "successful": len(successful),
            "failed": len(failed),
            "successful_files": successful,
            "failed_files": failed
        }

    async def get_presigned_url(
        self,
        bucket_name: str,
        object_name: str,
        expires: int = 3600
    ) -> Optional[str]:
        """Generate a presigned GET URL for an object."""
        try:
            url = self.client.presigned_get_object(
                bucket_name,
                object_name,
                expires
            )
            return url
        except Exception as e:
            logger.error(f"Failed to generate presigned URL for {object_name}: {e}")
            return None

    async def get_file_stat(self, bucket_name: str, object_name: str) -> Optional[Dict]:
        """Return metadata/statistics for an object."""
        try:
            stat = self.client.stat_object(bucket_name, object_name)
            return {
                "bucket_name": bucket_name,
                "object_name": object_name,
                "size": stat.size,
                "content_type": stat.content_type,
                "etag": stat.etag,
                "last_modified": stat.last_modified
            }
        except Exception as e:
            logger.error(f"Failed to get stat for {object_name}: {e}")
            return None

    async def delete_file(self, bucket_name: str, object_name: str) -> bool:
        """Delete an object asynchronously (still calls sync SDK)."""
        try:
            self.client.remove_object(bucket_name, object_name)
            return True
        except Exception as e:
            logger.error(f"Failed to delete {object_name}: {e}")
            return False

    async def list_objects(
        self,
        bucket_name: str,
        prefix: str = "",
        max_keys: int = 1000
    ) -> List[Dict]:
        """List objects under a prefix."""
        try:
            objects = []
            for obj in self.client.list_objects(bucket_name, prefix=prefix, recursive=True):
                objects.append({
                    "object_name": obj.object_name,
                    "size": obj.size,
                    "last_modified": obj.last_modified,
                    "etag": obj.etag
                })
                if len(objects) >= max_keys:
                    break

            return objects
        except Exception as e:
            logger.error(f"Failed to list objects: {e}")
            return []

    async def resume_upload(self, session_id: str) -> Dict[str, Any]:
        """Resume previously started multipart upload."""
        if session_id not in self.upload_sessions:
            raise ValueError(f"Session {session_id} not found")

        session = self.upload_sessions[session_id]
        bucket_name = session["bucket_name"]
        object_name = session["object_name"]
        file_path = session["file_path"]
        content_type = session["content_type"]

        try:
            # Query already uploaded parts
            upload_id = session["upload_id"]

            # List uploaded parts from MinIO
            uploaded_parts = self.client.client.list_multipart_parts(
                bucket_name, object_name, upload_id
            )

            uploaded_part_numbers = {part.part_number for part in uploaded_parts}

            # Mark chunks as uploaded based on part list
            for chunk in session["chunks"]:
                if chunk.chunk_id in uploaded_part_numbers:
                    chunk.uploaded = True

            # Continue uploading pending chunks
            pending_chunks = [chunk for chunk in session["chunks"] if not chunk.uploaded]

            if pending_chunks:
                logger.info(f"Resuming upload with {len(pending_chunks)} pending chunks")

                # Use a no-op progress callback
                async def no_progress(progress):
                    pass

                await self._upload_chunks_parallel(session_id, no_progress)

            # Finalize multipart upload
            etag = await self._complete_multipart_upload(session_id)

            upload_time = time.time() - session["start_time"]
            speed_mbps = (session["total_bytes"] / (1024 * 1024)) / upload_time

            # Remove session tracking
            del self.upload_sessions[session_id]

            return {
                "success": True,
                "object_name": object_name,
                "bucket_name": bucket_name,
                "uploaded_bytes": session["total_bytes"],
                "total_bytes": session["total_bytes"],
                "upload_time": upload_time,
                "speed_mbps": speed_mbps,
                "etag": etag
            }

        except Exception as e:
            logger.error(f"Failed to resume upload: {e}")
            return {
                "success": False,
                "object_name": object_name,
                "bucket_name": bucket_name,
                "error": str(e)
            }

    def save_upload_session(self, session_id: str, filepath: str):
        """Persist upload session metadata to disk."""
        if session_id in self.upload_sessions:
            session_data = self.upload_sessions[session_id].copy()
            # Remove non-serializable fields
            session_data.pop('chunks', None)

            with open(filepath, 'w') as f:
                json.dump(session_data, f, indent=2, default=str)

    def load_upload_session(self, session_id: str, filepath: str) -> bool:
        """Load upload session metadata from disk."""
        try:
            with open(filepath, 'r') as f:
                session_data = json.load(f)

            self.upload_sessions[session_id] = session_data
            return True

        except Exception as e:
            logger.error(f"Failed to load upload session: {e}")
            return False
