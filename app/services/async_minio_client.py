"""异步MinIO客户端实现"""
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
    """上传进度信息"""
    object_name: str
    uploaded_bytes: int
    total_bytes: int
    percentage: float
    speed_mbps: float
    eta_seconds: float

@dataclass
class ChunkInfo:
    """分片信息"""
    chunk_id: int
    offset: int
    size: int
    etag: Optional[str] = None
    uploaded: bool = False
    retry_count: int = 0

class AsyncMinioClient:
    """异步MinIO客户端"""

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
        初始化异步MinIO客户端

        Args:
            endpoint: MinIO服务器地址
            access_key: 访问密钥
            secret_key: 密钥
            secure: 是否使用HTTPS
            chunk_size: 分片大小（默认100MB）
            max_workers: 最大并发数
            max_retries: 最大重试次数
            retry_delay: 重试延迟
            connection_pool_size: 连接池大小
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

        # 创建MinIO客户端
        self.client = Minio(
            endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=secure
        )

        # 创建HTTP会话池
        self.session = None

        # 上传会话存储（用于断点续传）
        self.upload_sessions = {}

    async def __aenter__(self):
        """异步上下文管理器入口"""
        # 创建aiohttp会话
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
        """异步上下文管理器出口"""
        if self.session:
            await self.session.close()

    async def ensure_bucket_exists(self, bucket_name: str) -> bool:
        """确保bucket存在"""
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
        带进度追踪的文件上传

        Args:
            file_path: 本地文件路径
            bucket_name: bucket名称
            object_name: 对象名称
            content_type: MIME类型
            progress_callback: 进度回调函数

        Returns:
            上传结果字典
        """
        try:
            file_size = os.path.getsize(file_path)

            # 检查文件大小，决定是否使用分片上传
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
        """上传小文件（单片）"""
        start_time = time.time()
        uploaded_bytes = 0

        try:
            # 读取文件并计算大小
            file_size = os.path.getsize(file_path)

            async with aiofiles.open(file_path, 'rb') as file:
                # 读取文件内容
                file_data = await file.read()
                uploaded_bytes = len(file_data)

                # 将 bytes 包装成 file-like 对象（MinIO put_object 需要 file-like 对象）
                file_like = io.BytesIO(file_data)

                # 在线程池中运行同步的 put_object（避免阻塞事件循环）
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

            # 调用进度回调
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
                "etag": None  # 需要额外调用stat_object获取
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
        """上传大文件（分片并行上传）"""
        start_time = time.time()
        file_size = os.path.getsize(file_path)

        # 创建分片信息
        chunks = self._create_chunks(file_path, file_size)

        # 初始化上传会话
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
            # 创建上传ID
            upload_id = self.client.client.create_multipart_upload(
                bucket_name, object_name
            ).upload_id

            self.upload_sessions[session_id]["upload_id"] = upload_id

            # 并行上传分片
            await self._upload_chunks_parallel(
                session_id, progress_callback
            )

            # 完成上传
            etag = await self._complete_multipart_upload(session_id)

            upload_time = time.time() - start_time
            speed_mbps = (file_size / (1024 * 1024)) / upload_time if upload_time > 0 else 0

            # 清理会话
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

            # 尝试清理部分上传
            if "upload_id" in self.upload_sessions.get(session_id, {}):
                try:
                    self.client.client.abort_multipart_upload(
                        bucket_name, object_name,
                        self.upload_sessions[session_id]["upload_id"]
                    )
                except Exception as cleanup_error:
                    logger.warning(f"Failed to cleanup multipart upload: {cleanup_error}")

            # 清理会话
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
        """创建分片信息"""
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
        """并行上传分片"""
        session = self.upload_sessions[session_id]
        chunks = session["chunks"]

        # 创建信号量限制并发数
        semaphore = asyncio.Semaphore(self.max_workers)

        async def upload_chunk_with_semaphore(chunk: ChunkInfo):
            async with semaphore:
                return await self._upload_single_chunk(session_id, chunk, progress_callback)

        # 创建上传任务
        tasks = [
            upload_chunk_with_semaphore(chunk)
            for chunk in chunks
        ]

        # 并行执行所有上传任务
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 处理结果
        failed_chunks = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                failed_chunks.append(chunks[i])
                logger.error(f"Chunk {chunks[i].chunk_id} failed: {result}")

        # 重试失败的chunks
        await self._retry_failed_chunks(session_id, failed_chunks, progress_callback)

    async def _upload_single_chunk(
        self,
        session_id: str,
        chunk: ChunkInfo,
        progress_callback: Optional[Callable[[UploadProgress], None]]
    ) -> bool:
        """上传单个分片"""
        session = self.upload_sessions[session_id]
        bucket_name = session["bucket_name"]
        object_name = session["object_name"]
        file_path = session["file_path"]
        upload_id = session["upload_id"]

        for attempt in range(self.max_retries + 1):
            try:
                # 读取分片数据
                async with aiofiles.open(file_path, 'rb') as file:
                    await file.seek(chunk.offset)
                    chunk_data = await file.read(chunk.size)

                # 上传分片
                result = self.client.client.put_object_part(
                    bucket_name,
                    object_name,
                    upload_id,
                    chunk.chunk_id,
                    chunk_data,
                    len(chunk_data)
                )

                # 更新chunk状态
                chunk.etag = result.etag
                chunk.uploaded = True
                chunk.retry_count = attempt

                # 更新会话状态
                session["uploaded_chunks"] += 1
                session["uploaded_bytes"] += chunk.size

                # 调用进度回调
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
                    await asyncio.sleep(self.retry_delay * (2 ** attempt))  # 指数退避
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
        """重试失败的分片"""
        if not failed_chunks:
            return

        logger.info(f"Retrying {len(failed_chunks)} failed chunks for session {session_id}")

        for chunk in failed_chunks:
            success = await self._upload_single_chunk(session_id, chunk, progress_callback)
            if not success:
                raise Exception(f"Failed to upload chunk {chunk.chunk_id} after all retries")

    def _calculate_progress(self, session: Dict) -> UploadProgress:
        """计算上传进度"""
        uploaded_bytes = session["uploaded_bytes"]
        total_bytes = session["total_bytes"]
        start_time = session["start_time"]

        percentage = (uploaded_bytes / total_bytes) * 100 if total_bytes > 0 else 0

        # 计算速度
        elapsed_time = time.time() - start_time
        speed_mbps = (uploaded_bytes / (1024 * 1024)) / elapsed_time if elapsed_time > 0 else 0

        # 计算预计剩余时间
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
        """完成分片上传"""
        session = self.upload_sessions[session_id]
        bucket_name = session["bucket_name"]
        object_name = session["object_name"]
        chunks = session["chunks"]
        upload_id = session["upload_id"]

        # 收集已上传的chunks
        uploaded_chunks = [
            (chunk.chunk_id, chunk.etag)
            for chunk in chunks if chunk.uploaded and chunk.etag
        ]

        # 按chunk_id排序
        uploaded_chunks.sort(key=lambda x: x[0])

        if len(uploaded_chunks) != len(chunks):
            raise Exception(f"Not all chunks uploaded: {len(uploaded_chunks)}/{len(chunks)}")

        # 完成上传
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
        """并行上传多个文件"""

        async def upload_single_file(args) -> Dict[str, Any]:
            file_path, object_name, content_type = args
            return await self.upload_file_with_progress(
                file_path, bucket_name, object_name, content_type, progress_callback
            )

        # 创建信号量限制并发数
        semaphore = asyncio.Semaphore(self.max_workers)

        async def upload_with_semaphore(args):
            async with semaphore:
                return await upload_single_file(args)

        # 创建上传任务
        tasks = [upload_with_semaphore(file_info) for file_info in files]

        # 并行执行
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 统计结果
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
        """获取预签名URL"""
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
        """获取文件状态"""
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
        """删除文件"""
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
        """列出对象"""
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
        """恢复上传会话"""
        if session_id not in self.upload_sessions:
            raise ValueError(f"Session {session_id} not found")

        session = self.upload_sessions[session_id]
        bucket_name = session["bucket_name"]
        object_name = session["object_name"]
        file_path = session["file_path"]
        content_type = session["content_type"]

        try:
            # 获取已上传的分片
            upload_id = session["upload_id"]

            # 列出已上传的分片
            uploaded_parts = self.client.client.list_multipart_parts(
                bucket_name, object_name, upload_id
            )

            uploaded_part_numbers = {part.part_number for part in uploaded_parts}

            # 更新chunks状态
            for chunk in session["chunks"]:
                if chunk.chunk_id in uploaded_part_numbers:
                    chunk.uploaded = True

            # 继续上传未完成的分片
            pending_chunks = [chunk for chunk in session["chunks"] if not chunk.uploaded]

            if pending_chunks:
                logger.info(f"Resuming upload with {len(pending_chunks)} pending chunks")

                # 创建一个空的progress回调（不输出进度）
                async def no_progress(progress):
                    pass

                await self._upload_chunks_parallel(session_id, no_progress)

            # 完成上传
            etag = await self._complete_multipart_upload(session_id)

            upload_time = time.time() - session["start_time"]
            speed_mbps = (session["total_bytes"] / (1024 * 1024)) / upload_time

            # 清理会话
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
        """保存上传会话到文件"""
        if session_id in self.upload_sessions:
            session_data = self.upload_sessions[session_id].copy()
            # 不保存文件对象，只保存基本信息
            session_data.pop('chunks', None)

            with open(filepath, 'w') as f:
                json.dump(session_data, f, indent=2, default=str)

    def load_upload_session(self, session_id: str, filepath: str) -> bool:
        """从文件加载上传会话"""
        try:
            with open(filepath, 'r') as f:
                session_data = json.load(f)

            self.upload_sessions[session_id] = session_data
            return True

        except Exception as e:
            logger.error(f"Failed to load upload session: {e}")
            return False
