"""优化的MinIO服务，支持异步操作、连接池、进度追踪、分片上传、断点续传等功能"""
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
    """优化的MinIO服务类"""

    def __init__(self):
        """初始化MinIO服务"""
        logger.info(f"Initializing optimized MinIO service")

        # 初始化同步MinIO客户端（用于不支持异步的操作）
        self.sync_client = Minio(
            settings.minio_endpoint,
            access_key=settings.minio_access_key,
            secret_key=settings.minio_secret_key,
            secure=settings.minio_secure
        )

        self.bucket_name = settings.minio_bucket_name
        self.chunk_size = 100 * 1024 * 1024  # 100MB分片大小
        self.max_workers = 20  # 最大并发数
        self.max_retries = 3  # 最大重试次数
        self.retry_delay = 1.0  # 重试延迟
        self.connection_pool_size = 100  # 连接池大小

        # 确保bucket存在
        self._ensure_bucket_exists()

        # 上传会话存储
        self.upload_sessions = {}

        logger.info("Optimized MinIO service initialized successfully")

    def _ensure_bucket_exists(self) -> None:
        """确保bucket存在"""
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
        异步上传文件（支持进度追踪和分片上传）

        Args:
            file_path: 本地文件路径
            object_name: MinIO中的对象名称
            content_type: MIME类型
            progress_callback: 进度回调函数
            bucket_name: bucket名称（可选）

        Returns:
            上传结果字典
        """
        bucket = bucket_name or self.bucket_name

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        file_size = os.path.getsize(file_path)
        logger.info(f"Async upload started: {object_name} ({file_size / (1024**3):.2f} GB)")

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
            # 确保bucket存在
            await client.ensure_bucket_exists(bucket)

            # 上传文件
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
                    f"{result['uploaded_bytes'] / (1024**3):.2f} GB in "
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
        异步并行上传多个文件

        Args:
            files: 文件信息列表
            bucket_name: bucket名称
            progress_callback: 进度回调函数

        Returns:
            上传结果字典
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

        # 验证文件存在
        valid_files = []
        for file_path, object_name, content_type in files:
            if os.path.exists(file_path):
                valid_files.append((file_path, object_name, content_type))
            else:
                logger.warning(f"File not found: {file_path}")

        if not valid_files:
            raise ValueError("No valid files found")

        logger.info(f"Starting async parallel upload of {len(valid_files)} files")

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
            # 确保bucket存在
            await client.ensure_bucket_exists(bucket)

            # 并行上传文件
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
        带重试机制的文件上传（同步版本，兼容现有代码）

        Args:
            file_path: 本地文件路径
            object_name: MinIO中的对象名称
            content_type: MIME类型
            max_retries: 最大重试次数
            bucket_name: bucket名称
            enable_async: 是否启用异步（如果文件很大）

        Returns:
            上传结果字典
        """
        bucket = bucket_name or self.bucket_name
        max_retries = max_retries or self.max_retries

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        file_size = os.path.getsize(file_path)

        # 对于大文件，自动使用异步版本
        if enable_async and file_size > 50 * 1024 * 1024:  # 50MB+
            # 创建事件循环运行异步代码
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

        # 小文件使用同步上传
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
        """同步上传带重试"""
        file_size = os.path.getsize(file_path)

        for attempt in range(max_retries + 1):
            try:
                # 上传文件
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
                    "upload_time": None,  # 同步版本不计算时间
                    "speed_mbps": None,
                    "attempt": attempt + 1
                }

            except Exception as e:
                logger.warning(
                    f"Upload attempt {attempt + 1} failed for {object_name}: {e}"
                )

                if attempt < max_retries:
                    time.sleep(self.retry_delay * (2 ** attempt))  # 指数退避
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
        带进度追踪的批量文件上传

        Args:
            files: 文件列表 [(file_path, object_name), ...]
            max_workers: 最大并发数
            max_retries: 最大重试次数
            content_type: 默认内容类型
            progress_callback: 进度回调函数
            bucket_name: bucket名称
            enable_async: 是否启用异步

        Returns:
            上传结果字典
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

        # 分类文件：小文件用同步，大文件用异步
        small_files = []
        large_files = []

        for file_path, object_name in files:
            if os.path.exists(file_path):
                file_size = os.path.getsize(file_path)
                if file_size > 50 * 1024 * 1024:  # 大于50MB
                    large_files.append((file_path, object_name, content_type))
                else:
                    small_files.append((file_path, object_name, content_type))
            else:
                logger.warning(f"File not found: {file_path}")

        results = []
        success_list = []
        failed_list = []

        # 处理小文件（同步并发）
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

        # 处理大文件（异步）
        if large_files:
            logger.info(f"Processing {len(large_files)} large files with async upload")

            # 异步上传大文件
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

                # 合并结果
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
            "retry_info": {"total_retries": 0, "retry_attempts": []}  # 重试已在内部处理
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
        启动分片上传会话（用于断点续传）

        Args:
            file_path: 文件路径
            object_name: 对象名称
            bucket_name: bucket名称
            session_id: 会话ID（可选）

        Returns:
            会话ID
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

            # 启动分片上传
            result = await client.upload_file_with_progress(
                file_path=file_path,
                bucket_name=bucket,
                object_name=object_name,
                progress_callback=None  # 不显示进度，用于初始化会话
            )

            if result["success"]:
                # 保存会话信息
                self.upload_sessions[session_id] = {
                    "bucket_name": bucket,
                    "object_name": object_name,
                    "file_path": file_path,
                    "status": "initializing",
                    "created_at": time.time(),
                    "result": result
                }

                # 保存会话到文件
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
        恢复上传会话

        Args:
            session_id: 会话ID
            progress_callback: 进度回调函数

        Returns:
            上传结果
        """
        if session_id not in self.upload_sessions:
            # 尝试从文件加载会话
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
                # 更新会话状态
                session["status"] = "completed"
                session["completed_at"] = time.time()
                session["result"] = result

                # 清理会话文件
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
        """获取上传会话状态"""
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
        """清理已完成的会话"""
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600

        sessions_to_remove = []

        for session_id, session in self.upload_sessions.items():
            if session.get("status") in ["completed", "failed"]:
                if current_time - session.get("created_at", current_time) > max_age_seconds:
                    sessions_to_remove.append(session_id)

        for session_id in sessions_to_remove:
            del self.upload_sessions[session_id]
            # 清理会话文件
            session_file = f"/tmp/upload_session_{session_id.replace('/', '_')}.json"
            try:
                os.remove(session_file)
            except:
                pass

        logger.info(f"Cleaned up {len(sessions_to_remove)} old sessions")
        return len(sessions_to_remove)

    # 保持向后兼容的同步方法
    def upload_file(self, file_path: str, object_name: str, content_type: str = "application/octet-stream") -> str:
        """上传文件（同步版本，向后兼容）"""
        result = self.upload_file_with_retry(file_path, object_name, content_type)

        if not result["success"]:
            raise Exception(f"Upload failed: {result.get('error')}")

        return f"http://{settings.minio_endpoint}/{self.bucket_name}/{object_name}"

    def get_file_url(self, object_name: str) -> str:
        """获取预签名URL（同步版本）"""
        try:
            url = self.sync_client.presigned_get_object(self.bucket_name, object_name)
            return url
        except Exception as e:
            logger.error(f"Failed to generate URL for '{object_name}': {e}")
            raise Exception(f"Failed to generate URL: {e}")

    def delete_file(self, object_name: str) -> bool:
        """删除文件（同步版本）"""
        try:
            self.sync_client.remove_object(self.bucket_name, object_name)
            logger.info(f"File deleted successfully: {object_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete file '{object_name}': {e}")
            return False

    def file_exists(self, object_name: str) -> bool:
        """检查文件是否存在（同步版本）"""
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
        批量上传文件（向后兼容版本，自动选择最优策略）

        Args:
            file_list: 文件列表 [(file_path, object_name), ...]
            max_workers: 最大并发数
            max_retries: 最大重试次数
            retry_delay: 重试延迟
            content_type: 默认内容类型
            progress_callback: 进度回调函数

        Returns:
            上传结果字典
        """
        # 使用新的优化版本
        return self.upload_files_with_progress(
            files=file_list,
            max_workers=max_workers,
            max_retries=max_retries,
            content_type=content_type,
            progress_callback=progress_callback,
            enable_async=True  # 启用异步优化
        )

    def get_files_urls(self, object_names: List[str], max_workers: int = 20) -> Dict[str, Any]:
        """
        批量获取URL（同步版本，向后兼容）

        Args:
            object_names: 对象名称列表
            max_workers: 最大并发数

        Returns:
            结果字典
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

    # 性能监控方法
    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计信息"""
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
        """更新性能配置"""
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


# 全局优化后的MinIO服务实例
minio_service = MinioService()
