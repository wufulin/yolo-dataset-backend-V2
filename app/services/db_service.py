"""Database service for MongoDB connection management with Motor async support."""
import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import (
    Any,
    AsyncGenerator,
    Dict,
    Generic,
    List,
    Optional,
    Protocol,
    TypeVar,
    Union,
    runtime_checkable,
)

import bson
from bson import ObjectId
from motor.motor_asyncio import (
    AsyncIOMotorClient,
    AsyncIOMotorCollection,
    AsyncIOMotorDatabase,
)
from pymongo import MongoClient, ReturnDocument
from pymongo.collation import Collation
from pymongo.errors import BulkWriteError, DuplicateKeyError, PyMongoError

from app.config import settings
from app.core.decorators import async_retry, exception_handler, performance_monitor
from app.core.exceptions import DatabaseException
from app.utils.logger import get_logger

logger: logging.Logger = get_logger(__name__)


class DatabaseService:
    """Service class for MongoDB connection management with Motor async support (Singleton)."""

    _instance: Optional['DatabaseService'] = None

    def __new__(cls, *args, **kwargs):
        """Ensure only one instance is created."""
        if cls._instance is None:
            cls._instance = super(DatabaseService, cls).__new__(cls)
        return cls._instance

    @exception_handler(DatabaseException, "Database service initialization failed.")
    @performance_monitor("database_service_init")
    def __init__(self, max_pool_size: Optional[int] = None, retry_writes: Optional[bool] = None,
                 max_idle_time_ms: Optional[int] = None):
        """Initialize MongoDB clients with connection pooling."""
        # Prevent re-initialization if already initialized
        if hasattr(self, 'sync_client'):
            return

        # Use config settings if not provided
        max_pool_size = max_pool_size or settings.mongodb_max_pool_size
        retry_writes = retry_writes if retry_writes is not None else settings.mongodb_retry_writes
        max_idle_time_ms = max_idle_time_ms or settings.mongodb_max_idle_time_ms

        # Sync client for blocking operations
        self.sync_client = MongoClient(
            settings.mongodb_url,
            maxPoolSize=max_pool_size,
            maxIdleTimeMS=max_idle_time_ms,
            retryWrites=retry_writes,
            serverSelectionTimeoutMS=settings.mongodb_server_selection_timeout_ms,
            connectTimeoutMS=settings.mongodb_connect_timeout_ms,
            socketTimeoutMS=settings.mongodb_socket_timeout_ms,
            heartbeatFrequencyMS=settings.mongodb_heartbeat_frequency_ms,
            waitQueueMultiple=settings.mongodb_wait_queue_multiple,
            waitQueueTimeoutMS=settings.mongodb_wait_queue_timeout_ms
        )
        self.sync_db = self.sync_client[settings.mongo_db_name]

        # Async client for non-blocking operations
        try:
            loop = asyncio.get_event_loop()
            self.async_client = AsyncIOMotorClient(
                settings.mongodb_url,
                maxPoolSize=max_pool_size,
                maxIdleTimeMS=max_idle_time_ms,
                retryWrites=retry_writes,
                serverSelectionTimeoutMS=settings.mongodb_server_selection_timeout_ms,
                connectTimeoutMS=settings.mongodb_connect_timeout_ms,
                socketTimeoutMS=settings.mongodb_socket_timeout_ms,
                heartbeatFrequencyMS=settings.mongodb_heartbeat_frequency_ms,
                waitQueueMultiple=settings.mongodb_wait_queue_multiple,
                waitQueueTimeoutMS=settings.mongodb_wait_queue_timeout_ms,
                io_loop=loop if loop.is_running() else None
            )
        except Exception as e:
            logger.warning(f"Failed to initialize async client (may be in sync context): {e}")
            self.async_client = None
            self.async_db = None

        if self.async_client is not None:
            self.async_db = self.async_client[settings.mongo_db_name]
        else:
            self.async_db = None

        # Collections (both sync and async versions)
        self.datasets = self.sync_db.datasets
        self.images = self.sync_db.images
        self.upload_sessions = self.sync_db.upload_sessions
        self.dataset_statistics = self.sync_db.dataset_statistics
        self.users = self.sync_db.users
        self.annotations = self.sync_db.annotations

        # Async collections (only if async client is available)
        if self.async_db is not None:
            self.async_datasets = self.async_db.datasets
            self.async_images = self.async_db.images
            self.async_upload_sessions = self.async_db.upload_sessions
            self.async_dataset_statistics = self.async_db.dataset_statistics
            self.async_users = self.async_db.users
            self.async_annotations = self.async_db.annotations
        else:
            # Set to None if async client is not available
            self.async_datasets = None
            self.async_images = None
            self.async_upload_sessions = None
            self.async_dataset_statistics = None
            self.async_users = None
            self.async_annotations = None

        # Performance metrics
        self.query_stats = {
            'total_queries': 0,
            'slow_queries': 0,
            'bulk_operations': 0,
            'average_query_time': 0.0
        }

        # Create indexes
        # self._ensure_indexes()

        # Test connection
        self._test_connection()

        # Start background tasks
        self._start_background_tasks()

    def _test_connection(self):
        """Test database connections."""
        try:
            # Test sync connection
            self.sync_client.admin.command('ping')
            logger.info("Successfully connected to MongoDB (sync)")

            # Test async connection
            if self.async_client is not None:
                try:
                    # Use asyncio to test async connection
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        # If loop is running, create a new task
                        asyncio.create_task(self.async_db.admin.command('ping'))
                    else:
                        loop.run_until_complete(self.async_db.admin.command('ping'))
                    logger.info("Successfully connected to MongoDB (async)")
                except Exception as e:
                    logger.warning(f"Async connection test skipped (may be in sync context): {e}")
            else:
                logger.warning("Async client not available for connection testing")

        except PyMongoError as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            raise Exception(f"Failed to connect to MongoDB: {e}")

    def _ensure_indexes(self):
        """Create performance indexes for all collections."""
        try:
            # Define indexes with explicit names to avoid conflicts
            indexes_to_create = [
                # Dataset indexes
                ("datasets", [("name", 1)], {"unique": True, "background": True, "name": "dataset_name_unique"}),
                (
                "datasets", [("user_id", 1), ("created_at", -1)], {"background": True, "name": "dataset_user_created"}),
                ("datasets", [("status", 1)], {"background": True, "name": "dataset_status"}),

                # Image indexes (high performance)
                ("images", [("dataset_id", 1)], {"background": True, "name": "image_dataset"}),
                ("images", [("dataset_id", 1), ("split", 1)], {"background": True, "name": "image_dataset_split"}),
                ("images", [("dataset_id", 1), ("filename", 1)],
                 {"unique": True, "background": True, "name": "image_dataset_filename_unique"}),
                ("images", [("created_at", -1)], {"background": True, "name": "image_created_at"}),
                ("images", [("metadata.width", 1), ("metadata.height", 1)],
                 {"background": True, "name": "image_metadata_dims"}),

                # Annotation indexes
                ("annotations", [("image_id", 1)], {"background": True, "name": "annotation_image"}),
                ("annotations", [("dataset_id", 1), ("class_name", 1)],
                 {"background": True, "name": "annotation_dataset_class"}),
                ("annotations", [("image_id", 1), ("class_name", 1)],
                 {"background": True, "name": "annotation_image_class"}),
                ("annotations", [("created_at", -1)], {"background": True, "name": "annotation_created_at"}),

                # Upload session indexes
                ("upload_sessions", [("user_id", 1), ("status", 1)],
                 {"background": True, "name": "upload_session_user_status"}),
                ("upload_sessions", [("created_at", -1)], {"background": True, "name": "upload_session_created"}),

                # Dataset statistics indexes
                ("dataset_statistics", [("dataset_id", 1)],
                 {"unique": True, "background": True, "name": "dataset_stats_dataset_unique"}),
                ("dataset_statistics", [("last_updated", -1)], {"background": True, "name": "dataset_stats_updated"}),

                # User indexes
                ("users", [("email", 1)], {"unique": True, "background": True, "name": "user_email_unique"}),
                ("users", [("username", 1)], {"unique": True, "background": True, "name": "user_username_unique"}),
            ]

            for collection_name, keys, options in indexes_to_create:
                try:
                    collection = getattr(self, collection_name)
                    collection.create_index(keys, **options)
                except Exception as e:
                    logger.warning(f"Index creation warning for {collection_name}: {e}")
                    # Continue with other indexes even if one fails

            logger.info("Successfully created performance indexes")

        except Exception as e:
            logger.error(f"Error creating indexes: {e}")

    def _start_background_tasks(self):
        """Start background tasks for performance monitoring."""
        # This would typically be handled by the main application
        pass

    # Async versions of common operations
    async def async_find_one(self, collection: AsyncIOMotorCollection, filter_dict: Dict[str, Any]) -> Optional[
        Dict[str, Any]]:
        """Async version of find_one with performance tracking."""
        start_time = time.time()
        try:
            self.query_stats['total_queries'] += 1
            result = await collection.find_one(filter_dict)
            query_time = time.time() - start_time

            if query_time > 1.0:  # Log slow queries
                self.query_stats['slow_queries'] += 1
                logger.warning(f"Slow query detected: {query_time:.2f}s for {filter_dict}")

            return result
        except Exception as e:
            logger.error(f"Async find_one error: {e}")
            raise

    async def async_find_many(self, collection: AsyncIOMotorCollection, filter_dict: Dict[str, Any],
                              skip: int = 0, limit: int = 100, sort: Optional[List[tuple]] = None) -> List[
        Dict[str, Any]]:
        """Async version of find with performance tracking."""
        start_time = time.time()
        try:
            self.query_stats['total_queries'] += 1

            cursor = collection.find(filter_dict)

            if sort:
                cursor = cursor.sort(sort)

            cursor = cursor.skip(skip).limit(limit)
            results = await cursor.to_list(length=limit)

            query_time = time.time() - start_time

            if query_time > 2.0:  # Log slow queries
                self.query_stats['slow_queries'] += 1
                logger.warning(f"Slow find_many query: {query_time:.2f}s for {filter_dict}")

            return results
        except Exception as e:
            logger.error(f"Async find_many error: {e}")
            raise

    async def async_count_documents(self, collection: AsyncIOMotorCollection, filter_dict: Dict[str, Any]) -> int:
        """Async version of count_documents with performance tracking."""
        start_time = time.time()
        try:
            result = await collection.count_documents(filter_dict)
            query_time = time.time() - start_time

            if query_time > 2.0:
                self.query_stats['slow_queries'] += 1
                logger.warning(f"Slow count query: {query_time:.2f}s for {filter_dict}")

            return result
        except Exception as e:
            logger.error(f"Async count_documents error: {e}")
            raise

    async def bulk_insert_optimized(self, collection: AsyncIOMotorCollection,
                                    documents: List[Dict[str, Any]],
                                    batch_size: int = 1000,
                                    ordered: bool = False) -> Dict[str, Any]:
        """
        Optimized bulk insert with batching and error handling.

        Args:
            collection: AsyncIOMotorCollection to insert into
            documents: List of documents to insert
            batch_size: Size of each batch (default 1000)
            ordered: Whether to stop on first error

        Returns:
            Dict with insert statistics
        """
        if not documents:
            return {'inserted': 0, 'errors': 0, 'batches': 0}

        start_time = time.time()
        self.query_stats['bulk_operations'] += 1

        total_inserted = 0
        total_errors = 0
        batches_processed = 0

        # Process documents in batches for better performance
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            batches_processed += 1

            try:
                # Use ordered=False for better performance when not requiring strict ordering
                result = await collection.insert_many(batch, ordered=ordered)
                batch_inserted = len(result.inserted_ids)
                total_inserted += batch_inserted

                if batches_processed % 10 == 0:
                    logger.info(f"Processed batch {batches_processed}, inserted {batch_inserted} documents")

            except BulkWriteError as e:
                # Count successful inserts in this batch
                batch_inserted = e.details.get('nInserted', 0)
                batch_errors = len(e.details.get('writeErrors', []))
                total_inserted += batch_inserted
                total_errors += batch_errors

                logger.warning(f"Batch {batches_processed} had {batch_errors} errors, inserted {batch_inserted}")

            except Exception as e:
                total_errors += len(batch)
                logger.error(f"Batch {batches_processed} completely failed: {e}")

        total_time = time.time() - start_time
        logger.info(f"Bulk insert completed: {total_inserted} inserted, {total_errors} errors in {total_time:.2f}s")

        return {
            'inserted': total_inserted,
            'errors': total_errors,
            'batches': batches_processed,
            'time': total_time,
            'documents_per_second': total_inserted / total_time if total_time > 0 else 0
        }

    # Transaction support
    @asynccontextmanager
    async def start_transaction(self, max_retry: int = 3):
        """
        Context manager for MongoDB transactions with retry logic.

        Args:
            max_retry: Maximum number of retry attempts
        """
        session = None
        for attempt in range(max_retry):
            try:
                session = await self.async_client.start_session()
                async with session.start_transaction():
                    yield session
                break  # Transaction committed successfully

            except Exception as e:
                if attempt == max_retry - 1:
                    logger.error(f"Transaction failed after {max_retry} attempts: {e}")
                    raise
                else:
                    logger.warning(f"Transaction attempt {attempt + 1} failed, retrying: {e}")
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff

            finally:
                if session:
                    await session.end_session()

    async def transaction_with_retry(self, operations: list, max_retry: int = 3) -> Dict[str, Any]:
        """
        Execute multiple operations in a single transaction with retry logic.

        Args:
            operations: List of callable operations to execute
            max_retry: Maximum retry attempts

        Returns:
            Dict with operation results
        """
        for attempt in range(max_retry):
            session = await self.async_client.start_session()
            try:
                async with session.start_transaction() as s:
                    results = []
                    for operation in operations:
                        if asyncio.iscoroutinefunction(operation):
                            result = await operation(session)
                        else:
                            result = operation(session)
                        results.append(result)

                    await session.commit_transaction()
                    logger.info(f"Transaction committed successfully on attempt {attempt + 1}")
                    return {'success': True, 'results': results}

            except Exception as e:
                if attempt == max_retry - 1:
                    logger.error(f"Transaction failed after {max_retry} attempts: {e}")
                    return {'success': False, 'error': str(e)}
                else:
                    logger.warning(f"Transaction attempt {attempt + 1} failed, retrying: {e}")
                    await asyncio.sleep(2 ** attempt)

            finally:
                await session.end_session()

        return {'success': False, 'error': 'Max retries exceeded'}

    # Performance monitoring
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        avg_query_time = self.query_stats.get('average_query_time', 0.0)
        slow_query_rate = (self.query_stats.get('slow_queries', 0) /
                           max(self.query_stats.get('total_queries', 1), 1))

        return {
            **self.query_stats,
            'slow_query_rate': slow_query_rate,
            'connected_clients': {
                'sync': self.sync_client._topology._servers,
                'async': self.async_client._topology._servers
            }
        }

    def close(self):
        """Close both sync and async database connections."""
        if hasattr(self, 'sync_client') and self.sync_client:
            self.sync_client.close()
            logger.info("MongoDB sync connection closed")

        if hasattr(self, 'async_client') and self.async_client:
            self.async_client.close()
            logger.info("MongoDB async connection closed")

    def convert_objectids_to_str(self, doc: Dict[str, Any]) -> None:
        """
        Recursively convert ObjectId fields to strings in a document.

        Args:
            doc: Document dictionary to process (modified in place)
        """
        if not isinstance(doc, dict):
            return

        for key, value in list(doc.items()):
            if isinstance(value, ObjectId):
                doc[key] = str(value)
            elif isinstance(value, dict):
                self.convert_objectids_to_str(value)
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    if isinstance(item, ObjectId):
                        value[i] = str(item)
                    elif isinstance(item, dict):
                        self.convert_objectids_to_str(item)


# Global Database service instance
db_service = DatabaseService()
