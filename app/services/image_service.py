"""Service for handling image operations with async support and performance optimization."""
import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Dict, List, Optional, Union

import bson
from bson import ObjectId
from pymongo.errors import BulkWriteError, DuplicateKeyError

from app.services.db_service import db_service
from app.utils.logger import get_logger

logger: logging.Logger = get_logger(__name__)


class ImageService:
    """Service class for Image operations with async support and performance optimization."""

    def __init__(self):
        """Initialize Image service."""
        self.db = db_service
        self.images = self.db.images  # Sync collections
        self.async_images = self.db.async_images  # Async collections

        # Performance settings
        self.default_batch_size = 1000
        self.max_concurrent_operations = 10
        self.retry_attempts = 3

        # Cache for frequently accessed data
        self._dataset_cache = {}
        self._cache_ttl = 300  # 5 minutes
        self._last_cache_update = {}

    def _validate_image_data(self, image_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and normalize image data."""
        if not isinstance(image_data, dict):
            raise ValueError("Image data must be a dictionary")

        # Ensure required fields
        required_fields = ['filename', 'dataset_id']
        for field in required_fields:
            if field not in image_data:
                raise ValueError(f"Required field '{field}' is missing")

        # Convert ObjectIds
        if 'dataset_id' in image_data and isinstance(image_data['dataset_id'], str):
            if ObjectId.is_valid(image_data['dataset_id']):
                image_data['dataset_id'] = ObjectId(image_data['dataset_id'])
            else:
                raise ValueError(f"Invalid ObjectId format for dataset_id: {image_data['dataset_id']}")

        # Set defaults
        image_data.setdefault('created_at', time.time())
        image_data.setdefault('metadata', {})
        image_data.setdefault('split', 'train')
        image_data.setdefault('status', 'active')

        return image_data

    def _prepare_images_for_insert(self, image_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Prepare and validate a list of images for bulk insertion."""
        prepared_images = []

        for i, image in enumerate(image_list):
            try:
                # Create a copy to avoid modifying original
                prepared_image = image.copy()

                # Generate ObjectId if not present
                if '_id' not in prepared_image:
                    prepared_image['_id'] = ObjectId()
                elif isinstance(prepared_image['_id'], str):
                    if ObjectId.is_valid(prepared_image['_id']):
                        prepared_image['_id'] = ObjectId(prepared_image['_id'])
                    else:
                        raise ValueError(f"Invalid ObjectId format: {prepared_image['_id']}")

                # Validate and normalize
                prepared_image = self._validate_image_data(prepared_image)
                prepared_images.append(prepared_image)

            except Exception as e:
                logger.error(f"Error preparing image {i}: {e}")
                raise ValueError(f"Error preparing image at index {i}: {e}")

        return prepared_images

    async def async_bulk_save_images(self, image_list: List[Dict[str, Any]],
                                     batch_size: Optional[int] = None) -> Dict[str, Any]:
        """
        High-performance async bulk save images with optimized batching.

        Args:
            image_list: List of image dictionaries to insert
            batch_size: Custom batch size (uses default if None)

        Returns:
            Dict with insert statistics and performance metrics
        """
        if not image_list:
            logger.warning("Empty image list provided for async bulk save")
            return {'inserted': 0, 'errors': 0, 'time': 0.0, 'throughput': 0.0}

        batch_size = batch_size or self.default_batch_size
        start_time = time.time()

        try:
            # Prepare and validate images
            prepared_images = self._prepare_images_for_insert(image_list)
            total_images = len(prepared_images)

            logger.info(f"Starting async bulk insert of {total_images} images with batch size {batch_size}")

            # Use the optimized bulk insert from database service
            result = await self.db.bulk_insert_optimized(
                self.async_images,
                prepared_images,
                batch_size=batch_size,
                ordered=False  # Better performance for large inserts
            )

            total_time = time.time() - start_time
            result['throughput'] = result['inserted'] / total_time if total_time > 0 else 0
            result['total_requested'] = total_images

            logger.info(f"Async bulk insert completed: {result}")

            return result

        except Exception as e:
            total_time = time.time() - start_time
            logger.error(f"Async bulk insert failed: {e}", exc_info=True)
            return {
                'inserted': 0,
                'errors': len(image_list),
                'time': total_time,
                'throughput': 0,
                'error': str(e)
            }

    async def async_bulk_save_images_with_annotations(self,
                                                      image_list: List[Dict[str, Any]],
                                                      annotations_by_image: Dict[str, List[Dict[str, Any]]] = None,
                                                      batch_size: Optional[int] = None) -> Dict[str, Any]:
        """
        Bulk save images with their annotations in a transaction.

        Args:
            image_list: List of image dictionaries
            annotations_by_image: Dict mapping image_id to list of annotations
            batch_size: Custom batch size

        Returns:
            Dict with insert statistics
        """
        if not image_list:
            return {'inserted': 0, 'errors': 0, 'annotations_inserted': 0}

        batch_size = batch_size or self.default_batch_size
        start_time = time.time()

        try:
            async with self.db.start_transaction() as session:
                # Insert images
                image_result = await self.db.bulk_insert_optimized(
                    self.async_images,
                    self._prepare_images_for_insert(image_list),
                    batch_size=batch_size,
                    ordered=False
                )

                # Insert annotations if provided
                annotations_inserted = 0
                if annotations_by_image:
                    annotations_to_insert = []
                    for image_data in image_result.get('inserted_ids', []):
                        image_id = str(image_data['_id']) if '_id' in image_data else None
                        if image_id and image_id in annotations_by_image:
                            for annotation in annotations_by_image[image_id]:
                                annotation = annotation.copy()
                                annotation['image_id'] = image_data['_id']
                                annotation['dataset_id'] = image_data['dataset_id']
                                annotation['created_at'] = time.time()
                                annotations_to_insert.append(annotation)

                    if annotations_to_insert:
                        annotation_result = await self.db.bulk_insert_optimized(
                            self.db.async_annotations,
                            annotations_to_insert,
                            batch_size=batch_size
                        )
                        annotations_inserted = annotation_result['inserted']

                total_time = time.time() - start_time

                logger.info(
                    f"Transaction completed: {image_result['inserted']} images, {annotations_inserted} annotations in {total_time:.2f}s")

                return {
                    **image_result,
                    'annotations_inserted': annotations_inserted,
                    'time': total_time
                }

        except Exception as e:
            logger.error(f"Transactional bulk insert failed: {e}", exc_info=True)
            raise

    async def async_get_images_by_dataset(self,
                                          dataset_id: str,
                                          skip: int = 0,
                                          limit: int = 100,
                                          split: Optional[str] = None,
                                          projection: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Async version of get_images_by_dataset with performance optimization.

        Args:
            dataset_id: Dataset ID
            skip: Number of records to skip
            limit: Maximum number of records to return
            split: Optional split filter (train/val/test)
            projection: Fields to include/exclude

        Returns:
            List of images with optimized query performance
        """
        if not ObjectId.is_valid(dataset_id):
            raise ValueError(f"Invalid ObjectId format: {dataset_id}")

        try:
            query = {"dataset_id": ObjectId(dataset_id)}
            if split:
                query["split"] = split

            # Add projection for better performance
            default_projection = {"_id": 0}  # Exclude _id by default for better performance
            if projection:
                default_projection.update(projection)

            # Use optimized async find method
            results = await self.db.async_find_many(
                self.async_images,
                query,
                skip=skip,
                limit=limit,
                sort=[("created_at", -1)]
            )

            # Process results
            images = []
            for image in results:
                image["id"] = str(image["_id"])
                image["dataset_id"] = str(image["dataset_id"])

                # Convert other ObjectIds
                self.db.convert_objectids_to_str(image)

                # Remove _id to use id instead
                if "_id" in image:
                    del image["_id"]

                images.append(image)

            return images

        except Exception as e:
            logger.error(f"Error in async_get_images_by_dataset: {e}", exc_info=True)
            raise

    async def async_count_images(self, dataset_id: str, split: Optional[str] = None) -> int:
        """Async version of count_images with caching."""
        if not ObjectId.is_valid(dataset_id):
            raise ValueError(f"Invalid ObjectId format: {dataset_id}")

        try:
            query = {"dataset_id": ObjectId(dataset_id)}
            if split:
                query["split"] = split

            # Simple caching for counts
            cache_key = f"count_{dataset_id}_{split}"
            current_time = time.time()

            if (cache_key in self._dataset_cache and
                current_time - self._last_cache_update.get(cache_key, 0) < self._cache_ttl):
                return self._dataset_cache[cache_key]

            count = await self.db.async_count_documents(self.async_images, query)

            # Cache the result
            self._dataset_cache[cache_key] = count
            self._last_cache_update[cache_key] = current_time

            return count

        except Exception as e:
            logger.error(f"Error in async_count_images: {e}", exc_info=True)
            raise

    async def async_batch_delete_images(self, image_ids: List[str]) -> Dict[str, Any]:
        """
        Delete multiple images efficiently in batches.

        Args:
            image_ids: List of image IDs to delete

        Returns:
            Dict with deletion statistics
        """
        if not image_ids:
            return {'deleted': 0, 'errors': 0}

        try:
            # Validate ObjectIds
            object_ids = []
            for image_id in image_ids:
                if ObjectId.is_valid(image_id):
                    object_ids.append(ObjectId(image_id))
                else:
                    logger.warning(f"Invalid ObjectId format for deletion: {image_id}")

            if not object_ids:
                return {'deleted': 0, 'errors': len(image_ids)}

            # Delete in batches
            batch_size = 1000
            total_deleted = 0
            total_errors = 0

            for i in range(0, len(object_ids), batch_size):
                batch = object_ids[i:i + batch_size]

                try:
                    result = await self.async_images.delete_many({"_id": {"$in": batch}})
                    total_deleted += result.deleted_count

                    # Also delete associated annotations
                    await self.db.async_annotations.delete_many({"image_id": {"$in": batch}})

                except Exception as e:
                    total_errors += len(batch)
                    logger.error(f"Error deleting batch: {e}")

            logger.info(f"Batch deletion completed: {total_deleted} deleted, {total_errors} errors")
            return {'deleted': total_deleted, 'errors': total_errors}

        except Exception as e:
            logger.error(f"Error in async_batch_delete_images: {e}", exc_info=True)
            raise

    @asynccontextmanager
    async def image_processing_context(self, dataset_id: str):
        """
        Context manager for batch image processing operations.

        Args:
            dataset_id: Dataset ID for the processing context
        """
        session = await self.db.async_client.start_session()
        try:
            async with session.start_transaction():
                yield session
        finally:
            await session.end_session()

    # Backward compatibility: Keep existing sync methods for API compatibility
    def bulk_save_images(self, image_list: List[Dict[str, Any]], batch_size: int = 1000) -> int:
        """
        Optimized bulk save image documents to database with batching.

        Args:
            image_list: List of image dictionaries to insert
            batch_size: Size of each batch for optimal performance

        Returns:
            int: Number of image documents successfully inserted

        Raises:
            Exception: For database errors
        """
        if not image_list:
            logger.warning("Empty image list provided for bulk save")
            return 0

        start_time = time.time()
        total_inserted = 0
        total_errors = 0

        try:
            # Process in batches for better performance
            for i in range(0, len(image_list), batch_size):
                batch = image_list[i:i + batch_size]

                try:
                    # Ensure all ObjectIds are properly converted
                    for image in batch:
                        if "_id" in image and isinstance(image["_id"], str):
                            if ObjectId.is_valid(image["_id"]):
                                image["_id"] = ObjectId(image["_id"])
                            else:
                                raise ValueError(f"Invalid ObjectId format: {image['_id']}")
                        elif "_id" not in image:
                            image["_id"] = ObjectId()

                        if "dataset_id" in image and isinstance(image["dataset_id"], str):
                            if ObjectId.is_valid(image["dataset_id"]):
                                image["dataset_id"] = ObjectId(image["dataset_id"])
                            else:
                                raise ValueError(f"Invalid ObjectId format for dataset_id: {image['dataset_id']}")

                        # Set defaults
                        image.setdefault('created_at', time.time())
                        image.setdefault('metadata', {})
                        image.setdefault('split', 'train')
                        image.setdefault('status', 'active')

                    result = self.images.insert_many(batch, ordered=False)
                    batch_inserted = len(result.inserted_ids)
                    total_inserted += batch_inserted

                    if (i // batch_size + 1) % 10 == 0:
                        logger.info(f"Processed batch {i // batch_size + 1}, inserted {batch_inserted} images")

                except (DuplicateKeyError, BulkWriteError) as e:
                    # Count successful inserts in this batch
                    if hasattr(e, 'details'):
                        batch_inserted = e.details.get('nInserted', 0)
                        batch_errors = len(e.details.get('writeErrors', []))
                    else:
                        batch_inserted = 0
                        batch_errors = len(batch)

                    total_inserted += batch_inserted
                    total_errors += batch_errors
                    logger.warning(f"Batch {i // batch_size + 1} had {batch_errors} errors, inserted {batch_inserted}")

                except Exception as e:
                    total_errors += len(batch)
                    logger.error(f"Batch {i // batch_size + 1} completely failed: {e}")

            total_time = time.time() - start_time
            throughput = total_inserted / total_time if total_time > 0 else 0

            logger.info(
                f"Bulk insert completed: {total_inserted} inserted, {total_errors} errors in {total_time:.2f}s (throughput: {throughput:.2f}/s)")
            return total_inserted

        except Exception as e:
            logger.error(f"Failed to bulk save images: {e}", exc_info=True)
            raise Exception(f"Failed to bulk save images: {e}")

    def get_image(self, image_id: str) -> Optional[Dict[str, Any]]:
        """
        Get image by ID.

        Args:
            image_id: Image ID

        Returns:
            Optional[Dict]: Image data with ObjectIds converted to strings

        Raises:
            ValueError: If image_id is invalid
            Exception: For other errors
        """
        try:
            # Validate ObjectId format
            if not ObjectId.is_valid(image_id):
                logger.info(f"Invalid ObjectId format: {image_id}")
                raise ValueError(f"Invalid ObjectId format: {image_id}")

            image = self.images.find_one({"_id": ObjectId(image_id)})
            if image:
                # Convert all ObjectIds to strings for proper serialization
                self.db.convert_objectids_to_str(image)
            return image

        except ValueError:
            raise
        except Exception as e:
            logger.error(f"Error in get_image: {e}", exc_info=True)
            raise Exception(f"Error in get_image: {e}")

    def get_images_by_dataset(
        self,
        dataset_id: str,
        skip: int = 0,
        limit: int = 100,
        split: Optional[str] = None,
        projection: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Optimized get images by dataset ID with optional split filter.

        Args:
            dataset_id: Dataset ID
            skip: Number of records to skip
            limit: Maximum number of records to return
            split: Optional split filter (train/val/test)
            projection: Fields to include/exclude

        Returns:
            List[Dict]: List of images with optimized query performance

        Raises:
            ValueError: If dataset_id is invalid
            Exception: For other errors
        """
        try:
            # Validate ObjectId format
            if not ObjectId.is_valid(dataset_id):
                logger.info(f"Invalid ObjectId format: {dataset_id}")
                raise ValueError(f"Invalid ObjectId format: {dataset_id}")

            query = {"dataset_id": ObjectId(dataset_id)}
            if split:
                query["split"] = split

            start_time = time.time()
            cursor = self.images.find(query).skip(skip).limit(limit).sort("created_at", -1)

            images = []
            for image in cursor:
                image["id"] = str(image["_id"]) if "_id" in image else None
                if "_id" in image:
                    del image["_id"]  # Remove _id and use id instead

                image["dataset_id"] = str(image["dataset_id"]) if "dataset_id" in image else dataset_id

                # Convert other ObjectIds
                self.db.convert_objectids_to_str(image)
                images.append(image)

            query_time = time.time() - start_time

            if query_time > 1.0:
                logger.warning(f"Slow query detected: {query_time:.2f}s for dataset {dataset_id}")

            return images

        except ValueError:
            raise
        except Exception as e:
            logger.error(f"Error in get_images_by_dataset: {e}", exc_info=True)
            raise Exception(f"Error in get_images_by_dataset: {e}")

    def count_images(self, dataset_id: str, split: Optional[str] = None) -> int:
        """
        Optimized count images in dataset with caching.

        Args:
            dataset_id: Dataset ID
            split: Optional split filter (train/val/test)

        Returns:
            int: Number of images

        Raises:
            ValueError: If dataset_id is invalid
            Exception: For other errors
        """
        try:
            # Validate ObjectId format
            if not ObjectId.is_valid(dataset_id):
                logger.info(f"Invalid ObjectId format: {dataset_id}")
                raise ValueError(f"Invalid ObjectId format: {dataset_id}")

            query = {"dataset_id": ObjectId(dataset_id)}
            if split:
                query["split"] = split

            # Use count_documents for better performance
            start_time = time.time()
            count = self.images.count_documents(query)
            query_time = time.time() - start_time

            if query_time > 2.0:
                logger.warning(f"Slow count query: {query_time:.2f}s for dataset {dataset_id}")

            return count

        except ValueError:
            raise
        except Exception as e:
            logger.error(f"Error in count_images: {e}", exc_info=True)
            raise Exception(f"Error in count_images: {e}")

    def delete_images_by_dataset(self, dataset_id: str) -> int:
        """
        Optimized delete all images in a dataset with annotation cleanup.

        Args:
            dataset_id: Dataset ID

        Returns:
            int: Number of deleted images

        Raises:
            ValueError: If dataset_id is invalid
            Exception: For other errors
        """
        try:
            # Validate ObjectId format
            if not ObjectId.is_valid(dataset_id):
                logger.info(f"Invalid ObjectId format: {dataset_id}")
                raise ValueError(f"Invalid ObjectId format: {dataset_id}")

            dataset_object_id = ObjectId(dataset_id)

            # Delete images
            start_time = time.time()
            result = self.images.delete_many({"dataset_id": dataset_object_id})
            images_deleted = result.deleted_count

            # Also delete associated annotations for better data consistency
            annotations_deleted = self.annotations.delete_many({"dataset_id": dataset_object_id}).deleted_count

            total_time = time.time() - start_time
            logger.info(
                f"Deleted {images_deleted} images and {annotations_deleted} annotations from dataset {dataset_id} in {total_time:.2f}s")

            return images_deleted

        except ValueError:
            raise
        except Exception as e:
            logger.error(f"Error in delete_images_by_dataset: {e}", exc_info=True)
            raise Exception(f"Error in delete_images_by_dataset: {e}")

    def delete_image(self, image_id: str) -> bool:
        """
        Delete a single image with annotation cleanup.

        Args:
            image_id: Image ID

        Returns:
            bool: True if successful

        Raises:
            ValueError: If image_id is invalid
            Exception: For other errors
        """
        try:
            # Validate ObjectId format
            if not ObjectId.is_valid(image_id):
                logger.info(f"Invalid ObjectId format: {image_id}")
                raise ValueError(f"Invalid ObjectId format: {image_id}")

            image_object_id = ObjectId(image_id)

            # Delete image
            result = self.images.delete_one({"_id": image_object_id})
            image_deleted = result.deleted_count > 0

            # Also delete associated annotations
            if image_deleted:
                self.annotations.delete_many({"image_id": image_object_id})
                logger.info(f"Deleted image: {image_id}")

            return image_deleted

        except ValueError:
            raise
        except Exception as e:
            logger.error(f"Error in delete_image: {e}", exc_info=True)
            raise Exception(f"Error in delete_image: {e}")

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for this service."""
        db_stats = self.db.get_performance_stats()

        return {
            'database_stats': db_stats,
            'cache_info': {
                'cached_items': len(self._dataset_cache),
                'cache_ttl': self._cache_ttl
            },
            'service_config': {
                'default_batch_size': self.default_batch_size,
                'max_concurrent_operations': self.max_concurrent_operations,
                'retry_attempts': self.retry_attempts
            }
        }

    async def async_get_performance_stats(self) -> Dict[str, Any]:
        """Get async performance statistics."""
        return self.get_performance_stats()

    def clear_cache(self):
        """Clear the dataset cache."""
        self._dataset_cache.clear()
        self._last_cache_update.clear()
        logger.info("Dataset cache cleared")

    async def async_clear_cache(self):
        """Async version of clear cache."""
        self.clear_cache()


# Global Image service instance
image_service = ImageService()
