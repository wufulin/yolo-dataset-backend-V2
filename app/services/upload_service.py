"""Service for handling dataset upload operations with optimized performance."""

import asyncio
import hashlib
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from bson import ObjectId
from fastapi import HTTPException, status

from app.core.decorators import async_exception_handler
from app.core.exceptions import (
    BusinessLogicException,
    DatabaseException,
    ErrorCategory,
    SystemException,
    ValidationException,
)
from app.core.patterns import DatasetType
from app.models.dataset import Dataset
from app.services import dataset_service, image_service, minio_service
from app.utils import yolo_validator
from app.utils.file_utils import safe_remove
from app.utils.logger import get_logger

logger: logging.Logger = get_logger(__name__)


def dataset_upload_monitor(operation_name: Optional[str] = None):
    """Enhanced performance monitor specifically for dataset upload operations."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            op_name = operation_name or f"dataset_upload.{func.__name__}"
            start_time = time.time()
            
            try:
                result = await func(*args, **kwargs)
                execution_time = time.time() - start_time
                
                # Log performance metrics
                logger.info(
                    f"{op_name} completed successfully",
                    extra={
                        "operation": op_name,
                        "duration_ms": round(execution_time * 1000, 2),
                        "status": "success",
                        "timestamp": datetime.utcnow().isoformat()
                    }
                )
                return result
                
            except Exception as e:
                execution_time = time.time() - start_time
                logger.error(
                    f"{op_name} failed after {execution_time:.2f}s: {str(e)}",
                    extra={
                        "operation": op_name,
                        "duration_ms": round(execution_time * 1000, 2),
                        "status": "error",
                        "error_type": type(e).__name__,
                        "timestamp": datetime.utcnow().isoformat()
                    },
                    exc_info=True
                )
                raise
                
        return wrapper
    return decorator


class OptimizedUploadService:
    """
    High-performance dataset upload service with optimized processing for large datasets.
    
    Features:
    - Batch parallel processing
    - Memory optimization
    - Progress tracking
    - Automatic parameter adjustment
    - Resource cleanup
    """

    def __init__(self, max_workers: Optional[int] = None, batch_size: Optional[int] = None):
        """
        Initialize upload service.
        
        Args:
            max_workers: Maximum worker threads (default: CPU cores * 2)
            batch_size: Batch processing size (default: auto-adjusted based on dataset size)
        """
        cpu_count = os.cpu_count() or 4
        self.max_workers = max_workers or min(cpu_count * 2, 20)  # Cap at 20 workers
        self.batch_size = batch_size or 1000
        
        # Auto-adjust based on system resources
        self.max_workers = min(self.max_workers, 20)
        
        logger.info(
            f"Initialized OptimizedUploadService: workers={self.max_workers}, batch_size={self.batch_size}"
        )

    def _get_image_info_sync(self, image_path: Path) -> Tuple[int, int, str]:
        """Synchronous helper function to get image information."""
        try:
            from PIL import Image
            with Image.open(image_path) as img:
                width, height = img.size
                img_format = img.format.lower() if img.format else "jpg"
            return width, height, img_format
        except Exception as e:
            logger.warning(f"Failed to get image info for {image_path}: {e}")
            return 640, 480, "jpg"  # Default values

    def _calculate_file_hash_sync(self, file_path: Path) -> str:
        """Synchronous helper function to calculate file hash."""
        try:
            hasher = hashlib.md5()
            with open(file_path, 'rb') as f:
                # Chunked reading to avoid memory overflow
                for chunk in iter(lambda: f.read(8192), b""):
                    hasher.update(chunk)
            return hasher.hexdigest()
        except Exception as e:
            logger.warning(f"Failed to calculate hash for {file_path}: {e}")
            return ""

    def _scan_images_efficiently(self, images_dir: Path) -> Iterator[Path]:
        """
        Efficiently scan images directory using generator to avoid memory overflow.
        
        Args:
            images_dir: Directory containing images
            
        Yields:
            Path: Image file paths one by one
        """
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
        
        try:
            # Use os.scandir for better performance with large directories
            with os.scandir(images_dir) as entries:
                for entry in entries:
                    # Skip directories and only process files
                    if entry.is_file():
                        # Check extension without creating Path object for every file
                        ext = os.path.splitext(entry.name)[1].lower()
                        if ext in image_extensions:
                            yield Path(entry.path)
                            
        except Exception as e:
            logger.error(f"Error scanning directory {images_dir}: {e}")
            return

    def _estimate_total_files(self, images_dir: Path) -> int:
        """
        Efficiently estimate total file count without loading all files into memory.
        
        Args:
            images_dir: Directory containing images
            
        Returns:
            int: Estimated number of image files
        """
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
        count = 0
        
        try:
            with os.scandir(images_dir) as entries:
                for entry in entries:
                    if entry.is_file():
                        ext = os.path.splitext(entry.name)[1].lower()
                        if ext in image_extensions:
                            count += 1
        except Exception as e:
            logger.warning(f"Could not estimate file count: {e}")
            
        return count

    @async_exception_handler(
        exception_type=Exception,
        default_message="Dataset processing failed",
        log_error=True,
        reraise=True
    )
    @dataset_upload_monitor("dataset.upload")
    async def process_dataset(self, zip_path: str, dataset_info: Any) -> Dict[str, Any]:
        """
        Process uploaded dataset ZIP file.
        
        Args:
            zip_path: Path to ZIP file
            dataset_info: Dataset metadata
            
        Returns:
            Dict[str, Any]: Processing results
        """
        if not dataset_info:
            raise ValidationException("Dataset info is required", details={"field": "dataset_info"})

        dataset_type = getattr(dataset_info, "dataset_type", "detect")
        
        # Validate dataset type
        valid_types = [t.value for t in DatasetType]
        if dataset_type not in valid_types:
            raise ValidationException(
                f"Invalid dataset type: {dataset_type}",
                details={"valid_types": valid_types, "provided": dataset_type}
            )

        validation_result = await yolo_validator.validate_dataset(zip_path, dataset_type)

        if not validation_result.is_valid:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid YOLO dataset: {validation_result.message}",
            )

        class_names = validation_result.dataset_info.get("class_names", [])
        dataset_root = validation_result.dataset_info.get("dataset_root", None)
        

        # Create dataset record
        dataset = Dataset(
            name=getattr(dataset_info, "name", ""),
            description=getattr(dataset_info, "description", ""),
            dataset_type=dataset_type,
            class_names=class_names,
            num_images=0,
            num_annotations=0,
            splits={"train": 0, "val": 0, "test": 0},
            status="processing",
            error_message=None,
            file_size=0,
            storage_path=None,
            created_by="admin",
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            version=1,
        )

        dataset_id = await dataset_service.create_dataset(dataset)

        # Ensure dataset root directory exists
        if not dataset_root:
            raise SystemException(
                "Failed to extract dataset root directory",
                error_code="DATASET_ROOT_EXTRACTION_FAILED",
                category=ErrorCategory.FILE_SYSTEM
            )

        # Start background processing task
        asyncio.create_task(
            self._background_process_images_and_annotations(
                dataset_root=Path(dataset_root),
                dataset_id=dataset_id,
                dataset_type=dataset_type,
                class_names=class_names,
                zip_path=zip_path,
            )
        )

        return {
            "status": "success",
            "dataset_id": dataset_id,
            "message": "Dataset created successfully. Processing images in background.",
            "dataset_type": dataset_type,
            "class_count": len(class_names)
        }

    @async_exception_handler(
        exception_type=Exception,
        default_message="Background processing failed",
        log_error=True,
        reraise=False
    )
    @dataset_upload_monitor("dataset.background_processing")
    async def _background_process_images_and_annotations(
        self,
        dataset_root: Path,
        dataset_id: str,
        dataset_type: str,
        class_names: List[str],
        zip_path: str,
    ) -> None:
        """
        Background worker for processing images and annotations.
        
        Args:
            dataset_root: Dataset root directory
            dataset_id: Dataset ID
            dataset_type: Dataset type
            class_names: List of class names
            zip_path: ZIP file path (for cleanup)
        """
        try:
            logger.info(f"Starting background processing for dataset {dataset_id}")

            # Process images and annotations
            processed_count = await self._process_images_and_annotations(
                dataset_root=dataset_root,
                dataset_id=dataset_id,
                dataset_type=dataset_type,
                class_names=class_names,
            )

            # Update dataset status to active
            await dataset_service.repository.update(
                dataset_id,
                {
                    "status": "active",
                    "updated_at": datetime.utcnow(),
                },
            )

            logger.info(
                f"Dataset {dataset_id} background processing completed. Processed {processed_count} images"
            )

        except Exception as exc:
            logger.error(
                f"Dataset {dataset_id} background processing failed: {exc}",
                exc_info=True
            )

            # Update dataset status to error
            try:
                await dataset_service.repository.update(
                    dataset_id,
                    {
                        "status": "error",
                        "error_message": str(exc),
                        "updated_at": datetime.utcnow(),
                    },
                )
            except Exception as update_error:
                logger.error(
                    f"Failed to update dataset status after error: {update_error}",
                    exc_info=True
                )
        finally:
            # Cleanup temporary files
            await self._cleanup_temp_files(dataset_root, zip_path)

    async def _cleanup_temp_files(self, dataset_root: Optional[Path], zip_path: Optional[str]) -> None:
        """Clean up temporary files."""
        try:
            if dataset_root and Path(dataset_root).exists():
                safe_remove(dataset_root)
            if zip_path and Path(zip_path).exists():
                safe_remove(zip_path)
            if zip_path and Path(zip_path).parent.exists():
                safe_remove(Path(zip_path).parent)

            logger.debug(f"Cleaned up temporary files: {dataset_root}, {zip_path}")
        except Exception as cleanup_error:
            logger.warning(f"Failed to clean up temporary files: {cleanup_error}", exc_info=True)

    @dataset_upload_monitor("dataset.process_images_annotations")
    async def _process_images_and_annotations(
        self,
        dataset_root: Path,
        dataset_id: str,
        dataset_type: str,
        class_names: List[str],
    ) -> int:
        """
        Process all images and annotations.
        
        Args:
            dataset_root: Dataset root directory
            dataset_id: Dataset ID
            dataset_type: Dataset type
            class_names: List of class names
            
        Returns:
            int: Number of processed images
        """
        splits = ["train", "val", "test"]
        train_images = train_annotations = train_size = 0
        val_images = val_annotations = val_size = 0
        test_images = test_annotations = test_size = 0
        total_images = 0

        for split_name in splits:
            # Update dataset status
            await dataset_service.repository.update(
                dataset_id,
                {
                    "status": "processing",
                    "message": f"Processing {split_name} images in background.",
                    "updated_at": datetime.utcnow(),
                },
            )

            split_images, split_annotations, split_size = await self.process_split_optimized(
                dataset_root=dataset_root,
                split_name=split_name,
                dataset_id=dataset_id,
                dataset_type=dataset_type,
                class_names=class_names,
            )

            if split_name == "train":
                train_images = split_images
                train_annotations = split_annotations
                train_size = split_size
            elif split_name == "val":
                val_images = split_images
                val_annotations = split_annotations
                val_size = split_size
            elif split_name == "test":
                test_images = split_images
                test_annotations = split_annotations
                test_size = split_size
            
            total_images += split_images

        # Update split statistics
        await dataset_service.update_dataset_stats(
            dataset_id=dataset_id,
            train_images=train_images,
            train_annotations=train_annotations,
            train_size=train_size,
            val_images=val_images,
            val_annotations=val_annotations,
            val_size=val_size,
            test_images=test_images,
            test_annotations=test_annotations,
            test_size=test_size,
        )

        return total_images

    async def _process_single_image_batch(
        self,
        image_batch: List[Path],
        labels_dir: Path,
        dataset_id: str,
        user_id: str,
        split_name: str,
        dataset_type: str,
        class_names: List[str]
    ) -> Tuple[List, List]:
        """
        Process a single batch of images.
        
        Returns:
            Tuple[List, List]: (upload_list, image_doc_list)
        """
        upload_list = []
        image_doc_list = []

        # Use thread pool for parallel file I/O processing
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_path = {}
            
            for image_path in image_batch:
                # Submit file info retrieval tasks
                future_info = executor.submit(self._get_image_info_sync, image_path)
                future_hash = executor.submit(self._calculate_file_hash_sync, image_path)
                future_to_path[future_info] = ('info', image_path)
                future_to_path[future_hash] = ('hash', image_path)
            
            # Collect results
            image_info_cache = {}
            file_hash_cache = {}
            
            for future in as_completed(future_to_path):
                task_type, image_path = future_to_path[future]
                try:
                    if task_type == 'info':
                        image_info_cache[image_path] = future.result()
                    else:  # hash
                        file_hash_cache[image_path] = future.result()
                except Exception as e:
                    logger.error(f"Error processing {image_path}: {e}")
                    # Use default values
                    image_info_cache[image_path] = (640, 480, "jpg")
                    file_hash_cache[image_path] = ""
        
        # Process annotation parsing (this may still be sequential but can be further optimized)
        for image_path in image_batch:
            try:
                width, height, img_format = image_info_cache[image_path]
                file_size = image_path.stat().st_size
                file_hash = file_hash_cache[image_path]
                
                # Corresponding label file
                label_path = labels_dir / f"{image_path.stem}.txt"
                
                # Parse annotations
                annotation_objects = await yolo_validator.parse_annotations(
                    str(label_path),
                    dataset_type,
                    class_names,
                )
                
                # Convert annotations to dictionary format
                image_id = ObjectId()
                annotations = []
                for ann_obj in annotation_objects:
                    ann_dict = asdict(ann_obj)
                    ann_dict["image_id"] = image_id
                    ann_dict["dataset_id"] = ObjectId(dataset_id)
                    annotations.append(ann_dict)
                
                # MinIO path
                minio_file_path = f"{user_id}/{dataset_id}/images/{split_name}/{image_path.name}"
                
                # Content type detection
                content_type = "image/jpeg"
                suffix = image_path.suffix.lower()
                if suffix == ".png":
                    content_type = "image/png"
                elif suffix in [".jpg", ".jpeg"]:
                    content_type = "image/jpeg"
                elif suffix == ".bmp":
                    content_type = "image/bmp"
                elif suffix in [".tiff", ".tif"]:
                    content_type = "image/tiff"
                
                # Add to upload list
                upload_list.append((str(image_path), minio_file_path, content_type))
                
                # Image document data
                image_doc_list.append({
                    "_id": image_id,
                    "dataset_id": ObjectId(dataset_id),
                    "filename": image_path.name,
                    "file_path": minio_file_path,
                    "file_size": file_size,
                    "file_hash": file_hash,
                    "width": width,
                    "height": height,
                    "channels": 3,
                    "format": img_format,
                    "split": split_name,
                    "annotations": annotations,
                    "metadata": {},
                    "is_annotated": len(annotations) > 0,
                    "annotation_count": len(annotations),
                    "created_at": datetime.utcnow(),
                    "updated_at": datetime.utcnow(),
                })
                
            except Exception as e:
                logger.error(f"Failed to prepare {image_path.name}: {e}", exc_info=True)
                continue

        return upload_list, image_doc_list

    async def _process_large_dataset_metadata(
        self,
        images_dir: Path,
        labels_dir: Path,
        dataset_id: str,
        user_id: str,
        split_name: str,
        dataset_type: str,
        class_names: List[str]
    ) -> Tuple[List, List, int]:
        """
        Optimized large dataset processing method using generators to avoid memory overflow.
        
        Args:
            images_dir: Images directory
            labels_dir: Labels directory
            dataset_id: Dataset ID
            user_id: User ID
            split_name: Split name
            dataset_type: Dataset type
            class_names: List of class names
            
        Returns:
            Tuple[List, List, int]: (upload_list, image_doc_list, total_file_size)
        """
        total_file_size = 0
        upload_list = []
        image_doc_list = []
        
        # Estimate total files for progress reporting
        total_files = self._estimate_total_files(images_dir)
        logger.info(f"  Starting processing of {total_files} files")
        logger.info(f"  Batch size: {self.batch_size}, Max workers: {self.max_workers}")
        
        # Process in batches using generator to avoid loading all files into memory
        processed_count = 0
        batch_count = 0
        
        # Create batch from generator
        def create_batch(generator: Iterator[Path], batch_size: int) -> List[Path]:
            batch = []
            try:
                for _ in range(batch_size):
                    batch.append(next(generator))
                return batch
            except StopIteration:
                return batch if batch else []
        
        # Get file generator
        image_generator = self._scan_images_efficiently(images_dir)
        
        while True:
            # Create current batch from generator
            image_batch = create_batch(image_generator, self.batch_size)
            
            if not image_batch:
                break  # No more files to process
                
            batch_count += 1
            batch_start_time = time.time()
            
            try:
                # Process current batch
                batch_upload_list, batch_image_doc_list = await self._process_single_image_batch(
                    image_batch, labels_dir, dataset_id, user_id, split_name, dataset_type, class_names
                )
                
                upload_list.extend(batch_upload_list)
                image_doc_list.extend(batch_image_doc_list)
                
                # Calculate current batch file size
                batch_size_sum = sum(img_path.stat().st_size for img_path in image_batch)
                total_file_size += batch_size_sum
                
                # Update processed count
                processed_count += len(image_batch)
                
                batch_time = time.time() - batch_start_time
                files_per_second = len(image_batch) / batch_time if batch_time > 0 else 0
                
                # Progress reporting every 10 batches or when finished
                if batch_count % 10 == 0 or processed_count >= total_files:
                    progress_pct = (processed_count / total_files) * 100 if total_files > 0 else 0
                    logger.info(f"    Batch {batch_count} completed in {batch_time:.2f}s ({files_per_second:.1f} files/sec)")
                    logger.info(f"    Progress: {processed_count}/{total_files} files ({progress_pct:.1f}%)")
                    logger.info(f"    Total size so far: {total_file_size / 1024 / 1024:.2f} MB")
                
            except Exception as e:
                logger.error(f"  Error processing batch {batch_count}: {e}", exc_info=True)
                continue
        
        logger.info(f"  Processing completed: {len(image_doc_list)} documents created from {processed_count} files")
        return upload_list, image_doc_list, total_file_size

    @dataset_upload_monitor("dataset.process_split_optimized")
    async def process_split_optimized(
        self,
        dataset_root: Path,
        split_name: str,
        dataset_id: str,
        dataset_type: str,
        class_names: List[str],
        user_id: str = "691c3f00ca496bc2f41f0993",
    ) -> Tuple[int, int, int]:
        """
        Optimized dataset split processing method.
        
        Args:
            dataset_root: Dataset root directory
            split_name: Split name
            dataset_id: Dataset ID
            dataset_type: Dataset type
            class_names: List of class names
            user_id: User ID
            
        Returns:
            Tuple[int, int, int]: (image_count, annotation_count, total_file_size)
        """
        dataset_root_path = Path(dataset_root)
        images_dir = dataset_root_path / "images" / split_name
        labels_dir = dataset_root_path / "labels" / split_name

        if not images_dir.exists():
            logger.warning("  Images directory not found: %s", images_dir)
            return 0, 0, 0

        if not labels_dir.exists():
            logger.warning("  Labels directory not found: %s", labels_dir)
            return 0, 0, 0

        # Get total file count for optimization decisions
        estimated_file_count = self._estimate_total_files(images_dir)

        logger.info(f"\nProcessing {split_name} split:")
        logger.info(f"  Estimated number of images: {estimated_file_count}")

        if estimated_file_count == 0:
            logger.warning(f"  No image files found in {images_dir}")
            return 0, 0, 0

        # Auto-adjust parameters based on file count
        original_max_workers = self.max_workers
        original_batch_size = self.batch_size
        
        if estimated_file_count > 100000:
            # Very large dataset - use more aggressive optimization
            self.max_workers = min(16, self.max_workers * 2)
            self.batch_size = min(3000, self.batch_size * 3)
            logger.info(f"  Very large dataset detected, adjusting: workers={self.max_workers}, batch_size={self.batch_size}")
        elif estimated_file_count > 50000:
            # Large dataset
            self.max_workers = min(12, self.max_workers * 2)
            self.batch_size = min(2000, self.batch_size * 2)
            logger.info(f"  Large dataset detected, adjusting: workers={self.max_workers}, batch_size={self.batch_size}")
        elif estimated_file_count < 1000:
            # Small dataset
            self.max_workers = max(2, self.max_workers // 2)
            logger.info(f"  Small dataset detected, adjusting: workers={self.max_workers}")

        try:
            # Phase 1: Data preparation using generators
            logger.info(f"  Phase 1: Preparing image data (generator-based)...")
            start_time = time.time()

            upload_list, image_doc_list, total_file_size = await self._process_large_dataset_metadata(
                images_dir, labels_dir, dataset_id, user_id, split_name, dataset_type, class_names
            )

            phase1_time = time.time() - start_time
            logger.info(f"  Phase 1 completed in {phase1_time:.2f} seconds")
            logger.info(f"  Upload list size: {len(upload_list)}")
            logger.info(f"  Image docs size: {len(image_doc_list)}")

            if not upload_list:
                logger.error("  No files prepared for upload")
                return 0, 0, 0

            # Phase 2: Batch upload to MinIO
            logger.info(f"  Phase 2: Uploading files to MinIO...")
            upload_start_time = time.time()
            
            try:
                upload_result = await minio_service.upload_files_parallel_async(
                    upload_list, 
                    bucket_name="yolo-datasets",  # Use default bucket
                )
            except Exception as e:
                logger.error(f"  Upload failed: {e}")
                raise BusinessLogicException(
                    f"MinIO upload failed: {str(e)}",
                    error_code="MINIO_UPLOAD_FAILED",
                    category=ErrorCategory.STORAGE
                )

            upload_time = time.time() - upload_start_time
            logger.info(f"  Phase 2 completed in {upload_time:.2f} seconds")

            # Phase 3: Batch insert to database
            logger.info(f"  Phase 3: Inserting documents to database...")
            db_start_time = time.time()
            
            try:
                # Get successfully uploaded file paths
                successful_paths = set()
                if "success_list" in upload_result:
                    successful_paths = set(upload_result["success_list"])
                elif "successful_files" in upload_result:
                    successful_paths = set(
                        result.get("object_name", "") for result in upload_result["successful_files"]
                        if isinstance(result, dict)
                    )
                
                # Filter successfully uploaded image documents
                images_to_insert = []
                for image in image_doc_list:
                    if image["file_path"] in successful_paths:
                        images_to_insert.append(image)
                
                if images_to_insert:
                    inserted_count = await image_service.async_bulk_save_images(images_to_insert)
                    logger.info(f"  Inserted {inserted_count} images to database")
                    
            except Exception as e:
                logger.error(f"  Database insertion failed: {e}")
                raise DatabaseException(
                    f"Failed to insert images: {str(e)}",
                    error_code="DATABASE_INSERTION_FAILED",
                    original_error=e
                )

            db_time = time.time() - db_start_time
            logger.info(f"  Phase 3 completed in {db_time:.2f} seconds")

            # Calculate statistics
            image_count = len(images_to_insert)
            annotation_count = sum(doc.get('annotation_count', 0) for doc in images_to_insert)
            
            # Log failed uploads
            failed_list = upload_result.get("failed_list", upload_result.get("failed_files", []))
            if failed_list:
                logger.warning(f"  WARNING: {len(failed_list)} images failed to upload")
                for failed in failed_list[:5]:  # Show only first 5 failures
                    logger.warning(f"    - {failed}")

            # Summary
            total_time = time.time() - start_time
            logger.info(f"\n  split summary:")
            logger.info(f"    Images processed: {image_count}")
            logger.info(f"    Annotations: {annotation_count}")
            logger.info(f"    Total size: {total_file_size / 1024 / 1024:.2f} MB")
            logger.info(f"    Total time: {total_time:.2f} seconds")
            logger.info(f"    Speed: {image_count / total_time:.1f} files/sec")

            return image_count, annotation_count, total_file_size
            
        finally:
            # Restore original parameters
            self.max_workers = original_max_workers
            self.batch_size = original_batch_size


# For backward compatibility, keep original class name
class UploadService(OptimizedUploadService):
    """Backward compatibility alias"""
    pass


# Global upload service instance
upload_service = OptimizedUploadService()
