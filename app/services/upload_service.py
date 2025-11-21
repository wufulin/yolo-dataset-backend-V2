"""Service for handling dataset upload operations."""

import asyncio
import hashlib
import logging
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

from bson import ObjectId
from fastapi import HTTPException, status
from PIL import Image

from app.config import settings
from app.models.dataset import Dataset
from app.services import dataset_service, image_service, minio_service
from app.utils import resolve_target_directory, yolo_validator
from app.utils.file_utils import safe_remove
from app.utils.logger import get_logger

logger: logging.Logger = get_logger(__name__)


class UploadService:
    """Service class for upload operations."""

    def __init__(self):
        """Initialize Upload service."""
        pass

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
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Dataset info is required",
            )

        try:
            dataset_type = getattr(dataset_info, "dataset_type", "detect")

            validation_result = await yolo_validator.validate_dataset(zip_path, dataset_type)

            if not validation_result.is_valid:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid YOLO dataset: {validation_result.message}",
                )

            class_names = validation_result.dataset_info.get("class_names", [])
            dataset_root = validation_result.dataset_info.get("dataset_root", None)

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

            # Ensure dataset_root exists before scheduling the background task
            if dataset_root is None:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Failed to extract dataset root directory",
                )

            # Schedule background processing without blocking the request
            asyncio.create_task(
                self._background_process_images_and_annotations(
                    dataset_root,
                    dataset_id,
                    dataset_type,
                    class_names,
                    zip_path,
                )
            )

            # Return immediately so the client does not wait for processing
            return {
                "status": "success",
                "dataset_id": dataset_id,
                "message": "Dataset created successfully. Processing images in background.",
                "dataset_type": dataset_type,
            }
        except Exception:
            # Clean up temporary files if dataset creation fails
            if dataset_root:
                safe_remove(dataset_root)
                if dataset_root.parent.exists():
                    safe_remove(dataset_root.parent)
            safe_remove(zip_path)
            raise

    async def _background_process_images_and_annotations(
        self,
        dataset_root: Path,
        dataset_id: str,
        dataset_type: str,
        class_names: List[str],
        zip_path: str,
    ) -> None:
        """
        Background worker that processes images and annotations with proper cleanup.

        Args:
            dataset_root: Root directory of dataset
            dataset_id: Dataset ID in MongoDB
            dataset_type: Type of dataset
            class_names: List of class names
            zip_path: Path to ZIP file for cleanup
        """
        try:
            logger.info("Starting background processing for dataset %s", dataset_id)

            # Process images and annotations
            processed_count = await self._process_images_and_annotations(
                dataset_root,
                dataset_id,
                dataset_type,
                class_names,
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
                "Dataset %s background processing completed. Processed %s images",
                dataset_id,
                processed_count,
            )

        except Exception as exc:
            logger.error(
                "Dataset %s background processing failed: %s",
                dataset_id,
                exc,
                exc_info=True,
            )

            # Update dataset status to error and record message
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
                    "Failed to update dataset status after error: %s",
                    update_error,
                    exc_info=True,
                )
        finally:
            # Clean up temporary files created during processing
            try:
                if dataset_root and dataset_root.exists():
                    safe_remove(dataset_root)
                    if dataset_root.parent.exists():
                        safe_remove(dataset_root.parent)
                if zip_path:
                    safe_remove(zip_path)
                logger.debug(
                    "Cleaned up temporary files: %s, %s",
                    dataset_root,
                    zip_path,
                )
            except Exception as cleanup_error:
                logger.warning(
                    "Failed to clean up temporary files: %s",
                    cleanup_error,
                    exc_info=True,
                )

    async def _process_images_and_annotations(
        self,
        dataset_root: Path,
        dataset_id: str,
        dataset_type: str,
        class_names: List[str],
    ) -> int:
        """
        Process all images and annotations in dataset.

        Args:
            dataset_root: Root directory of dataset
            dataset_id: Dataset ID in MongoDB
            dataset_type: Type of dataset
            class_names: List of class names

        Returns:
            int: Number of processed images
        """

        # Update dataset status for train split
        await dataset_service.repository.update(
            dataset_id,
            {
                "status": "processing",
                "message": "Processing train images in background.",
                "updated_at": datetime.utcnow(),
            },
        )

        train_images, train_annotations, train_size = await self.process_split(
            dataset_root,
            "train",
            dataset_id,
            dataset_type,
            class_names,
        )

        # Update dataset status for validation split
        await dataset_service.repository.update(
            dataset_id,
            {
                "status": "processing",
                "message": "Processing val images in background.",
                "updated_at": datetime.utcnow(),
            },
        )

        val_images, val_annotations, val_size = await self.process_split(
            dataset_root,
            "val",
            dataset_id,
            dataset_type,
            class_names,
        )

        # Update dataset status for test split
        await dataset_service.repository.update(
            dataset_id,
            {
                "status": "processing",
                "message": "Processing test images in background.",
                "updated_at": datetime.utcnow(),
            },
        )

        test_images, test_annotations, test_size = await self.process_split(
            dataset_root,
            "test",
            dataset_id,
            dataset_type,
            class_names,
        )

        await dataset_service.update_dataset_stats(
            dataset_id,
            train_images,
            train_annotations,
            train_size,
            val_images,
            val_annotations,
            val_size,
            test_images,
            test_annotations,
            test_size,
        )

        return train_images + val_images + test_images

    def _calculate_file_hash(self, file_path: Path) -> str:
        """
        Calculate file MD5 hash.

        Args:
            file_path: Path to the file

        Returns:
            str: MD5 hash of the file
        """
        md5_hash = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                md5_hash.update(chunk)
        return md5_hash.hexdigest()

    def _get_image_info(self, image_path: Path) -> Tuple[int, int, str]:
        """
        Get image information.

        Args:
            image_path: Path to the image file

        Returns:
            Tuple[int, int, str]: (width, height, format)
        """
        with Image.open(image_path) as img:
            width, height = img.size
            img_format = img.format.lower() if img.format else "jpg"
            return width, height, img_format

    async def process_split(
        self,
        dataset_root: Path,
        split_name: str,
        dataset_id: str,
        dataset_type: str,
        class_names: List[str],
        user_id: str = "691c3f00ca496bc2f41f0993",
    ) -> Tuple[int, int, int]:
        """
        Process a single dataset split with batch upload for better performance.

        Args:
            dataset_root: Root directory path of the dataset
            split_name: Split name (train/val/test)
            dataset_id: Dataset ID
            class_names: List of class names
            user_id: User ID for MinIO path (default: "691c3f00ca496bc2f41f0993")

        Returns:
            Tuple[int, int, int]: (image_count, annotation_count, total_file_size)
        """
        dataset_root_path = Path(dataset_root)
        images_dir = dataset_root_path / "images" / split_name
        labels_dir = dataset_root_path / "labels" / split_name

        if not images_dir.exists():
            logger.error("  WARNING: Images directory not found: %s", images_dir)
            return 0, 0, 0

        if not labels_dir.exists():
            logger.error("  WARNING: Labels directory not found: %s", labels_dir)
            return 0, 0, 0

        # Get all image files
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
        # find all image files based on the image_extensions
        image_files = [
            f for f in images_dir.glob("*") if f.suffix.lower() in image_extensions
        ]

        logger.info(f"\nProcessing {split_name} split:")
        logger.info(f"  Number of images: {len(image_files)}")

        # Phase 1: Prepare all image documents and upload list
        logger.info(f"  Phase 1: Preparing image data...")
        upload_list = []
        image_doc_list = []
        total_file_size = 0

        for image_path in image_files:
            try:
                # Get image information
                width, height, img_format = self._get_image_info(image_path)
                file_size = image_path.stat().st_size
                file_hash = self._calculate_file_hash(image_path)

                # Accumulate file size
                total_file_size += file_size

                # Corresponding label file
                label_path = labels_dir / f"{image_path.stem}.txt"

                # Parse annotations
                annotation_objects = await yolo_validator.parse_annotations(
                    str(label_path),
                    dataset_type,
                    class_names,
                )

                # Convert AnnotationInfo objects to dictionaries and set image_id and dataset_id
                image_id = ObjectId()
                annotations = []
                for ann_obj in annotation_objects:
                    # Convert dataclass to dict
                    ann_dict = asdict(ann_obj)
                    # Add image_id and dataset_id
                    ann_dict["image_id"] = image_id
                    ann_dict["dataset_id"] = ObjectId(dataset_id)
                    annotations.append(ann_dict)

                # MinIO path format: {user_id}/{dataset_id}/images/{split}/{filename}
                minio_file_path = (
                    f"{user_id}/{dataset_id}/images/{split_name}/{image_path.name}"
                )

                # Determine content type
                content_type = "image/jpeg"
                if image_path.suffix.lower() in [".png"]:
                    content_type = "image/png"
                elif image_path.suffix.lower() in [".jpg", ".jpeg"]:
                    content_type = "image/jpeg"
                elif image_path.suffix.lower() in [".bmp"]:
                    content_type = "image/bmp"
                elif image_path.suffix.lower() in [".tiff", ".tif"]:
                    content_type = "image/tiff"

                # Add to upload list: (local_path, minio_path, content_type)
                upload_list.append((str(image_path), minio_file_path, content_type))

                # Store image data for later database insertion
                image_doc_list.append(
                    {
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
                    }
                )
            except Exception as e:
                logger.error(
                    "  Failed to prepare %s: %s",
                    image_path.name,
                    e,
                    exc_info=True,
                )
                continue

        # Phase 2: Batch upload to MinIO
        logger.info(
            f"\n  Phase 2: Batch uploading {len(upload_list)} images to MinIO..."
        )
        # Use upload_files_parallel_async which supports per-file content_type
        upload_result = await minio_service.upload_files_parallel_async(
            upload_list,  # List of (file_path, object_name, content_type)
            bucket_name=settings.minio_bucket_name,  # Use default bucket
        )

        logger.info(
            f"  Upload completed: {upload_result['successful']}/{upload_result['total']} successful"
        )
        # Check for retry info if available
        if "retry_info" in upload_result and upload_result["retry_info"].get("total_retries", 0) > 0:
            logger.info(
                f"  Retries performed: {upload_result['retry_info']['total_retries']}"
            )

        # Phase 3: Insert successfully uploaded image docs to database
        logger.info(f"\n  Phase 3: Inserting image docs into database...")
        # upload_files_parallel_async returns "successful_files" (list of result dicts) 
        # instead of "success_list" (list of object names)
        if "success_list" in upload_result:
            successful_paths = set(upload_result["success_list"])
        elif "successful_files" in upload_result:
            # Extract object names from successful_files
            successful_paths = set(
                result.get("object_name", "") for result in upload_result["successful_files"]
                if isinstance(result, dict)
            )
        else:
            successful_paths = set()

        image_count = 0
        annotation_count = 0
        images_to_insert = []

        for image in image_doc_list:
            if image["file_path"] in successful_paths:
                images_to_insert.append(image)
                annotation_count += len(image["annotations"])

        # Batch insert to database
        if images_to_insert:
            inserted_count = image_service.bulk_save_images(images_to_insert)
            image_count = inserted_count

        # Log failed uploads
        failed_list = upload_result.get("failed_list", upload_result.get("failed_files", []))
        if failed_list:
            logger.warning(
                "\n  WARNING: %d images failed to upload:",
                len(failed_list),
            )
            for failed in failed_list[:10]:  # Show first 10 failures
                if isinstance(failed, dict):
                    object_name = failed.get("object_name", failed.get("file_path", "unknown"))
                    error = failed.get("error", "Unknown error")
                    logger.warning(f"    - {object_name}: {error}")
                else:
                    logger.warning(f"    - {failed}")
            if len(failed_list) > 10:
                logger.warning(
                    f"    ... and {len(failed_list) - 10} more"
                )

        logger.info(f"\n  Split summary:")
        logger.info(f"    Images processed: {image_count}")
        logger.info(f"    Annotations: {annotation_count}")
        logger.info(f"    Total size: {total_file_size / 1024 / 1024:.2f} MB")

        return image_count, annotation_count, total_file_size


# Global Upload service instance
upload_service = UploadService()
