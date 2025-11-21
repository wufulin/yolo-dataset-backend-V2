"""YOLO Format Validation and Parsing Service
Uses modern Python features (dataclasses, asyncio, Protocol) while maintaining backward compatibility.
Handles validation and parsing for multiple YOLO dataset types (detect, segment, pose, obb, classify).
"""

import asyncio
import hashlib
import json
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Tuple, runtime_checkable

import yaml

# Project internal imports (sorted by module)
from app.core.exceptions import (
    FileNotFoundException,
    YOLODatasetTypeException,
    YOLOValidationException,
)

# from app.core.decorators import exception_handler, performance_monitor, cache_result, async_exception_handler  # Avoid circular import
from app.utils.file_utils import resolve_target_directory
from app.utils.logger import get_logger


logger: logging.Logger = get_logger(__name__)


class DatasetType(Enum):
    """Dataset type enumeration for YOLO formats."""
    DETECT = "detect"    # Object detection
    SEGMENT = "segment"  # Instance segmentation
    POSE = "pose"        # Pose estimation
    OBB = "obb"          # Oriented bounding box
    CLASSIFY = "classify"# Image classification


@dataclass
class ValidationResult:
    """Structure to hold dataset validation outcome and metadata."""
    is_valid: bool
    message: str
    errors: List[str] = None
    warnings: List[str] = None
    dataset_info: Dict[str, Any] = None
    validation_time: datetime = None

    def __post_init__(self):
        """Initialize optional fields with default values if not provided."""
        if self.errors is None:
            self.errors = []
        if self.warnings is None:
            self.warnings = []
        if self.validation_time is None:
            self.validation_time = datetime.utcnow()


@dataclass
class AnnotationInfo:
    """Container for structured annotation metadata."""
    annotation_type: str  # e.g., "detect", "segment", "pose"
    class_id: int         # Class index from dataset YAML
    class_name: str       # Class name corresponding to class_id
    bbox: Optional[Dict[str, float]] = None  # Bounding box (for detect/obb)
    points: Optional[List[float]] = None     # Polygon points (for segment)
    keypoints: Optional[List[float]] = None  # Keypoint coordinates (for pose)
    confidence: Optional[float] = None       # Confidence score (if available)
    area: Optional[float] = None             # Bounding box/segment area
    metadata: Dict[str, Any] = None          # Additional annotation metadata

    def __post_init__(self):
        """Initialize optional metadata field if not provided."""
        if self.metadata is None:
            self.metadata = {}


class YOLOAnnotationParser(ABC):
    """Abstract base class defining interface for annotation parsers."""

    @abstractmethod
    async def parse(self, annotation_path: str, class_names: List[str]) -> List[AnnotationInfo]:
        """
        Parse annotation file into structured AnnotationInfo objects.
        
        Args:
            annotation_path: Path to the annotation TXT file
            class_names: List of class names from dataset YAML
            
        Returns:
            List of structured AnnotationInfo objects
        """
        pass

    @abstractmethod
    def get_format_version(self) -> str:
        """Return the supported annotation format version."""
        pass


class DetectAnnotationParser(YOLOAnnotationParser):
    """Parser for YOLO object detection annotations (normalized bbox format)."""

    async def parse(self, annotation_path: str, class_names: List[str]) -> List[AnnotationInfo]:
        """Parse detection annotations (class_id + normalized x_center/y_center/width/height)."""
        annotations = []
        annotation_path = Path(annotation_path)

        try:
            with open(annotation_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            for line_num, line in enumerate(lines, 1):
                line = line.strip()
                if not line:
                    continue  # Skip empty lines

                parts = line.split()
                if len(parts) < 5:
                    logger.warning(f"Invalid detection annotation at line {line_num} in {annotation_path}: {line}")
                    continue

                try:
                    class_id = int(parts[0])
                    # Get class name (fallback to "class_{class_id}" if index is out of range)
                    class_name = class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"

                    # Parse normalized bounding box coordinates
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])

                    # Calculate bounding box area (normalized)
                    area = width * height

                    annotations.append(AnnotationInfo(
                        annotation_type="detect",
                        class_id=class_id,
                        class_name=class_name,
                        bbox={
                            "x_center": x_center,
                            "y_center": y_center,
                            "width": width,
                            "height": height
                        },
                        area=area
                    ))

                except (ValueError, IndexError) as e:
                    logger.warning(f"Failed to parse detection annotation at line {line_num} in {annotation_path}: {e}")
                    continue

        except Exception as e:
            logger.error(f"Error reading detection annotation file {annotation_path}: {e}")
            raise

        return annotations

    def get_format_version(self) -> str:
        return "1.0"


class SegmentAnnotationParser(YOLOAnnotationParser):
    """Parser for YOLO instance segmentation annotations (polygon format)."""

    async def parse(self, annotation_path: str, class_names: List[str]) -> List[AnnotationInfo]:
        """Parse segmentation annotations (class_id + normalized polygon points)."""
        annotations = []
        annotation_path = Path(annotation_path)

        try:
            with open(annotation_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            for line_num, line in enumerate(lines, 1):
                line = line.strip()
                if not line:
                    continue  # Skip empty lines

                parts = line.split()
                # Require class_id + at least 2 points (x1,y1) to form a polygon
                if len(parts) < 5:
                    logger.warning(f"Invalid segment annotation at line {line_num} in {annotation_path}: {line}")
                    continue

                try:
                    class_id = int(parts[0])
                    class_name = class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"

                    # Parse normalized polygon points (x1,y1,x2,y2,...xn,yn)
                    points = [float(coord) for coord in parts[1:]]

                    annotations.append(AnnotationInfo(
                        annotation_type="segment",
                        class_id=class_id,
                        class_name=class_name,
                        points=points
                    ))

                except (ValueError, IndexError) as e:
                    logger.warning(f"Failed to parse segment annotation at line {line_num} in {annotation_path}: {e}")
                    continue

        except Exception as e:
            logger.error(f"Error reading segment annotation file {annotation_path}: {e}")
            raise

        return annotations

    def get_format_version(self) -> str:
        return "1.0"


class PoseAnnotationParser(YOLOAnnotationParser):
    """Parser for YOLO pose estimation annotations (keypoint format)."""

    async def parse(self, annotation_path: str, class_names: List[str]) -> List[AnnotationInfo]:
        """Parse pose annotations (class_id + normalized keypoint coordinates)."""
        annotations = []
        annotation_path = Path(annotation_path)

        try:
            with open(annotation_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            for line_num, line in enumerate(lines, 1):
                line = line.strip()
                if not line:
                    continue  # Skip empty lines

                parts = line.split()
                # Require class_id + at least 1 keypoint (x,y)
                if len(parts) < 3:
                    logger.warning(f"Invalid pose annotation at line {line_num} in {annotation_path}: {line}")
                    continue

                try:
                    class_id = int(parts[0])
                    class_name = class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"

                    # Parse normalized keypoint coordinates (x1,y1,x2,y2,...xn,yn)
                    keypoints = [float(coord) for coord in parts[1:]]

                    annotations.append(AnnotationInfo(
                        annotation_type="pose",
                        class_id=class_id,
                        class_name=class_name,
                        keypoints=keypoints
                    ))

                except (ValueError, IndexError) as e:
                    logger.warning(f"Failed to parse pose annotation at line {line_num} in {annotation_path}: {e}")
                    continue

        except Exception as e:
            logger.error(f"Error reading pose annotation file {annotation_path}: {e}")
            raise

        return annotations

    def get_format_version(self) -> str:
        return "1.0"


class ClassifyAnnotationParser(YOLOAnnotationParser):
    """Parser for YOLO image classification annotations (single class_id format)."""

    async def parse(self, annotation_path: str, class_names: List[str]) -> List[AnnotationInfo]:
        """Parse classification annotations (single class_id per image)."""
        annotations = []
        annotation_path = Path(annotation_path)

        try:
            with open(annotation_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            for line_num, line in enumerate(lines, 1):
                line = line.strip()
                if not line:
                    continue  # Skip empty lines

                parts = line.split()
                if not parts:
                    continue

                try:
                    class_id = int(parts[0])
                    class_name = class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"

                    annotations.append(AnnotationInfo(
                        annotation_type="classify",
                        class_id=class_id,
                        class_name=class_name
                    ))

                except (ValueError, IndexError) as e:
                    logger.warning(f"Failed to parse classify annotation at line {line_num} in {annotation_path}: {e}")
                    continue

        except Exception as e:
            logger.error(f"Error reading classify annotation file {annotation_path}: {e}")
            raise

        return annotations

    def get_format_version(self) -> str:
        return "1.0"


class YOLOValidator:
    """Enhanced YOLO format validator and parser with multi-type support."""

    def __init__(self):
        """Initialize validator with registry of dataset-specific parsers."""
        # Get supported dataset types from DatasetType enum
        self.supported_types = [t.value for t in DatasetType]
        
        # Register annotation parsers (key: dataset type, value: parser instance)
        self._parsers: Dict[str, YOLOAnnotationParser] = {
            DatasetType.DETECT.value: DetectAnnotationParser(),
            DatasetType.SEGMENT.value: SegmentAnnotationParser(),
            DatasetType.POSE.value: PoseAnnotationParser(),
            DatasetType.CLASSIFY.value: ClassifyAnnotationParser(),
            # OBB parser can be added later (e.g., ObbAnnotationParser())
        }

    async def validate_dataset(self, dataset_path: str, dataset_type: str) -> ValidationResult:
        """
        Validate YOLO dataset format and structure.
        
        Args:
            dataset_path: Path to the root directory of the dataset
            dataset_type: Type of dataset to validate (must be in supported_types)
            
        Returns:
            ValidationResult object containing validation outcome and metadata
            
        Raises:
            YOLODatasetTypeException: If dataset_type is unsupported
            FileNotFoundException: If dataset_path does not exist
            YOLOValidationException: If validation fails unexpectedly
        """
        logger.info(f"Starting YOLO dataset validation: {dataset_path} (type: {dataset_type})")

        # Validate dataset type is supported
        if dataset_type not in self.supported_types:
            raise YOLODatasetTypeException(
                f"Unsupported dataset type: {dataset_type}. Supported types: {', '.join(self.supported_types)}",
                dataset_type=dataset_type
            )

        # Validate dataset path exists
        dataset_path = Path(dataset_path)
        if not dataset_path.exists():
            raise FileNotFoundException(f"Dataset directory not found: {dataset_path}")

        # Execute validation tasks concurrently (only ultralytics check enabled by default)
        try:
            validation_tasks = [
                self._validate_from_ultralytics_lib(dataset_path, dataset_type),
                # self._validate_directory_structure(dataset_path, dataset_type),
                # self._validate_yaml_file(dataset_path),
                # self._validate_annotations(dataset_path, dataset_type),
                # self._validate_images(dataset_path, dataset_type)
            ]

            # Run tasks concurrently and collect results (including exceptions)
            results = await asyncio.gather(*validation_tasks, return_exceptions=True)

            # Combine results from all validation tasks
            all_errors: List[str] = []
            all_warnings: List[str] = []
            dataset_info: Dict[str, Any] = {}

            for result in results:
                if isinstance(result, Exception):
                    # Add exceptions as errors
                    all_errors.append(str(result))
                elif isinstance(result, ValidationResult):
                    # Merge errors and warnings from task result
                    if not result.is_valid:
                        all_errors.extend(result.errors)
                    all_warnings.extend(result.warnings)
                    # Merge dataset metadata
                    if result.dataset_info:
                        dataset_info.update(result.dataset_info)

            # Determine overall validation status
            is_valid = len(all_errors) == 0
            status_message = "Dataset validation completed successfully" if is_valid else "Dataset validation failed"
            
            return ValidationResult(
                is_valid=is_valid,
                message=status_message,
                errors=all_errors,
                warnings=all_warnings,
                dataset_info=dataset_info
            )

        except Exception as e:
            logger.error(f"Unexpected error during dataset validation: {e}", exc_info=True)
            raise YOLOValidationException(
                f"Dataset validation failed for {dataset_path}: {str(e)}",
                dataset_path=str(dataset_path)
            )

    async def _validate_from_ultralytics_lib(self, dataset_path: Path, dataset_type: str) -> ValidationResult:
        """
        Validate dataset using Ultralytics Hub's built-in dataset checker.
        
        Args:
            dataset_path: Path to dataset root
            dataset_type: Type of dataset to validate
            
        Returns:
            ValidationResult with outcome from Ultralytics checker
        """
        # Lazy import to avoid dependency if not using this validation method
        from ultralytics.hub import check_dataset
            
        # Run Ultralytics dataset check
        check_result = check_dataset(str(dataset_path), dataset_type)

        # Process check result (Ultralytics returns error string or True for success)
        if isinstance(check_result, str) and "error" in check_result.lower():
            logger.error(f"Ultralytics dataset validation failed: {check_result}")
            return ValidationResult(
                is_valid=False,
                message="Ultralytics dataset validation failed",
                errors=[check_result],
                warnings=[],
                dataset_info={}
            )

        logger.info(f"Ultralytics dataset validation passed for: {dataset_path}")

        # Resolve dataset root directory and parse YAML for class information
        dataset_root = resolve_target_directory(str(dataset_path))
        logger.info(f"Resolved dataset root directory: {dataset_root}")
        
        # Find and parse dataset YAML file
        dataset_yaml_path = self.find_dataset_yaml(str(dataset_root))
        yaml_data = self.parse_dataset_yaml(str(dataset_yaml_path))
        
        # Extract sorted class names from YAML
        class_names = [yaml_data['names'][i] for i in sorted(yaml_data['names'].keys())]

        # Collect dataset metadata
        dataset_info = {
            "dataset_type": dataset_type,
            "class_names": class_names,
            "dataset_root": str(dataset_root),
            "yaml_path": str(dataset_yaml_path),
            "num_classes": len(class_names)
        }

        return ValidationResult(
            is_valid=True,
            message="Ultralytics dataset validation successful",
            errors=[],
            warnings=[],
            dataset_info=dataset_info
        )

    async def _validate_directory_structure(self, dataset_path: Path, dataset_type: str) -> ValidationResult:
        """
        Validate dataset directory structure matches YOLO specifications.
        
        Args:
            dataset_path: Path to dataset root
            dataset_type: Type of dataset (affects required directories)
            
        Returns:
            ValidationResult with structure validation outcome
        """
        errors: List[str] = []
        warnings: List[str] = []
        dataset_info: Dict[str, Any] = {}

        # Define required/optional directories based on dataset type
        if dataset_type == DatasetType.CLASSIFY.value:
            required_dirs = ["train"]  # Classification only requires train directory
        else:
            required_dirs = ["train", "val", "test"]  # Detection/segment/pose require train/val/test

        # Check each required directory
        for dir_name in required_dirs:
            dir_path = dataset_path / dir_name
            if not dir_path.exists():
                warnings.append(f"Optional directory missing: {dir_name} (recommended for complete dataset)")
            elif not dir_path.is_dir():
                errors.append(f"Invalid directory: {dir_name} exists but is not a directory")

        # Type-specific structure checks
        if dataset_type == DatasetType.SEGMENT.value:
            # Check for segmentation-specific directories
            seg_dirs = ["segments", "masks"]
            for seg_dir in seg_dirs:
                if (dataset_path / "train" / seg_dir).exists():
                    dataset_info[f"has_{seg_dir}"] = True
                    logger.info(f"Found segmentation directory: {seg_dir}")

        return ValidationResult(
            is_valid=len(errors) == 0,
            message="Directory structure validation completed",
            errors=errors,
            warnings=warnings,
            dataset_info=dataset_info
        )

    async def _validate_yaml_file(self, dataset_path: Path) -> ValidationResult:
        """
        Validate dataset YAML configuration file (required for YOLO datasets).
        
        Checks for mandatory fields and validates referenced paths.
        
        Args:
            dataset_path: Path to dataset root
            
        Returns:
            ValidationResult with YAML validation outcome
        """
        errors: List[str] = []
        warnings: List[str] = []
        dataset_info: Dict[str, Any] = {}

        # Find all YAML files in dataset root
        yaml_files = list(dataset_path.glob("*.yaml")) + list(dataset_path.glob("*.yml"))

        if not yaml_files:
            errors.append("No dataset YAML file found (required for YOLO format)")
            return ValidationResult(
                is_valid=False,
                message="YAML validation failed - no YAML file found",
                errors=errors,
                warnings=warnings
            )

        # Use the first found YAML file (standard for YOLO datasets)
        yaml_file = yaml_files[0]
        logger.info(f"Validating dataset YAML file: {yaml_file}")

        try:
            # Parse YAML file (use safe_load to prevent code execution)
            with open(yaml_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)

            # Check for mandatory fields in YAML
            mandatory_fields = ["train", "val", "nc", "names"]
            for field in mandatory_fields:
                if field not in config:
                    errors.append(f"Missing mandatory field in YAML: '{field}'")

            # Validate training/validation paths referenced in YAML
            if "train" in config:
                train_path = dataset_path / config["train"]
                if not train_path.exists():
                    errors.append(f"Training data path not found: {config['train']} (referenced in YAML)")
            
            if "val" in config:
                val_path = dataset_path / config["val"]
                if not val_path.exists():
                    errors.append(f"Validation data path not found: {config['val']} (referenced in YAML)")

            # Validate class names configuration
            if "names" in config:
                if isinstance(config["names"], dict):
                    dataset_info["num_classes"] = len(config["names"])
                    dataset_info["class_mapping"] = config["names"]
                elif isinstance(config["names"], list):
                    dataset_info["num_classes"] = len(config["names"])
                    dataset_info["class_mapping"] = {i: name for i, name in enumerate(config["names"])}
                else:
                    warnings.append("Invalid 'names' format in YAML - expected dict or list")
            else:
                warnings.append("No 'names' field found in YAML (class names will be unavailable)")

        except Exception as e:
            errors.append(f"Failed to parse YAML file {yaml_file}: {str(e)}")

        return ValidationResult(
            is_valid=len(errors) == 0,
            message="YAML configuration validation completed",
            errors=errors,
            warnings=warnings,
            dataset_info=dataset_info
        )

    async def _validate_annotations(self, dataset_path: Path, dataset_type: str) -> ValidationResult:
        """
        Validate annotation files (format, structure, and basic integrity).
        
        Args:
            dataset_path: Path to dataset root
            dataset_type: Type of dataset (affects annotation format checks)
            
        Returns:
            ValidationResult with annotation validation outcome
        """
        errors: List[str] = []
        warnings: List[str] = []
        dataset_info: Dict[str, Any] = {"annotation_stats": {}}

        # Find all annotation TXT files (recursive search)
        annotation_files = list(dataset_path.rglob("*.txt"))

        if not annotation_files:
            warnings.append("No annotation files found in dataset (normal for classification if using folder structure)")
            return ValidationResult(
                is_valid=True,
                message="Annotation validation completed (no files found)",
                errors=errors,
                warnings=warnings,
                dataset_info=dataset_info
            )

        # Basic annotation statistics
        total_annotations = 0
        invalid_annotations = 0

        for ann_file in annotation_files:
            try:
                with open(ann_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()

                for line in lines:
                    line = line.strip()
                    if not line:
                        continue  # Skip empty lines

                    total_annotations += 1
                    parts = line.split()

                    # Basic format check: first part must be class_id (integer)
                    if not parts or not parts[0].lstrip('-').isdigit():
                        invalid_annotations += 1
                        warnings.append(f"Invalid annotation format in {ann_file}: '{line}' (class_id must be integer)")

            except Exception as e:
                errors.append(f"Failed to read annotation file {ann_file}: {str(e)}")

        # Update annotation statistics
        dataset_info["annotation_stats"] = {
            "total_files": len(annotation_files),
            "total_annotations": total_annotations,
            "invalid_annotations": invalid_annotations
        }

        # Log summary
        logger.info(
            f"Annotation validation summary: {len(annotation_files)} files, "
            f"{total_annotations} annotations, {invalid_annotations} invalid entries"
        )

        return ValidationResult(
            is_valid=len(errors) == 0,
            message="Annotation validation completed",
            errors=errors,
            warnings=warnings,
            dataset_info=dataset_info
        )

    async def _validate_images(self, dataset_path: Path, dataset_type: str) -> ValidationResult:
        """
        Validate image files (presence, format, and basic statistics).
        
        Args:
            dataset_path: Path to dataset root
            dataset_type: Type of dataset (not used for basic image checks)
            
        Returns:
            ValidationResult with image validation outcome and statistics
        """
        errors: List[str] = []
        warnings: List[str] = []
        dataset_info: Dict[str, Any] = {"image_stats": {}}

        # Supported image formats (YOLO-compatible)
        supported_image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

        # Find all image files (recursive search)
        image_files = [
            f for f in dataset_path.rglob("*")
            if f.is_file() and f.suffix.lower() in supported_image_extensions
        ]

        if not image_files:
            errors.append("No valid image files found in dataset")
            return ValidationResult(
                is_valid=False,
                message="Image validation failed - no images found",
                errors=errors,
                warnings=warnings,
                dataset_info=dataset_info
            )

        # Calculate image statistics
        total_images = len(image_files)
        total_size_bytes = sum(f.stat().st_size for f in image_files)
        total_size_mb = round(total_size_bytes / 1024 / 1024, 2)

        # Optional: Check for extremely small/large images (configurable thresholds)
        small_image_threshold = 10 * 1024  # 10KB
        large_image_threshold = 100 * 1024 * 1024  # 100MB
        small_images = [f for f in image_files if f.stat().st_size < small_image_threshold]
        large_images = [f for f in image_files if f.stat().st_size > large_image_threshold]

        if small_images:
            warnings.append(f"Found {len(small_images)} unusually small images (size < {small_image_threshold/1024:.0f}KB)")
        if large_images:
            warnings.append(f"Found {len(large_images)} unusually large images (size > {large_image_threshold/1024/1024:.0f}MB)")

        # Update image statistics in dataset info
        dataset_info["image_stats"] = {
            "total_images": total_images,
            "total_size_mb": total_size_mb,
            "small_images_count": len(small_images),
            "large_images_count": len(large_images),
            "supported_formats": sorted(supported_image_extensions)
        }

        # Optional: Add image format validation (e.g., using PIL to check corrupt images)
        # Uncomment below to enable (requires Pillow installation)
        # try:
        #     from PIL import Image
        #     corrupt_images = []
        #     for img_file in image_files[:100]:  # Sample first 100 images to avoid performance issues
        #         try:
        #             with Image.open(img_file) as img:
        #                 img.verify()
        #         except Exception:
        #             corrupt_images.append(str(img_file))
        #     if corrupt_images:
        #         warnings.append(f"Found {len(corrupt_images)} corrupt image files: {', '.join(corrupt_images[:5])}...")
        # except ImportError:
        #     logger.warning("Pillow not installed - skipping image corruption check")

        return ValidationResult(
            is_valid=len(errors) == 0,
            message="Image validation completed",
            errors=errors,
            warnings=warnings,
            dataset_info=dataset_info
        )

    def find_dataset_yaml(self, directory: str) -> Path:
        """
        Locate the dataset YAML configuration file in a directory.
        
        Args:
            directory: Path to directory to search for YAML file
            
        Returns:
            Path to the first found YAML file
            
        Raises:
            FileNotFoundException: If directory does not exist
            YOLOValidationException: If no YAML file is found
        """
        dir_path = Path(directory)

        if not dir_path.exists():
            raise FileNotFoundException(f"Directory not found: {dir_path}")

        # Search for .yaml and .yml files in the directory
        yaml_files = list(dir_path.glob("*.yaml")) + list(dir_path.glob("*.yml"))

        if not yaml_files:
            raise YOLOValidationException(
                f"No dataset YAML file found in directory: {dir_path}. "
                "Expected a .yaml or .yml file with dataset configuration."
            )

        # Return the first found YAML file (YOLO datasets typically have one)
        return yaml_files[0]

    def parse_dataset_yaml(self, yaml_path: str) -> Dict[str, Any]:
        """
        Parse dataset YAML configuration file into a dictionary.
        
        Args:
            yaml_path: Path to the dataset YAML file
            
        Returns:
            Parsed YAML data as a dictionary
            
        Raises:
            FileNotFoundException: If YAML file does not exist
            YOLOValidationException: If parsing fails
        """
        yaml_path = Path(yaml_path)
        logger.info(f"Parsing dataset YAML file: {yaml_path}")

        if not yaml_path.exists():
            raise FileNotFoundException(f"Dataset YAML file not found: {yaml_path}")

        try:
            with open(yaml_path, 'r', encoding='utf-8') as f:
                # Use safe_load to prevent execution of arbitrary code
                parsed_data = yaml.safe_load(f)

            logger.info(f"Successfully parsed dataset YAML: {yaml_path}")
            return parsed_data or {}  # Return empty dict if YAML is empty

        except Exception as e:
            logger.error(f"Failed to parse YAML file {yaml_path}: {e}", exc_info=True)
            raise YOLOValidationException(
                f"Failed to parse dataset YAML: {str(e)}. Check YAML syntax and file permissions."
            )

    def get_dataset_type(self, dataset_path: str) -> str:
        """
        Detect dataset type automatically based on directory structure and content.
        
        Uses heuristic checks to identify:
        - OBB: Presence of OBB-related directory/filename patterns
        - Classify: Presence of classification-related patterns
        - Segment: Presence of segmentation directories (segments/masks)
        - Pose: Presence of pose-related patterns
        - Detect: Default if no other type is detected
        
        Args:
            dataset_path: Path to dataset root
            
        Returns:
            Detected dataset type (one of supported_types)
            
        Raises:
            FileNotFoundException: If dataset path does not exist
        """
        logger.info(f"Automatically detecting dataset type for: {dataset_path}")
        dataset_path = Path(dataset_path)

        if not dataset_path.exists():
            raise FileNotFoundException(f"Dataset path does not exist: {dataset_path}")

        # Heuristic checks (ordered by specificity to avoid false positives)
        type_detection_checks = [
            (self._has_obb_annotations, DatasetType.OBB.value),
            (self._has_classify_annotations, DatasetType.CLASSIFY.value),
            (self._has_segmentation_annotations, DatasetType.SEGMENT.value),
            (self._has_pose_annotations, DatasetType.POSE.value),
        ]

        # Run detection checks in order
        for detection_func, detected_type in type_detection_checks:
            if detection_func(str(dataset_path)):
                logger.info(f"Detected dataset type: {detected_type}")
                return detected_type

        # Default to detection if no other type is detected
        default_type = DatasetType.DETECT.value
        logger.info(f"No specific dataset type detected - defaulting to: {default_type}")
        return default_type

    def _has_classify_annotations(self, dataset_path: str) -> bool:
        """
        Heuristic check for classification dataset: presence of classification-related patterns.
        
        Args:
            dataset_path: Path to dataset root
            
        Returns:
            True if classification dataset is suspected, False otherwise
        """
        classify_patterns = ['classify', 'classification', 'cls', 'category']
        dir_path = Path(dataset_path)

        try:
            # Check directory names for classification patterns
            for item in dir_path.iterdir():
                if item.is_dir() and any(pattern in item.name.lower() for pattern in classify_patterns):
                    return True
        except Exception as e:
            logger.debug(f"Error checking classification patterns: {e}")

        return False

    def _has_obb_annotations(self, dataset_path: str) -> bool:
        """
        Heuristic check for OBB dataset: presence of OBB-related patterns.
        
        Args:
            dataset_path: Path to dataset root
            
        Returns:
            True if OBB dataset is suspected, False otherwise
        """
        obb_patterns = ['obb', 'rotated', 'rbox', 'oriented']
        dir_path = Path(dataset_path)

        try:
            # Check directory/filename patterns for OBB indicators
            for item in dir_path.rglob("*"):
                if any(pattern in item.name.lower() for pattern in obb_patterns):
                    return True
        except Exception as e:
            logger.debug(f"Error checking OBB patterns: {e}")

        return False

    def _has_segmentation_annotations(self, dataset_path: str) -> bool:
        """
        Heuristic check for segmentation dataset: presence of segmentation directories.
        
        Args:
            dataset_path: Path to dataset root
            
        Returns:
            True if segmentation dataset is suspected, False otherwise
        """
        seg_dir_patterns = ['segments', 'masks', 'polygons', 'segmentation']
        dir_path = Path(dataset_path)

        try:
            # Check for segmentation-specific directories
            for seg_pattern in seg_dir_patterns:
                if any(dir_path.rglob(seg_pattern)):
                    return True
        except Exception as e:
            logger.debug(f"Error checking segmentation patterns: {e}")

        return False

    def _has_pose_annotations(self, dataset_path: str) -> bool:
        """
        Heuristic check for pose dataset: presence of pose-related patterns.
        
        Args:
            dataset_path: Path to dataset root
            
        Returns:
            True if pose dataset is suspected, False otherwise
        """
        pose_patterns = ['keypoints', 'pose', 'skeleton', 'kps']
        dir_path = Path(dataset_path)

        try:
            # Check directory/filename patterns for pose indicators
            for item in dir_path.iterdir():
                if item.is_dir() and any(pattern in item.name.lower() for pattern in pose_patterns):
                    return True
        except Exception as e:
            logger.debug(f"Error checking pose patterns: {e}")

        return False

    async def parse_annotations(self, annotation_path: str, dataset_type: str,
                              class_names: List[str]) -> List[AnnotationInfo]:
        """
        Parse annotation file using the appropriate dataset-specific parser.
        
        Args:
            annotation_path: Path to annotation TXT file
            dataset_type: Type of dataset (determines parser to use)
            class_names: List of class names from dataset YAML
            
        Returns:
            List of structured AnnotationInfo objects
            
        Raises:
            YOLODatasetTypeException: If dataset_type is unsupported
            Exception: If parsing fails
        """
        # Validate dataset type is supported
        if dataset_type not in self._parsers:
            raise YOLODatasetTypeException(
                f"Unsupported dataset type for parsing: {dataset_type}. "
                f"Supported types: {', '.join(self._parsers.keys())}"
            )

        # Get the appropriate parser for the dataset type
        parser = self._parsers[dataset_type]

        try:
            logger.debug(f"Parsing annotations with {parser.__class__.__name__}: {annotation_path}")
            annotations = await parser.parse(annotation_path, class_names)
            logger.debug(f"Successfully parsed {len(annotations)} annotations from: {annotation_path}")
            return annotations
        except Exception as e:
            logger.error(f"Failed to parse annotations from {annotation_path}: {str(e)}", exc_info=True)
            raise


# Global YOLO validator instance (for convenient access across the application)
yolo_validator = YOLOValidator()