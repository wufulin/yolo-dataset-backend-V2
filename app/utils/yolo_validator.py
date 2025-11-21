"""YOLO format validation and parsing service - 改进版本，使用现代Python特性."""
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

from app.core.exceptions import (
    FileNotFoundException,
    YOLODatasetTypeException,
    YOLOValidationException,
)

# from app.core.decorators import exception_handler, performance_monitor, cache_result, async_exception_handler  # 避免循环导入
from app.utils.file_utils import resolve_target_directory
from app.utils.logger import get_logger

logger: logging.Logger = get_logger(__name__)


class DatasetType(Enum):
    """数据集类型枚举"""
    DETECT = "detect"
    SEGMENT = "segment"
    POSE = "pose"
    OBB = "obb"
    CLASSIFY = "classify"


@dataclass
class ValidationResult:
    """验证结果数据类"""
    is_valid: bool
    message: str
    errors: List[str] = None
    warnings: List[str] = None
    dataset_info: Dict[str, Any] = None
    validation_time: datetime = None

    def __post_init__(self):
        if self.errors is None:
            self.errors = []
        if self.warnings is None:
            self.warnings = []
        if self.validation_time is None:
            self.validation_time = datetime.utcnow()


@dataclass
class AnnotationInfo:
    """标注信息数据类"""
    annotation_type: str
    class_id: int
    class_name: str
    bbox: Optional[Dict[str, float]] = None
    points: Optional[List[float]] = None
    keypoints: Optional[List[float]] = None
    confidence: Optional[float] = None
    area: Optional[float] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class YOLOAnnotationParser(ABC):
    """YOLO标注解析器抽象基类"""

    @abstractmethod
    async def parse(self, annotation_path: str, class_names: List[str]) -> List[AnnotationInfo]:
        """解析标注文件"""
        pass

    @abstractmethod
    def get_format_version(self) -> str:
        """获取格式版本"""
        pass


class DetectAnnotationParser(YOLOAnnotationParser):
    """检测标注解析器"""

    async def parse(self, annotation_path: str, class_names: List[str]) -> List[AnnotationInfo]:
        """解析检测标注"""
        annotations = []

        try:
            with open(annotation_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            for line_num, line in enumerate(lines, 1):
                line = line.strip()
                if not line:
                    continue

                parts = line.split()
                if len(parts) < 5:
                    logger.warning(f"Invalid detection annotation format at line {line_num}: {line}")
                    continue

                try:
                    class_id = int(parts[0])
                    class_name = class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"

                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])

                    annotation = AnnotationInfo(
                        annotation_type="detect",
                        class_id=class_id,
                        class_name=class_name,
                        bbox={
                            "x_center": x_center,
                            "y_center": y_center,
                            "width": width,
                            "height": height
                        },
                        area=width * height
                    )
                    annotations.append(annotation)

                except (ValueError, IndexError) as e:
                    logger.warning(f"Error parsing detection annotation at line {line_num}: {e}")
                    continue

        except Exception as e:
            logger.error(f"Error reading detection annotation file {annotation_path}: {e}")
            raise

        return annotations

    def get_format_version(self) -> str:
        return "1.0"


class SegmentAnnotationParser(YOLOAnnotationParser):
    """分割标注解析器"""

    async def parse(self, annotation_path: str, class_names: List[str]) -> List[AnnotationInfo]:
        """解析分割标注"""
        annotations = []

        try:
            with open(annotation_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            for line_num, line in enumerate(lines, 1):
                line = line.strip()
                if not line:
                    continue

                parts = line.split()
                if len(parts) < 5:  # 至少需要class_id + x1,y1,x2,y2
                    logger.warning(f"Invalid segment annotation format at line {line_num}: {line}")
                    continue

                try:
                    class_id = int(parts[0])
                    class_name = class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"

                    # 分割标注的点是归一化坐标
                    points = [float(x) for x in parts[1:]]

                    annotation = AnnotationInfo(
                        annotation_type="segment",
                        class_id=class_id,
                        class_name=class_name,
                        points=points
                    )
                    annotations.append(annotation)

                except (ValueError, IndexError) as e:
                    logger.warning(f"Error parsing segment annotation at line {line_num}: {e}")
                    continue

        except Exception as e:
            logger.error(f"Error reading segment annotation file {annotation_path}: {e}")
            raise

        return annotations

    def get_format_version(self) -> str:
        return "1.0"


class PoseAnnotationParser(YOLOAnnotationParser):
    """姿态估计标注解析器"""

    async def parse(self, annotation_path: str, class_names: List[str]) -> List[AnnotationInfo]:
        """解析姿态估计标注"""
        annotations = []

        try:
            with open(annotation_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            for line_num, line in enumerate(lines, 1):
                line = line.strip()
                if not line:
                    continue

                parts = line.split()
                if len(parts) < 3:  # 至少需要class_id + x,y
                    logger.warning(f"Invalid pose annotation format at line {line_num}: {line}")
                    continue

                try:
                    class_id = int(parts[0])
                    class_name = class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"

                    # 姿态标注的关键点坐标
                    keypoints = [float(x) for x in parts[1:]]

                    annotation = AnnotationInfo(
                        annotation_type="pose",
                        class_id=class_id,
                        class_name=class_name,
                        keypoints=keypoints
                    )
                    annotations.append(annotation)

                except (ValueError, IndexError) as e:
                    logger.warning(f"Error parsing pose annotation at line {line_num}: {e}")
                    continue

        except Exception as e:
            logger.error(f"Error reading pose annotation file {annotation_path}: {e}")
            raise

        return annotations

    def get_format_version(self) -> str:
        return "1.0"


class ClassifyAnnotationParser(YOLOAnnotationParser):
    """分类标注解析器"""

    async def parse(self, annotation_path: str, class_names: List[str]) -> List[AnnotationInfo]:
        """解析分类标注"""
        annotations = []

        try:
            with open(annotation_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            for line_num, line in enumerate(lines, 1):
                line = line.strip()
                if not line:
                    continue

                parts = line.split()
                if not parts:
                    continue

                try:
                    class_id = int(parts[0])
                    class_name = class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"

                    annotation = AnnotationInfo(
                        annotation_type="classify",
                        class_id=class_id,
                        class_name=class_name
                    )
                    annotations.append(annotation)

                except (ValueError, IndexError) as e:
                    logger.warning(f"Error parsing classify annotation at line {line_num}: {e}")
                    continue

        except Exception as e:
            logger.error(f"Error reading classify annotation file {annotation_path}: {e}")
            raise

        return annotations

    def get_format_version(self) -> str:
        return "1.0"


class YOLOValidator:
    """YOLO格式验证和解析类 - 改进版本"""

    def __init__(self):
        """初始化验证器"""
        self.supported_types = [t.value for t in DatasetType]
        self._parsers: Dict[str, YOLOAnnotationParser] = {
            DatasetType.DETECT.value: DetectAnnotationParser(),
            DatasetType.SEGMENT.value: SegmentAnnotationParser(),
            DatasetType.POSE.value: PoseAnnotationParser(),
            DatasetType.CLASSIFY.value: ClassifyAnnotationParser(),
            # OBB解析器可以后续添加
        }

    async def validate_dataset(self, dataset_path: str, dataset_type: str) -> ValidationResult:
        """
        验证YOLO数据集格式

        Args:
            dataset_path: 数据集目录路径
            dataset_type: 数据集类型

        Returns:
            ValidationResult: 验证结果
        """
        logger.info(f"Validating YOLO dataset: {dataset_path} (type: {dataset_type})")

        # 验证数据集类型
        if dataset_type not in self.supported_types:
            raise YOLODatasetTypeException(
                f"Unsupported dataset type: {dataset_type}",
                dataset_type=dataset_type
            )

        # 验证路径存在
        dataset_path = Path(dataset_path)
        if not dataset_path.exists():
            raise FileNotFoundException(str(dataset_path))

        # 执行验证
        try:
            validation_tasks = [
                self._validate_from_ultralytics_lib(dataset_path, dataset_type),
                # self._validate_directory_structure(dataset_path, dataset_type),
                # self._validate_yaml_file(dataset_path),
                # self._validate_annotations(dataset_path, dataset_type),
                # self._validate_images(dataset_path, dataset_type)
            ]

            results = await asyncio.gather(*validation_tasks, return_exceptions=True)

            # 合并验证结果
            all_errors = []
            all_warnings = []
            dataset_info = {}

            for result in results:
                if isinstance(result, Exception):
                    all_errors.append(str(result))
                elif isinstance(result, ValidationResult):
                    if not result.is_valid:
                        all_errors.extend(result.errors)
                    all_warnings.extend(result.warnings)
                    if result.dataset_info:
                        dataset_info.update(result.dataset_info)

            is_valid = len(all_errors) == 0

            return ValidationResult(
                is_valid=is_valid,
                message="Dataset validation completed",
                errors=all_errors,
                warnings=all_warnings,
                dataset_info=dataset_info
            )

        except Exception as e:
            logger.error(f"Dataset validation error: {e}", exc_info=True)
            raise YOLOValidationException(
                f"Dataset validation failed: {str(e)}",
                dataset_path=dataset_path
            )

    async def _validate_from_ultralytics_lib(self, dataset_path: Path, dataset_type: str) -> ValidationResult:
        """使用ultralytics库验证数据集"""
        from ultralytics.hub import check_dataset
            
        result = check_dataset(dataset_path, dataset_type)
        if isinstance(result, str) and "error" in result.lower():
            logger.error(f"Dataset validation failed: {result}")
            return ValidationResult(
                is_valid=False,
                message=result,
                errors=[result],
                warnings=[],
                dataset_info={}
            )

        logger.info(f"Dataset validation successful for: {dataset_path}")

        dataset_root = resolve_target_directory(dataset_path)

        logger.info(f"Dataset root: {dataset_root}")
        
        dataset_yaml_path = yolo_validator.find_dataset_yaml(str(dataset_root))

        yaml_data = yolo_validator.parse_dataset_yaml(str(dataset_yaml_path))
        class_names = [yaml_data['names'][i] for i in sorted(yaml_data['names'].keys())]

        dataset_info = {
            "dataset_type": dataset_type,
            "class_names": class_names,
            "dataset_root": dataset_root
        }

        return ValidationResult(
            is_valid=True,
            message="Dataset validation successful",
            errors=[],
            warnings=[],
            dataset_info=dataset_info
        )

    async def _validate_directory_structure(self, dataset_path: Path, dataset_type: str) -> ValidationResult:
        """验证目录结构"""
        errors = []
        warnings = []
        dataset_info = {}

        # 检查必要的目录
        required_dirs = ["train", "val", "test"] if dataset_type != DatasetType.CLASSIFY.value else ["train"]

        for dir_name in required_dirs:
            dir_path = dataset_path / dir_name
            if not dir_path.exists():
                warnings.append(f"Optional directory missing: {dir_name}")
            elif not dir_path.is_dir():
                errors.append(f"{dir_name} exists but is not a directory")

        # 检查数据集类型特定的目录结构
        if dataset_type == DatasetType.SEGMENT.value:
            seg_dirs = ["segments", "masks"]
            for seg_dir in seg_dirs:
                if (dataset_path / "train" / seg_dir).exists():
                    dataset_info[f"has_{seg_dir}"] = True

        return ValidationResult(
            is_valid=len(errors) == 0,
            message="Directory structure validation completed",
            errors=errors,
            warnings=warnings,
            dataset_info=dataset_info
        )

    async def _validate_yaml_file(self, dataset_path: Path) -> ValidationResult:
        """验证YAML配置文件"""
        errors = []
        warnings = []

        yaml_files = list(dataset_path.glob("*.yaml")) + list(dataset_path.glob("*.yml"))

        if not yaml_files:
            errors.append("No dataset YAML file found")
            return ValidationResult(False, "YAML validation failed", errors, warnings)

        yaml_file = yaml_files[0]

        try:
            with open(yaml_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)

            # 验证必需的字段
            required_fields = ["train", "val", "nc", "names"]
            for field in required_fields:
                if field not in config:
                    errors.append(f"Missing required field in YAML: {field}")

            # 验证路径格式
            if "train" in config:
                train_path = dataset_path / config["train"]
                if not train_path.exists():
                    errors.append(f"Training data path does not exist: {config['train']}")

            # 验证类别名称
            if "names" in config and isinstance(config["names"], dict):
                dataset_info = {"num_classes": len(config["names"])}
            else:
                warnings.append("Invalid or missing class names")
                dataset_info = {}

        except Exception as e:
            errors.append(f"Error parsing YAML file: {e}")
            dataset_info = {}

        return ValidationResult(
            is_valid=len(errors) == 0,
            message="YAML validation completed",
            errors=errors,
            warnings=warnings,
            dataset_info=dataset_info
        )

    async def _validate_annotations(self, dataset_path: Path, dataset_type: str) -> ValidationResult:
        """验证标注文件"""
        errors = []
        warnings = []
        dataset_info = {"annotation_stats": {}}

        # 收集所有txt文件
        txt_files = list(dataset_path.rglob("*.txt"))

        if not txt_files:
            warnings.append("No annotation files found")
            return ValidationResult(True, "Annotation validation completed", errors, warnings)

        annotation_count = 0

        for txt_file in txt_files:
            try:
                with open(txt_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()

                for line in lines:
                    line = line.strip()
                    if line:
                        annotation_count += 1
                        parts = line.split()
                        if not parts or not parts[0].isdigit():
                            warnings.append(f"Invalid annotation format in {txt_file}: {line}")

            except Exception as e:
                errors.append(f"Error reading annotation file {txt_file}: {e}")

        dataset_info["annotation_stats"]["total_annotations"] = annotation_count
        dataset_info["annotation_stats"]["total_files"] = len(txt_files)

        return ValidationResult(
            is_valid=len(errors) == 0,
            message="Annotation validation completed",
            errors=errors,
            warnings=warnings,
            dataset_info=dataset_info
        )

    async def _validate_images(self, dataset_path: Path, dataset_type: str) -> ValidationResult:
        """验证图像文件"""
        errors = []
        warnings = []
        dataset_info = {"image_stats": {}}

        # 收集所有图像文件
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
        image_files = [f for f in dataset_path.rglob("*")
                      if f.is_file() and f.suffix.lower() in image_extensions]

        if not image_files:
            warnings.append("No image files found")
            return ValidationResult(True, "Image validation completed", errors, warnings)

        image_count = len(image_files)
        total_size = sum(f.stat().st_size for f in image_files)

        dataset_info["image_stats"]["total_images"] = image_count
        dataset_info["image_stats"]["total_size_mb"] = round(total_size / 1024 / 1024, 2)

        # 检查图像大小分布
        small_images = 0
        large_images = 0

        for image_file in image_files:
            try:
                # 这里可以添加图像格式验证逻辑
                pass
            except Exception as e:
                warnings.append(f"Error processing image {image_file}: {e}")

        return ValidationResult(
            is_valid=len(errors) == 0,
            message="Image validation completed",
            errors=errors,
            warnings=warnings,
            dataset_info=dataset_info
        )

    def find_dataset_yaml(self, directory: str) -> Path:
        """查找数据集YAML文件"""
        directory = Path(directory)

        if not directory.exists():
            raise FileNotFoundException(f"Directory does not exist: {directory}")

        yaml_files = list(directory.glob("*.yaml")) + list(directory.glob("*.yml"))

        if not yaml_files:
            raise YOLOValidationException(f"No dataset YAML found in: {directory}")

        # 返回第一个找到的YAML文件
        return yaml_files[0]

    def parse_dataset_yaml(self, yaml_path: str) -> Dict[str, Any]:
        """解析数据集YAML文件"""
        logger.info(f"Parsing dataset YAML: {yaml_path}")

        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundException(f"YAML file not found: {yaml_path}")

        try:
            with open(yaml_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)

            logger.info(f"Successfully parsed YAML file: {yaml_path}")
            return data or {}

        except Exception as e:
            logger.error(f"Failed to parse YAML {yaml_path}: {e}", exc_info=True)
            raise YOLOValidationException(f"Failed to parse YAML: {str(e)}")

    def get_dataset_type(self, dataset_path: str) -> str:
        """检测数据集类型"""
        logger.info(f"Detecting dataset type for: {dataset_path}")

        dataset_path = Path(dataset_path)

        if not dataset_path.exists():
            raise FileNotFoundException(f"Dataset path does not exist: {dataset_path}")

        # 检查特定的文件模式
        type_checks = [
            (self._has_obb_annotations, 'obb'),
            (self._has_classify_annotations, 'classify'),
            (self._has_segmentation_annotations, 'segment'),
            (self._has_pose_annotations, 'pose'),
        ]

        for check_func, dataset_type in type_checks:
            if check_func(str(dataset_path)):
                logger.info(f"Detected {dataset_type} dataset type")
                return dataset_type

        # 默认检测为检测数据集
        logger.info("Detected detection dataset type (default)")
        return 'detect'

    def _has_classify_annotations(self, dataset_path: str) -> bool:
        """检查是否有分类标注"""
        classify_patterns = ['classify', 'classification']

        try:
            for item in os.listdir(dataset_path):
                if any(pattern in item.lower() for pattern in classify_patterns):
                    return True
        except Exception:
            pass

        return False

    def _has_obb_annotations(self, dataset_path: str) -> bool:
        """检查是否有OBB标注"""
        obb_patterns = ['obb', 'rotated', 'rbox']

        try:
            for item in os.listdir(dataset_path):
                if any(pattern in item.lower() for pattern in obb_patterns):
                    return True
        except Exception:
            pass

        return False

    def _has_segmentation_annotations(self, dataset_path: str) -> bool:
        """检查是否有分割标注"""
        seg_dirs = ['segments', 'masks', 'polygons']

        try:
            for seg_dir in seg_dirs:
                if (Path(dataset_path) / seg_dir).exists():
                    return True
        except Exception:
            pass

        return False

    def _has_pose_annotations(self, dataset_path: str) -> bool:
        """检查是否有姿态估计标注"""
        pose_patterns = ['keypoints', 'pose', 'skeleton']

        try:
            for item in os.listdir(dataset_path):
                if any(pattern in item.lower() for pattern in pose_patterns):
                    return True
        except Exception:
            pass

        return False

    async def parse_annotations(self, annotation_path: str, dataset_type: str,
                              class_names: List[str]) -> List[AnnotationInfo]:
        """解析标注文件"""

        if dataset_type not in self._parsers:
            raise YOLODatasetTypeException(f"Unsupported dataset type: {dataset_type}")

        parser = self._parsers[dataset_type]

        try:
            annotations = await parser.parse(annotation_path, class_names)
            return annotations
        except Exception as e:
            logger.error(f"Error parsing annotations {annotation_path}: {str(e)}", exc_info=True)
            raise


# 全局YOLO验证器实例
yolo_validator = YOLOValidator()
