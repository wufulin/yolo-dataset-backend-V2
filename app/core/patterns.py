"""设计模式实现 - 工厂模式、策略模式、观察者模式等"""
import asyncio
import logging
import time
from abc import ABC, abstractmethod
from concurrent.futures import Future
from dataclasses import dataclass
from enum import Enum
from typing import (
    Any,
    Dict,
    Generic,
    List,
    Optional,
    Protocol,
    Type,
    TypeVar,
    runtime_checkable,
)

from app.core.exceptions import BusinessLogicException, YOLOException
from app.utils.logger import get_logger

logger: logging.Logger = get_logger(__name__)

# 类型变量
T = TypeVar('T')
ServiceType = TypeVar('ServiceType')


@runtime_checkable
class ServiceFactory(Protocol):
    """服务工厂协议"""
    def create_service(self, service_type: str, **kwargs) -> Any:
        """创建服务实例"""
        ...


class DatasetType(Enum):
    """YOLO数据集类型"""
    DETECT = "detect"
    SEGMENT = "segment"
    POSE = "pose"
    OBB = "obb"
    CLASSIFY = "classify"


class ServiceFactoryBase(ABC):
    """服务工厂基类"""

    @abstractmethod
    def create_service(self, service_type: str, **kwargs) -> Any:
        """创建服务实例"""
        pass

    def get_available_types(self) -> List[str]:
        """获取可用的服务类型"""
        return []


class DatasetServiceFactory(ServiceFactoryBase):
    """数据集服务工厂"""

    def __init__(self):
        self._services: Dict[str, Type] = {}
        self._instances: Dict[str, Any] = {}
        self._register_default_services()

    def _register_default_services(self):
        """注册默认服务类型"""
        # 这里导入具体的服务类，避免循环导入
        try:
            from app.services.dataset_service import DatasetService
            from app.services.image_service import ImageService
            from app.services.minio_service import MinioService

            self._services = {
                "dataset": DatasetService,
                "image": ImageService,
                "minio": MinioService,
                "db": lambda: None,  # 数据库服务是单例，不需要工厂创建
            }
        except ImportError as e:
            logger.warning(f"Could not import services for factory registration: {e}")

    def create_service(self, service_type: str, **kwargs) -> Any:
        """创建服务实例"""
        if service_type not in self._services:
            raise BusinessLogicException(f"Unknown service type: {service_type}")

        # 数据库服务特殊处理（单例模式）
        if service_type == "db":
            from app.services.db_service import db_service
            return db_service

        # 检查是否已经有缓存的实例（如果支持单例）
        if service_type in self._instances:
            return self._instances[service_type]

        # 创建新实例
        service_class = self._services[service_type]
        try:
            instance = service_class(**kwargs)

            # 缓存某些服务类型的实例
            if service_type in ["minio"]:
                self._instances[service_type] = instance

            return instance
        except Exception as e:
            logger.error(f"Failed to create service {service_type}: {e}")
            raise BusinessLogicException(f"Failed to create service {service_type}: {str(e)}")

    def get_available_types(self) -> List[str]:
        """获取可用的服务类型"""
        return list(self._services.keys())

    def register_service(self, service_type: str, service_class: Type):
        """注册新的服务类型"""
        self._services[service_type] = service_class
        logger.info(f"Registered service type: {service_type}")


class DatasetValidationStrategy(ABC, Generic[T]):
    """数据集验证策略基类"""

    @abstractmethod
    async def validate(self, dataset_path: str, **kwargs) -> T:
        """验证数据集"""
        pass

    @abstractmethod
    def get_supported_types(self) -> List[str]:
        """获取支持的数据集类型"""
        pass


class DetectDatasetValidator(DatasetValidationStrategy[bool]):
    """检测数据集验证策略"""

    async def validate(self, dataset_path: str, **kwargs) -> bool:
        """验证检测数据集"""
        try:
            from app.utils.yolo_validator import yolo_validator
            return yolo_validator.validate_dataset(dataset_path, "detect")[0]
        except Exception as e:
            logger.error(f"Detect dataset validation failed: {e}")
            return False

    def get_supported_types(self) -> List[str]:
        return [DatasetType.DETECT.value]


class SegmentDatasetValidator(DatasetValidationStrategy[bool]):
    """分割数据集验证策略"""

    async def validate(self, dataset_path: str, **kwargs) -> bool:
        """验证分割数据集"""
        try:
            from app.utils.yolo_validator import yolo_validator
            return yolo_validator.validate_dataset(dataset_path, "segment")[0]
        except Exception as e:
            logger.error(f"Segment dataset validation failed: {e}")
            return False

    def get_supported_types(self) -> List[str]:
        return [DatasetType.SEGMENT.value]


class PoseDatasetValidator(DatasetValidationStrategy[bool]):
    """姿态估计数据集验证策略"""

    async def validate(self, dataset_path: str, **kwargs) -> bool:
        """验证姿态估计数据集"""
        try:
            from app.utils.yolo_validator import yolo_validator
            return yolo_validator.validate_dataset(dataset_path, "pose")[0]
        except Exception as e:
            logger.error(f"Pose dataset validation failed: {e}")
            return False

    def get_supported_types(self) -> List[str]:
        return [DatasetType.POSE.value]


class OBBDatasetValidator(DatasetValidationStrategy[bool]):
    """定向边界框数据集验证策略"""

    async def validate(self, dataset_path: str, **kwargs) -> bool:
        """验证定向边界框数据集"""
        try:
            from app.utils.yolo_validator import yolo_validator
            return yolo_validator.validate_dataset(dataset_path, "obb")[0]
        except Exception as e:
            logger.error(f"OBB dataset validation failed: {e}")
            return False

    def get_supported_types(self) -> List[str]:
        return [DatasetType.OBB.value]


class ClassifyDatasetValidator(DatasetValidationStrategy[bool]):
    """分类数据集验证策略"""

    async def validate(self, dataset_path: str, **kwargs) -> bool:
        """验证分类数据集"""
        try:
            from app.utils.yolo_validator import yolo_validator
            return yolo_validator.validate_dataset(dataset_path, "classify")[0]
        except Exception as e:
            logger.error(f"Classify dataset validation failed: {e}")
            return False

    def get_supported_types(self) -> List[str]:
        return [DatasetType.CLASSIFY.value]


class DatasetValidatorContext:
    """数据集验证上下文 - 策略模式的上下文类"""

    def __init__(self):
        self._strategies: Dict[str, DatasetValidationStrategy] = {}
        self._register_default_strategies()

    def _register_default_strategies(self):
        """注册默认验证策略"""
        self._strategies = {
            DatasetType.DETECT.value: DetectDatasetValidator(),
            DatasetType.SEGMENT.value: SegmentDatasetValidator(),
            DatasetType.POSE.value: PoseDatasetValidator(),
            DatasetType.OBB.value: OBBDatasetValidator(),
            DatasetType.CLASSIFY.value: ClassifyDatasetValidator(),
        }

    def validate_dataset(self, dataset_path: str, dataset_type: str) -> bool:
        """验证数据集"""
        if dataset_type not in self._strategies:
            logger.error(f"No validator found for dataset type: {dataset_type}")
            return False

        try:
            strategy = self._strategies[dataset_type]
            # 同步调用async方法
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(strategy.validate(dataset_path))
                return result
            finally:
                loop.close()
        except Exception as e:
            logger.error(f"Dataset validation failed for type {dataset_type}: {e}")
            return False

    def register_strategy(self, dataset_type: str, strategy: DatasetValidationStrategy):
        """注册新的验证策略"""
        self._strategies[dataset_type] = strategy
        logger.info(f"Registered validation strategy for type: {dataset_type}")

    def get_available_types(self) -> List[str]:
        """获取支持的验证类型"""
        return list(self._strategies.keys())


class Observer(ABC):
    """观察者基类"""

    @abstractmethod
    async def update(self, event_type: str, data: Dict[str, Any]):
        """更新方法"""
        pass


class Observable:
    """可观察对象"""

    def __init__(self):
        self._observers: Dict[str, List[Observer]] = {}

    def attach(self, event_type: str, observer: Observer):
        """附加观察者"""
        if event_type not in self._observers:
            self._observers[event_type] = []
        self._observers[event_type].append(observer)
        logger.debug(f"Attached observer for event: {event_type}")

    def detach(self, event_type: str, observer: Observer):
        """分离观察者"""
        if event_type in self._observers:
            try:
                self._observers[event_type].remove(observer)
                logger.debug(f"Detached observer for event: {event_type}")
            except ValueError:
                logger.warning(f"Observer not found for event: {event_type}")

    async def notify(self, event_type: str, data: Dict[str, Any]):
        """通知观察者"""
        if event_type in self._observers:
            tasks = []
            for observer in self._observers[event_type]:
                try:
                    if asyncio.iscoroutinefunction(observer.update):
                        task = asyncio.create_task(observer.update(event_type, data))
                    else:
                        task = asyncio.create_task(self._sync_to_async(observer.update, event_type, data))
                    tasks.append(task)
                except Exception as e:
                    logger.error(f"Failed to notify observer for {event_type}: {e}")

            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

    def _sync_to_async(self, func, *args):
        """同步函数转异步"""
        def wrapper():
            return func(*args)

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(wrapper())
        finally:
            loop.close()


class UploadProgressObserver(Observer):
    """上传进度观察者"""

    def __init__(self, session_id: str):
        self.session_id = session_id

    async def update(self, event_type: str, data: Dict[str, Any]):
        """更新上传进度"""
        if event_type == "upload_progress":
            progress = data.get("progress", 0)
            filename = data.get("filename", "unknown")
            speed = data.get("speed_mbps", 0)

            logger.info(
                f"Upload progress [{self.session_id}]: {filename} - "
                f"{progress:.1f}% ({speed:.2f} MB/s)"
            )

            # 这里可以更新数据库或Redis中的进度信息
            # await self._update_progress_in_db(progress, speed)


class FileValidationObserver(Observer):
    """文件验证观察者"""

    async def update(self, event_type: str, data: Dict[str, Any]):
        """更新文件验证状态"""
        if event_type == "file_validation":
            filename = data.get("filename", "unknown")
            is_valid = data.get("is_valid", False)
            error = data.get("error")

            if is_valid:
                logger.info(f"File validation passed: {filename}")
            else:
                logger.error(f"File validation failed: {filename} - {error}")


class DatasetCreationObserver(Observer):
    """数据集创建观察者"""

    async def update(self, event_type: str, data: Dict[str, Any]):
        """更新数据集创建状态"""
        if event_type == "dataset_creation":
            dataset_name = data.get("dataset_name", "unknown")
            status = data.get("status", "unknown")

            logger.info(f"Dataset creation event [{dataset_name}]: {status}")


@dataclass
class UploadEvent:
    """上传事件数据类"""
    session_id: str
    filename: str
    file_size: int
    uploaded_bytes: int
    progress: float
    speed_mbps: float
    timestamp: float


class UploadProgressTracker(Observable):
    """上传进度追踪器"""

    def __init__(self):
        super().__init__()
        self._active_uploads: Dict[str, UploadEvent] = {}

    async def start_upload(self, session_id: str, filename: str, file_size: int):
        """开始上传追踪"""
        upload_event = UploadEvent(
            session_id=session_id,
            filename=filename,
            file_size=file_size,
            uploaded_bytes=0,
            progress=0.0,
            speed_mbps=0.0,
            timestamp=time.time()
        )

        self._active_uploads[session_id] = upload_event

        await self.notify("upload_started", {
            "session_id": session_id,
            "filename": filename,
            "file_size": file_size
        })

    async def update_progress(self, session_id: str, uploaded_bytes: int, speed_mbps: float):
        """更新上传进度"""
        if session_id not in self._active_uploads:
            logger.warning(f"Unknown upload session: {session_id}")
            return

        event = self._active_uploads[session_id]
        event.uploaded_bytes = uploaded_bytes
        event.progress = (uploaded_bytes / event.file_size) * 100 if event.file_size > 0 else 0
        event.speed_mbps = speed_mbps
        event.timestamp = time.time()

        # 通知观察者
        await self.notify("upload_progress", {
            "session_id": session_id,
            "filename": event.filename,
            "progress": event.progress,
            "speed_mbps": event.speed_mbps,
            "uploaded_bytes": uploaded_bytes,
            "total_bytes": event.file_size
        })

    async def complete_upload(self, session_id: str):
        """完成上传"""
        if session_id not in self._active_uploads:
            return

        event = self._active_uploads[session_id]
        await self.notify("upload_completed", {
            "session_id": session_id,
            "filename": event.filename,
            "file_size": event.file_size,
            "total_time": time.time() - event.timestamp
        })

        del self._active_uploads[session_id]

    def get_active_uploads(self) -> List[UploadEvent]:
        """获取活跃的上传任务"""
        return list(self._active_uploads.values())

