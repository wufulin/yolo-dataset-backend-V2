"""Design-pattern utilities: factories, strategies, observers, and helpers."""
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

# Type variables for generics
T = TypeVar('T')
ServiceType = TypeVar('ServiceType')


@runtime_checkable
class ServiceFactory(Protocol):
    """Protocol for service factories."""
    def create_service(self, service_type: str, **kwargs) -> Any:
        """Create a service instance."""
        ...


class DatasetType(Enum):
    """YOLO dataset type enumeration."""
    DETECT = "detect"
    SEGMENT = "segment"
    POSE = "pose"
    OBB = "obb"
    CLASSIFY = "classify"


class ServiceFactoryBase(ABC):
    """Abstract base class for service factories."""

    @abstractmethod
    def create_service(self, service_type: str, **kwargs) -> Any:
        """Create a concrete service instance."""
        pass

    def get_available_types(self) -> List[str]:
        """Return registered service type names."""
        return []


class DatasetServiceFactory(ServiceFactoryBase):
    """Factory responsible for dataset-related services."""

    def __init__(self):
        self._services: Dict[str, Type] = {}
        self._instances: Dict[str, Any] = {}
        self._register_default_services()

    def _register_default_services(self):
        """Register built-in services while avoiding circular imports."""
        try:
            from app.services.dataset_service import DatasetService
            from app.services.image_service import ImageService
            from app.services.minio_service import MinioService

            self._services = {
                "dataset": DatasetService,
                "image": ImageService,
                "minio": MinioService,
                "db": lambda: None,  # Database service is a singleton elsewhere
            }
        except ImportError as e:
            logger.warning(f"Could not import services for factory registration: {e}")

    def create_service(self, service_type: str, **kwargs) -> Any:
        """Instantiate or retrieve a service by type."""
        if service_type not in self._services:
            raise BusinessLogicException(f"Unknown service type: {service_type}")

        # Shortcut for singleton DB service
        if service_type == "db":
            from app.services.db_service import db_service
            return db_service

        # Return cached singleton when available
        if service_type in self._instances:
            return self._instances[service_type]

        # Create fresh instance
        service_class = self._services[service_type]
        try:
            instance = service_class(**kwargs)

            # Cache specific service types for reuse
            if service_type in ["minio"]:
                self._instances[service_type] = instance

            return instance
        except Exception as e:
            logger.error(f"Failed to create service {service_type}: {e}")
            raise BusinessLogicException(f"Failed to create service {service_type}: {str(e)}")

    def get_available_types(self) -> List[str]:
        """Return list of supported service types."""
        return list(self._services.keys())

    def register_service(self, service_type: str, service_class: Type):
        """Register additional service types at runtime."""
        self._services[service_type] = service_class
        logger.info(f"Registered service type: {service_type}")


class DatasetValidationStrategy(ABC, Generic[T]):
    """Base class for dataset validation strategies."""

    @abstractmethod
    async def validate(self, dataset_path: str, **kwargs) -> T:
        """Validate a dataset located at dataset_path."""
        pass

    @abstractmethod
    def get_supported_types(self) -> List[str]:
        """Return the dataset types supported by this strategy."""
        pass


class DetectDatasetValidator(DatasetValidationStrategy[bool]):
    """Validation strategy for detection datasets."""

    async def validate(self, dataset_path: str, **kwargs) -> bool:
        """Validate detection dataset structure."""
        try:
            from app.utils.yolo_validator import yolo_validator
            return yolo_validator.validate_dataset(dataset_path, "detect")[0]
        except Exception as e:
            logger.error(f"Detect dataset validation failed: {e}")
            return False

    def get_supported_types(self) -> List[str]:
        return [DatasetType.DETECT.value]


class SegmentDatasetValidator(DatasetValidationStrategy[bool]):
    """Validation strategy for segmentation datasets."""

    async def validate(self, dataset_path: str, **kwargs) -> bool:
        """Validate segmentation dataset structure."""
        try:
            from app.utils.yolo_validator import yolo_validator
            return yolo_validator.validate_dataset(dataset_path, "segment")[0]
        except Exception as e:
            logger.error(f"Segment dataset validation failed: {e}")
            return False

    def get_supported_types(self) -> List[str]:
        return [DatasetType.SEGMENT.value]


class PoseDatasetValidator(DatasetValidationStrategy[bool]):
    """Validation strategy for pose datasets."""

    async def validate(self, dataset_path: str, **kwargs) -> bool:
        """Validate pose dataset structure."""
        try:
            from app.utils.yolo_validator import yolo_validator
            return yolo_validator.validate_dataset(dataset_path, "pose")[0]
        except Exception as e:
            logger.error(f"Pose dataset validation failed: {e}")
            return False

    def get_supported_types(self) -> List[str]:
        return [DatasetType.POSE.value]


class OBBDatasetValidator(DatasetValidationStrategy[bool]):
    """Validation strategy for oriented bounding box datasets."""

    async def validate(self, dataset_path: str, **kwargs) -> bool:
        """Validate oriented bounding box dataset structure."""
        try:
            from app.utils.yolo_validator import yolo_validator
            return yolo_validator.validate_dataset(dataset_path, "obb")[0]
        except Exception as e:
            logger.error(f"OBB dataset validation failed: {e}")
            return False

    def get_supported_types(self) -> List[str]:
        return [DatasetType.OBB.value]


class ClassifyDatasetValidator(DatasetValidationStrategy[bool]):
    """Validation strategy for classification datasets."""

    async def validate(self, dataset_path: str, **kwargs) -> bool:
        """Validate classification dataset structure."""
        try:
            from app.utils.yolo_validator import yolo_validator
            return yolo_validator.validate_dataset(dataset_path, "classify")[0]
        except Exception as e:
            logger.error(f"Classify dataset validation failed: {e}")
            return False

    def get_supported_types(self) -> List[str]:
        return [DatasetType.CLASSIFY.value]


class DatasetValidatorContext:
    """Strategy context responsible for dataset validation orchestration."""

    def __init__(self):
        self._strategies: Dict[str, DatasetValidationStrategy] = {}
        self._register_default_strategies()

    def _register_default_strategies(self):
        """Register default strategies."""
        self._strategies = {
            DatasetType.DETECT.value: DetectDatasetValidator(),
            DatasetType.SEGMENT.value: SegmentDatasetValidator(),
            DatasetType.POSE.value: PoseDatasetValidator(),
            DatasetType.OBB.value: OBBDatasetValidator(),
            DatasetType.CLASSIFY.value: ClassifyDatasetValidator(),
        }

    def validate_dataset(self, dataset_path: str, dataset_type: str) -> bool:
        """Validate dataset using the registered strategy."""
        if dataset_type not in self._strategies:
            logger.error(f"No validator found for dataset type: {dataset_type}")
            return False

        try:
            strategy = self._strategies[dataset_type]
            # Execute async validation in a temporary loop
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
        """Register a new dataset validation strategy."""
        self._strategies[dataset_type] = strategy
        logger.info(f"Registered validation strategy for type: {dataset_type}")

    def get_available_types(self) -> List[str]:
        """Return supported dataset type keys."""
        return list(self._strategies.keys())


class Observer(ABC):
    """Base class for observers."""

    @abstractmethod
    async def update(self, event_type: str, data: Dict[str, Any]):
        """Consume events emitted by Observable."""
        pass


class Observable:
    """Subject implementation that manages observer lifecycles."""

    def __init__(self):
        self._observers: Dict[str, List[Observer]] = {}

    def attach(self, event_type: str, observer: Observer):
        """Subscribe an observer to a given event type."""
        if event_type not in self._observers:
            self._observers[event_type] = []
        self._observers[event_type].append(observer)
        logger.debug(f"Attached observer for event: {event_type}")

    def detach(self, event_type: str, observer: Observer):
        """Unsubscribe an observer from a given event type."""
        if event_type in self._observers:
            try:
                self._observers[event_type].remove(observer)
                logger.debug(f"Detached observer for event: {event_type}")
            except ValueError:
                logger.warning(f"Observer not found for event: {event_type}")

    async def notify(self, event_type: str, data: Dict[str, Any]):
        """Notify observers asynchronously."""
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
        """Helper to run sync callbacks in the background."""
        def wrapper():
            return func(*args)

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(wrapper())
        finally:
            loop.close()


class UploadProgressObserver(Observer):
    """Observer that logs upload progress updates."""

    def __init__(self, session_id: str):
        self.session_id = session_id

    async def update(self, event_type: str, data: Dict[str, Any]):
        """Handle upload progress events."""
        if event_type == "upload_progress":
            progress = data.get("progress", 0)
            filename = data.get("filename", "unknown")
            speed = data.get("speed_mbps", 0)

            logger.info(
                f"Upload progress [{self.session_id}]: {filename} - "
                f"{progress:.1f}% ({speed:.2f} MB/s)"
            )

            # Hook for persisting progress to Redis or database if desired
            # await self._update_progress_in_db(progress, speed)


class FileValidationObserver(Observer):
    """Observer that tracks file validation events."""

    async def update(self, event_type: str, data: Dict[str, Any]):
        """Handle file validation events."""
        if event_type == "file_validation":
            filename = data.get("filename", "unknown")
            is_valid = data.get("is_valid", False)
            error = data.get("error")

            if is_valid:
                logger.info(f"File validation passed: {filename}")
            else:
                logger.error(f"File validation failed: {filename} - {error}")


class DatasetCreationObserver(Observer):
    """Observer for dataset creation lifecycle."""

    async def update(self, event_type: str, data: Dict[str, Any]):
        """Handle dataset creation events."""
        if event_type == "dataset_creation":
            dataset_name = data.get("dataset_name", "unknown")
            status = data.get("status", "unknown")

            logger.info(f"Dataset creation event [{dataset_name}]: {status}")


@dataclass
class UploadEvent:
    """Upload event payload definition."""
    session_id: str
    filename: str
    file_size: int
    uploaded_bytes: int
    progress: float
    speed_mbps: float
    timestamp: float


class UploadProgressTracker(Observable):
    """Observable specialized for upload progress reporting."""

    def __init__(self):
        super().__init__()
        self._active_uploads: Dict[str, UploadEvent] = {}

    async def start_upload(self, session_id: str, filename: str, file_size: int):
        """Start tracking an upload session."""
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
        """Record upload progress and notify observers."""
        if session_id not in self._active_uploads:
            logger.warning(f"Unknown upload session: {session_id}")
            return

        event = self._active_uploads[session_id]
        event.uploaded_bytes = uploaded_bytes
        event.progress = (uploaded_bytes / event.file_size) * 100 if event.file_size > 0 else 0
        event.speed_mbps = speed_mbps
        event.timestamp = time.time()

        # Notify observers with updated metrics
        await self.notify("upload_progress", {
            "session_id": session_id,
            "filename": event.filename,
            "progress": event.progress,
            "speed_mbps": event.speed_mbps,
            "uploaded_bytes": uploaded_bytes,
            "total_bytes": event.file_size
        })

    async def complete_upload(self, session_id: str):
        """Finish tracking an upload session."""
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
        """Return active uploads as a list of events."""
        return list(self._active_uploads.values())

