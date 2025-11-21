"""Service interface protocol definitions for type-safe contracts."""
from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    Dict,
    Generic,
    List,
    Optional,
    Protocol,
    TypeVar,
    Union,
    runtime_checkable,
)

# Type variables
T = TypeVar('T')
ModelType = TypeVar('ModelType')
CreateSchemaType = TypeVar('CreateSchemaType')
UpdateSchemaType = TypeVar('UpdateSchemaType')
DatabaseModelType = TypeVar('DatabaseModelType')


class QueryFilter(Protocol):
    """Protocol for objects that can be converted to query dictionaries."""
    def to_dict(self) -> Dict[str, Any]:
        """Convert filter to dictionary representation."""
        ...


@runtime_checkable
class CRUDRepository(Protocol[ModelType, CreateSchemaType, UpdateSchemaType]):
    """Generic asynchronous CRUD repository protocol."""

    async def create(self, obj: CreateSchemaType, **kwargs) -> ModelType:
        """Create a new entity."""
        ...

    async def get_by_id(self, id: str) -> Optional[ModelType]:
        """Retrieve an entity by identifier."""
        ...

    async def get_multi(
        self,
        filters: Optional[QueryFilter] = None,
        skip: int = 0,
        limit: int = 100,
        sort: Optional[Union[str, List[str]]] = None
    ) -> List[ModelType]:
        """Retrieve multiple entities with optional filtering."""
        ...

    async def update(self, id: str, obj: UpdateSchemaType, **kwargs) -> Optional[ModelType]:
        """Update an entity."""
        ...

    async def delete(self, id: str) -> bool:
        """Delete an entity."""
        ...

    async def count(self, filters: Optional[QueryFilter] = None) -> int:
        """Count entities matching the filter."""
        ...


class StorageService(Protocol):
    """Protocol describing object storage capabilities."""

    async def upload_file(
        self,
        bucket_name: str,
        object_name: str,
        file_path: Union[str, Path],
        metadata: Optional[Dict[str, str]] = None
    ) -> str:
        """Upload a file and return object identifier."""
        ...

    async def download_file(
        self,
        bucket_name: str,
        object_name: str,
        file_path: Union[str, Path]
    ) -> bool:
        """Download a file to the given path."""
        ...

    async def delete_file(self, bucket_name: str, object_name: str) -> bool:
        """Delete an object from storage."""
        ...

    async def get_file_url(
        self,
        bucket_name: str,
        object_name: str,
        expires: int = 3600
    ) -> str:
        """Return a signed URL for the object."""
        ...

    async def list_files(
        self,
        bucket_name: str,
        prefix: Optional[str] = None,
        max_keys: int = 1000
    ) -> List[Dict[str, Any]]:
        """List objects within a prefix."""
        ...

    async def file_exists(self, bucket_name: str, object_name: str) -> bool:
        """Return whether an object exists."""
        ...


class YOLOValidatorService(Protocol):
    """Protocol describing YOLO dataset validation services."""

    async def validate_dataset(
        self,
        dataset_path: Union[str, Path],
        dataset_type: str
    ) -> Dict[str, Any]:
        """Validate dataset and return detailed result."""
        ...

    def get_supported_types(self) -> List[str]:
        """Return supported dataset types."""
        ...

    async def validate_single_annotation(
        self,
        annotation_path: Union[str, Path],
        dataset_type: str
    ) -> Dict[str, Any]:
        """Validate a single annotation file."""
        ...


class ProgressTracker(Protocol):
    """Protocol for tracking asynchronous task progress."""

    async def start_task(self, task_id: str, task_type: str, total_steps: int) -> None:
        """Begin tracking a task."""
        ...

    async def update_progress(self, task_id: str, current_step: int, message: str = "") -> None:
        """Update task progress."""
        ...

    async def complete_task(self, task_id: str, success: bool = True, message: str = "") -> None:
        """Mark task completion."""
        ...

    def get_progress(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Fetch progress information for a task."""
        ...

    def get_active_tasks(self) -> List[str]:
        """Return identifiers for active tasks."""
        ...


class CacheService(Protocol):
    """Protocol describing cache storage operations."""

    async def get(self, key: str, default: Any = None) -> Any:
        """Retrieve a cached value."""
        ...

    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None
    ) -> bool:
        """Store a value in cache."""
        ...

    async def delete(self, key: str) -> bool:
        """Delete a cached key."""
        ...

    async def exists(self, key: str) -> bool:
        """Return whether a key exists in cache."""
        ...

    async def clear_pattern(self, pattern: str) -> int:
        """Remove keys matching the supplied pattern."""
        ...


class MessageQueueService(Protocol):
    """Protocol for message queue interactions."""

    async def publish(
        self,
        exchange: str,
        routing_key: str,
        message: Dict[str, Any]
    ) -> bool:
        """Publish a message to an exchange."""
        ...

    async def subscribe(
        self,
        queue: str,
        exchange: str,
        routing_key: str,
        handler: callable
    ) -> None:
        """Subscribe a handler to a queue/exchange."""
        ...

    async def unsubscribe(self, queue: str) -> None:
        """Remove a subscription."""
        ...


class ConfigurationService(Protocol):
    """Protocol for configuration retrieval and persistence."""

    def get(self, key: str, default: Any = None) -> Any:
        """Fetch configuration value."""
        ...

    def set(self, key: str, value: Any) -> bool:
        """Persist configuration value."""
        ...

    def update(self, config_dict: Dict[str, Any]) -> None:
        """Bulk update configuration entries."""
        ...

    def validate_config(self, config_dict: Dict[str, Any]) -> Dict[str, List[str]]:
        """Validate configuration payload."""
        ...


class EventPublisher(Protocol):
    """Protocol for event publishing systems."""

    async def publish_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Publish a single event."""
        ...

    async def publish_batch(self, events: List[Dict[str, Any]]) -> None:
        """Publish a batch of events."""
        ...

    def subscribe(self, event_type: str, handler: callable) -> None:
        """Subscribe a handler to an event type."""
        ...

    def unsubscribe(self, event_type: str, handler: callable) -> None:
        """Remove an event subscription."""
        ...


class ValidationService(Protocol):
    """Protocol for validating files and annotations."""

    async def validate_file(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """Validate a file."""
        ...

    async def validate_image(self, image_path: Union[str, Path]) -> Dict[str, Any]:
        """Validate an image."""
        ...

    async def validate_yaml(self, yaml_path: Union[str, Path]) -> Dict[str, Any]:
        """Validate a YAML configuration."""
        ...

    async def validate_annotations(
        self,
        annotation_path: Union[str, Path],
        dataset_type: str
    ) -> Dict[str, Any]:
        """Validate annotation file payload."""
        ...


class MonitoringService(Protocol):
    """Protocol for metrics/monitoring backends."""

    def record_metric(self, metric_name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """Record a gauge metric."""
        ...

    def start_timer(self, operation_name: str) -> Any:
        """Start a timer context for an operation."""
        ...

    def increment_counter(self, counter_name: str, value: int = 1, tags: Optional[Dict[str, str]] = None) -> None:
        """Increment a counter metric."""
        ...

    def record_histogram(self, histogram_name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """Record a histogram sample."""
        ...

    async def health_check(self) -> Dict[str, Any]:
        """Return health-check information."""
        ...


class EmailService(Protocol):
    """Protocol for sending transactional/bulk email."""

    async def send_email(
        self,
        to: str,
        subject: str,
        body: str,
        html_body: Optional[str] = None,
        attachments: Optional[List[Dict[str, Any]]] = None
    ) -> bool:
        """Send a single email."""
        ...

    async def send_bulk_email(
        self,
        recipients: List[str],
        subject: str,
        body: str,
        html_body: Optional[str] = None
    ) -> Dict[str, List[str]]:
        """Send emails in bulk; return per-recipient status."""
        ...


class TaskSchedulerService(Protocol):
    """Protocol for scheduling recurring/deferred tasks."""

    async def schedule_task(
        self,
        task_id: str,
        task_type: str,
        schedule_expression: str,
        handler: callable,
        kwargs: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Schedule a task for execution."""
        ...

    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a scheduled task."""
        ...

    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Return status for a scheduled task."""
        ...

    def list_tasks(self, task_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """List scheduled tasks, optionally filtered by type."""
        ...


class SearchService(Protocol):
    """Protocol for search-index operations."""

    async def index_document(
        self,
        collection: str,
        document_id: str,
        document: Dict[str, Any]
    ) -> bool:
        """Index or upsert a search document."""
        ...

    async def search(
        self,
        collection: str,
        query: Dict[str, Any],
        filters: Optional[Dict[str, Any]] = None,
        sort: Optional[Dict[str, Any]] = None,
        limit: int = 10,
        offset: int = 0
    ) -> Dict[str, Any]:
        """Execute a search query."""
        ...

    async def delete_index(self, collection: str, document_id: str) -> bool:
        """Delete a document from the index."""
        ...

    async def reindex_collection(self, collection: str) -> bool:
        """Rebuild a collection index."""
        ...


class NotificationService(Protocol):
    """Protocol for sending notifications to users."""

    async def send_notification(
        self,
        user_id: str,
        title: str,
        message: str,
        notification_type: str = "info",
        data: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Send a notification to a user."""
        ...

    async def send_bulk_notification(
        self,
        user_ids: List[str],
        title: str,
        message: str,
        notification_type: str = "info"
    ) -> Dict[str, List[str]]:
        """Send notifications to multiple users."""
        ...

    async def get_user_notifications(
        self,
        user_id: str,
        limit: int = 50,
        unread_only: bool = False
    ) -> List[Dict[str, Any]]:
        """Fetch notifications for a user."""
        ...

    async def mark_as_read(self, notification_id: str) -> bool:
        """Mark a notification as read."""
        ...


class AuditService(Protocol):
    """Protocol for audit logging backends."""

    async def log_action(
        self,
        user_id: str,
        action: str,
        resource: str,
        resource_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        ip_address: Optional[str] = None
    ) -> bool:
        """Log an audited action."""
        ...

    async def get_audit_log(
        self,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """Query audit log entries."""
        ...

    async def export_audit_log(
        self,
        filters: Optional[Dict[str, Any]] = None,
        format: str = "json"
    ) -> str:
        """Export audit logs in the requested format."""
        ...
