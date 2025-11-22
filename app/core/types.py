"""Type aliases and protocols used throughout the project."""
from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    Generic,
    List,
    Literal,
    Optional,
    Protocol,
    TypedDict,
    TypeVar,
    Union,
    runtime_checkable,
)

# Base type aliases
ModelType = TypeVar('ModelType')
CreateSchemaType = TypeVar('CreateSchemaType')
UpdateSchemaType = TypeVar('UpdateSchemaType')
ResponseType = TypeVar('ResponseType')

# Dataset-related literals
YOLODatasetType = Literal['detect', 'segment', 'pose', 'obb', 'classify']

DatasetStatus = Literal['active', 'processing', 'validating', 'error', 'deleted']

ImageFormat = Literal['JPEG', 'JPG', 'PNG', 'BMP', 'TIFF', 'WEBP']

# File handling literals
FileType = Literal['image', 'annotation', 'config', 'archive']


# API response types
class APIResponse(TypedDict):
    """Standard API response envelope."""
    success: bool
    message: str
    data: Optional[Any]
    timestamp: datetime
    request_id: Optional[str]


class PaginationResponse(TypedDict, Generic[ResponseType]):
    """Paginated API response envelope."""
    items: List[ResponseType]
    page: int
    page_size: int
    total: int
    total_pages: int
    has_next: bool
    has_prev: bool


class SuccessResponse(TypedDict, Generic[ResponseType]):
    """Success response payload."""
    success: Literal[True]
    data: ResponseType
    message: str


class ErrorResponse(TypedDict):
    """Error response payload."""
    success: Literal[False]
    error_code: str
    message: str
    details: Optional[Dict[str, Any]]


class ValidationErrorResponse(TypedDict):
    """Validation error payload."""
    success: Literal[False]
    error_code: Literal['VALIDATION_ERROR']
    message: str
    field_errors: Dict[str, List[str]]


# Database model configuration types
class MongoDBConfig(TypedDict):
    """MongoDB configuration snapshot."""
    url: str
    db_name: str
    max_pool_size: int
    max_idle_time_ms: int
    retry_writes: bool


class StorageConfig(TypedDict):
    """Object storage configuration snapshot."""
    endpoint: str
    access_key: str
    secret_key: str
    bucket_name: str
    secure: bool


class CacheConfig(TypedDict):
    """Cache/Redis configuration snapshot."""
    host: str
    port: int
    db: int
    password: Optional[str]
    max_connections: int


# File processing types
class FileInfo(TypedDict):
    """Describes a file stored on disk."""
    path: str
    name: str
    size: int
    mime_type: str
    checksum: str
    created_at: datetime
    modified_at: datetime


class FileValidationResult(TypedDict):
    """Result of validating a file."""
    is_valid: bool
    file_type: FileType
    format: Optional[str]
    size: Optional[int]
    error_message: Optional[str]
    metadata: Optional[Dict[str, Any]]


class UploadProgress(TypedDict):
    """Upload progress metadata."""
    session_id: str
    filename: str
    file_size: int
    uploaded_bytes: int
    progress_percentage: float
    speed_mbps: float
    eta_seconds: Optional[float]


# YOLO-related types
class YOLOAnnotation(TypedDict):
    """Normalized YOLO annotation entry."""
    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float
    confidence: Optional[float]


class YOLODatasetConfig(TypedDict):
    """YOLO dataset configuration file."""
    train: str
    val: str
    test: Optional[str]
    nc: int
    names: List[str]


class DatasetValidationResult(TypedDict):
    """Dataset validation result details."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    stats: Optional[Dict[str, Any]]


class YOLODatasetStats(TypedDict):
    """Dataset statistics summary."""
    total_images: int
    total_annotations: int
    class_distribution: Dict[str, int]
    image_sizes: Dict[str, int]
    avg_objects_per_image: float


# Event system types
class EventData(TypedDict):
    """Payload delivered to event handlers."""
    event_type: str
    event_id: str
    timestamp: datetime
    source: str
    data: Dict[str, Any]


class EventHandler(Protocol):
    """Protocol for async event handlers."""

    def handle(self, event: EventData) -> Awaitable[None]:
        """Handle an incoming event."""
        ...


class EventPublisher(Protocol):
    """Protocol for event publishing components."""

    async def publish(self, event: EventData) -> None:
        """Publish a single event."""
        ...

    async def subscribe(self, event_type: str, handler: EventHandler) -> None:
        """Subscribe a handler to an event type."""
        ...

    async def unsubscribe(self, event_type: str, handler: EventHandler) -> None:
        """Remove subscription for an event type."""
        ...


# Monitoring and metrics types
class MetricData(TypedDict):
    """Generic metric datapoint."""
    name: str
    value: float
    timestamp: datetime
    tags: Optional[Dict[str, str]]


class HealthCheckResult(TypedDict):
    """Health-check structure describing service status."""
    status: Literal['healthy', 'unhealthy', 'degraded']
    service: str
    timestamp: datetime
    details: Optional[Dict[str, Any]]
    dependencies: Optional[Dict[str, str]]


# Configuration management types
class ConfigurationSchema(TypedDict):
    """Schema definition for configurable items."""
    key: str
    type: str
    required: bool
    default: Optional[Any]
    description: str
    validation_rules: Optional[Dict[str, Any]]


class ConfigurationGroup(TypedDict):
    """Grouping of related configuration entries."""
    name: str
    description: str
    configs: List[ConfigurationSchema]


# User and permission types
class UserInfo(TypedDict):
    """User profile snapshot."""
    user_id: str
    username: str
    email: Optional[str]
    roles: List[str]
    permissions: List[str]
    created_at: datetime
    last_login: Optional[datetime]


class Permission(TypedDict):
    """Permission definition."""
    resource: str
    action: str
    scope: Optional[str]


# Batch processing types
class BatchJob(TypedDict):
    """Batch-processing job metadata."""
    job_id: str
    job_type: str
    status: Literal['pending', 'running', 'completed', 'failed']
    progress: float
    total_steps: int
    current_step: int
    created_at: datetime
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    error_message: Optional[str]
    result: Optional[Dict[str, Any]]


class BatchJobHandler(Protocol):
    """Protocol for batch job handlers."""

    def prepare_job(self, job: BatchJob) -> Awaitable[None]:
        """Prepare a job before execution."""
        ...

    def execute_step(self, step: int, job: BatchJob) -> Awaitable[Dict[str, Any]]:
        """Execute one step of the job."""
        ...

    def cleanup_job(self, job: BatchJob, success: bool) -> Awaitable[None]:
        """Clean up resources after job completion."""
        ...


# Cache-related types
class CacheEntry(TypedDict):
    """Metadata for cached entries."""
    key: str
    value: Any
    ttl: Optional[int]
    created_at: datetime
    access_count: int
    last_accessed: datetime


class CacheStrategy(Protocol):
    """Protocol describing cache strategy hooks."""

    def get(self, key: str) -> Optional[Any]:
        """Fetch a cached value."""
        ...

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Store a cached value."""
        ...

    def delete(self, key: str) -> bool:
        """Remove a cached value."""
        ...

    def clear(self) -> bool:
        """Clear all cache entries."""
        ...

    def exists(self, key: str) -> bool:
        """Return whether the key exists."""
        ...


# Search-related types
class SearchQuery(TypedDict):
    """Represents a user search query."""
    query: str
    filters: Optional[Dict[str, Any]]
    sort: Optional[Dict[str, Literal['asc', 'desc']]]
    page: int
    page_size: int


class SearchResult(TypedDict, Generic[ResponseType]):
    """Represents a search result set."""
    items: List[ResponseType]
    total: int
    page: int
    page_size: int
    took_ms: float


class SearchIndex(Protocol):
    """Protocol describing search index operations."""

    async def index_document(self, doc_id: str, document: Dict[str, Any]) -> bool:
        """Index or update a document."""
        ...

    async def search(self, query: SearchQuery) -> SearchResult[Dict[str, Any]]:
        """Search documents."""
        ...

    async def delete_document(self, doc_id: str) -> bool:
        """Delete a document from the index."""
        ...

    async def rebuild_index(self) -> bool:
        """Rebuild the entire index."""
        ...


# Notification system types
class Notification(TypedDict):
    """Notification payload stored for a user."""
    notification_id: str
    user_id: str
    title: str
    message: str
    notification_type: Literal['info', 'warning', 'error', 'success']
    created_at: datetime
    read_at: Optional[datetime]
    data: Optional[Dict[str, Any]]


class NotificationChannel(Protocol):
    """Protocol for notification delivery channels."""

    async def send(self, notification: Notification) -> bool:
        """Send a single notification."""
        ...

    async def send_bulk(self, notifications: List[Notification]) -> Dict[str, List[str]]:
        """Send notifications in bulk."""
        ...


# Audit log types
class AuditLog(TypedDict):
    """Audit log entry."""
    log_id: str
    user_id: str
    action: str
    resource: str
    resource_id: Optional[str]
    details: Optional[Dict[str, Any]]
    ip_address: Optional[str]
    user_agent: Optional[str]
    timestamp: datetime


# Analytics types
class UsageStatistics(TypedDict):
    """Usage statistic datapoint."""
    metric_name: str
    value: float
    timestamp: datetime
    dimensions: Optional[Dict[str, str]]


class PerformanceMetrics(TypedDict):
    """Performance measurement datapoint."""
    operation: str
    duration_ms: float
    success: bool
    timestamp: datetime
    details: Optional[Dict[str, Any]]


# Error-handling types
class ErrorContext(TypedDict):
    """Context associated with an error."""
    error_id: str
    error_code: str
    message: str
    category: str
    severity: Literal['low', 'medium', 'high', 'critical']
    timestamp: datetime
    request_id: Optional[str]
    user_id: Optional[str]
    details: Optional[Dict[str, Any]]


class ErrorRecoveryStrategy(Protocol):
    """Protocol for defining retry/recovery strategies."""

    def should_retry(self, error: Exception, attempt: int) -> bool:
        """Return whether we should retry."""
        ...

    def get_retry_delay(self, attempt: int) -> float:
        """Return delay before the next retry."""
        ...

    def handle_failure(self, error: Exception, context: ErrorContext) -> Any:
        """Handle terminal failure."""
        ...


# Version/compatibility types
class APIVersion(TypedDict):
    """API version metadata."""
    version: str
    deprecated: bool
    sunset_date: Optional[datetime]
    migration_guide: Optional[str]


class CompatibilityMatrix(TypedDict):
    """Compatibility mapping between API and client versions."""
    api_version: str
    client_version: str
    compatible: bool
    features: List[str]


# Helper utilities
def create_success_response(data: ResponseType, message: str = "Success") -> SuccessResponse[ResponseType]:
    """Create a success response payload."""
    return {
        "success": True,
        "data": data,
        "message": message
    }


def create_error_response(error_code: str, message: str, details: Optional[Dict[str, Any]] = None) -> ErrorResponse:
    """Create an error response payload."""
    return {
        "success": False,
        "error_code": error_code,
        "message": message,
        "details": details
    }


def create_pagination_response(
    items: List[ResponseType],
    page: int,
    page_size: int,
    total: int
) -> PaginationResponse[ResponseType]:
    """Create a pagination response payload."""
    total_pages = (total + page_size - 1) // page_size

    return {
        "items": items,
        "page": page,
        "page_size": page_size,
        "total": total,
        "total_pages": total_pages,
        "has_next": page < total_pages,
        "has_prev": page > 1
    }
