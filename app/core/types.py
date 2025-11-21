"""项目的类型定义和协议 - 提供完整的类型支持"""
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

# 基础类型定义
ModelType = TypeVar('ModelType')
CreateSchemaType = TypeVar('CreateSchemaType')
UpdateSchemaType = TypeVar('UpdateSchemaType')
ResponseType = TypeVar('ResponseType')

# 数据集相关类型
YOLODatasetType = Literal['detect', 'segment', 'pose', 'obb', 'classify']

DatasetStatus = Literal['active', 'processing', 'validating', 'error', 'deleted']

ImageFormat = Literal['JPEG', 'JPG', 'PNG', 'BMP', 'TIFF', 'WEBP']

# 文件处理类型
FileType = Literal['image', 'annotation', 'config', 'archive']

# API响应类型
class APIResponse(TypedDict):
    """标准API响应格式"""
    success: bool
    message: str
    data: Optional[Any]
    timestamp: datetime
    request_id: Optional[str]

class PaginationResponse(TypedDict, Generic[ResponseType]):
    """分页响应格式"""
    items: List[ResponseType]
    page: int
    page_size: int
    total: int
    total_pages: int
    has_next: bool
    has_prev: bool

class SuccessResponse(TypedDict, Generic[ResponseType]):
    """成功响应格式"""
    success: Literal[True]
    data: ResponseType
    message: str

class ErrorResponse(TypedDict):
    """错误响应格式"""
    success: Literal[False]
    error_code: str
    message: str
    details: Optional[Dict[str, Any]]

class ValidationErrorResponse(TypedDict):
    """验证错误响应格式"""
    success: Literal[False]
    error_code: Literal['VALIDATION_ERROR']
    message: str
    field_errors: Dict[str, List[str]]

# 数据库模型类型
class MongoDBConfig(TypedDict):
    """MongoDB配置"""
    url: str
    db_name: str
    max_pool_size: int
    max_idle_time_ms: int
    retry_writes: bool

class StorageConfig(TypedDict):
    """存储配置"""
    endpoint: str
    access_key: str
    secret_key: str
    bucket_name: str
    secure: bool

class CacheConfig(TypedDict):
    """缓存配置"""
    host: str
    port: int
    db: int
    password: Optional[str]
    max_connections: int

# 文件处理类型
class FileInfo(TypedDict):
    """文件信息"""
    path: str
    name: str
    size: int
    mime_type: str
    checksum: str
    created_at: datetime
    modified_at: datetime

class FileValidationResult(TypedDict):
    """文件验证结果"""
    is_valid: bool
    file_type: FileType
    format: Optional[str]
    size: Optional[int]
    error_message: Optional[str]
    metadata: Optional[Dict[str, Any]]

class UploadProgress(TypedDict):
    """上传进度"""
    session_id: str
    filename: str
    file_size: int
    uploaded_bytes: int
    progress_percentage: float
    speed_mbps: float
    eta_seconds: Optional[float]

# YOLO相关类型
class YOLOAnnotation(TypedDict):
    """YOLO标注格式"""
    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float
    confidence: Optional[float]

class YOLODatasetConfig(TypedDict):
    """YOLO数据集配置"""
    train: str
    val: str
    test: Optional[str]
    nc: int
    names: List[str]

class DatasetValidationResult(TypedDict):
    """数据集验证结果"""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    stats: Optional[Dict[str, Any]]

class YOLODatasetStats(TypedDict):
    """数据集统计信息"""
    total_images: int
    total_annotations: int
    class_distribution: Dict[str, int]
    image_sizes: Dict[str, int]
    avg_objects_per_image: float

# 事件系统类型
class EventData(TypedDict):
    """事件数据"""
    event_type: str
    event_id: str
    timestamp: datetime
    source: str
    data: Dict[str, Any]

class EventHandler(Protocol):
    """事件处理器协议"""
    def handle(self, event: EventData) -> Awaitable[None]:
        """处理事件"""
        ...

class EventPublisher(Protocol):
    """事件发布器协议"""
    async def publish(self, event: EventData) -> None:
        """发布事件"""
        ...

    async def subscribe(self, event_type: str, handler: EventHandler) -> None:
        """订阅事件"""
        ...

    async def unsubscribe(self, event_type: str, handler: EventHandler) -> None:
        """取消订阅"""
        ...

# 监控和指标类型
class MetricData(TypedDict):
    """指标数据"""
    name: str
    value: float
    timestamp: datetime
    tags: Optional[Dict[str, str]]

class HealthCheckResult(TypedDict):
    """健康检查结果"""
    status: Literal['healthy', 'unhealthy', 'degraded']
    service: str
    timestamp: datetime
    details: Optional[Dict[str, Any]]
    dependencies: Optional[Dict[str, str]]

# 配置管理类型
class ConfigurationSchema(TypedDict):
    """配置模式"""
    key: str
    type: str
    required: bool
    default: Optional[Any]
    description: str
    validation_rules: Optional[Dict[str, Any]]

class ConfigurationGroup(TypedDict):
    """配置组"""
    name: str
    description: str
    configs: List[ConfigurationSchema]

# 用户和权限类型
class UserInfo(TypedDict):
    """用户信息"""
    user_id: str
    username: str
    email: Optional[str]
    roles: List[str]
    permissions: List[str]
    created_at: datetime
    last_login: Optional[datetime]

class Permission(TypedDict):
    """权限定义"""
    resource: str
    action: str
    scope: Optional[str]

# 批处理任务类型
class BatchJob(TypedDict):
    """批处理任务"""
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
    """批处理任务处理器协议"""
    def prepare_job(self, job: BatchJob) -> Awaitable[None]:
        """准备任务"""
        ...

    def execute_step(self, step: int, job: BatchJob) -> Awaitable[Dict[str, Any]]:
        """执行步骤"""
        ...

    def cleanup_job(self, job: BatchJob, success: bool) -> Awaitable[None]:
        """清理任务"""
        ...

# 缓存相关类型
class CacheEntry(TypedDict):
    """缓存条目"""
    key: str
    value: Any
    ttl: Optional[int]
    created_at: datetime
    access_count: int
    last_accessed: datetime

class CacheStrategy(Protocol):
    """缓存策略协议"""
    def get(self, key: str) -> Optional[Any]:
        """获取缓存值"""
        ...

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """设置缓存值"""
        ...

    def delete(self, key: str) -> bool:
        """删除缓存值"""
        ...

    def clear(self) -> bool:
        """清空缓存"""
        ...

    def exists(self, key: str) -> bool:
        """检查键是否存在"""
        ...

# 搜索相关类型
class SearchQuery(TypedDict):
    """搜索查询"""
    query: str
    filters: Optional[Dict[str, Any]]
    sort: Optional[Dict[str, Literal['asc', 'desc']]]
    page: int
    page_size: int

class SearchResult(TypedDict, Generic[ResponseType]):
    """搜索结果"""
    items: List[ResponseType]
    total: int
    page: int
    page_size: int
    took_ms: float

class SearchIndex(Protocol):
    """搜索索引协议"""
    async def index_document(self, doc_id: str, document: Dict[str, Any]) -> bool:
        """索引文档"""
        ...

    async def search(self, query: SearchQuery) -> SearchResult[Dict[str, Any]]:
        """搜索文档"""
        ...

    async def delete_document(self, doc_id: str) -> bool:
        """删除文档"""
        ...

    async def rebuild_index(self) -> bool:
        """重建索引"""
        ...

# 通知系统类型
class Notification(TypedDict):
    """通知"""
    notification_id: str
    user_id: str
    title: str
    message: str
    notification_type: Literal['info', 'warning', 'error', 'success']
    created_at: datetime
    read_at: Optional[datetime]
    data: Optional[Dict[str, Any]]

class NotificationChannel(Protocol):
    """通知渠道协议"""
    async def send(self, notification: Notification) -> bool:
        """发送通知"""
        ...

    async def send_bulk(self, notifications: List[Notification]) -> Dict[str, List[str]]:
        """批量发送通知"""
        ...

# 审计日志类型
class AuditLog(TypedDict):
    """审计日志"""
    log_id: str
    user_id: str
    action: str
    resource: str
    resource_id: Optional[str]
    details: Optional[Dict[str, Any]]
    ip_address: Optional[str]
    user_agent: Optional[str]
    timestamp: datetime

# 统计和分析类型
class UsageStatistics(TypedDict):
    """使用统计"""
    metric_name: str
    value: float
    timestamp: datetime
    dimensions: Optional[Dict[str, str]]

class PerformanceMetrics(TypedDict):
    """性能指标"""
    operation: str
    duration_ms: float
    success: bool
    timestamp: datetime
    details: Optional[Dict[str, Any]]

# 错误处理类型
class ErrorContext(TypedDict):
    """错误上下文"""
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
    """错误恢复策略协议"""
    def should_retry(self, error: Exception, attempt: int) -> bool:
        """是否应该重试"""
        ...

    def get_retry_delay(self, attempt: int) -> float:
        """获取重试延迟"""
        ...

    def handle_failure(self, error: Exception, context: ErrorContext) -> Any:
        """处理失败"""
        ...

# 版本和兼容性类型
class APIVersion(TypedDict):
    """API版本信息"""
    version: str
    deprecated: bool
    sunset_date: Optional[datetime]
    migration_guide: Optional[str]

class CompatibilityMatrix(TypedDict):
    """兼容性矩阵"""
    api_version: str
    client_version: str
    compatible: bool
    features: List[str]

# 工具类型
def create_success_response(data: ResponseType, message: str = "Success") -> SuccessResponse[ResponseType]:
    """创建成功响应"""
    return {
        "success": True,
        "data": data,
        "message": message
    }

def create_error_response(error_code: str, message: str, details: Optional[Dict[str, Any]] = None) -> ErrorResponse:
    """创建错误响应"""
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
    """创建分页响应"""
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
