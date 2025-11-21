"""服务接口协议定义 - 提供类型安全的接口规范"""
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

# 类型变量定义
T = TypeVar('T')
ModelType = TypeVar('ModelType')
CreateSchemaType = TypeVar('CreateSchemaType')
UpdateSchemaType = TypeVar('UpdateSchemaType')
DatabaseModelType = TypeVar('DatabaseModelType')


class QueryFilter(Protocol):
    """查询过滤器协议"""
    def to_dict(self) -> Dict[str, Any]:
        """转换为查询字典"""
        ...


@runtime_checkable
class CRUDRepository(Protocol[ModelType, CreateSchemaType, UpdateSchemaType]):
    """通用CRUD仓储协议"""

    async def create(self, obj: CreateSchemaType, **kwargs) -> ModelType:
        """创建对象"""
        ...

    async def get_by_id(self, id: str) -> Optional[ModelType]:
        """根据ID获取对象"""
        ...

    async def get_multi(
        self,
        filters: Optional[QueryFilter] = None,
        skip: int = 0,
        limit: int = 100,
        sort: Optional[Union[str, List[str]]] = None
    ) -> List[ModelType]:
        """获取多个对象"""
        ...

    async def update(self, id: str, obj: UpdateSchemaType, **kwargs) -> Optional[ModelType]:
        """更新对象"""
        ...

    async def delete(self, id: str) -> bool:
        """删除对象"""
        ...

    async def count(self, filters: Optional[QueryFilter] = None) -> int:
        """计算对象数量"""
        ...


class StorageService(Protocol):
    """存储服务协议"""

    async def upload_file(
        self,
        bucket_name: str,
        object_name: str,
        file_path: Union[str, Path],
        metadata: Optional[Dict[str, str]] = None
    ) -> str:
        """上传文件"""
        ...

    async def download_file(
        self,
        bucket_name: str,
        object_name: str,
        file_path: Union[str, Path]
    ) -> bool:
        """下载文件"""
        ...

    async def delete_file(self, bucket_name: str, object_name: str) -> bool:
        """删除文件"""
        ...

    async def get_file_url(
        self,
        bucket_name: str,
        object_name: str,
        expires: int = 3600
    ) -> str:
        """获取文件URL"""
        ...

    async def list_files(
        self,
        bucket_name: str,
        prefix: Optional[str] = None,
        max_keys: int = 1000
    ) -> List[Dict[str, Any]]:
        """列出文件"""
        ...

    async def file_exists(self, bucket_name: str, object_name: str) -> bool:
        """检查文件是否存在"""
        ...


class YOLOValidatorService(Protocol):
    """YOLO验证服务协议"""

    async def validate_dataset(
        self,
        dataset_path: Union[str, Path],
        dataset_type: str
    ) -> Dict[str, Any]:
        """验证数据集"""
        ...

    def get_supported_types(self) -> List[str]:
        """获取支持的数据集类型"""
        ...

    async def validate_single_annotation(
        self,
        annotation_path: Union[str, Path],
        dataset_type: str
    ) -> Dict[str, Any]:
        """验证单个标注文件"""
        ...


class ProgressTracker(Protocol):
    """进度追踪器协议"""

    async def start_task(self, task_id: str, task_type: str, total_steps: int) -> None:
        """开始任务"""
        ...

    async def update_progress(self, task_id: str, current_step: int, message: str = "") -> None:
        """更新进度"""
        ...

    async def complete_task(self, task_id: str, success: bool = True, message: str = "") -> None:
        """完成任务"""
        ...

    def get_progress(self, task_id: str) -> Optional[Dict[str, Any]]:
        """获取进度信息"""
        ...

    def get_active_tasks(self) -> List[str]:
        """获取活跃任务列表"""
        ...


class CacheService(Protocol):
    """缓存服务协议"""

    async def get(self, key: str, default: Any = None) -> Any:
        """获取缓存值"""
        ...

    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None
    ) -> bool:
        """设置缓存值"""
        ...

    async def delete(self, key: str) -> bool:
        """删除缓存值"""
        ...

    async def exists(self, key: str) -> bool:
        """检查键是否存在"""
        ...

    async def clear_pattern(self, pattern: str) -> int:
        """清除匹配模式的键"""
        ...


class MessageQueueService(Protocol):
    """消息队列服务协议"""

    async def publish(
        self,
        exchange: str,
        routing_key: str,
        message: Dict[str, Any]
    ) -> bool:
        """发布消息"""
        ...

    async def subscribe(
        self,
        queue: str,
        exchange: str,
        routing_key: str,
        handler: callable
    ) -> None:
        """订阅消息"""
        ...

    async def unsubscribe(self, queue: str) -> None:
        """取消订阅"""
        ...


class ConfigurationService(Protocol):
    """配置服务协议"""

    def get(self, key: str, default: Any = None) -> Any:
        """获取配置值"""
        ...

    def set(self, key: str, value: Any) -> bool:
        """设置配置值"""
        ...

    def update(self, config_dict: Dict[str, Any]) -> None:
        """批量更新配置"""
        ...

    def validate_config(self, config_dict: Dict[str, Any]) -> Dict[str, List[str]]:
        """验证配置"""
        ...


class EventPublisher(Protocol):
    """事件发布器协议"""

    async def publish_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """发布事件"""
        ...

    async def publish_batch(self, events: List[Dict[str, Any]]) -> None:
        """批量发布事件"""
        ...

    def subscribe(self, event_type: str, handler: callable) -> None:
        """订阅事件"""
        ...

    def unsubscribe(self, event_type: str, handler: callable) -> None:
        """取消订阅"""
        ...


class ValidationService(Protocol):
    """验证服务协议"""

    async def validate_file(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """验证文件"""
        ...

    async def validate_image(self, image_path: Union[str, Path]) -> Dict[str, Any]:
        """验证图像"""
        ...

    async def validate_yaml(self, yaml_path: Union[str, Path]) -> Dict[str, Any]:
        """验证YAML配置"""
        ...

    async def validate_annotations(
        self,
        annotation_path: Union[str, Path],
        dataset_type: str
    ) -> Dict[str, Any]:
        """验证标注文件"""
        ...


class MonitoringService(Protocol):
    """监控服务协议"""

    def record_metric(self, metric_name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """记录指标"""
        ...

    def start_timer(self, operation_name: str) -> Any:
        """开始计时"""
        ...

    def increment_counter(self, counter_name: str, value: int = 1, tags: Optional[Dict[str, str]] = None) -> None:
        """增加计数器"""
        ...

    def record_histogram(self, histogram_name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """记录直方图"""
        ...

    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        ...


class EmailService(Protocol):
    """邮件服务协议"""

    async def send_email(
        self,
        to: str,
        subject: str,
        body: str,
        html_body: Optional[str] = None,
        attachments: Optional[List[Dict[str, Any]]] = None
    ) -> bool:
        """发送邮件"""
        ...

    async def send_bulk_email(
        self,
        recipients: List[str],
        subject: str,
        body: str,
        html_body: Optional[str] = None
    ) -> Dict[str, List[str]]:
        """批量发送邮件"""
        ...


class TaskSchedulerService(Protocol):
    """任务调度器协议"""

    async def schedule_task(
        self,
        task_id: str,
        task_type: str,
        schedule_expression: str,
        handler: callable,
        kwargs: Optional[Dict[str, Any]] = None
    ) -> bool:
        """调度任务"""
        ...

    async def cancel_task(self, task_id: str) -> bool:
        """取消任务"""
        ...

    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """获取任务状态"""
        ...

    def list_tasks(self, task_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """列出任务"""
        ...


class SearchService(Protocol):
    """搜索服务协议"""

    async def index_document(
        self,
        collection: str,
        document_id: str,
        document: Dict[str, Any]
    ) -> bool:
        """索引文档"""
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
        """搜索文档"""
        ...

    async def delete_index(self, collection: str, document_id: str) -> bool:
        """删除索引"""
        ...

    async def reindex_collection(self, collection: str) -> bool:
        """重建索引"""
        ...


class NotificationService(Protocol):
    """通知服务协议"""

    async def send_notification(
        self,
        user_id: str,
        title: str,
        message: str,
        notification_type: str = "info",
        data: Optional[Dict[str, Any]] = None
    ) -> bool:
        """发送通知"""
        ...

    async def send_bulk_notification(
        self,
        user_ids: List[str],
        title: str,
        message: str,
        notification_type: str = "info"
    ) -> Dict[str, List[str]]:
        """批量发送通知"""
        ...

    async def get_user_notifications(
        self,
        user_id: str,
        limit: int = 50,
        unread_only: bool = False
    ) -> List[Dict[str, Any]]:
        """获取用户通知"""
        ...

    async def mark_as_read(self, notification_id: str) -> bool:
        """标记为已读"""
        ...


class AuditService(Protocol):
    """审计服务协议"""

    async def log_action(
        self,
        user_id: str,
        action: str,
        resource: str,
        resource_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        ip_address: Optional[str] = None
    ) -> bool:
        """记录操作日志"""
        ...

    async def get_audit_log(
        self,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """获取审计日志"""
        ...

    async def export_audit_log(
        self,
        filters: Optional[Dict[str, Any]] = None,
        format: str = "json"
    ) -> str:
        """导出审计日志"""
        ...
