"""统一异常类型和错误管理模块"""
from enum import Enum
from typing import Any, Dict, Optional


class ErrorSeverity(Enum):
    """错误严重级别"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """错误类别"""
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    VALIDATION = "validation"
    BUSINESS_LOGIC = "business_logic"
    DATABASE = "database"
    STORAGE = "storage"
    NETWORK = "network"
    FILE_SYSTEM = "file_system"
    EXTERNAL_SERVICE = "external_service"
    SYSTEM = "system"


class YOLOException(Exception):
    """基础业务异常类"""

    def __init__(
        self,
        message: str,
        error_code: str,
        category: ErrorCategory,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        details: Optional[Dict[str, Any]] = None,
        original_error: Optional[Exception] = None
    ):
        self.message = message
        self.error_code = error_code
        self.category = category
        self.severity = severity
        self.details = details or {}
        self.original_error = original_error
        super().__init__(self.message)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "error_code": self.error_code,
            "message": self.message,
            "category": self.category.value,
            "severity": self.severity.value,
            "details": self.details,
            "type": self.__class__.__name__
        }


# 认证相关异常
class AuthenticationException(YOLOException):
    """认证异常"""

    def __init__(self, message: str = "认证失败", details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="AUTH_FAILED",
            category=ErrorCategory.AUTHENTICATION,
            severity=ErrorSeverity.MEDIUM,
            details=details
        )


class AuthorizationException(YOLOException):
    """授权异常"""

    def __init__(self, message: str = "权限不足", details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="AUTHZ_INSUFFICIENT",
            category=ErrorCategory.AUTHORIZATION,
            severity=ErrorSeverity.HIGH,
            details=details
        )


# 数据验证异常
class ValidationException(YOLOException):
    """数据验证异常"""

    def __init__(self, message: str, field: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        validation_details = details or {}
        if field:
            validation_details["field"] = field
        super().__init__(
            message=message,
            error_code="VALIDATION_ERROR",
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.MEDIUM,
            details=validation_details
        )


# 业务逻辑异常
class BusinessLogicException(YOLOException):
    """业务逻辑异常"""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="BUSINESS_LOGIC_ERROR",
            category=ErrorCategory.BUSINESS_LOGIC,
            severity=ErrorSeverity.MEDIUM,
            details=details
        )


class DatasetNotFoundException(YOLOException):
    """数据集未找到异常"""

    def __init__(self, dataset_id: str):
        super().__init__(
            message=f"数据集未找到: {dataset_id}",
            error_code="DATASET_NOT_FOUND",
            category=ErrorCategory.BUSINESS_LOGIC,
            severity=ErrorSeverity.HIGH,
            details={"dataset_id": dataset_id}
        )


class DatasetAlreadyExistsException(YOLOException):
    """数据集已存在异常"""

    def __init__(self, dataset_name: str):
        super().__init__(
            message=f"数据集已存在: {dataset_name}",
            error_code="DATASET_ALREADY_EXISTS",
            category=ErrorCategory.BUSINESS_LOGIC,
            severity=ErrorSeverity.MEDIUM,
            details={"dataset_name": dataset_name}
        )


# 数据库相关异常
class DatabaseException(YOLOException):
    """数据库操作异常"""

    def __init__(self, message: str, operation: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        db_details = details or {}
        if operation:
            db_details["operation"] = operation
        super().__init__(
            message=message,
            error_code="DATABASE_ERROR",
            category=ErrorCategory.DATABASE,
            severity=ErrorSeverity.HIGH,
            details=db_details
        )


class DatabaseConnectionException(DatabaseException):
    """数据库连接异常"""

    def __init__(self, message: str = "数据库连接失败", details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            operation="connection",
            severity=ErrorSeverity.CRITICAL,
            details=details
        )


# 存储相关异常
class StorageException(YOLOException):
    """存储操作异常"""

    def __init__(self, message: str, operation: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        storage_details = details or {}
        if operation:
            storage_details["operation"] = operation
        super().__init__(
            message=message,
            error_code="STORAGE_ERROR",
            category=ErrorCategory.STORAGE,
            severity=ErrorSeverity.HIGH,
            details=storage_details
        )


class FileUploadException(StorageException):
    """文件上传异常"""

    def __init__(self, message: str, filename: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        upload_details = details or {}
        if filename:
            upload_details["filename"] = filename
        super().__init__(
            message=message,
            operation="upload",
            severity=ErrorSeverity.HIGH,
            details=upload_details
        )


class FileNotFoundException(StorageException):
    """文件未找到异常"""

    def __init__(self, file_path: str):
        super().__init__(
            message=f"文件未找到: {file_path}",
            operation="access",
            severity=ErrorSeverity.MEDIUM,
            details={"file_path": file_path}
        )


# YOLO相关异常
class YOLOValidationException(YOLOException):
    """YOLO格式验证异常"""

    def __init__(self, message: str, dataset_path: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        validation_details = details or {}
        if dataset_path:
            validation_details["dataset_path"] = dataset_path
        super().__init__(
            message=message,
            error_code="YOLO_VALIDATION_ERROR",
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.HIGH,
            details=validation_details
        )


class YOLODatasetTypeException(YOLOException):
    """YOLO数据集类型异常"""

    def __init__(self, message: str, dataset_type: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        type_details = details or {}
        if dataset_type:
            type_details["dataset_type"] = dataset_type
        super().__init__(
            message=message,
            error_code="YOLO_DATASET_TYPE_ERROR",
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.MEDIUM,
            details=type_details
        )


# 系统异常
class SystemException(YOLOException):
    """系统级异常"""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="SYSTEM_ERROR",
            category=ErrorCategory.SYSTEM,
            severity=ErrorSeverity.CRITICAL,
            details=details
        )


class ConfigurationException(YOLOException):
    """配置异常"""

    def __init__(self, message: str, config_key: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        config_details = details or {}
        if config_key:
            config_details["config_key"] = config_key
        super().__init__(
            message=message,
            error_code="CONFIG_ERROR",
            category=ErrorCategory.SYSTEM,
            severity=ErrorSeverity.CRITICAL,
            details=config_details
        )
