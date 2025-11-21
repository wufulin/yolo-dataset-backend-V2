"""Centralized exception definitions and error taxonomy."""
from enum import Enum
from typing import Any, Dict, Optional


class ErrorSeverity(Enum):
    """Describes severity for surfaced errors."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Categorization used for error routing and analytics."""
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
    """Base exception type for the application."""

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
        """Serialize exception metadata into a dict."""
        return {
            "error_code": self.error_code,
            "message": self.message,
            "category": self.category.value,
            "severity": self.severity.value,
            "details": self.details,
            "type": self.__class__.__name__
        }


# Authentication exceptions
class AuthenticationException(YOLOException):
    """Raised when authentication fails."""

    def __init__(self, message: str = "Authentication failed", details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="AUTH_FAILED",
            category=ErrorCategory.AUTHENTICATION,
            severity=ErrorSeverity.MEDIUM,
            details=details
        )


class AuthorizationException(YOLOException):
    """Raised when a user lacks required permissions."""

    def __init__(self, message: str = "Insufficient permissions", details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="AUTHZ_INSUFFICIENT",
            category=ErrorCategory.AUTHORIZATION,
            severity=ErrorSeverity.HIGH,
            details=details
        )


# Validation exceptions
class ValidationException(YOLOException):
    """Raised when request or model validation fails."""

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


# Business logic exceptions
class BusinessLogicException(YOLOException):
    """Raised when business rules cannot be satisfied."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="BUSINESS_LOGIC_ERROR",
            category=ErrorCategory.BUSINESS_LOGIC,
            severity=ErrorSeverity.MEDIUM,
            details=details
        )


class DatasetNotFoundException(YOLOException):
    """Raised when a dataset identifier cannot be located."""

    def __init__(self, dataset_id: str):
        super().__init__(
            message=f"Dataset not found: {dataset_id}",
            error_code="DATASET_NOT_FOUND",
            category=ErrorCategory.BUSINESS_LOGIC,
            severity=ErrorSeverity.HIGH,
            details={"dataset_id": dataset_id}
        )


class DatasetAlreadyExistsException(YOLOException):
    """Raised when attempting to create a duplicate dataset."""

    def __init__(self, dataset_name: str):
        super().__init__(
            message=f"Dataset already exists: {dataset_name}",
            error_code="DATASET_ALREADY_EXISTS",
            category=ErrorCategory.BUSINESS_LOGIC,
            severity=ErrorSeverity.MEDIUM,
            details={"dataset_name": dataset_name}
        )


# Database exceptions
class DatabaseException(YOLOException):
    """Raised for database operation failures."""

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
    """Raised when establishing a database connection fails."""

    def __init__(self, message: str = "Database connection failed", details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            operation="connection",
            severity=ErrorSeverity.CRITICAL,
            details=details
        )


# Storage exceptions
class StorageException(YOLOException):
    """Raised for storage layer failures."""

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
    """Raised when file upload operations fail."""

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
    """Raised when a file cannot be located."""

    def __init__(self, file_path: str):
        super().__init__(
            message=f"File not found: {file_path}",
            operation="access",
            severity=ErrorSeverity.MEDIUM,
            details={"file_path": file_path}
        )


# YOLO-specific exceptions
class YOLOValidationException(YOLOException):
    """Raised when YOLO format validation fails."""

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
    """Raised for unsupported YOLO dataset types."""

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


# System exceptions
class SystemException(YOLOException):
    """Raised for generic internal/system failures."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="SYSTEM_ERROR",
            category=ErrorCategory.SYSTEM,
            severity=ErrorSeverity.CRITICAL,
            details=details
        )


class ConfigurationException(YOLOException):
    """Raised for configuration/initialization failures."""

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
