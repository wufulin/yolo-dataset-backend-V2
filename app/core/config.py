"""统一的配置管理系统，支持环境变量、配置验证、不同环境切换"""
import json
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Union

import yaml
from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.core.exceptions import ConfigurationException, SystemException


class Environment(Enum):
    """环境类型"""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


class LogLevel(Enum):
    """日志级别"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class DatabaseConfig:
    """数据库配置"""
    url: str
    db_name: str
    max_pool_size: int = Field(default=100, ge=1, le=1000)
    max_idle_time_ms: int = Field(default=30000, ge=1000, le=300000)
    server_selection_timeout_ms: int = Field(default=5000, ge=1000, le=30000)
    connect_timeout_ms: int = Field(default=5000, ge=1000, le=30000)
    socket_timeout_ms: int = Field(default=30000, ge=1000, le=60000)
    heartbeat_frequency_ms: int = Field(default=10000, ge=1000, le=60000)
    retry_writes: bool = Field(default=True)
    wait_queue_multiple: int = Field(default=5, ge=1, le=20)
    wait_queue_timeout_ms: int = Field(default=30000, ge=1000, le=300000)

    # 性能配置
    batch_size: int = Field(default=1000, ge=1, le=10000)
    max_concurrent_operations: int = Field(default=10, ge=1, le=50)
    query_timeout_ms: int = Field(default=30000, ge=1000, le=300000)
    transaction_retry_attempts: int = Field(default=3, ge=1, le=10)


@dataclass
class MinioConfig:
    """MinIO配置"""
    endpoint: str
    access_key: str
    secret_key: str
    bucket_name: str = "yolo-datasets"
    secure: bool = Field(default=False)

    # 性能配置
    chunk_size: int = Field(default=100 * 1024 * 1024, ge=1024*1024, le=1024*1024*1024)  # 100MB
    max_workers: int = Field(default=20, ge=1, le=100)
    max_retries: int = Field(default=3, ge=0, le=10)
    retry_delay: float = Field(default=1.0, ge=0.1, le=10.0)
    connection_pool_size: int = Field(default=100, ge=10, le=1000)


@dataclass
class RedisConfig:
    """Redis配置"""
    url: str = Field(default="redis://localhost:6379/0")
    host: str = Field(default="localhost")
    port: int = Field(default=6379, ge=1, le=65535)
    db: int = Field(default=0, ge=0, le=15)
    password: Optional[str] = Field(default=None)
    max_connections: int = Field(default=20, ge=1, le=100)
    session_ttl: int = Field(default=3600, ge=60, le=86400)  # 1小时
    session_lock_timeout: int = Field(default=30, ge=10, le=300)  # 30秒
    connection_pool_timeout: int = Field(default=5, ge=1, le=60)


@dataclass
class SecurityConfig:
    """安全配置"""
    secret_key: str
    jwt_algorithm: str = Field(default="HS256")
    access_token_expire_minutes: int = Field(default=120, ge=5, le=1440)  # 2小时
    refresh_token_expire_days: int = Field(default=7, ge=1, le=30)
    password_salt_rounds: int = Field(default=12, ge=10, le=20)


@dataclass
class FileConfig:
    """文件处理配置"""
    allowed_image_formats: List[str] = field(default_factory=lambda: ["JPEG", "JPG", "PNG", "BMP", "TIFF"])
    max_upload_size: int = Field(default=100 * 1024 * 1024 * 1024, ge=1024*1024, le=1024*1024*1024*1024)  # 100GB
    upload_chunk_size: int = Field(default=10 * 1024 * 1024, ge=1024*1024, le=1024*1024*1024)  # 10MB
    temp_dir: str = Field(default="/tmp/yolo_datasets_upload")

    # 图像处理配置
    thumbnail_sizes: Dict[str, tuple] = field(default_factory=lambda: {
        "small": (150, 150),
        "medium": (300, 300)
    })
    default_thumbnail_quality: int = Field(default=80, ge=10, le=100)

    # YOLO配置
    max_annotations_per_image: int = Field(default=1000, ge=1, le=10000)
    annotation_confidence_threshold: float = Field(default=0.1, ge=0.0, le=1.0)
    yolo_validation_timeout: int = Field(default=300, ge=60, le=1800)  # 5分钟


@dataclass
class APIConfig:
    """API配置"""
    default_page_size: int = Field(default=20, ge=1, le=100)
    max_page_size: int = Field(default=50, ge=1, le=200)
    rate_limit_per_minute: int = Field(default=1000, ge=1, le=10000)
    request_timeout: int = Field(default=30, ge=5, le=300)


@dataclass
class LoggingConfig:
    """日志配置"""
    level: LogLevel = Field(default=LogLevel.INFO)
    format: str = Field(default="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_path: Optional[str] = Field(default=None)
    max_file_size: int = Field(default=100 * 1024 * 1024, ge=1024*1024, le=1024*1024*1024)  # 100MB
    backup_count: int = Field(default=5, ge=1, le=20)
    enable_console: bool = Field(default=True)
    enable_file: bool = Field(default=False)


class Settings(BaseSettings):
    """应用设置 - 使用Pydantic进行配置管理"""

    model_config = SettingsConfigDict(
        env_file=[".env.dev", ".env", ".env.local"],
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="allow"
    )

    # 应用基础配置
    app_name: str = Field(default="YOLO Dataset API", description="应用名称")
    app_version: str = Field(default="2.0.0", description="应用版本")
    debug: bool = Field(default=False, description="调试模式")
    environment: Environment = Field(default=Environment.DEVELOPMENT, description="运行环境")

    # 服务器配置
    host: str = Field(default="0.0.0.0", description="服务主机")
    port: int = Field(default=8000, ge=1, le=65535, description="服务端口")

    # 数据库配置
    mongodb_url: str = Field(default="mongodb://localhost:27017/yolo_datasets?authSource=admin")
    mongo_db_name: str = Field(default="yolo_datasets")
    mongodb_max_pool_size: int = Field(default=100, ge=1, le=1000)
    mongodb_max_idle_time_ms: int = Field(default=30000, ge=1000, le=300000)
    mongodb_server_selection_timeout_ms: int = Field(default=5000, ge=1000, le=30000)
    mongodb_connect_timeout_ms: int = Field(default=5000, ge=1000, le=30000)
    mongodb_socket_timeout_ms: int = Field(default=30000, ge=1000, le=60000)
    mongodb_heartbeat_frequency_ms: int = Field(default=10000, ge=1000, le=60000)
    mongodb_retry_writes: bool = Field(default=True)
    mongodb_wait_queue_multiple: int = Field(default=5, ge=1, le=20)
    mongodb_wait_queue_timeout_ms: int = Field(default=30000, ge=1000, le=300000)

    # 数据库性能配置
    mongodb_batch_size: int = Field(default=1000, ge=1, le=10000)
    mongodb_max_concurrent_operations: int = Field(default=10, ge=1, le=50)
    mongodb_query_timeout_ms: int = Field(default=30000, ge=1000, le=300000)
    mongodb_transaction_retry_attempts: int = Field(default=3, ge=1, le=10)

    # MinIO配置
    minio_endpoint: str = Field(default="localhost:9000")
    minio_access_key: str = Field(default="minioadmin")
    minio_secret_key: str = Field(default="minioadmin")
    minio_bucket_name: str = Field(default="yolo-datasets")
    minio_secure: bool = Field(default=False)
    minio_chunk_size: int = Field(default=100 * 1024 * 1024, ge=1024*1024, le=1024*1024*1024)
    minio_max_workers: int = Field(default=20, ge=1, le=100)
    minio_max_retries: int = Field(default=3, ge=0, le=10)
    minio_retry_delay: float = Field(default=1.0, ge=0.1, le=10.0)
    minio_connection_pool_size: int = Field(default=100, ge=10, le=1000)

    # 文件上传配置
    allowed_image_formats: List[str] = Field(default=["JPEG", "JPG", "PNG", "BMP", "TIFF"])
    max_upload_size: int = Field(default=100 * 1024 * 1024 * 1024, ge=1024*1024, le=1024*1024*1024*1024)
    upload_chunk_size: int = Field(default=10 * 1024 * 1024, ge=1024*1024, le=1024*1024*1024)
    temp_dir: str = Field(default="/tmp/yolo_datasets_upload")

    # 图像处理配置
    thumbnail_sizes: Dict[str, tuple] = Field(default={"small": (150, 150), "medium": (300, 300)})
    default_thumbnail_quality: int = Field(default=80, ge=10, le=100)

    # YOLO标注配置
    max_annotations_per_image: int = Field(default=1000, ge=1, le=10000)
    annotation_confidence_threshold: float = Field(default=0.1, ge=0.0, le=1.0)
    yolo_validation_timeout: int = Field(default=300, ge=60, le=1800)

    # 分页配置
    default_page_size: int = Field(default=20, ge=1, le=100)
    max_page_size: int = Field(default=50, ge=1, le=200)

    # Redis配置
    redis_url: str = Field(default="redis://localhost:6379/0")
    redis_host: str = Field(default="localhost")
    redis_port: int = Field(default=6379, ge=1, le=65535)
    redis_db: int = Field(default=0, ge=0, le=15)
    redis_password: Optional[str] = Field(default=None)
    redis_max_connections: int = Field(default=20, ge=1, le=100)
    redis_session_ttl: int = Field(default=3600, ge=60, le=86400)
    redis_session_lock_timeout: int = Field(default=30, ge=10, le=300)
    redis_connection_pool_timeout: int = Field(default=5, ge=1, le=60)

    # 简化JWT配置
    jwt_algorithm: str = Field(default="HS256")
    jwt_access_token_expire_minutes: int = Field(default=120, ge=5, le=1440)
    jwt_refresh_token_expire_days: int = Field(default=7, ge=1, le=30)
    secret_key: str = Field(default="yolo-secret-key-simplified", min_length=32)

    # 日志配置
    log_level: LogLevel = Field(default=LogLevel.INFO)
    log_format: str = Field(default="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    log_file_path: Optional[str] = Field(default=None)
    log_max_file_size: int = Field(default=100 * 1024 * 1024, ge=1024*1024, le=1024*1024*1024)
    log_backup_count: int = Field(default=5, ge=1, le=20)
    log_enable_console: bool = Field(default=True)
    log_enable_file: bool = Field(default=False)

    # 性能监控配置
    enable_performance_monitoring: bool = Field(default=True)
    slow_query_threshold: float = Field(default=1.0, ge=0.1, le=10.0)
    enable_metrics_export: bool = Field(default=False)

    @field_validator("secret_key")
    @classmethod
    def validate_secret_key(cls, v):
        """验证密钥强度"""
        if len(v) < 32:
            raise ValueError("Secret key must be at least 32 characters long")
        return v

    @field_validator("environment", mode="before")
    @classmethod
    def validate_environment(cls, v):
        """验证环境类型"""
        if isinstance(v, str):
            return Environment(v.lower())
        return v

    @field_validator("log_level", mode="before")
    @classmethod
    def validate_log_level(cls, v):
        """验证日志级别"""
        if isinstance(v, str):
            return LogLevel(v.upper())
        return v

    @field_validator("allowed_image_formats")
    @classmethod
    def validate_image_formats(cls, v):
        """验证图像格式"""
        valid_formats = {"JPEG", "JPG", "PNG", "BMP", "TIFF", "WEBP"}
        for fmt in v:
            if fmt.upper() not in valid_formats:
                raise ValueError(f"Invalid image format: {fmt}")
        return [fmt.upper() for fmt in v]

    @model_validator(mode="before")
    @classmethod
    def validate_dependencies(cls, values):
        """验证配置依赖关系"""
        # 开发环境和生产环境的不同要求
        environment = values.get("environment", Environment.DEVELOPMENT)

        if environment == Environment.PRODUCTION:
            # 生产环境要求更严格的安全配置
            if len(values.get("secret_key", "")) < 64:
                raise ValueError("Production environment requires secret key of at least 64 characters")

            if not values.get("debug", False):
                # 生产环境关闭调试
                values["debug"] = False

            # 确保日志文件配置
            if not values.get("log_enable_file", False):
                raise ValueError("Production environment requires file logging enabled")

        return values

    def get_database_config(self) -> DatabaseConfig:
        """获取数据库配置"""
        return DatabaseConfig(
            url=self.mongodb_url,
            db_name=self.mongo_db_name,
            max_pool_size=self.mongodb_max_pool_size,
            max_idle_time_ms=self.mongodb_max_idle_time_ms,
            server_selection_timeout_ms=self.mongodb_server_selection_timeout_ms,
            connect_timeout_ms=self.mongodb_connect_timeout_ms,
            socket_timeout_ms=self.mongodb_socket_timeout_ms,
            heartbeat_frequency_ms=self.mongodb_heartbeat_frequency_ms,
            retry_writes=self.mongodb_retry_writes,
            wait_queue_multiple=self.mongodb_wait_queue_multiple,
            wait_queue_timeout_ms=self.mongodb_wait_queue_timeout_ms,
            batch_size=self.mongodb_batch_size,
            max_concurrent_operations=self.mongodb_max_concurrent_operations,
            query_timeout_ms=self.mongodb_query_timeout_ms,
            transaction_retry_attempts=self.mongodb_transaction_retry_attempts
        )

    def get_minio_config(self) -> MinioConfig:
        """获取MinIO配置"""
        return MinioConfig(
            endpoint=self.minio_endpoint,
            access_key=self.minio_access_key,
            secret_key=self.minio_secret_key,
            bucket_name=self.minio_bucket_name,
            secure=self.minio_secure,
            chunk_size=self.minio_chunk_size,
            max_workers=self.minio_max_workers,
            max_retries=self.minio_max_retries,
            retry_delay=self.minio_retry_delay,
            connection_pool_size=self.minio_connection_pool_size
        )

    def get_redis_config(self) -> RedisConfig:
        """获取Redis配置"""
        return RedisConfig(
            url=self.redis_url,
            host=self.redis_host,
            port=self.redis_port,
            db=self.redis_db,
            password=self.redis_password,
            max_connections=self.redis_max_connections,
            session_ttl=self.redis_session_ttl,
            session_lock_timeout=self.redis_session_lock_timeout,
            connection_pool_timeout=self.redis_connection_pool_timeout
        )

    def get_security_config(self) -> SecurityConfig:
        """获取安全配置"""
        return SecurityConfig(
            secret_key=self.secret_key,
            jwt_algorithm=self.jwt_algorithm,
            access_token_expire_minutes=self.jwt_access_token_expire_minutes,
            refresh_token_expire_days=self.jwt_refresh_token_expire_days
        )

    def get_file_config(self) -> FileConfig:
        """获取文件处理配置"""
        return FileConfig(
            allowed_image_formats=self.allowed_image_formats,
            max_upload_size=self.max_upload_size,
            upload_chunk_size=self.upload_chunk_size,
            temp_dir=self.temp_dir,
            thumbnail_sizes=self.thumbnail_sizes,
            default_thumbnail_quality=self.default_thumbnail_quality,
            max_annotations_per_image=self.max_annotations_per_image,
            annotation_confidence_threshold=self.annotation_confidence_threshold,
            yolo_validation_timeout=self.yolo_validation_timeout
        )

    def get_logging_config(self) -> LoggingConfig:
        """获取日志配置"""
        return LoggingConfig(
            level=self.log_level,
            format=self.log_format,
            file_path=self.log_file_path,
            max_file_size=self.log_max_file_size,
            backup_count=self.log_backup_count,
            enable_console=self.log_enable_console,
            enable_file=self.log_enable_file
        )


class ConfigManager:
    """配置管理器 - 单例模式"""

    _instance: Optional['ConfigManager'] = None
    _settings: Optional[Settings] = None
    _config_cache: Dict[str, Any] = {}

    def __new__(cls, *args, **kwargs):
        """确保只创建一个实例"""
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
        return cls._instance

    def __init__(self, config_file: Optional[str] = None):
        """初始化配置管理器"""
        if hasattr(self, '_initialized'):
            return

        self._initialized = True
        self.config_file = config_file
        self._watchers: List[Callable] = []

        # 加载配置
        self._load_settings()

    def _load_settings(self):
        """加载设置"""
        try:
            if self.config_file and Path(self.config_file).exists():
                # 从配置文件加载
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config_data = yaml.safe_load(f)

                # 转换为环境变量格式
                env_vars = {}
                for key, value in config_data.items():
                    env_key = f"YOLO_{key.upper()}"
                    if isinstance(value, (dict, list)):
                        env_vars[env_key] = json.dumps(value)
                    else:
                        env_vars[env_key] = str(value)

                # 设置环境变量
                for key, value in env_vars.items():
                    os.environ[key] = value

            # 创建设置实例
            self._settings = Settings()

            # 验证配置
            self._validate_settings()

            # 设置配置缓存
            self._config_cache = {
                "database": self._settings.get_database_config(),
                "minio": self._settings.get_minio_config(),
                "redis": self._settings.get_redis_config(),
                "security": self._settings.get_security_config(),
                "file": self._settings.get_file_config(),
                "logging": self._settings.get_logging_config()
            }

        except Exception as e:
            raise ConfigurationException(f"Failed to load settings: {str(e)}")

    def _validate_settings(self):
        """验证设置"""
        if not self._settings:
            raise ConfigurationException("Settings not loaded")

        # 验证必要的配置项
        required_configs = [
            ("secret_key", self._settings.secret_key),
            ("mongodb_url", self._settings.mongodb_url),
            ("minio_endpoint", self._settings.minio_endpoint),
        ]

        for name, value in required_configs:
            if not value:
                raise ConfigurationException(f"Required configuration missing: {name}")

        # 验证环境特定配置
        if self._settings.environment == Environment.PRODUCTION:
            self._validate_production_config()

    def _validate_production_config(self):
        """验证生产环境配置"""
        # 生产环境安全检查
        if len(self._settings.secret_key) < 64:
            raise ConfigurationException("Production environment requires stronger secret key")

        if self._settings.debug:
            raise ConfigurationException("Debug mode should be disabled in production")

        if not self._settings.log_enable_file:
            raise ConfigurationException("File logging must be enabled in production")

    @property
    def settings(self) -> Settings:
        """获取设置实例"""
        if not self._settings:
            raise ConfigurationException("Settings not initialized")
        return self._settings

    def get(self, key: str, default: Any = None) -> Any:
        """获取配置值"""
        return getattr(self.settings, key, default)

    def get_typed_config(self, config_type: str) -> Any:
        """获取类型化配置"""
        if config_type not in self._config_cache:
            raise ConfigurationException(f"Unknown config type: {config_type}")
        return self._config_cache[config_type]

    def update(self, key: str, value: Any):
        """更新配置值"""
        if hasattr(self.settings, key):
            setattr(self.settings, key, value)

            # 更新缓存
            if key in ["mongodb_url", "mongo_db_name", "mongodb_max_pool_size",
                      "mongodb_max_idle_time_ms", "mongodb_server_selection_timeout_ms",
                      "mongodb_connect_timeout_ms", "mongodb_socket_timeout_ms",
                      "mongodb_heartbeat_frequency_ms", "mongodb_retry_writes",
                      "mongodb_wait_queue_multiple", "mongodb_wait_queue_timeout_ms",
                      "mongodb_batch_size", "mongodb_max_concurrent_operations",
                      "mongodb_query_timeout_ms", "mongodb_transaction_retry_attempts"]:
                self._config_cache["database"] = self.settings.get_database_config()
            elif key in ["minio_endpoint", "minio_access_key", "minio_secret_key",
                        "minio_bucket_name", "minio_secure", "minio_chunk_size",
                        "minio_max_workers", "minio_max_retries", "minio_retry_delay",
                        "minio_connection_pool_size"]:
                self._config_cache["minio"] = self.settings.get_minio_config()
            elif key in ["redis_url", "redis_host", "redis_port", "redis_db",
                        "redis_password", "redis_max_connections", "redis_session_ttl",
                        "redis_session_lock_timeout", "redis_connection_pool_timeout"]:
                self._config_cache["redis"] = self.settings.get_redis_config()
            elif key in ["secret_key", "jwt_algorithm", "jwt_access_token_expire_minutes",
                        "jwt_refresh_token_expire_days"]:
                self._config_cache["security"] = self.settings.get_security_config()
            elif key in ["allowed_image_formats", "max_upload_size", "upload_chunk_size",
                        "temp_dir", "thumbnail_sizes", "default_thumbnail_quality",
                        "max_annotations_per_image", "annotation_confidence_threshold",
                        "yolo_validation_timeout"]:
                self._config_cache["file"] = self.settings.get_file_config()
            elif key in ["log_level", "log_format", "log_file_path", "log_max_file_size",
                        "log_backup_count", "log_enable_console", "log_enable_file"]:
                self._config_cache["logging"] = self.settings.get_logging_config()

            # 通知观察者
            self._notify_watchers(key, value)
        else:
            raise ConfigurationException(f"Unknown configuration key: {key}")

    def add_watcher(self, watcher: Callable[[str, Any], None]):
        """添加配置变化观察者"""
        self._watchers.append(watcher)

    def _notify_watchers(self, key: str, value: Any):
        """通知观察者配置变化"""
        for watcher in self._watchers:
            try:
                watcher(key, value)
            except Exception as e:
                print(f"Failed to notify config watcher: {e}")

    def reload(self):
        """重新加载配置"""
        self._load_settings()

    def export_config(self, format: Literal['yaml', 'json', 'env'] = 'yaml') -> str:
        """导出配置"""
        config_dict = self.settings.model_dump()

        if format == 'json':
            return json.dumps(config_dict, indent=2, ensure_ascii=False)
        elif format == 'env':
            lines = []
            for key, value in config_dict.items():
                env_key = f"YOLO_{key.upper()}"
                if isinstance(value, (dict, list)):
                    lines.append(f"{env_key}='{json.dumps(value)}'")
                else:
                    lines.append(f"{env_key}={value}")
            return "\n".join(lines)
        else:  # yaml
            return yaml.dump(config_dict, default_flow_style=False, allow_unicode=True)


# 全局配置管理器实例
config_manager = ConfigManager()
