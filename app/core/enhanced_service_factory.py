"""改进的服务工厂模式实现"""
from abc import ABC, abstractmethod
from typing import (
    Any,
    Dict,
    Generic,
    Optional,
    Protocol,
    Type,
    TypeVar,
    runtime_checkable,
)

from app.core.exceptions import BusinessLogicException
from app.utils.logger import get_logger

logger: logging.Logger = get_logger(__name__)

# 类型变量
T = TypeVar('T')


@runtime_checkable
class ServiceProtocol(Protocol):
    """服务基础协议"""
    def initialize(self, **kwargs) -> None:
        """初始化服务"""
        ...

    def get_status(self) -> Dict[str, Any]:
        """获取服务状态"""
        ...


class BaseService(ABC, Generic[T]):
    """基础服务类"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self._initialized = False

    @abstractmethod
    def initialize(self, **kwargs) -> None:
        """初始化服务"""
        pass

    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        """获取服务状态"""
        pass

    def is_initialized(self) -> bool:
        """检查是否已初始化"""
        return self._initialized

    def _mark_initialized(self) -> None:
        """标记为已初始化"""
        self._initialized = True


class ServiceRegistry:
    """服务注册中心"""

    def __init__(self):
        self._services: Dict[str, Type[BaseService]] = {}
        self._instances: Dict[str, BaseService] = {}

    def register(self, service_name: str, service_class: Type[BaseService]) -> None:
        """注册服务"""
        if not issubclass(service_class, BaseService):
            raise BusinessLogicException(f"Service class must inherit from BaseService: {service_name}")

        self._services[service_name] = service_class
        logger.info(f"Registered service: {service_name}")

    def get_service_class(self, service_name: str) -> Type[BaseService]:
        """获取服务类"""
        if service_name not in self._services:
            raise BusinessLogicException(f"Service not registered: {service_name}")

        return self._services[service_name]

    def create_service(self, service_name: str, **kwargs) -> BaseService:
        """创建服务实例"""
        service_class = self.get_service_class(service_name)
        instance = service_class(**kwargs)
        return instance

    def get_cached_service(self, service_name: str, **kwargs) -> BaseService:
        """获取缓存的服务实例"""
        cache_key = f"{service_name}:{hash(str(sorted(kwargs.items())))}"

        if cache_key in self._instances:
            return self._instances[cache_key]

        instance = self.create_service(service_name, **kwargs)
        self._instances[cache_key] = instance
        return instance

    def unregister(self, service_name: str) -> None:
        """注销服务"""
        if service_name in self._services:
            del self._services[service_name]
            logger.info(f"Unregistered service: {service_name}")

    def list_services(self) -> list[str]:
        """列出已注册的服务"""
        return list(self._services.keys())


class EnhancedServiceFactory:
    """增强的服务工厂"""

    def __init__(self):
        self.registry = ServiceRegistry()
        self._setup_default_services()

    def _setup_default_services(self) -> None:
        """设置默认服务"""
        # 这里可以注册默认的服务类
        # 由于循环导入问题，暂时使用延迟注册
        pass

    def register_service(self, service_name: str, service_class: Type[BaseService]) -> None:
        """注册服务"""
        self.registry.register(service_name, service_class)

    def get_service(self, service_name: str, **kwargs) -> BaseService:
        """获取服务实例"""
        try:
            return self.registry.get_cached_service(service_name, **kwargs)
        except BusinessLogicException:
            # 如果没有找到，尝试使用延迟注册
            return self._lazy_create_service(service_name, **kwargs)

    def _lazy_create_service(self, service_name: str, **kwargs) -> BaseService:
        """延迟创建服务（解决循环导入问题）"""
        service_mappings = {
            'dataset': self._create_dataset_service,
            'image': self._create_image_service,
            'minio': self._create_minio_service,
            'db': self._create_db_service,
            'redis': self._create_redis_service,
        }

        if service_name not in service_mappings:
            raise BusinessLogicException(f"Unknown service type: {service_name}")

        return service_mappings[service_name](**kwargs)

    def _create_dataset_service(self, **kwargs) -> 'BaseService':
        """创建数据集服务"""
        try:
            from app.services.dataset_service import DatasetService
            service = DatasetService(**kwargs)
            service.initialize()
            return service
        except ImportError as e:
            raise BusinessLogicException(f"Cannot create dataset service: {e}")

    def _create_image_service(self, **kwargs) -> 'BaseService':
        """创建图像服务"""
        try:
            from app.services.image_service import ImageService
            service = ImageService(**kwargs)
            service.initialize()
            return service
        except ImportError as e:
            raise BusinessLogicException(f"Cannot create image service: {e}")

    def _create_minio_service(self, **kwargs) -> 'BaseService':
        """创建MinIO服务"""
        try:
            from app.services.minio_service import MinioService
            service = MinioService(**kwargs)
            service.initialize()
            return service
        except ImportError as e:
            raise BusinessLogicException(f"Cannot create minio service: {e}")

    def _create_db_service(self, **kwargs) -> 'BaseService':
        """创建数据库服务"""
        try:
            from app.services.db_service import DatabaseService

            # 数据库服务是单例
            return DatabaseService()
        except ImportError as e:
            raise BusinessLogicException(f"Cannot create db service: {e}")

    def _create_redis_service(self, **kwargs) -> 'BaseService':
        """创建Redis服务"""
        try:
            from app.services.redis_service import RedisService
            service = RedisService(**kwargs)
            service.initialize()
            return service
        except ImportError as e:
            raise BusinessLogicException(f"Cannot create redis service: {e}")

    def get_available_services(self) -> list[str]:
        """获取可用的服务列表"""
        return self.registry.list_services() + [
            'dataset', 'image', 'minio', 'db', 'redis'
        ]


# 全局服务工厂实例
service_factory = EnhancedServiceFactory()
