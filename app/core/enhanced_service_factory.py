"""Enhanced service factory pattern implementation."""
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

# Type variables
T = TypeVar('T')


@runtime_checkable
class ServiceProtocol(Protocol):
    """Base protocol shared by all services."""

    def initialize(self, **kwargs) -> None:
        """Initialize service resources."""
        ...

    def get_status(self) -> Dict[str, Any]:
        """Return health/status metadata."""
        ...


class BaseService(ABC, Generic[T]):
    """Abstract base class for concrete services."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self._initialized = False

    @abstractmethod
    def initialize(self, **kwargs) -> None:
        """Initialize service resources."""
        pass

    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        """Return health/status metadata."""
        pass

    def is_initialized(self) -> bool:
        """Return True if initialize() has been called."""
        return self._initialized

    def _mark_initialized(self) -> None:
        """Mark service as initialized."""
        self._initialized = True


class ServiceRegistry:
    """Registry for service classes and cached instances."""

    def __init__(self):
        self._services: Dict[str, Type[BaseService]] = {}
        self._instances: Dict[str, BaseService] = {}

    def register(self, service_name: str, service_class: Type[BaseService]) -> None:
        """Register a BaseService subclass under a name."""
        if not issubclass(service_class, BaseService):
            raise BusinessLogicException(f"Service class must inherit from BaseService: {service_name}")

        self._services[service_name] = service_class
        logger.info(f"Registered service: {service_name}")

    def get_service_class(self, service_name: str) -> Type[BaseService]:
        """Return registered service class by name."""
        if service_name not in self._services:
            raise BusinessLogicException(f"Service not registered: {service_name}")

        return self._services[service_name]

    def create_service(self, service_name: str, **kwargs) -> BaseService:
        """Instantiate a service by name."""
        service_class = self.get_service_class(service_name)
        instance = service_class(**kwargs)
        return instance

    def get_cached_service(self, service_name: str, **kwargs) -> BaseService:
        """Return cached service instance keyed by args."""
        cache_key = f"{service_name}:{hash(str(sorted(kwargs.items())))}"

        if cache_key in self._instances:
            return self._instances[cache_key]

        instance = self.create_service(service_name, **kwargs)
        self._instances[cache_key] = instance
        return instance

    def unregister(self, service_name: str) -> None:
        """Remove service registration."""
        if service_name in self._services:
            del self._services[service_name]
            logger.info(f"Unregistered service: {service_name}")

    def list_services(self) -> list[str]:
        """List registered service names."""
        return list(self._services.keys())


class EnhancedServiceFactory:
    """Higher-level factory that supports lazy creation and caching."""

    def __init__(self):
        self.registry = ServiceRegistry()
        self._setup_default_services()

    def _setup_default_services(self) -> None:
        """Register built-in services (left blank for lazy registration)."""
        # Default services can be registered here; deferred to avoid circular imports.
        pass

    def register_service(self, service_name: str, service_class: Type[BaseService]) -> None:
        """Register a service class explicitly."""
        self.registry.register(service_name, service_class)

    def get_service(self, service_name: str, **kwargs) -> BaseService:
        """Return cached or lazily-created service instance."""
        try:
            return self.registry.get_cached_service(service_name, **kwargs)
        except BusinessLogicException:
            # Attempt lazy creation when not pre-registered
            return self._lazy_create_service(service_name, **kwargs)

    def _lazy_create_service(self, service_name: str, **kwargs) -> BaseService:
        """Create services lazily to avoid circular imports."""
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
        """Create dataset service instance."""
        try:
            from app.services.dataset_service import DatasetService
            service = DatasetService(**kwargs)
            service.initialize()
            return service
        except ImportError as e:
            raise BusinessLogicException(f"Cannot create dataset service: {e}")

    def _create_image_service(self, **kwargs) -> 'BaseService':
        """Create image service instance."""
        try:
            from app.services.image_service import ImageService
            service = ImageService(**kwargs)
            service.initialize()
            return service
        except ImportError as e:
            raise BusinessLogicException(f"Cannot create image service: {e}")

    def _create_minio_service(self, **kwargs) -> 'BaseService':
        """Create MinIO service instance."""
        try:
            from app.services.minio_service import MinioService
            service = MinioService(**kwargs)
            service.initialize()
            return service
        except ImportError as e:
            raise BusinessLogicException(f"Cannot create minio service: {e}")

    def _create_db_service(self, **kwargs) -> 'BaseService':
        """Create database service instance."""
        try:
            from app.services.db_service import DatabaseService

            # Database service is a singleton provided by import side-effects
            return DatabaseService()
        except ImportError as e:
            raise BusinessLogicException(f"Cannot create db service: {e}")

    def _create_redis_service(self, **kwargs) -> 'BaseService':
        """Create Redis service instance."""
        try:
            from app.services.redis_service import RedisService
            service = RedisService(**kwargs)
            service.initialize()
            return service
        except ImportError as e:
            raise BusinessLogicException(f"Cannot create redis service: {e}")

    def get_available_services(self) -> list[str]:
        """List registered and lazily supported services."""
        return self.registry.list_services() + [
            'dataset', 'image', 'minio', 'db', 'redis'
        ]


# Global service factory instance
service_factory = EnhancedServiceFactory()
