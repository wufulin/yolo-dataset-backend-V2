"""Service for handling dataset operations - 改进版本，使用设计模式和异常处理."""
import asyncio
from datetime import datetime
from enum import Enum
import logging
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Protocol,
)

from bson import ObjectId
from pydantic import BaseModel
from pymongo.errors import DuplicateKeyError, PyMongoError

from app.core.decorators import (
    async_exception_handler,
    async_performance_monitor,
    async_retry,
)
from app.core.exceptions import (
    BusinessLogicException,
    DatabaseException,
    DatasetAlreadyExistsException,
    DatasetNotFoundException,
    ValidationException,
)
from app.core.patterns import DatasetValidatorContext
from app.models.dataset import Dataset
from app.services.db_service import db_service
from app.utils.logger import get_logger

logger: logging.Logger = get_logger(__name__)


class DatasetStatus(Enum):
    """数据集状态枚举"""
    ACTIVE = "active"
    PROCESSING = "processing"
    VALIDATING = "validating"
    ERROR = "error"
    DELETED = "deleted"


class DatasetFilter(BaseModel):
    """数据集过滤参数"""
    status: Optional[DatasetStatus] = None
    dataset_type: Optional[str] = None
    created_by: Optional[str] = None
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None
    search_text: Optional[str] = None


class DatasetRepository(Protocol):
    """数据集仓储协议"""

    async def create(self, dataset: Dataset) -> str: ...
    async def get_by_id(self, dataset_id: str) -> Optional[Dataset]: ...
    async def list(self, filters: DatasetFilter, skip: int, limit: int) -> List[Dataset]: ...
    async def update(self, dataset_id: str, updates: Dict[str, Any]) -> bool: ...
    async def delete(self, dataset_id: str) -> bool: ...
    async def count(self, filters: DatasetFilter) -> int: ...


class MongoDatasetRepository:
    """MongoDB数据集仓储实现"""

    def __init__(self):
        self.db = db_service
        self.collection = self.db.datasets
        self.validator_context = DatasetValidatorContext()

    @async_exception_handler(DatabaseException)
    @async_performance_monitor("dataset_create")
    async def create(self, dataset: Dataset) -> str:
        """创建数据集"""
        try:
            dataset_dict = dataset.to_mongo_dict()
            result = self.collection.insert_one(dataset_dict)

            if not result.acknowledged:
                raise DatabaseException("Dataset insertion was not acknowledged by MongoDB")
            if not result.inserted_id:
                raise DatabaseException("No dataset ID returned from MongoDB")

            logger.info(f"Dataset created: {result.inserted_id}")
            return str(result.inserted_id)

        except DuplicateKeyError:
            raise DatasetAlreadyExistsException(dataset.name)
        except PyMongoError as e:
            raise DatabaseException(f"Failed to create dataset: {e}")

    @async_exception_handler(DatabaseException)
    @async_performance_monitor("dataset_get_by_id")
    async def get_by_id(self, dataset_id: str) -> Optional[Dataset]:
        """根据ID获取数据集"""
        try:
            if not ObjectId.is_valid(dataset_id):
                raise ValidationException(f"Invalid ObjectId format: {dataset_id}")

            data = self.collection.find_one({"_id": ObjectId(dataset_id)})
            if data:
                dataset = Dataset.from_mongo_dict(data)
                self.db.convert_objectids_to_str(data)
                return dataset
            return None

        except PyMongoError as e:
            raise DatabaseException(f"Failed to get dataset: {e}")

    @async_exception_handler(DatabaseException)
    @async_performance_monitor("dataset_list")
    async def list(self, filters: DatasetFilter, skip: int, limit: int) -> List[Dataset]:
        """列出数据集"""
        try:
            query = self._build_query(filters)
            cursor = self.collection.find(query).skip(skip).limit(limit).sort("created_at", -1)

            datasets = []
            for data in cursor:
                dataset = Dataset.from_mongo_dict(data)
                datasets.append(dataset)

            return datasets

        except PyMongoError as e:
            raise DatabaseException(f"Failed to list datasets: {e}")

    def _build_query(self, filters: DatasetFilter) -> Dict[str, Any]:
        """构建查询条件"""
        query = {}

        if filters.status:
            query["status"] = filters.status.value

        if filters.dataset_type:
            query["dataset_type"] = filters.dataset_type

        if filters.created_by:
            query["created_by"] = filters.created_by

        if filters.date_from or filters.date_to:
            date_range = {}
            if filters.date_from:
                date_range["$gte"] = filters.date_from
            if filters.date_to:
                date_range["$lte"] = filters.date_to
            query["created_at"] = date_range

        if filters.search_text:
            query["$or"] = [
                {"name": {"$regex": filters.search_text, "$options": "i"}},
                {"description": {"$regex": filters.search_text, "$options": "i"}}
            ]

        return query

    @async_exception_handler(DatabaseException)
    @async_performance_monitor("dataset_update")
    async def update(self, dataset_id: str, updates: Dict[str, Any]) -> bool:
        """更新数据集"""
        try:
            if not ObjectId.is_valid(dataset_id):
                raise ValidationException(f"Invalid ObjectId format: {dataset_id}")

            updates["updated_at"] = datetime.utcnow()

            result = self.collection.update_one(
                {"_id": ObjectId(dataset_id)},
                {"$set": updates}
            )

            if result.matched_count == 0:
                raise DatasetNotFoundException(dataset_id)

            logger.info(f"Dataset updated: {dataset_id}")
            return True

        except (DatasetNotFoundException, ValidationException):
            raise
        except PyMongoError as e:
            raise DatabaseException(f"Failed to update dataset: {e}")

    @async_exception_handler(DatabaseException)
    @async_performance_monitor("dataset_delete")
    async def delete(self, dataset_id: str) -> bool:
        """删除数据集（软删除）"""
        try:
            if not ObjectId.is_valid(dataset_id):
                raise ValidationException(f"Invalid ObjectId format: {dataset_id}")

            result = self.collection.update_one(
                {"_id": ObjectId(dataset_id)},
                {"$set": {"status": DatasetStatus.DELETED.value, "updated_at": datetime.utcnow()}}
            )

            if result.matched_count == 0:
                raise DatasetNotFoundException(dataset_id)

            logger.info(f"Dataset soft deleted: {dataset_id}")
            return True

        except (DatasetNotFoundException, ValidationException):
            raise
        except PyMongoError as e:
            raise DatabaseException(f"Failed to delete dataset: {e}")

    @async_exception_handler(DatabaseException)
    async def count(self, filters: DatasetFilter) -> int:
        """计算数据集数量"""
        try:
            query = self._build_query(filters)
            return self.collection.count_documents(query)
        except PyMongoError as e:
            raise DatabaseException(f"Failed to count datasets: {e}")


class DatasetService:
    """数据集服务类 - 使用仓储模式和业务逻辑"""

    def __init__(self, repository: Optional[DatasetRepository] = None):
        """初始化数据集服务"""
        self.repository = repository or MongoDatasetRepository()
        self.validator_context = DatasetValidatorContext()

    @async_exception_handler
    @async_performance_monitor("create_dataset")
    async def create_dataset(self, dataset: Dataset) -> str:
        """创建数据集"""
        # 验证数据集类型
        if dataset.dataset_type not in self.validator_context.get_available_types():
            raise ValidationException(f"Unsupported dataset type: {dataset.dataset_type}")

        # 验证数据集名称唯一性
        await self._validate_dataset_name_unique(dataset.name)

        # 创建数据集
        dataset_id = await self.repository.create(dataset)

        # 异步验证数据集
        asyncio.create_task(self._validate_dataset_async(dataset_id, dataset))

        return dataset_id

    async def _validate_dataset_name_unique(self, name: str) -> None:
        """验证数据集名称唯一性"""
        filters = DatasetFilter(search_text=name)
        existing_count = await self.repository.count(filters)
        if existing_count > 0:
            raise DatasetAlreadyExistsException(name)

    async def _validate_dataset_async(self, dataset_id: str, dataset: Dataset) -> None:
        """异步验证数据集"""
        try:
            await self.repository.update(dataset_id, {"status": DatasetStatus.VALIDATING.value})

            # 这里应该根据实际的存储路径进行验证
            # 由于这是演示，我们跳过实际的验证过程
            is_valid = True  # await self.validator_context.validate_dataset(dataset.storage_path, dataset.dataset_type)

            if is_valid:
                await self.repository.update(dataset_id, {"status": DatasetStatus.ACTIVE.value})
            else:
                await self.repository.update(dataset_id, {"status": DatasetStatus.ERROR.value})

        except Exception as e:
            logger.error(f"Dataset validation failed for {dataset_id}: {e}")
            await self.repository.update(dataset_id, {"status": DatasetStatus.ERROR.value})

    @async_exception_handler
    @async_performance_monitor("get_dataset")
    async def get_dataset(self, dataset_id: str) -> Optional[Dataset]:
        """获取数据集"""
        return await self.repository.get_by_id(dataset_id)

    @async_exception_handler
    @async_performance_monitor("list_datasets")
    async def list_datasets(
        self,
        filters: Optional[DatasetFilter] = None,
        page: int = 1,
        page_size: int = 20
    ) -> Dict[str, Any]:
        """列出数据集"""
        filters = filters or DatasetFilter()

        skip = (page - 1) * page_size
        datasets = await self.repository.list(filters, skip, page_size)
        total = await self.repository.count(filters)

        return {
            "items": datasets,
            "total": total,
            "page": page,
            "page_size": page_size,
            "total_pages": (total + page_size - 1) // page_size
        }

    @async_exception_handler
    @async_performance_monitor("update_dataset_stats")
    @async_retry(max_attempts=3, delay=1.0)
    async def update_dataset_stats(
        self,
        dataset_id: str,
        train_images: int = 0,
        train_annotations: int = 0,
        train_size: int = 0,
        val_images: int = 0,
        val_annotations: int = 0,
        val_size: int = 0,
        test_images: int = 0,
        test_annotations: int = 0,
        test_size: int = 0
    ) -> bool:
        """更新数据集统计信息"""
        # 验证参数
        if any(count < 0 for count in [train_images, val_images, test_images]):
            raise ValidationException("Image counts cannot be negative")

        total_size = train_size + val_size + test_size
        total_images = train_images + val_images + test_images
        total_annotations = train_annotations + val_annotations + test_annotations

        updates = {
            "num_images": total_images,
            "num_annotations": total_annotations,
            "file_size": total_size,
            "splits": {
                "train": train_images,
                "val": val_images,
                "test": test_images
            },
            "stats_last_updated": datetime.utcnow()
        }

        return await self.repository.update(dataset_id, updates)

    @async_exception_handler
    @async_performance_monitor("delete_dataset")
    async def delete_dataset(self, dataset_id: str) -> bool:
        """删除数据集"""
        return await self.repository.delete(dataset_id)

    @async_exception_handler
    @async_performance_monitor("validate_dataset")
    async def validate_dataset(self, dataset_id: str) -> Dict[str, Any]:
        """验证数据集"""
        dataset = await self.get_dataset(dataset_id)
        if not dataset:
            raise DatasetNotFoundException(dataset_id)

        # 更新验证状态
        await self.repository.update(dataset_id, {"status": DatasetStatus.VALIDATING.value})

        try:
            # 验证数据集格式
            is_valid = await self._perform_validation(dataset)

            # 更新验证结果
            validation_result = {
                "is_valid": is_valid,
                "validated_at": datetime.utcnow(),
                "validation_message": "Validation completed successfully" if is_valid else "Validation failed"
            }

            await self.repository.update(dataset_id, {
                "validation_result": validation_result,
                "status": DatasetStatus.ACTIVE.value if is_valid else DatasetStatus.ERROR.value
            })

            return validation_result

        except Exception as e:
            logger.error(f"Dataset validation failed: {e}")
            await self.repository.update(dataset_id, {
                "status": DatasetStatus.ERROR.value,
                "validation_error": str(e)
            })
            raise BusinessLogicException(f"Dataset validation failed: {str(e)}")

    async def _perform_validation(self, dataset: Dataset) -> bool:
        """执行数据集验证"""
        # 这里实现具体的验证逻辑
        # 由于这是演示，我们返回True
        return True


# 全局数据集服务实例
dataset_service = DatasetService()
