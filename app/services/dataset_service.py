"""Service layer for handling dataset operations with structured patterns and error handling."""
import asyncio
import logging
from datetime import datetime
from enum import Enum
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
    """Enumeration of dataset lifecycle states."""
    ACTIVE = "active"
    PROCESSING = "processing"
    VALIDATING = "validating"
    ERROR = "error"
    DELETED = "deleted"


class DatasetFilter(BaseModel):
    """Filter parameters accepted by dataset listing endpoints."""
    status: Optional[DatasetStatus] = None
    dataset_type: Optional[str] = None
    created_by: Optional[str] = None
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None
    search_text: Optional[str] = None


class DatasetRepository(Protocol):
    """Protocol describing the repository contract for datasets."""

    async def create(self, dataset: Dataset) -> str: ...

    async def get_by_id(self, dataset_id: str) -> Optional[Dataset]: ...

    async def list(self, filters: DatasetFilter, skip: int, limit: int) -> List[Dataset]: ...

    async def update(self, dataset_id: str, updates: Dict[str, Any]) -> bool: ...

    async def delete(self, dataset_id: str) -> bool: ...

    async def count(self, filters: DatasetFilter) -> int: ...


class MongoDatasetRepository:
    """MongoDB-backed DatasetRepository implementation."""

    def __init__(self):
        self.db = db_service
        self.collection = self.db.datasets
        self.validator_context = DatasetValidatorContext()

    @async_exception_handler(DatabaseException)
    @async_performance_monitor("dataset_create")
    async def create(self, dataset: Dataset) -> str:
        """Create a dataset document."""
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
        """Return a dataset by its identifier."""
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
        """List datasets that match the provided filters."""
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
        """Convert incoming filters into a MongoDB query."""
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
        """Update a dataset document."""
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
        """Soft-delete a dataset by marking its status as deleted."""
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
        """Return total dataset count for the given filters."""
        try:
            query = self._build_query(filters)
            return self.collection.count_documents(query)
        except PyMongoError as e:
            raise DatabaseException(f"Failed to count datasets: {e}")


class DatasetService:
    """High-level dataset service that coordinates repository and validation logic."""

    def __init__(self, repository: Optional[DatasetRepository] = None):
        """Initialize the service with a repository implementation."""
        self.repository = repository or MongoDatasetRepository()
        self.validator_context = DatasetValidatorContext()

    @async_exception_handler
    @async_performance_monitor("create_dataset")
    async def create_dataset(self, dataset: Dataset) -> str:
        """Create a dataset and schedule async validation."""
        # Ensure dataset type is supported
        if dataset.dataset_type not in self.validator_context.get_available_types():
            raise ValidationException(f"Unsupported dataset type: {dataset.dataset_type}")

        # Ensure name uniqueness across datasets
        await self._validate_dataset_name_unique(dataset.name)

        # Persist dataset metadata
        dataset_id = await self.repository.create(dataset)

        # Trigger asynchronous validation
        asyncio.create_task(self._validate_dataset_async(dataset_id, dataset))

        return dataset_id

    async def _validate_dataset_name_unique(self, name: str) -> None:
        """Ensure that no other dataset uses the same name."""
        filters = DatasetFilter(search_text=name)
        existing_count = await self.repository.count(filters)
        if existing_count > 0:
            raise DatasetAlreadyExistsException(name)

    async def _validate_dataset_async(self, dataset_id: str, dataset: Dataset) -> None:
        """Validate dataset contents asynchronously."""
        try:
            await self.repository.update(dataset_id, {"status": DatasetStatus.VALIDATING.value})

            # Placeholder for actual validation process. In production this should
            # inspect storage_path + dataset type.
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
        """Fetch a dataset by id via repository."""
        return await self.repository.get_by_id(dataset_id)

    @async_exception_handler
    @async_performance_monitor("list_datasets")
    async def list_datasets(
        self,
        filters: Optional[DatasetFilter] = None,
        page: int = 1,
        page_size: int = 20
    ) -> Dict[str, Any]:
        """Return paginated datasets for UI consumption."""
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
        """Update aggregate dataset statistics."""
        # Validate incoming counts before persisting
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
        """Soft-delete dataset via repository."""
        return await self.repository.delete(dataset_id)

    @async_exception_handler
    @async_performance_monitor("validate_dataset")
    async def validate_dataset(self, dataset_id: str) -> Dict[str, Any]:
        """Run validation pipeline for a dataset."""
        dataset = await self.get_dataset(dataset_id)
        if not dataset:
            raise DatasetNotFoundException(dataset_id)

        # Mark dataset as being validated
        await self.repository.update(dataset_id, {"status": DatasetStatus.VALIDATING.value})

        try:
            # Run validation procedure against dataset data
            is_valid = await self._perform_validation(dataset)

            # Store validation output on dataset document
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
        """Placeholder for actual dataset validation logic."""
        # TODO: implement real validation. Demo returns True to unblock flow.
        return True


# Global dataset service instance
dataset_service = DatasetService()
