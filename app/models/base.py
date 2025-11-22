"""Base models and utilities for MongoDB integration."""
from datetime import datetime
from enum import Enum
from typing import Any, Generic, Protocol, TypeVar, runtime_checkable

from bson import ObjectId
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic.json_schema import JsonSchemaValue

from app.core.exceptions import ValidationException

T = TypeVar('T')


@runtime_checkable
class DatabaseModel(Protocol):
    """Protocol implemented by persisted models."""
    id: Any


class PyObjectId(ObjectId):
    """Custom type for handling MongoDB ObjectId with Pydantic v2."""

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type: Any, handler: Any):
        """Get Pydantic core schema for ObjectId."""
        from pydantic_core import core_schema
        return core_schema.union_schema([
            core_schema.is_instance_schema(ObjectId),
            core_schema.chain_schema([
                core_schema.str_schema(),
                core_schema.no_info_plain_validator_function(cls.validate),
            ])
        ], serialization=core_schema.plain_serializer_function_ser_schema(
            lambda x: str(x),
            when_used='json'
        ))

    @classmethod
    def validate(cls, v):
        """Validate ObjectId."""
        if isinstance(v, ObjectId):
            return v
        if isinstance(v, str) and ObjectId.is_valid(v):
            return ObjectId(v)
        raise ValueError("Invalid ObjectId")

    @classmethod
    def __get_pydantic_json_schema__(cls, schema, handler):
        """Get JSON schema for ObjectId."""
        return {"type": "string", "format": "objectid"}


class TimestampMixin(BaseModel):
    """Mixin that automatically manages created/updated timestamps."""
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Created at")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last updated at")

    @model_validator(mode='before')
    @classmethod
    def set_timestamps(cls, values):
        """Populate timestamp fields when absent."""
        current_time = datetime.utcnow()
        if isinstance(values, dict):
            if 'created_at' not in values:
                values['created_at'] = current_time
            values['updated_at'] = current_time
        return values


class AuditMixin(BaseModel):
    """Mixin that stores creator/updater identifiers."""
    created_by: str = Field(description="Created by user id")
    updated_by: str = Field(description="Updated by user id")

    @model_validator(mode='before')
    @classmethod
    def set_audit_info(cls, values):
        """Populate audit fields when missing."""
        if isinstance(values, dict) and 'created_by' not in values:
            # This should pull from authentication context in real usage
            values['created_by'] = values.get('updated_by', 'system')
        return values


class StatusEnum(str, Enum):
    """Common status enumeration shared by models."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    PENDING = "pending"
    PROCESSING = "processing"
    ERROR = "error"
    DELETED = "deleted"


class BaseDBModel(BaseModel):
    """Common base class for persisted MongoDB models."""

    model_config = ConfigDict(
        populate_by_name=True,
        arbitrary_types_allowed=True,
        json_encoders={
            ObjectId: str,
            datetime: lambda v: v.isoformat() if v else None
        }
    )

    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id", description="Database identifier")
    status: StatusEnum = Field(default=StatusEnum.ACTIVE, description="Record status")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation time")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last modification time")
    created_by: str = Field(description="Creator user id")
    updated_by: str = Field(description="Updater user id")

    @field_validator("id", mode="before")
    @classmethod
    def validate_object_id(cls, v):
        """Validate and coerce values into ObjectId."""
        if v is None:
            return PyObjectId()
        if isinstance(v, ObjectId):
            return v
        if isinstance(v, str):
            if ObjectId.is_valid(v):
                return ObjectId(v)
            raise ValidationException(f"Invalid ObjectId format: {v}")
        raise ValidationException(f"Invalid ObjectId type: {type(v)}")

    @field_validator("updated_at", mode="before")
    @classmethod
    def update_timestamp(cls, v, info):
        """Ensure updated_at reflects the latest mutation."""
        # When updating, automatically set updated_at
        if info.data and 'id' in info.data:
            return datetime.utcnow()
        return v

    def to_dict(self) -> dict:
        """Return dict representation with alias support."""
        return self.model_dump(by_alias=True)

    def to_mongo_dict(self) -> dict:
        """Return dict suitable for MongoDB writes, renaming id -> _id."""
        data = self.to_dict()
        if 'id' in data and data['id']:
            data['_id'] = data.pop('id')
        return data

    @classmethod
    def from_mongo_dict(cls, data: dict) -> 'BaseDBModel':
        """Instantiate from MongoDB result payload."""
        if '_id' in data:
            data['id'] = data.pop('_id')
        return cls(**data)


class PaginationQuery(BaseModel):
    """Common pagination query parameters."""
    page: int = Field(default=1, ge=1, description="Requested page number")
    page_size: int = Field(default=20, ge=1, le=100, description="Results per page")

    @property
    def skip(self) -> int:
        """Return number of records to skip."""
        return (self.page - 1) * self.page_size

    @property
    def limit(self) -> int:
        """Return limit for query."""
        return self.page_size


class PaginationResult(BaseModel, Generic[T]):
    """Standard paginated response payload."""
    items: list[T] = Field(description="Items on the current page")
    total: int = Field(description="Total item count")
    page: int = Field(description="Current page number")
    page_size: int = Field(description="Items per page")
    total_pages: int = Field(description="Total number of pages")
    has_next: bool = Field(description="Whether a next page exists")
    has_prev: bool = Field(description="Whether a previous page exists")

    @model_validator(mode='after')
    def validate_pagination(self):
        """Validate pagination invariants."""
        if self.page < 1:
            raise ValidationException("Page must be >= 1")
        if self.page_size < 1:
            raise ValidationException("Page size must be >= 1")
        if self.page > self.total_pages and self.total_pages > 0:
            raise ValidationException(f"Page {self.page} exceeds total pages {self.total_pages}")
        return self

    @classmethod
    def create(
        cls,
        items: list[T],
        total: int,
        page: int,
        page_size: int
    ) -> 'PaginationResult[T]':
        """Factory helper for building paginated responses."""
        total_pages = (total + page_size - 1) // page_size if total > 0 else 0

        return cls(
            items=items,
            total=total,
            page=page,
            page_size=page_size,
            total_pages=total_pages,
            has_next=page < total_pages,
            has_prev=page > 1
        )


class ErrorResponse(BaseModel):
    """Standard error response schema."""
    error_code: str = Field(description="Application-specific error code")
    message: str = Field(description="Human-readable error message")
    details: dict = Field(default_factory=dict, description="Additional error context")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Time error was generated")
    request_id: str = Field(description="Request correlation identifier")


class SuccessResponse(BaseModel, Generic[T]):
    """Standard success response wrapper."""
    data: T = Field(description="Payload data")
    message: str = Field(default="Success", description="Success message")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")
    request_id: str = Field(description="Request correlation identifier")


class ModelFactory:
    """Registry for resolving models by name."""

    _model_registry: dict[str, type] = {}

    @classmethod
    def register_model(cls, model_name: str, model_class: type):
        """Register a model class."""
        cls._model_registry[model_name] = model_class

    @classmethod
    def get_model(cls, model_name: str) -> type:
        """Retrieve a registered model class."""
        if model_name not in cls._model_registry:
            raise ValueError(f"Model {model_name} not registered")
        return cls._model_registry[model_name]

    @classmethod
    def create_model(cls, model_name: str, **kwargs) -> DatabaseModel:
        """Instantiate a registered model."""
        model_class = cls.get_model(model_name)
        return model_class(**kwargs)
