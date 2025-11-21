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
    """数据库模型协议"""
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
    """时间戳混入类"""
    created_at: datetime = Field(default_factory=datetime.utcnow, description="创建时间")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="更新时间")

    @model_validator(mode='before')
    @classmethod
    def set_timestamps(cls, values):
        """设置时间戳"""
        current_time = datetime.utcnow()
        if isinstance(values, dict):
            if 'created_at' not in values:
                values['created_at'] = current_time
            values['updated_at'] = current_time
        return values


class AuditMixin(BaseModel):
    """审计混入类"""
    created_by: str = Field(description="创建者")
    updated_by: str = Field(description="最后更新者")

    @model_validator(mode='before')
    @classmethod
    def set_audit_info(cls, values):
        """设置审计信息"""
        if isinstance(values, dict) and 'created_by' not in values:
            # 这里应该从认证上下文中获取用户信息
            values['created_by'] = values.get('updated_by', 'system')
        return values


class StatusEnum(str, Enum):
    """状态枚举"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    PENDING = "pending"
    PROCESSING = "processing"
    ERROR = "error"
    DELETED = "deleted"


class BaseDBModel(BaseModel):
    """基础数据库模型"""

    model_config = ConfigDict(
        populate_by_name=True,
        arbitrary_types_allowed=True,
        json_encoders={
            ObjectId: str,
            datetime: lambda v: v.isoformat() if v else None
        }
    )

    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id", description="数据库ID")
    status: StatusEnum = Field(default=StatusEnum.ACTIVE, description="状态")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="创建时间")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="更新时间")
    created_by: str = Field(description="创建者")
    updated_by: str = Field(description="最后更新者")

    @field_validator("id", mode="before")
    @classmethod
    def validate_object_id(cls, v):
        """验证并转换ObjectId"""
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
        """自动更新修改时间"""
        # 如果是更新操作，自动设置updated_at
        if info.data and 'id' in info.data:
            return datetime.utcnow()
        return v

    def to_dict(self) -> dict:
        """转换为字典格式"""
        return self.model_dump(by_alias=True)

    def to_mongo_dict(self) -> dict:
        """转换为MongoDB字典格式（处理ObjectId）"""
        data = self.to_dict()
        if 'id' in data and data['id']:
            data['_id'] = data.pop('id')
        return data

    @classmethod
    def from_mongo_dict(cls, data: dict) -> 'BaseDBModel':
        """从MongoDB字典创建模型实例"""
        if '_id' in data:
            data['id'] = data.pop('_id')
        return cls(**data)


class PaginationQuery(BaseModel):
    """分页查询参数"""
    page: int = Field(default=1, ge=1, description="页码")
    page_size: int = Field(default=20, ge=1, le=100, description="每页大小")

    @property
    def skip(self) -> int:
        """获取跳过数量"""
        return (self.page - 1) * self.page_size

    @property
    def limit(self) -> int:
        """获取限制数量"""
        return self.page_size


class PaginationResult(BaseModel, Generic[T]):
    """分页结果"""
    items: list[T] = Field(description="数据项列表")
    total: int = Field(description="总数量")
    page: int = Field(description="当前页码")
    page_size: int = Field(description="每页大小")
    total_pages: int = Field(description="总页数")
    has_next: bool = Field(description="是否有下一页")
    has_prev: bool = Field(description="是否有上一页")

    @model_validator(mode='after')
    def validate_pagination(self):
        """验证分页参数"""
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
        """创建分页结果"""
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
    """错误响应模型"""
    error_code: str = Field(description="错误代码")
    message: str = Field(description="错误消息")
    details: dict = Field(default_factory=dict, description="错误详情")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="时间戳")
    request_id: str = Field(description="请求ID")


class SuccessResponse(BaseModel, Generic[T]):
    """成功响应模型"""
    data: T = Field(description="响应数据")
    message: str = Field(default="Success", description="响应消息")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="时间戳")
    request_id: str = Field(description="请求ID")


class ModelFactory:
    """模型工厂类"""

    _model_registry: dict[str, type] = {}

    @classmethod
    def register_model(cls, model_name: str, model_class: type):
        """注册模型类"""
        cls._model_registry[model_name] = model_class

    @classmethod
    def get_model(cls, model_name: str) -> type:
        """获取模型类"""
        if model_name not in cls._model_registry:
            raise ValueError(f"Model {model_name} not registered")
        return cls._model_registry[model_name]

    @classmethod
    def create_model(cls, model_name: str, **kwargs) -> DatabaseModel:
        """创建模型实例"""
        model_class = cls.get_model(model_name)
        return model_class(**kwargs)

