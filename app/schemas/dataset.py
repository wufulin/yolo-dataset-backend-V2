"""Pydantic schemas for dataset-related API requests and responses."""
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class DatasetCreate(BaseModel):
    """Schema for dataset creation."""
    name: str = Field(..., min_length=1, max_length=100, description="Dataset name")
    description: Optional[str] = Field(None, max_length=500, description="Dataset description")
    dataset_type: str = Field(..., description="Dataset type: detect/obb/segment/pose/classify")
    class_names: Optional[List[str]] = Field(default=[], description="List of class names (optional)")

    class Config:
        json_schema_extra = {
            "example": {
                "name": "My YOLO Dataset",
                "description": "A custom object detection dataset",
                "dataset_type": "detect",
                "class_names": ["person", "car", "dog", "cat"]
            }
        }


class DatasetResponse(BaseModel):
    """Schema for dataset response."""
    id: str = Field(..., alias="_id", description="Dataset ID")
    name: str = Field(..., description="Dataset name")
    description: Optional[str] = Field(None, description="Dataset description")
    dataset_type: str = Field(..., description="Dataset type")
    class_names: List[str] = Field(..., description="List of class names")
    num_images: int = Field(..., description="Number of images")
    num_annotations: int = Field(..., description="Total annotations")
    splits: Dict[str, int] = Field(..., description="Split counts")
    status: str = Field(..., description="Dataset status")
    error_message: Optional[str] = Field(None, description="Error message if status is error")
    file_size: int = Field(..., description="File size in bytes")
    created_by: str = Field(..., description="User who created the dataset")
    version: int = Field(..., description="Dataset version")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Update timestamp")

    model_config = {
        "from_attributes": True,
        "populate_by_name": True
    }


class PaginatedResponse(BaseModel):
    """Schema for paginated responses."""
    items: List[Any] = Field(..., description="List of items")
    total: int = Field(..., description="Total number of items")
    page: int = Field(..., description="Current page number")
    page_size: int = Field(..., description="Page size")
    total_pages: int = Field(..., description="Total number of pages")
