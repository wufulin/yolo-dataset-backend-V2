"""Pydantic schemas for image-related API requests and responses."""
from datetime import datetime
from typing import Any, Dict, List

from pydantic import BaseModel, Field


class ImageResponse(BaseModel):
    """Schema for image response."""
    id: str = Field(..., alias="_id", description="Image ID")
    dataset_id: str = Field(..., description="Parent dataset ID")
    filename: str = Field(..., description="Image filename")
    file_url: str = Field(..., description="Image URL")
    width: int = Field(..., description="Image width")
    height: int = Field(..., description="Image height")
    split: str = Field(..., description="Dataset split")
    annotations: List[Dict[str, Any]] = Field(..., description="Image annotations")
    created_at: datetime = Field(..., description="Creation timestamp")

    model_config = {
        "from_attributes": True,
        "populate_by_name": True
    }
