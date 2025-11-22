"""Upload-related schemas."""
from typing import Optional

from pydantic import BaseModel, Field

from app.schemas.dataset import DatasetCreate


class UploadResponse(BaseModel):
    """Schema for upload response."""
    upload_id: str = Field(..., description="Upload session ID")
    chunk_size: int = Field(..., description="Chunk size in bytes")
    total_chunks: int = Field(..., description="Total number of chunks")


class UploadComplete(BaseModel):
    """Schema for upload completion."""
    filename: str = Field(..., description="Original filename")
    dataset_info: Optional[DatasetCreate] = Field(None, description="Dataset information")
