"""Upload session model."""
from datetime import datetime
from typing import List, Optional

from bson import ObjectId
from pydantic import BaseModel, Field

from .base import PyObjectId


class UploadSession(BaseModel):
    """Upload session model."""
    upload_id: str = Field(..., description="Upload session ID")
    user_id: str = Field(..., description="User ID")
    filename: str = Field(..., description="Original filename")
    file_size: int = Field(..., ge=0, description="File total size")
    total_chunks: int = Field(..., ge=1, description="Total number of chunks")
    chunk_size: int = Field(..., ge=1024, description="Chunk size in bytes")
    received_chunks: List[int] = Field(default_factory=list, description="Received chunk indices")
    temp_path: str = Field(..., description="Temporary file path")
    status: str = Field("uploading", description="Upload status")
    dataset_id: Optional[PyObjectId] = Field(None, description="Associated dataset ID")
    error_message: Optional[str] = Field(None, description="Error message")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: datetime = Field(description="Session expiration time")

    model_config = {
        "arbitrary_types_allowed": True,
        "json_encoders": {ObjectId: str},
        "json_schema_extra": {
            "example": {
                "upload_id": "1234567890",
                "user_id": "1234567890",
                "filename": "example.zip",
                "file_size": 1000000,
                "total_chunks": 10,
                "chunk_size": 102400,
                "received_chunks": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                "temp_path": "/tmp/example.zip",
                "status": "uploading",
                "dataset_id": "1234567890",
                "error_message": None,
                "created_at": "2021-01-01T00:00:00.000Z",
                "updated_at": "2021-01-01T00:00:00.000Z",
                "expires_at": "2021-01-01T00:00:00.000Z"
            }
        }
    }

