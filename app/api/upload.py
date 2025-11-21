"""File upload API endpoints."""
import logging
import os
import uuid
from datetime import datetime

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

from app.auth import authenticate_user
from app.config import settings
from app.schemas.upload import UploadComplete, UploadResponse
from app.services import upload_service
from app.services.redis_service import (
    add_chunk,
    cleanup_session,
    create_session,
    delete_session,
    get_redis_session_manager,
    get_session,
    recover_session,
    redis_session_manager,
)
from app.utils.file_utils import ensure_directory, safe_remove
from app.utils.logger import get_logger

logger: logging.Logger = get_logger(__name__)

router = APIRouter()


@router.post("/upload/start", response_model=UploadResponse)
async def start_upload(
    filename: str = Form(...),
    total_size: int = Form(...),
    total_chunks: int = Form(...),
    chunk_size: int = Form(...),
    username: str = Depends(authenticate_user)
):
    """
    Start a new file upload session.

    Args:
        filename: Original filename
        total_size: Total file size in bytes
        total_chunks: Total number of chunks
        chunk_size: Size of each chunk in bytes

    Returns:
        UploadResponse: Upload session information
    """
    # Validate file size
    if total_size > settings.max_upload_size:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large. Maximum size is {settings.max_upload_size} bytes"
        )

    # Create upload session
    upload_id = str(uuid.uuid4())
    temp_dir = os.path.join(settings.temp_dir, upload_id)
    ensure_directory(temp_dir)
    temp_file = os.path.join(temp_dir, filename)

    # Create session in Redis
    success = await create_session(
        upload_id=upload_id,
        filename=filename,
        total_size=total_size,
        total_chunks=total_chunks,
        chunk_size=chunk_size,
        temp_dir=temp_dir,
        temp_file=temp_file,
        user_id=username
    )

    if not success:
        # Cleanup on failure
        safe_remove(temp_dir)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create upload session"
        )

    logger.info("Created upload session %s for user %s", upload_id, username)

    return UploadResponse(
        upload_id=upload_id,
        chunk_size=chunk_size,
        total_chunks=total_chunks
    )


@router.post("/upload/chunk/{upload_id}/{chunk_index}")
async def upload_chunk(
    upload_id: str,
    chunk_index: int,
    file: UploadFile = File(...),
    username: str = Depends(authenticate_user)
):
    """
    Upload a file chunk.

    Args:
        upload_id: Upload session ID
        chunk_index: Index of the chunk
        file: Uploaded file chunk

    Returns:
        dict: Upload status
    """
    # Get session from Redis
    session = await get_session(upload_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Upload session not found"
        )

    # Validate chunk index
    if chunk_index < 0 or chunk_index >= session["total_chunks"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid chunk index"
        )

    # Check if chunk already exists (for recovery)
    received_chunks = session.get("received_chunks", set())
    if chunk_index in received_chunks:
        logger.info("Chunk already present for %s:%s", upload_id, chunk_index)
        return {"status": "success", "chunk": chunk_index, "recovered": True}

    # Save chunk
    chunk_path = f"{session['temp_file']}.part{chunk_index}"

    try:
        with open(chunk_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        # Update session in Redis
        success = await add_chunk(upload_id, chunk_index)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to update upload session"
            )

        logger.info("Chunk %s uploaded for session %s", chunk_index, upload_id)
        return {"status": "success", "chunk": chunk_index}

    except Exception as e:
        logger.error("Chunk upload failed for %s:%s: %s", upload_id, chunk_index, e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save chunk: {str(e)}"
        )


@router.post("/upload/complete/{upload_id}")
async def complete_upload(
    upload_id: str,
    upload_complete: UploadComplete,
    username: str = Depends(authenticate_user)
):
    """
    Complete file upload and process dataset.

    Args:
        upload_id: Upload session ID
        upload_complete: Upload completion data

    Returns:
        dict: Processing status
    """
    # Get session from Redis
    session = await get_session(upload_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Upload session not found"
        )

    # Check if all chunks are received
    received_chunks = session.get("received_chunks", set())
    if len(received_chunks) != session["total_chunks"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Not all chunks received"
        )

    # Reassemble file
    try:
        with open(session["temp_file"], "wb") as output:
            for i in range(session["total_chunks"]):
                chunk_path = f"{session['temp_file']}.part{i}"
                with open(chunk_path, "rb") as chunk:
                    output.write(chunk.read())
                safe_remove(chunk_path)

        # Process the dataset
        result = await upload_service.process_dataset(
            session["temp_file"],
            upload_complete.dataset_info,
        )

        # Cleanup session after successful processing
        # await cleanup_session(upload_id)

        logger.info("Upload completed for session %s", upload_id)
        return result

    except Exception as e:
        # Cleanup on error
        await cleanup_session(upload_id)
        logger.error("Dataset processing failed for session %s: %s", upload_id, e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process dataset: {str(e)}"
        )


@router.get("/upload/status/{upload_id}")
async def get_upload_status(
    upload_id: str,
    username: str = Depends(authenticate_user)
):
    """
    Get upload session status.

    Args:
        upload_id: Upload session ID

    Returns:
        dict: Upload status information
    """
    session = await get_session(upload_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Upload session not found"
        )

    received_chunks = session.get("received_chunks", set())
    total_chunks = session.get("total_chunks", 0)

    return {
        "upload_id": upload_id,
        "filename": session.get("filename"),
        "total_chunks": total_chunks,
        "received_chunks": len(received_chunks),
        "completion_percentage": (len(received_chunks) / total_chunks * 100) if total_chunks > 0 else 0,
        "status": session.get("status", "uploading"),
        "created_at": session.get("created_at"),
        "expires_at": session.get("expires_at"),
        "temp_file": session.get("temp_file")
    }


@router.post("/upload/recover/{upload_id}")
async def recover_upload_session(
    upload_id: str,
    username: str = Depends(authenticate_user)
):
    """
    Recover an upload session.

    Args:
        upload_id: Upload session ID

    Returns:
        dict: Recovery status
    """
    # Try to recover session
    session = await recover_session(upload_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Upload session not found or expired"
        )

    # Get received chunks
    received_chunks = session.get("received_chunks", set())

    logger.info("Upload recovery succeeded for session %s", upload_id)

    return {
        "status": "recovered",
        "upload_id": upload_id,
        "filename": session.get("filename"),
        "total_chunks": session.get("total_chunks"),
        "received_chunks": list(received_chunks),
        "missing_chunks": [
            i for i in range(session.get("total_chunks", 0))
            if i not in received_chunks
        ],
        "temp_file": session.get("temp_file"),
        "recovered_at": datetime.utcnow().isoformat()
    }


@router.delete("/upload/{upload_id}")
async def cancel_upload(
    upload_id: str,
    username: str = Depends(authenticate_user)
):
    """
    Cancel an upload session and cleanup resources.

    Args:
        upload_id: Upload session ID

    Returns:
        dict: Cancellation status
    """
    success = await cleanup_session(upload_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Upload session not found"
        )

    logger.info("Upload cancelled for session %s", upload_id)
    return {"status": "cancelled", "upload_id": upload_id}


@router.get("/upload/stats")
async def get_upload_stats(
    username: str = Depends(authenticate_user)
):
    """
    Get upload statistics.

    Returns:
        dict: Upload statistics
    """
    redis_manager = await get_redis_session_manager()
    stats = await redis_manager.get_session_stats()

    return {
        "redis_stats": stats,
        "timestamp": datetime.utcnow().isoformat()
    }
