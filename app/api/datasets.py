"""Dataset management API endpoints."""
from datetime import datetime
import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.auth import authenticate_user
from app.models.dataset import Dataset
from app.schemas.dataset import DatasetCreate, DatasetResponse, PaginatedResponse
from app.schemas.image import ImageResponse
from app.services.dataset_service import dataset_service
from app.services.image_service import image_service
from app.services.minio_service import minio_service
from app.utils.logger import get_logger

logger: logging.Logger = get_logger(__name__)

router = APIRouter()


@router.post("/datasets", response_model=DatasetResponse, status_code=status.HTTP_201_CREATED)
async def create_dataset(
    dataset_data: DatasetCreate,
    username: str = Depends(authenticate_user)
):
    """
    Create a new dataset.

    Args:
        dataset_data: Dataset creation data
        username: Authenticated username

    Returns:
        DatasetResponse: Created dataset information
    """
    logger.info(f"Creating dataset '{dataset_data.name}' of type '{dataset_data.dataset_type}' by user '{username}'")

    # Validate dataset_type
    valid_types = ['detect', 'obb', 'segment', 'pose', 'classify']
    if dataset_data.dataset_type not in valid_types:
        logger.error(f"Invalid dataset_type: {dataset_data.dataset_type}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid dataset_type. Must be one of: {', '.join(valid_types)}"
        )

    # Create Dataset model
    dataset = Dataset(
        name=dataset_data.name,
        description=dataset_data.description,
        dataset_type=dataset_data.dataset_type,
        class_names=dataset_data.class_names if dataset_data.class_names else [],
        num_images=0,
        num_annotations=0,
        splits={"train": 0, "val": 0, "test": 0},
        status="active",
        error_message=None,
        file_size=0,
        storage_path=None,
        created_by=username,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
        version=1
    )

    try:
        # Create dataset in MongoDB
        dataset_id = await dataset_service.create_dataset(dataset)

        # Retrieve and return created dataset
        created_dataset = await dataset_service.get_dataset(dataset_id)
        if not created_dataset:
            logger.error(f"Failed to retrieve created dataset with ID: {dataset_id}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to retrieve created dataset with ID: {dataset_id}"
            )

        # Convert Dataset object to dict and ensure _id is present
        dataset_dict = created_dataset.model_dump(by_alias=True) if hasattr(created_dataset, 'model_dump') else created_dataset.dict(by_alias=True)
        # Ensure _id is present for DatasetResponse (field name is 'id', alias is '_id')
        if '_id' not in dataset_dict or dataset_dict['_id'] is None:
            # Try to get id from the dataset object, or use the dataset_id parameter
            if hasattr(created_dataset, 'id') and created_dataset.id:
                dataset_dict['_id'] = str(created_dataset.id)
            else:
                dataset_dict['_id'] = dataset_id
        else:
            dataset_dict['_id'] = str(dataset_dict['_id'])
        
        logger.info(f"Dataset '{dataset_data.name}' created successfully with ID: {dataset_id}")
        return DatasetResponse(**dataset_dict)
    except HTTPException:
        # Re-raise HTTPException as-is
        raise
    except Exception as e:
        # Handle all other errors
        logger.error(f"Failed to create dataset: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create dataset: {str(e)}"
        )


@router.get("/datasets", response_model=PaginatedResponse)
async def list_datasets(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Page size"),
    username: str = Depends(authenticate_user)
):
    """
    List all datasets with pagination.

    Args:
        page: Page number (starting from 1)
        page_size: Number of items per page

    Returns:
        PaginatedResponse: Paginated list of datasets
    """
    logger.info(f"Listing datasets: page={page}, page_size={page_size}")

    try:
        result = await dataset_service.list_datasets(page=page, page_size=page_size)

        # Convert Dataset objects to dicts with _id
        items = []
        for dataset in result['items']:
            dataset_dict = dataset.model_dump(by_alias=True) if hasattr(dataset, 'model_dump') else dataset.dict(by_alias=True)
            # Ensure _id is present (it should be there from by_alias=True, but check just in case)
            if '_id' not in dataset_dict or dataset_dict['_id'] is None:
                # Try to get id from the dataset object (field name is 'id', alias is '_id')
                if hasattr(dataset, 'id') and dataset.id:
                    dataset_dict['_id'] = str(dataset.id)
            items.append(dataset_dict)

        logger.info(f"Retrieved {len(items)} datasets (total: {result['total']})")
        return PaginatedResponse(
            items=items,
            total=result['total'],
            page=result['page'],
            page_size=result['page_size'],
            total_pages=result['total_pages']
        )
    except Exception as e:
        logger.error(f"Failed to list datasets: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list datasets: {str(e)}"
        )


@router.get("/datasets/{dataset_id}", response_model=DatasetResponse)
async def get_dataset(
    dataset_id: str,
    username: str = Depends(authenticate_user)
):
    """
    Get dataset by ID.

    Args:
        dataset_id: Dataset ID

    Returns:
        DatasetResponse: Dataset information
    """
    logger.info(f"Retrieving dataset with ID: {dataset_id}")

    try:
        dataset = await dataset_service.get_dataset(dataset_id)
        if not dataset:
            logger.error(f"Dataset not found with ID: {dataset_id}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Dataset not found"
            )

        # Convert Dataset object to dict and ensure _id is present
        dataset_dict = dataset.model_dump(by_alias=True) if hasattr(dataset, 'model_dump') else dataset.dict(by_alias=True)
        # Ensure _id is present for DatasetResponse (field name is 'id', alias is '_id')
        if '_id' not in dataset_dict or dataset_dict['_id'] is None:
            # Try to get id from the dataset object, or use the dataset_id parameter
            if hasattr(dataset, 'id') and dataset.id:
                dataset_dict['_id'] = str(dataset.id)
            else:
                dataset_dict['_id'] = dataset_id
        else:
            dataset_dict['_id'] = str(dataset_dict['_id'])
        
        logger.info(f"Retrieved dataset: {dataset.name} (ID: {dataset_id})")
        return DatasetResponse(**dataset_dict)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get dataset {dataset_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve dataset: {str(e)}"
        )


@router.get("/datasets/{dataset_id}/images", response_model=PaginatedResponse)
async def get_dataset_images(
    dataset_id: str,
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Page size"),
    split: Optional[str] = Query(None, description="Filter by split"),
    username: str = Depends(authenticate_user)
):
    """
    Get images for a specific dataset.

    Args:
        dataset_id: Dataset ID
        page: Page number
        page_size: Page size
        split: Optional split filter

    Returns:
        PaginatedResponse: Paginated list of images
    """
    logger.info(f"Getting images for dataset {dataset_id}: page={page}, page_size={page_size}, split={split}")

    try:
        # Verify dataset exists
        dataset = await dataset_service.get_dataset(dataset_id)
        if not dataset:
            logger.error(f"Dataset not found with ID: {dataset_id}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Dataset not found"
            )

        skip = (page - 1) * page_size
        images = image_service.get_images_by_dataset(
            dataset_id, skip=skip, limit=page_size, split=split
        )

        # Generate presigned URLs for images
        for image in images:
            image["file_url"] = minio_service.get_file_url(image["file_path"])

        total = image_service.count_images(dataset_id, split=split)

        logger.info(f"Retrieved {len(images)} images for dataset {dataset_id} (total: {total})")
        return PaginatedResponse(
            items=images,
            total=total,
            page=page,
            page_size=page_size,
            total_pages=(total + page_size - 1) // page_size
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get images for dataset {dataset_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve images: {str(e)}"
        )


@router.get("/images/{image_id}", response_model=ImageResponse)
async def get_image(
    image_id: str,
    username: str = Depends(authenticate_user)
):
    """
    Get image by ID with annotations.

    Args:
        image_id: Image ID

    Returns:
        ImageResponse: Image information with annotations
    """
    logger.info(f"Retrieving image with ID: {image_id}")

    try:
        image = image_service.get_image(image_id)
        if not image:
            logger.error(f"Image not found with ID: {image_id}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Image not found"
            )

        # Generate presigned URL
        image["file_url"] = minio_service.get_file_url(image["file_path"])

        logger.info(f"Retrieved image {image_id} from dataset {image.get('dataset_id', 'Unknown')}")
        return ImageResponse(**image)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get image {image_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve image: {str(e)}"
        )
