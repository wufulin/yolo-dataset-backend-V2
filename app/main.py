"""Main FastAPI application."""
import asyncio
import logging
import sys
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Dict

from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import HTTPException, RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.api import datasets, upload
from app.auth import authenticate_user
from app.config import settings
from app.core.decorators import async_exception_handler, performance_monitor
from app.core.exceptions import (
    AuthenticationException,
    AuthorizationException,
    DatabaseException,
    ErrorCategory,
    ErrorSeverity,
    ValidationException as CustomValidationException,
    YOLOException,
)
from app.services.redis_service import redis_session_manager
from app.utils.logger import get_logger

logger: logging.Logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> Any:
    """Application lifecycle management
    
    Handles startup and shutdown logic for the FastAPI application,
    including external service connections (e.g., Redis).
    """
    # Startup logic
    logger.info("Starting YOLO Dataset API...")

    try:
        # Initialize Redis connection
        await redis_session_manager.connect()
        logger.info("Redis connection established successfully")
    except Exception as e:
        logger.error(f"Failed to connect to Redis: {e}")
        logger.warning(
            "Application will start without Redis - "
            "session management will use in-memory storage"
        )

    yield  # Application runtime

    # Shutdown logic
    logger.info("Shutting down YOLO Dataset API...")

    try:
        await redis_session_manager.disconnect()
        logger.info("Redis connection closed successfully")
    except Exception as e:
        logger.error(f"Error closing Redis connection: {e}")


def create_application() -> FastAPI:
    """Create and configure FastAPI application instance
    
    Configures CORS, exception handlers, routes, and other core settings.
    Returns a fully initialized FastAPI app.
    """
    application = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="YOLO Dataset Management API v2.0.0 - Modern, high-performance YOLO dataset management system",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan
    )

    # Add CORS middleware (configure specific origins in production)
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Replace with specific domains in production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register global exception handlers
    _register_exception_handlers(application)

    # Register API routes
    _register_routes(application)

    logger.info(
        f"Application '{settings.app_name}' v{settings.app_version} "
        "created successfully"
    )

    return application


def _register_exception_handlers(app: FastAPI) -> None:
    """Register global exception handlers for consistent error responses"""

    @app.exception_handler(YOLOException)
    async def yolo_exception_handler(request: Request, exc: YOLOException) -> JSONResponse:
        """Handle custom YOLO-specific exceptions"""
        logger.error(
            f"YOLO exception occurred: {exc.message}",
            extra={
                "error_code": exc.error_code,
                "category": exc.category.value,
                "severity": exc.severity.value,
                "details": exc.details,
                "request_path": request.url.path,
                "request_method": request.method
            }
        )

        status_code = _map_error_category_to_status_code(exc.category)

        return JSONResponse(
            status_code=status_code,
            content={
                "error": {
                    "code": exc.error_code,
                    "message": exc.message,
                    "category": exc.category.value,
                    "severity": exc.severity.value,
                    "details": exc.details,
                    "timestamp": datetime.utcnow().isoformat(),
                    "path": str(request.url.path)
                }
            }
        )

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
        """Handle FastAPI HTTP exceptions"""
        logger.warning(f"HTTP exception: {exc.status_code} - {exc.detail}")
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "code": f"HTTP_{exc.status_code}",
                    "message": exc.detail,
                    "timestamp": datetime.utcnow().isoformat(),
                    "path": str(request.url.path)
                }
            }
        )

    @app.exception_handler(ValidationError)
    async def validation_exception_handler(request: Request, exc: ValidationError) -> JSONResponse:
        """Handle Pydantic model validation exceptions"""
        logger.warning(f"Validation error: {exc}")
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "code": "VALIDATION_ERROR",
                    "message": "Request validation failed",
                    "details": exc.errors(),
                    "timestamp": datetime.utcnow().isoformat(),
                    "path": str(request.url.path)
                }
            }
        )

    @app.exception_handler(StarletteHTTPException)
    async def starlette_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        """Handle Starlette base HTTP exceptions"""
        logger.warning(f"Starlette HTTP exception: {exc.status_code} - {exc.detail}")
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "code": f"HTTP_{exc.status_code}",
                    "message": exc.detail,
                    "timestamp": datetime.utcnow().isoformat(),
                    "path": str(request.url.path)
                }
            }
        )

    @app.exception_handler(Exception)
    @performance_monitor("global_exception_handler")
    async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        """Global exception handler for uncaught exceptions
        
        Provides consistent error formatting and logs detailed exception info
        for debugging purposes.
        """
        logger.error(
            f"Unhandled exception: {type(exc).__name__}: {exc}",
            exc_info=True,  # Include full traceback in logs
            extra={
                "request_path": request.url.path,
                "request_method": request.method,
                "exception_type": type(exc).__name__
            }
        )

        # Map exception type to appropriate status code and message
        if isinstance(exc, ConnectionError):
            status_code = 503
            message = "Service temporarily unavailable"
        elif isinstance(exc, TimeoutError):
            status_code = 504
            message = "Request timeout"
        elif isinstance(exc, PermissionError):
            status_code = 403
            message = "Permission denied"
        else:
            status_code = 500
            message = "Internal server error"

        return JSONResponse(
            status_code=status_code,
            content={
                "error": {
                    "code": "INTERNAL_SERVER_ERROR",
                    "message": message,
                    "timestamp": datetime.utcnow().isoformat(),
                    "path": str(request.url.path),
                    "type": type(exc).__name__
                }
            }
        )


def _register_routes(app: FastAPI) -> None:
    """Register API routes with the FastAPI application"""
    # Dataset management routes
    app.include_router(
        datasets.router,
        prefix="/api/v1",
        tags=["datasets"]
    )
    # File upload routes
    app.include_router(
        upload.router,
        prefix="/api/v1",
        tags=["upload"]
    )


def _map_error_category_to_status_code(category: ErrorCategory) -> int:
    """Map custom error categories to standard HTTP status codes
    
    Args:
        category: ErrorCategory enum value
        
    Returns:
        Corresponding HTTP status code
    """
    status_code_mapping: Dict[ErrorCategory, int] = {
        ErrorCategory.AUTHENTICATION: 401,    # Unauthorized
        ErrorCategory.AUTHORIZATION: 403,     # Forbidden
        ErrorCategory.VALIDATION: 422,        # Unprocessable Entity
        ErrorCategory.BUSINESS_LOGIC: 400,    # Bad Request
        ErrorCategory.DATABASE: 500,          # Internal Server Error
        ErrorCategory.STORAGE: 500,           # Internal Server Error
        ErrorCategory.NETWORK: 503,           # Service Unavailable
        ErrorCategory.FILE_SYSTEM: 500,       # Internal Server Error
        ErrorCategory.EXTERNAL_SERVICE: 502,  # Bad Gateway
        ErrorCategory.SYSTEM: 500             # Internal Server Error
    }
    return status_code_mapping.get(category, 500)  # Default to 500


# Create FastAPI application instance
app: FastAPI = create_application()

if __name__ == "__main__":
    # Lazy import uvicorn to avoid dependency if not running directly
    import uvicorn

    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    logger.info(f"Environment: {settings.environment.value}")
    logger.info(f"Debug mode: {settings.debug}")

    # Start development server
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level=settings.log_level.value.lower(),
        access_log=True
    )