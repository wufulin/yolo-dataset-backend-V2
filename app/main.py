"""Main FastAPI application."""
import asyncio
import logging
import sys
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any

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
)
from app.core.exceptions import ValidationException as CustomValidationException
from app.core.exceptions import (
    YOLOException,
)
from app.services.redis_service import redis_session_manager
from app.utils.logger import get_logger

logger: logging.Logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> Any:
    """应用生命周期管理"""
    # 启动逻辑
    logger.info("Starting YOLO Dataset API...")

    try:
        # 初始化Redis连接
        await redis_session_manager.connect()
        logger.info("Redis connection established successfully")
    except Exception as e:
        logger.error(f"Failed to connect to Redis: {e}")
        logger.warning("Application will start without Redis - session management will use in-memory storage")

    yield  # 应用运行期间

    # 关闭逻辑
    logger.info("Shutting down YOLO Dataset API...")

    try:
        await redis_session_manager.disconnect()
        logger.info("Redis connection closed successfully")
    except Exception as e:
        logger.error(f"Error closing Redis connection: {e}")


def create_application() -> FastAPI:
    """创建和配置FastAPI应用"""
    application = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="YOLO Dataset Management API v2.0.0 - 现代化、高性能的YOLO数据集管理系统",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan
    )

    # 添加CORS中间件
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # 生产环境需要配置具体域名
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 注册全局异常处理器
    _register_exception_handlers(application)

    # 注册路由
    _register_routes(application)

    logger.info(f"Application '{settings.app_name}' v{settings.app_version} created successfully")

    return application


def _register_exception_handlers(app: FastAPI) -> Any:
    """注册全局异常处理器"""

    @app.exception_handler(YOLOException)
    async def yolo_exception_handler(request: Request, exc: YOLOException) -> Any:
        """处理自定义YOLO异常"""
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
    async def http_exception_handler(request: Request, exc: HTTPException) -> Any:
        """处理HTTP异常"""
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
    async def validation_exception_handler(request: Request, exc: ValidationError) -> Any:
        """处理Pydantic验证异常"""
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
    async def starlette_exception_handler(request: Request, exc: StarletteHTTPException) -> Any:
        """处理Starlette HTTP异常"""
        logger.warning(f"Starlette HTTP exception: {exc.status_code}")
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
    async def global_exception_handler(request: Request, exc: Exception) -> Any:
        """全局异常处理器"""
        logger.error(
            f"Unhandled exception: {type(exc).__name__}: {exc}",
            exc_info=True,
            extra={
                "request_path": request.url.path,
                "request_method": request.method,
                "exception_type": type(exc).__name__
            }
        )

        # 根据异常类型决定返回状态码
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


def _register_routes(app: FastAPI) -> Any:
    """注册路由"""
    app.include_router(
        datasets.router,
        prefix="/api/v1",
        tags=["datasets"]
    )
    app.include_router(
        upload.router,
        prefix="/api/v1",
        tags=["upload"]
    )


def _map_error_category_to_status_code(category: ErrorCategory) -> int:
    """将错误类别映射到HTTP状态码"""
    mapping = {
        ErrorCategory.AUTHENTICATION: 401,
        ErrorCategory.AUTHORIZATION: 403,
        ErrorCategory.VALIDATION: 422,
        ErrorCategory.BUSINESS_LOGIC: 400,
        ErrorCategory.DATABASE: 500,
        ErrorCategory.STORAGE: 500,
        ErrorCategory.NETWORK: 503,
        ErrorCategory.FILE_SYSTEM: 500,
        ErrorCategory.EXTERNAL_SERVICE: 502,
        ErrorCategory.SYSTEM: 500
    }
    return mapping.get(category, 500)


# 创建应用实例
app: FastAPI = create_application()


@app.get("/")
@performance_monitor("root_endpoint")
async def root() -> Any:
    """根端点"""
    logger.info("Root endpoint accessed")
    return {
        "message": f"Welcome to {settings.app_name}",
        "version": settings.app_version,
        "description": "Modern, high-performance YOLO dataset management system",
        "documentation": "/docs",
        "health_check": "/health"
    }


@app.get("/health")
@performance_monitor("health_check")
async def health_check() -> Any:
    """健康检查端点"""
    logger.debug("Health check endpoint accessed")

    health_status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": settings.app_version,
        "environment": settings.environment.value
    }

    # 检查Redis连接
    try:
        redis_manager = redis_session_manager
        if redis_manager.redis:
            await redis_manager.redis.ping()
            health_status["redis"] = {
                "status": "connected",
                "type": "primary"
            }
        else:
            health_status["redis"] = {
                "status": "not_initialized",
                "type": "memory_fallback"
            }
    except Exception as e:
        health_status["redis"] = {
            "status": "error",
            "error": str(e)
        }

    # 检查数据库连接（如果可用）
    try:
        from app.services.db_service import db_service

        # 这里可以添加数据库健康检查
        health_status["database"] = {
            "status": "connected",
            "type": "mongodb"
        }
    except Exception as e:
        health_status["database"] = {
            "status": "error",
            "error": str(e)
        }

    # 整体健康状态
    critical_services = ["redis"]
    service_statuses = [health_status.get(service, {}).get("status") for service in critical_services]

    if all(status == "connected" for status in service_statuses):
        overall_status = "healthy"
    elif any(status == "error" for status in service_statuses):
        overall_status = "unhealthy"
    else:
        overall_status = "degraded"

    health_status["overall_status"] = overall_status

    # 设置响应状态码
    status_code = 200 if overall_status == "healthy" else (503 if overall_status == "unhealthy" else 200)

    # 如果是不健康状态，记录日志
    if overall_status != "healthy":
        logger.warning(f"Health check failed: {health_status}")

    # 这里我们直接返回JSON，因为FastAPI会自动设置状态码
    return JSONResponse(content=health_status, status_code=status_code)


@app.get("/metrics")
@performance_monitor("metrics_endpoint")
async def get_metrics() -> Any:
    """获取系统指标"""
    logger.info("Metrics endpoint accessed")

    metrics = {
        "timestamp": datetime.utcnow().isoformat(),
        "version": settings.app_version,
        "uptime": "N/A",  # 实际部署中需要跟踪启动时间
        "configuration": {
            "environment": settings.environment.value,
            "debug": settings.debug,
            "default_page_size": settings.default_page_size,
            "max_page_size": settings.max_page_size
        }
    }

    # 添加Redis指标
    try:
        if redis_session_manager.redis:
            redis_info = await redis_session_manager.redis.info()
            metrics["redis"] = {
                "connected": True,
                "clients": redis_info.get("connected_clients", 0),
                "memory_used": redis_info.get("used_memory", 0)
            }
        else:
            metrics["redis"] = {"connected": False}
    except Exception as e:
        metrics["redis"] = {"connected": False, "error": str(e)}

    return metrics


@app.get("/config")
async def get_config_info() -> Any:
    """获取配置信息（敏感信息已过滤）"""
    logger.info("Config info endpoint accessed")

    # 返回非敏感的配置文件信息
    safe_config = {
        "app_name": settings.app_name,
        "app_version": settings.app_version,
        "environment": settings.environment.value,
        "debug": settings.debug,
        "features": {
            "performance_monitoring": settings.get("enable_performance_monitoring", True),
            "cache_enabled": True,
            "async_processing": True
        },
        "limits": {
            "max_upload_size": settings.max_upload_size,
            "default_page_size": settings.default_page_size,
            "max_page_size": settings.max_page_size
        }
    }

    return safe_config


if __name__ == "__main__":
    import uvicorn

    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    logger.info(f"Environment: {settings.environment.value}")
    logger.info(f"Debug mode: {settings.debug}")

    # 启动开发服务器
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level=settings.log_level.value.lower(),
        access_log=True
    )
