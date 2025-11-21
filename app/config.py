"""
应用配置文件 - 向后兼容版本
现在使用新的配置管理系统，同时保持向后兼容性
"""

from typing import Dict, Optional, Set

from app.core.config import config_manager

# 向后兼容：使用新的配置管理器，但保持原有的接口
settings = config_manager.settings

# 支持的图像格式 - 从新配置系统获取
ALLOWED_IMAGE_FORMATS = set(fmt.lower() for fmt in settings.allowed_image_formats)

# YOLO类别颜色映射（默认）
DEFAULT_CLASS_COLORS = {
    "0": "#FF0000",    # 红色
    "1": "#00FF00",    # 绿色
    "2": "#0000FF",    # 蓝色
    "3": "#FFFF00",    # 黄色
    "4": "#FF00FF",    # 紫色
    "5": "#00FFFF",    # 青色
    "6": "#FFA500",    # 橙色
    "7": "#800080",    # 紫红色
    "8": "#008000",    # 深绿色
    "9": "#000080",    # 海军蓝
}

# 图像质量阈值
IMAGE_QUALITY_THRESHOLDS = {
    "min_sharpness": 30.0,
    "max_blur_score": 20.0,
    "min_brightness": 20.0,
    "max_brightness": 90.0,
    "min_contrast": 30.0,
    "max_noise_level": 40.0
}

# API响应状态码
HTTP_STATUS_CODES = {
    "SUCCESS": 200,
    "CREATED": 201,
    "NO_CONTENT": 204,
    "BAD_REQUEST": 400,
    "UNAUTHORIZED": 401,
    "FORBIDDEN": 403,
    "NOT_FOUND": 404,
    "CONFLICT": 409,
    "UNSUPPORTED_MEDIA_TYPE": 415,
    "TOO_MANY_REQUESTS": 429,
    "INTERNAL_SERVER_ERROR": 500,
    "SERVICE_UNAVAILABLE": 503
}

# 错误代码
ERROR_CODES = {
    "AUTH_INVALID_CREDENTIALS": "AUTH_INVALID_CREDENTIALS",
    "AUTH_TOKEN_EXPIRED": "AUTH_TOKEN_EXPIRED",
    "AUTH_TOKEN_INVALID": "AUTH_TOKEN_INVALID",
    "AUTH_INSUFFICIENT_PERMISSIONS": "AUTH_INSUFFICIENT_PERMISSIONS",
    "RESOURCE_NOT_FOUND": "RESOURCE_NOT_FOUND",
    "RESOURCE_ALREADY_EXISTS": "RESOURCE_ALREADY_EXISTS",
    "VALIDATION_ERROR": "VALIDATION_ERROR",
    "RATE_LIMIT_EXCEEDED": "RATE_LIMIT_EXCEEDED",
    "FILE_TOO_LARGE": "FILE_TOO_LARGE",
    "UNSUPPORTED_FILE_TYPE": "UNSUPPORTED_FILE_TYPE",
    "STORAGE_QUOTA_EXCEEDED": "STORAGE_QUOTA_EXCEEDED",
    "INTERNAL_SERVER_ERROR": "INTERNAL_SERVER_ERROR",
    "SERVICE_UNAVAILABLE": "SERVICE_UNAVAILABLE"
}
