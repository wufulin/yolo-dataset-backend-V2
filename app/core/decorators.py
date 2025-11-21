"""装饰器模式实现 - 日志、缓存、监控等横切关注点"""
import asyncio
import functools
import hashlib
import json
import logging
import time
from datetime import datetime, timedelta
from functools import wraps
from typing import Any, Awaitable, Callable, Dict, Optional, TypeVar, Union

from app.core.exceptions import (
    DatabaseException,
    ErrorCategory,
    ErrorSeverity,
    SystemException,
    YOLOException,
)
from app.utils.logger import get_logger

logger: logging.Logger = get_logger(__name__)

# 类型变量定义
F = TypeVar('F', bound=Callable[..., Any])
AsyncF = TypeVar('AsyncF', bound=Callable[..., Awaitable[Any]])


def exception_handler(
    exception_type: type = Exception,
    default_message: str = "操作执行失败",
    log_error: bool = True,
    reraise: bool = True,
    custom_handler: Optional[Callable[[Exception], Any]] = None
):
    """
    异常处理装饰器
    支持两种用法：
    @exception_handler 或 @exception_handler()
    @exception_handler(DatabaseException)

    Args:
        exception_type: 要捕获的异常类型
        default_message: 默认错误消息
        log_error: 是否记录错误日志
        reraise: 是否重新抛出异常
        custom_handler: 自定义异常处理器
    """
    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # 记录错误日志
                if log_error:
                    if isinstance(e, YOLOException):
                        logger.error(
                            f"Function {func.__name__} failed: {e.message}",
                            extra={"error_details": e.to_dict()},
                            exc_info=True
                        )
                    else:
                        logger.error(
                            f"Function {func.__name__} failed: {str(e)}",
                            exc_info=True
                        )

                # 自定义异常处理
                if custom_handler:
                    try:
                        return custom_handler(e)
                    except Exception as custom_error:
                        logger.error(f"Custom exception handler failed: {custom_error}")

                # 转换异常类型
                if not isinstance(e, exception_type) and exception_type != Exception:
                    if isinstance(e, YOLOException):
                        raise e
                    else:
                        raise YOLOException(
                            message=default_message,
                            error_code="EXECUTION_ERROR",
                            category=ErrorCategory.SYSTEM,
                            severity=ErrorSeverity.MEDIUM,
                            original_error=e
                        )

                # 重新抛出原始异常或返回默认值
                if reraise:
                    raise
                else:
                    return None

        return wrapper  # type: ignore

    # 如果第一个参数是可调用对象但不是类型（即函数），说明是 @exception_handler 的用法（无括号）
    # 类型（如异常类）也是可调用的，所以需要排除类型
    if callable(exception_type) and not isinstance(exception_type, type):
        func = exception_type
        exception_type = Exception
        return decorator(func)  # type: ignore
    
    return decorator


def async_exception_handler(
    exception_type: type = Exception,
    default_message: str = "异步操作执行失败",
    log_error: bool = True,
    reraise: bool = True,
    custom_handler: Optional[Callable[[Exception], Any]] = None
):
    """
    异步异常处理装饰器
    支持两种用法：
    @async_exception_handler 或 @async_exception_handler()
    @async_exception_handler(DatabaseException)
    """
    def decorator(func: AsyncF) -> AsyncF:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                # 记录错误日志
                if log_error:
                    if isinstance(e, YOLOException):
                        logger.error(
                            f"Async function {func.__name__} failed: {e.message}",
                            extra={"error_details": e.to_dict()},
                            exc_info=True
                        )
                    else:
                        logger.error(
                            f"Async function {func.__name__} failed: {str(e)}",
                            exc_info=True
                        )

                # 自定义异常处理
                if custom_handler:
                    try:
                        return await custom_handler(e) if asyncio.iscoroutinefunction(custom_handler) else custom_handler(e)
                    except Exception as custom_error:
                        logger.error(f"Custom async exception handler failed: {custom_error}")

                # 转换异常类型
                if not isinstance(e, exception_type) and exception_type != Exception:
                    if isinstance(e, YOLOException):
                        raise e
                    else:
                        raise YOLOException(
                            message=default_message,
                            error_code="ASYNC_EXECUTION_ERROR",
                            category=ErrorCategory.SYSTEM,
                            severity=ErrorSeverity.MEDIUM,
                            original_error=e
                        )

                # 重新抛出原始异常或返回默认值
                if reraise:
                    raise
                else:
                    return None

        return wrapper  # type: ignore

    # 如果第一个参数是可调用对象但不是类型（即函数），说明是 @async_exception_handler 的用法（无括号）
    # 类型（如异常类）也是可调用的，所以需要排除类型
    if callable(exception_type) and not isinstance(exception_type, type):
        func = exception_type
        exception_type = Exception
        return decorator(func)  # type: ignore
    
    return decorator


def performance_monitor(
    operation_name: Optional[str] = None,
    log_slow_operations: bool = True,
    slow_threshold: float = 1.0,
    include_args: bool = False,
    include_result: bool = False
):
    """
    性能监控装饰器

    Args:
        operation_name: 操作名称（默认使用函数名）
        log_slow_operations: 是否记录慢操作
        slow_threshold: 慢操作阈值（秒）
        include_args: 是否包含参数信息
        include_result: 是否包含结果信息
    """
    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args, **kwargs):
            op_name = operation_name or f"{func.__module__}.{func.__name__}"
            start_time = time.time()

            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time

                # 记录性能日志
                log_info = {
                    "operation": op_name,
                    "execution_time": execution_time,
                    "status": "success"
                }

                if include_args:
                    log_info["args"] = str(args)[:200] + "..." if len(str(args)) > 200 else str(args)
                    log_info["kwargs"] = str(kwargs)[:200] + "..." if len(str(kwargs)) > 200 else str(kwargs)

                if include_result:
                    result_str = str(result)[:200] + "..." if len(str(result)) > 200 else str(result)
                    log_info["result"] = result_str

                if log_slow_operations and execution_time > slow_threshold:
                    logger.warning(f"Slow operation detected: {log_info}")
                else:
                    logger.debug(f"Operation completed: {log_info}")

                return result

            except Exception as e:
                execution_time = time.time() - start_time
                logger.error(
                    f"Operation failed: {op_name} in {execution_time:.2f}s",
                    extra={"error": str(e), "execution_time": execution_time},
                    exc_info=True
                )
                raise

        return wrapper  # type: ignore

    return decorator


def async_performance_monitor(
    operation_name: Optional[str] = None,
    log_slow_operations: bool = True,
    slow_threshold: float = 1.0,
    include_args: bool = False,
    include_result: bool = False
):
    """
    异步性能监控装饰器
    """
    def decorator(func: AsyncF) -> AsyncF:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            op_name = operation_name or f"{func.__module__}.{func.__name__}"
            start_time = time.time()

            try:
                result = await func(*args, **kwargs)
                execution_time = time.time() - start_time

                # 记录性能日志
                log_info = {
                    "operation": op_name,
                    "execution_time": execution_time,
                    "status": "success"
                }

                if include_args:
                    log_info["args"] = str(args)[:200] + "..." if len(str(args)) > 200 else str(args)
                    log_info["kwargs"] = str(kwargs)[:200] + "..." if len(str(kwargs)) > 200 else str(kwargs)

                if include_result:
                    result_str = str(result)[:200] + "..." if len(str(result)) > 200 else str(result)
                    log_info["result"] = result_str

                if log_slow_operations and execution_time > slow_threshold:
                    logger.warning(f"Slow async operation detected: {log_info}")
                else:
                    logger.debug(f"Async operation completed: {log_info}")

                return result

            except Exception as e:
                execution_time = time.time() - start_time
                logger.error(
                    f"Async operation failed: {op_name} in {execution_time:.2f}s",
                    extra={"error": str(e), "execution_time": execution_time},
                    exc_info=True
                )
                raise

        return wrapper  # type: ignore

    return decorator


def retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    exceptions: tuple = (Exception,)
):
    """
    重试装饰器

    Args:
        max_attempts: 最大重试次数
        delay: 初始延迟时间（秒）
        exponential_base: 指数退避基数
        jitter: 是否添加随机抖动
        exceptions: 需要重试的异常类型
    """
    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e

                    if attempt == max_attempts - 1:
                        logger.error(f"Function {func.__name__} failed after {max_attempts} attempts: {e}")
                        raise

                    # 计算延迟时间
                    if attempt > 0:
                        wait_time = delay * (exponential_base ** (attempt - 1))
                        if jitter:
                            import random
                            wait_time *= (0.5 + random.random() * 0.5)

                        logger.warning(
                            f"Function {func.__name__} attempt {attempt + 1} failed, retrying in {wait_time:.2f}s: {e}"
                        )
                        time.sleep(wait_time)

            raise last_exception

        return wrapper  # type: ignore

    return decorator


def async_retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    exceptions: tuple = (Exception,)
):
    """
    异步重试装饰器
    """
    def decorator(func: AsyncF) -> AsyncF:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e

                    if attempt == max_attempts - 1:
                        logger.error(f"Async function {func.__name__} failed after {max_attempts} attempts: {e}")
                        raise

                    # 计算延迟时间
                    if attempt > 0:
                        wait_time = delay * (exponential_base ** (attempt - 1))
                        if jitter:
                            import random
                            wait_time *= (0.5 + random.random() * 0.5)

                        logger.warning(
                            f"Async function {func.__name__} attempt {attempt + 1} failed, retrying in {wait_time:.2f}s: {e}"
                        )
                        await asyncio.sleep(wait_time)

            raise last_exception

        return wrapper  # type: ignore

    return decorator


def cache_result(
    ttl: int = 300,  # 默认5分钟
    key_func: Optional[Callable] = None,
    max_size: Optional[int] = None
):
    """
    结果缓存装饰器（简化版）

    Args:
        ttl: 缓存时间（秒）
        key_func: 自定义缓存键生成函数
        max_cache_size: 最大缓存条目数
    """
    cache: Dict[str, tuple] = {}
    cache_order: list = []  # 用于LRU

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 生成缓存键
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                # 使用函数名和参数生成简单键
                key_data = f"{func.__name__}:{str(args)}:{str(kwargs)}"
                cache_key = hashlib.md5(key_data.encode()).hexdigest()

            # 检查缓存
            current_time = datetime.now().timestamp()
            if cache_key in cache:
                cached_result, cached_time = cache[cache_key]
                if current_time - cached_time < ttl:
                    logger.debug(f"Cache hit for {func.__name__}")
                    # 更新访问顺序（LRU）
                    if cache_key in cache_order:
                        cache_order.remove(cache_key)
                    cache_order.append(cache_key)
                    return cached_result

            # 执行函数并缓存结果
            result = func(*args, **kwargs)
            cache[cache_key] = (result, current_time)
            cache_order.append(cache_key)

            # 检查缓存大小限制
            if max_size and len(cache) > max_size:
                # 移除最旧的条目
                oldest_key = cache_order.pop(0)
                del cache[oldest_key]

            logger.debug(f"Cache miss for {func.__name__}, result cached")
            return result

        return wrapper  # type: ignore

    return decorator


def validate_inputs(**validators):
    """
    输入验证装饰器

    Args:
        **validators: 参数名到验证函数的映射
    """
    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 合并位置参数和关键字参数
            bound_args = inspect.signature(func).bind(*args, **kwargs)
            bound_args.apply_defaults()

            # 验证参数
            for param_name, validator in validators.items():
                if param_name in bound_args.arguments:
                    value = bound_args.arguments[param_name]
                    if not validator(value):
                        raise ValueError(f"Parameter '{param_name}' failed validation")

            return func(*args, **kwargs)

        return wrapper  # type: ignore

    return decorator


def log_method_call(
    log_level: int = logging.DEBUG,
    include_args: bool = False,
    include_result: bool = False
):
    """
    方法调用日志装饰器

    Args:
        log_level: 日志级别
        include_args: 是否记录参数
        include_result: 是否记录返回值
    """
    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 记录调用信息
            log_info = {
                "function": f"{func.__module__}.{func.__name__}",
                "timestamp": datetime.now().isoformat()
            }

            if include_args:
                log_info["args"] = str(args)[:500] + "..." if len(str(args)) > 500 else str(args)
                log_info["kwargs"] = str(kwargs)[:500] + "..." if len(str(kwargs)) > 500 else str(kwargs)

            logger.log(log_level, f"Calling {log_info['function']}", extra=log_info)

            try:
                result = func(*args, **kwargs)

                if include_result:
                    result_str = str(result)[:500] + "..." if len(str(result)) > 500 else str(result)
                    logger.log(log_level, f"Function {log_info['function']} completed", extra={
                        **log_info,
                        "result": result_str
                    })
                else:
                    logger.log(log_level, f"Function {log_info['function']} completed")

                return result

            except Exception as e:
                logger.error(
                    f"Function {log_info['function']} failed: {str(e)}",
                    extra=log_info,
                    exc_info=True
                )
                raise

        return wrapper  # type: ignore

    return decorator


# 导入inspect模块（在函数内部使用）
import inspect
