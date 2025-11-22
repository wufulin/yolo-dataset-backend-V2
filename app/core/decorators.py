"""Decorator implementations for logging, caching, monitoring, and other cross-cutting concerns."""
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

# Type variables shared by decorators
F = TypeVar('F', bound=Callable[..., Any])
AsyncF = TypeVar('AsyncF', bound=Callable[..., Awaitable[Any]])


def exception_handler(
    exception_type: type = Exception,
    default_message: str = "Operation failed",
    log_error: bool = True,
    reraise: bool = True,
    custom_handler: Optional[Callable[[Exception], Any]] = None
):
    """
    Exception handling decorator supporting both @exception_handler and @exception_handler().

    Args:
        exception_type: Exception type to capture.
        default_message: Fallback user-facing message.
        log_error: Whether to log the error automatically.
        reraise: Whether to re-raise the original exception.
        custom_handler: Optional callback to transform the exception.
    """

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Log enriched error details
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

                # Execute user-provided exception hook
                if custom_handler:
                    try:
                        return custom_handler(e)
                    except Exception as custom_error:
                        logger.error(f"Custom exception handler failed: {custom_error}")

                # Normalize exception shape for consumers
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

                # Bubble original exception or swallow based on config
                if reraise:
                    raise
                else:
                    return None

        return wrapper  # type: ignore

    # Support decorator usage without parentheses while avoiding types (which are callable)
    if callable(exception_type) and not isinstance(exception_type, type):
        func = exception_type
        exception_type = Exception
        return decorator(func)  # type: ignore

    return decorator


def async_exception_handler(
    exception_type: type = Exception,
    default_message: str = "Async operation failed",
    log_error: bool = True,
    reraise: bool = True,
    custom_handler: Optional[Callable[[Exception], Any]] = None
):
    """Async-friendly version of exception_handler with identical behavior."""

    def decorator(func: AsyncF) -> AsyncF:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                # Log enriched error details
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

                # Execute optional handler hook
                if custom_handler:
                    try:
                        return await custom_handler(e) if asyncio.iscoroutinefunction(
                            custom_handler) else custom_handler(e)
                    except Exception as custom_error:
                        logger.error(f"Custom async exception handler failed: {custom_error}")

                # Normalize exception shape for consumers
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

                # Bubble or swallow based on configuration
                if reraise:
                    raise
                else:
                    return None

        return wrapper  # type: ignore

    # Support decorator usage without parentheses while avoiding type objects
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
    Decorator for measuring sync function latency and flagging slow calls.

    Args:
        operation_name: Custom label for the monitored operation.
        log_slow_operations: Emit warnings when threshold is exceeded.
        slow_threshold: Seconds beyond which the call is considered slow.
        include_args: Attach arguments to the log payload.
        include_result: Attach return value to the log payload.
    """

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args, **kwargs):
            op_name = operation_name or f"{func.__module__}.{func.__name__}"
            start_time = time.time()

            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time

                # Emit structured performance log entry
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
    """Async equivalent of performance_monitor."""

    def decorator(func: AsyncF) -> AsyncF:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            op_name = operation_name or f"{func.__module__}.{func.__name__}"
            start_time = time.time()

            try:
                result = await func(*args, **kwargs)
                execution_time = time.time() - start_time

                # Emit structured performance log entry
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
    Retry decorator with exponential backoff for synchronous functions.

    Args:
        max_attempts: Maximum number of attempts.
        delay: Initial delay in seconds.
        exponential_base: Base multiplier used for backoff.
        jitter: Whether to introduce randomness to delays.
        exceptions: Tuple of exception types that trigger retries.
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

                    # Calculate next delay duration with optional jitter
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
    """Async variant of retry with identical parameters."""

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

                    # Calculate next delay duration with optional jitter
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
    ttl: int = 300,  # Default five-minute TTL
    key_func: Optional[Callable] = None,
    max_size: Optional[int] = None
):
    """
    In-memory caching decorator with TTL and optional LRU eviction.

    Args:
        ttl: Cache lifetime in seconds.
        key_func: Optional callable that builds the cache key.
        max_size: Maximum entries to keep before evicting the oldest.
    """
    cache: Dict[str, tuple] = {}
    cache_order: list = []  # Track access order for LRU

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Build cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                # Fallback hash from function name and arguments
                key_data = f"{func.__name__}:{str(args)}:{str(kwargs)}"
                cache_key = hashlib.md5(key_data.encode()).hexdigest()

            # Serve cached response when still valid
            current_time = datetime.now().timestamp()
            if cache_key in cache:
                cached_result, cached_time = cache[cache_key]
                if current_time - cached_time < ttl:
                    logger.debug(f"Cache hit for {func.__name__}")
                    # Refresh LRU order
                    if cache_key in cache_order:
                        cache_order.remove(cache_key)
                    cache_order.append(cache_key)
                    return cached_result

            # Compute fresh result and cache it
            result = func(*args, **kwargs)
            cache[cache_key] = (result, current_time)
            cache_order.append(cache_key)

            # Enforce max cache size
            if max_size and len(cache) > max_size:
                # Drop least recently used entry
                oldest_key = cache_order.pop(0)
                del cache[oldest_key]

            logger.debug(f"Cache miss for {func.__name__}, result cached")
            return result

        return wrapper  # type: ignore

    return decorator


def validate_inputs(**validators):
    """
    Decorator for declarative parameter validation.

    Args:
        **validators: Mapping from parameter name to validation callable.
    """

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Merge positional + keyword arguments
            bound_args = inspect.signature(func).bind(*args, **kwargs)
            bound_args.apply_defaults()

            # Execute validators
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
    Decorator for logging method invocation metadata.

    Args:
        log_level: Logging level for emitted records.
        include_args: Whether to log arguments.
        include_result: Whether to log return value.
    """

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Build logging payload
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


# Import inspect lazily to avoid circular dependencies
import inspect
