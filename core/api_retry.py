"""
API Retry Infrastructure for ST-Bot.

Provides robust retry logic for external API calls (OpenAI, Pinecone, etc.)
with exponential backoff, jitter, and comprehensive logging.

Features:
- Exponential backoff with configurable base and max delays
- Random jitter to prevent thundering herd
- Retryable vs non-retryable error classification
- Comprehensive logging of retry attempts
- Decorator and context manager interfaces
- Graceful degradation support

Usage:
    # As a decorator
    @with_retry(max_attempts=3, base_delay=1.0)
    def call_openai(prompt: str) -> str:
        return client.chat.completions.create(...)

    # As a context manager
    async with RetryContext(max_attempts=3) as ctx:
        result = await api_call()

    # Manual retry
    retry = RetryHandler(max_attempts=3)
    result = retry.execute(api_call, fallback=default_value)
"""

import logging
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, Union
import traceback

# Type variable for generic return types
T = TypeVar('T')

# Module logger
_logger = logging.getLogger(__name__)


# =============================================================================
# Error Classification
# =============================================================================

class RetryableErrorType(Enum):
    """Categories of errors for retry decisions."""
    RATE_LIMIT = "rate_limit"           # 429 - should retry with backoff
    TIMEOUT = "timeout"                  # Request timeout - should retry
    SERVER_ERROR = "server_error"        # 5xx - should retry
    CONNECTION_ERROR = "connection_error" # Network issues - should retry
    TRANSIENT = "transient"              # Other transient errors - should retry

    # Non-retryable
    AUTH_ERROR = "auth_error"            # 401/403 - don't retry
    BAD_REQUEST = "bad_request"          # 400 - don't retry
    NOT_FOUND = "not_found"              # 404 - don't retry
    QUOTA_EXCEEDED = "quota_exceeded"    # Billing issue - don't retry
    UNKNOWN = "unknown"                  # Unknown error - don't retry by default


# Errors that should trigger retry
RETRYABLE_ERROR_TYPES = {
    RetryableErrorType.RATE_LIMIT,
    RetryableErrorType.TIMEOUT,
    RetryableErrorType.SERVER_ERROR,
    RetryableErrorType.CONNECTION_ERROR,
    RetryableErrorType.TRANSIENT,
}


def classify_error(error: Exception) -> RetryableErrorType:
    """
    Classify an error to determine if it should be retried.

    Args:
        error: The exception to classify

    Returns:
        RetryableErrorType indicating the error category
    """
    error_type = type(error).__name__
    error_msg = str(error).lower()

    # Check for OpenAI-specific errors
    if 'openai' in error_type.lower():
        if 'ratelimit' in error_type.lower() or '429' in error_msg:
            return RetryableErrorType.RATE_LIMIT
        if 'timeout' in error_type.lower():
            return RetryableErrorType.TIMEOUT
        if 'authentication' in error_type.lower() or '401' in error_msg:
            return RetryableErrorType.AUTH_ERROR
        if 'apierror' in error_type.lower():
            if '5' in error_msg[:3]:  # 5xx errors
                return RetryableErrorType.SERVER_ERROR
            if '400' in error_msg:
                return RetryableErrorType.BAD_REQUEST

    # Check for Pinecone-specific errors
    if 'pinecone' in error_type.lower():
        if 'timeout' in error_msg:
            return RetryableErrorType.TIMEOUT
        if 'unauthorized' in error_msg or '401' in error_msg:
            return RetryableErrorType.AUTH_ERROR
        if 'rate' in error_msg or '429' in error_msg:
            return RetryableErrorType.RATE_LIMIT

    # Check for common network errors
    if any(x in error_type.lower() for x in ['timeout', 'timedout']):
        return RetryableErrorType.TIMEOUT

    if any(x in error_type.lower() for x in ['connection', 'network', 'socket']):
        return RetryableErrorType.CONNECTION_ERROR

    # Check error message for common patterns
    if 'rate limit' in error_msg or 'too many requests' in error_msg:
        return RetryableErrorType.RATE_LIMIT

    if 'timeout' in error_msg:
        return RetryableErrorType.TIMEOUT

    if 'connection' in error_msg or 'network' in error_msg:
        return RetryableErrorType.CONNECTION_ERROR

    if 'unauthorized' in error_msg or 'authentication' in error_msg:
        return RetryableErrorType.AUTH_ERROR

    if 'quota' in error_msg or 'billing' in error_msg:
        return RetryableErrorType.QUOTA_EXCEEDED

    return RetryableErrorType.UNKNOWN


def is_retryable(error: Exception) -> bool:
    """
    Check if an error should be retried.

    Args:
        error: The exception to check

    Returns:
        True if the error is retryable
    """
    return classify_error(error) in RETRYABLE_ERROR_TYPES


# =============================================================================
# Retry Configuration
# =============================================================================

@dataclass
class RetryConfig:
    """
    Configuration for retry behavior.

    Attributes:
        max_attempts: Maximum number of retry attempts (including initial)
        base_delay: Base delay in seconds before first retry
        max_delay: Maximum delay between retries
        exponential_base: Base for exponential backoff (default: 2)
        jitter: Whether to add random jitter (default: True)
        jitter_range: Range of jitter as fraction of delay (0.0-1.0)
        retry_on: List of exception types to retry on (None = use classifier)
        log_retries: Whether to log retry attempts
    """
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    jitter_range: float = 0.25
    retry_on: Optional[List[Type[Exception]]] = None
    log_retries: bool = True


# Default configurations for different API types
DEFAULT_OPENAI_RETRY = RetryConfig(
    max_attempts=3,
    base_delay=1.0,
    max_delay=30.0,
    exponential_base=2.0,
)

DEFAULT_PINECONE_RETRY = RetryConfig(
    max_attempts=3,
    base_delay=0.5,
    max_delay=15.0,
    exponential_base=2.0,
)


# =============================================================================
# Retry Result
# =============================================================================

@dataclass
class RetryResult:
    """
    Result of a retry operation.

    Attributes:
        success: Whether the operation eventually succeeded
        value: The return value (if successful)
        attempts: Number of attempts made
        total_delay: Total time spent waiting between retries
        errors: List of errors encountered
        final_error: The last error if operation failed
    """
    success: bool
    value: Any = None
    attempts: int = 0
    total_delay: float = 0.0
    errors: List[Exception] = field(default_factory=list)
    final_error: Optional[Exception] = None


# =============================================================================
# Retry Handler
# =============================================================================

class RetryHandler:
    """
    Handles retry logic for API calls.

    Example:
        handler = RetryHandler(config=DEFAULT_OPENAI_RETRY)
        result = handler.execute(
            lambda: client.chat.completions.create(...),
            fallback="Sorry, I couldn't process that request."
        )
    """

    def __init__(
        self,
        config: Optional[RetryConfig] = None,
        session_id: Optional[str] = None,
        operation_name: str = "api_call",
    ):
        """
        Initialize retry handler.

        Args:
            config: Retry configuration (uses default if None)
            session_id: Session ID for logging
            operation_name: Name of the operation for logging
        """
        self.config = config or RetryConfig()
        self.session_id = session_id or "unknown"
        self.operation_name = operation_name

    def calculate_delay(self, attempt: int) -> float:
        """
        Calculate delay before next retry using exponential backoff.

        Args:
            attempt: Current attempt number (0-indexed)

        Returns:
            Delay in seconds
        """
        # Exponential backoff: base_delay * (exponential_base ^ attempt)
        delay = self.config.base_delay * (self.config.exponential_base ** attempt)

        # Cap at max delay
        delay = min(delay, self.config.max_delay)

        # Add jitter if enabled
        if self.config.jitter:
            jitter_amount = delay * self.config.jitter_range
            delay += random.uniform(-jitter_amount, jitter_amount)
            delay = max(0.1, delay)  # Ensure positive delay

        return delay

    def should_retry(self, error: Exception, attempt: int) -> bool:
        """
        Determine if we should retry after an error.

        Args:
            error: The exception that occurred
            attempt: Current attempt number (0-indexed)

        Returns:
            True if we should retry
        """
        # Check attempt limit
        if attempt >= self.config.max_attempts - 1:
            return False

        # Check if specific exception types are configured
        if self.config.retry_on:
            return any(isinstance(error, exc_type) for exc_type in self.config.retry_on)

        # Use error classifier
        return is_retryable(error)

    def log_retry(
        self,
        attempt: int,
        error: Exception,
        delay: float,
        will_retry: bool
    ) -> None:
        """Log a retry attempt."""
        if not self.config.log_retries:
            return

        error_type = classify_error(error)

        if will_retry:
            _logger.warning(
                f"Retry {attempt + 1}/{self.config.max_attempts} for {self.operation_name}: "
                f"{type(error).__name__} ({error_type.value}). Waiting {delay:.2f}s...",
                extra={
                    "event": "api_retry",
                    "session_id": self.session_id,
                    "operation": self.operation_name,
                    "attempt": attempt + 1,
                    "max_attempts": self.config.max_attempts,
                    "error_type": error_type.value,
                    "error_message": str(error)[:200],
                    "delay_seconds": round(delay, 2),
                    "will_retry": True,
                }
            )
        else:
            _logger.error(
                f"Failed {self.operation_name} after {attempt + 1} attempts: "
                f"{type(error).__name__} ({error_type.value})",
                extra={
                    "event": "api_retry_exhausted",
                    "session_id": self.session_id,
                    "operation": self.operation_name,
                    "attempts": attempt + 1,
                    "error_type": error_type.value,
                    "error_message": str(error)[:200],
                    "will_retry": False,
                }
            )

    def execute(
        self,
        func: Callable[[], T],
        fallback: Optional[T] = None,
        raise_on_failure: bool = False,
    ) -> Union[T, RetryResult]:
        """
        Execute a function with retry logic.

        Args:
            func: The function to execute
            fallback: Value to return if all retries fail (if not raising)
            raise_on_failure: Whether to raise the final exception

        Returns:
            The function's return value, or fallback/RetryResult on failure
        """
        errors = []
        total_delay = 0.0

        for attempt in range(self.config.max_attempts):
            try:
                result = func()

                # Log success after retries
                if attempt > 0 and self.config.log_retries:
                    _logger.info(
                        f"{self.operation_name} succeeded after {attempt + 1} attempts",
                        extra={
                            "event": "api_retry_success",
                            "session_id": self.session_id,
                            "operation": self.operation_name,
                            "attempts": attempt + 1,
                            "total_delay": round(total_delay, 2),
                        }
                    )

                return result

            except Exception as e:
                errors.append(e)
                will_retry = self.should_retry(e, attempt)

                if will_retry:
                    delay = self.calculate_delay(attempt)
                    self.log_retry(attempt, e, delay, will_retry=True)
                    time.sleep(delay)
                    total_delay += delay
                else:
                    self.log_retry(attempt, e, 0, will_retry=False)
                    break

        # All retries exhausted
        final_error = errors[-1] if errors else None

        if raise_on_failure and final_error:
            raise final_error

        if fallback is not None:
            return fallback

        return RetryResult(
            success=False,
            attempts=len(errors),
            total_delay=total_delay,
            errors=errors,
            final_error=final_error,
        )


# =============================================================================
# Decorator Interface
# =============================================================================

def with_retry(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    retry_on: Optional[List[Type[Exception]]] = None,
    fallback: Any = None,
    raise_on_failure: bool = True,
    operation_name: Optional[str] = None,
):
    """
    Decorator to add retry logic to a function.

    Args:
        max_attempts: Maximum retry attempts
        base_delay: Base delay between retries
        max_delay: Maximum delay between retries
        exponential_base: Base for exponential backoff
        jitter: Whether to add random jitter
        retry_on: Specific exception types to retry on
        fallback: Value to return on failure (if not raising)
        raise_on_failure: Whether to raise on final failure
        operation_name: Name for logging (defaults to function name)

    Example:
        @with_retry(max_attempts=3, base_delay=1.0)
        def call_openai(prompt: str) -> str:
            return client.chat.completions.create(...)
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            config = RetryConfig(
                max_attempts=max_attempts,
                base_delay=base_delay,
                max_delay=max_delay,
                exponential_base=exponential_base,
                jitter=jitter,
                retry_on=retry_on,
            )

            handler = RetryHandler(
                config=config,
                operation_name=operation_name or func.__name__,
            )

            return handler.execute(
                lambda: func(*args, **kwargs),
                fallback=fallback,
                raise_on_failure=raise_on_failure,
            )

        return wrapper
    return decorator


# =============================================================================
# Convenience Functions
# =============================================================================

def retry_api_call(
    func: Callable[[], T],
    config: Optional[RetryConfig] = None,
    session_id: Optional[str] = None,
    operation_name: str = "api_call",
    fallback: Optional[T] = None,
    raise_on_failure: bool = False,
) -> Union[T, RetryResult]:
    """
    Execute an API call with retry logic.

    Convenience function for one-off API calls.

    Args:
        func: The function to execute
        config: Retry configuration
        session_id: Session ID for logging
        operation_name: Name for logging
        fallback: Value to return on failure
        raise_on_failure: Whether to raise on final failure

    Returns:
        Function result or fallback/RetryResult

    Example:
        result = retry_api_call(
            lambda: openai_client.chat.completions.create(...),
            config=DEFAULT_OPENAI_RETRY,
            session_id="abc123",
            operation_name="openai_chat",
            fallback="Sorry, I couldn't process that."
        )
    """
    handler = RetryHandler(
        config=config or RetryConfig(),
        session_id=session_id,
        operation_name=operation_name,
    )

    return handler.execute(
        func,
        fallback=fallback,
        raise_on_failure=raise_on_failure,
    )


def with_graceful_degradation(
    func: Callable[[], T],
    fallback_func: Callable[[], T],
    config: Optional[RetryConfig] = None,
    session_id: Optional[str] = None,
    operation_name: str = "api_call",
) -> T:
    """
    Execute an API call with retry, falling back to a backup function.

    Useful when you have a local fallback for API failures.

    Args:
        func: Primary function to execute
        fallback_func: Fallback function if primary fails
        config: Retry configuration
        session_id: Session ID for logging
        operation_name: Name for logging

    Returns:
        Result from primary or fallback function

    Example:
        result = with_graceful_degradation(
            lambda: llm_parse_query(query),
            lambda: local_pattern_matching(query),
            config=DEFAULT_OPENAI_RETRY,
            operation_name="query_parsing"
        )
    """
    handler = RetryHandler(
        config=config or RetryConfig(),
        session_id=session_id,
        operation_name=operation_name,
    )

    result = handler.execute(func, raise_on_failure=False)

    if isinstance(result, RetryResult) and not result.success:
        _logger.info(
            f"Falling back to local implementation for {operation_name}",
            extra={
                "event": "api_fallback",
                "session_id": session_id,
                "operation": operation_name,
                "attempts_before_fallback": result.attempts,
            }
        )
        return fallback_func()

    return result
