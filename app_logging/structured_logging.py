"""
Structured logging infrastructure for ST-Bot.

Provides production-grade JSON logging with:
- Multiple log levels (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- Rotating file handlers (daily rotation, 30-day retention)
- Separate error log file
- Performance tracking (latency metrics)
- Context tracking (session_id, query, intent)

Usage:
    from core.structured_logging import get_logger, log_query, log_error

    logger = get_logger(__name__)
    logger.info("Starting search", extra={"query": "USB-C cable"})

    # Or use convenience functions:
    log_query(session_id="abc", query="USB-C cable", intent="new_search")
"""

import json
import logging
import sys
import time
import traceback
import uuid
from datetime import datetime
from functools import wraps
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from typing import Any, Callable, Dict, Optional


# =============================================================================
# JSON Formatter
# =============================================================================

class JSONFormatter(logging.Formatter):
    """
    Formats log records as JSON for structured logging.

    Output format:
    {
        "timestamp": "2024-12-08T10:30:00.123456Z",
        "level": "INFO",
        "logger": "core.intent",
        "message": "Intent classified",
        "event": "intent_classification",
        "session_id": "abc123",
        ...
    }
    """

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add extra fields from record
        # These are passed via logger.info("msg", extra={...})
        extra_fields = [
            # Core conversation fields (new)
            "event", "session_id", "user_query", "intent_result", "intent_confidence",
            # Legacy names (for backwards compatibility)
            "query", "intent", "confidence",
            # Filter and search fields
            "filters", "products_found", "products_shown", "product_skus",
            "response_time_ms", "tier", "dropped_filters",
            # Error tracking
            "error_type", "stack_trace", "user_id",
            # LLM metrics
            "llm_model", "llm_tokens", "llm_latency_ms",
            # Performance timing
            "search_latency_ms", "filter_extraction_ms",
            "intent_classification_ms", "total_latency_ms",
            # Filter details
            "category", "connector_from", "connector_to", "length", "features",
            # Misc
            "reasoning", "match_quality", "port_count", "setup_type",
            "guidance_phase", "api_endpoint", "status_code", "request_id",
        ]

        for field in extra_fields:
            if hasattr(record, field):
                value = getattr(record, field)
                if value is not None:
                    log_data[field] = value

        # Add exception info if present
        if record.exc_info:
            log_data["error_type"] = record.exc_info[0].__name__ if record.exc_info[0] else None
            log_data["stack_trace"] = self.formatException(record.exc_info)

        return json.dumps(log_data, default=str)


class ConsoleFormatter(logging.Formatter):
    """
    Human-readable formatter for console output.

    Output format:
    2024-12-08 10:30:00 | INFO | core.intent | Intent classified | session=abc123
    """

    COLORS = {
        "DEBUG": "\033[36m",    # Cyan
        "INFO": "\033[32m",     # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",    # Red
        "CRITICAL": "\033[35m", # Magenta
        "RESET": "\033[0m",
    }

    def format(self, record: logging.LogRecord) -> str:
        """Format log record for console."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        level = record.levelname

        # Color the level
        color = self.COLORS.get(level, "")
        reset = self.COLORS["RESET"]

        # Build base message
        msg = f"{timestamp} | {color}{level:8}{reset} | {record.name} | {record.getMessage()}"

        # Add key context fields
        context_parts = []
        for field in ["session_id", "event", "response_time_ms"]:
            if hasattr(record, field) and getattr(record, field) is not None:
                context_parts.append(f"{field}={getattr(record, field)}")

        if context_parts:
            msg += f" | {', '.join(context_parts)}"

        # Add exception info
        if record.exc_info:
            msg += f"\n{self.formatException(record.exc_info)}"

        return msg


# =============================================================================
# CSV Handler for Power BI
# =============================================================================

class CSVHandler(logging.Handler):
    """
    Custom handler that writes log records to CSV files.

    Creates daily rotating CSV files in logs/csv/ directory.
    Ideal for Power BI and Excel analysis.
    """

    # Standard CSV columns (order matters for Power BI)
    # Priority columns first for easy Power BI analysis
    CSV_COLUMNS = [
        # Core conversation data (most important for Power BI)
        'timestamp', 'session_id', 'user_query', 'intent_result', 'intent_confidence',
        # Event metadata
        'level', 'logger', 'message', 'event',
        # Search filters (flattened)
        'filters_category', 'filters_connector_from', 'filters_connector_to',
        'filters_length', 'filters_length_unit', 'filters_features',
        # Results
        'products_found', 'products_shown', 'product_skus',
        # Performance metrics
        'response_time_ms', 'search_latency_ms', 'llm_latency_ms',
        # Legacy columns (for backwards compatibility with existing logs)
        'query', 'intent', 'confidence', 'category', 'connector_from',
        'connector_to', 'length', 'error_type', 'tier', 'filters'
    ]

    def __init__(self, log_dir: str = "logs"):
        super().__init__()
        self.log_dir = Path(log_dir) / "csv"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.current_date = None
        self.csv_file = None
        self.csv_writer = None
        self._file_handle = None

    def _get_csv_path(self) -> Path:
        """Get path for today's CSV file."""
        date_str = datetime.now().strftime('%Y-%m-%d')
        return self.log_dir / f"stbot-{date_str}.csv"

    def _ensure_file_open(self):
        """Ensure CSV file is open and has headers."""
        today = datetime.now().date()

        # Check if we need to rotate to new day
        if self.current_date != today:
            self._close_file()
            self.current_date = today

        if self._file_handle is None:
            csv_path = self._get_csv_path()
            file_exists = csv_path.exists()

            self._file_handle = open(csv_path, 'a', newline='', encoding='utf-8')
            self.csv_writer = None  # Will create with DictWriter

            # Write header if new file
            if not file_exists:
                self._file_handle.write(','.join(self.CSV_COLUMNS) + '\n')
                self._file_handle.flush()

    def _close_file(self):
        """Close current CSV file."""
        if self._file_handle:
            self._file_handle.close()
            self._file_handle = None
            self.csv_writer = None

    def _flatten_value(self, value) -> str:
        """Convert value to CSV-safe string."""
        if value is None:
            return ''
        if isinstance(value, dict):
            # Flatten dict to key=value pairs
            return '; '.join(f"{k}={v}" for k, v in value.items() if v is not None)
        if isinstance(value, list):
            return ', '.join(str(v) for v in value)
        return str(value)

    def emit(self, record: logging.LogRecord):
        """Write log record to CSV."""
        try:
            self._ensure_file_open()

            # Build row from record
            row = {
                'timestamp': datetime.utcnow().isoformat() + 'Z',
                'level': record.levelname,
                'logger': record.name,
                'message': record.getMessage(),
            }

            # Map new column names to record attributes
            # New columns map to both new and legacy field names
            column_mappings = {
                'user_query': ['user_query', 'query'],  # Try new name first, fall back to legacy
                'intent_result': ['intent_result', 'intent'],
                'intent_confidence': ['intent_confidence', 'confidence'],
                'product_skus': ['product_skus'],
            }

            # Add mapped fields
            for col, attr_names in column_mappings.items():
                for attr in attr_names:
                    if hasattr(record, attr) and getattr(record, attr) is not None:
                        row[col] = self._flatten_value(getattr(record, attr))
                        break
                else:
                    row[col] = ''

            # Handle flattened filter fields from 'filters' dict
            filters = getattr(record, 'filters', None)
            if isinstance(filters, dict):
                row['filters_category'] = filters.get('category', '') or ''
                row['filters_connector_from'] = filters.get('connector_from', '') or ''
                row['filters_connector_to'] = filters.get('connector_to', '') or ''
                row['filters_length'] = filters.get('length', '') or ''
                row['filters_length_unit'] = filters.get('length_unit', '') or ''
                features = filters.get('features', [])
                row['filters_features'] = ', '.join(features) if features else ''

            # Add remaining fields directly from record
            direct_fields = [
                'event', 'session_id', 'products_found', 'products_shown',
                'response_time_ms', 'search_latency_ms', 'llm_latency_ms',
                'query', 'intent', 'confidence', 'category', 'connector_from',
                'connector_to', 'length', 'error_type', 'tier', 'filters'
            ]
            for col in direct_fields:
                if col not in row and hasattr(record, col):
                    row[col] = self._flatten_value(getattr(record, col))

            # Fill missing columns with empty strings
            for col in self.CSV_COLUMNS:
                if col not in row:
                    row[col] = ''

            # Write as CSV line
            values = [self._escape_csv(row.get(col, '')) for col in self.CSV_COLUMNS]
            self._file_handle.write(','.join(values) + '\n')
            self._file_handle.flush()

        except Exception:
            self.handleError(record)

    def _escape_csv(self, value: str) -> str:
        """Escape value for CSV (quote if contains comma, quote, or newline)."""
        value = str(value) if value else ''
        if ',' in value or '"' in value or '\n' in value:
            return '"' + value.replace('"', '""') + '"'
        return value

    def close(self):
        """Clean up handler."""
        self._close_file()
        super().close()


# =============================================================================
# Logger Setup
# =============================================================================

_loggers: Dict[str, logging.Logger] = {}
_initialized = False


def setup_logging(
    log_dir: str = "logs",
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
    enable_console: bool = True,
    enable_file: bool = True,
    enable_csv: bool = True,
    enable_error_log: bool = True,
) -> None:
    """
    Initialize the logging system.

    Creates:
    - logs/stbot.log (all logs, rotating daily, 30-day retention)
    - logs/errors.log (ERROR and above, rotating daily, 30-day retention)
    - logs/csv/stbot-YYYY-MM-DD.csv (CSV for Power BI, daily)
    - Console output (if enabled)

    Args:
        log_dir: Directory for log files
        console_level: Minimum level for console output
        file_level: Minimum level for file output
        enable_console: Whether to output to console
        enable_file: Whether to write to stbot.log (detailed logs)
        enable_csv: Whether to write CSV files for Power BI
        enable_error_log: Whether to write errors.log (ERROR and above)
    """
    global _initialized
    if _initialized:
        return

    # Create log directory
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # Get root logger for our application
    root_logger = logging.getLogger("stbot")
    root_logger.setLevel(logging.DEBUG)  # Capture all, handlers filter

    # Remove any existing handlers
    root_logger.handlers = []

    # Console handler (human-readable)
    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(console_level)
        console_handler.setFormatter(ConsoleFormatter())
        root_logger.addHandler(console_handler)

    # File handler - all logs (JSON, rotating daily)
    if enable_file:
        main_log_file = log_path / "stbot.log"
        file_handler = TimedRotatingFileHandler(
            filename=str(main_log_file),
            when="midnight",
            interval=1,
            backupCount=30,  # Keep 30 days
            encoding="utf-8",
        )
        file_handler.setLevel(file_level)
        file_handler.setFormatter(JSONFormatter())
        file_handler.suffix = "%Y-%m-%d"
        root_logger.addHandler(file_handler)

    # Error log file (ERROR and above only) - can be enabled independently
    if enable_error_log:
        error_log_file = log_path / "errors.log"
        error_handler = TimedRotatingFileHandler(
            filename=str(error_log_file),
            when="midnight",
            interval=1,
            backupCount=30,
            encoding="utf-8",
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(JSONFormatter())
        error_handler.suffix = "%Y-%m-%d"
        root_logger.addHandler(error_handler)

    # CSV handler for Power BI (daily files in logs/csv/)
    if enable_csv:
        csv_handler = CSVHandler(log_dir=log_dir)
        csv_handler.setLevel(logging.DEBUG)  # Capture all events
        root_logger.addHandler(csv_handler)

    _initialized = True


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a module.

    Args:
        name: Module name (usually __name__)

    Returns:
        Configured logger instance

    Example:
        logger = get_logger(__name__)
        logger.info("Search started", extra={"query": "USB-C cable"})
    """
    # Note: Don't auto-initialize here - let app.py control logging setup
    # If logging isn't set up yet, logs will go to root logger (console only)

    # Create child logger under stbot namespace
    if name.startswith("stbot."):
        logger_name = name
    else:
        logger_name = f"stbot.{name}"

    if logger_name not in _loggers:
        _loggers[logger_name] = logging.getLogger(logger_name)

    return _loggers[logger_name]


# =============================================================================
# Context Manager for Session Tracking
# =============================================================================

class LogContext:
    """
    Context manager for tracking session-level logging context.

    Usage:
        with LogContext(session_id="abc123") as ctx:
            ctx.log_query("USB-C cable")
            # ... do work ...
            ctx.log_response(products_found=5, response_time_ms=450)
    """

    def __init__(self, session_id: Optional[str] = None):
        """Initialize log context."""
        self.session_id = session_id or str(uuid.uuid4())[:8]
        self.start_time = None
        self.logger = get_logger("context")

    def __enter__(self) -> "LogContext":
        """Enter context."""
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """Exit context, log any errors."""
        if exc_type is not None:
            self.log_error(exc_val, exc_tb)
        return False  # Don't suppress exceptions

    def elapsed_ms(self) -> float:
        """Get elapsed time in milliseconds."""
        if self.start_time is None:
            return 0.0
        return (time.perf_counter() - self.start_time) * 1000

    def log_query(self, query: str, **extra) -> None:
        """Log an incoming query."""
        self.logger.info(
            "User query received",
            extra={
                "event": "user_query",
                "session_id": self.session_id,
                "query": query,
                **extra
            }
        )

    def log_response(self, **extra) -> None:
        """Log the response being sent."""
        self.logger.info(
            "Response sent",
            extra={
                "event": "response_sent",
                "session_id": self.session_id,
                "total_latency_ms": round(self.elapsed_ms(), 2),
                **extra
            }
        )

    def log_error(self, error: Exception, tb=None) -> None:
        """Log an error."""
        self.logger.error(
            f"Error: {error}",
            extra={
                "event": "error",
                "session_id": self.session_id,
                "error_type": type(error).__name__,
                "stack_trace": "".join(traceback.format_tb(tb)) if tb else None,
            },
            exc_info=(type(error), error, tb) if tb else None
        )


# =============================================================================
# Convenience Functions
# =============================================================================

def log_query(
    session_id: str,
    query: str,
    intent: Optional[str] = None,
    confidence: Optional[float] = None,
    **extra
) -> None:
    """
    Log a user query with context.

    Args:
        session_id: Session identifier
        query: User's query text
        intent: Classified intent type
        confidence: Intent classification confidence
        **extra: Additional fields to log
    """
    logger = get_logger("query")
    logger.info(
        "User query",
        extra={
            "event": "user_query",
            "session_id": session_id,
            "query": query,
            "intent": intent,
            "confidence": confidence,
            **extra
        }
    )


def log_intent(
    session_id: str,
    query: str,
    intent: str,
    confidence: float,
    reasoning: str,
    classification_time_ms: float,
    **extra
) -> None:
    """
    Log intent classification result.

    Args:
        session_id: Session identifier
        query: Original query
        intent: Classified intent type
        confidence: Classification confidence
        reasoning: Why this intent was chosen
        classification_time_ms: Time taken to classify
        **extra: Additional fields
    """
    logger = get_logger("intent")
    logger.info(
        f"Intent classified: {intent}",
        extra={
            "event": "intent_classification",
            "session_id": session_id,
            "query": query,
            "intent": intent,
            "confidence": confidence,
            "reasoning": reasoning,
            "intent_classification_ms": round(classification_time_ms, 2),
            **extra
        }
    )


def log_filters(
    session_id: str,
    query: str,
    filters: Dict[str, Any],
    extraction_time_ms: float,
    **extra
) -> None:
    """
    Log filter extraction result.

    Args:
        session_id: Session identifier
        query: Original query
        filters: Extracted filters
        extraction_time_ms: Time taken to extract
        **extra: Additional fields
    """
    logger = get_logger("filters")
    logger.info(
        "Filters extracted",
        extra={
            "event": "filter_extraction",
            "session_id": session_id,
            "query": query,
            "filters": filters,
            "filter_extraction_ms": round(extraction_time_ms, 2),
            **extra
        }
    )


def log_search(
    session_id: str,
    filters: Dict[str, Any],
    products_found: int,
    tier: str,
    search_time_ms: float,
    dropped_filters: Optional[list] = None,
    **extra
) -> None:
    """
    Log search result.

    Args:
        session_id: Session identifier
        filters: Filters used for search
        products_found: Number of products found
        tier: Search tier used (tier1, tier2, etc.)
        search_time_ms: Time taken to search
        dropped_filters: Any filters that were relaxed
        **extra: Additional fields
    """
    logger = get_logger("search")
    logger.info(
        f"Search complete: {products_found} products found (tier: {tier})",
        extra={
            "event": "search_complete",
            "session_id": session_id,
            "filters": filters,
            "products_found": products_found,
            "tier": tier,
            "search_latency_ms": round(search_time_ms, 2),
            "dropped_filters": dropped_filters,
            **extra
        }
    )


def log_llm_call(
    session_id: str,
    model: str,
    endpoint: str,
    latency_ms: float,
    tokens_used: Optional[int] = None,
    success: bool = True,
    error: Optional[str] = None,
    **extra
) -> None:
    """
    Log LLM API call.

    Args:
        session_id: Session identifier
        model: Model used (e.g., "gpt-4o")
        endpoint: API endpoint/function called
        latency_ms: Time taken for API call
        tokens_used: Tokens consumed (if available)
        success: Whether call succeeded
        error: Error message if failed
        **extra: Additional fields
    """
    logger = get_logger("llm")
    level = logging.INFO if success else logging.ERROR

    message = f"LLM call: {endpoint}" if success else f"LLM call failed: {endpoint}"

    logger.log(
        level,
        message,
        extra={
            "event": "llm_api_call",
            "session_id": session_id,
            "llm_model": model,
            "api_endpoint": endpoint,
            "llm_latency_ms": round(latency_ms, 2),
            "llm_tokens": tokens_used,
            "success": success,
            "error": error,
            **extra
        }
    )


def log_products_shown(
    session_id: str,
    products: list,
    query: str,
    **extra
) -> None:
    """
    Log products shown to user.

    Args:
        session_id: Session identifier
        products: List of products shown
        query: Original query
        **extra: Additional fields
    """
    logger = get_logger("products")

    product_skus = [p.product_number if hasattr(p, 'product_number') else str(p) for p in products[:10]]

    logger.info(
        f"Showing {len(products)} products",
        extra={
            "event": "products_shown",
            "session_id": session_id,
            "products_shown": len(products),
            "product_skus": product_skus,
            "query": query,
            **extra
        }
    )


def log_response(
    session_id: str,
    intent: str,
    products_found: int,
    response_time_ms: float,
    **extra
) -> None:
    """
    Log final response sent to user.

    Args:
        session_id: Session identifier
        intent: Intent type
        products_found: Number of products shown
        response_time_ms: Total response time
        **extra: Additional fields
    """
    logger = get_logger("response")
    logger.info(
        f"Response sent: {intent} with {products_found} products",
        extra={
            "event": "response_sent",
            "session_id": session_id,
            "intent": intent,
            "products_found": products_found,
            "response_time_ms": round(response_time_ms, 2),
            **extra
        }
    )


def log_error(
    session_id: str,
    error: Exception,
    context: Optional[str] = None,
    **extra
) -> None:
    """
    Log an error with full context.

    Args:
        session_id: Session identifier
        error: The exception
        context: Additional context about what was happening
        **extra: Additional fields
    """
    logger = get_logger("error")
    logger.error(
        f"Error: {type(error).__name__}: {error}",
        extra={
            "event": "error",
            "session_id": session_id,
            "error_type": type(error).__name__,
            "stack_trace": traceback.format_exc(),
            "context": context,
            **extra
        },
        exc_info=True
    )


def log_guidance(
    session_id: str,
    setup_type: str,
    phase: str,
    **extra
) -> None:
    """
    Log guidance flow progress.

    Args:
        session_id: Session identifier
        setup_type: Type of setup (multi_monitor, dock_selection, etc.)
        phase: Current guidance phase
        **extra: Additional fields
    """
    logger = get_logger("guidance")
    logger.info(
        f"Guidance: {setup_type} - {phase}",
        extra={
            "event": "guidance_progress",
            "session_id": session_id,
            "setup_type": setup_type,
            "guidance_phase": phase,
            **extra
        }
    )


def log_conversation_turn(
    session_id: str,
    user_query: str,
    intent_result: str,
    intent_confidence: float,
    products_found: int = 0,
    products_shown: int = 0,
    product_skus: Optional[list] = None,
    filters: Optional[Dict[str, Any]] = None,
    response_time_ms: Optional[float] = None,
    **extra
) -> None:
    """
    Log a complete conversation turn with all context for Power BI analysis.

    This is the primary log event for conversation analytics. It captures
    everything about a single user interaction in one row.

    Args:
        session_id: Unique session identifier (persists across conversation)
        user_query: The actual user message/question
        intent_result: Classified intent type (new_search, followup, greeting, etc.)
        intent_confidence: Intent classification confidence (0.0 to 1.0)
        products_found: Total matching products from search
        products_shown: Number of products displayed to user (usually top 3-5)
        product_skus: List of SKUs shown to user (pipe-separated in CSV)
        filters: Extracted search filters dict (category, connectors, length, etc.)
        response_time_ms: Total response time in milliseconds
        **extra: Additional fields to log

    Example:
        log_conversation_turn(
            session_id="session_20260101_134318",
            user_query="I need a 10ft HDMI cable",
            intent_result="new_search",
            intent_confidence=0.90,
            products_found=142,
            products_shown=3,
            product_skus=["HDMM10M", "HDMI2-CABLE-4K60-10M", "HD2AP-10M"],
            filters={"category": "Cables", "connector_from": "HDMI", "length": 10.0},
            response_time_ms=450.5
        )
    """
    logger = get_logger("conversation")

    # Format product SKUs as pipe-separated string for CSV
    skus_str = '|'.join(product_skus) if product_skus else ''

    logger.info(
        f"Conversation turn: {intent_result}",
        extra={
            "event": "conversation_turn",
            "session_id": session_id,
            # New primary fields for Power BI
            "user_query": user_query,
            "intent_result": intent_result,
            "intent_confidence": round(intent_confidence, 2) if intent_confidence else None,
            # Results
            "products_found": products_found,
            "products_shown": products_shown,
            "product_skus": skus_str,
            # Filters (will be flattened by CSVHandler)
            "filters": filters or {},
            # Performance
            "response_time_ms": round(response_time_ms, 2) if response_time_ms else None,
            # Legacy field mappings for backwards compatibility
            "query": user_query,
            "intent": intent_result,
            "confidence": round(intent_confidence, 2) if intent_confidence else None,
            **extra
        }
    )


# =============================================================================
# Performance Timing Decorator
# =============================================================================

def timed(event_name: str, logger_name: str = "performance"):
    """
    Decorator to time function execution and log it.

    Usage:
        @timed("intent_classification")
        def classify(query: str) -> Intent:
            ...

    Args:
        event_name: Name of the event for logging
        logger_name: Logger to use
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                elapsed_ms = (time.perf_counter() - start) * 1000

                logger = get_logger(logger_name)
                logger.debug(
                    f"{event_name} completed",
                    extra={
                        "event": f"{event_name}_timing",
                        "elapsed_ms": round(elapsed_ms, 2),
                        "function": func.__name__,
                    }
                )
                return result
            except Exception as e:
                elapsed_ms = (time.perf_counter() - start) * 1000
                logger = get_logger(logger_name)
                logger.error(
                    f"{event_name} failed after {elapsed_ms:.2f}ms",
                    extra={
                        "event": f"{event_name}_error",
                        "elapsed_ms": round(elapsed_ms, 2),
                        "function": func.__name__,
                        "error_type": type(e).__name__,
                    },
                    exc_info=True
                )
                raise
        return wrapper
    return decorator


class Timer:
    """
    Context manager for timing code blocks.

    Usage:
        with Timer() as t:
            # ... do work ...
        print(f"Took {t.elapsed_ms}ms")
    """

    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.elapsed_ms = 0.0

    def __enter__(self) -> "Timer":
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args) -> None:
        self.end_time = time.perf_counter()
        self.elapsed_ms = (self.end_time - self.start_time) * 1000


# =============================================================================
# Initialize on import (with defaults)
# =============================================================================

# Don't auto-initialize - let the app control this
# setup_logging()
