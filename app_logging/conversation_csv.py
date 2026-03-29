"""
Clean Conversation CSV Logger for Power BI Analysis.

Writes ONE row per user interaction with all relevant data.
This is separate from the debug logs - it's designed for easy Power BI import.

Output file: logs/conversations.csv
"""

import csv
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from app_logging.structured_logging import get_logger

# Module-level logger
_logger = get_logger("core.conversation_csv")


class ConversationCSVLogger:
    """
    Logs conversation turns to a clean CSV file for Power BI analysis.

    Each row represents ONE user interaction with:
    - Timestamp and session info
    - User's query text
    - Intent classification
    - Search filters applied
    - Products shown
    - Response time

    Usage:
        logger = ConversationCSVLogger()
        logger.log(
            session_id="session_123",
            user_query="I need a USB-C to HDMI cable",
            intent="new_search",
            confidence=0.9,
            category="Cables",
            products_found=50,
            products_shown=3,
            product_skus=["SKU1", "SKU2", "SKU3"],
            response_time_ms=250.5
        )
    """

    # Clean column order optimized for Power BI
    COLUMNS = [
        'timestamp',
        'session_id',
        'user_query',
        'bot_response',
        'intent',
        'confidence',
        'category',
        'connector_from',
        'connector_to',
        'length',
        'length_unit',
        'features',
        'products_found',
        'products_shown',
        'product_skus',
        'response_time_ms',
    ]

    def __init__(self, log_dir: str = "logs"):
        """
        Initialize the conversation CSV logger.

        Args:
            log_dir: Directory to store the CSV file
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.log_dir / "conversations.csv"
        self._ensure_headers()

    def _ensure_headers(self):
        """Create CSV with headers if it doesn't exist."""
        if not self.csv_path.exists():
            with open(self.csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(self.COLUMNS)

    def log(
        self,
        session_id: str,
        user_query: str,
        bot_response: str,
        intent: str,
        confidence: float,
        category: Optional[str] = None,
        connector_from: Optional[str] = None,
        connector_to: Optional[str] = None,
        length: Optional[float] = None,
        length_unit: Optional[str] = None,
        features: Optional[List[str]] = None,
        products_found: int = 0,
        products_shown: int = 0,
        product_skus: Optional[List[str]] = None,
        response_time_ms: Optional[float] = None,
    ) -> None:
        """
        Log a single conversation turn.

        Args:
            session_id: Unique session identifier
            user_query: The user's message
            bot_response: The bot's response text
            intent: Classified intent (new_search, followup, greeting, etc.)
            confidence: Intent confidence score (0-1)
            category: Product category filter (e.g., "Cables")
            connector_from: Source connector filter
            connector_to: Target connector filter
            length: Length filter value
            length_unit: Length unit (ft, m, in)
            features: Feature filters (e.g., ["4K", "HDR"])
            products_found: Total matching products
            products_shown: Products displayed to user
            product_skus: List of SKUs shown
            response_time_ms: Total response time
        """
        row = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'session_id': session_id or '',
            'user_query': user_query or '',
            'bot_response': bot_response or '',
            'intent': intent or '',
            'confidence': f"{confidence:.2f}" if confidence is not None else '',
            'category': category or '',
            'connector_from': connector_from or '',
            'connector_to': connector_to or '',
            'length': str(length) if length is not None else '',
            'length_unit': length_unit or '',
            'features': '|'.join(features) if features else '',
            'products_found': str(products_found),
            'products_shown': str(products_shown),
            'product_skus': '|'.join(product_skus) if product_skus else '',
            'response_time_ms': f"{response_time_ms:.2f}" if response_time_ms is not None else '',
        }

        # Write row (with error handling for locked files)
        try:
            with open(self.csv_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([self._escape(row.get(col, '')) for col in self.COLUMNS])
        except PermissionError:
            # File is locked (open in Excel/VS Code) - log warning but don't crash
            _logger.warning("Could not write to CSV (file locked)", extra={"path": str(self.csv_path)})

    def log_from_filters(
        self,
        session_id: str,
        user_query: str,
        bot_response: str,
        intent: str,
        confidence: float,
        filters: Optional[Dict[str, Any]] = None,
        products_found: int = 0,
        products_shown: int = 0,
        product_skus: Optional[List[str]] = None,
        response_time_ms: Optional[float] = None,
    ) -> None:
        """
        Log a conversation turn using a filters dict.

        This is a convenience method that extracts filter values
        from a dict (as returned by FilterExtractor).

        Args:
            session_id: Unique session identifier
            user_query: The user's message
            bot_response: The bot's response text
            intent: Classified intent
            confidence: Intent confidence score
            filters: Dict with keys like category, connector_from, etc.
            products_found: Total matching products
            products_shown: Products displayed to user
            product_skus: List of SKUs shown
            response_time_ms: Total response time
        """
        filters = filters or {}

        self.log(
            session_id=session_id,
            user_query=user_query,
            bot_response=bot_response,
            intent=intent,
            confidence=confidence,
            category=filters.get('category'),
            connector_from=filters.get('connector_from'),
            connector_to=filters.get('connector_to'),
            length=filters.get('length'),
            length_unit=filters.get('length_unit'),
            features=filters.get('features'),
            products_found=products_found,
            products_shown=products_shown,
            product_skus=product_skus,
            response_time_ms=response_time_ms,
        )

    def _escape(self, value: str) -> str:
        """Escape value for CSV - handles commas, quotes, newlines."""
        value = str(value) if value else ''
        # Replace newlines with spaces for clean single-line output
        value = value.replace('\n', ' ').replace('\r', ' ')
        return value


# Global instance for convenience
_conversation_logger: Optional[ConversationCSVLogger] = None


def log_conversation(
    session_id: str,
    user_query: str,
    bot_response: str,
    intent: str,
    confidence: float,
    filters: Optional[Dict[str, Any]] = None,
    products_found: int = 0,
    products_shown: int = 0,
    product_skus: Optional[List[str]] = None,
    response_time_ms: Optional[float] = None,
) -> None:
    """
    Convenience function to log a conversation turn.

    This is the primary function to call from the orchestrator.
    """
    global _conversation_logger
    if _conversation_logger is None:
        _conversation_logger = ConversationCSVLogger()
    _conversation_logger.log_from_filters(
        session_id=session_id,
        user_query=user_query,
        bot_response=bot_response,
        intent=intent,
        confidence=confidence,
        filters=filters,
        products_found=products_found,
        products_shown=products_shown,
        product_skus=product_skus,
        response_time_ms=response_time_ms,
    )
