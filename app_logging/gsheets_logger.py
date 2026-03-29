"""
Google Sheets Logger for ST-Bot Cloud Deployment.

Logs conversations and errors to Google Sheets for persistent storage
when running on Streamlit Cloud (where local files are ephemeral).

Sheets structure:
- conversations-YYYY-MM-DD: Daily conversation logs
- errors-YYYY-MM-DD: Daily error logs
"""

import sys
from datetime import datetime
from typing import Any, Dict, List, Optional

from app_logging.structured_logging import get_logger

# Module-level logger
_logger = get_logger("core.gsheets_logger")

try:
    import gspread
    from google.oauth2.service_account import Credentials
    GSPREAD_AVAILABLE = True
except ImportError:
    GSPREAD_AVAILABLE = False


# Column order for conversations (matches local CSV)
CONVERSATION_COLUMNS = [
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

# Column order for errors
ERROR_COLUMNS = [
    'timestamp',
    'session_id',
    'error_type',
    'error_message',
    'stack_trace',
    'context',
]


class GoogleSheetsLogger:
    """
    Logs data to Google Sheets with daily sheet rotation.

    Each day gets its own sheet (tab) within the spreadsheet:
    - conversations-2026-01-03
    - errors-2026-01-03
    """

    SCOPES = [
        'https://www.googleapis.com/auth/spreadsheets',
        'https://www.googleapis.com/auth/drive'
    ]

    def __init__(self, spreadsheet_id: str, credentials_dict: Dict[str, Any]):
        """
        Initialize the Google Sheets logger.

        Args:
            spreadsheet_id: The Google Sheets spreadsheet ID
            credentials_dict: Service account credentials as a dict
        """
        self.spreadsheet_id = spreadsheet_id
        self.credentials_dict = credentials_dict
        self._client = None
        self._spreadsheet = None
        self._sheet_cache = {}  # Cache worksheet references

    def _get_client(self):
        """Get or create the gspread client."""
        if self._client is None:
            credentials = Credentials.from_service_account_info(
                self.credentials_dict,
                scopes=self.SCOPES
            )
            self._client = gspread.authorize(credentials)
        return self._client

    def _get_spreadsheet(self):
        """Get or open the spreadsheet."""
        if self._spreadsheet is None:
            client = self._get_client()
            self._spreadsheet = client.open_by_key(self.spreadsheet_id)
        return self._spreadsheet

    def _get_or_create_sheet(self, sheet_name: str, columns: List[str]):
        """Get existing sheet or create new one with headers."""
        # Check cache first
        if sheet_name in self._sheet_cache:
            return self._sheet_cache[sheet_name]

        spreadsheet = self._get_spreadsheet()

        try:
            # Try to get existing sheet
            worksheet = spreadsheet.worksheet(sheet_name)
        except gspread.WorksheetNotFound:
            # Create new sheet with headers
            worksheet = spreadsheet.add_worksheet(
                title=sheet_name,
                rows=1000,
                cols=len(columns)
            )
            # Add headers
            worksheet.append_row(columns, value_input_option='RAW')

        self._sheet_cache[sheet_name] = worksheet
        return worksheet

    def _get_today_sheet_name(self, prefix: str) -> str:
        """Get sheet name for today."""
        return f"{prefix}-{datetime.now().strftime('%Y-%m-%d')}"

    def log_conversation(
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
    ) -> bool:
        """
        Log a conversation turn to Google Sheets.

        Returns:
            True if logged successfully, False otherwise
        """
        try:
            sheet_name = self._get_today_sheet_name("conversations")
            worksheet = self._get_or_create_sheet(sheet_name, CONVERSATION_COLUMNS)

            row = [
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                session_id or '',
                user_query or '',
                bot_response or '',
                intent or '',
                f"{confidence:.2f}" if confidence is not None else '',
                category or '',
                connector_from or '',
                connector_to or '',
                str(length) if length is not None else '',
                length_unit or '',
                '|'.join(features) if features else '',
                str(products_found),
                str(products_shown),
                '|'.join(product_skus) if product_skus else '',
                f"{response_time_ms:.2f}" if response_time_ms is not None else '',
            ]

            worksheet.append_row(row, value_input_option='RAW')
            return True

        except Exception as e:
            _logger.warning("Failed to log to Google Sheets", extra={"error": str(e)})
            return False

    def log_conversation_from_filters(
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
    ) -> bool:
        """Log conversation using a filters dict."""
        filters = filters or {}
        return self.log_conversation(
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

    def log_error(
        self,
        session_id: str,
        error_type: str,
        error_message: str,
        stack_trace: Optional[str] = None,
        context: Optional[str] = None,
    ) -> bool:
        """
        Log an error to Google Sheets.

        Returns:
            True if logged successfully, False otherwise
        """
        try:
            sheet_name = self._get_today_sheet_name("errors")
            worksheet = self._get_or_create_sheet(sheet_name, ERROR_COLUMNS)

            row = [
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                session_id or '',
                error_type or '',
                error_message or '',
                (stack_trace or '')[:50000],  # Limit stack trace length
                context or '',
            ]

            worksheet.append_row(row, value_input_option='RAW')
            return True

        except Exception as e:
            _logger.warning("Failed to log error to Google Sheets", extra={"error": str(e)})
            return False


# Global instance
_gsheets_logger: Optional[GoogleSheetsLogger] = None


def init_gsheets_logger(spreadsheet_id: str, credentials_dict: Dict[str, Any]) -> Optional[GoogleSheetsLogger]:
    """
    Initialize the global Google Sheets logger.

    Args:
        spreadsheet_id: The Google Sheets spreadsheet ID
        credentials_dict: Service account credentials as a dict

    Returns:
        GoogleSheetsLogger instance or None if gspread not available
    """
    global _gsheets_logger

    if not GSPREAD_AVAILABLE:
        _logger.warning("gspread not installed - Google Sheets logging disabled")
        return None

    _gsheets_logger = GoogleSheetsLogger(spreadsheet_id, credentials_dict)
    return _gsheets_logger


def get_gsheets_logger() -> Optional[GoogleSheetsLogger]:
    """Get the global Google Sheets logger instance."""
    return _gsheets_logger


def log_to_gsheets(
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
) -> bool:
    """
    Convenience function to log a conversation to Google Sheets.

    Returns True if logged, False if logger not initialized or failed.
    """
    logger = get_gsheets_logger()
    if logger is None:
        return False

    return logger.log_conversation_from_filters(
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


def log_error_to_gsheets(
    session_id: str,
    error_type: str,
    error_message: str,
    stack_trace: Optional[str] = None,
    context: Optional[str] = None,
) -> bool:
    """
    Convenience function to log an error to Google Sheets.

    Returns True if logged, False if logger not initialized or failed.
    """
    logger = get_gsheets_logger()
    if logger is None:
        return False

    return logger.log_error(
        session_id=session_id,
        error_type=error_type,
        error_message=error_message,
        stack_trace=stack_trace,
        context=context,
    )
