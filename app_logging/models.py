"""Logging data models."""

from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime


@dataclass
class ConversationLog:
    """Single conversation log entry."""
    timestamp: datetime
    session_id: str
    query_number: int
    user_message: str
    bot_response: str
    products_shown: list[dict] = field(default_factory=list)
    intent_type: Optional[str] = None
    filters_applied: dict = field(default_factory=dict)
    match_status: str = "other"
