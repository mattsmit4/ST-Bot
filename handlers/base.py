"""
Base handler and context classes for ST-Bot intent handlers.

Provides the common interface and shared context for all handlers.
"""

from dataclasses import dataclass, field
from typing import Any, Optional, List
from abc import ABC, abstractmethod

from core.models import (
    ConversationContext, Intent, Product, SearchFilters, SearchResult
)


@dataclass
class HandlerContext:
    """
    Context passed to all intent handlers.

    Contains everything a handler needs to process a query:
    - The query itself
    - Classified intent
    - Conversation context (products shown, pending guidance, etc.)
    - All products for searching
    - Component references
    """
    query: str
    intent: Intent
    context: ConversationContext
    all_products: List[Product]
    debug_mode: bool = False

    # Component references (set by orchestrator)
    filter_extractor: Any = None
    search_engine: Any = None
    query_analyzer: Any = None

    # Debug output collector
    debug_lines: List[str] = field(default_factory=list)

    def add_debug(self, message: str) -> None:
        """Add a debug message."""
        if self.debug_mode:
            self.debug_lines.append(message)


@dataclass
class HandlerResult:
    """
    Result returned by intent handlers.

    Contains the response text and any side effects:
    - Products to set in context
    - Whether to save/clear clarification state
    - Whether to save/clear narrowing state
    - Extracted filters (for logging)
    """
    response: str
    products_to_set: Optional[List[Product]] = None
    save_pending_clarification: bool = False
    clear_pending_clarification: bool = False
    save_pending_narrowing: bool = False
    clear_pending_narrowing: bool = False
    # For logging - extracted search filters (optional)
    filters_for_logging: Optional[dict] = None
    # For logging - total products found before filtering to top N
    products_found: int = 0


class BaseHandler(ABC):
    """
    Base class for all intent handlers.

    Each handler processes a specific intent type and returns a HandlerResult.
    Handlers should be stateless - all state is in HandlerContext.
    """

    @abstractmethod
    def handle(self, ctx: HandlerContext) -> HandlerResult:
        """
        Process the intent and return a result.

        Args:
            ctx: Handler context with query, intent, and all components

        Returns:
            HandlerResult with response and any side effects
        """
        pass

    def _clear_stale_context(self, ctx: HandlerContext) -> None:
        """
        Clear stale context when starting a new flow.

        Call this when handling intents that start fresh (new search,
        explicit SKU, new guidance flow).
        """
        # Clear comparison context
        if ctx.context.has_comparison_context():
            ctx.add_debug(f"CLEARING STALE COMPARISON: {ctx.context.last_comparison_indices}")
            ctx.context.clear_comparison_context()

        # Clear pending clarification
        if ctx.context.has_pending_clarification():
            ctx.add_debug(f"CLEARING STALE CLARIFICATION: {ctx.context.pending_clarification.vague_type}")
            ctx.context.clear_pending_clarification()
