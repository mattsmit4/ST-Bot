"""Conversation state tracking."""

from dataclasses import dataclass, field
from typing import Optional
from enum import Enum
from datetime import datetime

from core.models.product import Product


@dataclass
class Message:
    """A single message in the conversation history."""
    role: str       # 'user' or 'assistant'
    content: str
    timestamp: datetime = field(default_factory=datetime.now)


class VagueQueryType(Enum):
    """Types of vague queries that need clarification."""
    CABLE = "cable"
    GENERIC = "generic"
    PORTS = "ports"
    CONNECTOR = "connector"
    UNCERTAIN = "uncertain"


class ClarificationMissing(Enum):
    """Types of information that may be missing from a vague query."""
    CONNECTOR_FROM = "connector_from"
    CONNECTOR_TO = "connector_to"
    USE_CASE = "use_case"
    DEVICE_TYPE = "device_type"


@dataclass
class PendingClarification:
    """
    Tracks a vague query that needs clarification before searching.
    Pure state — business logic (get_search_filters) lives in the clarification handler.
    """
    vague_type: VagueQueryType
    original_query: str
    missing_info: list[ClarificationMissing]
    collected_info: dict = field(default_factory=dict)
    questions_asked: int = 0
    original_category: Optional[str] = None

    def has_enough_info(self) -> bool:
        """Check if we have enough info to search (pure state check)."""
        return 'connector_from' in self.collected_info or 'use_case' in self.collected_info


@dataclass
class PendingNarrowing:
    """Tracks a product narrowing conversation when search returned too many tied results."""
    original_query: str
    original_filters: dict
    product_skus: list[str] = field(default_factory=list)
    product_pool: list = field(default_factory=list)
    questions_asked: int = 0
    max_questions: int = 3
    last_attribute: Optional[str] = None
    asked_attributes: list[str] = field(default_factory=list)
    last_options: list[str] = field(default_factory=list)
    option_filters: list = field(default_factory=list)
    is_fallback: bool = False


@dataclass
class SearchHistoryEntry:
    """Entry in search history for recall functionality."""
    query: str
    products: list
    category_hint: str
    timestamp: datetime
    filters: Optional[dict] = None


@dataclass
class ConversationContext:
    """
    Conversation state tracking.
    Tracks current products, pending flows, and search history.
    """
    current_products: Optional[list[Product]] = None
    previous_products: Optional[list[Product]] = None
    last_product: Optional[Product] = None
    last_filters: Optional[dict] = None
    last_query: Optional[str] = None
    query_count: int = 0
    session_id: Optional[str] = None
    pending_clarification: Optional[PendingClarification] = None
    pending_narrowing: Optional[PendingNarrowing] = None
    search_history: list[SearchHistoryEntry] = field(default_factory=list)
    max_search_history: int = 10
    last_comparison_indices: Optional[list[int]] = None
    messages: list[Message] = field(default_factory=list)
    max_messages: int = 50

    def has_multi_product_context(self) -> bool:
        return bool(self.current_products)

    def has_single_product_context(self) -> bool:
        return bool(self.last_product)

    def clear_products(self) -> None:
        self.current_products = None
        self.last_product = None

    def set_multi_products(self, products: list[Product]) -> None:
        self.current_products = products
        self.last_product = None

    def set_single_product(self, product: Product) -> None:
        self.last_product = product
        self.current_products = None

    def save_products_before_filter(self) -> None:
        if self.current_products and len(self.current_products) > 1:
            self.previous_products = self.current_products.copy()

    def restore_previous_products(self) -> bool:
        if self.previous_products:
            self.current_products = self.previous_products
            self.previous_products = None
            return True
        return False

    def has_previous_products(self) -> bool:
        return self.previous_products is not None and len(self.previous_products) > 0

    def add_to_search_history(
        self, query: str, products: list, category_hint: str,
        filters: Optional[dict] = None
    ) -> None:
        entry = SearchHistoryEntry(
            query=query, products=products, category_hint=category_hint,
            timestamp=datetime.now(), filters=filters
        )
        self.search_history.append(entry)
        if len(self.search_history) > self.max_search_history:
            self.search_history = self.search_history[-self.max_search_history:]

    def find_in_history(self, category_hint: str) -> Optional[SearchHistoryEntry]:
        category_lower = category_hint.lower()
        for entry in reversed(self.search_history):
            if category_lower in entry.category_hint.lower():
                return entry
        return None

    def get_history_categories(self) -> list[str]:
        return [entry.category_hint for entry in self.search_history]

    # --- Message history ---
    def add_message(self, role: str, content: str) -> None:
        """Add a message to conversation history."""
        self.messages.append(Message(role=role, content=content))
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]

    def get_last_message(self, role: str = None) -> Optional[Message]:
        """Get the last message, optionally filtered by role."""
        if role:
            for msg in reversed(self.messages):
                if msg.role == role:
                    return msg
            return None
        return self.messages[-1] if self.messages else None

    def get_conversation_history(self, limit: int = 10, role: str = None) -> list[Message]:
        """Get recent messages, optionally filtered by role."""
        msgs = self.messages
        if role:
            msgs = [m for m in msgs if m.role == role]
        return msgs[-limit:]

    # --- Clarification state ---
    def has_pending_clarification(self) -> bool:
        return self.pending_clarification is not None

    def set_pending_clarification(self, clarification: PendingClarification) -> None:
        self.pending_clarification = clarification

    def clear_pending_clarification(self) -> None:
        self.pending_clarification = None

    def update_clarification(self, collected_info: dict) -> None:
        if self.pending_clarification:
            self.pending_clarification.collected_info.update(collected_info)
            self.pending_clarification.questions_asked += 1

    # --- Narrowing state ---
    def has_pending_narrowing(self) -> bool:
        return self.pending_narrowing is not None

    def set_pending_narrowing(self, narrowing: PendingNarrowing) -> None:
        self.pending_narrowing = narrowing

    def clear_pending_narrowing(self) -> None:
        self.pending_narrowing = None

    def escape_narrowing(self) -> None:
        """Clear narrowing state while preserving context for re-classification."""
        if self.pending_narrowing:
            self.current_products = self.pending_narrowing.product_pool
            if not self.last_filters:
                self.last_filters = self.pending_narrowing.original_filters
            if not self.last_query:
                self.last_query = self.pending_narrowing.original_query
        self.pending_narrowing = None

    # --- Comparison state ---
    def set_comparison_context(self, indices: list[int]) -> None:
        self.last_comparison_indices = indices

    def clear_comparison_context(self) -> None:
        self.last_comparison_indices = None

    def has_comparison_context(self) -> bool:
        return self.last_comparison_indices is not None and len(self.last_comparison_indices) >= 2

    def get_compared_products(self) -> Optional[list[Product]]:
        if not self.has_comparison_context() or not self.current_products:
            return None
        result = []
        for idx in self.last_comparison_indices:
            if 1 <= idx <= len(self.current_products):
                result.append(self.current_products[idx - 1])
        return result if len(result) >= 2 else None
