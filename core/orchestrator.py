"""
Query orchestrator for ST-Bot v2.0.

Coordinates the flow: intent classification -> handler routing -> response building.
8 intents mapped to 8 handlers, no Streamlit coupling.
"""

import re
import time
from typing import Any, List, Optional, Tuple
from dataclasses import dataclass

from app_logging.structured_logging import get_logger, log_conversation_turn
from app_logging.conversation_csv import log_conversation as log_conversation_csv

from core.models import ConversationContext, Intent, IntentType, Product

from handlers.base import HandlerContext, HandlerResult
from handlers.greeting import GreetingHandler, FarewellHandler, OutOfScopeHandler
from handlers.educational import EducationalHandler
from handlers.search import NewSearchHandler
from handlers.sku import SkuHandler
from handlers.clarification import VagueSearchHandler
from handlers.followup import FollowupHandler

_logger = get_logger(__name__)


# =============================================================================
# Input Validation
# =============================================================================

MAX_QUERY_LENGTH = 1000

# Patterns that may indicate prompt injection attempts
BLOCKED_PATTERNS = [
    r'ignore\s+(previous|all|above)\s+instructions?',
    r'system\s*:\s*',
    r'<\|.*?\|>',  # Common prompt injection delimiters
    r'\[INST\]',   # Llama-style instruction delimiters
    r'<<SYS>>',    # Llama system prompt delimiter
]


def sanitize_query(query: str) -> str:
    """
    Sanitize user query before processing.

    Prevents prompt injection and handles malformed input.

    Args:
        query: Raw user input

    Returns:
        Sanitized query string
    """
    if not query:
        return ""

    # Limit length to prevent abuse
    if len(query) > MAX_QUERY_LENGTH:
        query = query[:MAX_QUERY_LENGTH]

    # Strip potential prompt injection patterns
    for pattern in BLOCKED_PATTERNS:
        query = re.sub(pattern, '', query, flags=re.IGNORECASE)

    # Basic cleanup
    query = query.strip()

    return query


# =============================================================================
# Helper Functions
# =============================================================================

def _extract_category_hint(products: list, query: str) -> str:
    """
    Extract category hint from products or query for history matching.

    Used by search history to enable recall functionality.
    e.g., "Show me docks" -> category_hint="dock"
    """
    # Check product categories
    categories = set()
    for p in products:
        cat = p.metadata.get('category', '').lower()
        if cat:
            categories.add(cat)

    # If all products are same category, use that
    if len(categories) == 1:
        return list(categories)[0]

    # Check query for category keywords (single source in category_config)
    from core.category_config import CATEGORY_KEYWORDS
    query_lower = query.lower()
    for cat, keywords in CATEGORY_KEYWORDS.items():
        if any(kw in query_lower for kw in keywords):
            return cat

    # Fallback to first product category or generic "products"
    return list(categories)[0] if categories else "products"


# =============================================================================
# Orchestrator Components
# =============================================================================

@dataclass
class OrchestratorComponents:
    """
    All components needed by the orchestrator.

    Created by the main app and passed to process_query().
    """
    intent_classifier: Any  # LLMIntentClassifier
    filter_extractor: Any   # LLMFilterExtractor
    search_engine: Any      # SearchEngineWrapper (has .search() and .engine)
    query_analyzer: Any     # QueryAnalyzer


# =============================================================================
# Handler Registry
# =============================================================================

# Maps each of the 8 v2.0 intents to its handler
HANDLERS = {
    IntentType.GREETING: GreetingHandler(),
    IntentType.FAREWELL: FarewellHandler(),
    IntentType.OUT_OF_SCOPE: OutOfScopeHandler(),
    IntentType.EDUCATIONAL: EducationalHandler(),
    IntentType.NEW_SEARCH: NewSearchHandler(),
    IntentType.FOLLOWUP: FollowupHandler(),       # absorbs narrowing + clarification responses via meta_info
    IntentType.CLARIFICATION: VagueSearchHandler(),
    IntentType.SPECIFIC_SKU: SkuHandler(),
}


# =============================================================================
# Main Entry Point
# =============================================================================

def process_query(
    query: str,
    context: ConversationContext,
    components: OrchestratorComponents,
    all_products: List[Product],
    debug_mode: bool = False
) -> Tuple[str, str]:
    """
    Process a user query and return a response.

    Main entry point for the orchestrator. Routes queries through:
    1. Input sanitization
    2. Intent classification
    3. Handler dispatch
    4. Side effect application (context updates)

    Args:
        query: User's query text
        context: Conversation context (products shown, pending flows, etc.)
        components: All orchestrator components
        all_products: List of all products for searching
        debug_mode: Whether to include debug output

    Returns:
        Tuple of (response_text, intent_type_value)
    """
    start_time = time.perf_counter()
    debug_lines = []

    # Sanitize user input
    query = sanitize_query(query)
    if not query:
        return "Please enter a message.", "error"

    # Record user message
    context.add_message('user', query)

    # Debug: Show context state
    if debug_mode:
        context_count = len(context.current_products) if context.current_products else 0
        debug_lines.append(f"CONTEXT: {context_count} products in context")
        if context.current_products:
            sku_list = [p.product_number for p in context.current_products[:5]]
            debug_lines.append(f"   SKUs: {', '.join(sku_list)}")
        if context.has_pending_clarification():
            debug_lines.append(f"PENDING CLARIFICATION: {context.pending_clarification.vague_type.value}")
        if context.has_pending_narrowing():
            debug_lines.append(f"PENDING NARROWING: {len(context.pending_narrowing.product_pool)} products in pool")

    # Step 1: Classify intent
    # Track pre-classification state so we can detect classifier escape patterns
    had_clarification_before = context.has_pending_clarification()
    had_narrowing_before = context.has_pending_narrowing()

    intent = components.intent_classifier.classify(query, context, debug_lines if debug_mode else None)
    if debug_mode:
        debug_lines.append(f"INTENT: {intent.type.value} (confidence={intent.confidence:.2f})")

    # Log if classifier cleared pending state (escape/followup detected)
    if had_clarification_before and not context.has_pending_clarification():
        if debug_mode:
            debug_lines.append("CLEARED: clarification (classifier escaped)")

    if had_narrowing_before and not context.has_pending_narrowing():
        if debug_mode:
            debug_lines.append("CLEARED: narrowing (classifier escaped)")

    # Clear stale clarification for NEW_SEARCH to prevent category leak
    if intent.type == IntentType.NEW_SEARCH and context.has_pending_clarification():
        context.clear_pending_clarification()
        if debug_mode:
            debug_lines.append("CLEARED: stale clarification for new search")

    # Clear stale narrowing for NEW_SEARCH
    if intent.type == IntentType.NEW_SEARCH and context.has_pending_narrowing():
        context.clear_pending_narrowing()
        if debug_mode:
            debug_lines.append("CLEARED: stale narrowing for new search")

    # Step 2: Get handler for intent
    handler = HANDLERS.get(intent.type)
    if not handler:
        handler = NewSearchHandler()
        if debug_mode:
            debug_lines.append(f"No handler for {intent.type.value}, using NewSearchHandler")

    # Step 3: Build handler context
    handler_ctx = HandlerContext(
        query=query,
        intent=intent,
        context=context,
        all_products=all_products,
        debug_mode=debug_mode,
        filter_extractor=components.filter_extractor,
        search_engine=components.search_engine,
        query_analyzer=components.query_analyzer,
        debug_lines=debug_lines,
    )

    # Step 4: Execute handler
    try:
        result = handler.handle(handler_ctx)
    except Exception as e:
        _logger.error(f"Handler error for {intent.type.value}: {e}", exc_info=True)
        if debug_mode:
            debug_lines.append(f"ERROR: {type(e).__name__}: {str(e)}")
        result = HandlerResult(
            response="I encountered an issue processing your request. "
                     "Could you try rephrasing your question?"
        )

    # Step 5: Apply side effects
    if result.products_to_set:
        context.set_multi_products(result.products_to_set)
        if debug_mode:
            debug_lines.append(f"SAVED: {len(result.products_to_set)} products to context")

        # Save filters for correction detection in next query
        if result.filters_for_logging:
            context.last_filters = result.filters_for_logging
            context.last_query = query

        # Record to search history (only for NEW_SEARCH)
        if intent.type == IntentType.NEW_SEARCH:
            category_hint = _extract_category_hint(result.products_to_set, query)
            context.add_to_search_history(
                query=query,
                products=result.products_to_set,
                category_hint=category_hint,
                filters=result.filters_for_logging
            )
            if debug_mode:
                debug_lines.append(f"HISTORY: Added '{category_hint}' ({len(result.products_to_set)} products)")

    # Handle clarification state
    if result.save_pending_clarification:
        if debug_mode and context.pending_clarification:
            debug_lines.append(f"SAVED: pending clarification ({context.pending_clarification.vague_type.value})")
    elif result.clear_pending_clarification:
        context.clear_pending_clarification()
        if debug_mode:
            debug_lines.append("CLEARED: pending clarification")

    # Handle narrowing state
    if result.save_pending_narrowing:
        if debug_mode and context.pending_narrowing:
            debug_lines.append(f"SAVED: pending narrowing ({context.pending_narrowing.questions_asked} questions asked)")
    elif result.clear_pending_narrowing:
        context.clear_pending_narrowing()
        if debug_mode:
            debug_lines.append("CLEARED: pending narrowing")

    # Step 6: Build final response
    response = result.response

    if debug_mode and debug_lines:
        debug_header = "**DEBUG OUTPUT:**\n```\n" + "\n".join(debug_lines) + "\n```\n\n---\n\n"
        response = debug_header + response

    # Record bot response (use clean response, not debug-augmented)
    context.add_message('assistant', result.response)

    # Step 7: Structured logging
    response_time_ms = (time.perf_counter() - start_time) * 1000
    products_for_logging = result.products_to_set or context.current_products
    products_shown_count = len(products_for_logging) if products_for_logging else 0
    product_skus = [p.product_number for p in products_for_logging] if products_for_logging else []

    log_conversation_turn(
        session_id=context.session_id or "",
        user_query=query,
        intent_result=intent.type.value,
        intent_confidence=intent.confidence,
        products_found=result.products_found,
        products_shown=products_shown_count,
        product_skus=product_skus,
        filters=result.filters_for_logging,
        response_time_ms=response_time_ms,
    )

    log_conversation_csv(
        session_id=context.session_id or "",
        user_query=query,
        bot_response=result.response,
        intent=intent.type.value,
        confidence=intent.confidence,
        filters=result.filters_for_logging,
        products_found=result.products_found,
        products_shown=products_shown_count,
        product_skus=product_skus,
        response_time_ms=response_time_ms,
    )

    return response, intent.type.value
