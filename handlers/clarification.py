"""
Clarification handlers for ST-Bot.

Handles vague queries by asking clarifying questions and processing responses.
"""

import re
from handlers.base import BaseHandler, HandlerContext, HandlerResult
from core.models import (
    PendingClarification,
    ClarificationMissing,
    SearchFilters,
)
from core.vagueness import LLMVaguenessDetector
from core.clarification import (
    ClarificationQuestionBuilder,
    ClarificationResponseParser,
)


# Patterns for things we can't know (limitations)
LIMITATION_PATTERNS = {
    'purchase_history': [
        r'\b(?:same|last|previous)\s+(?:one|cable|product|order)\b',
        r'\bbought\s+(?:last|before|previously)\b',
        r'\b(?:my|the)\s+(?:last|previous)\s+(?:order|purchase)\b',
        r'\breorder\b',
    ],
    'friend_recommendation': [
        r'\bfriend\s+(?:recommended|suggested|told)\b',
        r'\bsomeone\s+(?:recommended|suggested|told)\b',
        r'\bheard\s+(?:about|of)\b',
        r'\bforgot\s+(?:what|the\s+name)\b',
    ],
    'best_seller': [
        r'\bbest\s*sell(?:er|ing)\b',
        r'\bmost\s+popular\b',
        r'\btop\s+(?:rated|selling)\b',
        r'\beveryone\s+(?:uses|buys|gets)\b',
    ],
    'unknown_setup': [
        r'\bmy\s+setup\b',
        r'\bwhatever\s+works\b',
        r'\bwith\s+my\s+(?:current\s+)?(?:setup|system|config)\b',
    ],
}

# Limitation acknowledgment responses
LIMITATION_RESPONSES = {
    'purchase_history': (
        "I don't have access to your purchase history, but I can help you find what you need. "
        "What devices are you trying to connect?"
    ),
    'friend_recommendation': (
        "I don't know what your friend recommended, but I can help you find the right product. "
        "What are you trying to accomplish with it?"
    ),
    'best_seller': (
        "I don't have access to sales data, but I can recommend products based on your needs. "
        "What devices are you trying to connect?"
    ),
    'unknown_setup': (
        "I don't know your current setup, but I can help if you tell me more. "
        "What devices do you have and what are you trying to do?"
    ),
}

# Pricing patterns - should trigger OUT_OF_SCOPE
PRICING_PATTERNS = [
    r'\bcheap(?:est|er)?\b',
    r'\baffordable\b',
    r'\bbudget\b',
    r'\b(?:under|below|less\s+than)\s*\$?\d+',
    r'\$\d+',
    r'\bpric(?:e|es|ing)\b',
    r'\bcost\b',
    r'\bexpensive\b',
    r'\binexpensive\b',
]

# Warranty patterns - redirect to StarTech.com support
WARRANTY_PATTERNS = [
    r'\bwarrant(?:y|ies)\b',
    r'\breturn\s+polic(?:y|ies)\b',
    r'\brma\b',
    r'\bguarantee\b',
]


class VagueSearchHandler(BaseHandler):
    """
    Handles vague product searches that need clarification.

    Instead of dumping random products for vague queries like "I need a cable",
    this handler asks clarifying questions to understand what the user actually needs.

    This is called from NewSearchHandler when a vague query is detected.
    """

    def __init__(self):
        """Initialize vague search handler."""
        self.detector = LLMVaguenessDetector()
        self.question_builder = ClarificationQuestionBuilder()
        self._debug_lines = None  # Set by caller for debug output

    def handle(self, ctx: HandlerContext) -> HandlerResult:
        """
        Start a clarification flow for a vague query.

        Args:
            ctx: Handler context

        Returns:
            HandlerResult with clarifying question
        """
        # Extract filters to check what we already know
        llm_result = ctx.filter_extractor.extract(ctx.query)
        filters = llm_result.filters if llm_result else SearchFilters()

        # Detect vague query type
        vague_type = self.detector.detect(ctx.query, filters)

        if not vague_type:
            # Not actually vague - shouldn't happen but fall back to normal search
            ctx.add_debug("⚠️ VagueSearchHandler called but query not vague")
            from handlers.search import NewSearchHandler
            return NewSearchHandler().handle(ctx)

        ctx.add_debug(f"🤔 VAGUE QUERY: type={vague_type.value}, query='{ctx.query}'")

        # Determine what information is missing
        missing_info = self._determine_missing_info(filters)

        # Create pending clarification state
        # Preserve original category so we can fall back to it after clarification
        clarification = PendingClarification(
            vague_type=vague_type,
            original_query=ctx.query,
            missing_info=missing_info,
            original_category=filters.product_category,
        )

        # Set pending clarification in context
        ctx.context.set_pending_clarification(clarification)

        # Build the initial response with clarifying question
        response = self.question_builder.build_initial_response(
            clarification,
            conversation_history=ctx.context.get_conversation_history(limit=6)
        )

        ctx.add_debug(f"🤔 ASKING: missing={[m.value for m in missing_info]}")

        return HandlerResult(
            response=response,
            save_pending_clarification=True,
        )

    def _determine_missing_info(self, filters: SearchFilters) -> list[ClarificationMissing]:
        """
        Determine what information is missing from the query.

        Args:
            filters: Extracted search filters

        Returns:
            List of missing information types
        """
        missing = []

        # If we don't know source connector, we need use case
        if not filters.connector_from:
            missing.append(ClarificationMissing.USE_CASE)

        # If we don't know destination either
        if not filters.connector_to:
            missing.append(ClarificationMissing.CONNECTOR_TO)

        # If no missing info detected, default to use case
        if not missing:
            missing.append(ClarificationMissing.USE_CASE)

        return missing

    def is_vague_query(self, query: str, filters: SearchFilters, debug_lines: list = None) -> bool:
        """
        Check if a query is vague and needs clarification.

        This is used by NewSearchHandler to decide whether to route
        to this handler.

        Performance optimization: Skip LLM vagueness check if filters are clear.
        Clear filters = category + connector(s), which means user specified enough.

        Args:
            query: User's query
            filters: Extracted filters
            debug_lines: Optional list for debug output

        Returns:
            True if query is vague
        """
        # Store debug_lines for later use in handle()
        self._debug_lines = debug_lines

        # Fast path: Skip vagueness check if filters are already clear
        # This avoids an LLM call for obvious queries like "show me HDMI cables"
        has_clear_filters = self._has_clear_filters(filters)
        if has_clear_filters:
            if debug_lines is not None:
                debug_lines.append("⚡ SKIP VAGUE CHECK: Clear filters found")
            return False

        # LLMVaguenessDetector accepts debug_lines, VagueQueryDetector doesn't
        if hasattr(self.detector, 'detect') and 'debug_lines' in self.detector.detect.__code__.co_varnames:
            return self.detector.detect(query, filters, debug_lines) is not None
        return self.detector.detect(query, filters) is not None

    # Categories specific enough that the category alone is a meaningful search.
    # These don't need connector/length/port_count to skip the vagueness check.
    SELF_SUFFICIENT_CATEGORIES = {
        'video_splitter', 'video_switch', 'kvm', 'kvm_switch', 'kvm_extender',
        'rack', 'privacy_screen', 'display_mount', 'mount', 'enclosure',
        'laptop_lock', 'power', 'dock', 'hub', 'ethernet_switch',
    }

    def _has_clear_filters(self, filters: SearchFilters) -> bool:
        """
        Check if filters are clear enough to skip vagueness detection.

        Clear filters mean user has specified enough info for a meaningful search:
        - Has a product category AND at least one connector
        - OR has both connector_from and connector_to
        - OR has category + length (e.g., "10ft ethernet cable")

        Args:
            filters: Extracted search filters

        Returns:
            True if filters are clear enough to skip vagueness check
        """
        if not filters:
            return False

        has_category = bool(filters.product_category)
        has_connector_from = bool(filters.connector_from)
        has_connector_to = bool(filters.connector_to)
        has_length = bool(filters.length)

        # Pattern 0: Category is specific enough on its own
        if has_category and filters.product_category in self.SELF_SUFFICIENT_CATEGORIES:
            return True

        # Pattern 1: Category + connector (e.g., "HDMI cables", "USB-C adapter")
        if has_category and (has_connector_from or has_connector_to):
            return True

        # Pattern 2: Both connectors (e.g., "USB-C to HDMI")
        if has_connector_from and has_connector_to:
            return True

        # Pattern 3: Category + length (e.g., "10ft ethernet cable")
        if has_category and has_length:
            return True

        # Pattern 4: Category + port count (e.g., "7 port hub", "dock with 8 ports")
        if has_category and filters.port_count:
            return True

        # Pattern 5: Category + monitor count (e.g., "dock that supports 3 monitors")
        if has_category and filters.min_monitors:
            return True

        return False


class ClarificationResponseHandler(BaseHandler):
    """
    Handles user responses to clarification questions.

    Parses the response to extract useful information, then either:
    - Asks another question if more info is needed
    - Performs the search with collected information
    """

    def __init__(self):
        self.response_parser = ClarificationResponseParser()
        self.question_builder = ClarificationQuestionBuilder()

    def handle(self, ctx: HandlerContext) -> HandlerResult:
        """
        Process a response to a clarification question.

        Args:
            ctx: Handler context

        Returns:
            HandlerResult with either another question or search results
        """
        clarification = ctx.context.pending_clarification
        query_lower = ctx.query.lower()

        if not clarification:
            # No pending clarification - shouldn't happen
            ctx.add_debug("⚠️ ClarificationResponseHandler called without pending clarification")
            return HandlerResult(
                response="I'm not sure what you're referring to. What products are you looking for?"
            )

        # Check for pricing patterns FIRST - should trigger pricing guardrail
        if self._is_pricing_request(query_lower):
            ctx.add_debug("💰 PRICING REQUEST DETECTED - triggering guardrail")
            ctx.context.clear_pending_clarification()
            return HandlerResult(
                response=(
                    "I can't filter by price, but you can check pricing at StarTech.com. "
                    "Would you like me to help find products by features instead? "
                    "What devices are you trying to connect?"
                ),
                clear_pending_clarification=True,
            )

        # Check for warranty patterns - redirect to StarTech.com support
        if self._is_warranty_request(query_lower):
            ctx.add_debug("🛡️ WARRANTY REQUEST DETECTED - triggering guardrail")
            ctx.context.clear_pending_clarification()
            return HandlerResult(
                response=(
                    "For warranty details, return policies, and RMA requests, "
                    "please visit www.startech.com/warranty or contact StarTech.com support directly. "
                    "Is there anything else I can help you find?"
                ),
                clear_pending_clarification=True,
            )

        ctx.add_debug(f"🤔 PARSING RESPONSE: '{ctx.query}'")
        ctx.add_debug(f"🤔 BEFORE: collected={clarification.collected_info}")

        # Parse the user's response
        clarification = self.response_parser.parse_response(ctx.query, clarification)

        ctx.add_debug(f"🤔 AFTER: collected={clarification.collected_info}")

        # Check if we have enough info to search
        if clarification.has_enough_info():
            ctx.add_debug("🤔 ENOUGH INFO - proceeding with search")
            return self._perform_search(ctx, clarification)

        # Only check for limitation patterns if parsing didn't extract anything useful
        # This ensures "adapter everyone uses for MacBooks" extracts MacBook first
        limitation = self._detect_limitation(query_lower)
        if limitation:
            ctx.add_debug(f"⚠️ LIMITATION DETECTED: {limitation}")
            response = LIMITATION_RESPONSES.get(limitation)
            return HandlerResult(
                response=response,
                save_pending_clarification=True,
            )

        # Need more info - ask another question (with variation)
        ctx.add_debug("🤔 NEED MORE INFO - asking follow-up")
        question = self._build_varied_question(clarification, ctx)

        return HandlerResult(
            response=question,
            save_pending_clarification=True,
        )

    def _is_pricing_request(self, query: str) -> bool:
        """Check if query contains pricing-related terms."""
        for pattern in PRICING_PATTERNS:
            if re.search(pattern, query):
                return True
        return False

    def _is_warranty_request(self, query: str) -> bool:
        """Check if query contains warranty-related terms."""
        for pattern in WARRANTY_PATTERNS:
            if re.search(pattern, query):
                return True
        return False

    def _detect_limitation(self, query: str) -> str | None:
        """Detect if query asks about something we can't know."""
        for limitation_type, patterns in LIMITATION_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, query):
                    return limitation_type
        return None

    def _build_varied_question(self, clarification: PendingClarification,
                               ctx: HandlerContext) -> str:
        """Build a varied question, using LLM with hard-coded fallback."""
        # LLM handles escalation naturally via questions_asked count
        question = self.question_builder.build_question(
            clarification,
            conversation_history=ctx.context.get_conversation_history(limit=6)
        )
        return question

    def _perform_search(self, ctx: HandlerContext, clarification: PendingClarification) -> HandlerResult:
        """
        Perform search with collected clarification info.

        Args:
            ctx: Handler context
            clarification: Completed clarification state

        Returns:
            HandlerResult with search results
        """
        # Convert collected info to search filters
        filter_params = self._get_search_filters(clarification)
        ctx.add_debug(f"🔍 CLARIFICATION FILTERS: {filter_params}")

        # Determine product category with smart fallback
        product_category = filter_params.get('product_category')
        if not product_category:
            # Use original category from vague query detection, or default to Cables
            product_category = clarification.original_category or 'Cables'

        # Build SearchFilters object
        filters = SearchFilters(
            connector_from=filter_params.get('connector_from'),
            connector_to=filter_params.get('connector_to'),
            product_category=product_category,
        )

        # Perform search
        results = ctx.search_engine.search(filters)
        ctx.add_debug(f"🔍 SEARCH: Found {len(results.products)} products")

        if not results.products:
            # No results - be helpful
            ctx.context.clear_pending_clarification()
            return HandlerResult(
                response=self._build_no_results_response(clarification),
                clear_pending_clarification=True,
            )

        # Top 3 products (already ranked by scored search)
        products_list = results.products[:3]

        # Build response
        from llm.llm_response_generator import generate_response, ResponseType
        response = generate_response(
            products=products_list,
            query=clarification.original_query,
            response_type=ResponseType.SEARCH_RESULTS,
            context={
                'original_filters': filters,
            }
        )

        # Add context about what we understood
        intro = self._build_search_intro(clarification)
        if intro:
            response = f"{intro}\n\n{response}"

        top_products = products_list

        # Clear clarification state
        ctx.context.clear_pending_clarification()

        return HandlerResult(
            response=response,
            products_to_set=top_products,
            clear_pending_clarification=True,
            filters_for_logging={
                'connector_from': filters.connector_from,
                'connector_to': filters.connector_to,
                'category': filters.product_category,
                'clarification_collected': clarification.collected_info,
            },
            products_found=len(results.products)
        )

    def _get_search_filters(self, clarification: PendingClarification) -> dict:
        """
        Convert collected clarification info into search filter parameters.

        Args:
            clarification: Clarification state with collected info

        Returns:
            dict: Filter parameters ready for SearchFilters
        """
        filters = {}

        if 'connector_from' in clarification.collected_info:
            filters['connector_from'] = clarification.collected_info['connector_from']

        if 'connector_to' in clarification.collected_info:
            filters['connector_to'] = clarification.collected_info['connector_to']

        # PRIORITY 1: Check if product_category was directly extracted
        # (e.g., user said "dock" explicitly in their clarification response)
        if 'product_category' in clarification.collected_info:
            filters['product_category'] = clarification.collected_info['product_category']
            return filters  # Don't override with use_case mapping

        # PRIORITY 2: Determine product category from use_case or fall back to original
        category_set = False
        if 'use_case' in clarification.collected_info:
            use_case = clarification.collected_info['use_case']
            if use_case == 'video_output':
                if clarification.original_category not in ('dock', 'multiport_adapter'):
                    filters['product_category'] = 'Cables'
                    category_set = True
            elif use_case == 'ports':
                filters['product_category'] = 'USB Hubs'
                category_set = True
            elif use_case == 'dock':
                filters['product_category'] = 'Docks'
                category_set = True
            elif use_case == 'charging':
                if clarification.original_category in ('dock', 'multiport_adapter'):
                    filters['product_category'] = clarification.original_category
                else:
                    if self._is_dock_context(clarification):
                        filters['product_category'] = 'Docks'
                    else:
                        filters['product_category'] = 'Cables'
                category_set = True

        # Fall back to original category if no use_case mapping set category
        if not category_set and clarification.original_category:
            filters['product_category'] = clarification.original_category

        return filters

    def _is_dock_context(self, clarification: PendingClarification) -> bool:
        """
        Check if collected info suggests a dock context.

        Returns True if user mentioned multiple monitors, keyboard/mouse,
        or other indicators of needing a docking station.
        """
        text_to_check = clarification.original_query.lower()
        for value in clarification.collected_info.values():
            if isinstance(value, str):
                text_to_check += ' ' + value.lower()

        dock_patterns = [
            r'\b(?:two|2|dual|triple|three|3|multiple)\s+(?:external\s+)?monitors?',
            r'\bmonitors?\s+(?:and|while|with)\s+(?:charg|keyboard|mouse)',
            r'\bkeyboard\s+(?:and|with)\s+mouse',
            r'\bdock(?:ing)?\s+station',
            r'\bwork\s+(?:from\s+home|station|setup|desk)',
            r'\bdesktop\s+(?:setup|experience|mode)',
        ]

        for pattern in dock_patterns:
            if re.search(pattern, text_to_check):
                return True

        return False

    def _build_search_intro(self, clarification: PendingClarification) -> str:
        """Build an intro message summarizing what we understood."""
        collected = clarification.collected_info

        if 'connector_from' in collected and 'connector_to' in collected:
            return f"Based on your {collected['connector_from']} device connecting to {collected.get('connector_to', 'display')}, here's what I found:"
        elif 'connector_from' in collected:
            return f"Here are {collected['connector_from']} cables that should work:"
        elif 'use_case' in collected:
            use_case = collected['use_case']
            if use_case == 'video_output':
                return "Here are video cables for connecting to a display:"
            elif use_case == 'ports':
                return "Here are options for expanding your ports:"

        return ""

    def _build_no_results_response(self, clarification: PendingClarification) -> str:
        """Build a helpful response when no results are found."""
        collected = clarification.collected_info

        if 'connector_from' in collected:
            connector = collected['connector_from']
            return (
                f"I couldn't find cables matching those exact specs. "
                f"Could you tell me more about what you're trying to connect your {connector} device to?"
            )

        return (
            "I couldn't find products matching that description. "
            "Could you give me more details about what devices you're trying to connect?"
        )
