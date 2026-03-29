"""
LLM-Based Intent Classifier for ST-Bot v2.0.

Replaces regex patterns with natural language understanding using OpenAI GPT-4o-mini.
On LLM failure, returns CLARIFICATION (no regex fallback in v2).

Key benefits over regex:
- Handles edge cases naturally (typos, unusual phrasing)
- No "death by a thousand rules" maintenance burden
- Better context awareness for followup detection
- Graceful handling of ambiguous queries

v2.0 changes:
- 8 intents (not 9): GREETING, FAREWELL, OUT_OF_SCOPE, NEW_SEARCH, FOLLOWUP,
  EDUCATIONAL, CLARIFICATION, SPECIFIC_SKU
- AMBIGUOUS renamed to CLARIFICATION
- PRODUCT_EXISTENCE absorbed into NEW_SEARCH
- NARROWING_RESPONSE and CLARIFICATION_RESPONSE absorbed into FOLLOWUP
- SPECIFIC_SKU added for bare SKU queries
- No regex fallback (core.intent module removed)
"""

import os
import json
import re
import logging
from typing import Optional, List
from dataclasses import dataclass

from core.models import Intent, IntentType, ConversationContext
from core.api_retry import RetryHandler, RetryConfig
from core.openai_client import get_openai_client
from config.patterns import GREETING_PATTERNS, FAREWELL_PATTERNS, has_pattern

# Module-level logger
_logger = logging.getLogger(__name__)


# System prompt for intent classification
INTENT_CLASSIFICATION_PROMPT = '''You are an intent classifier for StarTech.com's product assistant chatbot.

Classify the user's query into exactly one of these intents:

## Intent Types

1. **GREETING** - Simple greetings only
   - Examples: "hi", "hello", "hey there", "good morning"
   - Also includes generic help requests with NO product specifics:
     "I need some help", "can you help me?", "I need assistance"
   - Key test: Does the message mention ANY product, category, use case, or setup? If not → GREETING
   - "I need help" → GREETING (no product context)
   - "I need help finding a cable" → NEW_SEARCH (has product context)
   - NOT: "hi, show me cables" (that's NEW_SEARCH)

2. **FAREWELL** - Goodbye messages, thanks for help
   - Examples: "bye", "thanks", "goodbye", "that's all"
   - NOT: "thanks, but show me more options" (that's FOLLOWUP)

3. **OUT_OF_SCOPE** - Returns, competitor focus, off-topic requests
   - customer_service: "how do I return this?", "shipping info", "order status"
   - pricing: "how much does this cost?", "what's the price?"
   - competitor: "Amazon has better prices", "what about Belkin cables?"
   - fictional: "quantum HDMI cable", made-up products
   - off_topic: weather, jokes, essays, politics, unrelated questions
   - setup_help: asking for help configuring, installing, or troubleshooting a product they already own
     - "Can you help me set up my router?" → OUT_OF_SCOPE (configuration help, not product search)
     - "How do I configure my monitor settings?" → OUT_OF_SCOPE
   - not_our_products: asking about product types StarTech doesn't sell (laptops, computers, monitors, printers, phones, TVs, software)
     - "Do you sell laptops?" → OUT_OF_SCOPE (StarTech sells connectivity accessories, not computers/devices)
     - "I need a new monitor" → OUT_OF_SCOPE (StarTech sells mounts and cables for monitors, not monitors themselves)
     - vs. "I need to set up a dual monitor workstation" → NEW_SEARCH (needs products)
   - **"from a different brand" / "from another manufacturer"** → OUT_OF_SCOPE (explicitly not StarTech)
   - **CRITICAL**: If products are currently shown in context, assume ambiguous or vague questions are about those products (FOLLOWUP), NOT off_topic. Only classify as OUT_OF_SCOPE if the question is clearly unrelated to ANY product (e.g., weather, politics, jokes).
   - Set meta_info.out_of_scope_type to one of the above values
   - IMPORTANT: Price MENTIONS in product questions are NOT out of scope!
     - "Which is cheapest?" with products shown → FOLLOWUP (not out of scope)
     - "I need a budget-friendly HDMI cable" → NEW_SEARCH (not out of scope)
     - "Should I just pick the cheapest?" → FOLLOWUP (asking for advice)

4. **NEW_SEARCH** also covers "Do you have X?" availability questions
   - Examples: "do you have Cat7?", "do you sell fiber cables?", "any USB4 docks?"
   - These are product searches — treat them as NEW_SEARCH
   - Extract the product type in meta_info.query (e.g., "Cat7", "fiber cables")
   - **WITH products shown, SAME category**: "do you have any 10ft or longer?" → FOLLOWUP (filtering shown products)
   - **WITH products shown, DIFFERENT category**: "do you have Cat6a cables?" when docks are shown → NEW_SEARCH
   - **WITHOUT products shown**: "do you have Cat7?" → NEW_SEARCH

5. **EDUCATIONAL** - Pure technical questions seeking knowledge, WITHOUT purchase intent
   - **Category/technology comparisons**: "what's the difference between Cat6 and Cat6a?" (comparing TYPES, not specific products)
   - Explanations: "what is Thunderbolt?", "how does HDMI ARC work?"
   - NOT: "which Cat6 cable should I buy?" (that's NEW_SEARCH)
   - NOT: Technical questions embedded in a purchase inquiry (that's NEW_SEARCH)
   - **NOT**: Questions about specific products shown in context (that's FOLLOWUP)
   - **CRITICAL**: "what's the difference between #2 and #3?" → FOLLOWUP (comparing products shown)
   - **CRITICAL**: Comparing specific SKUs when those SKUs are in context → FOLLOWUP
   - **Key test**: Is user asking about general technology/categories OR about specific products shown?

6. **NEW_SEARCH** - Looking for products OR describing what you need
   - Product requests: "I need HDMI cables", "show me USB-C docks"
   - **"Connect X to Y" patterns**: "connect my laptop to my TV" → NEW_SEARCH
   - **Setup descriptions**: "I have a MacBook Pro and need two monitors" → NEW_SEARCH
     - This means the user is describing what PRODUCTS they need — not asking for help configuring/setting up a device they already own
   - Use cases: "something for 4K gaming", "working from home setup"
   - Even with typos: "HDMI calbe" should be NEW_SEARCH
   - **Technical questions WITH purchase context**: "I need to run cables 250 feet, do I need shielded?" → NEW_SEARCH
   - **"What do you recommend?" with requirements** = NEW_SEARCH (user describing their needs)
   - If user describes their situation AND asks for recommendations, it's NEW_SEARCH
   - **BUT**: "Which would you recommend?" with products already shown = FOLLOWUP (not NEW_SEARCH)
   - Same for "help me decide", "help me choose", "which works best" — if products are shown, it's FOLLOWUP

7. **FOLLOWUP** - Questions/requests about THE SPECIFIC products already shown
   - Must have products currently displayed in context
   - **Product comparisons**: "what's the difference between #2 and #3?", "compare these two", "how do they differ?"
   - **SKU comparisons**: "what's the difference between SKU1 and SKU2?" (when those SKUs are in context)
   - Questions about shown products: "does it support 4K?", "what about the second one?"
   - Comparisons: "which is better?", "compare these", "which would you recommend?"
   - Modifications: "shorter one?", "different color?"
   - **Product specs**: "what's the warranty?", "what are the dimensions?", "how heavy is it?" → FOLLOWUP when products shown
   - **Filtering requests**: "do you have any 10ft or longer?", "any in black?" → FOLLOWUP (filtering current results)
   - **Corrections**: "sorry, I actually need DVI not VGA", "I meant USB-A not USB-C", "switch to DisplayPort" → FOLLOWUP (user is changing a spec on current products, not starting fresh)
     - Key signals: "actually", "sorry", "I meant", "not X but Y", "instead of", "switch to"
     - These change a connector/feature/spec, NOT the product category — still FOLLOWUP
     - **BUT**: "actually I need cables not docks" → NEW_SEARCH (changing the category itself)
   - **Recommendations**: "which would you recommend?", "which one should I get?", "help me decide", "help me choose", "which would work best?" → FOLLOWUP when products shown
   - **NOT FOLLOWUP**: Describing a new device or setup (even if products are shown)
   - Example: "I have a MacBook and need monitors" → NEW_SEARCH (new need, not about shown products)
   - **CRITICAL - Different product category = NOT FOLLOWUP**: If products shown are one category (e.g., docks) and user asks about a completely different category (e.g., cables, mounts, adapters), that is NEW_SEARCH — even if the query sounds like "do you have any [X]?". The filtering pattern ONLY applies when refining the SAME type of product.
     - Docks shown + "Do you have Cat6a ethernet cables?" → NEW_SEARCH (cables ≠ docks)
     - HDMI cables shown + "do you have any 10ft or longer?" → FOLLOWUP (still about cables)
   - If no products in context, this should be CLARIFICATION or NEW_SEARCH

8. **SPECIFIC_SKU** - User provides a specific product SKU/part number for lookup
   - Examples: "TB3CDK2DH", "HDMM2M", "what about HDMM10?", "tell me about CDPVGDVHDBP"
   - Some SKUs are all-letter codes that look like nonsense words (e.g., CDPVGDVHDBP) — these are still SKUs
   - Single SKU mention for direct product lookup
   - NOT: comparing multiple SKUs (that's FOLLOWUP if products shown)
   - NOT: common English words like "adapter", "network", "connector" — those are product descriptions, not SKUs

## Context Information

{context_info}

## Response Format

Return ONLY valid JSON (no markdown, no explanation):
{{
    "intent": "INTENT_TYPE",
    "confidence": 0.0-1.0,
    "reasoning": "Brief explanation",
    "sku": "SKU if detected, null otherwise",
    "meta_info": {{
        "query": "extracted search query if NEW_SEARCH",
        "topic": "topic if EDUCATIONAL",
        "out_of_scope_type": "one of: customer_service, pricing, competitor, fictional, off_topic, setup_help (if OUT_OF_SCOPE)"
    }}
}}

## Important Rules

1. SKU Detection: Product numbers are typically 5-15 chars. Most contain letters + digits (e.g., TB3CDK2DH, HDMM2M), but some are all-letter codes (e.g., CDPVGDVHDBP). An unfamiliar all-caps word that doesn't read like normal English may be a SKU.
2. Typo Tolerance: "calbe" = "cable", "HDIM" = "HDMI" - still NEW_SEARCH
3. **CRITICAL - Context Matters**:
   - "does it support 4K?" → FOLLOWUP only if products shown
   - "do you have any 10ft?" → FOLLOWUP if products shown (filtering), NEW_SEARCH if not
   - "which would you recommend?" → FOLLOWUP if products shown, NEW_SEARCH if not
4. Greeting + Search: "hi, show me cables" should be NEW_SEARCH (search takes priority)
5. When unsure: NEW_SEARCH is safer than CLARIFICATION for product-related queries
6. **Setup = NEW_SEARCH**: "I have [device] and need [X]" or "connect my [X] to [Y]" describes what user NEEDS, not a question about products shown
7. **Purchase + Technical Questions = NEW_SEARCH**: When user describes requirements AND asks technical questions (like "do I need shielded?", "what's the max length?"), classify as NEW_SEARCH not EDUCATIONAL. The technical questions are part of the buying decision, not pure curiosity.
8. **Generic Help**: "I need help/assistance" WITHOUT product specifics = GREETING. Only NEW_SEARCH if they mention what they need help WITH.'''


@dataclass
class LLMClassificationResult:
    """Result from LLM classification."""
    intent: IntentType
    confidence: float
    reasoning: str
    sku: Optional[str] = None
    meta_info: Optional[dict] = None


class LLMIntentClassifier:
    """
    LLM-based intent classifier using OpenAI GPT-4o-mini.

    Replaces regex patterns with natural language understanding.
    On LLM failure, returns CLARIFICATION (no regex fallback in v2).

    Usage:
        classifier = LLMIntentClassifier()
        intent = classifier.classify("I need HDMI cables", context)
    """

    def __init__(self, model: str = None, temperature: float = 0.1, valid_skus: set = None):
        """
        Initialize LLM intent classifier.

        Args:
            model: OpenAI model (default: gpt-4o-mini)
            temperature: Model temperature (lower = more deterministic)
            valid_skus: Set of uppercase product SKUs for validating all-letter candidates
        """
        self.model = model or os.environ.get('OPENAI_CHAT_MODEL', 'gpt-5.4-nano')
        self.temperature = temperature
        self._valid_skus = valid_skus

        self.retry_config = RetryConfig(
            max_attempts=2,
            base_delay=0.5,
            max_delay=5.0
        )

    def _fast_path_check(self, prompt: str, context: ConversationContext) -> Optional[Intent]:
        """
        Fast path for truly unambiguous patterns - no LLM needed.

        Handles ONLY:
        - SKU detection (exact product lookup)
        - Pure greetings (<=4 words, no product keywords)
        - Pure farewells (<=4 words)
        - Gibberish (keyboard mashing, no recognizable words)

        Everything else returns None -> goes to LLM.

        Args:
            prompt: User's message
            context: Conversation context

        Returns:
            Intent if unambiguous, None otherwise (-> LLM)
        """
        prompt_lower = prompt.lower().strip()
        word_count = len(prompt.split())

        # 1. SKU detection (structural pattern, not semantic)
        sku = self._extract_sku(prompt)
        if sku:
            # If SKU is already in context, let LLM classify (likely followup)
            if context.current_products:
                for p in context.current_products:
                    if p.product_number.upper() == sku.upper():
                        return None  # Let LLM handle as followup
            return Intent(
                type=IntentType.SPECIFIC_SKU,
                confidence=0.95,
                reasoning=f"Fast path: SKU lookup for {sku}",
                sku=sku
            )

        # 2. Pure greetings (short, no product keywords)
        if self._is_pure_greeting(prompt_lower, word_count):
            return Intent(
                type=IntentType.GREETING,
                confidence=1.0,
                reasoning="Fast path: Pure greeting"
            )

        # 3. Pure farewells
        if self._is_pure_farewell(prompt_lower, word_count):
            return Intent(
                type=IntentType.FAREWELL,
                confidence=1.0,
                reasoning="Fast path: Pure farewell"
            )

        # Everything else -> LLM
        return None

    def _extract_sku(self, text: str) -> Optional[str]:
        """
        Extract a product SKU ONLY for direct SKU lookup.

        Returns None (lets LLM handle) when:
        - Multiple SKUs mentioned (comparison context)
        - Query contains comparison words ("between", "difference", "compare", "vs")
        - Query is asking a question about products

        StarTech SKUs: 5-20 chars, alphanumeric with hyphens, must have letter + digit.
        Examples: HDMM10, CDP2HD1MBNL, TB3CDK2DH
        """
        text = text.strip()
        text_lower = text.lower()

        # 1. Skip SKU fast path for comparison queries - let LLM handle
        comparison_signals = [
            r'\bdifference\s+between\b',
            r'\bcompare\b',
            r'\bvs\.?\b',
            r'\bversus\b',
            r'\bbetween\s+\S+\s+and\b',
            r"\bwhat'?s\s+the\s+difference\b",
        ]
        if any(re.search(p, text_lower) for p in comparison_signals):
            return None  # Let LLM handle comparison

        # 2. Skip SKU extraction if query contains device context
        # These indicate the user is describing a use case, not looking up a SKU
        device_context_patterns = [
            r'\b(?:monitor|monitors|screen|display|laptop|computer|tv|television)\b',
            r'\b(?:connect(?:ing)?|for\s+my|to\s+my|with\s+my)\b',
            r'\b(?:daisy[- ]?chain(?:ing)?|chain(?:ing)?)\b',
        ]
        if any(re.search(p, text_lower) for p in device_context_patterns):
            return None  # Let LLM handle contextual query

        # Patterns that look like SKUs but aren't
        NOT_SKU_PATTERNS = [
            # Numeric patterns
            r'^\d+-?ports?$',   # "4-port"
            r'^\d+ft$',         # "6ft"
            r'^\d+-?foot$',     # "30-foot"
            r'^\d+-?feet$',     # "10-feet"
            r'^\d+m$',          # "2m"
            r'^\d+-?inch$',     # "27-inch"
            r'^\d+hz$',         # "60hz"
            r'^\d+p$',          # "1080p", "1440p", "720p"
            r'^\d+k$',          # "4k"
            r'^\d+g$',          # "5g"
            r'^\d+gb$',         # "8gb"
            r'^\d+tb$',         # "1tb"
            r'^\d+w$',          # "65w"
            r'^\d+mbps$',       # "100mbps"
            r'^\d+gbps$',       # "10gbps"
            # Measurement ranges and values (NOT product SKUs)
            r'^\d+[\-–]\d+\s*(?:ft|feet|m|mm|in|cm)$',  # "5-6ft", "10-15m"
            r'^\d+(?:\.\d+)?\s*(?:ft|feet|foot|m|meters?|mm|in|inch|inches|cm)$',  # "6ft", "0.4m"
            # Cable category names (NOT product SKUs)
            r'^cat\d+[a-z]?$',  # "cat5", "cat5e", "cat6", "cat6a", "cat7"
            # Monitor/device model numbers (not StarTech SKUs)
            r'^[a-z]\d{3,5}[a-z]{0,2}$',     # P2725HE, U2723QE, S2421HS
            r'^[a-z]{2}\d{2,4}[a-z]{0,3}$',  # LG27GP850, VG27AQ
        ]

        def is_not_sku(word):
            word_lower = word.lower()
            return any(re.match(p, word_lower) for p in NOT_SKU_PATTERNS)

        # 3. Count SKU-like patterns - if multiple, it's a comparison
        sku_candidates = []
        for word in text.split():
            word_clean = word.strip('.,!?').upper()
            if re.match(r'^[A-Z0-9\-]{5,20}$', word_clean):
                if re.search(r'[A-Z]', word_clean) and re.search(r'\d', word_clean):
                    # Catalog match overrides NOT_SKU_PATTERNS (avoids false positives
                    # where real SKUs like RK1233BKM match monitor model patterns)
                    if self._valid_skus and word_clean in self._valid_skus:
                        sku_candidates.append(word_clean)
                    elif not is_not_sku(word_clean):
                        sku_candidates.append(word_clean)

        # Multiple SKUs = comparison context, let LLM handle
        if len(sku_candidates) > 1:
            return None

        # Single SKU in longer text = direct lookup
        if len(sku_candidates) == 1:
            return sku_candidates[0]

        # Single SKU-like token (entire query is just the SKU)
        if re.match(r'^[A-Za-z0-9\-]{4,25}$', text) and re.search(r'\d', text):
            if self._valid_skus and text.upper() in self._valid_skus:
                return text.upper()
            if not is_not_sku(text):
                return text.upper()

        # All-letter token validated against catalog
        text_upper = text.strip().upper()
        if self._valid_skus and re.match(r'^[A-Z]{4,25}$', text_upper) and text_upper in self._valid_skus:
            return text_upper

        return None

    def _is_pure_greeting(self, text: str, word_count: int) -> bool:
        """Check if text is a pure greeting (short, no product keywords)."""
        if word_count > 4:
            return False

        # Not a greeting if it contains product-related words
        product_keywords = r'\b(?:cable|cables|adapter|adapters|dock|docks|hub|hubs|' \
                          r'hdmi|displayport|usb|thunderbolt|tb3|tb4|ethernet|monitor|' \
                          r'need|looking|find|show|want)\b'
        if re.search(product_keywords, text):
            return False

        return has_pattern(text, GREETING_PATTERNS)

    def _is_pure_farewell(self, text: str, word_count: int) -> bool:
        """Check if text is a pure farewell (short)."""
        if word_count > 6:
            return False
        return has_pattern(text, FAREWELL_PATTERNS)


    def classify(self, prompt: str, context: ConversationContext, debug_lines: list = None) -> Intent:
        """
        Classify user intent using LLM-first approach.

        Strategy:
        1. Handle pending clarification (special state)
        2. Fast path for unambiguous patterns (SKU, greeting, farewell, gibberish)
        3. LLM for everything else

        Args:
            prompt: User's message
            context: Conversation context
            debug_lines: Optional list to append debug messages to

        Returns:
            Intent object with classification
        """
        # Priority 0a: Handle pending narrowing BEFORE any classification
        if context.has_pending_narrowing():
            # Check "show me all" BEFORE escape (since "show me" matches escape patterns)
            if self._is_narrowing_show_all(prompt):
                if debug_lines is not None:
                    debug_lines.append("NARROWING: User chose 'show all'")
                return Intent(
                    type=IntentType.FOLLOWUP,
                    confidence=0.95,
                    reasoning="User wants to see all remaining products",
                    meta_info={'narrowing_response': True, 'show_all': True}
                )
            if self._is_option_selection(prompt):
                if debug_lines is not None:
                    debug_lines.append("NARROWING RESPONSE: Routing to narrowing handler")
                return Intent(
                    type=IntentType.FOLLOWUP,
                    confidence=0.95,
                    reasoning="User selecting option by number/ordinal",
                    meta_info={'narrowing_response': True}
                )
            # SKU lookup escapes narrowing (user wants a specific product)
            sku = self._extract_sku(prompt)
            if sku:
                context.escape_narrowing()
                if debug_lines is not None:
                    debug_lines.append(f"SKU ESCAPE: Breaking out of narrowing for SKU {sku}")
                return Intent(
                    type=IntentType.SPECIFIC_SKU,
                    confidence=0.95,
                    reasoning=f"SKU lookup for {sku} during narrowing",
                    sku=sku
                )
            if self._is_correction_pattern(prompt):
                context.escape_narrowing()
                if debug_lines is not None:
                    debug_lines.append("CORRECTION: Breaking out of narrowing for correction")
                # Fall through to classification
            elif self._is_escape_pattern(prompt, during_narrowing=True):
                context.escape_narrowing()
                if debug_lines is not None:
                    debug_lines.append("ESCAPE: Breaking out of narrowing flow")
                # Fall through to classification
            else:
                if debug_lines is not None:
                    debug_lines.append("NARROWING RESPONSE: Routing to narrowing handler")
                return Intent(
                    type=IntentType.FOLLOWUP,
                    confidence=0.95,
                    reasoning="User responding to narrowing question",
                    meta_info={'narrowing_response': True}
                )

        # Priority 0b: Handle pending clarification BEFORE any classification
        # This prevents the infinite loop where clarification responses get
        # misclassified as NEW_SEARCH and trigger the same clarification question
        if context.has_pending_clarification():
            if self._is_escape_pattern(prompt):
                # User wants to break out of clarification flow
                context.clear_pending_clarification()
                if debug_lines is not None:
                    debug_lines.append("ESCAPE: Breaking out of clarification flow")
                # Fall through to classification
            elif self._is_followup_about_products(prompt, context):
                # User is asking about products in context, not responding to clarification
                context.clear_pending_clarification()
                if debug_lines is not None:
                    debug_lines.append("FOLLOWUP DETECTED: Breaking out of clarification for product question")
                # Fall through to classification (will become FOLLOWUP)
            elif self._contains_filter_specification(prompt):
                # User provided a filter spec (like "6 feet") - treat as search/filter, not clarification
                context.clear_pending_clarification()
                if debug_lines is not None:
                    debug_lines.append("FILTER SPEC DETECTED: Breaking out of clarification for filter specification")
                # Fall through to classification
            else:
                # Treat as clarification response
                if debug_lines is not None:
                    debug_lines.append("CLARIFICATION RESPONSE: Routing to clarification handler")
                return Intent(
                    type=IntentType.FOLLOWUP,
                    confidence=0.95,
                    reasoning="User responding to clarification question",
                    meta_info={'clarification_response': True}
                )

        # Priority 0c: Bare product reference when products are in context
        if context.current_products and self._is_option_selection(prompt):
            if debug_lines is not None:
                debug_lines.append("PRODUCT REFERENCE: Routing to followup handler")
            return Intent(
                type=IntentType.FOLLOWUP,
                confidence=0.95,
                reasoning="User selecting a product by number",
                meta_info={}
            )

        # Priority 0d: "show me all" with products in context (outside narrowing)
        if context.current_products and self._is_narrowing_show_all(prompt):
            if debug_lines is not None:
                debug_lines.append("SHOW ALL: Products in context, routing to followup")
            return Intent(
                type=IntentType.FOLLOWUP,
                confidence=0.90,
                reasoning="User wants to see all products in context"
            )

        # Step 1: Fast path for unambiguous patterns (no LLM needed)
        fast_result = self._fast_path_check(prompt, context)
        if fast_result:
            if debug_lines is not None:
                debug_lines.append(f"FAST PATH: {fast_result.type.value} ({fast_result.confidence:.2f})")
                if fast_result.reasoning:
                    debug_lines.append(f"   Reason: {fast_result.reasoning[:80]}")
            return fast_result

        # Step 2: LLM classification for everything else
        try:
            intent = self._classify_with_llm(prompt, context)

            # Post-LLM sanity check: clarification_response requires pending clarification
            # LLM sometimes hallucinates this intent for short answers like "around 6 feet"
            # Note: LLM may still return CLARIFICATION_RESPONSE from legacy mapping
            if intent.type == IntentType.FOLLOWUP and intent.meta_info and intent.meta_info.get('clarification_response') and not context.has_pending_clarification():
                # LLM incorrectly classified as clarification_response - re-classify
                # IMPORTANT: Check for new search criteria FIRST, before checking products in context
                # This fixes Issue 24: "USB-C I think" should search, not check compatibility
                if self._contains_new_search_criteria(prompt):
                    # Contains connector/category keywords = user wants a NEW search
                    intent = Intent(
                        type=IntentType.NEW_SEARCH,
                        confidence=0.80,
                        reasoning="Reclassified: clarification_response with new search criteria (connector/category)"
                    )
                    if debug_lines is not None:
                        debug_lines.append(f"RECLASSIFIED: clarification_response -> new_search (search criteria detected)")
                elif context.current_products:
                    # Has products in context but no new search criteria = likely a followup question
                    intent = Intent(
                        type=IntentType.FOLLOWUP,
                        confidence=0.80,
                        reasoning="Reclassified: clarification_response without pending clarification, products in context"
                    )
                    if debug_lines is not None:
                        debug_lines.append(f"RECLASSIFIED: clarification_response -> followup (products in context)")
                elif self._contains_filter_specification(prompt):
                    # Has filter specs = likely a new search
                    intent = Intent(
                        type=IntentType.NEW_SEARCH,
                        confidence=0.80,
                        reasoning="Reclassified: clarification_response without pending clarification, filter specs detected"
                    )
                    if debug_lines is not None:
                        debug_lines.append(f"RECLASSIFIED: clarification_response -> new_search (filter specs)")

            # Post-LLM sanity check: CLARIFICATION with search criteria should be new_search
            # This fixes Issue 25: "Or maybe Thunderbolt?" should search, not ask for clarification
            # User is suggesting an ALTERNATIVE search, not asking about current products
            if intent.type == IntentType.CLARIFICATION and self._contains_new_search_criteria(prompt):
                intent = Intent(
                    type=IntentType.NEW_SEARCH,
                    confidence=0.80,
                    reasoning="Reclassified: clarification query with new search criteria (connector/category)"
                )
                if debug_lines is not None:
                    debug_lines.append(f"RECLASSIFIED: clarification -> new_search (search criteria detected)")

            if debug_lines is not None:
                debug_lines.append(f"LLM INTENT: {intent.type.value} ({intent.confidence:.2f})")
                if intent.reasoning:
                    debug_lines.append(f"   Reason: {intent.reasoning[:80]}")
            return intent
        except Exception as e:
            if debug_lines is not None:
                debug_lines.append(f"LLM INTENT FAILED: {type(e).__name__}: {str(e)[:100]}")
            # No regex fallback in v2 - return CLARIFICATION on failure
            return Intent(
                type=IntentType.CLARIFICATION,
                confidence=0.3,
                reasoning="LLM classification failed"
            )

    def _is_narrowing_show_all(self, prompt: str) -> bool:
        """Check if user wants to skip narrowing and see all products."""
        text = prompt.lower().strip()
        patterns = [
            r'\bshow\s+(?:me\s+)?(?:all|everything|them\s+all)\b',
            r'\bjust\s+show\b',
            r'\blist\s+(?:all|them)\b',
            r'\bi\s+don\'?t\s+(?:care|mind)\b',
            r'\bany\s+(?:of\s+them|will\s+do|is\s+fine)\b',
            r'\bnot\s+sure\b',
        ]
        return any(re.search(pat, text) for pat in patterns)

    def _is_option_selection(self, prompt: str) -> bool:
        """Detect if user is trying to select a narrowing option by ordinal or number."""
        prompt_lower = prompt.lower().strip()
        # Questions about products are not option selections
        if '?' in prompt_lower or re.match(r'^(?:what|how|does|is|can|which|where|why|who)\b', prompt_lower):
            return False
        # "1st", "first", "2nd option", "the second one", etc.
        if re.search(r'\b(?:1st|first|2nd|second|3rd|third|4th|fourth)\b', prompt_lower):
            return True
        # "product 2", "option 3", "#2", bare "2"
        if re.match(r'^(?:product|option)?\s*#?\s*\d+$', prompt_lower):
            return True
        return False

    def _is_escape_pattern(self, prompt: str, during_narrowing: bool = False) -> bool:
        """
        Detect if user wants to escape from clarification/narrowing flow.

        Escape patterns allow users to:
        - Say goodbye/thanks
        - Ask educational questions
        - Start an explicit new search

        Args:
            prompt: User's message

        Returns:
            True if user wants to escape clarification flow
        """
        prompt_lower = prompt.lower().strip()

        # Strip conversational filler prefixes so "ok, do you have..." matches "do you have"
        filler_prefixes = [
            'ok, ', 'ok ', 'okay, ', 'okay ', 'well, ', 'well ',
            'so, ', 'so ', 'alright, ', 'alright ', 'hey, ', 'hey ',
            'hmm, ', 'hmm ', 'um, ', 'um ', 'uh, ', 'uh ',
        ]
        stripped = prompt_lower
        for prefix in filler_prefixes:
            if stripped.startswith(prefix):
                stripped = stripped[len(prefix):].lstrip()
                break

        # Farewells - let user exit gracefully
        farewell_patterns = [
            'bye', 'goodbye', 'thanks', 'thank you', 'no thanks',
            'never mind', 'nevermind', 'forget it', 'cancel'
        ]
        if any(p in prompt_lower for p in farewell_patterns):
            return True

        # Educational / general questions - user is asking, not answering
        # During narrowing, answers are short ("active", "6 feet", "the first one")
        # Questions start with question words and are longer
        question_starts = [
            'what ', "what's ", 'how ', 'why ', 'which ',
            'can you ', 'could you ',
            'is there ', 'does ', 'do ', 'is it ', 'are there ',
            'difference between', 'explain', 'tell me about'
        ]
        if any(stripped.startswith(p) for p in question_starts) and len(prompt_lower.split()) >= 4:
            return True

        # Explicit new search - user is starting fresh with clear intent
        # These are strong signals that override clarification context
        explicit_search_patterns = [
            'show me', 'find me', 'search for', 'look for',
            'i want', 'i need', 'i\'m looking for', 'do you have'
        ]
        if any(stripped.startswith(p) for p in explicit_search_patterns):
            # During narrowing, "i need/want" might be answering the question
            # ("I need one with at least 2 ports"). Only escape if a product
            # category word follows, indicating a genuinely new search.
            if during_narrowing and any(stripped.startswith(p) for p in ['i want', 'i need']):
                category_words = [
                    'cable', 'adapter', 'dock', 'hub', 'switch', 'mount',
                    'splitter', 'extender', 'enclosure', 'rack', 'kvm',
                    'privacy', 'monitor', 'display', 'hdmi', 'usb-c',
                    'thunderbolt', 'fiber', 'ethernet', 'network card',
                ]
                if not any(w in stripped for w in category_words):
                    return False  # Likely answering the narrowing question
            return True

        # Product-reference questions - user is asking about specific products
        product_patterns = [
            r'\bcompare\b', r'\bproduct\s+\d', r'\b#\d+\b',
            r'\bfirst\s+one\b', r'\bsecond\s+one\b', r'\bwhich\s+one\b',
        ]
        if any(re.search(p, prompt_lower) for p in product_patterns):
            return True

        return False

    def _is_correction_pattern(self, prompt: str) -> bool:
        """Detect if user is correcting their previous search during narrowing."""
        prompt_lower = prompt.lower()
        correction_patterns = [
            r'\b(?:actually|sorry|oops)\b',
            r'\bi\s+meant\b',
            r'\binstead\s+of\b',
            r'\bnot\s+\S+.*\b(?:hdmi|displayport|usb|vga|thunderbolt|ethernet|dvi)\b',
            r'\b(?:wrong|different)\s+(?:type|connector|cable)\b',
        ]
        return any(re.search(p, prompt_lower) for p in correction_patterns)

    def _contains_filter_specification(self, prompt: str) -> bool:
        """
        Detect if prompt contains filter specifications (length, color, features).

        This allows users to provide filter specs like "6 feet" or "shorter" during
        a clarification flow, which should be treated as a search intent rather than
        a clarification response.

        Args:
            prompt: User's message

        Returns:
            True if prompt contains extractable filter specifications
        """
        import re
        prompt_lower = prompt.lower().strip()

        filter_patterns = [
            # Length specifications
            r'\d+\s*(?:ft|feet|foot|m|meter|meters|inch|inches|cm)',
            r'\baround\s+\d+',  # "around 6 feet"
            r'\babout\s+\d+\s*(?:ft|feet|foot|m|meter)',  # "about 6 feet"

            # Relative length
            r'\b(?:shorter|longer|smaller|bigger|shorter|taller)\b',

            # Color specifications
            r'\b(?:black|white|gray|grey|silver|blue|red|green)\s+(?:one|cable|version)?\b',

            # Feature specifications
            r'\b(?:4k|8k|1080p|1440p|144hz|60hz|120hz)\b',

            # Connector specifications (when clarifying)
            r'\b(?:hdmi|displayport|usb-?c|usb-?a|vga|dvi|thunderbolt)\s+(?:to|cable|end)\b',
        ]

        return any(re.search(p, prompt_lower, re.IGNORECASE) for p in filter_patterns)

    def _contains_new_search_criteria(self, text: str) -> bool:
        """
        Check if text contains new search criteria like connector types or product categories.

        These indicate the user wants a NEW search, not a followup on current products.
        Used to properly reclassify clarification_response when it contains search criteria.

        For example:
        - "USB-C I think" -> should search for USB-C products, not check compatibility
        - "HDMI cable" -> should search for HDMI cables, not filter current products

        Args:
            text: User's message

        Returns:
            True if text contains new search criteria (connectors, product categories)
        """
        text_lower = text.lower()

        # Standalone connector types (no "to/cable/end" required)
        # These are strong signals that user wants a specific product type
        connector_keywords = [
            'usb-c', 'usb c', 'type-c', 'type c',
            'usb-a', 'usb a', 'type-a', 'type a',
            'hdmi', 'displayport', 'dp ',  # "dp " with space to avoid matching "display"
            'thunderbolt', 'tb3', 'tb4',
            'vga', 'dvi', 'ethernet', 'rj45', 'rj-45',
            'lightning', 'micro usb', 'mini usb',
        ]

        # Product category keywords
        category_keywords = [
            'cable', 'cables', 'adapter', 'adapters',
            'hub', 'hubs', 'dock', 'docking',
            'charger', 'charging',
        ]

        # Check for connector keywords
        for kw in connector_keywords:
            if kw in text_lower:
                return True

        # Check for category keywords
        for cat in category_keywords:
            if cat in text_lower:
                return True

        return False

    def _is_followup_about_products(self, prompt: str, context: ConversationContext) -> bool:
        """
        Detect if user is asking a followup question about products in context.

        This prevents followup questions from being trapped in clarification flow
        when user has products shown and asks about them.

        Args:
            prompt: User's message
            context: Conversation context with product history

        Returns:
            True if this is a followup about shown products
        """
        # Must have products in context to be a followup about products
        if not context.current_products:
            return False

        prompt_lower = prompt.lower().strip()

        # Superlative patterns - asking about best/most/least
        superlative_patterns = [
            r'\bwhich\s+(?:one\s+)?(?:has\s+)?(?:the\s+)?(?:most|best|cheapest|longest|shortest|fastest|smallest|largest)',
            r'\bwhich\s+(?:one\s+)?is\s+(?:the\s+)?(?:best|cheapest|longest|shortest|fastest)',
            r'\bwhat(?:\'s|\s+is)\s+the\s+(?:best|cheapest|longest|shortest|fastest)',
        ]

        # Comparison patterns
        comparison_patterns = [
            r'\bcompare\s+(?:these|them|the)',
            r'\bwhat(?:\'s|\s+is)\s+the\s+difference',
            r'\bhow\s+(?:do\s+)?(?:these|they)\s+compare',
            r'\bdifference\s+between\s+(?:these|them|the)',
        ]

        # Context pronoun patterns - questions about "these", "they", "them", ordinals
        context_pronoun_patterns = [
            r'\b(?:do|does|can|will|are|is)\s+(?:these|they|them|it|any\s+of\s+them)',
            r'\b(?:what|which)\s+(?:ports?|features?|specs?)\s+(?:do|does)\s+(?:these|they|it)',
            r'\btell\s+me\s+(?:more\s+)?about\s+(?:the\s+)?(?:first|second|third|\d+(?:st|nd|rd|th)|#\d+)',
            r'\bwhat\s+about\s+(?:the\s+)?(?:first|second|third|\d+(?:st|nd|rd|th)|#\d+)',
            r'\b(?:the\s+)?(?:first|second|third|\d+(?:st|nd|rd|th)|#\d+)\s+one',
        ]

        import re
        all_patterns = superlative_patterns + comparison_patterns + context_pronoun_patterns

        for pattern in all_patterns:
            if re.search(pattern, prompt_lower):
                return True

        return False

    def _classify_with_llm(self, prompt: str, context: ConversationContext) -> Intent:
        """Perform LLM-based classification."""
        client = get_openai_client()
        if not client:
            raise RuntimeError("OpenAI client not available")

        # Build context summary
        context_info = self._build_context_summary(context)

        # Prepare system prompt with context
        system_prompt = INTENT_CLASSIFICATION_PROMPT.format(context_info=context_info)

        # Call OpenAI with retry
        handler = RetryHandler(
            config=self.retry_config,
            operation_name="llm_intent_classification"
        )

        response = handler.execute(
            lambda: self._call_openai(client, system_prompt, prompt),
            fallback=None,
            raise_on_failure=True
        )

        if response is None:
            raise RuntimeError("LLM returned no response")

        # Parse response into Intent
        return self._parse_response(response, prompt)

    def _build_context_summary(self, context: ConversationContext) -> str:
        """Build context summary for LLM."""
        parts = []

        # Products in context - show ALL SKUs so LLM can detect references
        if context.current_products:
            products_info = []
            for i, p in enumerate(context.current_products[:10], 1):
                category = p.metadata.get('category', 'product')
                products_info.append(f"#{i} {p.product_number} ({category})")
            parts.append(f"**Products shown:** {', '.join(products_info)}")
            parts.append("(User can ask followup questions about these — if they mention a SKU listed here, it's FOLLOWUP not NEW_SEARCH)")
        else:
            parts.append("**No products currently shown**")
            parts.append("(FOLLOWUP intent requires products in context)")

        # Pending clarification
        if context.pending_clarification:
            parts.append(f"**Pending clarification:** Bot asked user for more info about: {context.pending_clarification.vague_type.value}")
            parts.append("(Short answers are likely FOLLOWUP with clarification_response=True)")

        # Last search filters (stored as dict)
        if context.last_filters:
            filter_parts = []
            # Handle both dict and object access for backwards compatibility
            if isinstance(context.last_filters, dict):
                category = context.last_filters.get('category')
                connector = context.last_filters.get('connector_from')
            else:
                category = getattr(context.last_filters, 'product_category', None)
                connector = getattr(context.last_filters, 'connector_from', None)
            if category:
                filter_parts.append(f"category={category}")
            if connector:
                filter_parts.append(f"connector={connector}")
            if filter_parts:
                parts.append(f"**Previous search:** {', '.join(filter_parts)}")

        # Recent conversation turns — helps LLM resolve pronouns like "it" after recommendations
        # Only include after 2+ full exchanges (5+ messages) to avoid biasing simple followups
        recent = context.get_conversation_history(limit=4)
        if recent and len(context.messages) >= 5:
            turns = []
            for msg in recent:
                role_label = "User" if msg.role == "user" else "Bot"
                # Truncate long bot responses to keep context summary compact
                content = msg.content[:150] + "..." if len(msg.content) > 150 else msg.content
                turns.append(f"  {role_label}: {content}")
            parts.append("**Recent conversation:**\n" + "\n".join(turns))

        return "\n".join(parts) if parts else "No prior context"

    def _call_openai(self, client, system_prompt: str, user_prompt: str) -> str:
        """Make OpenAI API call."""
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=self.temperature,
            max_completion_tokens=300,
            response_format={"type": "json_object"},  # Force valid JSON output
            timeout=30.0
        )
        return response.choices[0].message.content

    def _parse_response(self, response: str, original_prompt: str) -> Intent:
        """Parse LLM response into Intent object."""
        try:
            # With json_object mode, response should be valid JSON directly
            data = json.loads(response)

            # Map intent string to IntentType
            intent_str = data.get('intent', 'CLARIFICATION').upper()
            intent_type = self._map_intent_type(intent_str)

            # Extract optional fields
            confidence = float(data.get('confidence', 0.8))
            reasoning = data.get('reasoning', '')
            sku = data.get('sku')
            meta_info = data.get('meta_info')

            # Clean up meta_info (remove null values)
            if meta_info:
                meta_info = {k: v for k, v in meta_info.items() if v is not None}
                if not meta_info:
                    meta_info = None

            return Intent(
                type=intent_type,
                confidence=confidence,
                reasoning=reasoning,
                sku=sku,
                meta_info=meta_info
            )

        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
            _logger.error("Error parsing LLM response", extra={
                "error": str(e),
                "response_preview": response[:200] if response else None
            })
            # Return CLARIFICATION on parse error
            return Intent(
                type=IntentType.CLARIFICATION,
                confidence=0.5,
                reasoning=f"Failed to parse LLM response: {e}"
            )

    def _map_intent_type(self, intent_str: str) -> IntentType:
        """Map intent string to IntentType enum."""
        mapping = {
            'GREETING': IntentType.GREETING,
            'FAREWELL': IntentType.FAREWELL,
            'OUT_OF_SCOPE': IntentType.OUT_OF_SCOPE,
            'NEW_SEARCH': IntentType.NEW_SEARCH,
            'FOLLOWUP': IntentType.FOLLOWUP,
            'EDUCATIONAL': IntentType.EDUCATIONAL,
            'CLARIFICATION': IntentType.CLARIFICATION,
            'SPECIFIC_SKU': IntentType.SPECIFIC_SKU,
            # Legacy safety net
            'PRODUCT_EXISTENCE': IntentType.NEW_SEARCH,
            'AMBIGUOUS': IntentType.CLARIFICATION,
            'CLARIFICATION_RESPONSE': IntentType.FOLLOWUP,
            'NARROWING_RESPONSE': IntentType.FOLLOWUP,
        }
        return mapping.get(intent_str, IntentType.CLARIFICATION)


# Convenience function for testing
def classify_with_llm(prompt: str, context: ConversationContext = None) -> Intent:
    """
    Classify intent using LLM (convenience function).

    Args:
        prompt: User's message
        context: Conversation context (creates empty one if None)

    Returns:
        Intent object
    """
    if context is None:
        context = ConversationContext()

    classifier = LLMIntentClassifier()
    return classifier.classify(prompt, context)


# Test function
if __name__ == "__main__":
    print("LLM Intent Classifier Test")
    print("=" * 60)

    # Test queries covering all intent types
    test_cases = [
        ("hello", "GREETING"),
        ("goodbye, thanks for your help", "FAREWELL"),
        ("how do I return this cable?", "OUT_OF_SCOPE"),
        ("cheapest HDMI cable", "OUT_OF_SCOPE"),
        ("asdfghjkl random text", "OUT_OF_SCOPE"),
        ("do you have Cat7?", "NEW_SEARCH"),
        ("what's the difference between Cat6 and Cat6a?", "EDUCATIONAL"),
        ("I need HDMI cables", "NEW_SEARCH"),
        ("TB3CDK2DH", "SPECIFIC_SKU"),
        ("HDMI calbe under 6 feet", "NEW_SEARCH"),  # typo
        ("USB-C to HDMI for my MacBook", "NEW_SEARCH"),
        ("does it support 4K?", "CLARIFICATION"),  # no context
        ("cable", "CLARIFICATION"),
    ]

    context = ConversationContext()
    classifier = LLMIntentClassifier()

    for query, expected in test_cases:
        intent = classifier.classify(query, context)
        status = "PASS" if intent.type.value.upper() == expected.lower() else "FAIL"
        print(f"{status} '{query}'")
        print(f"   Expected: {expected}")
        print(f"   Got: {intent.type.value.upper()} ({intent.confidence:.2f})")
        print(f"   Reason: {intent.reasoning}")
        print()
