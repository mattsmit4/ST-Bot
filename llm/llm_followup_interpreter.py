"""
LLM-based Followup Interpreter for ST-Bot.

When a user asks a followup question about products in context,
this interpreter uses LLM to understand what the user wants:
- FILTER: Show only products matching criteria (e.g., "2 meter ones")
- COMPARE: Compare specific products
- DETAILS: Show details for specific product(s)
- SPECS: Show specs for all products
- CLARIFY: Ask what they mean

This prevents showing all products when user asks something specific.
"""

import os
import json
import re
from typing import Optional, List
from dataclasses import dataclass
from enum import Enum

from core.models import Product
from core.openai_client import get_openai_client


class FollowupAction(Enum):
    """Types of actions for followup queries."""
    FILTER = "filter"          # Filter to products matching criteria
    COMPARE = "compare"        # Compare specific products
    DETAILS = "details"        # Show details for specific product(s)
    SPECS = "specs"            # Show specs for all products
    COMPATIBILITY = "compatibility"  # Check if products support a feature (yes/no for all)
    EDUCATIONAL = "educational"  # Answer general knowledge/physics questions
    RECOMMEND = "recommend"    # Make a product recommendation
    RECALL = "recall"          # Recall products from earlier in conversation
    CLARIFY = "clarify"        # Ask for clarification


@dataclass
class FollowupInterpretation:
    """Result of interpreting a followup query."""
    action: FollowupAction
    criteria: Optional[dict] = None      # For FILTER: {"length_m": 2, "color": "black"}
    product_indices: Optional[List[int]] = None  # For DETAILS/COMPARE: [1, 2]
    reasoning: str = ""
    clarification_question: Optional[str] = None  # For CLARIFY


FOLLOWUP_INTERPRETATION_PROMPT = '''You are interpreting follow-up questions about products.

User has these products in context:
{products_summary}
{conversation_context}
User asks: "{query}"

IMPORTANT - Resolving References:
When user uses words like "that", "this", "it", "the same", look at the bot's last response
to understand what they're referring to. For example:
- If bot mentioned "shielding" and user asks "does the third one support that too?" → "that" = shielding
- If bot discussed "4K support" and user asks "what about that?" → "that" = 4K support
- Don't guess features that weren't discussed - use the conversation context!

IMPORTANT - Product Targeting:
- IMPORTANT: product_indices are 1-based. #1 = index 1, #4 = index 4. Do NOT use 0-based indexing.
- If user mentions a specific SKU, the [User's query mentions: #N SKU] note provides the correct 1-based index. Use it.
- If user says "that one" / "that product" / "it", refer to the [Last discussed: #N] product noted above.
- If user asks "is there one with more/less/better X?", they're comparing across ALL current products — include all indices.
- If user says "they" / "these" / "all of them", include ALL product indices.
- Only set product_indices to ALL products when the user is clearly asking about the entire set.

Determine what the user wants:

## FILTER
User wants to see only products matching a specific criterion.

### Absolute Length (specific measurement):
- "2 meter ones" → filter by length_m = 2
- "the 6ft one" → filter by length_ft = 6
- "show me just the 6ft options" → filter by length_ft = 6
- "around 6 feet" → filter by length_ft = 6

### Relative Length - IMPORTANT: Distinguish these two patterns!

#### Superlative QUESTIONS about current products (NOT a filter):
User is asking ABOUT the current products, not requesting different ones.
- "which one is longest?" → action="specs" (answer which current product is longest)
- "what's the shortest one?" → action="specs" (identify shortest among current products)
- "which is the longer option?" → action="specs" (compare current products)
- "is there a longer one among these?" → action="specs" (check current products)
→ Return action="specs" with criteria.question="length" - DO NOT use length_comparison!

#### Superlative/comparative questions about ANY attribute (NOT a filter):
Same pattern as length — when user asks "is there one with more/fewer/better/faster X?"
about the CURRENT products, they want to know WHICH product has the most/least/best X.
- "is there one with more USB ports?" → action="specs", criteria.question="USB ports"
- "which one has the best resolution?" → action="specs", criteria.question="resolution"
- "does any have faster data transfer?" → action="specs", criteria.question="data transfer"
→ Return action="specs" — NOT action="filter"!

#### Comparative REQUESTS for different products (filter/search):
User wants NEW products that are shorter/longer than current ones.
- "show me longer cables" → length_comparison = "longer"
- "I need something shorter" → length_comparison = "shorter"
- "do you have longer ones?" → length_comparison = "longer"
- "no, shorter" → length_comparison = "shorter"
- "something longer" → length_comparison = "longer"
- "I need them shorter" → length_comparison = "shorter"
→ Return action="filter" with criteria.length_comparison

### Length range filters:
When user specifies a length range, extract BOTH values as length_ft (convert meters to feet if needed):
- "between 3 and 6 feet" → criteria: {{"length_ft": 3, "length_max_ft": 6}}
- "1 to 2 meters" → criteria: {{"length_ft": 3.28, "length_max_ft": 6.56}}
- "around 10 feet" → criteria: {{"length_ft": 10}}
IMPORTANT: Always extract a numeric length — do NOT return criteria: null for length queries.

### Other filters:
- "the black one" → filter by color = black
- "the 4K one" → filter by feature = 4K
- "only USB-C" → filter by connector
- "single mode" / "multimode" / "OM3" / "OS2" → filter by feature (for fiber cables)

### Alternative product requests ("is there one with...?"):
IMPORTANT: "Is there one with..." / "do you have one with..." means the user wants a DIFFERENT product — NOT asking about the current one. This is a FILTER or new search, not SPECS.
- "is there one with more USB ports?" → filter (user wants alternatives with more ports)
- "do you have one with wireless?" → filter (user wants a wireless version)
- "is there a version with PoE?" → filter by feature = PoE
- "is there a longer option?" → length_comparison = "longer"
Key distinction: "is there one with X?" = wants DIFFERENT products (filter). "Which one has X?" = asking ABOUT current products (specs).

### Corrections / Requirement changes:
User is changing what they need — extract the NEW requirement as a filter.
- "I actually need DVI not VGA" → filter by connector = DVI
- "Sorry, I need USB-A not USB-C" → filter by connector = USB-A
- "Actually, I want the 4K ones" → filter by feature = 4K
- "Not that, I want the 10 foot ones" → filter by length_ft = 10
- "Switch to DisplayPort" → filter by connector = DisplayPort
- "I meant single mode, not multimode" → filter by feature = single mode
Key signals: "actually", "sorry", "I meant", "not X, Y", "instead of", "switch to"
→ Extract the NEW requirement (ignore the old one). Return action="filter".

## COMPARE
User wants to compare specific products.
Examples:
- "compare 1 and 2" → compare products [1, 2]
- "what's the difference between them" → compare all

## DETAILS
User wants details about a specific product.
Examples:
- "tell me more about the first one" → details for [1]
- "what are the specs on product 2" → details for [2]

## SPECS
User wants to know about specific specs or has a question about product details.
This includes:
- Questions about ALL products: "show me the specs", "what are all the features"
- Questions about a SPECIFIC product's field: "how many ports does the second one have?", "what's the wattage on #1?"
- Any "how many", "what is the", "does it have" question about a product

Examples:
- "show me the specs" → specs for all
- "what are all the features" → specs for all
- "how many ports does the second one have?" → specs for [2], criteria.question="ports"
- "what's the resolution on product 1?" → specs for [1], criteria.question="resolution"
- "how heavy is it?" → specs for [1] (or all), criteria.question="weight"
- "what wattage does it support?" → specs for all, criteria.question="wattage"
- "does that one support 4K?" → specs for [N], criteria.question="4K support"
- "what's the warranty on that?" → specs for [N], criteria.question="warranty"
- "what's the maximum cable distance?" → specs for [N], criteria.question="cable length"

IMPORTANT: When the user asks a specific question about a product, use action="specs" with product_indices and criteria.question — NOT "details" (which shows everything) and NOT "clarify" (the question is clear). Questions about warranty, cable length, weight, dimensions, etc. are all SPECS questions — they have clear answers in the product data.

## COMPATIBILITY
User is asking a yes/no question about whether ALL products have a feature or are compatible with something.
When user uses plural pronouns (they, them, these, those) asking about a feature or compatibility, this is a compatibility check.
IMPORTANT: When user says "they" referring to products, they want to know about ALL products - don't ask for clarification!
IMPORTANT: This action is ONLY for plural/all products. If the user asks about a SINGLE product ("that one", "it", "the first one", "product 3"), use SPECS instead — even for yes/no feature questions like "does that one support 4K?".

Examples:
- "Do they work with Thunderbolt 3?" → compatibility check for "Thunderbolt 3"
- "Are they compatible with Mac?" → compatibility check for "Mac"
- "Do they support 4K?" → compatibility check for "4K"
- "Can they handle 100W power delivery?" → compatibility check for "100W power delivery"
- "Do they all have USB-C?" → compatibility check for "USB-C"
- "Will they work with my MacBook?" → compatibility check for "MacBook"

Return action="compatibility" with criteria.feature describing what to check.

## EDUCATIONAL
User is asking a general knowledge question about technology, physics, or how things work.
NOT about specific products in context - about the underlying concepts.
Key indicators: "Does X affect Y?", "How does X work?", "Why is X better?", "What's the difference between X and Y?"

Examples:
- "Does cable length affect charging speed?" → educational (cable physics)
- "How does USB-C deliver power?" → educational (technology explanation)
- "What's the difference between Cat6 and Cat6a?" → educational (category comparison)
- "Why is shielded better than unshielded?" → educational (explanation)
- "Will longer cables slow down my data transfer?" → educational (physics)
- "Does that length affect charging speed or data transfer rates?" → educational (physics)

IMPORTANT: These are NOT about the specific products shown - they're general technology questions.
Compare to COMPATIBILITY which asks "Do THESE SPECIFIC products support X?"

Return action="educational" with criteria.topic describing the topic (e.g., "cable length and charging").

## RECOMMEND
User wants a recommendation about which product to choose.
Examples:
- "which would you recommend?" → recommend (compare all and suggest best)
- "which one should I get?" → recommend
- "should I pick the cheapest?" → recommend (address price/value)
- "what's the best option?" → recommend
- "which is best for gaming?" → recommend (with criteria.use_case = "gaming")
- "which of those would work best if I need dual 4K and 90W charging?" → recommend (criteria from question)
- "which one handles dual 4K 60Hz, 90W, and gigabit ethernet?" → recommend (evaluating shown products)
- "do any of these support 4K 60Hz and USB-C?" → recommend (checking shown products against criteria)

IMPORTANT: When user says "which of those/these" + specific criteria, it is ALWAYS recommend — they are explicitly asking about the current products. Do NOT clarify.

Return action="recommend" with optional criteria.use_case if specified.

## RECALL
User wants to see products that were shown EARLIER in the conversation (not the current products).
Key indicators: "from earlier", "you showed before", "remember when", "those [category] earlier",
"go back to", "what about those [category]", "still have those", "back to the"

Examples:
- "Do you still have those docks from earlier?" → recall category_hint="dock"
- "Can I see those cables you showed before?" → recall category_hint="cable"
- "Go back to the hubs" → recall category_hint="hub"
- "What about those HDMI cables from before?" → recall category_hint="hdmi cable"
- "Show me the adapters you had earlier" → recall category_hint="adapter"

IMPORTANT: This is NOT a filter - user wants DIFFERENT products than what's currently shown.
The category they mention is what they want to recall, not a filter on current products.

Return action="recall" with criteria.category_hint describing what to find in history.

## CLARIFY
Can't determine what user wants, need to ask.
Use ONLY when truly ambiguous - NOT when user says "they" (that means all products).
Examples:
- "the one" (which one? - singular, ambiguous)
- "something different" (what specifically?)

Return JSON:
{{
    "action": "filter" | "compare" | "details" | "specs" | "compatibility" | "educational" | "recommend" | "recall" | "clarify",
    "criteria": {{"length_m": 2}} or {{"length_ft": 6}} or {{"length_comparison": "shorter"}} or {{"length_comparison": "longer"}} or {{"color": "black"}} or {{"feature": "4K"}} or {{"connector": "DVI"}} or {{"category_hint": "dock"}} or null,
    "product_indices": [1, 2] or null,  // 1-based, matching #N labels. NOT 0-based.
    "reasoning": "brief explanation",
    "clarification_question": "What would you like me to clarify?" or null
}}'''


class LLMFollowupInterpreter:
    """
    LLM-based interpreter for followup questions.

    Uses GPT-4o-mini to understand what the user wants when asking
    about products in context. Prevents showing all products when
    user asks something specific like "2 meter ones".
    """

    def __init__(self, model: str = None):
        """
        Initialize LLM followup interpreter.

        Args:
            model: OpenAI model (default: gpt-4o-mini)
        """
        self.model = model or os.environ.get('OPENAI_CHAT_MODEL', 'gpt-5.4-nano')

    def interpret(
        self,
        query: str,
        products: List[Product],
        debug_lines: list = None,
        last_bot_response: Optional[str] = None,
        recent_bot_responses: Optional[list] = None,
    ) -> Optional[FollowupInterpretation]:
        """
        Interpret a followup query to understand what user wants.

        Args:
            query: User's followup question
            products: Products currently in context
            debug_lines: Optional list for debug output
            last_bot_response: Bot's last response for resolving anaphoric references
            recent_bot_responses: Earlier bot responses for pronoun resolution across turns

        Returns:
            FollowupInterpretation with action and criteria, or None on error
        """
        try:
            result = self._interpret_with_llm(query, products, last_bot_response, recent_bot_responses)

            if debug_lines is not None:
                debug_lines.append(f"🤖 FOLLOWUP INTERPRET: {result.action.value}")
                debug_lines.append(f"   Criteria: {result.criteria}")
                if result.product_indices:
                    debug_lines.append(f"   Products: {result.product_indices}")

            return result

        except Exception as e:
            if debug_lines is not None:
                debug_lines.append(f"⚠️ FOLLOWUP INTERPRET FAILED: {type(e).__name__}: {str(e)[:50]}")
            return None

    def _interpret_with_llm(
        self,
        query: str,
        products: List[Product],
        last_bot_response: Optional[str] = None,
        recent_bot_responses: Optional[list] = None,
    ) -> FollowupInterpretation:
        """Perform LLM-based interpretation."""
        client = get_openai_client()
        if not client:
            raise RuntimeError("OpenAI client not available")

        # Build products summary
        products_summary = self._build_products_summary(products)

        # Build conversation context for anaphoric reference resolution
        # This helps the LLM understand what "that", "this", "it" refer to
        conversation_context = ""
        if last_bot_response:
            # Search recent responses for last single-product mention
            # (handles cases where last response was an error with no SKUs)
            all_responses = [last_bot_response] + (recent_bot_responses or [])
            for response in all_responses:
                if not response:
                    continue
                mentioned = []
                for i, prod in enumerate(products, 1):
                    if prod.product_number and prod.product_number in response:
                        mentioned.append((i, prod.product_number))
                if len(mentioned) >= 1:
                    idx, sku = mentioned[0]
                    conversation_context = f"\n[Last discussed: #{idx} {sku}]\n"
                    break

            # If no SKU mentions found (e.g., after out_of_scope redirect), list products in context
            if not conversation_context and products:
                sku_list = ', '.join(f"#{i} {p.product_number}" for i, p in enumerate(products[:5], 1))
                conversation_context = f"\n[Products in context: {sku_list}]\n"

            # Still include truncated response for feature/topic references
            truncated = last_bot_response[-500:] if len(last_bot_response) > 500 else last_bot_response
            conversation_context += f"\nBot's last response:\n{truncated}\n"

        # Check if user's query mentions any product SKUs
        query_upper = query.upper()
        query_refs = []
        for i, prod in enumerate(products, 1):
            if prod.product_number and prod.product_number.upper() in query_upper:
                query_refs.append((i, prod.product_number))

        if query_refs:
            refs_str = ", ".join(f"#{idx} {sku}" for idx, sku in query_refs)
            conversation_context += f"\n[User's query mentions: {refs_str}]\n"

        # Prepare prompt
        prompt = FOLLOWUP_INTERPRETATION_PROMPT.format(
            products_summary=products_summary,
            query=query,
            conversation_context=conversation_context
        )

        # Call OpenAI
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": query}
            ],
            temperature=0.1,
            max_completion_tokens=200,
            response_format={"type": "json_object"},
            timeout=30.0
        )

        result = self._parse_response(response.choices[0].message.content)

        # Fix 0-based indexing: LLM sometimes uses 0-based despite 1-based prompt
        if result.product_indices and 0 in result.product_indices:
            result.product_indices = [i + 1 for i in result.product_indices]

        return result

    def _build_products_summary(self, products: List[Product]) -> str:
        """Build a summary of products for the prompt."""
        lines = []

        # Calculate length range for context (helps with "shorter"/"longer" requests)
        lengths_ft = [p.metadata.get('length_ft', 0) for p in products if p.metadata.get('length_ft')]
        if lengths_ft:
            min_len = min(lengths_ft)
            max_len = max(lengths_ft)
            if min_len == max_len:
                lines.append(f"[Current products are all {max_len}ft]")
            else:
                lines.append(f"[Current products range from {min_len}ft to {max_len}ft]")
            lines.append("")

        for i, prod in enumerate(products, 1):
            meta = prod.metadata
            sku = prod.product_number

            # Get key attributes
            length = meta.get('length_display', '')
            color = meta.get('color', '')
            features = meta.get('features', [])
            connectors = meta.get('connectors', [])

            parts = [f"#{i} {sku}"]
            if length:
                parts.append(f"length={length}")
            if color:
                parts.append(f"color={color}")
            if connectors and len(connectors) >= 2:
                parts.append(f"connectors={connectors[0]}→{connectors[1]}")
            if features:
                parts.append(f"features={','.join(features[:3])}")

            lines.append(" | ".join(parts))

        return "\n".join(lines)

    def _parse_response(self, response: str) -> FollowupInterpretation:
        """Parse LLM response into interpretation object."""
        data = json.loads(response)

        action_str = data.get('action', 'specs').lower()
        action_map = {
            'filter': FollowupAction.FILTER,
            'compare': FollowupAction.COMPARE,
            'details': FollowupAction.DETAILS,
            'specs': FollowupAction.SPECS,
            'compatibility': FollowupAction.COMPATIBILITY,
            'educational': FollowupAction.EDUCATIONAL,
            'recommend': FollowupAction.RECOMMEND,
            'recall': FollowupAction.RECALL,
            'clarify': FollowupAction.CLARIFY,
        }
        action = action_map.get(action_str, FollowupAction.SPECS)

        # Clean criteria — LLM sometimes returns string instead of dict, or wraps keys in quotes
        criteria = data.get('criteria')
        if isinstance(criteria, str):
            try:
                criteria = json.loads(criteria)
            except (json.JSONDecodeError, ValueError):
                criteria = None
        if isinstance(criteria, dict):
            criteria = {k.strip('"'): v for k, v in criteria.items()}

        return FollowupInterpretation(
            action=action,
            criteria=criteria,
            product_indices=data.get('product_indices'),
            reasoning=data.get('reasoning', ''),
            clarification_question=data.get('clarification_question'),
        )


def filter_products_by_criteria(
    products: List[Product],
    criteria: dict
) -> List[Product]:
    """
    Filter products based on interpreted criteria.

    Args:
        products: All products in context
        criteria: Filter criteria from LLM interpretation

    Returns:
        Filtered list of products
    """
    if not criteria:
        return products

    # Handle relative length comparison separately (shorter/longer)
    if 'length_comparison' in criteria:
        return _filter_by_relative_length(products, criteria['length_comparison'])

    filtered = []

    for prod in products:
        meta = prod.metadata
        matches = True

        # Length filter (meters)
        if 'length_m' in criteria:
            target_m = criteria['length_m']
            length_ft = meta.get('length_ft', 0)
            # Convert to meters for comparison (1m = 3.28ft)
            length_m = length_ft / 3.28 if length_ft else 0
            # ±15% tolerance for imperial/metric conversions
            tolerance_m = max(target_m * 0.15, 0.3)
            if abs(length_m - target_m) > tolerance_m:
                matches = False

        # Length filter (feet) — range or single value
        if 'length_ft' in criteria:
            target_ft = criteria['length_ft']
            length_ft = meta.get('length_ft', 0)
            if 'length_max_ft' in criteria:
                # Range query: between min and max with 15% tolerance
                max_ft = criteria['length_max_ft']
                if not (target_ft * 0.85 <= length_ft <= max_ft * 1.15):
                    matches = False
            else:
                # Single value: ±15% tolerance for imperial/metric conversions
                tolerance = max(target_ft * 0.15, 0.5)
                if abs(length_ft - target_ft) > tolerance:
                    matches = False

        # Color filter
        if 'color' in criteria:
            target_color = criteria['color'].lower()
            product_color = (meta.get('color', '') or '').lower()
            if target_color not in product_color:
                matches = False

        # Feature filter (4K, HDR, fiber type, etc.)
        if 'feature' in criteria:
            target_feature = criteria['feature'].lower()
            features = [f.lower() for f in meta.get('features', [])]
            content = (prod.content or '').lower()
            if not any(target_feature in f for f in features) and target_feature not in content:
                # Also search short metadata values (catches fiber_type, network_rating, etc.
                # but excludes long text fields like General_Specifications)
                meta_match = any(
                    isinstance(v, str) and len(v) < 50 and target_feature in v.lower()
                    for v in meta.values()
                )
                if not meta_match:
                    matches = False

        # Connector filter — handle both simple ("LC") and compound ("LC Duplex→SC Duplex") formats
        if 'connector' in criteria:
            target_conn = criteria['connector'].lower()
            connectors = [str(c).lower() for c in meta.get('connectors', [])]
            # Strip arrow notation ("LC Duplex→LC Duplex" → check each side)
            conn_parts = [p.strip() for p in target_conn.replace('→', ' ').replace('->', ' ').split() if p.strip()]
            # Also check fiber_duplex, fiber_connector for fiber-specific filtering
            duplex_mode = str(meta.get('fiber_duplex', '')).lower()
            fiber_conn = str(meta.get('fiber_connector', '')).lower()
            all_searchable = connectors + [duplex_mode, fiber_conn]
            if not any(any(part in s for part in conn_parts) for s in all_searchable if s):
                matches = False

        # Duplex/simplex filter (for fiber cables)
        if 'duplex' in criteria:
            target_duplex = str(criteria['duplex']).lower()
            fiber_duplex = str(meta.get('fiber_duplex', '')).lower()
            if target_duplex not in fiber_duplex:
                matches = False

        if matches:
            filtered.append(prod)

    return filtered


def _filter_by_relative_length(products: List[Product], comparison: str) -> List[Product]:
    """
    Filter products by relative length (shorter/longer than current).

    Args:
        products: Products currently in context
        comparison: "shorter" or "longer"

    Returns:
        Products matching the relative comparison, or empty list if none match
        (caller should handle empty list by doing a new search)
    """
    if not products:
        return []

    # Get all lengths
    lengths_ft = [(p, p.metadata.get('length_ft', 0)) for p in products]

    if comparison == 'shorter':
        # Find the max length in current products
        max_length = max(length for _, length in lengths_ft if length > 0)
        # Return products shorter than the max
        return [p for p, length in lengths_ft if 0 < length < max_length]
    elif comparison == 'longer':
        # Find the min length in current products
        min_length = min(length for _, length in lengths_ft if length > 0)
        # Return products longer than the min
        return [p for p, length in lengths_ft if length > min_length]

    return products


def format_filtered_products(products: List[Product], criteria: dict, original_count: int) -> str:
    """
    Format response for filtered products.

    Args:
        products: Filtered products
        criteria: The filter criteria used
        original_count: Original number of products before filtering

    Returns:
        Formatted response string
    """
    if not products:
        # No matches - explain what we looked for
        # Note: Auto-search in followup handler should find products before we get here.
        # If we're here, it means no products match even in the full catalog.
        criteria_desc = _describe_criteria(criteria)
        return (
            f"I couldn't find any {criteria_desc} in our catalog. "
            f"Would you like to try a different specification?"
        )

    if len(products) == original_count:
        # All match - confirm they all have the requested attribute
        criteria_desc = _describe_criteria(criteria)
        if original_count == 1:
            return f"Yes, this product is available in {criteria_desc}."
        return f"Yes, all {original_count} products are available in {criteria_desc}."

    # Some filtered - use LLM to generate proper product cards
    criteria_desc = _describe_criteria(criteria)
    try:
        from llm.llm_response_generator import generate_response, ResponseType
        response = generate_response(
            products=products[:5],
            query=f"Show me the {criteria_desc} option{'s' if len(products) != 1 else ''}",
            response_type=ResponseType.SEARCH_RESULTS,
        )
        if response:
            return response
    except Exception:
        pass

    # Fallback: simple listing if LLM fails
    lines = [f"Here {'is the' if len(products) == 1 else 'are the'} {criteria_desc}:"]
    lines.append("")
    for i, prod in enumerate(products[:5], 1):
        name = prod.metadata.get('name', prod.product_number)
        length = prod.metadata.get('length_display', '')
        line = f"{i}. **{prod.product_number}** - {name}"
        if length:
            line += f" ({length})"
        lines.append(line)
    return "\n".join(lines)


def _describe_criteria(criteria: dict) -> str:
    """Convert criteria dict to human-readable description."""
    if not criteria:
        return "product"

    parts = []
    if 'length_m' in criteria:
        parts.append(f"{criteria['length_m']}m")
    if 'length_ft' in criteria:
        parts.append(f"{criteria['length_ft']}ft")
    if 'length_comparison' in criteria:
        comparison = criteria['length_comparison']
        parts.append(f"{comparison}")
    if 'color' in criteria:
        parts.append(criteria['color'])
    if 'feature' in criteria:
        parts.append(criteria['feature'])
    if 'connector' in criteria:
        parts.append(criteria['connector'])

    if parts:
        return " ".join(parts) + " option" + ("s" if len(parts) > 1 else "")
    return "matching products"
