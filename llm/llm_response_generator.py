"""
LLM-Based Response Generator for ST-Bot.

Generates natural, contextually appropriate responses using LLM + product metadata.
Replaces hardcoded templates in formatters/*, response_builder, followup_handler.

Usage:
    generator = LLMResponseGenerator()
    response = generator.generate_response(
        products=products,
        query="I need HDMI cables",
        response_type=ResponseType.SEARCH_RESULTS
    )
"""

import os
import json
import logging
import re
from typing import List, Optional, TYPE_CHECKING
from enum import Enum

from core.openai_client import get_openai_client
from config.category_columns import get_tier1_columns, get_tier2_columns, get_tier3_columns, get_field_label
from config.unit_converter import format_measurement

if TYPE_CHECKING:
    from core.models import Product


class ResponseType(Enum):
    """Types of responses the generator can produce."""
    SEARCH_RESULTS = "search_results"      # Show products matching search
    COMPARISON = "comparison"              # Compare 2+ products
    RECOMMENDATION = "recommendation"      # Recommend best product for use case
    PRODUCT_DETAILS = "product_details"    # Deep dive on one product
    SPECS_TABLE = "specs_table"           # Show specs for all products
    COMPATIBILITY = "compatibility"        # Yes/no feature check for all products
    FILTER_RESULTS = "filter_results"     # Filtered subset of shown products
    TWO_PRODUCT_SOLUTION = "two_product_solution"  # Adapter + long cable combo
    SPECIFIC_QUESTION = "specific_question"        # Answer a specific question concisely


# Response type specific instructions
RESPONSE_TYPE_INSTRUCTIONS = {
    ResponseType.SEARCH_RESULTS: """
Show ALL provided products as a numbered list with relevant specs.
Format each product as:
**1. SKU** - Brief description
   Key specs relevant to user's query (e.g., length, features, connectors)

Important: Always include every product provided — never omit any.

If there are 2+ products, add a brief "Key Differences" summary highlighting the most notable
differences between them. Prioritize differences relevant to the User Query above
(e.g., if the user asked about fiber distance, highlight distance differences first).
Focus on specs that actually differ — skip anything identical across all products.
Do NOT repeat the same difference using different field names (e.g., "Length" and "Cable Length" are the same thing — pick one).
If only one difference exists, just mention that one — don't pad.
If there is only 1 product, skip the Key Differences section entirely.""",

    ResponseType.COMPARISON: """
Compare the products highlighting similarities and differences.
Do NOT include an introductory sentence or greeting — the product listing already provides context.
Structure:
1. Key similarities (1-2 sentences)
2. Key differences (the 2-3 MOST important differences only — not every spec)
3. "Bottom line" recommendation based ONLY on listed specs

Be CONCISE. For 5+ products, group by distinguishing trait (e.g., "4K 60Hz models: X, Y"
vs "4K 30Hz models: Z, W") rather than listing every product individually.
Focus on specs that actually differ between the products.
Only reference specifications explicitly shown in the product data. Do not infer or assume capabilities not listed.""",

    ResponseType.RECOMMENDATION: """
FIRST check if the products can actually meet the user's requirements. If a requirement is clearly
impossible for this product type (e.g., charging/ethernet on an HDMI splitter), say so immediately
in 1-2 sentences and suggest the right product type instead. Do NOT try to pick a "best match"
when no product meets the core requirements.

Only if products CAN meet the requirements: lead with your pick and why in 2-3 sentences,
then briefly note which requirements are met vs not met.
Do NOT include an "alternative" or "runner-up" section.
Base your reasoning on the user's stated criteria, not on specs they didn't ask about.""",

    ResponseType.PRODUCT_DETAILS: """
If the user asked a specific question about the product, answer it directly first in a brief sentence or two before showing the full details.

Provide detailed information about the product(s), organized into logical spec groups.

Format each group as:

### [Group Name]
- **[Spec Name]**: [Value]
- **[Spec Name]**: [Value]

Replace [Spec Name] with the actual specification name (e.g., "Displays Supported", "Power Delivery"). Do NOT use the word "Spec" as a label.

Use groups like:
- Display & Video (resolution, number of displays, 4K support, standards)
- Connectivity (ports, USB types, host connector, interfaces)
- Power (power delivery, wattage, charging)
- Networking (speed, cable type, PXE)
- Physical (dimensions, weight, color)
- Package & Warranty (package dimensions, weight, quantity, included items, warranty)
- Compatibility (OS, system requirements)

Only include groups that have relevant data for this product.
Do NOT write sentences or paragraphs — use the bullet-point format shown above.
Start with a one-line product summary, then go straight into spec groups.

End with a brief question that helps the user decide (e.g., what they're connecting, or which feature matters most) — not yes/no questions about showing more specs.""",

    ResponseType.SPECS_TABLE: """
Show specifications for all products.
Use a clear format - either markdown table or structured list.
Highlight any notable differences between products.""",

    ResponseType.COMPATIBILITY: """
Answer the compatibility/feature question for each product.
Format as:
- **SKU**: Yes/No - brief reason

Be clear about which products have the feature and which don't.""",

    ResponseType.FILTER_RESULTS: """
Show the filtered products that match the user's criteria.
Format as numbered list with relevant specs.
Mention how many products matched vs total shown before.""",

    ResponseType.TWO_PRODUCT_SOLUTION: """
A single cable doesn't exist at the requested length. The solution is two products used together.

Structure your response:
1. Briefly explain why a single cable isn't available at this length
2. Present the two-product solution — adapter first, then the long cable
3. Explain how they connect: [source device] → [adapter SKU] → [cable SKU] → [display/destination]
4. Key specs for each product (connectors, length, notable features)
5. End with a question to confirm this solves their setup (or if they need anything else)

Be practical and solution-oriented. The user needs a complete, working setup.""",

    ResponseType.SPECIFIC_QUESTION: """
Answer the user's specific question about the product(s) directly and concisely.
Carefully check ALL provided data fields — the answer may be under a related field name
(e.g., "power_adapter" answers questions about included power supplies).

CRITICAL for multiple products: Check the relevant data field in EACH product individually.
- If ALL products share the SAME value: state it ONCE (e.g., "All 20 cables support 100Gbps"). Do NOT list each product — just give the single answer.
- If values DIFFER between products: list each product with its specific value.

Do NOT list all specs — just answer what was asked.
When citing evidence, lead with the most relevant data points — e.g., for OS compatibility,
mention the newest supported versions first, not the oldest.
If the answer isn't in the data, consider whether the spec even applies to this product type.
If it doesn't apply (e.g., power delivery on a passive cable), briefly explain why.
If it should apply but is missing, say the data isn't available.
Do not end with offers to show more details — the user will ask if they want more.""",
}


RESPONSE_PROMPT = '''You are a helpful product expert for StarTech.com, a computer peripherals company.

## Products
{products_json}

## User Query
"{query}"

## Response Type
{response_type}

## Context
{context}

## Instructions
{type_instructions}

Important: The products listed above are the ones the user is asking about. If the user refers to products by number (e.g., "product 6", "the first one", "#2"), those references have already been resolved — the products below ARE the ones they mean. Do not say a product doesn't exist.

General guidelines:
- Use **bold** for SKUs and key terms
- Keep responses concise but complete
- Be conversational and helpful
- Only ask a follow-up question when it helps the user make a buying decision (e.g., "Do you need PoE?" or "What screen size is your monitor?"). Do NOT ask if they want more spec details — they'll ask if they do. Do NOT offer information that isn't in the product data above.
- Never make up specs not in the product data
- Never speculate or guess — only state facts from the provided data. If unsure, omit it.
- Never mention price, cost, or budget — we do not have pricing data
- If a product lacks data for a spec, don't mention that spec
- When showing measurements in mm, also include converted units: e.g., "5000 mm (16.4 ft / 5 m)". For product dimensions, use cm and inches.
- Always include every product provided in the data — never skip or omit any. {product_count_note}

Respond in markdown format.'''


class LLMResponseGenerator:
    """
    Generates natural responses using LLM + product metadata.

    This replaces the hardcoded templates in formatters/*, response_builder,
    and followup_handler with a unified LLM-based approach.
    """

    def __init__(self, model: str = None):
        """
        Initialize the response generator.

        Args:
            model: OpenAI model to use (default: gpt-4o-mini or env var)
        """
        self.model = model or os.environ.get('OPENAI_CHAT_MODEL', 'gpt-5.4-nano')

    def generate_response(
        self,
        products: List["Product"],
        query: str,
        response_type: ResponseType,
        context: Optional[dict] = None
    ) -> str:
        """
        Generate a response using LLM with product metadata.

        Args:
            products: Products to include in response
            query: User's original query
            response_type: What kind of response to generate
            context: Optional context (dropped_filters, use_case, etc.)

        Returns:
            Generated response string
        """
        if not products:
            return self._handle_no_products(query, response_type, context)

        # Build prompt - tier1 for quick overview, tier2 for specs/differences, tier3 for full detail
        use_tier3 = response_type in (
            ResponseType.PRODUCT_DETAILS, ResponseType.SPECS_TABLE,
        )
        use_tier2 = response_type in (
            ResponseType.SEARCH_RESULTS, ResponseType.COMPARISON,
            ResponseType.COMPATIBILITY, ResponseType.TWO_PRODUCT_SOLUTION,
            ResponseType.FILTER_RESULTS,
            ResponseType.RECOMMENDATION,
            ResponseType.SPECIFIC_QUESTION,
        )
        tier = 3 if use_tier3 else (2 if use_tier2 else 1)
        if context and 'tier_override' in context:
            override = context['tier_override']
            if override in (1, 2, 3):
                tier = override
        products_json = self._serialize_products(products, tier=tier)
        context_str = self._format_context(context)
        type_instructions = RESPONSE_TYPE_INSTRUCTIONS.get(
            response_type,
            "Provide a helpful response about these products."
        )

        # For fallback comparisons, replace the template entirely — the standard
        # COMPARISON structure (intro + similarities + bottom line) is too verbose
        # and fights with the fallback context.
        if context and context.get('is_fallback') and response_type == ResponseType.COMPARISON:
            type_instructions = (
                "These are fallback alternatives — not exact matches for the user's query.\n"
                "IMPORTANT: Follow ONLY these instructions. Ignore the General guidelines section below entirely.\n\n"
                "STRICT OUTPUT FORMAT:\n"
                "Line 1: One sentence stating which requirements from the User Query above "
                "are NOT met by these products based on the available data. "
                "If a requirement cannot be verified from the data (field not present), "
                "say the information is not available — do NOT claim it is unsupported.\n"
                "Then: Bullet points showing ONLY specs that DIFFER between products.\n\n"
                "Each bullet MUST use this exact format:\n"
                "- **Cable Length**: SKU1 is 3.3 ft; SKU2 is 1 ft\n"
                "- **Port Count**: SKU1 has 7 ports; SKU2 has 10\n\n"
                "EXAMPLE of complete output:\n"
                "These products don't include an SD card reader; Cat6 compatibility info is not available in the data.\n\n"
                "- **USB-C Ports**: ABC-HUB has 2 USB-C ports; XYZ-HUB has none\n"
                "- **Power Delivery**: ABC-HUB supports 100W PD pass-through; XYZ-HUB does not\n\n"
                "Rules:\n"
                "- If a spec has the SAME value for all products, it is NOT a difference — omit it entirely, "
                "even if it seems important\n"
                "- Do NOT create per-product sections or use SKUs as headers\n"
                "- ONLY include differences a customer would use to choose between these products: "
                "ports/connectivity, power delivery, display support, or features "
                "directly mentioned in the user's query\n"
                "- NEVER include: chipset, operating temperature, certifications, MTBF, ESD ratings, "
                "material, weight, dimensions, packaging, humidity, storage temperature, "
                "OS compatibility, driver requirements\n"
                "- Keep to 3 bullet points max — even 1 bullet is fine if only 1 meaningful difference exists\n"
                "- Do NOT include section headers, intro paragraphs, or 'Bottom line'\n"
                "- Do NOT end with a question or suggestion of any kind\n"
                "- The product listing above already shows full specs — be concise\n\n"
                "CRITICAL: Every difference line MUST begin with '- **' — no exceptions."
            )

        # For fallback results on the direct path (≤5 products, no narrowing),
        # prepend instructions so the LLM calls out mismatched requirements.
        if context and context.get('is_fallback') and response_type == ResponseType.SEARCH_RESULTS:
            type_instructions = (
                "These are fallback/alternative results — not exact matches for the user's query.\n"
                "Start with one sentence noting which specific requirements from the User Query "
                "are NOT met by these products based on the available data. "
                "If a requirement cannot be verified (no field for it in the data), "
                "say the info is not available — do NOT claim it is unsupported.\n"
                "If one product is clearly the best match, recommend it and explain why.\n"
                "Then show the product list as normal.\n\n"
            ) + type_instructions

        if context and context.get('selected_from_total'):
            total = context['selected_from_total']
            n = len(products)
            product_count_note = (
                f"The user selected {'this product' if n == 1 else f'these {n} products'} "
                f"from {total} currently shown. Answer about {'it' if n == 1 else 'them'} only."
            )
        else:
            product_count_note = (
                f"There are {len(products)} products — "
                f"your response MUST include all {len(products)}"
            )

        prompt = RESPONSE_PROMPT.format(
            products_json=products_json,
            query=query,
            response_type=response_type.value,
            context=context_str,
            type_instructions=type_instructions,
            product_count_note=product_count_note
        )

        # Call LLM
        return self._call_llm(prompt)

    def _serialize_products(self, products: List["Product"], tier: int = 1) -> str:
        """
        Convert products to JSON with relevant metadata.

        Uses CATEGORY_COLUMNS config to include category-specific fields.
        Tier 1 = key overview fields, Tier 2 = all relevant fields.
        """
        data = []
        for i, p in enumerate(products, 1):
            meta = p.metadata
            category = meta.get('category', '')
            product_data = {
                'position': i,
                'sku': p.product_number,
                'category': category,
            }
            # Only include name if it differs from SKU (avoids duplicate display)
            name = meta.get('name', '')
            if name and name != p.product_number:
                product_data['name'] = name

            # Always include sub_category for product type context
            sub_cat = meta.get('sub_category')
            if sub_cat:
                product_data['sub_category'] = sub_cat

            # Always include derived connector types (ground truth from search engine)
            connectors = meta.get('connectors')
            if connectors and isinstance(connectors, list) and len(connectors) >= 2:
                product_data['connector_from'] = connectors[0]
                product_data['connector_to'] = connectors[1]

            # Get category-specific fields from config
            if tier == 3:
                columns = get_tier3_columns(category)
            elif tier == 2:
                columns = get_tier2_columns(category)
            else:
                columns = get_tier1_columns(category)
            for field in columns:
                value = meta.get(field)
                if value:
                    if isinstance(value, float) and value.is_integer():
                        value = int(value)
                    label = get_field_label(field, category)
                    product_data[label] = format_measurement(str(value), field)

            # Add screen size for privacy screens (extracted from SKU pattern)
            # Prevents LLM hallucination about screen sizes
            if meta.get('category') == 'privacy_screen':
                sku = p.product_number
                # Pattern: 135CT → 13.5", 156W → 15.6", 24MAM → 24"
                match = re.match(r'^(\d{3})', sku)
                if match:
                    digits = match.group(1)
                    product_data['screen_size'] = f"{digits[0]}{digits[1]}.{digits[2]} inch"
                else:
                    match = re.match(r'^(\d{2})', sku)
                    if match:
                        product_data['screen_size'] = f"{match.group(1)} inch"

            # Include product description if available
            # Prefer meta['description'] (product title with port/feature breakdown)
            # over p.content (generic generated summary)
            desc = meta.get('description', '') or ''
            if desc:
                product_data['description'] = desc[:300]
            elif p.content:
                product_data['description'] = p.content[:300]

            data.append(product_data)

        return json.dumps(data, indent=2)

    def _format_context(self, context: Optional[dict]) -> str:
        """Format context dict into readable string for prompt."""
        if not context:
            return "None"

        parts = []

        # Handle dropped filters
        if context.get('dropped_filters'):
            dropped = context['dropped_filters']
            if isinstance(dropped, list) and dropped:
                # Convert DroppedFilter objects to readable strings
                dropped_strs = []
                for d in dropped:
                    if hasattr(d, 'filter_name') and hasattr(d, 'requested_value'):
                        dropped_strs.append(f"{d.filter_name}={d.requested_value}")
                    else:
                        dropped_strs.append(str(d))
                if dropped_strs:
                    parts.append(f"Filters we couldn't match: {', '.join(dropped_strs)}")
            elif dropped:
                parts.append(f"Filters we couldn't match: {dropped}")

        # Handle original filters
        if context.get('original_filters'):
            filters = context['original_filters']
            filter_parts = []
            field_labels = {
                'product_category': 'category',
                'connector_from': 'host connection',
                'connector_to': 'output connection',
                'port_count': 'minimum ports',
                'keywords': 'must-have features',
                'features': 'features',
                'length': 'length (ft)',
                'color': 'color',
                'min_monitors': 'minimum monitors',
                'cable_type': 'cable type',
                'screen_size': 'screen size',
                'usb_version': 'USB version',
                'thunderbolt_version': 'Thunderbolt version',
                'requested_refresh_rate': 'refresh rate',
                'requested_power_wattage': 'power wattage',
                'requested_network_speed': 'network speed',
            }
            if isinstance(filters, dict):
                for key, label in field_labels.items():
                    val = filters.get(key)
                    if val:
                        if isinstance(val, list):
                            filter_parts.append(f"{label}={', '.join(str(v) for v in val)}")
                        else:
                            filter_parts.append(f"{label}={val}")
            elif hasattr(filters, '__dict__'):
                for attr, label in field_labels.items():
                    val = getattr(filters, attr, None)
                    if val:
                        if isinstance(val, list):
                            filter_parts.append(f"{label}={', '.join(str(v) for v in val)}")
                        else:
                            filter_parts.append(f"{label}={val}")
            if filter_parts:
                parts.append(f"User originally requested: {', '.join(filter_parts)}")

        # Handle use case
        if context.get('use_case'):
            parts.append(f"Use case: {context['use_case']}")

        # Handle feature being checked
        if context.get('feature'):
            parts.append(f"Feature to check: {context['feature']}")

        # Handle alternatives for existence NO responses
        if context.get('alternatives'):
            alts = context['alternatives']
            if isinstance(alts, list) and alts:
                parts.append(f"Alternative products we DO carry: {'; '.join(alts)}")

        if context.get('selected_from_total'):
            refs = ', '.join(context.get('user_product_refs', []))
            parts.append(
                f"The user selected {refs} from {context['selected_from_total']} products currently shown. "
                f"Answer about the selected product(s) only — do not comment on the list size."
            )

        return '\n'.join(parts) if parts else "None"

    def _handle_no_products(
        self,
        query: str,
        response_type: ResponseType,
        context: Optional[dict]
    ) -> str:
        """Handle case when no products are provided."""
        if response_type == ResponseType.SEARCH_RESULTS:
            return (
                "I couldn't find any products matching your search. "
                "Could you try different terms or let me know what you're looking for?"
            )
        return "I don't have any products to show right now. What are you looking for?"

    def _call_llm(self, prompt: str) -> str:
        """Call OpenAI API to generate response."""
        client = get_openai_client()
        if not client:
            return "I'm having trouble generating a response right now. Please try again."

        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful product expert for StarTech.com."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_completion_tokens=3000,
                timeout=30.0
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logging.getLogger(__name__).error(f"LLM call failed: {e}")
            return f"I found some products but had trouble formatting the response. Please try again."


# Singleton instance for convenience
_generator = None


def get_response_generator() -> LLMResponseGenerator:
    """Get or create singleton response generator."""
    global _generator
    if _generator is None:
        _generator = LLMResponseGenerator()
    return _generator


def generate_response(
    products: List["Product"],
    query: str,
    response_type: ResponseType,
    context: Optional[dict] = None
) -> str:
    """
    Convenience function to generate a response.

    Args:
        products: Products to include
        query: User query
        response_type: Type of response
        context: Optional context

    Returns:
        Generated response string
    """
    generator = get_response_generator()
    return generator.generate_response(products, query, response_type, context)
