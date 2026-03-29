"""
SKU lookup handler for ST-Bot.

Handles SPECIFIC_SKU intent - looks up products by SKU/product number.
"""

import re
from handlers.base import BaseHandler, HandlerContext, HandlerResult


class SkuHandler(BaseHandler):
    """Handle direct SKU/product number lookups."""

    # SKU pattern: alphanumeric with optional hyphens, 3+ chars
    SKU_PATTERN = re.compile(r'\b([A-Za-z0-9][\w\-]{2,})\b')

    def handle(self, ctx: HandlerContext) -> HandlerResult:
        # Get SKU from intent (classifier already extracted it)
        sku = ctx.intent.sku

        # If no SKU from intent, try to extract from query
        if not sku:
            sku = self._extract_sku_from_query(ctx.query)

        if not sku:
            return HandlerResult(
                response="I couldn't identify a product SKU in your query. "
                         "Please provide the full product number (e.g., HD2MM2M or USB31HD4K60)."
            )

        ctx.add_debug(f"🔍 SKU LOOKUP: {sku}")

        # Search for the SKU in all products
        matching_products = []
        sku_upper = sku.upper()

        for product in ctx.all_products:
            product_sku = product.product_number.upper()
            if product_sku == sku_upper or product_sku.startswith(sku_upper):
                matching_products.append(product)

        if not matching_products:
            ctx.add_debug(f"🔍 SKU LOOKUP: No products found for {sku}")
            return HandlerResult(
                response=f"I couldn't find a product with SKU **{sku}**. "
                         f"Please check the SKU and try again, or describe what you're looking for."
            )

        ctx.add_debug(f"🔍 SKU LOOKUP: Found {len(matching_products)} products")

        # If exact match, show that product with details
        exact_match = next(
            (p for p in matching_products if p.product_number.upper() == sku_upper),
            None
        )

        if exact_match:
            from llm.llm_response_generator import generate_response, ResponseType
            # If query is more than just the SKU, user is asking about it
            sku_only = ctx.query.strip().upper() == exact_match.product_number.upper()
            rtype = ResponseType.PRODUCT_DETAILS if sku_only else ResponseType.SPECIFIC_QUESTION
            response = generate_response(
                products=[exact_match],
                query=ctx.query,
                response_type=rtype,
            )
            # Prepend SKU product to existing context instead of replacing
            existing = ctx.context.current_products or []
            merged = [exact_match] + [p for p in existing if p.product_number != exact_match.product_number]
            return HandlerResult(
                response=response,
                products_to_set=merged
            )

        # Multiple partial matches - show list
        response_parts = [f"I found {len(matching_products)} products matching **{sku}**:", ""]
        for i, product in enumerate(matching_products[:5], 1):
            name = product.metadata.get('name', product.product_number)
            response_parts.append(f"{i}. **{product.product_number}** - {name}")

        response_parts.append("")
        response_parts.append("Which one would you like to know more about?")

        # Prepend matched products to existing context instead of replacing
        new_products = matching_products[:5]
        new_skus = {p.product_number for p in new_products}
        existing = ctx.context.current_products or []
        merged = new_products + [p for p in existing if p.product_number not in new_skus]
        return HandlerResult(
            response="\n".join(response_parts),
            products_to_set=merged
        )

    def _extract_sku_from_query(self, query: str) -> str | None:
        """Try to extract a SKU pattern from the query text."""
        matches = self.SKU_PATTERN.findall(query)
        # Filter out common words that match the pattern
        stop_words = {
            'the', 'and', 'for', 'can', 'you', 'show', 'tell', 'about',
            'what', 'how', 'does', 'this', 'that', 'with', 'from',
            'have', 'has', 'are', 'was', 'were', 'been', 'being',
            'need', 'want', 'looking', 'find', 'search', 'info',
            'details', 'product', 'sku', 'number', 'model',
        }
        for match in matches:
            if match.lower() not in stop_words and len(match) >= 3:
                return match
        return None
