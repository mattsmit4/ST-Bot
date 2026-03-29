"""
Product Narrowing Handler for ST-Bot.

Replaces the vagueness check system. When a search returns > 5 products with
tied scores, this handler analyzes the pool, asks differentiating questions,
and filters until <= 5 products remain.

Like "21 questions" -- finds the attribute that splits the pool most evenly
and asks the user about it.
"""

import re

from handlers.base import BaseHandler, HandlerContext, HandlerResult
from core.models import PendingNarrowing


def start_narrowing(
    ctx: HandlerContext,
    product_pool: list,
    filters,
    original_query: str,
    total_count: int = None,
    is_fallback: bool = False,
) -> HandlerResult:
    """
    Initialize a narrowing flow when search returned too many tied products.

    Called from NewSearchHandler when _should_narrow() returns True.

    Args:
        ctx: Handler context
        product_pool: All products from search (tied scores)
        filters: SearchFilters used for the search
        original_query: The user's original query

    Returns:
        HandlerResult with narrowing question
    """
    from llm.llm_narrowing_analyzer import LLMNarrowingAnalyzer

    # Serialize filters to dict for persistence AND to inform narrowing LLM
    filter_dict = {}
    if filters:
        for field_name in ['product_category', 'connector_from', 'connector_to',
                           'length', 'features', 'port_count', 'color',
                           'screen_size', 'cable_type', 'kvm_video_type',
                           'bay_count', 'rack_height', 'drive_size',
                           'usb_version', 'thunderbolt_version',
                           'requested_refresh_rate', 'requested_power_wattage',
                           'requested_network_speed', 'min_monitors']:
            val = getattr(filters, field_name, None)
            if val:
                filter_dict[field_name] = val

    analyzer = LLMNarrowingAnalyzer()
    question_result = analyzer.analyze_and_question(
        products=product_pool,
        original_query=original_query,
        questions_asked=0,
        original_filters=filter_dict,
        is_fallback=is_fallback,
    )

    if not question_result:
        # LLM failed -- fall back to showing top 3
        ctx.add_debug("⚠️ NARROWING: LLM and code fallback both failed, showing top products")
        return _fallback_show_products(ctx, product_pool, original_query)

    # Create narrowing state
    narrowing = PendingNarrowing(
        original_query=original_query,
        original_filters=filter_dict,
        product_skus=[p.product_number for p in product_pool],
        product_pool=product_pool,
        questions_asked=1,
        last_attribute=question_result['attribute'],
        asked_attributes=[question_result['attribute']],
        last_options=question_result['options'],
        option_filters=question_result['option_filters'],
        is_fallback=is_fallback,
    )
    ctx.context.set_pending_narrowing(narrowing)

    # Format response
    count = question_result.get('total_accounted', len(product_pool))
    question_text = question_result['question']
    options = question_result['options']

    ctx.add_debug(f"🔍 NARROWING: {count} tied products, asking about '{question_result['attribute']}'")
    ctx.add_debug(f"🔍 NARROWING OPTIONS: {options}")

    # Build response with LLM-generated intro (falls back to generic if missing)
    intro = question_result.get('intro') or f"I found {count} products that match. Let me help you narrow it down."
    # Fix intro count if validation dropped unmatched products
    if count < len(product_pool):
        intro = re.sub(r'\b' + str(len(product_pool)) + r'\b', str(count), intro)
    response = f"{intro}\n\n{question_text}"
    for opt in options:
        response += f"\n- {opt}"

    return HandlerResult(
        response=response,
        save_pending_narrowing=True,
        products_found=count,
    )


def _fallback_show_products(ctx, product_pool, original_query):
    """Fall back to showing top 5 products when narrowing can't proceed."""
    display = product_pool[:5]

    # Deterministic product listing
    listing = _format_product_listing(display)

    # LLM for key differences only
    key_diff = None
    if len(display) > 1:
        from llm.llm_response_generator import generate_response, ResponseType
        key_diff = generate_response(
            products=display,
            query=original_query,
            response_type=ResponseType.COMPARISON,
        )

    response = listing
    if key_diff:
        response += f"\n\n**Key Differences:**\n{key_diff}"
    if len(product_pool) > 5:
        response += f"\n\nShowing 5 of {len(product_pool)} matching products. Ask to see more or narrow further!"

    return HandlerResult(
        response=response,
        products_to_set=product_pool,
        products_found=len(product_pool),
    )


def _format_product_listing(products):
    """Build deterministic product listing with key specs from tier1 config."""
    from config.category_columns import get_tier1_columns, get_field_label
    from config.unit_converter import format_measurement

    lines = []
    for i, p in enumerate(products, 1):
        meta = p.metadata
        category = meta.get('category', '')
        name = meta.get('name', '')
        # Avoid showing SKU as name
        if not name or name == p.product_number:
            sub_cat = meta.get('sub_category', category)
            name = sub_cat if sub_cat else p.product_number

        lines.append(f"**{i}. SKU: {p.product_number}** - {name}")

        # Add connectors if available
        connectors = meta.get('connectors')
        if connectors and isinstance(connectors, list) and len(connectors) >= 2:
            lines.append(f"   - Connectors: {connectors[0]} to {connectors[1]}")

        # Add length if available
        length = meta.get('length_ft')
        if length:
            length_m = meta.get('length_m', '')
            if length_m:
                lines.append(f"   - Length: {length} ft [{length_m} m]")
            else:
                lines.append(f"   - Length: {length} ft")

        # Add tier1 fields with clean labels (skip fields already shown above)
        skip_fields = {'cable_length_raw'}  # Already shown as Length
        columns = get_tier1_columns(category)
        for field in columns:
            if field in skip_fields:
                continue
            value = meta.get(field)
            if value:
                if isinstance(value, float) and value.is_integer():
                    value = int(value)
                formatted = format_measurement(str(value), field)
                label = get_field_label(field, category)
                lines.append(f"   - {label}: {formatted}")

        lines.append("")  # blank line between products

    return "\n".join(lines)


class NarrowingResponseHandler(BaseHandler):
    """
    Handles user responses to narrowing questions.

    Parses the response, filters the product pool, and either:
    - Shows results if pool is <= 5
    - Asks another narrowing question if pool is still > 5
    - Falls back to top 5 if max questions reached or LLM fails
    """

    def handle(self, ctx: HandlerContext) -> HandlerResult:
        """Process a response to a narrowing question."""
        narrowing = ctx.context.pending_narrowing
        if not narrowing:
            ctx.add_debug("⚠️ NarrowingResponseHandler called without pending narrowing")
            return HandlerResult(
                response="I'm not sure what you're referring to. What products are you looking for?"
            )

        ctx.add_debug(f"🔍 NARROWING RESPONSE: '{ctx.query}'")
        ctx.add_debug(f"🔍 POOL SIZE: {len(narrowing.product_pool)}, questions asked: {narrowing.questions_asked}")

        # Check if intent already identified this as "show all"
        if ctx.intent.meta_info and ctx.intent.meta_info.get('show_all'):
            ctx.add_debug(f"🔍 NARROWING: User chose 'show all', showing all {len(narrowing.product_pool)} products")
            return self._show_results(ctx, narrowing, narrowing.product_pool, show_all=True)

        # Parse user's answer and filter the pool
        from llm.llm_narrowing_analyzer import LLMNarrowingAnalyzer
        analyzer = LLMNarrowingAnalyzer()

        filter_result = analyzer.parse_response_and_filter(
            user_response=ctx.query,
            product_pool=narrowing.product_pool,
            last_options=narrowing.last_options,
            option_filters=narrowing.option_filters,
        )

        if not filter_result:
            # Couldn't parse as a narrowing option — try extracting a filter from the response
            # (e.g., user said "1 to 2 meters" during a connector question)
            try:
                from llm.llm_followup_interpreter import filter_products_by_criteria
                extraction = ctx.filter_extractor.extract(ctx.query)
                if extraction and extraction.filters:
                    f = extraction.filters
                    criteria = {}
                    # Single length value
                    if f.length:
                        unit = f.length_unit or 'ft'
                        if unit in ('m', 'meters'):
                            criteria['length_m'] = f.length
                        else:
                            criteria['length_ft'] = f.length
                    # Range query (e.g., "1 to 2 meters")
                    elif f.length_min is not None and f.length_max is not None:
                        unit = f.length_unit or 'ft'
                        conv = 3.28084 if unit in ('m', 'meters') else 1.0
                        min_ft = f.length_min * conv
                        max_ft = f.length_max * conv
                        # Filter pool by range directly
                        range_filtered = [
                            p for p in narrowing.product_pool
                            if min_ft * 0.85 <= (p.metadata.get('length_ft') or 0) <= max_ft * 1.15
                        ]
                        if range_filtered and len(range_filtered) < len(narrowing.product_pool):
                            ctx.add_debug(f"⚠️ NARROWING: Parsed as range: {f.length_min}-{f.length_max}{unit}, {len(range_filtered)} match")
                            ctx.context.clear_pending_narrowing()
                            return self._show_results(ctx, narrowing, range_filtered[:5])
                    if f.connector_from:
                        criteria['connector'] = f.connector_from
                    if f.features:
                        criteria['feature'] = f.features[0]
                    if f.port_count:
                        criteria['port_count'] = f.port_count
                    if f.color:
                        criteria['color'] = f.color
                    if f.min_monitors:
                        criteria['min_monitors'] = f.min_monitors
                    if criteria:
                        filtered = filter_products_by_criteria(narrowing.product_pool, criteria)
                        if filtered and len(filtered) < len(narrowing.product_pool):
                            ctx.add_debug(f"⚠️ NARROWING: Parsed as filter: {criteria}, {len(filtered)} match")
                            ctx.context.clear_pending_narrowing()
                            return self._show_results(ctx, narrowing, filtered[:5])
            except Exception:
                pass
            ctx.add_debug("⚠️ NARROWING: Couldn't parse response, showing top 3")
            return self._show_results(ctx, narrowing, narrowing.product_pool[:3])

        new_pool = filter_result['filtered_pool']
        selected = filter_result['selected']

        ctx.add_debug(f"🔍 NARROWING: Selected '{selected}', pool {len(narrowing.product_pool)} → {len(new_pool)}")

        # "Show all" -- show all remaining products
        if selected == 'show_all':
            ctx.add_debug(f"🔍 NARROWING: User chose 'show all', showing all {len(new_pool)} products")
            return self._show_results(ctx, narrowing, new_pool, show_all=True)

        # Pool is now <= 5 -- show results
        if len(new_pool) <= 5:
            ctx.add_debug(f"🔍 NARROWING: Pool narrowed to {len(new_pool)}, showing results")
            return self._show_results(ctx, narrowing, new_pool)

        # Max questions reached -- show all remaining
        if narrowing.questions_asked >= narrowing.max_questions:
            ctx.add_debug(f"🔍 NARROWING: Max questions ({narrowing.max_questions}) reached, showing {len(new_pool)}")
            return self._show_results(ctx, narrowing, new_pool, show_all=True)

        # Ask another narrowing question (exclude attributes already asked + original filters)
        question_result = analyzer.analyze_and_question(
            products=new_pool,
            original_query=narrowing.original_query,
            questions_asked=narrowing.questions_asked,
            exclude_attributes=narrowing.asked_attributes,
            original_filters=narrowing.original_filters,
        )

        if not question_result:
            # No useful differentiating question found -- show all remaining
            ctx.add_debug(f"⚠️ NARROWING: No useful follow-up question, showing all {len(new_pool)}")
            return self._show_results(ctx, narrowing, new_pool, show_all=True)

        # Update narrowing state
        narrowing.product_pool = new_pool
        narrowing.product_skus = [p.product_number for p in new_pool]
        narrowing.questions_asked += 1
        narrowing.last_attribute = question_result['attribute']
        narrowing.asked_attributes.append(question_result['attribute'])
        narrowing.last_options = question_result['options']
        narrowing.option_filters = question_result['option_filters']

        remaining = question_result.get('total_accounted', len(new_pool))
        question_text = question_result['question']
        options = question_result['options']

        ctx.add_debug(f"🔍 NARROWING: {remaining} remaining, asking about '{question_result['attribute']}'")

        response = f"That narrows it to {len(new_pool)} products.\n\n{question_text}"
        for opt in options:
            response += f"\n- {opt}"

        return HandlerResult(
            response=response,
            save_pending_narrowing=True,
        )

    def _show_results(self, ctx, narrowing, products, show_all=False):
        """Show the final narrowed product list using hybrid code+LLM approach."""
        ctx.context.clear_pending_narrowing()

        # Only cap display when narrowing ran out of questions (not user's "show all")
        if show_all or len(products) <= 5:
            display = products
        else:
            display = products[:5]

        from llm.llm_response_generator import generate_response, ResponseType

        if len(display) == 1:
            # Single product: PRODUCT_DETAILS prompt (designed for single-product deep dive)
            # with tier 1 data — concise confirmation after narrowing conversation.
            # User can ask followup questions for deeper specs (those use tier 3).
            response = generate_response(
                products=display,
                query=narrowing.original_query,
                response_type=ResponseType.PRODUCT_DETAILS,
                context={'tier_override': 1},
            )
        else:
            # Multi-product: deterministic listing + LLM for key differences
            listing = _format_product_listing(display)
            intro = f"Here are {len(display)} products:\n\n"

            context = {}
            if getattr(narrowing, 'is_fallback', False):
                context['is_fallback'] = True
                context['original_filters'] = narrowing.original_filters
            key_diff = generate_response(
                products=display,
                query=narrowing.original_query,
                response_type=ResponseType.COMPARISON,
                context=context if context else None,
            )

            response = intro + listing
            if key_diff:
                response += f"\n\n{key_diff}"
        if not show_all and len(products) > 5:
            response += f"\n\nShowing 5 of {len(products)} matching products. Ask to see more or narrow further!"

        return HandlerResult(
            response=response,
            products_to_set=products,  # Save ALL to context
            filters_for_logging=narrowing.original_filters,
            clear_pending_narrowing=True,
            products_found=len(products),
        )

