"""
Search intent handlers - Simplified MVP.

Handles new product searches with simple, reliable logic.
Includes LLM fallback for complex queries when rule-based extraction fails.

Query Routing Strategy:
- Simple queries -> Rules (fast path)
- Complex queries -> LLM (directly, bypassing rules)

This avoids "wrong results" when rules extract something incorrect but
search still returns products. The LLM understands complex, multi-part
queries that rules struggle with.
"""

import re
from dataclasses import replace
from handlers.base import BaseHandler, HandlerContext, HandlerResult
from core.models import SearchFilters, Product


from config import get_config_value

# LLM-based product verification (port types, refresh rate, power delivery)
USE_LLM_VERIFICATION = get_config_value('USE_LLM_VERIFICATION') == 'true'

# Patterns for detecting query corrections (e.g., "Actually, I meant DisplayPort")
CORRECTION_PATTERNS = [
    r'\b(?:actually|wait|no|sorry|oops)\b',
    r'\bi\s+meant\b',
    r'\binstead\s+of\b',
    r'\bnot\s+(?:hdmi|displayport|usb|vga|thunderbolt|ethernet)\b',
    r'\b(?:wrong|different)\s+(?:type|connector|cable)\b',
]


class NewSearchHandler(BaseHandler):
    """Handle new product search queries."""

    # Categories that use connector gates (benefit from connector relaxation)
    CONNECTOR_CATEGORIES = {'cables', 'cable', 'adapters', 'adapter'}

    # Category -> key SearchFilters attribute to preserve in category-only fallback
    CATEGORY_KEY_FILTERS = {
        'docks': 'min_monitors', 'dock': 'min_monitors',
        'display mounts': 'min_monitors', 'display_mount': 'min_monitors',
        'kvm switches': 'kvm_video_type', 'kvm_switch': 'kvm_video_type',
        'kvm_extender': 'kvm_video_type',
        'storage enclosures': 'bay_count', 'storage_enclosure': 'bay_count',
        'server racks': 'rack_height', 'rack': 'rack_height',
        'privacy screens': 'screen_size', 'privacy_screen': 'screen_size',
        'hubs': 'connector_from', 'hub': 'connector_from',
        'video_splitter': 'connector_from', 'video_switch': 'connector_from',
    }

    # When the primary key filter is empty, try these semantically related fields
    KEY_FILTER_FALLBACKS = {
        'kvm_video_type': ['connector_to', 'connector_from'],
    }

    @staticmethod
    def _build_filters_for_log(filters: 'SearchFilters') -> dict:
        return {
            'category': filters.product_category,
            'connector_from': filters.connector_from,
            'connector_to': filters.connector_to,
            'length': filters.length,
            'length_unit': getattr(filters, 'length_unit', 'ft'),
            'features': filters.features or [],
            'keywords': filters.keywords or [],
        }

    def handle(self, ctx: HandlerContext) -> HandlerResult:
        # Clear stale context
        self._clear_stale_context(ctx)

        # Check for query correction (e.g., "Actually, I meant DisplayPort")
        correction_filters = self._check_for_correction(ctx)
        if correction_filters:
            ctx.add_debug(f"🔄 CORRECTION DETECTED: Merging with previous filters")
            # Use correction filters instead of extracting fresh ones
            return self._handle_corrected_search(ctx, correction_filters)

        # Check for direct SKU lookup
        if ctx.intent.sku:
            return self._handle_sku_lookup(ctx, ctx.intent.sku)

        # Extract filters from query (LLM-based, scalable)
        previous_category = (ctx.context.last_filters.get('category')
                             or ctx.context.last_filters.get('product_category')) if ctx.context.last_filters else None
        llm_result = ctx.filter_extractor.extract(ctx.query, previous_category=previous_category)

        if llm_result and llm_result.confidence >= 0.7:
            filters = llm_result.filters
            ctx.add_debug(f"🤖 LLM FILTERS (conf={llm_result.confidence:.2f}): {llm_result.reasoning[:60]}")
        else:
            # Low confidence — use what we got but log it
            filters = llm_result.filters if llm_result else SearchFilters()
            ctx.add_debug("⚠️ LLM FILTER LOW CONFIDENCE: Using best-effort extraction")

        # Carry forward previous category when LLM didn't extract one.
        # When products are in context, always carry forward — the user is still
        # in the same product category unless they explicitly asked for something else.
        # EXCEPTION: if the last bot response suggested a different category (e.g.,
        # "you need a dock, not a splitter"), don't carry forward the old category.
        if not filters.product_category and previous_category:
            has_context = bool(ctx.context.current_products)

            # Check if last bot response mentioned a different product category
            bot_suggested_different = False
            if has_context:
                last_bot_msg = ctx.context.get_last_message(role='assistant')
                if last_bot_msg and last_bot_msg.content:
                    response_lower = last_bot_msg.content.lower()
                    # Look for category suggestion phrases
                    suggestion_phrases = [
                        'you need a', 'you\'ll need a', 'you would need a',
                        'try a', 'look for a', 'search for a',
                        'different product type', 'different type of',
                    ]
                    if any(phrase in response_lower for phrase in suggestion_phrases):
                        from core.category_config import CATEGORY_DESCRIPTIONS
                        for cat in CATEGORY_DESCRIPTIONS:
                            if cat != previous_category and cat.replace('_', ' ') in response_lower:
                                bot_suggested_different = True
                                ctx.add_debug(f"🚫 SKIPPED CARRY-FORWARD: Last response suggested '{cat}', not '{previous_category}'")
                                break

            if not bot_suggested_different and (has_context or (llm_result and llm_result.confidence >= 0.7)):
                filters.product_category = previous_category
                ctx.add_debug(f"🔄 CATEGORY CARRY-FORWARD: Using previous category '{previous_category}'"
                              f"{' (products in context)' if has_context else ''}")
            elif not bot_suggested_different:
                ctx.add_debug(f"⚠️ SKIPPED CARRY-FORWARD: Low filter confidence, not assuming category '{previous_category}'")

        # Device-to-connector post-processing: LLM sometimes extracts generic "USB"
        # when the device implies a specific connector type
        query_lower = ctx.query.lower()
        if filters.connector_to and filters.connector_to.upper() == 'USB':
            if any(w in query_lower for w in ['printer', 'scanner']):
                filters.connector_to = 'USB-B'
                ctx.add_debug("🔄 CONNECTOR OVERRIDE: printer/scanner → USB-B")

        ctx.add_debug(f"🔍 FILTERS: {filters}")

        # Vagueness gate: if filters aren't specific enough, ask for clarification
        # before searching. This catches queries like "I need something for my network"
        # where a category was extracted but nothing else.
        if filters.has_search_criteria:
            from handlers.clarification import VagueSearchHandler
            vague_handler = VagueSearchHandler()
            if not vague_handler._has_clear_filters(filters):
                vague_type = vague_handler.detector.detect(ctx.query, filters)
                if vague_type:
                    ctx.add_debug(f"🤔 VAGUE QUERY: type={vague_type.value}, routing to clarification")
                    return vague_handler.handle(ctx)

        # Check if current products already meet the new search criteria
        # (e.g., user says "single mode" when products ARE already single mode)
        # Only for small working sets (≤5 products) — not stale pools from narrowing.
        # Large context counts mean the user likely wants a fresh search, not a confirmation.
        current_count = len(ctx.context.current_products) if ctx.context.current_products else 0
        if current_count > 0 and current_count <= 5 and filters.keywords:
            keyword_match_count = 0
            for p in ctx.context.current_products:
                content = (p.content or '').lower()
                meta_str = ' '.join(str(v).lower() for v in p.metadata.values() if isinstance(v, str))
                searchable = content + ' ' + meta_str
                if all(kw.lower() in searchable for kw in filters.keywords):
                    keyword_match_count += 1
            if keyword_match_count == len(ctx.context.current_products) and keyword_match_count > 0:
                kw_str = ', '.join(filters.keywords)
                ctx.add_debug(f"✅ ALREADY MEETS: All {keyword_match_count} products match keywords [{kw_str}]")
                return HandlerResult(
                    response=f"All {keyword_match_count} products currently shown already match **{kw_str}**. "
                             f"Would you like to narrow by a different spec?"
                )

        # Guard: don't search with completely empty filters
        if not filters.has_search_criteria:
            ctx.add_debug("⚠️ EMPTY FILTERS: No meaningful filters extracted, treating as greeting")
            return HandlerResult(
                response="I'd be happy to help you find StarTech.com products! What are you looking for? For example:\n\n"
                         "- \"USB-C dock for my laptop\"\n"
                         "- \"HDMI cable, 10 feet\"\n"
                         "- \"KVM switch for two monitors\""
            )

        # Perform search
        results = ctx.search_engine.search(filters)
        ctx.add_debug(f"🔍 SEARCH: Found {len(results.products)} products (match_quality={results.match_quality})")
        if results.match_quality == 'relaxed':
            ctx.add_debug(f"⚠️ FALLBACK: Relaxed search, filters_used={results.filters_used}")

        if not results.products:
            # Try fallback searches
            return self._handle_no_results(ctx, filters)

        # Two-product solution: relaxed search dropped connectors, but user asked for a specific length
        if (results.match_quality == 'relaxed'
                and filters.length and filters.connector_from and filters.connector_to):
            two_product = self._try_two_product_solution(ctx, filters, results.products)
            if two_product:
                return two_product

        # Build unmet-criteria warning for partial matches
        unmet_note = None
        if results.match_quality == 'partial' and filters.length:
            max_length = max((p.metadata.get('length_ft', 0) for p in results.products), default=0)
            if filters.length > max_length * 1.5:
                suggestion = ""
                if filters.length > 100:
                    suggestion = " For long-distance runs, consider a KVM extender or fiber optic solution instead."
                unmet_note = (
                    f"**Note:** No {int(filters.length)}ft options available — "
                    f"the longest we carry is {max_length:.0f}ft.{suggestion} "
                    f"Here are the closest matches.\n\n"
                )

        # Product narrowing: if > 5 products with tied scores, ask narrowing questions
        if self._should_narrow(results):
            from handlers.narrowing import start_narrowing
            ctx.add_debug(f"🔍 NARROWING: {len(results.products)} tied products, starting narrowing")
            result = start_narrowing(ctx, results.products, filters, ctx.query, total_count=results.total_count)
            if unmet_note and result.response:
                result.response = unmet_note + result.response
            return result

        # Top 5 products (already ranked by scored search)
        products_list = results.products[:5]

        # Verify products match requested specs (refresh rate, power wattage)
        spec_warning = self._verify_spec_match(
            products_list, filters, ctx
        )

        intro_text = None
        if unmet_note:
            intro_text = unmet_note
        if spec_warning:
            intro_text = (intro_text or '') + spec_warning + "\n"

        # LLM-based response generation
        from llm.llm_response_generator import generate_response, ResponseType
        response = generate_response(
            products=products_list,
            query=ctx.query,
            response_type=ResponseType.SEARCH_RESULTS,
            context={
                'dropped_filters': results.dropped_filters,
                'original_filters': filters,
            }
        )
        # Prepend intro text if any (spec warnings, etc.)
        if intro_text:
            response = intro_text + response

        top_products = results.products if len(results.products) > len(products_list) else products_list

        return HandlerResult(
            response=response,
            products_to_set=top_products,
            filters_for_logging=self._build_filters_for_log(filters),
            products_found=len(results.products)
        )

    def _is_connector_category(self, category: str) -> bool:
        """Check if category uses connector gates in search."""
        return (category or '').lower() in self.CONNECTOR_CATEGORIES

    def _should_narrow(self, results) -> bool:
        """
        Check if product narrowing should be triggered.

        Returns True when > 5 products have tied relevance scores,
        meaning the query didn't have enough detail to rank them.
        """
        if len(results.products) <= 5:
            return False

        # Check search_scores for ties
        scores = results.search_scores
        if not scores or len(scores) < 4:
            return False

        # Count how many of the top products share the same score
        top_score = scores[0]
        tied_count = sum(1 for s in scores[:min(len(scores), 10)] if abs(s - top_score) < 0.01)

        return tied_count > 5

    def _handle_no_results(self, ctx: HandlerContext, filters: SearchFilters) -> HandlerResult:
        """Try recovery strategies in order when initial search finds nothing."""
        original_cat = (filters.product_category or '').lower()

        # Port count guard (early exit, not a recovery strategy)
        if filters.port_count and filters.port_count > 10:
            category = filters.product_category or "products"
            return HandlerResult(
                response=f"We don't have {filters.port_count}-port {category.lower()}. "
                         f"Our {category.lower()} typically come in 4, 7, or 10-port configurations.\n\n"
                         f"Would you like me to show you our highest port count options?"
            )

        # Try recovery strategies in order — first success wins
        result = (
            self._recover_category_swap(ctx, filters, original_cat)
            or self._recover_keyword_relaxation(ctx, filters)
            or self._recover_connector_relaxation(ctx, filters, original_cat)
            or self._recover_structured_filter_relaxation(ctx, filters, original_cat)
            or self._recover_category_key_filter(ctx, filters, original_cat)
        )

        # Connector relaxation may return a complete HandlerResult (two-product solution)
        if isinstance(result, HandlerResult):
            return result

        if result:
            fallback_results, fallback_note = result
            return self._finalize_fallback(ctx, filters, fallback_results, fallback_note)

        return self._no_results_response(ctx, filters)

    def _recover_category_swap(self, ctx: HandlerContext, filters: SearchFilters, original_cat: str):
        """Swap cables <-> adapters. Returns (results, fallback_note) or None."""
        category_swaps = {
            'cables': 'Adapters', 'cable': 'Adapters',
            'adapters': 'Cables', 'adapter': 'Cables',
        }
        if original_cat not in category_swaps:
            return None

        swapped = SearchFilters()
        swapped.product_category = category_swaps[original_cat]
        swapped.connector_from = filters.connector_from
        swapped.connector_to = filters.connector_to
        swapped.features = filters.features
        swapped.keywords = filters.keywords
        swapped.cable_type = filters.cable_type
        swapped.required_port_types = filters.required_port_types

        ctx.add_debug(f"🔄 FALLBACK 0: Category swap → {category_swaps[original_cat]}")
        fallback_results = ctx.search_engine.search(swapped)

        if not fallback_results.products:
            return None

        fallback_note = (
            f"**Note:** No {filters.product_category.lower()} found, "
            f"but we have matching {category_swaps[original_cat].lower()}:"
        )
        return (fallback_results, fallback_note)

    def _recover_keyword_relaxation(self, ctx: HandlerContext, filters: SearchFilters):
        """Progressively drop keywords. Returns (results, None) or None."""
        if not filters.keywords:
            return None

        for keep_count in range(len(filters.keywords) - 1, -1, -1):
            relaxed = replace(filters, keywords=list(filters.keywords[:keep_count]))
            fallback_results = ctx.search_engine.search(relaxed)
            if fallback_results.products:
                dropped = filters.keywords[keep_count:]
                ctx.add_debug(
                    f"🔄 KEYWORD RELAXATION: Dropped {dropped}, "
                    f"found {len(fallback_results.products)}"
                )
                return (fallback_results, None)

        return None

    def _recover_connector_relaxation(self, ctx: HandlerContext, filters: SearchFilters, original_cat: str):
        """Connector relaxation for cables/adapters. Returns (results, note), HandlerResult, or None."""
        if not self._is_connector_category(original_cat):
            return None

        fallback_results = None
        fallback_note = None

        # 1a: Drop length/features, keep both connectors
        if filters.connector_from or filters.connector_to:
            relaxed = SearchFilters()
            relaxed.connector_from = filters.connector_from
            relaxed.connector_to = filters.connector_to
            relaxed.product_category = filters.product_category

            ctx.add_debug("🔄 FALLBACK 1a: Connectors only (dropped length/features)")
            fallback_results = ctx.search_engine.search(relaxed)

            if fallback_results.products:
                conn_from = filters.connector_from or "?"
                conn_to = filters.connector_to or "?"
                if filters.length:
                    fallback_note = (
                        f"**Note:** No {int(filters.length)}ft {conn_from} to {conn_to} "
                        f"cables found. Here are available options:"
                    )
                elif filters.features:
                    feat_str = ', '.join(filters.features)
                    fallback_note = (
                        f"**Note:** No {conn_from} to {conn_to} cables with "
                        f"{feat_str} found. Here are available options:"
                    )

        # Two-product solution: adapter + long cable when length was requested
        if (fallback_results and fallback_results.products
                and filters.length and filters.connector_from and filters.connector_to):
            two_product = self._try_two_product_solution(ctx, filters, fallback_results.products)
            if two_product:
                return two_product

        # 1b: Single connector (drop one connector gate)
        # Only try this when ONE connector was specified — if the user said both
        # (e.g., "USB to parallel"), dropping one gives irrelevant results.
        has_both_connectors = filters.connector_from and filters.connector_to
        if not has_both_connectors and (not fallback_results or not fallback_results.products):
            for connector in [filters.connector_from, filters.connector_to]:
                if connector:
                    simple = SearchFilters()
                    simple.connector_from = connector
                    simple.product_category = filters.product_category or 'Cables'

                    ctx.add_debug(f"🔄 FALLBACK 1b: Single connector: {connector}")
                    fallback_results = ctx.search_engine.search(simple)

                    if fallback_results.products:
                        conn_from = filters.connector_from or "?"
                        conn_to = filters.connector_to or "?"
                        fallback_note = (
                            f"**Note:** No {conn_from} to {conn_to} products found. "
                            f"Here are {connector} products:"
                        )
                        break

        if fallback_results and fallback_results.products:
            return (fallback_results, fallback_note)

        return None

    def _recover_structured_filter_relaxation(self, ctx: HandlerContext, filters: SearchFilters, original_cat: str):
        """Progressively drop structured filters, least important first."""
        key_filter = self.CATEGORY_KEY_FILTERS.get(original_cat)

        # Drop order: cosmetic → interface-specific → functional
        drop_order = [
            ('color', None),
            ('connector_from', None),
            ('connector_to', None),
            ('features', []),
            ('cable_type', None),
            ('kvm_video_type', None),
            ('thunderbolt_version', None),
            ('screen_size', None),
            ('rack_height', None),
            ('port_count', None),
            ('min_monitors', None),
            ('usb_version', None),
            ('drive_size', None),
            ('bay_count', None),
        ]

        # Version filters represent explicit technology choices — never relax these
        protected_filters = {key_filter, 'thunderbolt_version', 'usb_version'} - {None}

        relaxed = filters
        dropped = []
        for field, default in drop_order:
            if field in protected_filters:
                continue
            val = getattr(relaxed, field, None)
            if not val or (isinstance(val, list) and not val):
                continue
            relaxed = replace(relaxed, **{field: default})
            dropped.append(field)

            # If both connectors are now gone, the core ask is lost —
            # stop before searching so we don't return irrelevant results.
            if field in ('connector_from', 'connector_to'):
                if not getattr(relaxed, 'connector_from', None) and not getattr(relaxed, 'connector_to', None):
                    ctx.add_debug("🔄 FILTER RELAXATION: Both connectors dropped, stopping")
                    return None

            results = ctx.search_engine.search(relaxed)
            if results.products:
                ctx.add_debug(
                    f"🔄 FILTER RELAXATION: Dropped {dropped}, "
                    f"found {len(results.products)}"
                )
                return (results, None)

        return None

    def _recover_category_key_filter(self, ctx: HandlerContext, filters: SearchFilters, original_cat: str):
        """Category + one key attribute. Returns (results, None) or None."""
        if not filters.product_category:
            return None

        category_only = SearchFilters()
        category_only.product_category = filters.product_category

        # Always preserve keywords — they're universally useful for narrowing within any category
        if filters.keywords:
            category_only.keywords = list(filters.keywords)

        # Version filters represent explicit technology choices — preserve through fallback
        for version_field in ('thunderbolt_version', 'usb_version'):
            version_val = getattr(filters, version_field, None)
            if version_val:
                setattr(category_only, version_field, version_val)

        # Preserve the single most important filter for this category
        key_attr = self.CATEGORY_KEY_FILTERS.get(original_cat)
        key_value = getattr(filters, key_attr, None) if key_attr else None

        # If primary key filter is empty, try semantically related fields
        if key_attr and not key_value:
            for fallback_attr in self.KEY_FILTER_FALLBACKS.get(key_attr, []):
                fallback_value = getattr(filters, fallback_attr, None)
                if fallback_value:
                    key_value = fallback_value
                    key_attr = fallback_attr
                    break

        if key_attr and key_value:
            setattr(category_only, key_attr, key_value)
            kw_note = f" + keywords={filters.keywords}" if filters.keywords else ""
            ctx.add_debug(
                f"🔄 FALLBACK 2: Category '{filters.product_category}' "
                f"+ key filter {key_attr}={key_value}{kw_note}"
            )
        else:
            kw_note = f" + keywords={filters.keywords}" if filters.keywords else ""
            ctx.add_debug(
                f"🔄 FALLBACK 2: Category-only search for "
                f"'{filters.product_category}'{kw_note}"
            )

        fallback_results = ctx.search_engine.search(category_only)

        # If keywords returned 0 results, retry without them
        if not fallback_results.products and category_only.keywords:
            ctx.add_debug("🔄 FALLBACK 2: Keywords too restrictive, retrying without")
            category_only.keywords = []
            fallback_results = ctx.search_engine.search(category_only)

        # Post-filter by monitor count (scored search treats it as weighted,
        # but for fallback we want strict enforcement)
        if (fallback_results.products and key_attr == 'min_monitors'
                and key_value):
            filtered = self._filter_by_monitor_count(
                fallback_results.products, key_value
            )
            if filtered:
                fallback_results.products = filtered
                ctx.add_debug(
                    f"🔍 FALLBACK 2 MONITOR FILTER: min {key_value} → "
                    f"{len(filtered)} products"
                )

        if not fallback_results.products:
            return None

        return (fallback_results, None)

    def _finalize_fallback(self, ctx: HandlerContext, filters: SearchFilters, fallback_results, fallback_note):
        """Post-filter, narrow, or return top results from fallback."""
        # Post-filter fallback results by original requirements
        post_filtered = self._post_filter_by_original(
            fallback_results.products, filters, ctx
        )
        if post_filtered:
            fallback_results.products = post_filtered
        elif not post_filtered and (filters.connector_from or filters.connector_to):
            # Post-filter eliminated everything AND original query had specific connectors.
            # The fallback products don't match the core ask — say so clearly.
            conn_from = filters.connector_from or "?"
            conn_to = filters.connector_to or "?"
            ctx.add_debug("🚫 FALLBACK: Post-filter eliminated all products — catalog gap")
            return HandlerResult(
                response=f"We don't currently carry **{conn_from} to {conn_to}** products in our catalog. "
                         f"Would you like to search for something similar, or try a different connector type?"
            )

        # Only mark as fallback if meaningful filters (not just keywords) were dropped
        meaningful_fallback = any(
            getattr(filters, f, None)
            for f in ['connector_from', 'connector_to', 'length', 'features',
                      'port_count', 'color', 'screen_size', 'cable_type',
                      'kvm_video_type', 'bay_count', 'rack_height', 'drive_size',
                      'usb_version', 'thunderbolt_version', 'requested_refresh_rate',
                      'requested_power_wattage', 'requested_network_speed', 'min_monitors']
        )

        # Route through narrowing if many tied results (same as main search path)
        if self._should_narrow(fallback_results):
            from handlers.narrowing import start_narrowing
            ctx.add_debug(f"🔍 NARROWING: {len(fallback_results.products)} fallback products, starting narrowing")
            return start_narrowing(ctx, fallback_results.products, filters, ctx.query, total_count=fallback_results.total_count, is_fallback=meaningful_fallback)

        products_list = fallback_results.products[:3]

        from llm.llm_response_generator import generate_response, ResponseType
        response = generate_response(
            products=products_list,
            query=ctx.query,
            response_type=ResponseType.SEARCH_RESULTS,
            context={
                'original_filters': filters,
                'is_fallback': meaningful_fallback,
            }
        )
        if fallback_note:
            response = fallback_note + "\n" + response

        return HandlerResult(
            response=response,
            products_to_set=products_list,
            filters_for_logging=self._build_filters_for_log(filters),
            products_found=len(fallback_results.products)
        )

    def _no_results_response(self, ctx: HandlerContext, filters: SearchFilters) -> HandlerResult:
        """Vague query → clarification; category set but not found → suggestions."""
        if not filters.product_category:
            ctx.add_debug("🤔 NO RESULTS + VAGUE QUERY: Starting clarification")
            from handlers.clarification import VagueSearchHandler
            return VagueSearchHandler().handle(ctx)

        # Category was set but genuinely not in catalog
        category = filters.product_category or 'that product'
        category_display = category.replace('_', ' ')
        return HandlerResult(
            response=f"I wasn't able to find **{category_display}** products matching your request. "
                     f"Could you try describing what you need differently, or tell me more about what you're trying to connect?"
        )

    def _try_two_product_solution(
        self,
        ctx: HandlerContext,
        filters: SearchFilters,
        adapter_candidates: list,
    ):
        """
        Recommend adapter + long cable when no single cable covers the requested length.
        e.g., "30ft USB-C to HDMI" -> USB-C->HDMI adapter + 30ft HDMI cable
        Returns HandlerResult or None if no two-product solution found.
        """
        intermediate = filters.connector_to  # e.g., "HDMI"

        # Search for a cable of the intermediate type at the requested length
        cable_filters = SearchFilters()
        cable_filters.connector_from = intermediate
        cable_filters.connector_to = intermediate
        cable_filters.length = filters.length
        cable_filters.length_unit = getattr(filters, 'length_unit', 'ft') or 'ft'

        ctx.add_debug(
            f"🔄 TWO-PRODUCT: Searching {intermediate}→{intermediate} cable at {filters.length}ft"
        )
        cable_results = ctx.search_engine.search(cable_filters)

        if not cable_results.products or cable_results.match_quality == 'relaxed':
            ctx.add_debug("🔄 TWO-PRODUCT: No length-matched cable found, skipping")
            return None

        # Pick best adapter and cable (already ranked by scored search)
        if not adapter_candidates or not cable_results.products:
            return None

        adapter = adapter_candidates[0]
        cable = cable_results.products[0]

        ctx.add_debug(f"✅ TWO-PRODUCT SOLUTION: {adapter.product_number} + {cable.product_number}")

        from llm.llm_response_generator import generate_response, ResponseType
        length_val = filters.length
        length_display = int(length_val) if length_val == int(length_val) else length_val
        response = generate_response(
            products=[adapter, cable],
            query=ctx.query,
            response_type=ResponseType.TWO_PRODUCT_SOLUTION,
            context={
                'original_request': f'{length_display}ft {filters.connector_from} to {filters.connector_to}',
                'adapter_sku': adapter.product_number,
                'cable_sku': cable.product_number,
                'intermediate_connector': intermediate,
            }
        )

        return HandlerResult(
            response=response,
            products_to_set=[adapter, cable]
        )

    def _check_for_correction(self, ctx: HandlerContext) -> SearchFilters | None:
        """
        Check if query is a correction to the previous search.

        Examples:
        - "Actually, I meant DisplayPort" (after searching HDMI cables)
        - "No, USB-C not HDMI"
        - "Wait, I need Thunderbolt instead"

        Returns merged filters if correction detected, None otherwise.
        """
        # Need previous filters to correct
        if not ctx.context.last_filters or not ctx.context.last_query:
            return None

        query_lower = ctx.query.lower()

        # Check for correction patterns
        is_correction = False
        for pattern in CORRECTION_PATTERNS:
            if re.search(pattern, query_lower, re.IGNORECASE):
                is_correction = True
                break

        if not is_correction:
            return None

        # Clean the query before extraction -- strip correction preamble and the negated
        # connector ("not HDMI", "not USB-A") so the extractor only sees the positive intent
        clean_query = ctx.query
        clean_query = re.sub(r'\b(?:actually|wait|no|sorry|oops)\b,?\s*', '', clean_query, flags=re.IGNORECASE)
        clean_query = re.sub(r'\bi\s+meant\s*', '', clean_query, flags=re.IGNORECASE)
        clean_query = re.sub(r'\bi\s+need\s*', '', clean_query, flags=re.IGNORECASE)
        clean_query = re.sub(r',?\s*\bnot\s+[\w\-]+', '', clean_query, flags=re.IGNORECASE)
        clean_query = clean_query.strip(' ,.')

        # Extract what the user wants instead
        llm_result = ctx.filter_extractor.extract(clean_query)
        new_filters = llm_result.filters if llm_result else SearchFilters()

        # If we extracted a connector but no category, inherit category from previous search
        prev_category = (ctx.context.last_filters.get('category')
                         or ctx.context.last_filters.get('product_category'))
        if not new_filters.product_category and prev_category:
            new_filters.product_category = prev_category

        # Smart connector merge: put the new connector in the slot the user is correcting
        new_connector = new_filters.connector_from or new_filters.connector_to
        if new_connector:
            prev_from = ctx.context.last_filters.get('connector_from')
            prev_to = ctx.context.last_filters.get('connector_to')

            if prev_to and not prev_from:
                # Previous had only connector_to → replace that slot
                new_filters.connector_to = new_connector
                new_filters.connector_from = None
            elif prev_from and not prev_to:
                # Previous had only connector_from → replace that slot
                new_filters.connector_from = new_connector
                new_filters.connector_to = None
            elif prev_from and prev_to:
                # Both were set — check which one the user negated
                negated = re.search(r'\bnot\s+([\w\-]+)', ctx.query, re.IGNORECASE)
                if negated:
                    neg_val = negated.group(1).lower()
                    from_negated = neg_val in prev_from.lower()
                    to_negated = neg_val in prev_to.lower()
                    if from_negated and to_negated:
                        # Same-connector cable correction (e.g., TB→TB to USB-C→USB-C)
                        new_filters.connector_from = new_connector
                        new_filters.connector_to = new_connector
                    elif to_negated:
                        new_filters.connector_to = new_connector
                        new_filters.connector_from = prev_from
                    elif from_negated:
                        new_filters.connector_from = new_connector
                        new_filters.connector_to = prev_to
                # else: use what the LLM extracted as-is
        else:
            # No connector extracted — can't determine what to correct
            return None

        return new_filters

    def _handle_corrected_search(self, ctx: HandlerContext, filters: SearchFilters) -> HandlerResult:
        """Handle a corrected search using merged filters."""
        ctx.add_debug(f"🔍 CORRECTED FILTERS: {filters}")

        # Perform search with corrected filters
        results = ctx.search_engine.search(filters)
        ctx.add_debug(f"🔍 SEARCH: Found {len(results.products)} products (match_quality={results.match_quality})")

        if not results.products:
            return self._handle_no_results(ctx, filters)

        # Route through narrowing if many tied results (same as main search path)
        if self._should_narrow(results):
            from handlers.narrowing import start_narrowing
            ctx.add_debug(f"🔍 NARROWING: {len(results.products)} corrected results, starting narrowing")
            return start_narrowing(ctx, results.products, filters, ctx.query,
                                   total_count=results.total_count)

        # Top 3 products (already ranked by scored search)
        products_list = results.products[:3]

        # Build response with acknowledgment of correction
        from llm.llm_response_generator import generate_response, ResponseType
        response = generate_response(
            products=products_list,
            query=ctx.query,
            response_type=ResponseType.SEARCH_RESULTS,
            context={
                'dropped_filters': results.dropped_filters,
                'original_filters': filters,
                'is_correction': True,
            }
        )

        top_products = products_list

        return HandlerResult(
            response=response,
            products_to_set=top_products,
            filters_for_logging=self._build_filters_for_log(filters),
            products_found=len(results.products)
        )

    def _filter_by_port_types(
        self,
        products: list[Product],
        required_port_types: list[str]
    ) -> list[Product]:
        """
        Filter products to only those that have all required port types.

        Args:
            products: List of products to filter
            required_port_types: Port types the product must have (e.g., ["USB-C"])

        Returns:
            Filtered list of products that have all required port types
        """
        if not required_port_types:
            return products

        return [p for p in products if self._product_has_port_types(p, required_port_types)]

    def _product_has_port_types(
        self,
        product: Product,
        required_port_types: list[str]
    ) -> bool:
        """
        Check if a product has all the required port types.

        When USE_LLM_VERIFICATION is enabled, uses LLM to handle
        various port naming conventions and formats.

        Args:
            product: Product to check
            required_port_types: Port types to look for (e.g., ["USB-C", "USB-A"])

        Returns:
            True if product has all required port types
        """
        # Try LLM verification first if enabled
        if USE_LLM_VERIFICATION:
            try:
                from llm.llm_product_verifier import get_verifier
                result = get_verifier().verify_port_types(product, required_port_types)
                if result.confidence >= 0.7:
                    return result.meets_requirement
                # Fall through to regex if LLM is uncertain
            except Exception:
                pass  # Fall through to regex

        # Regex-based verification (fallback or primary)
        # Get conn_type field (contains port information)
        conntype = product.metadata.get('conn_type', '')

        if not conntype:
            # No port info - can't verify, exclude from filtered results
            return False

        conntype_lower = str(conntype).lower()

        # Port type patterns to match in conn_type
        port_patterns = {
            'USB-C': [r'type[\s\-]?c', r'usb[\s\-]?c', r'usb\s+3\.\d+\s+type-c'],
            'USB-A': [r'type[\s\-]?a', r'usb[\s\-]?a', r'usb\s+\d+\.\d+\s+type-a'],
            'USB': [r'\busb\b'],  # Generic USB
            'HDMI': [r'\bhdmi\b'],
            'DisplayPort': [r'\bdisplayport\b', r'\bdp\b'],
            'Thunderbolt': [r'\bthunderbolt\b'],
            'Ethernet': [r'\bethernet\b', r'\brj[\s\-]?45\b', r'\bgigabit\b'],
        }

        # Check if all required port types are present
        for port_type in required_port_types:
            patterns = port_patterns.get(port_type, [port_type.lower()])

            # Check if any pattern matches
            found = False
            for pattern in patterns:
                if re.search(pattern, conntype_lower):
                    found = True
                    break

            if not found:
                return False

        return True

    def _filter_by_monitor_count(
        self,
        products: list[Product],
        min_monitors: int
    ) -> list[Product]:
        """
        Filter products to only those that support at least min_monitors.

        Checks multiple metadata fields since different product types store
        monitor count differently:
        - Docks: dock_num_displays
        - Mounts: mount_num_displays (derived from NUMOFDISPLAY)

        Args:
            products: List of products to filter
            min_monitors: Minimum number of monitors the product must support

        Returns:
            Filtered list of products that support at least min_monitors
        """
        if not min_monitors:
            return products

        filtered = []
        for product in products:
            # Check multiple fields where monitor count might be stored
            # Different product types use different fields
            num_displays = None

            # Try dock field first (dock_num_displays)
            dock_displays = product.metadata.get('dock_num_displays')
            if dock_displays:
                try:
                    num_displays = int(float(dock_displays))
                except (ValueError, TypeError):
                    pass

            # Try mount field (mount_num_displays, derived from NUMOFDISPLAY)
            if num_displays is None:
                mount_displays = product.metadata.get('mount_num_displays')
                if mount_displays:
                    try:
                        num_displays = int(float(mount_displays))
                    except (ValueError, TypeError):
                        pass

            # Check if product meets the minimum requirement
            if num_displays is not None and num_displays >= min_monitors:
                filtered.append(product)

        return filtered

    def _post_filter_by_original(
        self,
        products: list,
        original_filters: 'SearchFilters',
        ctx: HandlerContext,
    ) -> list:
        """
        Post-filter fallback results by original filter requirements.

        When fallback drops filters to get results, this re-applies the dropped
        filters as a best-effort post-filter. Only returns the filtered list
        if it's non-empty (caller keeps the unfiltered pool otherwise).
        """
        filtered = ctx.search_engine.filter_products(products, original_filters)

        if filtered and len(filtered) < len(products):
            ctx.add_debug(
                f"🔍 FALLBACK POST-FILTER: {len(products)} → {len(filtered)} "
                f"products (applied original filters)"
            )

        return filtered

    def _handle_sku_lookup(self, ctx: HandlerContext, sku: str) -> HandlerResult:
        """
        Handle direct SKU lookup.

        Args:
            ctx: Handler context
            sku: Product SKU to look up

        Returns:
            HandlerResult with product info or not found message
        """
        ctx.add_debug(f"🔍 SKU LOOKUP: {sku}")

        # Search for the exact SKU in all products
        matching_products = []
        sku_upper = sku.upper()

        for product in ctx.all_products:
            product_sku = product.product_number.upper()
            # Exact match or starts with (for partial SKUs)
            if product_sku == sku_upper or product_sku.startswith(sku_upper):
                matching_products.append(product)

        if not matching_products:
            ctx.add_debug(f"🔍 SKU LOOKUP: No products found for {sku}")
            return HandlerResult(
                response=f"I couldn't find a product with SKU **{sku}**. "
                         f"Please check the SKU and try again, or describe what you're looking for."
            )

        ctx.add_debug(f"🔍 SKU LOOKUP: Found {len(matching_products)} products")

        # If exact match, show that product
        exact_match = next((p for p in matching_products if p.product_number.upper() == sku_upper), None)

        if exact_match:
            from llm.llm_response_generator import generate_response, ResponseType
            response = generate_response(
                products=[exact_match],
                query=ctx.query,
                response_type=ResponseType.PRODUCT_DETAILS,
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

    def _verify_spec_match(
        self,
        ranked_products: list,
        filters: SearchFilters,
        ctx: HandlerContext
    ) -> str | None:
        """
        Verify that ranked products match requested specs (refresh rate, power wattage).

        If products don't meet the requested specs, returns a warning message
        to be displayed to the user. This ensures users know when shown products
        may not fully meet their requirements.

        Args:
            ranked_products: List of Product objects to verify
            filters: Extracted search filters including spec requirements
            ctx: Handler context for debug logging

        Returns:
            Warning message string if specs don't match, None otherwise
        """
        if not ranked_products:
            return None

        warnings = []

        # Check refresh rate (144Hz, 120Hz, etc.)
        if filters.requested_refresh_rate:
            hz = filters.requested_refresh_rate
            verified_count = 0
            for product in ranked_products:
                if self._product_supports_refresh_rate(product, hz):
                    verified_count += 1

            ctx.add_debug(
                f"⚡ SPEC CHECK: {verified_count}/{len(ranked_products)} "
                f"products verified for {hz}Hz"
            )

            if verified_count == 0:
                warnings.append(
                    f"**Note:** I couldn't verify {hz}Hz support for these products. "
                    f"Please check the product specifications for your required refresh rate."
                )
            elif verified_count < len(ranked_products):
                warnings.append(
                    f"**Note:** Only some products shown may support {hz}Hz. "
                    f"Check individual specs for your required refresh rate."
                )

        # Check power delivery wattage (100W, 60W, etc.)
        if filters.requested_power_wattage:
            wattage = filters.requested_power_wattage
            verified_count = 0
            for product in ranked_products:
                if self._product_supports_power_delivery(product, wattage):
                    verified_count += 1

            ctx.add_debug(
                f"⚡ SPEC CHECK: {verified_count}/{len(ranked_products)} "
                f"products verified for {wattage}W PD"
            )

            if verified_count == 0:
                warnings.append(
                    f"**Note:** I couldn't verify {wattage}W power delivery for these products. "
                    f"Please check the product specifications for your required wattage."
                )
            elif verified_count < len(ranked_products):
                warnings.append(
                    f"**Note:** Only some products shown support {wattage}W power delivery. "
                    f"Check individual specs for your required wattage."
                )

        if warnings:
            return "\n".join(warnings)
        return None

    def _product_supports_refresh_rate(self, product: Product, hz: int) -> bool:
        """
        Check if a product supports the requested refresh rate.

        When USE_LLM_VERIFICATION is enabled, uses LLM to understand
        resolution context (144Hz at 1080p vs 4K).

        Args:
            product: Product to check
            hz: Refresh rate in Hz (e.g., 144, 120, 240)

        Returns:
            True if product supports the refresh rate, False otherwise
        """
        # Try LLM verification first if enabled
        if USE_LLM_VERIFICATION:
            try:
                from llm.llm_product_verifier import get_verifier
                result = get_verifier().verify_refresh_rate(product, hz)
                if result.confidence >= 0.7:
                    return result.meets_requirement
                # Fall through to regex if LLM is uncertain
            except Exception:
                pass  # Fall through to regex

        # Regex-based verification (fallback or primary)
        meta = product.metadata
        content = (product.content or '').lower()

        # Fields that might contain refresh rate info
        max_res = str(meta.get('max_resolution', '') or '')
        sup_res = str(meta.get('supported_resolutions', '') or '')

        # Combine all sources
        all_text = f"{max_res} {sup_res} {content}".lower()

        # Look for the Hz value in text
        # Patterns: "144hz", "144 hz", "@144hz", "144hz", "p144"
        patterns = [
            rf'\b{hz}\s*hz\b',           # "144hz", "144 hz"
            rf'@\s*{hz}\s*hz\b',         # "@144hz"
            rf'p{hz}\b',                 # "1080p144" style
            rf'\b{hz}\s*hertz\b',        # "144 hertz"
        ]

        for pattern in patterns:
            if re.search(pattern, all_text, re.IGNORECASE):
                return True

        # Special case: 60Hz is default for most cables, assume supported
        # if no specific Hz is mentioned and product supports 4K or 1080p
        if hz <= 60:
            if '4k' in all_text or '2160' in all_text or '1080' in all_text:
                return True

        return False

    def _product_supports_power_delivery(self, product: Product, wattage: int) -> bool:
        """
        Check if a product supports the requested power delivery wattage.

        When USE_LLM_VERIFICATION is enabled, uses LLM to understand semantic
        differences like "supports 100W" vs "requires 100W input".

        Args:
            product: Product to check
            wattage: Power in watts (e.g., 100, 60, 240)

        Returns:
            True if product supports the wattage, False otherwise
        """
        # Try LLM verification first if enabled
        if USE_LLM_VERIFICATION:
            try:
                from llm.llm_product_verifier import get_verifier
                result = get_verifier().verify_power_delivery(product, wattage)
                if result.confidence >= 0.7:
                    return result.meets_requirement
                # Fall through to regex if LLM is uncertain
            except Exception:
                pass  # Fall through to regex

        # Regex-based verification (fallback or primary)
        meta = product.metadata
        content = (product.content or '').lower()

        # Check power_delivery field
        pd_field = str(meta.get('power_delivery', '') or '')

        # Extract numeric wattage from field
        if pd_field:
            # Pattern: "100W", "60 W", etc.
            match = re.search(r'(\d+)\s*w', pd_field.lower())
            if match:
                product_wattage = int(match.group(1))
                # Product supports requested wattage if it's >= requested
                if product_wattage >= wattage:
                    return True

        # Check content for wattage mentions
        # Pattern: "100W power delivery", "supports 100W", etc.
        patterns = [
            rf'\b{wattage}\s*w(?:att)?\b',                    # "100W", "100 watt"
            rf'\b(?:up\s+to\s+)?{wattage}\s*w\s*pd\b',       # "100W PD", "up to 100W PD"
            rf'power\s+delivery[^.]*{wattage}\s*w',          # "power delivery...100W"
        ]

        for pattern in patterns:
            if re.search(pattern, content, re.IGNORECASE):
                return True

        # Also check for higher wattages that would support the requested amount
        # e.g., 240W cable supports 100W
        match = re.search(r'(\d{2,3})\s*w(?:att)?\s*(?:pd|power)', content.lower())
        if match:
            product_wattage = int(match.group(1))
            if product_wattage >= wattage:
                return True

        return False
