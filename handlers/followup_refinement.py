"""
Refinement and search helper methods for the FollowupHandler.

Extracted from followup.py to reduce file size and improve maintainability.
These methods handle:
- Length refinement (absolute, range, threshold, relative)
- Feature-based refinement
- Auto-search when filters return 0 results
- Product type/connector/category helpers
"""

import re
from handlers.base import HandlerContext, HandlerResult
from core.models import LengthPreference
from core.models.filters import SearchFilters
from core.product_validator import get_best_cable
from core.search import SearchStrategy, SearchConfig


class RefinementMixin:
    """Mixin providing refinement and search helper methods for FollowupHandler."""

    def _collect_context_info(self, ctx: HandlerContext):
        """Collect connector pairs, categories, and subcategory keywords from context products.

        Returns (unique_pairs, categories, subcategory_keywords) — shared by all refine methods.
        """
        unique_pairs = {}
        categories = set()
        subcategory_keywords = set()
        for prod in ctx.context.current_products:
            connectors = prod.metadata.get('connectors', [])
            if connectors and len(connectors) >= 2:
                source = self._normalize_connector(connectors[0])
                target = self._normalize_connector(connectors[1])
                if source:
                    key = (source, target or source)
                    if key not in unique_pairs:
                        unique_pairs[key] = prod
            cat = prod.metadata.get('category')
            if cat:
                categories.add(cat)
            subcat = prod.metadata.get('sub_category', '')
            if subcat:
                subcat_lower = subcat.lower()
                for kw in ['ethernet', 'fiber', 'hdmi', 'displayport']:
                    if kw in subcat_lower:
                        subcategory_keywords.add(kw)
                        break
        return unique_pairs, categories, subcategory_keywords

    def _handle_refinement(self, ctx: HandlerContext) -> HandlerResult:
        """Handle refinement requests (e.g., different length)."""
        if not ctx.context.current_products:
            return HandlerResult(
                response="I don't have any products to refine. What are you looking for?"
            )

        query_lower = ctx.query.lower()

        # Extract new filters first to check for range queries
        extraction = ctx.filter_extractor.extract(ctx.query)
        new_filters = extraction.filters if extraction else SearchFilters()

        # Check for range query FIRST (e.g., "between 5 and 7 feet")
        if new_filters.length_min is not None and new_filters.length_max is not None:
            return self._refine_by_length_range(
                ctx,
                new_filters.length_min,
                new_filters.length_max,
                new_filters.length_unit or 'ft'
            )

        # Check for "longer/shorter than X feet" patterns - these have explicit lengths
        # "longer than 100 feet" should search for cables > 100ft, not relative to context
        longer_than_match = re.search(
            r'\blonger\s+than\s+(\d+(?:\.\d+)?)\s*(ft|feet|foot|m|meters?)\b',
            query_lower
        )
        shorter_than_match = re.search(
            r'\bshorter\s+than\s+(\d+(?:\.\d+)?)\s*(ft|feet|foot|m|meters?)\b',
            query_lower
        )

        if longer_than_match:
            min_length = float(longer_than_match.group(1))
            unit = 'ft' if longer_than_match.group(2).startswith('f') else 'm'
            return self._refine_by_length_threshold(ctx, min_length, unit, 'longer')

        if shorter_than_match:
            max_length = float(shorter_than_match.group(1))
            unit = 'ft' if shorter_than_match.group(2).startswith('f') else 'm'
            return self._refine_by_length_threshold(ctx, max_length, unit, 'shorter')

        # Check for relative length refinement (shorter/longer without explicit length)
        if re.search(r'\bshorter\b', query_lower):
            return self._refine_by_relative_length(ctx, 'shorter')
        if re.search(r'\blonger\b', query_lower):
            return self._refine_by_relative_length(ctx, 'longer')

        # Extract new filters for absolute length
        requested_length = new_filters.length
        length_unit = new_filters.length_unit or 'ft'

        if requested_length:
            return self._refine_by_length(ctx, requested_length, length_unit)

        # Check for feature-based refinement
        keywords = self._extract_requirement_keywords(ctx.query)
        if keywords:
            return self._refine_by_requirements(ctx, keywords)

        return HandlerResult(
            response="I couldn't determine what you're looking for. "
                     "Could you specify what features or length you need?"
        )

    def _refine_by_length(self, ctx: HandlerContext, length: float, unit: str) -> HandlerResult:
        """Re-search for products with new length."""
        ctx.add_debug(f"REFINEMENT: length={length}{unit}")

        unique_pairs, _, _ = self._collect_context_info(ctx)

        parts = [f"Here are {int(length)}{unit} options for your setup:", ""]
        all_products = []

        for (source, target), _ in unique_pairs.items():
            extraction = ctx.filter_extractor.extract("")
            filters = extraction.filters if extraction else SearchFilters()
            filters.connector_from = source
            filters.connector_to = target
            filters.length = length
            filters.length_unit = unit

            results = ctx.search_engine.search(filters)

            cable_desc = f"{source} to {target}" if source != target else f"{source} cable"
            parts.append(f"**{cable_desc}:**")

            if results.products:
                length_ft = length if unit == 'ft' else length * 3.28
                best = get_best_cable(results.products, source, target, preferred_length_ft=length_ft)

                if best:
                    all_products.append(best)
                    name = best.metadata.get('name', best.product_number)
                    length_display = best.metadata.get('length_display', '')
                    parts.append(f"Recommended: **{name}** ({best.product_number})")
                    if length_display:
                        parts.append(f"Length: {length_display}")
                else:
                    parts.append(f"_No {int(length)}{unit} cable found_")
            else:
                parts.append(f"_No {int(length)}{unit} version available_")
            parts.append("")

        parts.append("Would you like more details on any of these?")

        return HandlerResult(
            response="\n".join(parts),
            products_to_set=all_products if all_products else None
        )

    def _refine_by_length_range(
        self,
        ctx: HandlerContext,
        min_length: float,
        max_length: float,
        unit: str
    ) -> HandlerResult:
        """Handle length range refinement (e.g., "between 5 and 7 feet")."""
        ctx.add_debug(f"RANGE REFINEMENT: {min_length}-{max_length}{unit}")

        unique_pairs, categories, subcategory_keywords = self._collect_context_info(ctx)

        all_products = []

        # Convert to feet for consistent comparison
        min_ft = min_length if unit == 'ft' else min_length * 3.28
        max_ft = max_length if unit == 'ft' else max_length * 3.28

        # Primary search: by connector pairs
        for (source, target), _ in unique_pairs.items():
            extraction = ctx.filter_extractor.extract("")
            filters = extraction.filters if extraction else SearchFilters()
            filters.connector_from = source
            filters.connector_to = target

            results = ctx.search_engine.search(filters)

            for prod in results.products:
                length_ft = prod.metadata.get('length_ft')
                if length_ft and min_ft <= length_ft <= max_ft:
                    if prod.product_number not in [p.product_number for p in all_products]:
                        all_products.append(prod)

        # Fallback: if no results from connector search, try category search
        if not all_products and categories:
            ctx.add_debug(f"FALLBACK: Searching by category {categories} with keywords {subcategory_keywords}")
            fallback_config = SearchConfig(max_results=100)
            fallback_strategy = SearchStrategy(fallback_config)
            for cat in categories:
                extraction = ctx.filter_extractor.extract("")
                filters = extraction.filters if extraction else SearchFilters()
                filters.product_category = self._map_category_to_search_format(cat)
                if subcategory_keywords:
                    filters.keywords = list(subcategory_keywords)

                results = fallback_strategy.search(filters, engine=ctx.search_engine.engine)

                for prod in results.products:
                    length_ft = prod.metadata.get('length_ft')
                    if length_ft and min_ft <= length_ft <= max_ft:
                        if prod.product_number not in [p.product_number for p in all_products]:
                            all_products.append(prod)

        if not all_products:
            cable_type = self._get_product_type_from_context(ctx)
            return HandlerResult(
                response=f"I couldn't find any {cable_type} between {min_length} and {max_length} {unit}. "
                         f"Would you like me to search for a different length range?"
            )

        # Sort by length and limit to 5
        all_products.sort(key=lambda p: p.metadata.get('length_ft', 0))
        all_products = all_products[:5]

        cable_type = self._get_product_type_from_context(ctx)
        parts = [f"Here are {cable_type} between {min_length} and {max_length} {unit}:", ""]

        for i, prod in enumerate(all_products, 1):
            name = prod.metadata.get('name', prod.product_number)
            length_display = prod.metadata.get('length_display', '')
            parts.append(f"{i}. **{prod.product_number}** - {name}")
            if length_display:
                parts.append(f"   Length: {length_display}")
            parts.append("")

        parts.append("Would you like more details on any of these?")

        return HandlerResult(
            response="\n".join(parts),
            products_to_set=all_products
        )

    def _refine_by_length_threshold(
        self,
        ctx: HandlerContext,
        threshold: float,
        unit: str,
        direction: str
    ) -> HandlerResult:
        """Handle threshold length queries (e.g., "longer than 100 feet")."""
        ctx.add_debug(f"THRESHOLD REFINEMENT: {direction} than {threshold}{unit}")

        threshold_ft = threshold if unit == 'ft' else threshold * 3.28

        unique_pairs, categories, subcategory_keywords = self._collect_context_info(ctx)

        all_matches = []

        # Primary search: by connector pairs
        for (source, target), _ in unique_pairs.items():
            extraction = ctx.filter_extractor.extract("")
            filters = extraction.filters if extraction else SearchFilters()
            filters.connector_from = source
            filters.connector_to = target

            results = ctx.search_engine.search(filters)

            for prod in results.products:
                length_ft = prod.metadata.get('length_ft')
                if length_ft:
                    if direction == 'longer' and length_ft > threshold_ft:
                        all_matches.append((prod, length_ft))
                    elif direction == 'shorter' and length_ft < threshold_ft:
                        all_matches.append((prod, length_ft))

        # Fallback: category search
        if not all_matches and categories:
            ctx.add_debug(f"FALLBACK: Searching by category {categories} with keywords {subcategory_keywords}")
            fallback_config = SearchConfig(max_results=100)
            fallback_strategy = SearchStrategy(fallback_config)

            for cat in categories:
                extraction = ctx.filter_extractor.extract("")
                filters = extraction.filters if extraction else SearchFilters()
                filters.product_category = self._map_category_to_search_format(cat)
                if subcategory_keywords:
                    filters.keywords = list(subcategory_keywords)

                filters.length = threshold
                filters.length_unit = unit
                if direction == 'longer':
                    filters.length_preference = LengthPreference.EXACT_OR_LONGER
                else:
                    filters.length_preference = LengthPreference.EXACT_OR_SHORTER

                ctx.add_debug(f"FALLBACK FILTERS: category={filters.product_category}, keywords={filters.keywords}, length={filters.length}{filters.length_unit}")

                results = fallback_strategy.search(filters, engine=ctx.search_engine.engine)
                ctx.add_debug(f"FALLBACK RESULTS: {len(results.products)} products found")

                for prod in results.products:
                    length_ft = prod.metadata.get('length_ft')
                    if length_ft:
                        if direction == 'longer' and length_ft > threshold_ft:
                            all_matches.append((prod, length_ft))
                        elif direction == 'shorter' and length_ft < threshold_ft:
                            all_matches.append((prod, length_ft))

        if not all_matches:
            cable_type = self._get_product_type_from_context(ctx)
            if direction == 'longer':
                return HandlerResult(
                    response=f"I couldn't find any {cable_type} longer than {threshold} {unit}. "
                             f"Would you like to see our longest {cable_type} instead?"
                )
            else:
                return HandlerResult(
                    response=f"I couldn't find any {cable_type} shorter than {threshold} {unit}. "
                             f"Would you like to see our shortest {cable_type} instead?"
                )

        # Sort by length
        if direction == 'shorter':
            all_matches.sort(key=lambda x: x[1])
        else:
            all_matches.sort(key=lambda x: -x[1])

        # Get unique products (top 5)
        seen_skus = set()
        top_products = []
        for prod, length in all_matches:
            if prod.product_number not in seen_skus:
                seen_skus.add(prod.product_number)
                top_products.append(prod)
                if len(top_products) >= 5:
                    break

        cable_type = self._get_product_type_from_context(ctx)
        direction_word = "longer" if direction == 'longer' else "shorter"
        parts = [f"Here are {cable_type} {direction_word} than {threshold} {unit}:", ""]

        for i, prod in enumerate(top_products, 1):
            name = prod.metadata.get('name', prod.product_number)
            length_display = prod.metadata.get('length_display', '')
            parts.append(f"{i}. **{prod.product_number}** - {name}")
            if length_display:
                parts.append(f"   Length: {length_display}")
            parts.append("")

        parts.append("Would you like more details on any of these?")

        return HandlerResult(
            response="\n".join(parts),
            products_to_set=top_products
        )

    def _refine_by_relative_length(self, ctx: HandlerContext, direction: str) -> HandlerResult:
        """Handle relative length/size refinement (shorter/longer, smaller/larger)."""
        ctx.add_debug(f"RELATIVE REFINEMENT: {direction}")

        # Detect which dimension field these products use
        dimension_field = None
        current_dims = []
        for field in ('length_ft', 'screen_size_inches'):
            for prod in ctx.context.current_products:
                val = prod.metadata.get(field)
                if val:
                    if dimension_field is None:
                        dimension_field = field
                    current_dims.append((prod, val))
            if current_dims:
                break

        if not current_dims:
            return HandlerResult(
                response="I don't have length information for these products. "
                         "Could you specify what length you need?"
            )

        is_screen = dimension_field == 'screen_size_inches'
        unit = '"' if is_screen else 'ft'

        if direction == 'shorter':
            reference = min(d for _, d in current_dims)
        else:
            reference = max(d for _, d in current_dims)

        ctx.add_debug(f"Reference {dimension_field}: {reference}{unit}, looking for {direction}")

        # Reuse original search filters to preserve full search context
        all_matches = []
        last = ctx.context.last_filters

        if last:
            filters = SearchFilters(
                connector_from=last.get('connector_from'),
                connector_to=last.get('connector_to'),
                product_category=last.get('category') or last.get('product_category'),
                keywords=last.get('keywords', []),
                features=last.get('features', []),
            )
            ctx.add_debug(f"REUSING FILTERS: category={filters.product_category}, keywords={filters.keywords}")
            strategy = SearchStrategy(SearchConfig(max_results=100))
            results = strategy.search(filters, engine=ctx.search_engine.engine)

            for prod in results.products:
                val = prod.metadata.get(dimension_field)
                if val:
                    if direction == 'shorter' and val < reference:
                        all_matches.append((prod, val))
                    elif direction == 'longer' and val > reference:
                        all_matches.append((prod, val))

        # Fallback: derive category from current products
        if not all_matches and last and not (last.get('category') or last.get('product_category')):
            categories = set()
            for prod in ctx.context.current_products:
                cat = prod.metadata.get('category')
                if cat:
                    categories.add(cat)
            if categories:
                ctx.add_debug(f"FALLBACK: Deriving category from products: {categories}")
                for cat in categories:
                    filters = SearchFilters(
                        product_category=self._map_category_to_search_format(cat),
                    )
                    strategy = SearchStrategy(SearchConfig(max_results=100))
                    results = strategy.search(filters, engine=ctx.search_engine.engine)
                    for prod in results.products:
                        val = prod.metadata.get(dimension_field)
                        if val:
                            if direction == 'shorter' and val < reference:
                                all_matches.append((prod, val))
                            elif direction == 'longer' and val > reference:
                                all_matches.append((prod, val))

        if not all_matches:
            product_type = self._get_product_type_from_context(ctx)
            if is_screen:
                direction_word = "smallest" if direction == 'shorter' else "largest"
            else:
                direction_word = "shortest" if direction == 'shorter' else "longest"
            return HandlerResult(
                response=f"These are already the {direction_word} {product_type} available. "
                         f"The {direction_word} is {reference}{unit}."
            )

        # Sort by dimension
        if direction == 'shorter':
            all_matches.sort(key=lambda x: -x[1])
        else:
            all_matches.sort(key=lambda x: x[1])

        # Get top 3 unique products
        seen_skus = set()
        top_products = []
        for prod, _ in all_matches:
            if prod.product_number not in seen_skus:
                seen_skus.add(prod.product_number)
                top_products.append(prod)
                if len(top_products) >= 3:
                    break

        # Build response using LLM
        from llm.llm_response_generator import generate_response, ResponseType
        product_type = self._get_product_type_from_context(ctx)
        if is_screen:
            direction_word = "smaller" if direction == 'shorter' else "larger"
        else:
            direction_word = "shorter" if direction == 'shorter' else "longer"
        response = generate_response(
            products=top_products,
            query=f"Show me {direction_word} {product_type}",
            response_type=ResponseType.SEARCH_RESULTS,
        )

        return HandlerResult(
            response=response,
            products_to_set=top_products
        )

    def _refine_by_requirements(self, ctx: HandlerContext, keywords: list) -> HandlerResult:
        """Filter products by requirement keywords."""
        products = ctx.context.current_products
        scored = [(p, self._score_by_requirements(p, keywords)) for p in products]
        scored.sort(key=lambda x: -x[1])

        # Filter to only products that match at least one requirement
        matches = [(p, s) for p, s in scored if s > 0]

        if not matches:
            return HandlerResult(
                response="None of the current products match those requirements. "
                         "Would you like me to search for different products?"
            )

        parts = [f"Based on your requirements ({', '.join(keywords)}), here's how they rank:", ""]
        for i, (prod, score) in enumerate(matches[:5], 1):
            name = prod.metadata.get('name', prod.product_number)
            parts.append(f"{i}. **{prod.product_number}** - {name} ({score}/{len(keywords)} requirements met)")
        parts.append("")
        parts.append("Would you like more details on any of these?")

        matched_products = [p for p, _ in matches[:5]]
        return HandlerResult(
            response="\n".join(parts),
            products_to_set=matched_products
        )

    # =========================================================================
    # Helper methods used by refinement
    # =========================================================================

    def _normalize_connector(self, conn: str) -> str | None:
        """Normalize connector name to standard form."""
        from core.normalization import normalize_connector
        result = normalize_connector(conn)
        return result if result != conn else None

    def _map_category_to_search_format(self, stored_cat: str) -> str:
        """Map stored product category to search filter format."""
        mapping = {
            'cable': 'Cables',
            'cables': 'Cables',
            'network_cable': 'Network Cables',
            'fiber_cable': 'Fiber Cables',
            'adapter': 'Adapters',
            'adapters': 'Adapters',
            'dock': 'Docks',
            'docks': 'Docks',
            'hub': 'Hubs',
            'hubs': 'Hubs',
            'switch': 'Switches',
            'kvm': 'KVM Switches',
            'privacy_screen': 'Privacy Screens',
        }
        return mapping.get(stored_cat.lower(), stored_cat)

    def _get_product_type_from_context(self, ctx: HandlerContext) -> str:
        """Get a human-readable product type from context products."""
        products = ctx.context.current_products
        if not products:
            return "products"

        categories = set()
        connector_pairs = set()
        for prod in products:
            cat = prod.metadata.get('category', '')
            if cat:
                categories.add(cat.lower())
            connectors = prod.metadata.get('connectors', [])
            if connectors and len(connectors) >= 2:
                source = self._normalize_connector(connectors[0])
                target = self._normalize_connector(connectors[1])
                if source:
                    connector_pairs.add((source, target or source))

        cable_categories = {'cable', 'cables', 'network_cable', 'fiber_cable'}
        if categories & cable_categories and connector_pairs:
            if len(connector_pairs) == 1:
                source, target = list(connector_pairs)[0]
                if source == target:
                    return f"{source} cables"
                return f"{source} to {target} cables"
            all_sources = {p[0] for p in connector_pairs}
            if len(all_sources) == 1:
                return f"{list(all_sources)[0]} cables"

        if len(categories) == 1:
            cat = list(categories)[0]
            CATEGORY_DISPLAY = {
                'privacy_screen': 'privacy screens',
                'dock': 'docking stations',
                'hub': 'hubs',
                'kvm': 'KVM switches',
                'adapter': 'adapters',
                'switch': 'switches',
                'cable': 'cables',
                'network_cable': 'network cables',
                'fiber_cable': 'fiber cables',
                'enclosure': 'enclosures',
                'rack_mount': 'rack mount products',
            }
            if cat in CATEGORY_DISPLAY:
                return CATEGORY_DISPLAY[cat]
            return cat.replace('_', ' ') + 's'

        return "products"

    def _apply_criteria_to_filters(self, filters, criteria: dict) -> float | None:
        """Apply all criteria from LLM interpretation to search filters."""
        target_length_ft = criteria.get('length_ft')
        target_length_m = criteria.get('length_m')
        if target_length_m:
            target_length_ft = target_length_m * 3.28
        if target_length_ft:
            filters.length = target_length_ft
            filters.length_unit = 'ft'
            filters.length_preference = LengthPreference.EXACT_OR_LONGER

        if criteria.get('color'):
            filters.color = criteria['color']

        if criteria.get('feature'):
            if not filters.keywords:
                filters.keywords = []
            filters.keywords.append(criteria['feature'])

        if criteria.get('connector'):
            filters.connector_from = criteria['connector']

        return target_length_ft

    def _auto_search_for_criteria(
        self,
        ctx: HandlerContext,
        criteria: dict,
        filter_reason: str = "0 matches"
    ) -> HandlerResult | None:
        """Auto-search for products matching criteria when filter returns 0."""
        ctx.add_debug(f"AUTO-SEARCH: {filter_reason}, searching catalog...")

        if 'length_comparison' in criteria:
            direction = criteria['length_comparison']
            ctx.add_debug(f"AUTO-SEARCH: Relative length '{direction}' - delegating to _refine_by_relative_length")
            return self._refine_by_relative_length(ctx, direction)

        connector_pairs = set()
        categories = set()
        subcategory_keywords = set()

        for prod in ctx.context.current_products:
            connectors = prod.metadata.get('connectors', [])
            if connectors and len(connectors) >= 2:
                source = self._normalize_connector(connectors[0])
                target = self._normalize_connector(connectors[1])
                if source:
                    connector_pairs.add((source, target or source))

            cat = prod.metadata.get('category')
            if cat:
                categories.add(cat)

            subcat = prod.metadata.get('sub_category', '')
            if subcat:
                subcat_lower = subcat.lower()
                for kw in ['ethernet', 'fiber', 'hdmi', 'displayport', 'usb']:
                    if kw in subcat_lower:
                        subcategory_keywords.add(kw)

        all_results = []

        for source, target in connector_pairs:
            extraction = ctx.filter_extractor.extract("")
            filters = extraction.filters if extraction else SearchFilters()
            filters.connector_from = source
            filters.connector_to = target
            target_length_ft = self._apply_criteria_to_filters(filters, criteria)

            results = ctx.search_engine.search(filters)

            for prod in results.products:
                if prod.product_number not in [p.product_number for p in all_results]:
                    if target_length_ft:
                        prod_length = prod.metadata.get('length_ft', 0)
                        if prod_length >= target_length_ft:
                            all_results.append(prod)
                    else:
                        all_results.append(prod)

        # Fallback to category search ONLY when no connector context exists
        if not all_results and categories and not connector_pairs:
            ctx.add_debug(f"FALLBACK: Searching by category {categories}")
            for cat in categories:
                extraction = ctx.filter_extractor.extract("")
                filters = extraction.filters if extraction else SearchFilters()
                filters.product_category = self._map_category_to_search_format(cat)
                if subcategory_keywords:
                    filters.keywords = list(subcategory_keywords)
                target_length_ft = self._apply_criteria_to_filters(filters, criteria)

                results = ctx.search_engine.search(filters)

                for prod in results.products:
                    if prod.product_number not in [p.product_number for p in all_results]:
                        if target_length_ft:
                            prod_length = prod.metadata.get('length_ft', 0)
                            if prod_length >= target_length_ft:
                                all_results.append(prod)
                        else:
                            all_results.append(prod)

        if not all_results:
            return None

        has_length = criteria.get('length_ft') or criteria.get('length_m')
        if has_length:
            all_results.sort(key=lambda p: p.metadata.get('length_ft', 0))
        all_results = all_results[:5]

        ctx.add_debug(f"AUTO-SEARCH: Found {len(all_results)} products")

        from llm.llm_response_generator import generate_response, ResponseType
        response = generate_response(
            products=all_results,
            query=ctx.query,
            response_type=ResponseType.SEARCH_RESULTS,
        )

        return HandlerResult(
            response=response,
            products_to_set=all_results
        )

    def _extract_requirement_keywords(self, query: str) -> list:
        """Extract requirement keywords from query."""
        query_lower = query.lower()
        keywords = []

        if re.search(r'\b(?:dual|2|two)\s*monitors?\b', query_lower):
            keywords.append('dual monitor')
        if re.search(r'\b(?:triple|3|three)\s*monitors?\b', query_lower):
            keywords.append('triple monitor')

        if re.search(r'\b4k\b', query_lower):
            keywords.append('4K')
        if re.search(r'\b60\s*hz\b', query_lower):
            keywords.append('60Hz')

        if re.search(r'\bcharg(?:e|ing)\b|\bpower\s*delivery\b|\bpd\b', query_lower):
            keywords.append('power delivery')

        if 'ethernet' in query_lower:
            keywords.append('ethernet')
        if re.search(r'\busb[\s-]?a\b', query_lower):
            keywords.append('USB-A')
        if 'sd card' in query_lower:
            keywords.append('SD card')

        color_match = re.search(r'\b(black|white|gray|grey|silver|blue|red|green)\b', query_lower)
        if color_match:
            keywords.append(f'color:{color_match.group(1)}')

        return list(set(keywords))

    def _score_by_requirements(self, product, requirements: list) -> int:
        """Score product by how many requirements it matches."""
        from core.search.resolution import supports_4k

        score = 0
        content_lower = product.content.lower()
        features_lower = [f.lower() for f in product.metadata.get('features', [])]
        meta = product.metadata

        for req in requirements:
            req_lower = req.lower()

            if req_lower == '4k':
                if supports_4k(product):
                    score += 1
                continue

            if req_lower.startswith('color:'):
                color = req_lower.split(':', 1)[1]
                product_color = meta.get('color', '').lower()
                if color in product_color:
                    score += 1
                    continue
                if color in content_lower:
                    score += 1
                continue

            if meta.get('category') in ('dock', 'hub'):
                if req_lower == 'power delivery':
                    if meta.get('power_delivery') or meta.get('hub_power_delivery'):
                        score += 1
                        continue
                if req_lower == 'ethernet':
                    if meta.get('network_speed') or 'RJ-45' in meta.get('conn_type', ''):
                        score += 1
                        continue

            if req_lower in content_lower:
                score += 1
            elif any(req_lower in f for f in features_lower):
                score += 1

        return score
