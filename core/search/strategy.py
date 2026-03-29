"""
Search strategies for ST-Bot.

Implements cascading search with progressive filter relaxation:
- Tier 1: Strict search with all filters
- Tier 2: Relaxed search (drop optional filters like length)
- Tier 3: Broad search (category only)

Includes deduplication, result ranking, and product validation
(filtering out couplers/gender changers from cable searches).
"""

import logging
import time
from typing import Optional
from dataclasses import dataclass
from core.models import Product, SearchFilters, SearchResult, DroppedFilter, LengthPreference
from core.product_validator import is_actual_cable
from core.search.resolution import supports_resolution

# Module-level logger
_logger = logging.getLogger(__name__)


@dataclass
class SearchConfig:
    """
    Configuration for search behavior.

    Attributes:
        min_results: Minimum results to accept a search pass
        max_results: Maximum results to return
        enable_deduplication: Remove duplicate products
        max_search_time_ms: Maximum search time in milliseconds (default 5000)
    """
    min_results: int = 1
    max_results: int = 25
    enable_deduplication: bool = True
    max_search_time_ms: int = 5000

    # Backward compatibility
    @property
    def tier1_min_results(self) -> int:
        return self.min_results

    @property
    def tier2_min_results(self) -> int:
        return self.min_results


class SearchStrategy:
    """
    Implements cascading search strategy.

    Uses progressive filter relaxation to find products:
    1. Tier 1 (Strict): Apply all filters
    2. Tier 2 (Relaxed): Drop optional filters (length, features)
    3. Tier 3 (Broad): Category only

    Example:
        strategy = SearchStrategy()
        filters = SearchFilters(
            length=6.0,
            connector_from="USB-C",
            connector_to="HDMI",
            product_category="Cables"
        )

        result = strategy.search(filters, search_func=pinecone_search)
        # Returns: SearchResult with products, tier used, filters applied
    """

    # TODO [Testing]: Add integration tests for cascading search tiers
    # Current: No tests verifying tier progression and filter relaxation
    # Required test file: tests/test_search_strategy.py
    # Test cases needed:
    #   1. Tier1 exact match - verify no relaxation
    #   2. Tier1 fail -> Tier2 - verify length dropped
    #   3. All tiers fail - verify graceful empty result
    #   4. Length preference modes (EXACT_OR_LONGER, CLOSEST)
    #   5. Timeout protection - verify partial results returned on timeout
    # Acceptance: Each tier transition has explicit test coverage
    #
    # NOTE: Timeout protection implemented (2024) - needs testing:
    #   - max_search_time_ms config added (default 5000ms)
    #   - _is_timeout() checks between each tier
    #   - Returns best partial results with search_timed_out=True flag

    def __init__(self, config: Optional[SearchConfig] = None):
        """
        Initialize search strategy.

        Args:
            config: Search configuration (uses defaults if None)
        """
        self.config = config or SearchConfig()

    def search(
        self,
        filters: SearchFilters,
        search_func: callable = None,
        available_lengths: Optional[list[float]] = None,
        engine=None,
    ) -> SearchResult:
        """
        Execute scored search against the product catalog.

        Each product is scored 0.0-1.0 based on weighted filter matching.
        Partial matches surface naturally — no multi-pass fallback needed.
        If zero results, the handler's _handle_no_results() owns fallback logic.

        Args:
            filters: Extracted search filters
            search_func: Legacy search function (unused when engine provided)
            available_lengths: Optional list of available lengths in meters
            engine: ProductSearchEngine instance for scored search

        Returns:
            SearchResult with products, match_quality, and filters used
        """
        if engine is None:
            raise ValueError("engine is required — legacy search has been removed")

        filter_dict = self._build_tier1_filters(filters)

        _logger.debug(
            "Starting scored search",
            extra={"event": "search_start", "filters": filter_dict}
        )

        scored_results = engine.search_scored(filter_dict)
        scored_results = self._filter_invalid_scored(scored_results, filters)

        _logger.debug(
            f"Scored search: {len(scored_results)} products",
            extra={"event": "search_results", "products_found": len(scored_results)}
        )

        if scored_results:
            return self._build_scored_result(
                scored_results, filter_dict, filters,
                match_quality="exact", category_relaxed=False,
                available_lengths=available_lengths
            )

        # No results — return empty, handler's _handle_no_results() owns fallbacks
        return SearchResult(
            products=[],
            filters_used=filter_dict,
            match_quality="exact",
        )

    # === Filter Building Methods ===

    def _build_tier1_filters(self, filters: SearchFilters) -> dict:
        """
        Build Tier 1 filters (strict - all filters applied).

        Args:
            filters: Extracted search filters

        Returns:
            Dictionary of filters for search
        """
        filter_dict = {}

        # Category
        if filters.product_category:
            filter_dict['category'] = filters.product_category

        # Connectors
        if filters.connector_from:
            filter_dict['connector_from'] = filters.connector_from
        if filters.connector_to:
            filter_dict['connector_to'] = filters.connector_to
            # Flag same-connector cables (HDMI-to-HDMI, USB-C-to-USB-C)
            if filters.connector_to == filters.connector_from:
                filter_dict['same_connector'] = True

        # Length (optional but included in Tier 1)
        if filters.length and filters.length_unit:
            filter_dict['length'] = filters.length
            filter_dict['length_unit'] = filters.length_unit
            filter_dict['length_preference'] = filters.length_preference

        # Features
        if filters.features:
            filter_dict['features'] = filters.features

        # Port count (for hubs, switches)
        if filters.port_count:
            filter_dict['port_count'] = filters.port_count

        # Color (optional)
        if filters.color:
            filter_dict['color'] = filters.color

        # Keywords for text matching (critical for non-cable products)
        if filters.keywords:
            filter_dict['keywords'] = filters.keywords

        # Screen size for privacy screens
        if filters.screen_size:
            filter_dict['screen_size'] = filters.screen_size

        # Cable type for network cables (Cat7, Cat6a, Cat6, Cat5e)
        if filters.cable_type:
            filter_dict['cable_type'] = filters.cable_type

        # Network speed for network cables (10Gbps, 1Gbps, etc.)
        if filters.requested_network_speed:
            filter_dict['requested_network_speed'] = filters.requested_network_speed

        # KVM video interface type (HDMI, DisplayPort, VGA, DVI)
        if filters.kvm_video_type:
            filter_dict['kvm_video_type'] = filters.kvm_video_type

        # Thunderbolt version (3 or 4)
        if filters.thunderbolt_version:
            filter_dict['thunderbolt_version'] = filters.thunderbolt_version

        # Bay count for storage enclosures
        if filters.bay_count:
            filter_dict['bay_count'] = filters.bay_count

        # Rack height for racks/cabinets
        if filters.rack_height:
            filter_dict['rack_height'] = filters.rack_height

        # Required port types (for docks/hubs: "dock with Ethernet")
        if filters.required_port_types:
            filter_dict['required_port_types'] = filters.required_port_types

        # Minimum monitor count ("dock for 3 monitors")
        if filters.min_monitors:
            filter_dict['min_monitors'] = filters.min_monitors

        # Refresh rate ("144Hz cable")
        if filters.requested_refresh_rate:
            filter_dict['requested_refresh_rate'] = filters.requested_refresh_rate

        # Power delivery wattage ("100W USB-C dock")
        if filters.requested_power_wattage:
            filter_dict['requested_power_wattage'] = filters.requested_power_wattage

        # Drive size for storage enclosures ("2.5 inch enclosure")
        if filters.drive_size:
            filter_dict['drive_size'] = filters.drive_size

        # USB version ("USB 3.0 hub")
        if filters.usb_version:
            filter_dict['usb_version'] = filters.usb_version

        return filter_dict

    def _build_scored_result(
        self,
        scored_results: list,
        filters_used: dict,
        original_filters: SearchFilters,
        match_quality: str,
        category_relaxed: bool,
        available_lengths: Optional[list[float]] = None,
    ) -> SearchResult:
        """Build SearchResult from scored results with post-processing."""
        # Extract products from scored tuples
        products = [item[0] for item in scored_results]
        total_count = len(products)

        # Compute dropped filters from unmatched keys
        dropped_filters = self._compute_dropped_filters(
            scored_results, original_filters, available_lengths
        )

        # Determine match quality from top score
        if scored_results and match_quality == "exact":
            top_score = scored_results[0][1]
            if top_score < 0.5:
                match_quality = "partial"

        # Apply post-processing (existing methods)
        products = self._filter_invalid_products(products, original_filters)
        products = self._deduplicate(products) if self.config.enable_deduplication else products
        products = self._rank_by_length_preference(products, original_filters)
        products = self._filter_unreasonable_lengths(products, original_filters)
        products = self._rank_and_limit(products, original_filters)

        # Compute per-product relevance scores for narrowing detection
        search_scores = [self._calculate_relevance(p, original_filters) for p in products]

        return SearchResult(
            products=products,
            filters_used=filters_used,
            match_quality=match_quality,
            total_count=total_count,
            original_filters=self._build_tier1_filters(original_filters),
            dropped_filters=dropped_filters,
            category_relaxed=category_relaxed,
            search_scores=search_scores,
        )

    def _filter_invalid_scored(
        self,
        scored_results: list,
        filters: SearchFilters,
        actual_category: str = None,
    ) -> list:
        """Filter invalid products from scored results (preserving scores)."""
        cable_categories = {'cables', 'cable', 'hdmi cables', 'displayport cables',
                           'usb cables', 'digital display cables'}
        category = (actual_category or filters.product_category or '').lower()

        if category in cable_categories:
            return [
                (product, score, unmatched)
                for product, score, unmatched in scored_results
                if is_actual_cable(product)
            ]
        return scored_results

    def _compute_dropped_filters(
        self,
        scored_results: list,
        original_filters: SearchFilters,
        available_lengths: Optional[list[float]] = None,
    ) -> list[DroppedFilter]:
        """
        Compute dropped filters from unmatched keys of top scored results.

        If ALL top results fail a particular filter, it is "dropped".
        """
        if not scored_results:
            return []

        # Consider top N results (what will actually be shown)
        top_n = scored_results[:self.config.max_results]

        # Collect unmatched keys across all top results
        all_unmatched_sets = [set(item[2]) for item in top_n]
        if not all_unmatched_sets:
            return []

        # Keys unmatched in ALL top results
        universally_unmatched = set.intersection(*all_unmatched_sets)

        dropped = []

        # Build DroppedFilter for each universally unmatched key
        filter_meta = {
            'length': lambda: DroppedFilter(
                filter_name="length",
                requested_value=f"{original_filters.length}{original_filters.length_unit or 'ft'}",
                reason=f"No exact {original_filters.length}{original_filters.length_unit or 'ft'} cables available",
                alternatives=self._format_length_alternatives(available_lengths) if available_lengths else None,
            ),
            'features': lambda: DroppedFilter(
                filter_name="features",
                requested_value=original_filters.features,
                reason="No products with all requested features",
            ),
            'color': lambda: DroppedFilter(
                filter_name="color",
                requested_value=original_filters.color,
                reason=f"No {original_filters.color} products found",
            ),
            'requested_refresh_rate': lambda: DroppedFilter(
                filter_name="refresh_rate",
                requested_value=f"{original_filters.requested_refresh_rate}Hz",
                reason=f"No verified {original_filters.requested_refresh_rate}Hz products found",
            ),
            'requested_power_wattage': lambda: DroppedFilter(
                filter_name="power_wattage",
                requested_value=f"{original_filters.requested_power_wattage}W",
                reason=f"No verified {original_filters.requested_power_wattage}W products found",
            ),
            'required_port_types': lambda: DroppedFilter(
                filter_name="required_port_types",
                requested_value=original_filters.required_port_types,
                reason="Not all required port types available",
            ),
            'port_count': lambda: DroppedFilter(
                filter_name="port_count",
                requested_value=original_filters.port_count,
                reason=f"No {original_filters.port_count}-port products found",
            ),
            'min_monitors': lambda: DroppedFilter(
                filter_name="min_monitors",
                requested_value=original_filters.min_monitors,
                reason=f"No products supporting {original_filters.min_monitors}+ monitors found",
            ),
            'keywords': lambda: DroppedFilter(
                filter_name="keywords",
                requested_value=original_filters.keywords,
                reason="No products matching all keywords",
            ),
        }

        for key in universally_unmatched:
            factory = filter_meta.get(key)
            if factory:
                dropped.append(factory())

        return dropped

    # === Result Processing Methods ===

    def _deduplicate(self, products: list[Product]) -> list[Product]:
        """
        Remove duplicate products based on product_number.

        Also treats marketplace variants (e.g., -VAMZ suffix for Amazon) as
        duplicates of the base product to avoid showing essentially identical
        products.

        Args:
            products: List of products (may contain duplicates)

        Returns:
            List of unique products (first occurrence kept)
        """
        seen = set()
        unique_products = []

        for product in products:
            # Get base SKU by stripping marketplace variant suffixes
            base_sku = self._get_base_sku(product.product_number)

            if base_sku not in seen:
                seen.add(base_sku)
                unique_products.append(product)

        return unique_products

    def _get_base_sku(self, sku: str) -> str:
        """
        Get base SKU by normalizing variants to avoid duplicates.

        Handles:
        - Marketplace variants: -VAMZ (Amazon)
        - Color variants: MBNL (black) vs MWNL (white) at end of SKU

        Args:
            sku: Full product SKU

        Returns:
            Normalized SKU for deduplication

        Examples:
            "CDP2HD2MBNL-VAMZ" -> "CDP2HD2MxNL"
            "CDP2HD2MBNL" -> "CDP2HD2MxNL"
            "CDP2HD2MWNL" -> "CDP2HD2MxNL"
            "CDP2HD1MBNL" -> "CDP2HD1MxNL"
            "CDP2HD1MWNL" -> "CDP2HD1MxNL"
        """
        result = sku

        # Strip marketplace variant suffixes
        variant_suffixes = ['-VAMZ']
        for suffix in variant_suffixes:
            if result.endswith(suffix):
                result = result[:-len(suffix)]

        # Normalize color variants at end of SKU
        # Pattern: ...M[B/W]NL where B=black, W=white
        # Replace with ...MxNL to treat as same product
        import re
        result = re.sub(r'M[BW]NL$', 'MxNL', result)

        return result

    def _filter_invalid_products(
        self,
        products: list[Product],
        filters: SearchFilters,
        actual_category: str = None
    ) -> list[Product]:
        """
        Filter out products that don't match the requested category type.

        When searching for cables, excludes couplers/gender changers that
        are miscategorized in the data (e.g., GCHDMIFF is in "HDMI Cables"
        but is actually a coupler with no length).

        Args:
            products: Raw products from search
            filters: Search filters (used to determine category)
            actual_category: Override category (used when tier 2.5 swaps cable→adapter)

        Returns:
            Filtered list of valid products for the category
        """
        # Only apply cable validation when searching cable categories
        cable_categories = {'cables', 'cable', 'hdmi cables', 'displayport cables',
                           'usb cables', 'digital display cables'}

        # Use actual_category if provided (tier 2.5), otherwise use filters
        category = (actual_category or filters.product_category or '').lower()

        if category in cable_categories:
            # Filter out couplers/gender changers from cable searches
            valid_products = [p for p in products if is_actual_cable(p)]
            return valid_products

        # For non-cable categories (including adapters), return as-is
        return products

    def _rank_and_limit(
        self,
        products: list[Product],
        filters: SearchFilters
    ) -> list[Product]:
        """
        Rank products by relevance and limit to max_results.

        Ranking criteria:
        1. Exact length match (if length specified)
        2. Has all requested features
        3. Similarity score (already in Product.score)

        When user indicates length flexibility (EXACT_OR_SHORTER or CLOSEST),
        ensures variety by including products at different lengths.

        Args:
            products: List of products to rank
            filters: Original search filters (for relevance scoring)

        Returns:
            Ranked and limited list of products
        """
        # Score each product
        scored_products = []
        for product in products:
            relevance_score = self._calculate_relevance(product, filters)
            scored_products.append((relevance_score, product))

        # Sort by relevance (descending) then by original score
        scored_products.sort(key=lambda x: (x[0], x[1].score), reverse=True)

        # Extract just products for further processing
        ranked_products = [product for _, product in scored_products]

        # Apply length variety if user indicated flexibility
        if self._should_diversify_lengths(filters):
            ranked_products = self._diversify_by_length(
                ranked_products, filters, limit=self.config.max_results
            )
        else:
            ranked_products = ranked_products[:self.config.max_results]

        return ranked_products

    def _should_diversify_lengths(self, filters: SearchFilters) -> bool:
        """
        Check if we should diversify results by length.

        Only diversify when user indicated length flexibility AND
        specified a length preference.

        Args:
            filters: Search filters

        Returns:
            True if length diversification should be applied
        """
        if not filters.length:
            return False

        # Diversify when user said "shorter is fine" or wants "closest"
        return filters.length_preference in (
            LengthPreference.EXACT_OR_SHORTER,
            LengthPreference.CLOSEST
        )

    def _diversify_by_length(
        self,
        products: list[Product],
        filters: SearchFilters,
        limit: int
    ) -> list[Product]:
        """
        Select products ensuring variety in cable lengths.

        When user indicates flexibility (e.g., "shorter is fine"), include
        products at different lengths rather than multiple products at the
        same length.

        Strategy:
        1. Always include the best match (closest to requested length)
        2. Include one shorter option if available and user accepts shorter
        3. Include one longer option for comparison
        4. Fill remaining slots by relevance

        Args:
            products: Ranked list of products
            filters: Search filters with length preference
            limit: Maximum number of products to return

        Returns:
            Diversified list of products
        """
        if not products or limit <= 0:
            return []

        if not filters.length or not filters.length_unit:
            return products[:limit]

        requested_m = self._normalize_length(filters.length, filters.length_unit)

        # Categorize products by length relative to request
        shorter = []  # Products shorter than requested
        longer = []  # Products longer than requested (includes "at length")

        for product in products:
            product_length = product.metadata.get('length')
            product_unit = product.metadata.get('length_unit', 'm')

            if not product_length:
                longer.append(product)  # No length info, put at end
                continue

            product_m = self._normalize_length(float(product_length), product_unit)
            diff = product_m - requested_m

            if diff < -0.05:  # Clearly shorter (more than ~2 inches under)
                shorter.append(product)
            else:
                longer.append(product)  # At or above requested length

        # Sort each category by distance from requested (closest first)
        def distance_key(p):
            pl = p.metadata.get('length')
            if not pl:
                return float('inf')
            pm = self._normalize_length(float(pl), p.metadata.get('length_unit', 'm'))
            return abs(pm - requested_m)

        shorter.sort(key=distance_key)
        longer.sort(key=distance_key)

        # Build diverse result set
        result = []

        # 1. Best match first - closest to requested (usually from longer/at_length)
        if longer:
            result.append(longer[0])
            longer = longer[1:]

        # 2. Add shorter option if user accepts shorter and one exists
        if (shorter and len(result) < limit and
            filters.length_preference in (LengthPreference.EXACT_OR_SHORTER,
                                          LengthPreference.CLOSEST)):
            result.append(shorter[0])
            shorter = shorter[1:]

        # 3. Add another longer option for comparison if we have room
        if longer and len(result) < limit:
            result.append(longer[0])
            longer = longer[1:]

        # 4. Fill remaining slots from remaining products by distance
        remaining = shorter + longer
        remaining.sort(key=distance_key)
        for product in remaining:
            if len(result) >= limit:
                break
            if product not in result:
                result.append(product)

        return result[:limit]

    def _calculate_relevance(
        self,
        product: Product,
        filters: SearchFilters
    ) -> float:
        """
        Calculate relevance score for a product based on filters.

        Args:
            product: Product to score
            filters: Search filters

        Returns:
            Relevance score (0.0 - 2.0, where 1.0+ indicates primary products)
        """
        score = 0.0
        checks = 0

        # Check length match (if specified)
        if filters.length and filters.length_unit:
            checks += 1
            product_length = product.metadata.get('length')
            product_unit = product.metadata.get('length_unit')

            if product_length and product_unit:
                # Convert to same unit for comparison
                filter_length_normalized = self._normalize_length(
                    filters.length, filters.length_unit
                )
                product_length_normalized = self._normalize_length(
                    product_length, product_unit
                )

                # Exact match or within 10% tolerance
                if abs(filter_length_normalized - product_length_normalized) / filter_length_normalized < 0.1:
                    score += 1.0

        # Check features match (if specified)
        if filters.features:
            checks += 1
            product_features = set(product.metadata.get('features', []))
            requested_features = set(filters.features)

            # Score based on how many requested features are present
            if requested_features:
                # Use unified resolution methods for resolution features
                # This ensures consistent 4K/8K/1440p/1080p detection
                resolution_features = {'4K', '8K', '1080p', '1440p'}
                matching_count = 0

                for feature in requested_features:
                    feature_upper = feature.upper() if feature else ''
                    if feature_upper in resolution_features:
                        # Use standalone resolution function
                        if supports_resolution(product, feature.lower()):
                            matching_count += 1
                    elif feature in product_features or feature.lower() in [f.lower() for f in product_features]:
                        # Standard feature matching for non-resolution features
                        matching_count += 1

                score += matching_count / len(requested_features)

        # Category-specific relevance boost
        # Boost "primary" products over accessories when searching for a category
        category_boost = self._calculate_category_boost(product, filters)

        # Return average score (or 0.5 if no checks) plus category boost
        base_score = score / checks if checks > 0 else 0.5
        return base_score + category_boost

    def _calculate_category_boost(
        self,
        product: Product,
        filters: SearchFilters
    ) -> float:
        """
        Calculate relevance boost based on category-specific metadata.

        When a user searches for "server rack", boost actual racks (have UHEIGHT)
        over rack accessories (no UHEIGHT). When searching for "PCI card", boost
        actual cards (have usb_type/PCI bus type) over tools/accessories.

        Args:
            product: Product to score
            filters: Search filters

        Returns:
            Boost value (0.0 = accessory, 1.0 = primary product)
        """
        category = (filters.product_category or '').lower()
        metadata = product.metadata
        sub_category = (metadata.get('sub_category', '') or '').lower()

        # Server racks: boost products with BOTH UHEIGHT and RACKTYPE (actual racks)
        # Products with only UHEIGHT (drawers, shelves) get partial boost
        if category in ('server racks', 'racks'):
            has_uheight = metadata.get('u_height') and str(metadata.get('u_height')).lower() not in ('nan', '')
            has_racktype = metadata.get('rack_type') and str(metadata.get('rack_type')).lower() not in ('nan', '')

            if has_uheight and has_racktype:
                return 1.0  # This is an actual rack (has both U-height and rack type)
            elif has_uheight:
                return 0.5  # Rack accessory with U-height (drawer, shelf) - partial boost
            elif 'accessories' in sub_category or 'shelves' in sub_category:
                return 0.0  # Pure accessory (cage nuts, screws)
            return 0.3  # Unknown

        # Computer cards: boost products with usb_type containing "PCI"
        if category in ('computer cards', 'cards'):
            bus_type = metadata.get('usb_type', '')
            if bus_type and 'pci' in str(bus_type).lower():
                return 1.0  # This is an actual PCI card
            elif 'accessories' in sub_category or 'tools' in sub_category:
                return 0.0  # This is a tool/accessory
            return 0.3  # Unknown

        # Storage enclosures: boost actual enclosures over accessories
        if category in ('storage enclosures', 'enclosures'):
            drive_size = metadata.get('drive_size') or metadata.get('drive_size_raw')
            if drive_size and str(drive_size).lower() not in ('nan', ''):
                return 1.0  # This is an actual enclosure
            return 0.3

        # KVM switches: boost actual switches (have kvm_ports) over KVM cables
        # KVM cables are sometimes miscategorized as "Desktop KVMs" but don't have port counts
        if category in ('kvm switches', 'kvm'):
            # Check kvm_ports field (lowercase - as stored by excel_loader)
            kvm_ports = metadata.get('kvm_ports')
            if kvm_ports and str(kvm_ports).lower() not in ('nan', 'none', ''):
                try:
                    port_count = int(float(kvm_ports))
                    if port_count >= 2:
                        return 1.0  # This is an actual KVM switch
                except (ValueError, TypeError):
                    pass
            # Check ZCONTENTITEM for "cable" indication
            content_item = str(metadata.get('content_item', '')).lower()
            if 'cable' in content_item:
                return 0.0  # This is a KVM cable, not a switch
            # Check sub_category for "cables"
            if 'cable' in sub_category:
                return 0.0  # This is a KVM cable, not a switch
            return 0.3  # Unknown - no port count, might be accessory

        # No category-specific boost needed
        return 0.0

    def _normalize_length(self, length: float, unit: str) -> float:
        """
        Normalize length to meters for comparison.

        Args:
            length: Length value
            unit: Length unit

        Returns:
            Length in meters
        """
        if unit == 'm':
            return length
        elif unit == 'ft':
            return length * 0.3048
        elif unit == 'in':
            return length * 0.0254
        elif unit == 'cm':
            return length * 0.01
        else:
            return length


    # _identify_dropped_filters removed — replaced by _compute_dropped_filters above

    def _format_length_alternatives(self, lengths_in_meters: list[float]) -> list[str]:
        """
        Format available lengths for display.

        Args:
            lengths_in_meters: List of lengths in meters

        Returns:
            List of formatted strings like "1m (3.3ft)", "2m (6.6ft)"
        """
        formatted = []
        for length_m in sorted(lengths_in_meters):
            length_ft = length_m * 3.28084
            formatted.append(f"{length_m:.0f}m ({length_ft:.1f}ft)")
        return formatted

    def _rank_by_length_preference(
        self,
        products: list[Product],
        filters: SearchFilters
    ) -> list[Product]:
        """
        Rank products by length according to user preference.

        Default behavior (EXACT_OR_LONGER): Products closest to but >= requested length first.
        EXACT_OR_SHORTER: Products closest to but <= requested length first.
        CLOSEST: Products closest in either direction first.

        Args:
            products: List of products to rank
            filters: Search filters with length preference

        Returns:
            Products sorted by length preference
        """
        if not filters.length or not filters.length_unit:
            return products

        # Convert requested length to meters for comparison
        requested_m = self._normalize_length(filters.length, filters.length_unit)

        def length_sort_key(product: Product) -> tuple:
            """
            Generate sort key for a product based on length preference.

            Returns tuple: (priority_group, distance_from_requested)
            - priority_group: 0 = preferred direction, 1 = other direction
            - distance: absolute distance from requested length
            """
            product_length = product.metadata.get('length')
            product_unit = product.metadata.get('length_unit', 'm')

            if not product_length:
                # Products without length go to the end
                return (2, float('inf'))

            product_m = self._normalize_length(float(product_length), product_unit)
            diff = product_m - requested_m

            preference = filters.length_preference

            if preference == LengthPreference.EXACT_OR_LONGER:
                # Prefer products >= requested length, sorted by smallest excess
                if diff >= -0.05:  # Small tolerance for "exact" match
                    return (0, abs(diff))
                else:
                    return (1, abs(diff))

            elif preference == LengthPreference.EXACT_OR_SHORTER:
                # Prefer products <= requested length, sorted by smallest deficit
                if diff <= 0.05:  # Small tolerance
                    return (0, abs(diff))
                else:
                    return (1, abs(diff))

            else:  # LengthPreference.CLOSEST
                # Just sort by absolute distance
                return (0, abs(diff))

        return sorted(products, key=length_sort_key)

    def _filter_unreasonable_lengths(
        self,
        products: list[Product],
        filters: SearchFilters,
        min_ratio: float = 0.25,
        max_ratio: float = 4.0
    ) -> list[Product]:
        """
        Filter out products with lengths wildly different from requested.

        When user asks for 6ft cable, returning 0.3ft or 1ft cables is unhelpful
        even if they match connectors. This filters products outside a reasonable
        range of the requested length.

        Args:
            products: List of products to filter
            filters: Search filters with length requirement
            min_ratio: Minimum acceptable ratio (0.25 = at least 25% of requested)
            max_ratio: Maximum acceptable ratio (4.0 = at most 400% of requested)

        Returns:
            Products within reasonable length range, or original list if no length filter
        """
        if not filters.length or not filters.length_unit:
            return products

        # Convert requested length to meters for comparison
        requested_m = self._normalize_length(filters.length, filters.length_unit)

        # Calculate acceptable bounds
        min_length_m = requested_m * min_ratio
        max_length_m = requested_m * max_ratio

        filtered = []
        for product in products:
            product_length = product.metadata.get('length')
            product_unit = product.metadata.get('length_unit', 'm')

            if not product_length:
                # Products without length info - include them (might be adapters, etc.)
                filtered.append(product)
                continue

            try:
                product_m = self._normalize_length(float(product_length), product_unit)
            except (ValueError, TypeError):
                filtered.append(product)
                continue

            # Check if within reasonable range
            if min_length_m <= product_m <= max_length_m:
                filtered.append(product)

        # If all products were filtered out, return original list
        # (better to show something than nothing)
        if not filtered:
            return products

        return filtered


class SearchError(Exception):
    """Raised when search fails."""
    pass
