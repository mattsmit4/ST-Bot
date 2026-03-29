"""
Product Search Engine for ST-Bot.

Handles the actual product filtering logic against the product catalog.
Extracted from app.py to maintain proper separation of concerns.

This module is responsible for:
- Category matching (with normalization and plural handling)
- Connector matching (with strict type validation)
- Keyword matching (for text-based filtering)
- Length filtering (with unit conversion and preferences)
- Color filtering

Architecture:
- SearchStrategy (core/search.py) builds filter dictionaries for each tier
- ProductSearchEngine (this module) executes the actual filtering
- app.py creates the engine and passes it to SearchStrategy
"""

import re
import logging
from typing import List, Optional, Callable
from core.models import Product, LengthPreference
from config.category_columns import get_tier2_columns
from core.search.config import GENERIC_CATEGORIES, CONNECTOR_MATCH_CONFIG, FILTER_DISPATCH, AUXILIARY_KEYS

_logger = logging.getLogger(__name__)


class ProductSearchEngine:
    """
    Executes product searches against an in-memory product catalog.

    This class encapsulates all the filtering logic that was previously
    embedded in app.py's product_search_func.

    Usage:
        engine = ProductSearchEngine(products)
        results = engine.search(filter_dict)
    """

    # Cable type hierarchy - higher performance types are NOT compatible with lower
    # Cat7 > Cat6a > Cat6 > Cat5e > Cat5
    # If user asks for Cat7, Cat5e is NOT acceptable (wrong performance tier)
    CABLE_TYPE_HIERARCHY = {
        'Cat7': 7,
        'Cat6a': 6.5,  # 6a is between 6 and 7
        'Cat6': 6,
        'Cat5e': 5.5,  # 5e is between 5 and 6
        'Cat5': 5,
    }

    def __init__(self, products: List[Product]):
        """
        Initialize the search engine with a product catalog.

        Args:
            products: List of Product objects to search
        """
        self.products = products

    # === Scored Search Methods ===

    def _score_product(self, product: 'Product', filter_dict: dict) -> tuple:
        """
        Score a product against filter criteria.

        Returns:
            Tuple of (score: float 0.0-1.0, unmatched_keys: list[str]).
            Score 0.0 means a mandatory gate failed.
        """
        # Phase 1: Mandatory gates (any filter with is_gate=True in FILTER_DISPATCH)
        checked_gate_matchers = set()
        for key, value in filter_dict.items():
            if not value:
                continue
            dispatch = FILTER_DISPATCH.get(key)
            if not dispatch:
                continue
            method_name, weight, is_gate = dispatch
            if not is_gate or method_name in checked_gate_matchers:
                continue
            checked_gate_matchers.add(method_name)
            matcher = getattr(self, method_name)
            if not matcher(product, filter_dict):
                return (0.0, [key])

        # Phase 2: Scored filters
        total_weight = 0.0
        earned_weight = 0.0
        unmatched = []

        # Track which matchers we've already called (connector matcher handles both keys)
        called_matchers = set()

        for key, value in filter_dict.items():
            if key in AUXILIARY_KEYS or not value:
                continue

            dispatch = FILTER_DISPATCH.get(key)
            if not dispatch:
                continue

            method_name, weight, is_gate = dispatch
            if is_gate or method_name in called_matchers:
                continue

            called_matchers.add(method_name)
            total_weight += weight

            # Length uses partial credit (0.0-1.0) instead of binary
            if key == 'length':
                length_score = self._score_length(product, filter_dict)
                earned_weight += weight * length_score
                if length_score == 0.0:
                    unmatched.append(key)
            else:
                matcher = getattr(self, method_name)
                if matcher(product, filter_dict):
                    earned_weight += weight
                else:
                    unmatched.append(key)

        # Normalize score
        if total_weight == 0.0:
            return (1.0, [])

        score = earned_weight / total_weight
        return (score, unmatched)

    def search_scored(self, filter_dict: dict) -> list:
        """
        Score all products against filters. Return sorted by score descending.

        Returns:
            List of (product, score, unmatched_keys) tuples with score > 0.
        """
        results = []

        for product in self.products:
            score, unmatched = self._score_product(product, filter_dict)
            if score > 0:
                results.append((product, score, unmatched))

        # Sort by score descending, tiebreak by existing product.score
        results.sort(key=lambda x: (x[1], x[0].score), reverse=True)

        return results

    def search(self, filter_dict: dict) -> List[Product]:
        """
        Search products based on filter criteria.

        This is the main entry point called by SearchStrategy.

        Args:
            filter_dict: Dictionary of filters including:
                - category: Product category to match
                - connector_from: Source connector type
                - connector_to: Target connector type
                - same_connector: Flag for same-connector cables (HDMI-to-HDMI)
                - length: Requested length value
                - length_unit: Unit for length (ft, m, in, cm)
                - length_preference: LengthPreference enum value
                - color: Requested color
                - keywords: List of keywords for text matching
                - features: List of required features
                - port_count: Minimum port count (for hubs/switches)

        Returns:
            List of matching Product objects
        """
        results = []

        for product in self.products:
            if self._product_matches(product, filter_dict):
                results.append(product)

        return results

    def _product_matches(self, product: Product, filter_dict: dict) -> bool:
        """
        Check if a single product matches all filter criteria.

        Args:
            product: Product to check
            filter_dict: Filter criteria

        Returns:
            True if product matches all criteria
        """
        # Category filter
        if not self._matches_category(product, filter_dict):
            return False

        # Connector filters
        if not self._matches_connectors(product, filter_dict):
            return False

        # Length filter
        if not self._matches_length(product, filter_dict):
            return False

        # Color filter
        if not self._matches_color(product, filter_dict):
            return False

        # Keyword filter (NEW - was missing from original)
        if not self._matches_keywords(product, filter_dict):
            return False

        # Screen size filter (for privacy screens)
        if not self._matches_screen_size(product, filter_dict):
            return False

        # Cable type filter (for network cables - Cat7, Cat6a, etc.)
        if not self._matches_cable_type(product, filter_dict):
            return False

        # Network speed filter (for network cables - 10Gbps, 1Gbps, etc.)
        if not self._matches_network_speed(product, filter_dict):
            return False

        # KVM video type filter (HDMI, DisplayPort, VGA, DVI)
        if not self._matches_kvm_video_type(product, filter_dict):
            return False

        # Thunderbolt version filter (TB3 vs TB4)
        if not self._matches_thunderbolt_version(product, filter_dict):
            return False

        # Bay count filter (for storage enclosures)
        if not self._matches_bay_count(product, filter_dict):
            return False

        # Rack height filter (for racks, cabinets, PDUs)
        if not self._matches_rack_height(product, filter_dict):
            return False

        # Port count filter (for KVMs, hubs, switches)
        if not self._matches_port_count(product, filter_dict):
            return False

        # Drive size filter (for storage enclosures)
        if not self._matches_drive_size(product, filter_dict):
            return False

        # USB version filter (for hubs, adapters, docks)
        if not self._matches_usb_version(product, filter_dict):
            return False

        # Features filter (4K, Active, Shielded, etc.)
        if not self._matches_features(product, filter_dict):
            return False

        # Required port types (for docks/hubs: "dock with Ethernet and USB-C")
        if not self._matches_required_port_types(product, filter_dict):
            return False

        # Minimum monitor support ("dock for 3 monitors")
        if not self._matches_min_monitors(product, filter_dict):
            return False

        # Refresh rate ("144Hz DisplayPort cable")
        if not self._matches_refresh_rate(product, filter_dict):
            return False

        # Power delivery wattage ("100W USB-C cable")
        if not self._matches_power_wattage(product, filter_dict):
            return False

        return True

    # =========================================================================
    # CATEGORY MATCHING
    # =========================================================================

    def _matches_category(self, product: Product, filter_dict: dict) -> bool:
        """
        Check if product matches the requested category.

        Handles:
        - Normalization (underscores to spaces)
        - Plural forms (cables -> cable, switches -> switch)
        - Exact vs suffix matching based on category type

        Args:
            product: Product to check
            filter_dict: Filter dict with optional 'category' key

        Returns:
            True if category matches (or no category filter specified)
        """
        if 'category' not in filter_dict or not filter_dict['category']:
            return True

        product_cat = product.metadata.get('category', '').lower()
        search_cat = filter_dict['category'].lower()

        # Normalize: replace underscores with spaces
        product_cat = product_cat.replace('_', ' ')
        search_cat = search_cat.replace('_', ' ')

        # Normalize plural forms
        product_cat = self._normalize_plural(product_cat)
        search_cat = self._normalize_plural(search_cat)

        # Exact match always works
        if product_cat == search_cat:
            return True

        # For generic categories, require exact match only
        if product_cat in GENERIC_CATEGORIES:
            return False

        # For non-generic categories, allow suffix match
        # e.g., "server rack" matches product category "rack"
        if search_cat.endswith(product_cat):
            return True

        # Also check if product category ends with search category
        # e.g., search "rack" matches product "server rack"
        if product_cat.endswith(search_cat):
            return True

        return False

    def _normalize_plural(self, category: str) -> str:
        """
        Normalize plural forms and common variations in category names.

        Args:
            category: Category string to normalize

        Returns:
            Normalized category with trailing plurals removed
        """
        # Handle common variations first
        # "networking" -> "network" (extracted vs data mismatch)
        category_variations = {
            'networking': 'network',
            'storage enclosures': 'storage_enclosure',
            'storage enclosure': 'storage_enclosure',
            'display mounts': 'display_mount',
            'display mount': 'display_mount',
        }
        if category in category_variations:
            return category_variations[category]

        # Handle -ches, -shes endings (switches -> switch)
        if category.endswith('ches') or category.endswith('shes'):
            return category[:-2]
        # Handle regular -s endings (cables -> cable)
        elif category.endswith('s') and not category.endswith('ss'):
            return category[:-1]
        return category

    # =========================================================================
    # CONNECTOR MATCHING
    # =========================================================================

    def _matches_connectors(self, product: Product, filter_dict: dict) -> bool:
        """
        Check if product has the requested connectors.

        Uses strict matching to avoid false positives like matching
        "USB-C DisplayPort Alt Mode" when searching for DisplayPort.

        Args:
            product: Product to check
            filter_dict: Filter dict with optional connector keys

        Returns:
            True if connectors match (or no connector filter specified)
        """
        connectors = product.metadata.get('connectors', [])

        # Check connector_from (source/input connector)
        if 'connector_from' in filter_dict and filter_dict['connector_from']:
            if not connectors or len(connectors) < 1:
                return False

            source_connector = str(connectors[0]).lower()
            search_term = filter_dict['connector_from'].lower()

            if not self._connector_matches(source_connector, search_term):
                return False

        # Check connector_to (target/output connector)
        if 'connector_to' in filter_dict and filter_dict['connector_to']:
            # For same-connector cables (HDMI-to-HDMI, USB-C-to-USB-C), the product
            # metadata often has only ONE connector entry (e.g., ['HDMI']) since
            # both ends are identical. We already verified connector[0] matches
            # connector_from above, so for same-connector cables, we're done.
            if filter_dict.get('same_connector'):
                # Still verify connector_to to catch adapters (e.g., Mini HDMI to HDMI)
                # Even same-connector searches should reject products where one end differs
                if len(connectors) > 1:
                    target_connector = str(connectors[1]).lower()
                    search_term = filter_dict['connector_to'].lower()
                    if not self._connector_matches(target_connector, search_term):
                        return False
            else:
                # Different connectors - need 2 entries in metadata
                if not connectors or len(connectors) < 2:
                    return False

                target_connector = str(connectors[1]).lower()
                search_term = filter_dict['connector_to'].lower()

                if not self._connector_matches(target_connector, search_term):
                    return False

        return True

    def _connector_matches(self, product_connector: str, search_term: str) -> bool:
        """
        Check if a product connector matches the search term.

        Uses CONNECTOR_MATCH_CONFIG for data-driven matching. Each connector
        type defines search aliases, product patterns, and exclusions.

        For example, "1 x USB-C (24 pin) DisplayPort Alt Mode" should:
        - Match search term "USB-C" (it's the primary connector)
        - NOT match search term "DisplayPort" (excluded by 'alt mode')

        Args:
            product_connector: The connector string from product metadata
            search_term: The connector type being searched for

        Returns:
            True if the product connector is of the searched type
        """
        product_lower = product_connector.lower()
        search_lower = search_term.lower()

        # Look up connector type from config
        for config in CONNECTOR_MATCH_CONFIG.values():
            if search_lower not in config['search_aliases']:
                continue
            # Check exclusions first
            for exclusion in config['exclusions']:
                if exclusion in product_lower:
                    return False
            # Check product patterns
            for pattern in config['product_patterns']:
                if re.search(pattern, product_lower):
                    return True
            return False

        # Fallback: simple substring match for unknown connectors
        return search_lower in product_lower

    # =========================================================================
    # LENGTH MATCHING
    # =========================================================================

    def _matches_length(self, product: Product, filter_dict: dict) -> bool:
        """
        Check if product length matches the requested length.

        Handles unit conversion and length preferences (exact, shorter, longer).

        Args:
            product: Product to check
            filter_dict: Filter dict with optional length keys

        Returns:
            True if length matches (or no length filter specified)
        """
        if 'length' not in filter_dict or not filter_dict['length']:
            return True

        product_length_ft = product.metadata.get('length_ft')
        if not product_length_ft:
            return False

        requested_length = filter_dict['length']
        length_unit = filter_dict.get('length_unit', 'ft')
        length_pref = filter_dict.get('length_preference', LengthPreference.EXACT_OR_LONGER)

        # Convert requested length to feet
        if length_unit == 'm':
            requested_ft = requested_length * 3.28084
        elif length_unit == 'in':
            requested_ft = requested_length / 12.0
        elif length_unit == 'cm':
            requested_ft = requested_length / 30.48
        else:
            requested_ft = requested_length

        # Apply length preference
        if length_pref == LengthPreference.EXACT_OR_SHORTER:
            # "under X", "up to X" - only match products <= requested length
            # Use small tolerance for rounding (0.1ft = ~1 inch)
            return product_length_ft <= requested_ft + 0.1

        elif length_pref == LengthPreference.EXACT_OR_LONGER:
            # Default - match products >= requested length (with tolerance)
            tolerance_ft = max(0.5, requested_ft * 0.1)
            return product_length_ft >= requested_ft - tolerance_ft

        else:  # CLOSEST
            # Use tolerance around requested length
            tolerance_ft = max(0.5, requested_ft * 0.2)
            return abs(product_length_ft - requested_ft) <= tolerance_ft

    def _score_length(self, product: Product, filter_dict: dict) -> float:
        """
        Score length proximity on a 0.0-1.0 scale (partial credit).

        Used by _score_product() instead of binary _matches_length() so that
        products close to the requested length rank higher than distant ones.

        Scoring curve:
            Exact match (diff == 0):    1.0
            Within 0.5ft:               0.95
            Within 1ft:                 0.9
            Within 3ft:                 0.7
            Within 6ft:                 0.4
            Beyond 6ft:                 0.1

        Directional penalty (0.7x) if product is in the wrong direction
        for the user's length preference (e.g., shorter when they want longer).
        """
        if 'length' not in filter_dict or not filter_dict['length']:
            return 1.0  # No length requested — not applicable

        product_length_ft = product.metadata.get('length_ft')
        if not product_length_ft:
            return 0.0  # Product has no length data

        requested_length = filter_dict['length']
        length_unit = filter_dict.get('length_unit', 'ft')
        length_pref = filter_dict.get('length_preference', LengthPreference.EXACT_OR_LONGER)

        # Convert requested length to feet
        if length_unit == 'm':
            requested_ft = requested_length * 3.28084
        elif length_unit == 'in':
            requested_ft = requested_length / 12.0
        elif length_unit == 'cm':
            requested_ft = requested_length / 30.48
        else:
            requested_ft = requested_length

        diff = abs(product_length_ft - requested_ft)

        # Proximity score
        if diff == 0:
            score = 1.0
        elif diff <= 0.5:
            score = 0.95
        elif diff <= 1.0:
            score = 0.9
        elif diff <= 3.0:
            score = 0.7
        elif diff <= 6.0:
            score = 0.4
        else:
            score = 0.1

        # Directional penalty — wrong direction gets 0.7x
        if length_pref == LengthPreference.EXACT_OR_LONGER:
            if product_length_ft < requested_ft - 0.1:
                score *= 0.7
        elif length_pref == LengthPreference.EXACT_OR_SHORTER:
            if product_length_ft > requested_ft + 0.1:
                score *= 0.7

        return score

    # =========================================================================
    # COLOR MATCHING
    # =========================================================================

    def _matches_color(self, product: Product, filter_dict: dict) -> bool:
        """
        Check if product color matches the requested color.

        Args:
            product: Product to check
            filter_dict: Filter dict with optional 'color' key

        Returns:
            True if color matches (or no color filter specified)
        """
        if 'color' not in filter_dict or not filter_dict['color']:
            return True

        product_color = product.metadata.get('color', '').lower()
        requested_color = filter_dict['color'].lower()

        return requested_color in product_color

    # =========================================================================
    # TIER2 SEARCHABLE TEXT HELPER
    # =========================================================================

    def _get_tier2_searchable_text(self, product: Product) -> str:
        """Build searchable text from product's tier2 column values."""
        category = (product.metadata.get('category', '') or '').lower()
        columns = get_tier2_columns(category)
        parts = []
        for col in columns:
            value = product.metadata.get(col)
            if value and str(value).strip().lower() not in ('', 'nan', 'none'):
                parts.append(f"{col.lower().replace('_', ' ')} {str(value).lower()}")
        return ' '.join(parts)

    # =========================================================================
    # KEYWORD MATCHING (NEW)
    # =========================================================================

    def _matches_keywords(self, product: Product, filter_dict: dict) -> bool:
        """
        Check if product matches the requested keywords.

        Keywords are matched against:
        - Product SKU
        - Product name
        - Sub-category
        - Content string
        - Relevant metadata fields

        This is essential for queries like "serial cable" where "serial"
        is a keyword that should match products in the serial cable subcategory.

        Args:
            product: Product to check
            filter_dict: Filter dict with optional 'keywords' key

        Returns:
            True if keywords match (or no keyword filter specified)
        """
        keywords = filter_dict.get('keywords', [])
        if not keywords:
            return True

        # Build searchable text from product
        searchable_parts = [
            product.product_number.lower(),
            product.metadata.get('name', '').lower(),
            product.metadata.get('description', '').lower(),
            product.metadata.get('excel_category', '').lower(),
            product.content.lower() if product.content else '',
        ]

        # Add specific metadata fields that are commonly searched
        for field in ['nw_cable_type', 'network_rating_raw', 'usb_type', 'kvm_interface', 'fiber_type']:
            value = product.metadata.get(field, '')
            if value:
                searchable_parts.append(str(value).lower())

        # Add tier2 column values for comprehensive matching
        tier2_text = self._get_tier2_searchable_text(product)
        if tier2_text:
            searchable_parts.append(tier2_text)

        searchable_text = ' '.join(searchable_parts)

        # Check if ALL keywords match (AND logic for precision)
        # "media converter gigabit fiber" requires ALL three terms present
        for keyword in keywords:
            keyword_lower = keyword.lower()

            # Direct substring match
            if keyword_lower in searchable_text:
                continue

            # Check common variations before failing
            variations = self._get_keyword_variations(keyword_lower)
            if any(v in searchable_text for v in variations):
                continue

            # This keyword wasn't found — product doesn't match
            return False

        return True

    def _get_keyword_variations(self, keyword: str) -> List[str]:
        """
        Get common variations of a keyword for flexible matching.

        Handles:
        - Cat6/Cat6a variations
        - PoE variations
        - Common abbreviations

        Args:
            keyword: The keyword to get variations for

        Returns:
            List of keyword variations to try
        """
        variations = []

        # Cat6 should also match Cat6a (Cat6a is better than Cat6)
        # Also match just "6a" since SKUs like 6ASPAT contain "6a" not "cat6a"
        if keyword == 'cat6':
            variations.extend(['cat6a', 'cat 6', 'cat 6a', '6aspat', '6a'])
        elif keyword == 'cat5':
            variations.extend(['cat5e', 'cat 5', 'cat 5e', '45pat'])

        # PoE variations
        if keyword == 'poe':
            variations.extend(['power over ethernet', 'poe+', 'poe++'])

        # Serial variations
        if keyword == 'serial':
            variations.extend(['rs232', 'rs-232', 'db9', 'db25', 'com port'])

        # Network variations
        if keyword == 'network':
            variations.extend(['ethernet', 'lan', 'gigabit'])

        # PCIe variations for network cards
        if keyword == 'pcie':
            variations.extend(['pci express', 'pci-e'])

        # SD card reader variations (product data uses "SDHC", "SDXC", "microSD", etc.)
        if 'sd card' in keyword or 'card reader' in keyword:
            variations.extend(['sdhc', 'sdxc', 'microsd', 'secure digital', 'memory media', 'cfast'])

        return variations


    # =========================================================================
    # SCREEN SIZE MATCHING (for Privacy Screens)
    # =========================================================================

    def _matches_screen_size(self, product: Product, filter_dict: dict) -> bool:
        """
        Check if product screen size matches the requested size (for privacy screens).

        Uses derived metadata['screen_size_inches'] (float) computed at load
        time from Screen_Size column. Allows 0.7 inch tolerance for rounding
        (e.g., user says 15, product is 15.6).

        Args:
            product: Product to check
            filter_dict: Filter dict with optional 'screen_size' key

        Returns:
            True if screen size matches (or no screen_size filter specified)
        """
        requested_size = filter_dict.get('screen_size')
        if not requested_size:
            return True

        product_size = product.metadata.get('screen_size_inches')
        if product_size is None:
            return False

        # Allow 0.7 inch tolerance (user says 15, product is 15.6)
        return abs(product_size - float(requested_size)) <= 0.7

    # =========================================================================
    # CABLE TYPE MATCHING (Network Cables)
    # =========================================================================

    def _matches_cable_type(self, product: Product, filter_dict: dict) -> bool:
        """
        Check if product cable type matches the requested type.

        Network cables have specific performance tiers. When user asks for Cat7,
        they should NOT get Cat5e cables - that's a significant downgrade.

        Args:
            product: Product to check
            filter_dict: Filter dict with optional 'cable_type' key

        Returns:
            True if cable type matches (or no cable_type filter specified)
        """
        requested_type = filter_dict.get('cable_type')
        if not requested_type:
            return True

        # Build searchable text from product
        sku = product.product_number.upper()
        name = product.metadata.get('name', '').upper()
        nw_cable_type = str(product.metadata.get('nw_cable_type', '')).upper()
        content = (product.content or '').upper()

        # Extract cable type from product
        product_type = self._extract_product_cable_type(sku, name, nw_cable_type, content)

        if not product_type:
            # Product doesn't have a recognizable cable type - exclude it
            return False

        # Get hierarchy values
        requested_value = self.CABLE_TYPE_HIERARCHY.get(requested_type, 0)
        product_value = self.CABLE_TYPE_HIERARCHY.get(product_type, 0)

        # Product must be at least as good as requested
        # Cat7 request → only Cat7 (exact match for highest tier)
        # Cat6 request → Cat6 or Cat6a (equal or better)
        # Cat5 request → Cat5 or Cat5e (equal or better)
        return product_value >= requested_value

    def _extract_product_cable_type(
        self,
        sku: str,
        name: str,
        nw_cable_type: str,
        content: str
    ) -> Optional[str]:
        """
        Extract cable type from product metadata.

        Checks SKU, name, nw_cable_type metadata field, and content.

        Args:
            sku: Product SKU (uppercase)
            name: Product name (uppercase)
            nw_cable_type: nw_cable_type metadata field (uppercase)
            content: Product content/description (uppercase)

        Returns:
            Cable type (e.g., "Cat7") or None if not identifiable
        """
        searchable = f"{sku} {name} {nw_cable_type} {content}"

        # Check in order of specificity (Cat6a before Cat6, Cat5e before Cat5)
        cable_patterns = [
            (r'\bCAT\s*7\b', 'Cat7'),
            (r'\bCAT\s*6A\b', 'Cat6a'),
            (r'\bCAT\s*6\b', 'Cat6'),
            (r'\bCAT\s*5E\b', 'Cat5e'),
            (r'\bCAT\s*5\b', 'Cat5'),
        ]

        for pattern, cable_type in cable_patterns:
            if re.search(pattern, searchable):
                return cable_type

        return None

    # =========================================================================
    # NETWORK SPEED MATCHING
    # =========================================================================

    def _matches_network_speed(self, product: Product, filter_dict: dict) -> bool:
        """
        Check if product supports the requested network speed.

        Products with speed >= requested speed are considered matches.
        E.g., Cat6a (10Gbps) matches a 1Gbps request.

        Args:
            product: Product to check
            filter_dict: Filter dict with optional 'requested_network_speed' key

        Returns:
            True if product supports at least the requested speed
        """
        requested_speed = filter_dict.get('requested_network_speed')
        if not requested_speed:
            return True

        product_speed = self._parse_product_network_speed(product)

        if product_speed is None:
            # Can't determine speed - exclude when user explicitly requested a speed
            # This prevents products with unknown cable type from appearing in
            # "10Gbps ethernet cable" results
            return False

        return product_speed >= requested_speed

    def _parse_product_network_speed(self, product: Product) -> Optional[int]:
        """
        Parse network speed from product metadata.

        Extracts speed in Mbps from network_speed field or infers from cable type.

        Args:
            product: Product to parse

        Returns:
            Speed in Mbps (e.g., 10000 for 10Gbps) or None if unknown
        """
        # Try network_speed metadata field first
        network_speed = str(product.metadata.get('network_speed', ''))

        if network_speed:
            # Parse common formats: "10 Gigabit", "10Gbps", "10G", "1000Mbps"
            speed_patterns = [
                (r'(\d+)\s*gbps', 1000),       # "10Gbps" → 10000
                (r'(\d+)\s*gigabit', 1000),    # "10 Gigabit" → 10000
                (r'(\d+)\s*g\b', 1000),        # "10G" → 10000
                (r'(\d+)\s*mbps', 1),          # "1000Mbps" → 1000
            ]

            for pattern, multiplier in speed_patterns:
                match = re.search(pattern, network_speed, re.IGNORECASE)
                if match:
                    return int(match.group(1)) * multiplier

        # Fallback: infer from cable type
        # Cable type → speed mapping (conservative estimates)
        cable_speed_map = {
            'Cat8': 25000,   # 25Gbps (conservative, can do 40Gbps)
            'Cat7': 10000,   # 10Gbps
            'Cat6a': 10000,  # 10Gbps
            'Cat6': 1000,    # 1Gbps (10Gbps at short distances, but rated 1Gbps)
            'Cat5e': 1000,   # 1Gbps
            'Cat5': 100,     # 100Mbps
        }

        # First check network_rating field (most reliable source)
        network_rating = product.metadata.get('network_rating', '')
        if network_rating:
            # Normalize to match our map keys (e.g., "Cat5e" → "Cat5e")
            for cable_type in cable_speed_map.keys():
                if cable_type.lower() == network_rating.lower():
                    return cable_speed_map[cable_type]

        # Fallback to pattern matching in other fields
        cable_type = self._extract_product_cable_type(
            product.product_number.upper(),
            product.metadata.get('name', '').upper(),
            str(product.metadata.get('nw_cable_type', '')).upper(),
            (product.content or '').upper()
        )

        return cable_speed_map.get(cable_type)

    # =========================================================================
    # KVM VIDEO TYPE MATCHING
    # =========================================================================

    def _matches_kvm_video_type(self, product: Product, filter_dict: dict) -> bool:
        """
        Check if KVM product has the requested video interface.

        KVM switches support specific video interfaces (HDMI, DisplayPort, VGA, DVI).
        When user asks for "HDMI KVM", we should only show KVMs with HDMI support.

        Args:
            product: Product to check
            filter_dict: Filter dict with optional 'kvm_video_type' key

        Returns:
            True if video type matches (or no kvm_video_type filter specified)
        """
        requested_type = filter_dict.get('kvm_video_type')
        if not requested_type:
            return True

        # Check kvm_video_type metadata field directly (reliably populated)
        kvm_video = str(product.metadata.get('kvm_video_type', '')).strip()
        if not kvm_video:
            return False  # No video type data = can't match

        return requested_type.lower() in kvm_video.lower()

    # =========================================================================
    # THUNDERBOLT VERSION MATCHING
    # =========================================================================

    def _matches_thunderbolt_version(self, product: Product, filter_dict: dict) -> bool:
        """
        Check if product has the requested Thunderbolt version.

        Thunderbolt 4 products should match TB4 requests.
        Thunderbolt 3 products should NOT match TB4 requests (TB4 has stricter requirements).
        TB4 products CAN match TB3 requests (backwards compatible).

        Args:
            product: Product to check
            filter_dict: Filter dict with optional 'thunderbolt_version' key

        Returns:
            True if Thunderbolt version matches (or no version filter specified)
        """
        requested_version = filter_dict.get('thunderbolt_version')
        if not requested_version:
            return True

        # Build searchable text — include metadata fields that carry version info
        sku = product.product_number.upper()
        name = product.metadata.get('name', '').upper()
        content = (product.content or '').upper()
        meta_fields = ' '.join(
            str(product.metadata.get(f, '')).upper()
            for f in ('usb_type', 'hub_usb_type', 'type_and_rate', 'sub_category')
        )

        searchable = f"{sku} {name} {content} {meta_fields}"

        # Detect product's Thunderbolt version
        product_version = None

        if re.search(r'\bTHUNDERBOLT\s*5\b|\bTB5\b', searchable):
            product_version = 5
        elif re.search(r'\bTHUNDERBOLT\s*4\b|\bTB4\b', searchable):
            product_version = 4
        elif re.search(r'\bTHUNDERBOLT\s*3\b|\bTB3\b', searchable):
            product_version = 3

        if not product_version:
            # Product says "Thunderbolt" without version - check if it's a Thunderbolt product
            # Generic Thunderbolt products are typically TB3 compatible
            has_thunderbolt = re.search(r'\bTHUNDERBOLT\b', searchable)
            if has_thunderbolt:
                # Generic Thunderbolt - accept for TB3 requests, not for TB4+
                return requested_version == 3
            return False

        # Version matching logic (backward compatibility):
        # - TB5 request: only TB5
        # - TB4 request: TB4 or TB5 (TB5 is backward compatible)
        # - TB3 request: TB3, TB4, or TB5
        if requested_version == 5:
            return product_version == 5
        elif requested_version == 4:
            return product_version >= 4  # TB4 or TB5
        elif requested_version == 3:
            return product_version >= 3  # TB3, TB4, or TB5

        return True

    # =========================================================================
    # BAY COUNT MATCHING (Storage Enclosures)
    # =========================================================================

    def _matches_bay_count(self, product: Product, filter_dict: dict) -> bool:
        """
        Check if storage enclosure has the requested number of drive bays.

        Uses derived metadata['bay_count'] computed at load time from
        Number_of_Drives, Number_of_2.5_Inch_Bays, etc.

        Args:
            product: Product to check
            filter_dict: Filter dict with optional 'bay_count' key

        Returns:
            True if bay count matches (or no bay_count filter specified)
        """
        requested_bays = filter_dict.get('bay_count')
        if not requested_bays:
            return True

        product_bays = product.metadata.get('bay_count')
        if product_bays is None:
            return False

        return product_bays == requested_bays

    # =========================================================================
    # RACK HEIGHT MATCHING (Racks, Cabinets, PDUs)
    # =========================================================================

    def _matches_rack_height(self, product: Product, filter_dict: dict) -> bool:
        """
        Check if rack product matches requested U-height (or larger).

        Uses derived metadata['rack_height_u'] computed at load time
        from U_Height column.

        Args:
            product: Product to check
            filter_dict: Filter dict with optional 'rack_height' key

        Returns:
            True if rack height matches (or no rack_height filter specified)
        """
        requested_height = filter_dict.get('rack_height')
        if not requested_height:
            return True

        product_height = product.metadata.get('rack_height_u')
        if product_height is None:
            return False

        return product_height >= requested_height

    # =========================================================================
    # PORT COUNT MATCHING (KVMs, Hubs, Switches)
    # =========================================================================

    def _matches_port_count(self, product: Product, filter_dict: dict) -> bool:
        """
        Check if product has at least the requested number of ports.

        Uses derived metadata['port_count'] computed at load time from
        the appropriate source (kvm_ports, hub_ports, total_ports) based
        on product category.

        Args:
            product: Product to check
            filter_dict: Filter dict with optional 'port_count' key

        Returns:
            True if port count matches (or no port_count filter specified)
        """
        requested_ports = filter_dict.get('port_count')
        if not requested_ports:
            return True

        product_ports = product.metadata.get('port_count')
        if product_ports is None:
            return False

        return product_ports >= requested_ports

    # =========================================================================
    # DRIVE SIZE MATCHING (Storage Enclosures)
    # =========================================================================

    def _matches_drive_size(self, product: Product, filter_dict: dict) -> bool:
        """
        Check if storage enclosure supports the requested drive size.

        Uses derived metadata['drive_size'] computed at load time from
        Drive_Size column. Normalized to "2.5\"", "3.5\"", "M.2 NVMe", etc.

        Args:
            product: Product to check
            filter_dict: Filter dict with optional 'drive_size' key

        Returns:
            True if drive size matches (or no drive_size filter specified)
        """
        requested_size = filter_dict.get('drive_size')
        if not requested_size:
            return True

        product_size = product.metadata.get('drive_size')
        if product_size is None:
            return False

        req_lower = requested_size.lower()
        prod_lower = product_size.lower()

        # M.2 NVMe / M.2 SATA / M.2 matching
        if 'm.2' in req_lower:
            if 'nvme' in req_lower:
                return 'nvme' in prod_lower
            elif 'sata' in req_lower:
                return 'm.2 sata' in prod_lower or 'msata' in prod_lower
            else:
                return 'm.2' in prod_lower
        if 'msata' in req_lower:
            return 'msata' in prod_lower or 'm.2 sata' in prod_lower

        # Multi-size: "2.5, 3.5" → check each size individually
        if ',' in req_lower:
            sizes = [s.strip().replace('"', '') for s in req_lower.split(',')]
            return all(
                size in prod_lower.replace('"', '') for size in sizes if size
            )

        # Numeric sizes: "2.5" matches "2.5\"" or "2.5\"/3.5\""
        if req_lower.replace('"', '') in prod_lower.replace('"', ''):
            return True

        return False

    # =========================================================================
    # USB VERSION MATCHING (Hubs, Adapters, Docks)
    # =========================================================================

    def _matches_usb_version(self, product: Product, filter_dict: dict) -> bool:
        """
        Check if product meets the requested USB version.

        Uses derived metadata['usb_version'] computed at load time from
        Bus_Type column. Normalized to "USB 3.2 Gen 2 (10Gbps)", "USB 3.0 (5Gbps)", etc.

        USB 3.2 Gen 2 >= USB 3.0 >= USB 2.0 (higher versions satisfy lower requests).

        Args:
            product: Product to check
            filter_dict: Filter dict with optional 'usb_version' key

        Returns:
            True if USB version matches (or no usb_version filter specified)
        """
        requested_version = filter_dict.get('usb_version')
        if not requested_version:
            return True

        product_version = product.metadata.get('usb_version')
        if product_version is None:
            return False

        # Map versions to numeric ranks for comparison
        def _usb_rank(version_str: str) -> float:
            v = version_str.lower()
            if '3.2 gen 2' in v or '10g' in v:
                return 3.2
            elif '3.2 gen 1' in v or '3.1' in v or '3.0' in v or '5g' in v:
                return 3.0
            elif '2.0' in v:
                return 2.0
            elif '1.1' in v:
                return 1.1
            return 0.0

        return _usb_rank(product_version) >= _usb_rank(requested_version)

    # =========================================================================
    # FEATURES MATCHING (4K, Active, Shielded, etc.)
    # =========================================================================

    def _matches_features(self, product: Product, filter_dict: dict) -> bool:
        """
        Check if product has ALL requested features.

        Features come from queries like "4K HDMI cable", "active DisplayPort adapter".
        Checks product metadata features list, name, and content.

        Args:
            product: Product to check
            filter_dict: Filter dict with optional 'features' key

        Returns:
            True if all features match (or no features filter specified)
        """
        requested_features = filter_dict.get('features')
        if not requested_features:
            return True

        # Build searchable text from product
        product_features = product.metadata.get('features', [])
        if isinstance(product_features, str):
            product_features = [f.strip() for f in product_features.split(',')]

        features_lower = [f.lower() for f in product_features if f]

        name = product.metadata.get('name', '').lower()
        content = (product.content or '').lower()
        tier2_text = self._get_tier2_searchable_text(product)

        # Combined text for word-level fallback
        combined_text = ' '.join(features_lower) + ' ' + name + ' ' + content + ' ' + tier2_text

        # Check each requested feature
        for feature in requested_features:
            feature_lower = feature.lower()

            # Check in features list
            if any(feature_lower in f for f in features_lower):
                continue

            # Check in name or content
            if feature_lower in name or feature_lower in content:
                continue

            # Check in tier2 metadata values
            if feature_lower in tier2_text:
                continue

            # Word-level fallback for multi-word features:
            # Split into words, apply minimal stemming, check all appear
            words = feature_lower.split()
            if len(words) >= 2:
                stemmed = []
                for w in words:
                    for suffix in ('ing', 'ed', 's'):
                        if w.endswith(suffix) and len(w) - len(suffix) >= 3:
                            w = w[:-len(suffix)]
                            break
                    stemmed.append(w)
                if all(sw in combined_text for sw in stemmed):
                    continue

            # Feature not found
            return False

        return True

    # =========================================================================
    # REQUIRED PORT TYPES MATCHING (for Docks/Hubs)
    # =========================================================================

    def _matches_required_port_types(self, product: Product, filter_dict: dict) -> bool:
        """
        Check if product has all required port types.

        Used for queries like "dock with Ethernet and USB-C", "hub with HDMI output".
        Checks conn_type, name, content, and features for port type mentions.

        Args:
            product: Product to check
            filter_dict: Filter dict with optional 'required_port_types' key

        Returns:
            True if all port types found (or no required_port_types filter specified)
        """
        required_ports = filter_dict.get('required_port_types')
        if not required_ports:
            return True

        # Build searchable text
        conn_type = str(product.metadata.get('conn_type', '')).lower()
        name = product.metadata.get('name', '').lower()
        content = (product.content or '').lower()

        product_features = product.metadata.get('features', [])
        if isinstance(product_features, str):
            product_features = [f.strip() for f in product_features.split(',')]
        features_text = ' '.join(str(f).lower() for f in product_features if f)

        searchable = f"{conn_type} {name} {content} {features_text}"

        # Port type patterns
        port_patterns = {
            'USB-C': [r'usb[\s\-]?c', r'type[\s\-]?c'],
            'USB-A': [r'usb[\s\-]?a', r'type[\s\-]?a'],
            'USB': [r'\busb\b'],
            'HDMI': [r'\bhdmi\b'],
            'DisplayPort': [r'\bdisplayport\b', r'\bdp\b'],
            'Thunderbolt': [r'\bthunderbolt\b'],
            'Ethernet': [r'\bethernet\b', r'\brj[\s\-]?45\b', r'\bgigabit\b'],
            'VGA': [r'\bvga\b'],
            'DVI': [r'\bdvi\b'],
        }

        for port_type in required_ports:
            patterns = port_patterns.get(port_type, [re.escape(port_type.lower())])
            found = any(re.search(p, searchable) for p in patterns)
            if not found:
                return False

        return True

    # =========================================================================
    # MINIMUM MONITORS MATCHING (Docks, Mounts)
    # =========================================================================

    def _matches_min_monitors(self, product: Product, filter_dict: dict) -> bool:
        """
        Check if product supports at least the requested number of monitors.

        Checks dock_num_displays, mount_num_displays, and Displays_Supported.
        Handles values like "3", "Up to 3", etc.

        Args:
            product: Product to check
            filter_dict: Filter dict with optional 'min_monitors' key

        Returns:
            True if monitor count sufficient (or no min_monitors filter specified)
        """
        min_monitors = filter_dict.get('min_monitors')
        if not min_monitors:
            return True

        metadata = product.metadata

        # Check multiple fields where monitor count might be stored
        for field in ['dock_num_displays', 'mount_num_displays', 'Displays_Supported']:
            value = metadata.get(field)
            if value:
                # Extract number from value (handles "3", "Up to 3", "3.0")
                match = re.search(r'(\d+)', str(value))
                if match:
                    product_displays = int(match.group(1))
                    if product_displays >= min_monitors:
                        return True

        # Fallback: check name and content for monitor count
        name = metadata.get('name', '').lower()
        content = (product.content or '').lower()
        searchable = f"{name} {content}"

        # Patterns like "triple monitor", "dual monitor", "supports 3 displays"
        word_to_num = {'dual': 2, 'triple': 3, 'quad': 4}
        for word, num in word_to_num.items():
            if word in searchable and num >= min_monitors:
                return True

        monitor_match = re.search(r'(\d+)\s*(?:monitor|display|screen)', searchable)
        if monitor_match:
            if int(monitor_match.group(1)) >= min_monitors:
                return True

        return False

    # =========================================================================
    # REFRESH RATE MATCHING (144Hz, 120Hz, etc.)
    # =========================================================================

    def _matches_refresh_rate(self, product: Product, filter_dict: dict) -> bool:
        """
        Check if product supports at least the requested refresh rate.

        Uses derived metadata['max_refresh_rate'] (integer Hz) computed
        at load time from resolution columns.

        Args:
            product: Product to check
            filter_dict: Filter dict with optional 'requested_refresh_rate' key

        Returns:
            True if refresh rate sufficient (or no refresh rate filter specified)
        """
        requested_hz = filter_dict.get('requested_refresh_rate')
        if not requested_hz:
            return True

        max_rate = product.metadata.get('max_refresh_rate')
        if max_rate is not None:
            return max_rate >= requested_hz

        # 60Hz is default for most video cables — assume supported if not stated
        if requested_hz <= 60:
            features = product.metadata.get('features', [])
            if any(f in features for f in ['4K', '8K', '1440p', '1080p']):
                return True

        return False

    # =========================================================================
    # POWER WATTAGE MATCHING (100W, 60W PD, etc.)
    # =========================================================================

    def _matches_power_wattage(self, product: Product, filter_dict: dict) -> bool:
        """
        Check if product supports at least the requested power delivery wattage.

        Uses derived metadata['power_delivery_watts'] (integer) computed
        at load time from power-related columns.

        Args:
            product: Product to check
            filter_dict: Filter dict with optional 'requested_power_wattage' key

        Returns:
            True if wattage sufficient (or no power wattage filter specified)
        """
        requested_watts = filter_dict.get('requested_power_wattage')
        if not requested_watts:
            return True

        product_watts = product.metadata.get('power_delivery_watts')
        if product_watts is None:
            return True  # No data = benefit of the doubt

        return product_watts >= requested_watts


def create_search_func(products: List[Product]) -> Callable[[dict], List[Product]]:
    """
    Factory function to create a search function for use with SearchStrategy.

    This provides the same interface as the old product_search_func but
    uses the new ProductSearchEngine internally.

    Args:
        products: List of products to search

    Returns:
        A callable that takes filter_dict and returns matching products

    Usage:
        search_func = create_search_func(products)
        results = search_func({'category': 'Cables', 'connector_from': 'HDMI'})
    """
    engine = ProductSearchEngine(products)
    return engine.search
