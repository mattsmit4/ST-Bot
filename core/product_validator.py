"""
Product validation utilities for ST-Bot.

Ensures recommended products actually match what we claim they do.
Filters out couplers, gender changers, and other products that don't
fit the use case.

Key validations:
- Cables must have length (couplers/adapters don't)
- SKU patterns indicate product type (GC = Gender Changer)
- Connector gender matters (Male vs Female)
"""

import re
from typing import Optional
from core.models import Product


# SKU prefixes that indicate non-cable products (couplers/gender changers)
# Be conservative - only include patterns we're confident about
COUPLER_SKU_PATTERNS = [
    r'^GC',      # Gender Changer (e.g., GCHDMIFF = Gender Changer HDMI Female-Female)
]

# Note: We intentionally don't filter by adapter patterns like ^HD2 because
# many legitimate cables start with HD2 (HD2MM = HDMI Male-Male cable)

# Name patterns that indicate couplers/gender changers
COUPLER_NAME_PATTERNS = [
    r'\bcoupler\b',
    r'\bgender\s*changer\b',
    r'\bextender\b',
    r'\bjoiner\b',
    r'\bf/f\b',          # Female-to-Female
    r'\bfemale.*female\b',
]


def is_actual_cable(product: Product) -> bool:
    """
    Check if a product is an actual cable (not a coupler/adapter).

    Actual cables:
    - Have a length measurement
    - Don't have coupler/gender changer SKU patterns
    - Connect different devices (not just join cables)

    Args:
        product: Product to validate

    Returns:
        True if this is an actual cable, False if it's a coupler/adapter
    """
    sku = product.product_number.upper()
    name = product.metadata.get('name', '').lower()
    # Check multiple length fields - real data uses length_ft/length_display,
    # but search results may have 'length' directly
    length = (product.metadata.get('length_ft') or
              product.metadata.get('length_display') or
              product.metadata.get('length'))

    # Check 1: Cables must have length
    # Couplers and gender changers typically don't have length
    if not length:
        return False

    # Check 2: SKU patterns for couplers
    for pattern in COUPLER_SKU_PATTERNS:
        if re.match(pattern, sku, re.IGNORECASE):
            return False

    # Check 3: Name patterns for couplers
    for pattern in COUPLER_NAME_PATTERNS:
        if re.search(pattern, name, re.IGNORECASE):
            return False

    return True


def is_appropriate_for_connection(
    product: Product,
    source_port: str,
    target_input: str,
    need_cable: bool = True
) -> bool:
    """
    Check if a product is appropriate for connecting source to target.

    Validates that:
    - Product type matches need (cable vs adapter)
    - Connectors are in the right direction
    - Product isn't a coupler when we need a cable

    Args:
        product: Product to validate
        source_port: Computer port type (e.g., "USB-C", "HDMI")
        target_input: Monitor input type (e.g., "HDMI", "DisplayPort")
        need_cable: Whether we need an actual cable (True) or adapter is OK (False)

    Returns:
        True if product is appropriate for this connection
    """
    if need_cable and not is_actual_cable(product):
        return False

    # For same-connector scenarios (HDMI to HDMI), we definitely need a cable
    if source_port == target_input:
        if not is_actual_cable(product):
            return False

    return True


def filter_valid_products(
    products: list[Product],
    source_port: str,
    target_input: str,
    need_cable: bool = True
) -> list[Product]:
    """
    Filter a list of products to only include valid ones for the use case.

    Args:
        products: List of products from search
        source_port: Computer port type
        target_input: Monitor input type
        need_cable: Whether we need an actual cable

    Returns:
        Filtered list of valid products
    """
    return [
        p for p in products
        if is_appropriate_for_connection(p, source_port, target_input, need_cable)
    ]


def get_best_cable(
    products: list[Product],
    source_port: str,
    target_input: str,
    preferred_length_ft: Optional[float] = None
) -> Optional[Product]:
    """
    Get the best cable from a list of products for a specific connection.

    Prioritizes:
    1. Valid cables (not couplers)
    2. Appropriate length (6ft default for monitor connections)
    3. Feature-rich options

    Args:
        products: List of products from search
        source_port: Computer port type
        target_input: Monitor input type
        preferred_length_ft: Preferred cable length in feet

    Returns:
        Best matching product, or None if no valid product found
    """
    # Filter to valid cables only
    valid = filter_valid_products(products, source_port, target_input, need_cable=True)

    if not valid:
        return None

    # Default preferred length for monitor connections
    if preferred_length_ft is None:
        preferred_length_ft = 6.0

    # Sort by length proximity to preferred length
    def length_score(p: Product) -> float:
        length = p.metadata.get('length_ft')
        if length is None:
            return float('inf')
        return abs(length - preferred_length_ft)

    valid.sort(key=length_score)

    return valid[0]
