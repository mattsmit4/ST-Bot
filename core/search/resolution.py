"""Resolution capability checking for products."""

from core.models import Product


def supports_4k(product: Product) -> bool:
    """
    Check if product supports 4K resolution.

    Checks ALL possible 4K indicators for consistency across the codebase:
    1. dock_4k_support field (for docks)
    2. features list contains '4K'
    3. max_resolution / max_dvi_resolution contains 4K indicators
    4. product content contains '4k'
    5. Inherent capability based on cable/connector type (HDMI, DisplayPort, Thunderbolt)

    Returns:
        True if product supports 4K, False otherwise
    """
    meta = product.metadata
    content_lower = (product.content or '').lower()

    # Check 1: dock_4k_support field (most authoritative for docks)
    dock_4k = meta.get('dock_4k_support', '')
    if dock_4k and str(dock_4k).lower() not in ('no', '', 'nan'):
        return True

    # Check 2: features list
    features = meta.get('features', [])
    if any('4k' in str(f).lower() for f in features):
        return True

    # Check 3: Resolution fields (max_resolution, max_dvi_resolution)
    resolution_fields = ['max_resolution', 'max_dvi_resolution']
    for field in resolution_fields:
        res_value = meta.get(field, '')
        if res_value:
            res_str = str(res_value).lower()
            # Check for 4K indicators: '4k', '2160', '3840', 'uhd', 'ultra hd'
            if any(ind in res_str for ind in ('4k', '2160', '3840', 'uhd', 'ultra hd')):
                return True

    # Check 4: Content fallback (least reliable but catches edge cases)
    if '4k' in content_lower:
        return True

    # Check 5: Inherent capability based on cable/connector type
    # HDMI, DisplayPort, and Thunderbolt cables inherently support 4K
    if _has_inherent_4k_capability(product):
        return True

    return False


def _has_inherent_4k_capability(product: Product) -> bool:
    """
    Check if product type inherently supports 4K resolution.

    Cable types that inherently support 4K:
    - HDMI cables (High Speed HDMI 1.4+ supports 4K@30Hz, HDMI 2.0+ supports 4K@60Hz)
    - DisplayPort cables (DP 1.2+ supports 4K@60Hz)
    - Thunderbolt cables (TB3/TB4 supports 4K and higher)

    Returns:
        True if cable type inherently supports 4K
    """
    meta = product.metadata
    content_lower = (product.content or '').lower()
    name_lower = meta.get('name', '').lower()
    connectors = meta.get('connectors', [])
    connector_str = ' '.join(str(c).lower() for c in connectors)
    sub_category = (meta.get('sub_category', '') or '').lower()
    category = (meta.get('category', '') or '').lower()

    # Check if this is a cable/adapter product
    is_cable = category in ('cable', 'adapter') or meta.get('length_ft') or 'cable' in sub_category

    if not is_cable:
        return False

    # VGA is analog and can't do 4K - if VGA is involved, no inherent 4K
    if 'vga' in connector_str or 'vga' in name_lower or 'vga' in sub_category:
        return False

    # DisplayPort cables inherently support 4K
    if 'displayport' in connector_str or 'displayport' in name_lower or 'dp cable' in sub_category:
        return True

    # HDMI cables inherently support 4K (HDMI 1.4+ at 30Hz, HDMI 2.0+ at 60Hz)
    if 'hdmi' in connector_str or 'hdmi' in name_lower or 'hdmi' in sub_category:
        return True

    # Thunderbolt cables inherently support 4K
    if 'thunderbolt' in connector_str or 'thunderbolt' in name_lower or 'thunderbolt' in sub_category:
        return True

    return False


def supports_resolution(product: Product, resolution: str) -> bool:
    """
    Check if product supports a specific resolution.

    Args:
        product: The product to check
        resolution: Resolution to check ('4k', '8k', '1440p', '1080p')

    Returns:
        True if product supports the resolution
    """
    res_lower = resolution.lower()

    if res_lower == '4k':
        return supports_4k(product)

    meta = product.metadata
    content_lower = (product.content or '').lower()
    features = meta.get('features', [])

    # Resolution indicators
    resolution_indicators = {
        '8k': ['8k', '4320', '7680'],
        '1440p': ['1440p', '1440', '2560', 'qhd', '2k'],
        '1080p': ['1080p', '1080', '1920', 'full hd', 'fhd'],
    }

    indicators = resolution_indicators.get(res_lower, [res_lower])

    # Check features list
    for f in features:
        f_lower = str(f).lower()
        if any(ind in f_lower for ind in indicators):
            return True

    # Check resolution fields
    resolution_fields = ['max_resolution', 'max_dvi_resolution']
    for field in resolution_fields:
        res_value = meta.get(field, '')
        if res_value:
            res_str = str(res_value).lower()
            if any(ind in res_str for ind in indicators):
                return True

    # Check content
    if any(ind in content_lower for ind in indicators):
        return True

    # Check inherent capability for common resolutions
    # If a cable supports 4K, it also supports 1080p and 1440p
    if res_lower in ('1080p', '1440p'):
        if _has_inherent_4k_capability(product):
            return True

    return False
