"""
Single source of truth for connector normalization.

All connector name normalization across the codebase delegates here.
This prevents drift bugs from having the same mapping in 4+ places.
"""


# Combined mapping from all previous implementations:
# - core/filters.py _normalize_connector()
# - handlers/followup.py _normalize_connector()
# - llm/llm_filter_extractor.py _normalize_connector()
# - llm/response_builder.py _normalize_connector_name()
CONNECTOR_NORMALIZATION = {
    # USB connectors
    'usb-c': 'USB-C',
    'usb c': 'USB-C',
    'type-c': 'USB-C',
    'type c': 'USB-C',
    'usb-a': 'USB-A',
    'usb a': 'USB-A',
    'type-a': 'USB-A',
    'type a': 'USB-A',
    'usb': 'USB',
    # Display connectors
    'hdmi': 'HDMI',
    'mini hdmi': 'Mini HDMI',
    'mini-hdmi': 'Mini HDMI',
    'micro hdmi': 'Micro HDMI',
    'micro-hdmi': 'Micro HDMI',
    'displayport': 'DisplayPort',
    'display port': 'DisplayPort',
    'display-port': 'DisplayPort',
    'dp': 'DisplayPort',
    'mini displayport': 'Mini DisplayPort',
    'mini-displayport': 'Mini DisplayPort',
    'mdp': 'Mini DisplayPort',
    'vga': 'VGA',
    'dvi': 'DVI',
    'thunderbolt': 'Thunderbolt',
    # Network connectors
    'rj45': 'RJ45',
    'rj-45': 'RJ45',
    'ethernet': 'RJ45',
    # Audio connectors
    '3.5mm': '3.5mm',
    'audio': '3.5mm',
    'aux': '3.5mm',
    'rca': 'RCA',
    # Power connectors
    'iec': 'IEC',
    'nema': 'NEMA',
    'c14': 'IEC',
    'c13': 'IEC',
}


def normalize_connector(connector: str) -> str:
    """
    Normalize a connector name to its standard form.

    Args:
        connector: Raw connector string (e.g. 'usb-c', 'displayport', 'hdmi')

    Returns:
        Normalized connector name (e.g. 'USB-C', 'DisplayPort', 'HDMI')
    """
    if not connector:
        return ""

    connector_lower = connector.lower().replace('_', '-')

    # Exact match first
    result = CONNECTOR_NORMALIZATION.get(connector_lower)
    if result:
        return result

    # Substring match for compound names (e.g. "micro hdmi")
    for key, value in CONNECTOR_NORMALIZATION.items():
        if key in connector_lower:
            return value

    # Return as-is if unknown
    return connector
