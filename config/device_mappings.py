"""
Single source of truth for device-to-connector mappings.

Used when users mention devices instead of explicit connectors.
e.g., "iPad to projector" -> USB-C to HDMI

Imported by:
- core/clarification.py (ClarificationResponseParser)
- llm/llm_filter_extractor.py (LLMFilterExtractor)
"""

# Device -> likely connector mapping (use regex patterns for flexibility)
# More specific patterns must come BEFORE generic patterns
DEVICE_CONNECTOR_PATTERNS = [
    # Source devices (laptops) - specific models first
    (r'\bmacbook\s*pros?\b', 'USB-C'),
    (r'\bmacbook\s*airs?\b', 'USB-C'),
    (r'\bmacbooks?\b', 'USB-C'),
    (r'\bchromebooks?\b', 'USB-C'),
    (r'\bsurface\b', 'USB-C'),
    (r'\bdell\s*xps\b', 'USB-C'),
    (r'\bthinkpads?\b', 'USB-C'),

    # Generic laptop brands (most modern laptops have USB-C)
    (r'\bdell\b', 'USB-C'),
    (r'\bhp\b', 'USB-C'),
    (r'\basus\b', 'USB-C'),
    (r'\bacer\b', 'USB-C'),
    (r'\blenovo\b', 'USB-C'),
    (r'\blaptops?\b', 'USB-C'),
    (r'\bcomputers?\b', 'USB-C'),
    (r'\bpcs?\b', 'USB-C'),

    # Phones/tablets
    (r'\bipad\s*pros?\b', 'USB-C'),
    (r'\bipads?\b', 'USB-C'),  # Modern iPads use USB-C
    (r'\biphones?\b', 'Lightning'),  # iPhones still use Lightning
    (r'\bandroid\b', 'USB-C'),
    (r'\bsamsung\b', 'USB-C'),
    (r'\bpixel\b', 'USB-C'),

    # Desktop computers
    (r'\bimacs?\b', 'USB-C'),
    (r'\bmac\s*minis?\b', 'USB-C'),
    (r'\bmac\s*studios?\b', 'USB-C'),

    # Destination devices (monitors, TVs, projectors) - what receives signal
    (r'\bmonitors?\b', 'HDMI'),
    (r'\btvs?\b', 'HDMI'),
    (r'\btelevisions?\b', 'HDMI'),
    (r'\bprojectors?\b', 'HDMI'),
    (r'\bdisplays?\b', 'HDMI'),
    (r'\bscreens?\b', 'HDMI'),

    # Peripherals - what connects via USB
    (r'\bprinters?\b', 'USB-B'),
    (r'\bscanners?\b', 'USB-B'),
]

# Devices that are typically sources (laptops, phones) - used for direction inference
SOURCE_DEVICE_PATTERNS = [
    r'\bmacbook\s*pros?\b', r'\bmacbook\s*airs?\b', r'\bmacbooks?\b',
    r'\bchromebooks?\b', r'\bsurface\b', r'\bdell\s*xps\b', r'\bthinkpads?\b',
    r'\biphones?\b', r'\bipad\s*pros?\b', r'\bipads?\b',
    r'\bandroid\b', r'\bsamsung\b', r'\bpixel\b',
    r'\bimacs?\b', r'\bmac\s*minis?\b', r'\bmac\s*studios?\b',
    r'\blaptops?\b', r'\bcomputers?\b', r'\bpcs?\b',
    r'\bdell\b', r'\bhp\b', r'\basus\b', r'\bacer\b', r'\blenovo\b',
]

# Devices that are typically destinations (displays, peripherals)
DEST_DEVICE_PATTERNS = [
    r'\bmonitors?\b', r'\btvs?\b', r'\btelevisions?\b',
    r'\bprojectors?\b', r'\bdisplays?\b', r'\bscreens?\b',
    r'\bprinters?\b', r'\bscanners?\b',
]
