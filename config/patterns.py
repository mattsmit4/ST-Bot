"""
Regex patterns for structured data extraction.

Contains only essential patterns for extracting structured data
like lengths, SKUs, and explicit connectors.
"""

import re

# === Length/Distance Patterns ===

# Length units
LENGTH_UNIT = r'(?:ft|feet|foot|in(?:ch(?:es)?)?|cm|centimeter(?:s)?|centimetre(?:s)?|m|meter(?:s)?|metre(?:s)?)'

# Number + unit pattern (with word boundaries)
NUM_WITH_UNIT = rf'\b\d+(?:\.\d+)?\s*{LENGTH_UNIT}\b'

# Number + unit pattern (without boundaries for internal use)
NUM_WITH_UNIT_NOB = rf'\d+(?:\.\d+)?\s*(?:ft|feet|foot|in(?:ch(?:es)?)?|cm|centimeter(?:s)?|centimetre(?:s)?|m|meter(?:s)?|metre(?:s)?)'


# === SKU Patterns ===

# StarTech product numbers (alphanumeric with optional hyphens, 3+ chars)
PRODUCT_NUMBER_PATTERN = r'[A-Z0-9-]{3,}'


# === Connector Patterns ===

# Connector types (for detection, not extraction)
CONNECTOR_PATTERN = r'\b(usb[\s\-]?c|usb[\s\-]?a|usb|hdmi|displayport|display\s*port|dp|dvi|vga|thunderbolt|3\.5\s*mm|aux|audio|rca)\b'

# Specific connector-to-connector patterns
CONNECTOR_TO_PATTERNS = {
    'usb-c_to_hdmi': r'\busb[\s\-]?c\s+to\s+hdmi\b',
    'usb-c_to_dp': r'\busb[\s\-]?c\s+to\s+(?:displayport|display\s*port|dp)\b',
    'usb-c_to_vga': r'\busb[\s\-]?c\s+to\s+vga\b',
    'hdmi_to_usb-c': r'\bhdmi\s+to\s+usb[\s\-]?c\b',
    'vga_to_hdmi': r'\bvga\s+to\s+hdmi\b',
    'hdmi_to_vga': r'\bhdmi\s+to\s+vga\b',
    'dp_to_hdmi': r'\b(?:displayport|display\s*port|dp)\s+to\s+hdmi\b',
    'dp_to_vga': r'\b(?:displayport|display\s*port|dp)\s+to\s+vga\b',
    'hdmi_to_dp': r'\bhdmi\s+to\s+(?:displayport|display\s*port|dp)\b',
    'dvi_to_hdmi': r'\bdvi\s+to\s+hdmi\b',
    'hdmi_to_dvi': r'\bhdmi\s+to\s+dvi\b',
    'dvi_to_vga': r'\bdvi\s+to\s+vga\b',
    'vga_to_dvi': r'\bvga\s+to\s+dvi\b',
}

# Single connector cable patterns (e.g., "HDMI cable", "USB-C cable")
SINGLE_CONNECTOR_PATTERNS = {
    'hdmi': r'\bhdmi\s+cables?\b',
    'displayport': r'\b(?:displayport|display\s*port|dp)\s+cables?\b',
    'vga': r'\bvga\s+cables?\b',
    'dvi': r'\bdvi\s+cables?\b',
    'usb-c': r'\busb[\s\-]?c\s+cables?\b',
    'usb-a': r'\busb[\s\-]?a\s+cables?\b',
    '3.5mm': r'\b3\.?5\s*mm\s+(?:audio\s+)?cables?\b',
    'audio': r'\baudio\s+cables?\b',
    'aux': r'\baux\s+cables?\b',
    'rca': r'\brca\s+cables?\b',
}


# === Number Patterns ===

# Numeric words for parsing
NUMBER_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
    "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "twenty": 20,
}


# === Intent Detection Patterns ===

# Greeting patterns
GREETING_PATTERNS = [
    r'\bhello\b',
    r'\bhi\b',
    r'\bhey\b',
    r'\bgood\s+morning\b',
    r'\bgood\s+afternoon\b',
    r'\bgood\s+evening\b',
]

# Farewell patterns
FAREWELL_PATTERNS = [
    r'\bthank\s*you\b',
    r'\bthanks\b',
    r'\bbye\b',
    r'\bgoodbye\b',
    r'\bsee\s+you\b',
    r'\bappreciate\s+it\b',
    r'\bcheers\b',
]

# Installation/setup request patterns (blocked)
# Note: "setup" alone is intentionally NOT included here because "multi-monitor setup"
# and "dual monitor setup" should trigger SETUP_GUIDANCE, not INSTALL_HELP.
# INSTALL_HELP is for actual installation/configuration instructions.
#
# IMPORTANT: "mount" as a noun is a product category ("monitor mount", "wall mount")
# Only match "mount" when used as a VERB (installation action):
# - "how do I mount this?" → INSTALL_HELP (blocked)
# - "mounting the bracket" → INSTALL_HELP (blocked)
# - "I need a monitor mount" → NEW_SEARCH (product search)
INSTALL_PATTERNS = [
    r'\binstall\b',
    r'\binstallation\b',
    # "set up" patterns - be careful not to match "set up dual monitors" (product search)
    r'\bhow\s+(?:do\s+i\s+|to\s+)?set\s*up\b',  # "how do I set up", "how to set up"
    r'\bset\s*up\s+(?:help|guide|instructions?|steps?)\b',  # "set up help", "set up instructions"
    r'\bsetup\s+(?:instructions?|guide|help|steps?)\b',  # "setup instructions", "setup help"
    r'\bhow\s+(?:do\s+i\s+|to\s+)?setup\b',  # "how do I setup", "how to setup"
    r'\bconfigure\b',
    r'\bconfiguration\b',
    r'\bwiring\b',
    # "mount" as VERB only - not as noun (product category)
    r'\bhow\s+(?:do\s+i\s+|to\s+)?mount\b',  # "how do I mount", "how to mount"
    r'\bmount(?:ing)?\s+(?:it|this|the|my|a)\b',  # "mount it", "mounting this", "mount the bracket"
    r'\bcan\s+(?:i|you)\s+mount\b',  # "can I mount", "can you mount"
    r'\bfirmware\b',
    r'\btroubleshoot\b',
    r'\bfix\b',
    r'\brepair\b',
]

# Pricing/discount patterns (blocked - redirect to sales)
PRICING_PATTERNS = [
    r'\b(?:price|pricing|prices|priced)\b',
    r'\bdiscounts?\b',
    r'\bcoupons?\b',
    r'\bpromo(?:tion|tional)?\s*(?:code)?\b',
    r'\bcheap(?:er|est)?\b',
    r'\bexpensive\b',
    r'\bbest\s+(?:deal|price)\b',
    r'\bquotes?\b',
    r'\bbulk\s+(?:pricing|discount|order)\b',
    r'\bwholesale\b',
    r'\bcost(?:s|ing)?\b',
    r'\bhow\s+much\s+(?:does|is|do)\b',
    r'\bsale\b',
    r'\bbudget\b',
    r'\bafford(?:able)?\b',
]

# Warranty/returns patterns (redirect to support, unless product data can answer)
# NOTE: "return" patterns must NOT match "Audio Return Channel" (ARC feature)
WARRANTY_PATTERNS = [
    r'\bwarrant(?:y|ies)\b',
    r'\bguarantee\b',
    r'\breturn\s+(?:policy|it|this|the|my|a)\b',
    r'\breturn(?:s|ed|ing)\b',
    r'\bcan\s+i\s+return\b',
    r'\brefund\b',
    r'\brma\b',
    r'\bexchange\b',
    r'\bdefective\b',
    r'\bbroken\b',
    r'\bdamaged\b',
]

# === Compiled Patterns (for performance) ===

# Pre-compile frequently used patterns
LENGTH_PATTERN = re.compile(NUM_WITH_UNIT, re.IGNORECASE)
SKU_PATTERN = re.compile(PRODUCT_NUMBER_PATTERN)
CONNECTOR_DETECT = re.compile(CONNECTOR_PATTERN, re.IGNORECASE)


# === Helper Functions ===

def extract_lengths(text: str) -> list[tuple[float, str]]:
    """
    Extract all length measurements from text.

    Args:
        text: Input text

    Returns:
        List of (value, unit) tuples

    Example:
        >>> extract_lengths("I need a 6ft or 2m cable")
        [(6.0, 'ft'), (2.0, 'm')]
    """
    matches = LENGTH_PATTERN.finditer(text)
    results = []

    for match in matches:
        text_match = match.group(0)
        # Parse number and unit
        num_match = re.search(r'(\d+(?:\.\d+)?)', text_match)
        unit_match = re.search(LENGTH_UNIT, text_match)

        if num_match and unit_match:
            value = float(num_match.group(1))
            unit = unit_match.group(0)
            results.append((value, unit))

    return results


def has_pattern(text: str, patterns: list[str]) -> bool:
    """
    Check if any pattern matches text.

    Args:
        text: Input text
        patterns: List of regex patterns

    Returns:
        True if any pattern matches
    """
    text_lower = text.lower()
    return any(re.search(pat, text_lower) for pat in patterns)
