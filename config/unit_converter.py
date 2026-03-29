"""
Smart unit conversion for product dimensions and weights.

Converts raw mm/g values to the most readable metric unit
and adds imperial equivalents in parentheses.

Examples:
    1847.400 mm  →  1.85 m (6.06 ft)
    20.500 mm    →  2.05 cm (0.81 in)
    88 g         →  88 g (3.10 oz)
    5000 g       →  5.0 kg (11.02 lbs)
"""

import re
from typing import Optional

# Fields that contain length values (stored in mm)
LENGTH_FIELDS = {
    'product_length', 'product_width', 'product_height',
    'package_length', 'package_width', 'package_height',
}

# Fields that contain weight values (stored in g)
WEIGHT_FIELDS = {
    'product_weight', 'package_weight',
}

# Conversion constants
MM_PER_INCH = 25.4
MM_PER_FOOT = 304.8
G_PER_OZ = 28.3495
G_PER_LB = 453.592


def _parse_numeric(value_str: str) -> Optional[float]:
    """Extract the numeric portion from a value string like '1847.400 mm' or '88 g'."""
    match = re.match(r'^[\s]*([\d,]+\.?\d*)', value_str.strip())
    if match:
        try:
            return float(match.group(1).replace(',', ''))
        except ValueError:
            return None
    return None


def _format_length_mm(mm: float) -> str:
    """Convert mm to the most readable metric unit + imperial equivalent."""
    # Metric scaling
    if mm < 10:
        metric = f"{mm:.1f} mm"
    elif mm < 1000:
        cm = mm / 10
        metric = f"{cm:.1f} cm"
    else:
        m = mm / 1000
        metric = f"{m:.2f} m"

    # Imperial equivalent
    inches = mm / MM_PER_INCH
    if inches < 24:
        imperial = f"{inches:.2f} in"
    else:
        feet = mm / MM_PER_FOOT
        imperial = f"{feet:.2f} ft"

    return f"{metric} ({imperial})"


def _format_weight_g(g: float) -> str:
    """Convert grams to the most readable metric unit + imperial equivalent."""
    # Metric scaling
    if g < 1000:
        metric = f"{g:.0f} g"
    else:
        kg = g / 1000
        metric = f"{kg:.1f} kg"

    # Imperial equivalent
    oz = g / G_PER_OZ
    if oz < 16:
        imperial = f"{oz:.2f} oz"
    else:
        lbs = g / G_PER_LB
        imperial = f"{lbs:.2f} lbs"

    return f"{metric} ({imperial})"


def format_measurement(value_str: str, field_name: str) -> str:
    """
    Smart-format a measurement value based on its field name.

    If the field is a known dimension or weight field, parses the numeric
    value and converts to readable metric + imperial. Otherwise returns
    the value unchanged.

    Args:
        value_str: Raw value string (e.g., "1847.400 mm", "88 g")
        field_name: Metadata field name (e.g., "product_length", "product_weight")

    Returns:
        Formatted string with smart units, or original value if not a measurement field.
    """
    if field_name in LENGTH_FIELDS:
        num = _parse_numeric(value_str)
        if num is not None and num > 0:
            return _format_length_mm(num)

    elif field_name in WEIGHT_FIELDS:
        num = _parse_numeric(value_str)
        if num is not None and num > 0:
            return _format_weight_g(num)

    # Not a measurement field or couldn't parse — return as-is
    return value_str
