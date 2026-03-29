"""
Unit tests for data/derived.py — field normalization and parsing logic.

Tests the core parsing functions that convert raw Excel data into
normalized metadata fields used by the search engine and LLM responses.
"""

import pytest
import pandas as pd
from data.derived import (
    parse_cable_length,
    _normalize_usb_version,
    extract_features,
    extract_network_rating,
    compute_derived_fields,
)


# =============================================================================
# USB VERSION NORMALIZATION
# =============================================================================

class TestNormalizeUsbVersion:
    """Test _normalize_usb_version() — maps raw strings to canonical versions."""

    def test_usb_3_2_gen_2(self):
        assert _normalize_usb_version("USB 3.2 Gen 2") == "USB 3.2 Gen 2 (10Gbps)"

    def test_usb_3_2_gen_2_with_speed(self):
        assert _normalize_usb_version("USB 3.2 Gen 2 (10Gbps)") == "USB 3.2 Gen 2 (10Gbps)"

    def test_10gbit(self):
        assert _normalize_usb_version("USB 3.2 Gen 2 - 10 Gbit/s") == "USB 3.2 Gen 2 (10Gbps)"

    def test_usb_3_0(self):
        assert _normalize_usb_version("USB 3.0") == "USB 3.0 (5Gbps)"

    def test_usb_3_2_gen_1(self):
        assert _normalize_usb_version("USB 3.2 Gen 1") == "USB 3.0 (5Gbps)"

    def test_usb_2_0(self):
        assert _normalize_usb_version("USB 2.0") == "USB 2.0"

    def test_thunderbolt_4(self):
        assert _normalize_usb_version("Thunderbolt 4") == "Thunderbolt 4"

    def test_thunderbolt_3(self):
        assert _normalize_usb_version("Thunderbolt 3 (40Gbps)") == "Thunderbolt 3"

    def test_usb4(self):
        assert _normalize_usb_version("USB4 (40Gbps)") == "USB4"

    def test_compound_thunderbolt_usb4(self):
        assert _normalize_usb_version("Thunderbolt / USB4") == "Thunderbolt / USB4"

    def test_usb_c_returns_none(self):
        """USB-C is a connector type, not a version — should return None."""
        assert _normalize_usb_version("USB-C") is None

    def test_usb_a_returns_none(self):
        """USB-A is a connector type, not a version — should return None."""
        assert _normalize_usb_version("USB-A") is None

    def test_none_returns_none(self):
        assert _normalize_usb_version(None) is None

    def test_empty_returns_none(self):
        assert _normalize_usb_version("") is None

    def test_nan_returns_none(self):
        assert _normalize_usb_version("nan") is None


# =============================================================================
# CABLE LENGTH PARSING
# =============================================================================

class TestParseCableLength:
    """Test parse_cable_length() — converts various formats to (feet, meters)."""

    def test_millimeters(self):
        ft, m = parse_cable_length("1000")
        assert abs(m - 1.0) < 0.1
        assert abs(ft - 3.28) < 0.1

    def test_small_number_as_feet(self):
        """Numbers ≤10 are assumed to be feet, not mm."""
        ft, m = parse_cable_length("6")
        assert abs(ft - 6.0) < 0.1

    def test_meters_with_unit(self):
        ft, m = parse_cable_length("2 m")
        assert abs(m - 2.0) < 0.1
        assert abs(ft - 6.56) < 0.2

    def test_feet_with_unit(self):
        ft, m = parse_cable_length("10 ft")
        assert abs(ft - 10.0) < 0.1

    def test_none_returns_none_tuple(self):
        result = parse_cable_length(None)
        assert result is None or result == (0, 0) or result == (None, None)

    def test_empty_returns_none_tuple(self):
        result = parse_cable_length("")
        assert result is None or result == (0, 0) or result == (None, None)

    def test_large_mm_value(self):
        """5000mm = 5m ≈ 16.4ft"""
        ft, m = parse_cable_length("5000")
        assert abs(m - 5.0) < 0.1
        assert abs(ft - 16.4) < 0.2


# =============================================================================
# FEATURE EXTRACTION
# =============================================================================

class TestExtractFeatures:
    """Test extract_features() — derives feature tags from product data."""

    def _make_row(self, **kwargs):
        """Create a pandas Series mimicking a product row."""
        defaults = {
            'PoE': None,
            'Description_of_Product': '',
            'Maximum_Digital_Resolutions': None,
            'Supported_Resolutions': None,
            '4K_Support': None,
        }
        defaults.update(kwargs)
        return pd.Series(defaults)

    def test_poe_from_column(self):
        row = self._make_row(PoE='Yes')
        features = extract_features(row)
        assert 'PoE' in features

    def test_poe_from_description(self):
        row = self._make_row(
            Description_of_Product='Industrial 8 Port Gigabit PoE+ Switch'
        )
        features = extract_features(row)
        assert 'PoE' in features

    def test_4k_from_support_field(self):
        row = self._make_row(**{'4K_Support': 'Yes'})
        features = extract_features(row)
        assert '4K' in features

    def test_4k_from_resolution(self):
        row = self._make_row(
            Maximum_Digital_Resolutions='3840x2160 @ 60Hz'
        )
        features = extract_features(row)
        assert '4K' in features

    def test_no_features(self):
        row = self._make_row()
        features = extract_features(row)
        assert isinstance(features, list)


# =============================================================================
# NETWORK RATING
# =============================================================================

class TestExtractNetworkRating:
    """Test extract_network_rating() — parses cable category ratings."""

    def _make_row(self, **kwargs):
        defaults = {
            'Cable_Rating': None,
            'Description_of_Product': '',
            'ProductNumber': 'TEST',
        }
        defaults.update(kwargs)
        return pd.Series(defaults)

    def test_cat6a(self):
        row = self._make_row(Cable_Rating='Cat6a')
        result = extract_network_rating(row)
        assert result is not None
        assert result['rating'] == 'Cat6a'

    def test_cat5e(self):
        row = self._make_row(Cable_Rating='Cat5e')
        result = extract_network_rating(row)
        assert result is not None
        assert result['rating'] == 'Cat5e'

    def test_case_insensitive(self):
        row = self._make_row(Cable_Rating='CAT6')
        result = extract_network_rating(row)
        assert result is not None
        assert result['rating'] == 'Cat6'

    def test_no_rating(self):
        row = self._make_row()
        result = extract_network_rating(row)
        assert result is None


# =============================================================================
# PORT COUNT DERIVATION
# =============================================================================

class TestPortCountDerivation:
    """Test port_count computation for different product categories."""

    def test_video_splitter_from_description(self):
        """8-Port splitter should get port_count=8, not 9 from raw Ports column.
        Uses real products to test the full derivation pipeline."""
        from data.loader import load_startech_products
        products = load_startech_products('ProductAttributeValues_Cleaned_Exported.xlsx')
        p = next(p for p in products if p.product_number == 'ST128HD20')
        assert p.metadata.get('port_count') == 8

    def test_dock_from_total_ports(self):
        from data.loader import load_startech_products
        products = load_startech_products('ProductAttributeValues_Cleaned_Exported.xlsx')
        p = next(p for p in products if p.product_number == 'DK31C3MNCR')
        assert p.metadata.get('port_count') == 13

    def test_hub_port_count(self):
        from data.loader import load_startech_products
        products = load_startech_products('ProductAttributeValues_Cleaned_Exported.xlsx')
        # Find any hub with known port count
        hubs = [p for p in products if p.metadata.get('category') == 'hub' and p.metadata.get('port_count')]
        assert len(hubs) > 0, "Should have at least one hub with port_count"
        for h in hubs[:3]:
            assert isinstance(h.metadata['port_count'], int)


# =============================================================================
# POWER WATTAGE EXTRACTION
# =============================================================================

class TestPowerWattageExtraction:
    """Test power_delivery_watts derivation."""

    def _compute(self, **meta):
        row = pd.Series({'ProductNumber': 'TEST', 'Description_of_Product': ''})
        metadata = {'category': 'dock'}
        metadata.update(meta)
        compute_derived_fields(row, metadata)
        return metadata

    def test_from_power_delivery_field(self):
        meta = self._compute(power_delivery='60W')
        assert meta.get('power_delivery_watts') == 60

    def test_from_100w(self):
        meta = self._compute(power_delivery='100W')
        assert meta.get('power_delivery_watts') == 100

    def test_no_pd_data(self):
        meta = self._compute()
        assert meta.get('power_delivery_watts') is None
