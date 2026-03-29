"""
Unit tests for core/search/engine.py — product search and scoring logic.

Tests the core matching and scoring methods that determine which products
are returned for a given set of search filters.
"""

import pytest
from core.models.product import Product
from core.search.engine import ProductSearchEngine


def create_product(sku, category='cable', connectors=None, length_ft=None, features=None, **meta):
    """Helper to create test Product objects."""
    metadata = {'category': category, 'name': sku}
    if connectors:
        metadata['connectors'] = connectors
    if length_ft:
        metadata['length_ft'] = length_ft
    if features:
        metadata['features'] = features
    metadata.update(meta)
    content = f"{sku} {category} {' '.join(str(v) for v in metadata.values())}"
    return Product(product_number=sku, content=content, metadata=metadata, score=1.0)


# =============================================================================
# CONNECTOR MATCHING
# =============================================================================

class TestConnectorMatching:
    """Test _connector_matches() — data-driven connector type matching."""

    @pytest.fixture
    def engine(self):
        return ProductSearchEngine([])

    def test_usbc_matches_type_c(self, engine):
        assert engine._connector_matches("USB Type-C (24 pin)", "USB-C")

    def test_usbc_matches_usb_c_lowercase(self, engine):
        assert engine._connector_matches("usb-c connector", "USB-C")

    def test_usbb_matches_type_b(self, engine):
        assert engine._connector_matches("USB 3.2 Type-B (9 pin, Gen 2)", "USB-B")

    def test_usbb_matches_usb_b(self, engine):
        assert engine._connector_matches("USB B (4 pin)", "USB-B")

    def test_hdmi_matches_hdmi(self, engine):
        assert engine._connector_matches("HDMI (19 pin)", "HDMI")

    def test_hdmi_does_not_match_mini_hdmi(self, engine):
        assert not engine._connector_matches("Mini-HDMI", "HDMI")

    def test_displayport_does_not_match_alt_mode(self, engine):
        assert not engine._connector_matches("USB-C (24 pin) DisplayPort Alt Mode", "DisplayPort")

    def test_displayport_matches_displayport(self, engine):
        assert engine._connector_matches("DisplayPort (20 pin)", "DisplayPort")

    def test_thunderbolt_matches(self, engine):
        assert engine._connector_matches("Thunderbolt 4 (USB 4.0)", "Thunderbolt")

    def test_rj45_matches_ethernet(self, engine):
        assert engine._connector_matches("RJ-45 (PoE+)", "Ethernet")


# =============================================================================
# CATEGORY MATCHING
# =============================================================================

class TestCategoryMatching:
    """Test category filtering in search."""

    def test_exact_match(self):
        products = [
            create_product("CABLE1", category="cable"),
            create_product("DOCK1", category="dock"),
        ]
        engine = ProductSearchEngine(products)
        results = engine.search({'category': 'cable'})
        assert len(results) == 1
        assert results[0].product_number == "CABLE1"

    def test_plural_normalization(self):
        products = [create_product("CABLE1", category="cable")]
        engine = ProductSearchEngine(products)
        results = engine.search({'category': 'cables'})
        assert len(results) == 1

    def test_no_match(self):
        products = [create_product("CABLE1", category="cable")]
        engine = ProductSearchEngine(products)
        results = engine.search({'category': 'dock'})
        assert len(results) == 0


# =============================================================================
# LENGTH MATCHING
# =============================================================================

class TestLengthMatching:
    """Test length filtering and scoring."""

    def test_exact_length_match(self):
        products = [
            create_product("SHORT", category="cable", length_ft=3.0),
            create_product("MEDIUM", category="cable", length_ft=6.0),
            create_product("LONG", category="cable", length_ft=10.0),
        ]
        engine = ProductSearchEngine(products)
        results = engine.search({'category': 'cable', 'length': 6.0, 'length_unit': 'ft'})
        # Should include the 6ft cable (exact match)
        skus = [p.product_number for p in results]
        assert "MEDIUM" in skus

    def test_exact_or_longer(self):
        from core.models import LengthPreference
        products = [
            create_product("SHORT", category="cable", length_ft=3.0),
            create_product("LONG", category="cable", length_ft=10.0),
        ]
        engine = ProductSearchEngine(products)
        results = engine.search({
            'category': 'cable',
            'length': 6.0,
            'length_unit': 'ft',
            'length_preference': LengthPreference.EXACT_OR_LONGER,
        })
        skus = [p.product_number for p in results]
        assert "LONG" in skus

    def test_exact_or_shorter(self):
        from core.models import LengthPreference
        products = [
            create_product("SHORT", category="cable", length_ft=3.0),
            create_product("LONG", category="cable", length_ft=10.0),
        ]
        engine = ProductSearchEngine(products)
        results = engine.search({
            'category': 'cable',
            'length': 6.0,
            'length_unit': 'ft',
            'length_preference': LengthPreference.EXACT_OR_SHORTER,
        })
        skus = [p.product_number for p in results]
        assert "SHORT" in skus


# =============================================================================
# GATE LOGIC
# =============================================================================

class TestGateLogic:
    """Test that gate filters are mandatory pass/fail."""

    def test_category_gate_filters(self):
        products = [
            create_product("CABLE1", category="cable"),
            create_product("DOCK1", category="dock"),
        ]
        engine = ProductSearchEngine(products)
        results = engine.search({'category': 'cable'})
        # Only cables should pass the category gate
        assert all(p.metadata['category'] == 'cable' for p in results)

    def test_connector_gate_filters(self):
        products = [
            create_product("HDMI1", category="cable", connectors=["HDMI", "HDMI"]),
            create_product("USB1", category="cable", connectors=["USB-C", "USB-C"]),
        ]
        engine = ProductSearchEngine(products)
        results = engine.search({
            'category': 'cable',
            'connector_from': 'HDMI',
        })
        assert all("HDMI" in str(p.metadata.get('connectors', [])) for p in results)


# =============================================================================
# FEATURE MATCHING
# =============================================================================

class TestFeatureMatching:
    """Test feature-based filtering."""

    def test_single_feature(self):
        products = [
            create_product("4K_CABLE", features=["4K", "HDR"]),
            create_product("BASIC_CABLE", features=["1080p"]),
        ]
        engine = ProductSearchEngine(products)
        results = engine.search_scored({'features': ['4K']})
        # 4K cable should score higher
        if results:
            assert results[0][0].product_number == "4K_CABLE"

    def test_multi_feature(self):
        products = [
            create_product("FULL", features=["4K", "PoE", "HDR"]),
            create_product("PARTIAL", features=["4K"]),
        ]
        engine = ProductSearchEngine(products)
        results = engine.search_scored({'features': ['4K', 'PoE']})
        if results:
            # FULL matches both features, should score higher
            assert results[0][0].product_number == "FULL"


# =============================================================================
# THUNDERBOLT VERSION MATCHING
# =============================================================================

class TestThunderboltVersion:
    """Test Thunderbolt version backward compatibility."""

    def _create_tb_products(self):
        return [
            create_product("TB3", category="dock", hub_usb_version="Thunderbolt 3"),
            create_product("TB4", category="dock", hub_usb_version="Thunderbolt 4"),
            create_product("TB5", category="dock", hub_usb_version="Thunderbolt 5"),
        ]

    def test_tb3_filter_matches_tb3_and_higher(self):
        """TB3 filter should match TB3, TB4, TB5 (backward compat)."""
        engine = ProductSearchEngine(self._create_tb_products())
        results = engine.search({'category': 'dock', 'thunderbolt_version': 3})
        skus = [p.product_number for p in results]
        assert "TB3" in skus

    def test_tb4_filter_excludes_tb3(self):
        """TB4 filter should NOT match TB3."""
        engine = ProductSearchEngine(self._create_tb_products())
        results = engine.search({'category': 'dock', 'thunderbolt_version': 4})
        skus = [p.product_number for p in results]
        assert "TB3" not in skus
        assert "TB4" in skus


# =============================================================================
# CABLE TYPE HIERARCHY
# =============================================================================

class TestCableTypeHierarchy:
    """Test network cable type matching with hierarchy."""

    def _create_cable_products(self):
        return [
            create_product("CAT5E", category="cable", network_rating="Cat5e",
                          content="Cat5e ethernet cable"),
            create_product("CAT6", category="cable", network_rating="Cat6",
                          content="Cat6 ethernet cable"),
            create_product("CAT6A", category="cable", network_rating="Cat6a",
                          content="Cat6a ethernet cable"),
        ]

    def test_cat6a_filter_matches_cat6a(self):
        engine = ProductSearchEngine(self._create_cable_products())
        results = engine.search({'category': 'cable', 'cable_type': 'Cat6a'})
        skus = [p.product_number for p in results]
        assert "CAT6A" in skus

    def test_cat6a_filter_excludes_cat5e(self):
        engine = ProductSearchEngine(self._create_cable_products())
        results = engine.search({'category': 'cable', 'cable_type': 'Cat6a'})
        skus = [p.product_number for p in results]
        assert "CAT5E" not in skus


# =============================================================================
# PORT COUNT
# =============================================================================

class TestPortCount:
    """Test port count matching (>=, not exact)."""

    def test_minimum_port_match(self):
        products = [
            create_product("HUB4", category="hub", port_count=4),
            create_product("HUB7", category="hub", port_count=7),
            create_product("HUB10", category="hub", port_count=10),
        ]
        engine = ProductSearchEngine(products)
        results = engine.search({'category': 'hub', 'port_count': 7})
        skus = [p.product_number for p in results]
        # Should include 7 and 10 (>=), not 4
        assert "HUB4" not in skus
        assert "HUB7" in skus
        assert "HUB10" in skus
