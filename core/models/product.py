"""Product data model — pure data container, no business logic."""

from dataclasses import dataclass
from typing import Any


@dataclass
class Product:
    """
    Product information.

    Attributes:
        product_number: StarTech product SKU
        content: Full product specification text
        metadata: Product metadata dict. Common derived keys include:
            - category (str): Normalized category (e.g., 'cable', 'dock')
            - connectors (list[str]): Derived from Connector_A/B
            - features (list[str]): Derived features (e.g., ['4K', 'Power Delivery'])
            - length_ft (float): Cable length in feet
            - length_m (float): Cable length in meters
            - length_display (str): Formatted length (e.g., '6.0 ft [1.8 m]')
            - name (str): Product name (SKU)
            - sku (str): Product SKU
            - port_count (int): Number of ports
            - bay_count (int): Number of drive bays
            - network_rating (str): Cable rating (e.g., 'Cat6a')
            - hub_usb_version (str): USB version for hubs
            - max_refresh_rate (int): Max refresh rate in Hz
            - power_delivery_watts (int): PD wattage
        score: Relevance score from search
    """
    product_number: str
    content: str
    metadata: dict[str, Any]
    score: float = 1.0

    def get(self, key: str, default: Any = None) -> Any:
        """Get metadata value safely."""
        return self.metadata.get(key, default)
