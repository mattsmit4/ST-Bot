"""Search filter models — what the user is looking for."""

from dataclasses import dataclass, field, fields
from typing import Optional, Any
from enum import Enum


class LengthPreference(Enum):
    """User preference for length alternatives when exact match unavailable."""
    EXACT_OR_LONGER = "exact_or_longer"
    EXACT_OR_SHORTER = "exact_or_shorter"
    CLOSEST = "closest"


@dataclass
class DroppedFilter:
    """Information about a filter that was relaxed during search."""
    filter_name: str
    requested_value: Any
    reason: str
    alternatives: Optional[list[Any]] = None


@dataclass
class SearchFilters:
    """Extracted search filters from user query. 25 optional fields covering all product categories."""
    length: Optional[float] = None
    length_unit: Optional[str] = None
    length_preference: LengthPreference = LengthPreference.EXACT_OR_LONGER
    length_min: Optional[float] = None
    length_max: Optional[float] = None
    connector_from: Optional[str] = None
    connector_to: Optional[str] = None
    features: list[str] = field(default_factory=list)
    product_category: Optional[str] = None
    port_count: Optional[int] = None
    color: Optional[str] = None
    keywords: list[str] = field(default_factory=list)
    required_port_types: list[str] = field(default_factory=list)
    min_monitors: Optional[int] = None
    screen_size: Optional[float] = None
    cable_type: Optional[str] = None
    kvm_video_type: Optional[str] = None
    thunderbolt_version: Optional[int] = None
    bay_count: Optional[int] = None
    rack_height: Optional[int] = None
    drive_size: Optional[str] = None
    usb_version: Optional[str] = None
    requested_refresh_rate: Optional[int] = None
    requested_power_wattage: Optional[int] = None
    requested_network_speed: Optional[int] = None

    @property
    def has_search_criteria(self) -> bool:
        """True if any meaningful search filter is set."""
        _SKIP = {'length_unit', 'length_preference'}
        for f in fields(self):
            if f.name in _SKIP:
                continue
            val = getattr(self, f.name)
            if val is not None and val != []:
                return True
        return False


@dataclass
class LLMExtractionResult:
    """Result from LLM filter extraction."""
    filters: SearchFilters
    confidence: float = 0.0
    reasoning: str = ""


@dataclass
class FilterConfig:
    """Configuration for filter extraction."""
    categorical_values: dict[str, list]
    sku_set: set[str]
    sku_map: dict[str, str]
