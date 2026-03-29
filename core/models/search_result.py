"""Search result model."""

from dataclasses import dataclass, field
from typing import Optional

from core.models.product import Product
from core.models.filters import DroppedFilter


@dataclass
class SearchResult:
    """
    Search result with products and metadata.

    match_quality values: "exact", "partial", "relaxed"
    """
    products: list[Product]
    filters_used: dict
    match_quality: str = "exact"
    total_count: int = 0
    original_filters: Optional[dict] = None
    dropped_filters: list[DroppedFilter] = field(default_factory=list)
    category_relaxed: bool = False
    search_timed_out: bool = False
    search_scores: list[float] = field(default_factory=list)

    def had_filter_relaxation(self) -> bool:
        return len(self.dropped_filters) > 0

    def get_dropped_filter(self, name: str) -> Optional[DroppedFilter]:
        for df in self.dropped_filters:
            if df.filter_name == name:
                return df
        return None
