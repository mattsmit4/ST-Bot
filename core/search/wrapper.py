"""Search engine wrapper — binds SearchStrategy + ProductSearchEngine."""

from typing import List, Optional

from core.search.strategy import SearchStrategy, SearchConfig
from core.search.engine import ProductSearchEngine
from core.models import SearchFilters, SearchResult, Product


class SearchEngineWrapper:
    """
    Wraps SearchStrategy + ProductSearchEngine into a single interface.

    Handlers call:
      - wrapper.search(filters)  -> delegates to strategy.search(filters, engine=engine)
      - wrapper.engine           -> raw ProductSearchEngine (for fallback patterns in followup)
    """

    def __init__(self, products: List[Product], config: Optional[SearchConfig] = None):
        self.engine = ProductSearchEngine(products)
        self._strategy = SearchStrategy(config)

    def search(self, filters: SearchFilters) -> SearchResult:
        """Search products using the configured strategy."""
        return self._strategy.search(filters, engine=self.engine)

    def filter_products(self, products: List[Product], filters: SearchFilters) -> List[Product]:
        """Filter a provided list of products against SearchFilters criteria."""
        filter_dict = self._strategy._build_tier1_filters(filters)
        return [p for p in products if self.engine._product_matches(p, filter_dict)]
