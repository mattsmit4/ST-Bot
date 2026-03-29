"""Search engine package — filtering, scoring, and ranking."""

from core.search.engine import ProductSearchEngine, create_search_func
from core.search.strategy import SearchStrategy, SearchConfig
from core.search.wrapper import SearchEngineWrapper
from core.search.resolution import supports_4k, supports_resolution

__all__ = [
    'ProductSearchEngine', 'create_search_func',
    'SearchStrategy', 'SearchConfig',
    'SearchEngineWrapper',
    'supports_4k', 'supports_resolution',
]
