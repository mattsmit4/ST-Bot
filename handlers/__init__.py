"""Intent handlers — business logic for each intent type."""

from handlers.base import BaseHandler, HandlerContext, HandlerResult
from handlers.greeting import GreetingHandler, FarewellHandler, OutOfScopeHandler
from handlers.educational import EducationalHandler
from handlers.search import NewSearchHandler
from handlers.sku import SkuHandler
from handlers.clarification import VagueSearchHandler, ClarificationResponseHandler
from handlers.narrowing import NarrowingResponseHandler, start_narrowing
from handlers.followup import FollowupHandler

__all__ = [
    'BaseHandler', 'HandlerContext', 'HandlerResult',
    'GreetingHandler', 'FarewellHandler', 'OutOfScopeHandler',
    'EducationalHandler',
    'NewSearchHandler',
    'SkuHandler',
    'VagueSearchHandler', 'ClarificationResponseHandler',
    'NarrowingResponseHandler', 'start_narrowing',
    'FollowupHandler',
]
