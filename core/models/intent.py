"""Intent types and routing sets."""

from enum import Enum
from dataclasses import dataclass
from typing import Optional


class IntentType(Enum):
    """
    User intent types — v2.0 architecture with 8 intents, 2 paths.

    Fast exit (pre-built responses, no filter extraction):
        GREETING, FAREWELL, OUT_OF_SCOPE

    Deep pipeline (filter extraction → handler → assigned output):
        NEW_SEARCH, FOLLOWUP, EDUCATIONAL, CLARIFICATION, SPECIFIC_SKU
    """
    # Fast exit
    GREETING = "greeting"
    FAREWELL = "farewell"
    OUT_OF_SCOPE = "out_of_scope"

    # Deep pipeline
    NEW_SEARCH = "new_search"
    FOLLOWUP = "followup"
    EDUCATIONAL = "educational"
    CLARIFICATION = "clarification"
    SPECIFIC_SKU = "specific_sku"


FAST_EXIT_INTENTS = {
    IntentType.GREETING,
    IntentType.FAREWELL,
    IntentType.OUT_OF_SCOPE,
}

DEEP_PIPELINE_INTENTS = {
    IntentType.NEW_SEARCH,
    IntentType.FOLLOWUP,
    IntentType.EDUCATIONAL,
    IntentType.CLARIFICATION,
    IntentType.SPECIFIC_SKU,
}


@dataclass
class Intent:
    """User intent with metadata."""
    type: IntentType
    confidence: float
    reasoning: str
    sku: Optional[str] = None
    meta_info: Optional[dict] = None

    def __str__(self) -> str:
        return f"Intent({self.type.value}, confidence={self.confidence:.2f})"
