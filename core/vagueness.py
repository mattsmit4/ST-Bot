"""
Vague query detection for ST-Bot.

Detects when a user query is too vague to perform a meaningful search
and needs clarification before proceeding.

Two implementations:
- VagueQueryDetector: Regex-based (fast, but misses edge cases)
- LLMVaguenessDetector: LLM-based (accurate, understands context)
"""

import os
import re
import json
from typing import Optional
from dataclasses import dataclass

from core.models import VagueQueryType, SearchFilters
from core.openai_client import get_openai_client


# =============================================================================
# LLM-Based Vagueness Detection
# =============================================================================

VAGUENESS_DETECTION_PROMPT = '''You are a query analyzer for StarTech.com's product assistant.

Determine if this query is SPECIFIC ENOUGH to search for products, or TOO VAGUE and needs clarification.

## SEARCHABLE (return is_vague: false)
User provides enough info to find relevant products:
- Specific connector + category: "HDMI cables", "USB-C adapter", "DisplayPort cable", "VGA cables"
- Connector pair: "USB-C to HDMI cable", "DisplayPort to HDMI adapter"
- Device + destination: "connect my MacBook to a monitor", "laptop to projector"
- Specific use case: "4K gaming cable", "video cable for presentations"
- Product category + specs: "10ft ethernet cable", "thunderbolt dock"

## CRITICAL RULE
If user mentions a SPECIFIC connector type (HDMI, USB-C, DisplayPort, VGA, DVI, Thunderbolt, Serial, etc.)
plus a product category (cable, adapter, dock, hub), this is ALWAYS SEARCHABLE.
Examples that are ALWAYS SEARCHABLE:
- "HDMI cables" → SEARCHABLE (specific connector + category)
- "USB-C adapters" → SEARCHABLE (specific connector + category)
- "DisplayPort cable" → SEARCHABLE (specific connector + category)
- "show me HDMI cables" → SEARCHABLE (specific connector + category)
- "serial cable" → SEARCHABLE (serial is a specific connector type)
- "audio cable" → SEARCHABLE (audio is a specific cable type)

## SELF-CONTAINED CATEGORIES (ALWAYS SEARCHABLE)
Some product categories don't require connector specifications:
- Privacy Screens: "privacy screens", "privacy filter for monitor"
- KVM Switches: "KVM switch", "2-port KVM"
- Server Racks/Mounts: "server rack", "monitor arm", "TV mount"
- Storage Enclosures: "hard drive enclosure", "NVMe enclosure", "drive dock"
- Network Equipment: "ethernet switch", "patch panel", "network card", "network adapter", "NIC", "PCIe network"
- Audio Equipment: "audio cable", "speaker cable", "3.5mm cable"

If user specifies a SELF-CONTAINED product category, this is SEARCHABLE even without connector info.

## TOO VAGUE (return is_vague: true)
User hasn't specified what they actually need:
- Generic requests: "I need a cable", "something for my computer"
- No connectors or devices: "do you have cables?", "show me adapters"
- Asking for help deciding: "what should I get?", "help me choose"

## IMPORTANT CONTEXT RULES
- "Not sure if you can help, but [clear request]" → Look at the ACTUAL request, not the preamble
- "I need to connect my laptop to a projector" → SEARCHABLE (has devices, clear need)
- "connect laptop to TV" → SEARCHABLE (has devices)
- "I need something for my computer" → VAGUE (no specifics)
- "HDMI cables" → SEARCHABLE (has specific connector type)

Query: "{{query}}"
Filters extracted: {{filters_summary}}

Return JSON:
{{{{
    "is_vague": true/false,
    "vague_type": "cable" | "generic" | "ports" | "connector" | "uncertain" | null,
    "reasoning": "brief explanation",
    "what_is_missing": "what info would help" or null
}}}}'''


@dataclass
class LLMVaguenessResult:
    """Result from LLM vagueness detection."""
    is_vague: bool
    vague_type: Optional[VagueQueryType]
    reasoning: str
    what_is_missing: Optional[str]


class LLMVaguenessDetector:
    """
    LLM-based vagueness detector.

    Uses GPT-4o-mini to understand if a query is specific enough to search,
    considering context and nuance that regex patterns miss.

    Falls back to regex-based VagueQueryDetector on API errors.
    """

    def __init__(self, model: str = None, enable_fallback: bool = True):
        """
        Initialize LLM vagueness detector.

        Args:
            model: OpenAI model (default: gpt-4o-mini)
            enable_fallback: Whether to fall back to regex on errors
        """
        self.model = model or os.environ.get('OPENAI_CHAT_MODEL', 'gpt-5.4-nano')
        self.enable_fallback = enable_fallback
        self._regex_detector = None  # Lazy-loaded fallback

    @property
    def regex_detector(self):
        """Lazy-load regex detector for fallback."""
        if self._regex_detector is None:
            self._regex_detector = VagueQueryDetector()
        return self._regex_detector

    def detect(self, query: str, filters: Optional[SearchFilters] = None, debug_lines: list = None) -> Optional[VagueQueryType]:
        """
        Detect if a query is too vague to search.

        Args:
            query: The user's query string
            filters: Extracted search filters (optional)
            debug_lines: Optional list for debug output

        Returns:
            VagueQueryType if query is vague, None otherwise
        """
        try:
            result = self._detect_with_llm(query, filters)

            if debug_lines is not None:
                if result.is_vague:
                    debug_lines.append(f"🤖 LLM VAGUE: {result.vague_type.value if result.vague_type else 'generic'}")
                    debug_lines.append(f"   Reason: {result.reasoning[:60]}")
                else:
                    debug_lines.append(f"🤖 LLM SEARCHABLE: {result.reasoning[:60]}")

            return result.vague_type if result.is_vague else None

        except Exception as e:
            if debug_lines is not None:
                debug_lines.append(f"⚠️ LLM VAGUE FAILED: {type(e).__name__}: {str(e)[:50]}")

            if self.enable_fallback:
                if debug_lines is not None:
                    debug_lines.append("   Falling back to regex detector")
                return self.regex_detector.detect(query, filters)

            return None  # Assume searchable on error

    def _detect_with_llm(self, query: str, filters: Optional[SearchFilters]) -> LLMVaguenessResult:
        """Perform LLM-based vagueness detection."""
        client = get_openai_client()
        if not client:
            raise RuntimeError("OpenAI client not available")

        # Build filters summary
        filters_summary = self._build_filters_summary(filters)

        # Prepare prompt - use .format() with escaped braces
        prompt = VAGUENESS_DETECTION_PROMPT.format(
            query=query,
            filters_summary=filters_summary
        )

        # Call OpenAI
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": query}
            ],
            temperature=0.1,
            max_completion_tokens=200,
            response_format={"type": "json_object"},
            timeout=30.0
        )

        return self._parse_response(response.choices[0].message.content)

    def _build_filters_summary(self, filters: Optional[SearchFilters]) -> str:
        """Build a summary of extracted filters for the prompt."""
        if not filters:
            return "None extracted"

        parts = []
        if filters.product_category:
            parts.append(f"category={filters.product_category}")
        if filters.connector_from:
            parts.append(f"from={filters.connector_from}")
        if filters.connector_to:
            parts.append(f"to={filters.connector_to}")
        if filters.features:
            parts.append(f"features={filters.features}")
        if filters.keywords:
            parts.append(f"keywords={filters.keywords}")
        if filters.port_count:
            parts.append(f"port_count={filters.port_count}")
        if filters.min_monitors:
            parts.append(f"min_monitors={filters.min_monitors}")

        return ", ".join(parts) if parts else "No specific filters"

    def _parse_response(self, response: str) -> LLMVaguenessResult:
        """Parse LLM response into result object."""
        data = json.loads(response)

        is_vague = data.get('is_vague', False)
        vague_type_str = data.get('vague_type')
        reasoning = data.get('reasoning', '')
        what_is_missing = data.get('what_is_missing')

        # Map vague type string to enum
        vague_type = None
        if is_vague and vague_type_str:
            type_mapping = {
                'cable': VagueQueryType.CABLE,
                'generic': VagueQueryType.GENERIC,
                'ports': VagueQueryType.PORTS,
                'connector': VagueQueryType.CONNECTOR,
                'uncertain': VagueQueryType.UNCERTAIN,
            }
            vague_type = type_mapping.get(vague_type_str.lower(), VagueQueryType.GENERIC)
        elif is_vague:
            vague_type = VagueQueryType.GENERIC

        return LLMVaguenessResult(
            is_vague=is_vague,
            vague_type=vague_type,
            reasoning=reasoning,
            what_is_missing=what_is_missing
        )


# =============================================================================
# Regex-Based Vagueness Detection (Original)
# =============================================================================


class VagueQueryDetector:
    """
    Detects vague queries that need clarification before searching.

    Vague queries are ones where the user hasn't provided enough specifics
    to return useful results. For example:
    - "I need a cable" (what kind? what connectors?)
    - "Something for my computer" (what purpose?)
    - "I need ports" (what type? how many?)
    """

    # Patterns for vague cable requests
    VAGUE_CABLE_PATTERNS = [
        r'^i\s+need\s+(?:a\s+)?cable[s]?\.?$',
        r'^(?:can\s+you\s+)?(?:show\s+me\s+)?(?:your\s+)?cables?\.?$',
        r'^(?:what\s+)?cables?\s+(?:do\s+you\s+have|are\s+available)\.?$',
        r'^(?:i\'?m\s+)?looking\s+for\s+(?:a\s+)?cable[s]?\.?$',
        r'^(?:i\s+)?need\s+(?:a\s+)?cable\s+for\s+(?:my\s+)?(?:computer|laptop|pc)\.?$',
    ]

    # Patterns for vague generic requests
    VAGUE_GENERIC_PATTERNS = [
        r'^something\s+for\s+(?:my\s+)?(?:computer|laptop|pc)\.?$',
        r'^whatever\s+works\.?$',
        r'^(?:i\s+)?need\s+something\.?$',
        r'^what\s+(?:do\s+you\s+)?(?:have|recommend)\??$',
    ]

    # Patterns for vague port/hub requests
    VAGUE_PORTS_PATTERNS = [
        r'^i\s+need\s+(?:more\s+)?ports?\.?$',
        r'^i\s+need\s+(?:a\s+)?hub\.?$',
        r'^(?:i\'?m\s+)?(?:looking\s+for|need)\s+(?:more\s+)?(?:usb\s+)?ports?\.?$',
    ]

    # Patterns for vague connector requests
    VAGUE_CONNECTOR_PATTERNS = [
        r'^(?:the\s+)?thing\s+that\s+connects\.?$',
        r'^(?:i\s+need\s+)?(?:a\s+)?connector\.?$',
        r'^(?:an?\s+)?adapter\.?$',
    ]

    # Patterns that indicate user explicitly doesn't know what they need
    # These should trigger clarification regardless of other keywords
    UNCERTAINTY_PHRASES = [
        r"\bdon'?t\s+know\b",           # "I don't know"
        r"\bnot\s+sure\b",               # "I'm not sure"
        r"\bno\s+idea\b",                # "I have no idea"
        r"\bwhat\s+(?:kind|type)\s+of\b.*\bneed\b",  # "what kind of cables do I need"
        r"\bwhich\s+(?:cable|adapter|one)\s+(?:should|do)\b",  # "which cable should I"
        r"\bhelp\s+me\s+(?:figure|choose|pick|decide)\b",  # "help me figure out"
        r"\bconfused\s+(?:about|by)\b",  # "confused about cables"
        r"\bunsure\b",                   # "unsure what I need"
    ]

    # Valid connector types - used to validate extracted connectors
    VALID_CONNECTORS = {
        'hdmi', 'displayport', 'dp', 'usb-c', 'usbc', 'type-c', 'typec',
        'usb-a', 'usba', 'type-a', 'typea', 'thunderbolt', 'tb3', 'tb4',
        'vga', 'dvi', 'ethernet', 'rj45', 'rj-45', 'mini-dp', 'minidp',
        'lightning', 'micro-usb', 'microusb', 'usb-b', 'usbb',
        # Serial connectors
        'serial', 'rs232', 'rs-232', 'db9', 'db-9', 'db25', 'db-25',
        # Audio connectors
        '3.5mm', 'aux', 'audio', 'toslink', 'optical', 'spdif',
    }

    def detect(self, query: str, filters: Optional[SearchFilters] = None) -> Optional[VagueQueryType]:
        """
        Detect if a query is too vague to search.

        Args:
            query: The user's query string
            filters: Extracted search filters (optional)

        Returns:
            VagueQueryType if query is vague, None otherwise
        """
        query_lower = query.lower().strip()

        # FIRST: Check for uncertainty phrases
        # "I don't know what cables I need" should ALWAYS trigger clarification
        # even if other keywords are present (like "computer" or "monitor")
        if self._has_uncertainty_phrase(query_lower):
            return VagueQueryType.UNCERTAIN

        # Check cable patterns
        for pattern in self.VAGUE_CABLE_PATTERNS:
            if re.match(pattern, query_lower):
                return VagueQueryType.CABLE

        # Check generic patterns
        for pattern in self.VAGUE_GENERIC_PATTERNS:
            if re.match(pattern, query_lower):
                return VagueQueryType.GENERIC

        # Check ports patterns
        for pattern in self.VAGUE_PORTS_PATTERNS:
            if re.match(pattern, query_lower):
                return VagueQueryType.PORTS

        # Check connector patterns
        for pattern in self.VAGUE_CONNECTOR_PATTERNS:
            if re.match(pattern, query_lower):
                return VagueQueryType.CONNECTOR

        # Check for sparse filters - category set but no valid connector info
        if filters:
            if self._has_sparse_filters(query, filters):
                return VagueQueryType.CABLE

        return None

    def _has_uncertainty_phrase(self, text: str) -> bool:
        """
        Detect phrases indicating user explicitly doesn't know what they need.

        Args:
            text: Query text (lowercase)

        Returns:
            True if uncertainty phrase found
        """
        for pattern in self.UNCERTAINTY_PHRASES:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False

    def _is_valid_connector(self, connector: str) -> bool:
        """
        Check if a connector value is a valid connector type.

        "Computer" and "monitor" are NOT valid connectors - they're devices.
        "HDMI", "USB-C", "DisplayPort" are valid connectors.

        Args:
            connector: Connector string to validate

        Returns:
            True if valid connector type
        """
        if not connector:
            return False

        connector_lower = connector.lower().strip()

        # Check direct match
        if connector_lower in self.VALID_CONNECTORS:
            return True

        # Check if any valid connector is a substring
        # This catches "USB-C" in "usb-c", "hdmi 2.1" containing "hdmi", etc.
        for valid in self.VALID_CONNECTORS:
            if valid in connector_lower:
                return True

        return False

    def _has_sparse_filters(self, query: str, filters: SearchFilters) -> bool:
        """
        Check if filters indicate a vague query.

        A query is considered vague if:
        - It has a product category but no VALID connector info
        - It's relatively short (8 words or less)
        - No specific features requested

        Note: "Computer" and "Monitor" are NOT valid connectors - they're devices.
        If the filter extractor put "Computer" as connector_from, we should still
        treat this as a vague query because the user hasn't specified actual connector types.
        """
        # Must have category to be considered sparse
        if not filters.product_category:
            return False

        # Check if connectors are VALID (not just populated)
        # "Computer" and "Monitor" are not valid connectors - they're device names
        has_valid_from = self._is_valid_connector(filters.connector_from)
        has_valid_to = self._is_valid_connector(filters.connector_to)

        # If connectors are populated but INVALID, treat as sparse/vague
        if filters.connector_from and not has_valid_from:
            return True  # Connector populated with invalid value like "Computer"
        if filters.connector_to and not has_valid_to:
            return True  # Connector populated with invalid value like "Monitor"

        # If we have valid connector info, not vague
        if has_valid_from or has_valid_to:
            return False

        # Short query check - longer queries usually have more context
        word_count = len(query.split())
        if word_count > 8:
            return False

        # Specific features make it not vague
        if filters.features:
            return False

        # Specific keywords make it not vague
        if filters.keywords:
            return False

        return True
