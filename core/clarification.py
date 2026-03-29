"""
Clarification question building and response parsing for ST-Bot.

Handles the multi-turn conversation when a user's query is too vague
to search directly.
"""

import re
from typing import Optional

from core.models import (
    VagueQueryType,
    ClarificationMissing,
    PendingClarification,
)
from config.device_mappings import (
    DEVICE_CONNECTOR_PATTERNS,
    SOURCE_DEVICE_PATTERNS,
    DEST_DEVICE_PATTERNS,
)


class ClarificationQuestionBuilder:
    """
    Builds clarifying questions for vague queries.

    Tries LLM-generated questions first for natural, context-aware responses.
    Falls back to hard-coded templates if LLM is unavailable or fails.
    """

    def __init__(self):
        self._llm_generator = None  # Lazy-loaded

    @property
    def llm_generator(self):
        if self._llm_generator is None:
            from llm.llm_clarification_generator import LLMClarificationGenerator
            self._llm_generator = LLMClarificationGenerator()
        return self._llm_generator

    # Questions organized by vague type and what info is missing
    QUESTIONS = {
        VagueQueryType.CABLE: {
            ClarificationMissing.USE_CASE: (
                "What devices are you trying to connect? "
                "For example: laptop to monitor, phone to TV, or computer to projector."
            ),
            ClarificationMissing.CONNECTOR_FROM: (
                "What type of port does your source device have? "
                "Common types: USB-C, HDMI, DisplayPort, USB-A, Thunderbolt."
            ),
            ClarificationMissing.CONNECTOR_TO: (
                "What type of port does your destination device have?"
            ),
        },
        VagueQueryType.GENERIC: {
            ClarificationMissing.USE_CASE: (
                "What are you trying to accomplish?\n"
                "- Connect to an external monitor\n"
                "- Add more USB ports\n"
                "- Set up a docking station\n"
                "- Something else"
            ),
        },
        VagueQueryType.PORTS: {
            ClarificationMissing.USE_CASE: (
                "What kind of ports do you need more of?\n"
                "- USB-A ports (for mice, keyboards, drives)\n"
                "- USB-C ports\n"
                "- Video outputs (HDMI, DisplayPort)\n"
                "- Something else"
            ),
        },
        VagueQueryType.CONNECTOR: {
            ClarificationMissing.USE_CASE: (
                "What are you trying to connect? "
                "Tell me about the devices and I can help find the right cable or adapter."
            ),
        },
        VagueQueryType.UNCERTAIN: {
            ClarificationMissing.USE_CASE: (
                "I'd be happy to help! What ports do you see on each device?\n\n"
                "**Look for labels like:** HDMI, USB-C, DisplayPort, or VGA\n\n"
                "Or just tell me your devices (e.g., 'MacBook Pro' and 'Dell monitor') "
                "and I'll figure out what you need."
            ),
        },
    }

    # Fallback question when no specific question matches
    DEFAULT_QUESTION = (
        "Could you tell me more about what you're looking for? "
        "What devices are you trying to connect, or what problem are you trying to solve?"
    )

    def build_question(self, clarification: PendingClarification,
                       conversation_history: list = None) -> str:
        """
        Build the next clarifying question to ask.

        Tries LLM-generated question first, falls back to hard-coded templates.

        Args:
            clarification: Current clarification state
            conversation_history: Recent Message objects for context

        Returns:
            The question string to ask the user
        """
        # Try LLM-generated question first
        try:
            llm_question = self.llm_generator.generate_question(
                original_query=clarification.original_query,
                vague_type=clarification.vague_type,
                missing_info=clarification.missing_info,
                collected_info=clarification.collected_info,
                questions_asked=clarification.questions_asked,
                conversation_history=conversation_history,
            )
            if llm_question:
                return llm_question
        except Exception:
            pass  # Fall through to hard-coded fallback

        # Fallback: hard-coded question templates
        vague_type = clarification.vague_type
        type_questions = self.QUESTIONS.get(vague_type, {})

        for missing in clarification.missing_info:
            if missing.value in clarification.collected_info:
                continue

            question = type_questions.get(missing)
            if question:
                return question

        return self.DEFAULT_QUESTION

    def build_initial_response(self, clarification: PendingClarification,
                               conversation_history: list = None) -> str:
        """
        Build the initial response when we detect a vague query.

        Tries LLM-generated response first, falls back to hard-coded templates.

        Args:
            clarification: New clarification state
            conversation_history: Recent Message objects for context

        Returns:
            Response string with acknowledgment and question
        """
        # Try LLM-generated initial response
        try:
            llm_response = self.llm_generator.generate_initial_response(
                original_query=clarification.original_query,
                vague_type=clarification.vague_type,
                missing_info=clarification.missing_info,
                collected_info=clarification.collected_info,
            )
            if llm_response:
                return llm_response
        except Exception:
            pass  # Fall through to hard-coded fallback

        # Fallback: hard-coded acknowledgment + question
        acknowledgments = {
            VagueQueryType.CABLE: "I can help you find the right cable.",
            VagueQueryType.GENERIC: "I'd be happy to help you find what you need.",
            VagueQueryType.PORTS: "I can help you find the right solution for more ports.",
            VagueQueryType.CONNECTOR: "I can help you find the right connector.",
            VagueQueryType.UNCERTAIN: "No problem!",
        }

        ack = acknowledgments.get(clarification.vague_type, "I can help with that.")
        question = self.build_question(clarification, conversation_history)

        return f"{ack} {question}"


class ClarificationResponseParser:
    """
    Parses user responses to clarification questions.

    Extracts useful information from natural language responses
    like device types, connector types, use cases, and product categories.
    """

    # Product category patterns - what type of product user is looking for
    PRODUCT_CATEGORY_PATTERNS = {
        'dock': [
            r'\bdock(?:ing)?(?:\s+station)?\b',
            r'\b(?:usb[- ]?c|thunderbolt)\s+dock\b',
        ],
        'hub': [
            r'\busb\s+hub\b',
            r'\bport\s+hub\b',
        ],
        'adapter': [
            r'\badapter\b',
            r'\bdongle\b',
        ],
        # 'Cables' is default, no pattern needed
    }

    # Device connector patterns imported from config.device_mappings
    DEST_DEVICE_PATTERNS = DEST_DEVICE_PATTERNS

    # Direct connector mentions
    CONNECTOR_PATTERNS = {
        'usb-c': 'USB-C',
        'usb c': 'USB-C',
        'type-c': 'USB-C',
        'type c': 'USB-C',
        'usb-a': 'USB-A',
        'usb a': 'USB-A',
        'hdmi': 'HDMI',
        'displayport': 'DisplayPort',
        'display port': 'DisplayPort',
        'dp': 'DisplayPort',
        'thunderbolt': 'Thunderbolt',
        'vga': 'VGA',
        'dvi': 'DVI',
        'lightning': 'Lightning',
        'mini displayport': 'Mini DisplayPort',
        'mini dp': 'Mini DisplayPort',
    }

    # Use case pattern matching
    USE_CASE_PATTERNS = {
        'video_output': [
            r'\bmonitors?\b',
            r'\bdisplays?\b',
            r'\btv\b',
            r'\btelevisions?\b',
            r'\bprojectors?\b',
            r'\bscreens?\b',
            r'\bexternal\s+displays?\b',
            r'\bsecond\s+(?:monitor|screen)\b',
            r'\bdual\s+monitors?\b',
        ],
        'data_transfer': [
            r'\bhard\s+drive\b',
            r'\bexternal\s+drive\b',
            r'\bssd\b',
            r'\busb\s+drive\b',
            r'\bflash\s+drive\b',
            r'\btransfer\s+files?\b',
        ],
        'charging': [
            r'\bcharg(?:e|ing)\b',
            r'\bpower\s+(?:delivery|bank|supply|adapter)\b',
            r'\bbattery\b',
        ],
        'ports': [
            r'\boutlets?\b',
            r'\bports?\b',
            r'\bexpand\b',
        ],
        'ports': [
            r'\bmore\s+ports?\b',
            r'\bhub\b',
            r'\bdock(?:ing)?\b',
            r'\bexpand\b',
        ],
    }

    def parse_response(
        self,
        response: str,
        clarification: PendingClarification
    ) -> PendingClarification:
        """
        Parse a user's response and extract information.

        Args:
            response: The user's response text
            clarification: Current clarification state

        Returns:
            Updated clarification with extracted info
        """
        response_lower = response.lower()

        # For detailed responses (>8 words), delegate to LLM filter extraction
        # This handles specs like "3.5mm 10ft shielded audio cable" that simple patterns miss
        word_count = len(response.split())
        if word_count > 8:
            try:
                from llm.llm_filter_extractor import LLMFilterExtractor
                extractor = LLMFilterExtractor()
                result = extractor.extract(response)

                if result and result.filters:
                    full_filters = result.filters
                    # Transfer relevant filters to collected_info
                    if full_filters.connector_from:
                        clarification.collected_info['connector_from'] = full_filters.connector_from
                    if full_filters.connector_to:
                        clarification.collected_info['connector_to'] = full_filters.connector_to
                    if full_filters.length:
                        clarification.collected_info['length'] = full_filters.length
                    if full_filters.product_category:
                        clarification.collected_info['product_category'] = full_filters.product_category
                    if full_filters.features:
                        clarification.collected_info['features'] = full_filters.features
                    if full_filters.cable_type:
                        clarification.collected_info['cable_type'] = full_filters.cable_type
                    if full_filters.requested_network_speed:
                        clarification.collected_info['network_speed'] = full_filters.requested_network_speed
                    if full_filters.keywords:
                        clarification.collected_info['keywords'] = full_filters.keywords

                    clarification.questions_asked += 1
                    return clarification
            except Exception:
                pass  # Fall through to simple extraction

        # For short responses (or if LLM failed), use simple extraction patterns
        # Try to extract connector from
        connector_from = self._extract_connector_from(response_lower)
        if connector_from:
            clarification.collected_info['connector_from'] = connector_from

        # Try to extract connector to
        connector_to = self._extract_connector_to(response_lower)
        if connector_to:
            clarification.collected_info['connector_to'] = connector_to

        # Try to extract use case
        use_case = self._extract_use_case(response_lower)
        if use_case:
            clarification.collected_info['use_case'] = use_case

        # Try to extract product category (dock, hub, adapter, etc.)
        product_category = self._extract_product_category(response_lower)
        if product_category:
            clarification.collected_info['product_category'] = product_category

        # Increment question count
        clarification.questions_asked += 1

        return clarification

    def _extract_connector_from(self, response: str) -> Optional[str]:
        """Extract source connector type from response."""
        # Check for direct connector mentions first
        for pattern, connector in self.CONNECTOR_PATTERNS.items():
            if pattern in response:
                return connector

        # Check for source device patterns (MacBooks, laptops, etc.)
        for pattern in SOURCE_DEVICE_PATTERNS:
            if re.search(pattern, response):
                # Found a source device, get its connector
                for device_pattern, connector in DEVICE_CONNECTOR_PATTERNS:
                    if device_pattern == pattern and connector:
                        return connector

        return None

    def _extract_connector_to(self, response: str) -> Optional[str]:
        """Extract destination connector type from response."""
        response = response.lower()
        # PRIORITY 1: Check for explicit connector mentions first
        # "my monitor has DisplayPort" -> DisplayPort (not HDMI inferred from "monitor")
        # Look for patterns like "has DisplayPort", "with HDMI", "[device] has [connector]"
        explicit_connector_patterns = [
            # "[device] has [connector]" - captures connector after "has"
            r'\b(?:monitor|tv|display|projector|screen)s?\s+(?:has|have|with)\s+(?:a\s+)?(\w+[\w\-]*)',
            # "to [connector]" when connector is explicit (not a device)
            r'\bto\s+(?:a\s+)?(?:my\s+)?(displayport|hdmi|vga|dvi|usb[- ]?c|type[- ]?c)\b',
            # "[connector] on my [device]" or "[connector] port on"
            r'\b(displayport|hdmi|vga|dvi|usb[- ]?c|type[- ]?c)\s+(?:port\s+)?on\s+(?:my\s+)?(?:monitor|tv|display)',
        ]

        for pattern in explicit_connector_patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                captured = match.group(1).lower()
                # Check if captured text is a known connector type
                for connector_pattern, connector_name in self.CONNECTOR_PATTERNS.items():
                    if connector_pattern in captured or captured in connector_pattern:
                        return connector_name

        # PRIORITY 2: Check for any direct connector mention in the response
        # This catches cases like "laptop USB-C, monitor DisplayPort"
        # We want the connector associated with destination devices
        found_dest_device = False
        for pattern in self.DEST_DEVICE_PATTERNS:
            if re.search(pattern, response):
                found_dest_device = True
                break

        if found_dest_device:
            # Look for connector mentioned near destination device keywords
            dest_match = re.search(
                r'\b(?:monitor|tv|display|projector|screen)s?\b.*?\b(displayport|display\s+port|dp|hdmi|vga|dvi)\b',
                response, re.IGNORECASE
            )
            if dest_match:
                connector_text = dest_match.group(1).lower()
                for connector_pattern, connector_name in self.CONNECTOR_PATTERNS.items():
                    if connector_pattern in connector_text or connector_text in connector_pattern:
                        return connector_name

        # PRIORITY 3: Fallback - infer from destination device if no explicit connector
        for pattern in self.DEST_DEVICE_PATTERNS:
            if re.search(pattern, response):
                # Found a destination device, get its DEFAULT connector (HDMI for displays)
                for device_pattern, connector in DEVICE_CONNECTOR_PATTERNS:
                    if device_pattern == pattern:
                        return connector

        # Look for "to X" patterns
        to_match = re.search(r'\bto\s+(?:my\s+)?(\w+)', response)
        if to_match:
            after_to = to_match.group(1)
            # Check if what follows "to" is a destination device
            for pattern in self.DEST_DEVICE_PATTERNS:
                if re.search(pattern, after_to):
                    return 'HDMI'  # Most common for displays

        return None

    def _extract_use_case(self, response: str) -> Optional[str]:
        """Extract use case from response."""
        for use_case, patterns in self.USE_CASE_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, response):
                    return use_case

        return None

    def _extract_product_category(self, response: str) -> Optional[str]:
        """Extract product category if mentioned (dock, hub, adapter, etc.)."""
        for category, patterns in self.PRODUCT_CATEGORY_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, response):
                    return category

        return None
